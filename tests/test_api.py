import pytest
from fastapi.testclient import TestClient

from governance.api import create_app
from governance.llm import FakeLLM
from governance.pipeline import GovernancePipeline

GOOD = "Passwords must be at least 14 characters long."


@pytest.fixture
def client(index):
    return TestClient(create_app(GovernancePipeline(index)))


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["judge"] == "lexical" and body["generator"] is None


def test_audit_returns_structured_report(client):
    body = client.post("/audit", json={"question": "Rules?", "answer": GOOD}).json()
    assert body["decision"] == "approve"
    assert body["claims"][0]["label"] == "supported"
    assert body["claims"][0]["evidence_id"].startswith("security#")


def test_audit_never_echoes_raw_pii(client):
    body = client.post("/audit", json={"question": "q", "answer": GOOD + " Card 4111 1111 1111 1111."}).json()
    assert body["pii"] == [{"type": "credit_card", "severity": "high", "start": 52, "end": 71}]
    assert "4111" not in body["final_answer"]


@pytest.mark.parametrize(
    "payload", [{}, {"question": "", "answer": "x"}, {"question": "q"}, {"question": "q", "answer": "x" * 8001}]
)
def test_audit_validates_input(client, payload):
    assert client.post("/audit", json=payload).status_code == 422


def test_trace_roundtrip(client):
    trace_id = client.post("/audit", json={"question": "q", "answer": GOOD}).json()["trace_id"]
    trace = client.get(f"/traces/{trace_id}").json()
    assert [s["name"] for s in trace["spans"]] == ["critic", "synthesizer", "governance"]


@pytest.mark.parametrize("trace_id", ["doesnotexist", "..%2F..%2Fetc%2Fpasswd", "abc-def"])
def test_unknown_or_malicious_trace_ids_404(client, trace_id):
    assert client.get(f"/traces/{trace_id}").status_code == 404


def test_ask_without_llm_is_503(client):
    assert client.post("/ask", json={"question": "q"}).status_code == 503


def test_ask_with_llm(index):
    c = TestClient(create_app(GovernancePipeline(index, generator=FakeLLM([GOOD]))))
    assert c.post("/ask", json={"question": "Password rules?"}).json()["decision"] == "approve"


def test_console_page(client):
    r = client.get("/")
    assert r.status_code == 200 and "AI Governance Console" in r.text
