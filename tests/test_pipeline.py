import json

import pytest

from governance.judges import Label
from governance.llm import FakeLLM, LLMError
from governance.pipeline import NO_SUPPORT, REFUSAL, Decision, GovernancePipeline, Policy
from governance.tracing import TraceStore

GOOD = (
    "Passwords must be at least 14 characters long. Multi-factor authentication is mandatory for all company accounts."
)


def test_fully_supported_answer_is_approved_unchanged(pipeline):
    r = pipeline.audit("What are the password rules?", GOOD)
    assert r.decision is Decision.APPROVE and r.risk_score == 0
    assert r.final_answer == GOOD


def test_partially_unsupported_answer_is_revised_to_supported_claims(pipeline):
    answer = GOOD + " Passwords must be changed every 90 days."
    r = pipeline.audit("What are the password rules?", answer)
    assert r.decision is Decision.REVISE
    assert "90 days" not in r.final_answer
    assert r.final_answer.count("[security#") == 2  # each kept claim is cited
    assert r.counts() == {"supported": 2, "unsupported": 1, "contradicted": 0}


def test_mostly_contradicted_answer_is_blocked(pipeline):
    r = pipeline.audit(
        "Rules?", "Passwords must be at least 8 characters long. Laptops lock after 15 minutes of inactivity."
    )
    assert r.decision is Decision.BLOCK and r.final_answer == REFUSAL
    assert r.risk_score == 1.0


def test_high_severity_pii_blocks_even_if_facts_are_right(pipeline):
    r = pipeline.audit("Rules?", GOOD + " My card is 4111 1111 1111 1111.")
    assert r.decision is Decision.BLOCK
    assert {f.type for f in r.pii} == {"credit_card"}


def test_medium_pii_is_redacted_in_revision(pipeline):
    answer = "Security incidents must be reported to the security team within 24 hours. Contact sec@example.com."
    r = pipeline.audit("How do I report an incident?", answer)
    assert r.decision is Decision.REVISE
    assert "sec@example.com" not in r.final_answer


def test_empty_answer_is_blocked(pipeline):
    r = pipeline.audit("Anything?", "   ")
    assert r.decision is Decision.BLOCK and "no checkable claims" in r.reasons[0]


def test_revision_with_nothing_supported(index):
    lenient = GovernancePipeline(index, policy=Policy(block_threshold=1.1))
    r = lenient.audit("Food?", "The cafeteria serves free lunch on Fridays.")
    assert r.decision is Decision.REVISE and r.final_answer == NO_SUPPORT


@pytest.mark.parametrize("threshold,expected", [(0.15, Decision.BLOCK), (0.9, Decision.REVISE)])
def test_policy_threshold_is_configurable(index, threshold, expected):
    p = GovernancePipeline(index, policy=Policy(block_threshold=threshold))
    r = p.audit("Rules?", GOOD + " Passwords must be changed every 90 days.")
    assert r.decision is expected


def test_ask_uses_generator_with_retrieved_context(index):
    fake = FakeLLM([GOOD])
    p = GovernancePipeline(index, generator=fake)
    r = p.ask("What are the password rules?")
    assert r.decision is Decision.APPROVE
    system, prompt = fake.calls[0]
    assert "ONLY the context" in system and "security#" in prompt


def test_ask_without_llm_raises(pipeline):
    with pytest.raises(LLMError):
        pipeline.ask("Anything?")


def test_trace_records_every_stage_and_is_persisted(index, tmp_path):
    p = GovernancePipeline(index, generator=FakeLLM([GOOD]), store=TraceStore(tmp_path))
    r = p.ask("What are the password rules?")
    saved = json.loads((tmp_path / f"{r.trace_id}.json").read_text())
    assert [s["name"] for s in saved["spans"]] == ["retriever", "generator", "critic", "synthesizer", "governance"]
    assert all(s["duration_ms"] >= 0 for s in saved["spans"])
    assert saved["decision"] == "approve"
    assert saved["counts"]["supported"] == 2


def test_trace_records_errors(index):
    def boom(*_):
        raise TimeoutError("provider timeout")

    p = GovernancePipeline(index, generator=FakeLLM(boom))
    from governance.tracing import Trace

    trace = Trace()
    with pytest.raises(TimeoutError):
        p.generate("q", trace)
    assert trace.spans[-1].name == "generator" and "TimeoutError" in trace.spans[-1].error


def test_every_verdict_label_is_a_known_label(pipeline):
    r = pipeline.audit("Leave?", "Maternity leave is 26 weeks and fully paid. Paternity leave is 5 working days.")
    assert {c.verdict.label for c in r.claims} <= set(Label)
