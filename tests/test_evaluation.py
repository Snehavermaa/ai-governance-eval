"""Quality gate: the evaluation suite runs in CI and fails the build if the system regresses."""

import pytest

from governance.evaluation import evaluate_pii, prf, run, to_markdown


@pytest.fixture(scope="module")
def results():
    return run("lexical")


def test_no_hallucinated_claim_slips_through(results):
    assert results["claims"]["flagging"]["recall"] == 1.0


def test_flagging_quality_floor(results):
    assert results["claims"]["flagging"]["f1"] >= 0.90
    assert results["claims"]["accuracy"] >= 0.85


def test_governance_never_approves_an_answer_that_should_be_blocked(results):
    assert results["decisions"]["unsafe_approvals"] == 0
    assert results["decisions"]["accuracy"] >= 0.9


def test_pii_quality_floor(results):
    assert results["pii"]["precision"] >= 0.95 and results["pii"]["recall"] >= 0.95


def test_report_renders(results):
    md = to_markdown(results)
    assert "Confusion matrix" in md and "judge: `lexical`" in md


def test_prf_edge_cases():
    assert prf(0, 0, 0) == {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    assert prf(1, 1, 0) == {"precision": 0.5, "recall": 1.0, "f1": 0.667}


def test_evaluate_pii_counts_errors():
    out = evaluate_pii([{"id": "x", "text": "a@b.com", "expected_types": ["phone"]}])
    assert out["precision"] == 0.0 and len(out["errors"]) == 1
