import json

import pytest

from governance.judges import Label, LexicalJudge, LLMJudge
from governance.llm import FakeLLM, LLMError

J = LexicalJudge()


def judge(index, claim, question=""):
    return J.judge(claim, index.search(f"{question} {claim}", 4))


def test_lexical_supported_cites_evidence(index):
    v = judge(index, "Passwords must be at least 14 characters long.")
    assert v.label is Label.SUPPORTED and v.evidence_id.startswith("security#")


def test_lexical_number_mismatch_is_contradiction(index):
    v = judge(index, "Passwords must be at least 8 characters long.")
    assert v.label is Label.CONTRADICTED and "8" in v.rationale


def test_lexical_negation_mismatch_is_contradiction(index):
    v = judge(index, "Multi-factor authentication is not mandatory for company accounts.")
    assert v.label is Label.CONTRADICTED


def test_lexical_unrelated_claim_is_unsupported(index):
    assert judge(index, "The office cafeteria serves free lunch on Fridays.").label is Label.UNSUPPORTED


def test_lexical_no_evidence():
    v = J.judge("Anything at all.", [])
    assert v.label is Label.UNSUPPORTED and v.evidence_id is None


def llm_reply(label, evidence_id, rationale="r"):
    return json.dumps({"label": label, "evidence_id": evidence_id, "rationale": rationale})


def evidence(index):
    return index.search("password characters", 2)


def test_llm_judge_parses_valid_json(index):
    ev = evidence(index)
    fake = FakeLLM([llm_reply("supported", ev[0].passage.id)])
    v = LLMJudge(fake).judge("Passwords are long.", ev)
    assert v.label is Label.SUPPORTED and v.evidence_id == ev[0].passage.id
    assert v.judge == "llm:fake"
    assert ev[0].passage.id in fake.calls[0][1]  # evidence ids were shown to the model


def test_llm_judge_accepts_fenced_json(index):
    ev = evidence(index)
    fake = FakeLLM(["```json\n" + llm_reply("contradicted", ev[0].passage.id) + "\n```"])
    assert LLMJudge(fake).judge("x", ev).label is Label.CONTRADICTED


def test_llm_judge_repairs_invalid_output(index):
    ev = evidence(index)
    fake = FakeLLM(["not json at all", llm_reply("supported", ev[0].passage.id)])
    v = LLMJudge(fake).judge("x", ev)
    assert v.label is Label.SUPPORTED
    assert len(fake.calls) == 2 and "previous reply was invalid" in fake.calls[1][1]


def test_llm_judge_rejects_hallucinated_evidence_id(index):
    ev = evidence(index)
    fake = FakeLLM([llm_reply("supported", "made_up#9"), llm_reply("supported", ev[0].passage.id)])
    assert LLMJudge(fake).judge("x", ev).evidence_id == ev[0].passage.id
    assert "must be one of" in fake.calls[1][1]  # the repair prompt told the model why


def test_llm_judge_rejects_unknown_label(index):
    ev = evidence(index)
    fake = FakeLLM([llm_reply("maybe", None), llm_reply("unsupported", None)])
    assert LLMJudge(fake).judge("x", ev).label is Label.UNSUPPORTED


def test_llm_judge_gives_up_after_repairs(index):
    fake = FakeLLM(["bad", "still bad", "nope"])
    with pytest.raises(LLMError, match="after repairs"):
        LLMJudge(fake).judge("x", evidence(index))
    assert len(fake.calls) == 3


def test_llm_judge_skips_model_when_no_evidence():
    fake = FakeLLM([])
    assert LLMJudge(fake).judge("x", []).label is Label.UNSUPPORTED
    assert fake.calls == []
