"""Claim-level judges: decide whether retrieved evidence supports each claim."""

from dataclasses import dataclass
from enum import Enum
from typing import Literal, Protocol

from pydantic import BaseModel

from .llm import LLMClient, complete_structured
from .retrieval import Retrieved
from .text import content_terms, numbers, split_sentences, tokenize


class Label(str, Enum):
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    CONTRADICTED = "contradicted"


@dataclass
class Verdict:
    label: Label
    evidence_id: str | None
    evidence_text: str | None
    confidence: float
    rationale: str
    judge: str


class Judge(Protocol):
    name: str

    def judge(self, claim: str, evidence: list[Retrieved]) -> Verdict: ...


NEGATIONS = frozenset({"not", "no", "never", "cannot", "without", "none", "nor"})


def _negated(text: str) -> bool:
    toks = tokenize(text.replace("n't", " not"))
    return any(t in NEGATIONS for t in toks)


@dataclass
class LexicalJudge:
    """Offline, deterministic judge based on term coverage, number agreement and negation.

    Thresholds are fixed constants (not tuned on the evaluation set).
    """

    support_threshold: float = 0.75
    related_threshold: float = 0.5
    name: str = "lexical"

    def judge(self, claim: str, evidence: list[Retrieved]) -> Verdict:
        claim_nums = numbers(claim)
        claim_terms = content_terms(claim) - claim_nums
        best = None  # (coverage, sentence, passage_id)
        for r in evidence:
            for sent in split_sentences(r.passage.text):
                terms = content_terms(sent)
                cov = len(claim_terms & terms) / len(claim_terms) if claim_terms else 0.0
                if best is None or cov > best[0]:
                    best = (cov, sent, r.passage.id)

        if best is None or best[0] < self.related_threshold:
            cov = best[0] if best else 0.0
            return Verdict(
                Label.UNSUPPORTED,
                None,
                None,
                round(1 - cov, 2),
                f"No evidence sentence covers the claim (best coverage {cov:.2f}).",
                self.name,
            )

        cov, sent, pid = best
        ev_nums = numbers(sent)
        if claim_nums and not claim_nums <= ev_nums and ev_nums:
            return Verdict(
                Label.CONTRADICTED,
                pid,
                sent,
                round(cov, 2),
                f"Numbers {sorted(claim_nums - ev_nums)} disagree with evidence {sorted(ev_nums)}.",
                self.name,
            )
        if _negated(claim) != _negated(sent) and cov >= self.support_threshold:
            return Verdict(
                Label.CONTRADICTED, pid, sent, round(cov, 2), "Negation differs from the evidence.", self.name
            )
        if cov >= self.support_threshold and claim_nums <= ev_nums:
            return Verdict(
                Label.SUPPORTED, pid, sent, round(cov, 2), f"Evidence covers {cov:.0%} of claim terms.", self.name
            )
        return Verdict(
            Label.UNSUPPORTED,
            pid,
            sent,
            round(1 - cov, 2),
            f"Related evidence found but it covers only {cov:.0%} of the claim.",
            self.name,
        )


class _LLMVerdict(BaseModel):
    label: Literal["supported", "unsupported", "contradicted"]
    evidence_id: str | None = None
    rationale: str


JUDGE_SYSTEM = (
    "You are a strict fact-checking critic. Judge ONE claim using ONLY the evidence passages provided. "
    "supported = the evidence states it; contradicted = the evidence states something incompatible "
    "(e.g. a different number, the opposite rule); unsupported = the evidence does not settle it. "
    "Never use outside knowledge. Cite the single best evidence id, or null if unsupported."
)


@dataclass
class LLMJudge:
    client: LLMClient
    name: str = "llm"

    def judge(self, claim: str, evidence: list[Retrieved]) -> Verdict:
        if not evidence:
            return Verdict(Label.UNSUPPORTED, None, None, 1.0, "No evidence retrieved.", self.name)
        by_id = {r.passage.id: r.passage.text for r in evidence}
        passages = "\n".join(f"[{pid}] {text}" for pid, text in by_id.items())
        prompt = f"Evidence passages:\n{passages}\n\nClaim: {claim}"

        def check(v: _LLMVerdict) -> None:
            if v.label != "unsupported" and v.evidence_id not in by_id:
                raise ValueError(f"evidence_id must be one of {list(by_id)} when label is {v.label}")

        result = complete_structured(self.client, JUDGE_SYSTEM, prompt, _LLMVerdict, check=check)
        v: _LLMVerdict = result.value  # type: ignore[assignment]
        eid = v.evidence_id if v.evidence_id in by_id else None
        return Verdict(
            Label(v.label), eid, by_id.get(eid) if eid else None, 1.0, v.rationale, f"{self.name}:{self.client.name}"
        )
