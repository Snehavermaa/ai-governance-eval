"""The multi-agent pipeline: Retriever -> Generator -> Critic -> Synthesizer -> Governance."""

from dataclasses import dataclass, field
from enum import Enum

from .judges import Judge, Label, LexicalJudge, Verdict
from .llm import LLMClient, LLMError
from .pii import PIIFinding, detect_pii, redact
from .retrieval import BM25Index
from .text import split_sentences
from .tracing import Trace, TraceStore


class Decision(str, Enum):
    APPROVE = "approve"
    REVISE = "revise"
    BLOCK = "block"


@dataclass(frozen=True)
class Policy:
    """Risk weights and thresholds. Risk is clipped to [0, 1]."""

    unsupported_weight: float = 0.5
    contradicted_weight: float = 1.0
    pii_medium_penalty: float = 0.2
    pii_high_penalty: float = 0.6
    block_threshold: float = 0.6
    top_k: int = 4


@dataclass
class ClaimResult:
    claim: str
    verdict: Verdict


@dataclass
class Report:
    trace_id: str
    question: str
    original_answer: str
    claims: list[ClaimResult]
    pii: list[PIIFinding]
    risk_score: float
    decision: Decision
    final_answer: str
    reasons: list[str] = field(default_factory=list)

    def counts(self) -> dict[str, int]:
        out = {label.value: 0 for label in Label}
        for c in self.claims:
            out[c.verdict.label.value] += 1
        return out


GENERATOR_SYSTEM = (
    "Answer the user's question using ONLY the context passages. Be concise (at most 4 sentences). "
    "If the context does not contain the answer, say you don't know."
)
REFUSAL = (
    "This response was withheld by the governance policy because it could not be verified or contained sensitive data."
)
NO_SUPPORT = "I couldn't find support for this answer in the provided documents."


class GovernancePipeline:
    def __init__(
        self,
        index: BM25Index,
        judge: Judge | None = None,
        generator: LLMClient | None = None,
        policy: Policy | None = None,
        store: TraceStore | None = None,
    ):
        self.index = index
        self.judge = judge or LexicalJudge()
        self.generator = generator
        self.policy = policy or Policy()
        self.store = store or TraceStore()

    # ---- agents -------------------------------------------------------------------------
    def generate(self, question: str, trace: Trace) -> str:
        if self.generator is None:
            raise LLMError("No LLM configured for generation. Set GOV_LLM_PROVIDER, or use audit mode.")
        with trace.span("retriever", query=question) as s:
            hits = self.index.search(question, self.policy.top_k)
            s.attributes["passages"] = [h.passage.id for h in hits]
        context = "\n".join(f"[{h.passage.id}] {h.passage.text}" for h in hits)
        with trace.span("generator", provider=self.generator.name) as s:
            answer = self.generator.complete(GENERATOR_SYSTEM, f"Context:\n{context}\n\nQuestion: {question}").strip()
            s.attributes["answer_chars"] = len(answer)
        return answer

    def critique(self, question: str, answer: str, trace: Trace) -> list[ClaimResult]:
        results = []
        with trace.span("critic", judge=self.judge.name) as s:
            for claim in split_sentences(answer):
                evidence = self.index.search(f"{question} {claim}", self.policy.top_k)
                results.append(ClaimResult(claim, self.judge.judge(claim, evidence)))
            s.attributes["claims"] = len(results)
            s.attributes["labels"] = [r.verdict.label.value for r in results]
        return results

    def synthesize(self, claims: list[ClaimResult], trace: Trace) -> str:
        """Extractive: keep only supported claims, each with its citation. Adds no new text, so it
        cannot introduce new hallucinations."""
        with trace.span("synthesizer") as s:
            kept = [f"{c.claim} [{c.verdict.evidence_id}]" for c in claims if c.verdict.label is Label.SUPPORTED]
            s.attributes["kept"] = len(kept)
            s.attributes["dropped"] = len(claims) - len(kept)
        return " ".join(kept) if kept else NO_SUPPORT

    def govern(self, answer: str, claims: list[ClaimResult], revised: str, trace: Trace):
        p = self.policy
        with trace.span("governance") as s:
            pii = detect_pii(answer)
            n = len(claims) or 1
            unsupported = sum(c.verdict.label is Label.UNSUPPORTED for c in claims) / n
            contradicted = sum(c.verdict.label is Label.CONTRADICTED for c in claims) / n
            high_pii = any(f.severity == "high" for f in pii)
            risk = p.unsupported_weight * unsupported + p.contradicted_weight * contradicted
            if pii:
                risk += p.pii_high_penalty if high_pii else p.pii_medium_penalty
            risk = round(min(1.0, risk), 3)

            reasons = []
            if unsupported:
                reasons.append(f"{unsupported:.0%} of claims unsupported by the documents")
            if contradicted:
                reasons.append(f"{contradicted:.0%} of claims contradicted by the documents")
            if pii:
                reasons.append("contains PII: " + ", ".join(sorted({f.type for f in pii})))
            if not claims:
                reasons.append("answer contained no checkable claims")

            if high_pii or risk >= p.block_threshold or not claims:
                decision, final = Decision.BLOCK, REFUSAL
            elif risk > 0:
                decision, final = Decision.REVISE, redact(revised)
            else:
                decision, final = Decision.APPROVE, answer
            s.attributes.update(risk=risk, decision=decision.value, pii_types=sorted({f.type for f in pii}))
        return pii, risk, decision, final, reasons

    # ---- entry points ---------------------------------------------------------------------
    def audit(self, question: str, answer: str, trace: Trace | None = None) -> Report:
        """Check an existing answer. Works fully offline with the lexical judge."""
        trace = trace or Trace()
        claims = self.critique(question, answer, trace)
        revised = self.synthesize(claims, trace)
        pii, risk, decision, final, reasons = self.govern(answer, claims, revised, trace)
        report = Report(trace.trace_id, question, answer, claims, pii, risk, decision, final, reasons)
        self.store.save(
            trace, {"question": question, "decision": decision.value, "risk_score": risk, "counts": report.counts()}
        )
        return report

    def ask(self, question: str) -> Report:
        """Generate an answer with the configured LLM, then audit it."""
        trace = Trace()
        answer = self.generate(question, trace)
        return self.audit(question, answer, trace)
