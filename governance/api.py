"""FastAPI service exposing the governance pipeline.

Run: uvicorn governance.api:app --reload     then open http://127.0.0.1:8000
"""

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from .judges import LexicalJudge, LLMJudge
from .llm import LLMError, client_from_env
from .pipeline import GovernancePipeline, Report
from .retrieval import BM25Index
from .tracing import TraceStore

ROOT = Path(__file__).resolve().parent.parent


class AuditRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    answer: str = Field(max_length=8000)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


def report_to_dict(r: Report) -> dict:
    return {
        "trace_id": r.trace_id,
        "question": r.question,
        "decision": r.decision.value,
        "risk_score": r.risk_score,
        "reasons": r.reasons,
        "final_answer": r.final_answer,
        "original_answer": r.original_answer,
        "counts": r.counts(),
        "claims": [
            {
                "claim": c.claim,
                "label": c.verdict.label.value,
                "evidence_id": c.verdict.evidence_id,
                "evidence": c.verdict.evidence_text,
                "rationale": c.verdict.rationale,
                "judge": c.verdict.judge,
            }
            for c in r.claims
        ],
        # never echo the raw PII value back
        "pii": [{"type": f.type, "severity": f.severity, "start": f.start, "end": f.end} for f in r.pii],
    }


def pipeline_from_env() -> GovernancePipeline:
    index = BM25Index.from_directory(os.environ.get("GOV_CORPUS_DIR", ROOT / "data" / "corpus"))
    client = client_from_env()
    judge = LLMJudge(client) if client and os.environ.get("GOV_JUDGE", "lexical") == "llm" else LexicalJudge()
    return GovernancePipeline(
        index, judge, generator=client, store=TraceStore(os.environ.get("GOV_TRACE_DIR", ROOT / "traces"))
    )


def create_app(pipeline: GovernancePipeline | None = None) -> FastAPI:
    app = FastAPI(title="AI Governance & Evaluation Service", version="0.1.0")
    app.state.pipeline = pipeline or pipeline_from_env()

    @app.get("/health")
    def health():
        p: GovernancePipeline = app.state.pipeline
        return {
            "status": "ok",
            "judge": p.judge.name,
            "generator": p.generator.name if p.generator else None,
            "passages": len(p.index.passages),
        }

    @app.post("/audit")
    def audit(req: AuditRequest):
        return report_to_dict(app.state.pipeline.audit(req.question, req.answer))

    @app.post("/ask")
    def ask(req: AskRequest):
        try:
            return report_to_dict(app.state.pipeline.ask(req.question))
        except LLMError as exc:
            raise HTTPException(status_code=503, detail=str(exc))

    @app.get("/traces/{trace_id}")
    def get_trace(trace_id: str):
        trace = app.state.pipeline.store.get(trace_id)
        if trace is None:
            raise HTTPException(status_code=404, detail="trace not found")
        return trace

    @app.get("/", response_class=HTMLResponse)
    def home():
        return (Path(__file__).parent / "static" / "index.html").read_text(encoding="utf-8")

    return app


app = create_app() if os.environ.get("GOV_SKIP_APP") != "1" else None
