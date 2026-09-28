# Multi-Agent AI Governance & Evaluation System

![CI](https://github.com/Snehavermaa/ai-governance-eval/actions/workflows/ci.yml/badge.svg)

A guardrail service that checks LLM answers **claim by claim** against a trusted document corpus, rewrites
them to keep only verified content, detects personal data, and returns an auditable **approve / revise / block**
decision, with a trace of every step.

```
question ──► Retriever ──► Generator ──► Critic ──► Synthesizer ──► Governance ──► decision + final answer
              (BM25)        (LLM)        (per-claim    (keeps only      (PII, risk
                                         verdicts)     cited, verified   score, policy)
                                                       claims)
                         every stage is a timed span in a JSON trace  ──►  GET /traces/{id}
```

## What each agent does

| Agent | Responsibility | Implementation |
|---|---|---|
| Retriever | Find evidence passages for a question/claim | BM25 over sentence-chunked Markdown docs (pure Python) |
| Generator | Draft an answer from retrieved context only | Anthropic / OpenAI / Gemini via REST, temperature 0 |
| Critic | Split the answer into claims; label each **supported / unsupported / contradicted** with a cited evidence id | `LexicalJudge` (offline: term coverage + number agreement + negation) or `LLMJudge` (structured JSON output) |
| Synthesizer | Rebuild the answer from supported claims only, each with a citation | Extractive by design, so it cannot introduce new hallucinations |
| Governance | PII detection, risk score, decision | Regex + Luhn (cards) + Verhoeff (Aadhaar) checksums; configurable `Policy` weights and thresholds |

**Structured output with self-repair.** The LLM critic must return JSON matching a Pydantic schema, and its
`evidence_id` must be one of the passages it was shown. Invalid replies (bad JSON, unknown label, hallucinated
evidence id) are sent back with the validation error for up to two repair attempts before failing loudly.

**Risk and decision.** `risk = 0.5 × unsupported ratio + 1.0 × contradicted ratio + PII penalty` (clipped to 1).
High-severity PII (card, Aadhaar, PAN) or risk ≥ 0.6 → **block**; any risk → **revise** (verified claims only, PII
redacted); otherwise **approve** the original. All weights live in `Policy`.

## Evaluation

`python -m governance.evaluation --judge lexical` (offline) or `--judge llm` (uses your API key) writes
`reports/eval_<judge>.md` and `.json`. Datasets are in `data/eval/`:

- `claims.jsonl`: 71 hand-labelled claims (37 supported, 16 unsupported, 18 contradicted), including paraphrases
  and non-numeric contradictions designed to be hard for lexical matching.
- `answers.jsonl`: 16 multi-claim answers with the expected governance decision.
- `pii.jsonl`: 20 texts, including hard negatives (order numbers, invalid card and Aadhaar checksums).

Results with the offline lexical critic (thresholds fixed before evaluation, not tuned on this data):

| Metric | Lexical critic |
|---|---|
| Hallucination flagging (unsupported + contradicted) — recall | **100%** |
| Hallucination flagging — precision | 89.5% |
| Hallucination flagging — F1 | 94.4% |
| 3-class accuracy | 87.3% |
| Governance decision accuracy (16 answers) | 100% |
| Unsafe approvals | 0 |
| PII precision / recall (20 texts) | 100% / 100% |

Where the lexical critic fails (see `reports/eval_lexical.md`): paraphrases ("two weeks" vs "10 working days"),
and contradictions that don't involve a number ("approved by your manager" vs "requires a Vice President").
These are the cases the LLM critic is meant to handle. Run `--judge llm` to measure it.

The datasets are small and written for this project, so the numbers show the method works and let us catch
regressions; they are not a benchmark.

**Quality gate:** `tests/test_evaluation.py` runs the evaluation in CI and fails the build if hallucination recall
drops below 100%, F1 below 90%, or any answer that should be blocked is approved.

## Run it

```bash
python -m venv .venv && .venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
pip install -r requirements-dev.txt
uvicorn governance.api:app --reload                     # console at http://127.0.0.1:8000
```

Works fully offline. To enable generation and the LLM critic, copy `.env.example` values into your environment:
`GOV_LLM_PROVIDER=anthropic` (or `openai` / `gemini`), the matching API key, and `GOV_JUDGE=llm`.

| Endpoint | Purpose |
|---|---|
| `POST /audit` `{question, answer}` | Audit an existing answer (offline-capable) |
| `POST /ask` `{question}` | Generate with the LLM, then audit (503 if no LLM configured) |
| `GET /traces/{trace_id}` | Per-stage spans with timings, inputs and errors |
| `GET /health` | Active judge, generator and corpus size |

Docker: `docker build -t ai-governance . && docker run -p 8000:8000 ai-governance`

## Testing

```bash
pytest --cov=governance        # 94 tests, ~96% coverage, < 1 s, no network or API keys needed
flake8 && black --check .
```

LLM behaviour is tested with a deterministic `FakeLLM` (valid, fenced, malformed and hallucinated-id replies);
provider request/response shapes are tested with a mocked HTTP layer. Mutation spot-checks (disabling the Luhn
check, the negation check, PII blocking, or the synthesizer filter) each cause test failures.

CI (GitHub Actions): lint → tests on Python 3.10–3.12 with a 90% coverage gate → evaluation report published to the
job summary → Docker build and container smoke test.

## Project layout

```
governance/   retrieval.py  judges.py  llm.py  pii.py  pipeline.py  tracing.py  evaluation.py  api.py  static/
data/corpus/  fictional company policy documents (the trusted source)
data/eval/    labelled evaluation sets
reports/      committed offline evaluation report
tests/        unit, API and evaluation-gate tests
```

## Limitations and next steps

- BM25 retrieval misses synonyms; dense embeddings (e.g. sentence-transformers + FAISS) would improve recall.
- Claims are split by sentence; compound sentences carrying two facts are judged as one.
- The PII detector is rule-based and India-focused; names and addresses are not detected.
- Evaluation sets are small; a larger, independently labelled set would give tighter estimates.
