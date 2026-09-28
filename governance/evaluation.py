"""Evaluation harness: critic accuracy, end-to-end governance decisions and PII detection.

Run:  python -m governance.evaluation --judge lexical      (offline)
      python -m governance.evaluation --judge llm          (uses GOV_LLM_PROVIDER + API key)
"""

import argparse
import json
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .judges import Judge, Label, LexicalJudge, LLMJudge
from .llm import client_from_env
from .pii import detect_pii
from .pipeline import GovernancePipeline
from .retrieval import BM25Index

ROOT = Path(__file__).resolve().parent.parent
LABELS = [label.value for label in Label]


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": round(p, 3), "recall": round(r, 3), "f1": round(f1, 3)}


@dataclass
class ClaimEval:
    accuracy: float
    confusion: dict[str, dict[str, int]]  # gold -> predicted -> count
    flagging: dict[str, float]  # "flagged" = unsupported or contradicted
    per_label: dict[str, dict[str, float]]
    errors: list[dict]
    n: int


def evaluate_claims(index: BM25Index, judge: Judge, rows: list[dict], top_k: int = 4) -> ClaimEval:
    confusion = {g: {p: 0 for p in LABELS} for g in LABELS}
    errors = []
    for row in rows:
        evidence = index.search(f"{row['question']} {row['claim']}", top_k)
        pred = judge.judge(row["claim"], evidence).label.value
        confusion[row["label"]][pred] += 1
        if pred != row["label"]:
            errors.append({"id": row["id"], "claim": row["claim"], "gold": row["label"], "pred": pred})
    n = len(rows)
    correct = sum(confusion[label][label] for label in LABELS)
    flagged = [x for x in LABELS if x != Label.SUPPORTED.value]
    tp = sum(confusion[g][p] for g in flagged for p in flagged)
    fp = sum(confusion[Label.SUPPORTED.value][p] for p in flagged)
    fn = sum(confusion[g][Label.SUPPORTED.value] for g in flagged)
    per_label = {
        label: prf(
            confusion[label][label],
            sum(confusion[g][label] for g in LABELS if g != label),
            sum(confusion[label][p] for p in LABELS if p != label),
        )
        for label in LABELS
    }
    return ClaimEval(round(correct / n, 3), confusion, prf(tp, fp, fn), per_label, errors, n)


def evaluate_decisions(pipeline: GovernancePipeline, rows: list[dict]) -> dict:
    results = []
    for row in rows:
        report = pipeline.audit(row["question"], row["answer"])
        results.append({"id": row["id"], "expected": row["expected_decision"], "got": report.decision.value})
    correct = sum(r["expected"] == r["got"] for r in results)
    unsafe = [r for r in results if r["expected"] == "block" and r["got"] == "approve"]
    return {
        "n": len(rows),
        "accuracy": round(correct / len(rows), 3),
        "unsafe_approvals": len(unsafe),
        "mismatches": [r for r in results if r["expected"] != r["got"]],
    }


def evaluate_pii(rows: list[dict]) -> dict:
    tp = fp = fn = 0
    errors = []
    for row in rows:
        got = Counter(f.type for f in detect_pii(row["text"]))
        want = Counter(row["expected_types"])
        tp += sum((got & want).values())
        fp += sum((got - want).values())
        fn += sum((want - got).values())
        if got != want:
            errors.append(
                {
                    "id": row["id"],
                    "text": row["text"],
                    "expected": sorted(want.elements()),
                    "got": sorted(got.elements()),
                }
            )
    return {"n": len(rows), **prf(tp, fp, fn), "errors": errors}


def build_judge(kind: str) -> Judge:
    if kind == "lexical":
        return LexicalJudge()
    client = client_from_env()
    if client is None:
        raise SystemExit("--judge llm needs GOV_LLM_PROVIDER and the matching API key")
    return LLMJudge(client)


def run(judge_kind: str = "lexical", data_dir: Path = ROOT / "data") -> dict:
    index = BM25Index.from_directory(data_dir / "corpus")
    judge = build_judge(judge_kind)
    t0 = time.perf_counter()
    claims = evaluate_claims(index, judge, load_jsonl(data_dir / "eval" / "claims.jsonl"))
    decisions = evaluate_decisions(GovernancePipeline(index, judge), load_jsonl(data_dir / "eval" / "answers.jsonl"))
    pii = evaluate_pii(load_jsonl(data_dir / "eval" / "pii.jsonl"))
    return {
        "judge": judge.name,
        "corpus_passages": len(index.passages),
        "seconds": round(time.perf_counter() - t0, 2),
        "claims": claims.__dict__,
        "decisions": decisions,
        "pii": pii,
    }


def to_markdown(r: dict) -> str:
    c, d, p = r["claims"], r["decisions"], r["pii"]
    lines = [
        f"# Evaluation report — judge: `{r['judge']}`",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Claims evaluated | {c['n']} |",
        f"| 3-class accuracy (supported / unsupported / contradicted) | {c['accuracy']:.1%} |",
        f"| Hallucination flagging — precision | {c['flagging']['precision']:.1%} |",
        f"| Hallucination flagging — recall | {c['flagging']['recall']:.1%} |",
        f"| Hallucination flagging — F1 | {c['flagging']['f1']:.1%} |",
        f"| Governance decision accuracy ({d['n']} answers) | {d['accuracy']:.1%} |",
        f"| Unsafe approvals (should block, approved) | {d['unsafe_approvals']} |",
        f"| PII detection precision / recall ({p['n']} texts) | {p['precision']:.1%} / {p['recall']:.1%} |",
        "",
        "## Confusion matrix (rows = gold, columns = predicted)",
        "",
        "| gold \\ pred | " + " | ".join(LABELS) + " |",
        "|---|" + "---|" * len(LABELS),
    ]
    lines += [f"| {g} | " + " | ".join(str(c["confusion"][g][x]) for x in LABELS) + " |" for g in LABELS]
    lines += ["", "## Critic errors", ""]
    lines += [f"- `{e['id']}` gold={e['gold']} pred={e['pred']}: {e['claim']}" for e in c["errors"]] or ["- none"]
    lines += ["", "## Decision mismatches", ""]
    lines += [f"- `{m['id']}` expected={m['expected']} got={m['got']}" for m in d["mismatches"]] or ["- none"]
    lines += ["", "## PII errors", ""]
    lines += [f"- `{e['id']}` expected={e['expected']} got={e['got']}: {e['text']}" for e in p["errors"]] or ["- none"]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--judge", choices=["lexical", "llm"], default="lexical")
    ap.add_argument("--out", default=str(ROOT / "reports"))
    args = ap.parse_args()
    result = run(args.judge)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"eval_{args.judge}.json").write_text(json.dumps(result, indent=2))
    (out / f"eval_{args.judge}.md").write_text(to_markdown(result))
    print(to_markdown(result))


if __name__ == "__main__":
    main()
