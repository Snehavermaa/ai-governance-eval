"""Rule-based PII detection and redaction (Indian + international formats)."""

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class PIIFinding:
    type: str
    severity: str  # "high" | "medium"
    start: int
    end: int
    value: str


def luhn_valid(digits: str) -> bool:
    total, parity = 0, len(digits) % 2
    for i, ch in enumerate(digits):
        d = int(ch)
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


# Verhoeff checksum tables (used by Aadhaar's final check digit)
_VD = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
    [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
    [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
    [4, 0, 1, 2, 3, 9, 5, 6, 7, 8],
    [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
    [6, 5, 9, 8, 7, 1, 0, 4, 3, 2],
    [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
    [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
    [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
]
_VP = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
    [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
    [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
    [9, 4, 5, 3, 1, 2, 6, 8, 7, 0],
    [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
    [2, 7, 9, 3, 8, 0, 6, 4, 1, 5],
    [7, 0, 4, 6, 9, 1, 3, 2, 5, 8],
]


def verhoeff_valid(digits: str) -> bool:
    c = 0
    for i, ch in enumerate(reversed(digits)):
        c = _VD[c][_VP[i % 8][int(ch)]]
    return c == 0


def _aadhaar_ok(m: re.Match) -> bool:
    return verhoeff_valid(re.sub(r"\D", "", m.group(0)))


def _card_ok(m: re.Match) -> bool:
    digits = re.sub(r"\D", "", m.group(0))
    return 13 <= len(digits) <= 19 and luhn_valid(digits)


# (type, severity, pattern, extra validator). Order matters: earlier matches win on overlap.
RULES = [
    ("email", "medium", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"), None),
    ("credit_card", "high", re.compile(r"\b(?:\d[ -]?){12,18}\d\b"), _card_ok),
    ("aadhaar", "high", re.compile(r"\b[2-9]\d{3}[ -]?\d{4}[ -]?\d{4}\b"), _aadhaar_ok),
    ("pan", "high", re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"), None),
    ("phone", "medium", re.compile(r"(?<!\w)(?:\+91[ -]?|0)?[6-9]\d{4}[ -]?\d{5}\b"), None),
    ("phone", "medium", re.compile(r"(?<!\w)\+\d{1,3}[ -]?\(?\d{2,4}\)?[ -]?\d{3,4}[ -]?\d{3,4}\b"), None),
]


def detect_pii(text: str) -> list[PIIFinding]:
    findings: list[PIIFinding] = []
    taken: list[tuple[int, int]] = []
    for kind, severity, pattern, validator in RULES:
        for m in pattern.finditer(text):
            if validator and not validator(m):
                continue
            if any(m.start() < e and s < m.end() for s, e in taken):
                continue
            taken.append((m.start(), m.end()))
            findings.append(PIIFinding(kind, severity, m.start(), m.end(), m.group(0)))
    return sorted(findings, key=lambda f: f.start)


def redact(text: str, findings: list[PIIFinding] | None = None) -> str:
    findings = detect_pii(text) if findings is None else findings
    for f in sorted(findings, key=lambda f: f.start, reverse=True):
        text = text[: f.start] + f"[REDACTED_{f.type.upper()}]" + text[f.end :]
    return text
