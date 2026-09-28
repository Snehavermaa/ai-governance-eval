"""Small text utilities shared by retrieval, claim splitting and judging."""

import re

STOPWORDS = frozenset(
    """a an the and or but if of to in on at by for with from as is are was were be been being
    this that these those it its their there here they them we you your our i he she his her
    do does did done can could should would may might must will shall not no nor so than then
    also any all each every some such only own same very just into over under about after before
    between during through per via who whom which what when where why how""".split()
)

_TOKEN = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")
_NUMBER = re.compile(r"\b\d+(?:\.\d+)?\b")
_WORD_NUMBERS = {
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "twelve": "12",
    "fifteen": "15",
    "twenty": "20",
    "thirty": "30",
    "ninety": "90",
}


def tokenize(text: str) -> list[str]:
    tokens = _TOKEN.findall(text.lower())
    return [_WORD_NUMBERS.get(t, t) for t in tokens]


def stem(token: str) -> str:
    """Very light suffix stripping so 'days'/'day' and 'approved'/'approve' match."""
    for suffix in ("ing", "ed", "es", "s"):
        if len(token) > len(suffix) + 2 and token.endswith(suffix):
            return token[: -len(suffix)]
    return token


def content_terms(text: str) -> set[str]:
    return {stem(t) for t in tokenize(text) if t not in STOPWORDS}


def numbers(text: str) -> set[str]:
    words = {_WORD_NUMBERS[w] for w in re.findall(r"[a-z]+", text.lower()) if w in _WORD_NUMBERS}
    digits = {n.rstrip("0").rstrip(".") if "." in n else n for n in _NUMBER.findall(text.replace(",", ""))}
    return digits | words


def split_sentences(text: str) -> list[str]:
    """Split prose into sentences, keeping abbreviations like 'e.g.' and decimals intact."""
    text = re.sub(r"\s+", " ", text.strip())
    if not text:
        return []
    protected = re.sub(r"\b(e\.g|i\.e|etc|vs|approx|Dr|Mr|Ms|Inc)\.", lambda m: m.group(0).replace(".", "<DOT>"), text)
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])", protected)
    return [p.replace("<DOT>", ".").strip() for p in parts if p.strip()]
