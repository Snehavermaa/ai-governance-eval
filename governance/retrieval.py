"""Document store with passage-level BM25 retrieval."""

import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .text import split_sentences, stem, tokenize


@dataclass(frozen=True)
class Passage:
    id: str  # "<doc_id>#<n>"
    doc_id: str
    title: str
    text: str


@dataclass
class Retrieved:
    passage: Passage
    score: float


def load_corpus(directory: str | Path) -> list[tuple[str, str, str]]:
    """Load *.md files as (doc_id, title, body). Title = first '# ' heading, else file stem."""
    docs = []
    for path in sorted(Path(directory).glob("*.md")):
        lines = path.read_text(encoding="utf-8").splitlines()
        title = next((ln[2:].strip() for ln in lines if ln.startswith("# ")), path.stem)
        body = "\n".join(ln for ln in lines if not ln.startswith("# "))
        docs.append((path.stem, title, body))
    return docs


def chunk(doc_id: str, title: str, body: str, sentences_per_passage: int = 2) -> list[Passage]:
    passages: list[Passage] = []
    for para in [p for p in body.split("\n\n") if p.strip()]:
        sents = split_sentences(para)
        for i in range(0, len(sents), sentences_per_passage):
            text = " ".join(sents[i : i + sentences_per_passage])
            passages.append(Passage(f"{doc_id}#{len(passages)}", doc_id, title, text))
    return passages


@dataclass
class BM25Index:
    passages: list[Passage]
    k1: float = 1.5
    b: float = 0.75
    _tf: list[Counter] = field(init=False, repr=False)
    _idf: dict[str, float] = field(init=False, repr=False)
    _avgdl: float = field(init=False, repr=False)

    def __post_init__(self):
        self._tf = [Counter(stem(t) for t in tokenize(p.title + " " + p.text)) for p in self.passages]
        n = len(self.passages)
        df = Counter(term for tf in self._tf for term in tf)
        self._idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}
        self._avgdl = (sum(sum(tf.values()) for tf in self._tf) / n) if n else 0.0

    @classmethod
    def from_directory(cls, directory: str | Path, sentences_per_passage: int = 2) -> "BM25Index":
        passages = [p for d in load_corpus(directory) for p in chunk(*d, sentences_per_passage)]
        return cls(passages)

    @classmethod
    def from_texts(cls, texts: dict[str, str]) -> "BM25Index":
        return cls([p for doc_id, body in texts.items() for p in chunk(doc_id, doc_id, body)])

    def search(self, query: str, k: int = 4) -> list[Retrieved]:
        terms = [stem(t) for t in tokenize(query)]
        scored = []
        for passage, tf in zip(self.passages, self._tf):
            dl = sum(tf.values())
            score = 0.0
            for t in terms:
                if t in tf:
                    f = tf[t]
                    score += self._idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self._avgdl))
            if score > 0:
                scored.append(Retrieved(passage, score))
        scored.sort(key=lambda r: (-r.score, r.passage.id))
        return scored[:k]
