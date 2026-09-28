"""Lightweight per-request tracing: timed spans written as one JSON document per trace."""

import json
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Span:
    name: str
    started_at: float
    duration_ms: float = 0.0
    attributes: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass
class Trace:
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    started_at: float = field(default_factory=time.time)
    spans: list[Span] = field(default_factory=list)

    @contextmanager
    def span(self, name: str, **attributes: Any):
        s = Span(name, time.time(), attributes=dict(attributes))
        t0 = time.perf_counter()
        try:
            yield s
        except Exception as exc:
            s.error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            s.duration_ms = round((time.perf_counter() - t0) * 1000, 3)
            self.spans.append(s)

    def to_dict(self) -> dict[str, Any]:
        return {"trace_id": self.trace_id, "started_at": self.started_at, "spans": [asdict(s) for s in self.spans]}


class TraceStore:
    """Persists traces as JSON files; keeps them in memory when no directory is given."""

    def __init__(self, directory: str | Path | None = None):
        self.directory = Path(directory) if directory else None
        self._memory: dict[str, dict] = {}
        if self.directory:
            self.directory.mkdir(parents=True, exist_ok=True)

    def save(self, trace: Trace, extra: dict[str, Any] | None = None) -> None:
        doc = trace.to_dict() | (extra or {})
        if self.directory:
            (self.directory / f"{trace.trace_id}.json").write_text(json.dumps(doc, indent=2, default=str))
        else:
            self._memory[trace.trace_id] = doc

    def get(self, trace_id: str) -> dict | None:
        if not trace_id.isalnum():  # blocks path traversal like "../x"
            return None
        if self.directory:
            path = self.directory / f"{trace_id}.json"
            return json.loads(path.read_text()) if path.exists() else None
        return self._memory.get(trace_id)
