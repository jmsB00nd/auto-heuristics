import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BudgetTracker:
    """Dual stopping criterion for the search: function-evaluation count and
    wall-clock time. The search stops as soon as EITHER limit is reached.

    ``max_seconds=None`` disables the time limit (FE-only, original behavior).
    The clock starts at construction, so it covers the whole pipeline.
    """
    max_fe: int
    max_seconds: Optional[float] = None
    current: int = 0
    _start: float = field(default_factory=time.monotonic, init=False, repr=False)

    def increment(self, n: int = 1) -> None:
        self.current += n

    def elapsed(self) -> float:
        return time.monotonic() - self._start

    def fe_exhausted(self) -> bool:
        return self.current >= self.max_fe

    def time_exhausted(self) -> bool:
        return self.max_seconds is not None and self.elapsed() >= self.max_seconds

    def exhausted(self) -> bool:
        return self.fe_exhausted() or self.time_exhausted()

    def remaining(self) -> int:
        """Effective FE the search may still spend. Returns 0 once the time
        limit is hit, so every FE-based gate also respects the deadline."""
        if self.time_exhausted():
            return 0
        return max(0, self.max_fe - self.current)

    def stop_reason(self) -> Optional[str]:
        """Why the search would stop now: ``'time'``, ``'fe'``, or ``None``."""
        if self.time_exhausted():
            return "time"
        if self.fe_exhausted():
            return "fe"
        return None
