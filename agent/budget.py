from dataclasses import dataclass


@dataclass
class BudgetTracker:
    max_fe: int
    current: int = 0

    def increment(self, n: int = 1) -> None:
        self.current += n

    def remaining(self) -> int:
        return max(0, self.max_fe - self.current)

    def exhausted(self) -> bool:
        return self.current >= self.max_fe
