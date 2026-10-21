"""Sorting metrics: pick success rate, miss rate, sorting accuracy."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from trash_sorter import config

END_BIN = "end_bin"
LOST = "lost"  # fell on the floor / stuck


@dataclass
class SortingStats:
    """Counts filled in by the simulator; ground truth is used here for *evaluation only*."""

    attempts: int = 0
    successful_picks: int = 0  # grasped AND ended up in the bin the arm aimed for
    outcomes: list[tuple[str, str]] = field(default_factory=list)  # (true class, final location)

    def record_attempt(self, success: bool) -> None:
        self.attempts += 1
        self.successful_picks += int(success)

    def record_outcome(self, true_class: str, location: str) -> None:
        self.outcomes.append((true_class, location))

    def summary(self) -> dict[str, float | int | dict[str, int]]:
        n = len(self.outcomes)
        locs = Counter(loc for _, loc in self.outcomes)
        in_class_bins = [(c, loc) for c, loc in self.outcomes if loc in config.CLASS_NAMES]
        correct = sum(c == loc for c, loc in in_class_bins)
        return {
            "objects_finished": n,
            "pick_attempts": self.attempts,
            "successful_picks": self.successful_picks,
            "pick_success_rate": self.successful_picks / self.attempts if self.attempts else 0.0,
            "miss_rate": locs[END_BIN] / n if n else 0.0,
            "lost_rate": locs[LOST] / n if n else 0.0,
            "sorting_accuracy": correct / len(in_class_bins) if in_class_bins else 0.0,
            "locations": dict(locs),
        }

    def format(self) -> str:
        s = self.summary()
        return "\n".join(
            [
                "==== Sorting metrics ====",
                f"objects finished      : {s['objects_finished']}",
                f"pick attempts         : {s['pick_attempts']}",
                f"pick success rate     : {s['pick_success_rate']:.1%}  ({s['successful_picks']}/{s['pick_attempts']})",
                f"miss rate (end bin)   : {s['miss_rate']:.1%}",
                f"lost (floor / stuck)  : {s['lost_rate']:.1%}",
                f"sorting accuracy      : {s['sorting_accuracy']:.1%}  (class bin == true class)",
                f"final locations       : {s['locations']}",
            ]
        )
