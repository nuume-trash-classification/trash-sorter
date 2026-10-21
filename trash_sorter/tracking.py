"""Belt-aware multi-object tracker.

The camera sees each object in several consecutive frames. The tracker merges those
detections into one ``Track`` per object, using the fact that everything on the belt
moves at the (known) belt speed along +X.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from trash_sorter import config
from trash_sorter.geometry import predict_position
from trash_sorter.perception import Detection


@dataclass
class Track:
    """One physical object on the belt, as believed by the perception pipeline."""

    track_id: int
    position: np.ndarray  # last estimated pick point
    t_observed: float  # time ``position`` refers to
    class_scores: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    hits: int = 0
    claimed: bool = False  # the arm has committed to picking it
    done: bool = False  # picked, abandoned or gone

    def position_at(self, t: float, belt_speed: float = config.BELT_SPEED) -> np.ndarray:
        """Predicted pick point at time ``t``."""
        return predict_position(self.position, self.t_observed, t, belt_speed)

    @property
    def class_name(self) -> str:
        return max(self.class_scores, key=self.class_scores.__getitem__)

    @property
    def confidence(self) -> float:
        """Mean confidence of the winning class over all hits."""
        return self.class_scores[self.class_name] / max(self.hits, 1)


class BeltTracker:
    """Greedy nearest-neighbour association in belt coordinates."""

    def __init__(
        self,
        match_radius: float = 0.06,
        belt_speed: float = config.BELT_SPEED,
        drop_x: float = config.BELT_X_END,
    ) -> None:
        self.match_radius = match_radius
        self.belt_speed = belt_speed
        self.drop_x = drop_x
        self.tracks: list[Track] = []
        self._next_id = 0

    def update(self, detections: list[Detection]) -> None:
        """Merge one frame of detections (all with the same ``t_observed``)."""
        free = [tr for tr in self.tracks if not tr.done]
        used: set[int] = set()
        for det in sorted(detections, key=lambda d: -d.confidence):
            best, best_d = None, self.match_radius
            for tr in free:
                if tr.track_id in used:
                    continue
                pred = tr.position_at(det.t_observed, self.belt_speed)
                d = float(np.linalg.norm(pred[:2] - det.position[:2]))
                if d < best_d:
                    best, best_d = tr, d
            if best is None:
                best = Track(self._next_id, det.position.copy(), det.t_observed)
                self._next_id += 1
                self.tracks.append(best)
            else:
                # Average the old (predicted) and new estimate; reduces pixel noise.
                pred = best.position_at(det.t_observed, self.belt_speed)
                w = 1.0 / (best.hits + 1)
                best.position = (1 - w) * pred + w * det.position
                best.t_observed = det.t_observed
            best.class_scores[det.class_name] += det.confidence
            best.hits += 1
            used.add(best.track_id)

    def prune(self, t: float) -> None:
        """Forget tracks that have left the belt."""
        for tr in self.tracks:
            if not tr.done and tr.position_at(t, self.belt_speed)[0] > self.drop_x:
                tr.done = True
        self.tracks = [tr for tr in self.tracks if not tr.done or tr.claimed]

    def active(self) -> list[Track]:
        return [tr for tr in self.tracks if not tr.done]
