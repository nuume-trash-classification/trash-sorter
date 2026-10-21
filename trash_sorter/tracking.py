"""Belt-aware multi-object tracker.

The camera sees each object in several consecutive frames. The tracker merges those
detections into one ``Track`` per object. Objects move at roughly the belt speed along +X,
but round objects roll and slip a little, so each track also fits its *own* velocity
from its observations (camera data only) and uses that to predict where it will be.
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
    observations: list[tuple[float, np.ndarray]] = field(default_factory=list)
    velocity: np.ndarray | None = None  # fitted (vx, vy); None = not enough data yet

    def position_at(self, t: float, belt_speed: float = config.BELT_SPEED) -> np.ndarray:
        """Predicted pick point at time ``t``."""
        if self.velocity is None:
            return predict_position(self.position, self.t_observed, t, belt_speed)
        p = self.position.copy()
        p[:2] += self.velocity * (t - self.t_observed)
        return p

    def add_observation(
        self, t: float, position: np.ndarray, belt_speed: float, min_span: float = 0.5, window: float = 6.0
    ) -> None:
        """Add a measurement and refit position + velocity (least squares over the last ``window`` s)."""
        self.observations.append((t, position.copy()))
        self.observations = [(ti, pi) for ti, pi in self.observations if t - ti <= window]
        ts = np.array([ti for ti, _ in self.observations])
        ps = np.array([pi for _, pi in self.observations])
        self.t_observed = t
        if ts[-1] - ts[0] < min_span:
            # Too short to fit: average the measurements, shifted to time t at belt speed.
            shifted = ps.copy()
            shifted[:, 0] += belt_speed * (t - ts)
            self.position = shifted.mean(axis=0)
            return
        A = np.column_stack([np.ones_like(ts), ts - t])
        coef, *_ = np.linalg.lstsq(A, ps[:, :2], rcond=None)
        vx = float(np.clip(coef[1, 0], 0.6 * belt_speed, 1.3 * belt_speed))  # sanity limits
        vy = float(np.clip(coef[1, 1], -0.05, 0.05))
        self.velocity = np.array([vx, vy])
        self.position = np.array([coef[0, 0], coef[0, 1], ps[:, 2].mean()])

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
            best.add_observation(det.t_observed, det.position, self.belt_speed)
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
