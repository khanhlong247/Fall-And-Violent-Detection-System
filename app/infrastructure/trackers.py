from __future__ import annotations

from itertools import count
from math import hypot

from app.domain.entities import Body


class _SimpleTrackIdGenerator:
    _counter = count(1)

    @classmethod
    def next_id(cls) -> int:
        return next(cls._counter)


class TrackerIoU:
    """
    Placeholder tracker nhẹ để giữ interface giống code cũ.
    Nếu bạn đã có TrackerIoU thực tế, chỉ cần thay file này bằng implementation cũ.
    """

    def __init__(self) -> None:
        self.previous_bodies: list[Body] = []

    @staticmethod
    def _center(body: Body) -> tuple[float, float]:
        return ((body.xmin + body.xmax) / 2.0, (body.ymin + body.ymax) / 2.0)

    def apply(self, bodies: list[Body], timestamp: float) -> list[Body]:
        used_previous: set[int] = set()
        for body in bodies:
            best_idx = None
            best_dist = float("inf")
            cx, cy = self._center(body)

            for idx, prev in enumerate(self.previous_bodies):
                if idx in used_previous:
                    continue
                px, py = self._center(prev)
                dist = hypot(cx - px, cy - py)
                if dist < best_dist:
                    best_dist = dist
                    best_idx = idx

            if best_idx is not None and best_dist < 80:
                body.track_id = self.previous_bodies[best_idx].track_id
                used_previous.add(best_idx)
            else:
                body.track_id = _SimpleTrackIdGenerator.next_id()

        self.previous_bodies = bodies
        return bodies


class TrackerOKS(TrackerIoU):
    """
    Dùng chung logic đơn giản để không làm vỡ kiến trúc.
    Bạn có thể thay bằng TrackerOKS thật của dự án cũ.
    """
    pass
