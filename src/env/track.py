import numpy as np
from scipy.interpolate import splprep, splev

TRACK_WIDTH = 65.0
N_SAMPLES = 800
RAY_ANGLES = np.deg2rad([-90.0, -45.0, 0.0, 45.0, 90.0])

# F1-inspired circuit: main straight → T1 right complex → fast sweep → chicane → left hairpin
_CONTROL_POINTS = np.array([
    [190, 490],  # start/finish
    [560, 490],
    [660, 455],  # T1
    [720, 360],
    [700, 255],
    [620, 170],  # fast sweep
    [480, 115],
    [360, 110],  # top straight
    [265, 145],  # chicane
    [235, 210],
    [200, 290],
    [105, 360],  # left hairpin
    [ 90, 425],
    [135, 472],
], dtype=float)


class Track:
    def __init__(self, width: float = TRACK_WIDTH) -> None:
        self.half_width = width / 2.0
        self._build()

    def _build(self) -> None:
        pts = _CONTROL_POINTS
        tck, _ = splprep([pts[:, 0], pts[:, 1]], s=0, per=True)
        u = np.linspace(0, 1, N_SAMPLES, endpoint=False)
        cx, cy = splev(u, tck)
        dx, dy = splev(u, tck, der=1)

        mag = np.hypot(dx, dy)
        tx, ty = dx / mag, dy / mag
        # Left-side normal (inward for a clockwise track in screen coords)
        nx, ny = -ty, tx

        cl = np.column_stack([cx, cy])
        tang = np.column_stack([tx, ty])
        norm = np.column_stack([nx, ny])
        hw = self.half_width

        self.centerline: np.ndarray = cl
        self.tangents: np.ndarray = tang
        self.normals: np.ndarray = norm
        self.inner: np.ndarray = cl + norm * hw
        self.outer: np.ndarray = cl - norm * hw

        step_lens = np.linalg.norm(np.diff(cl, axis=0, append=cl[:1]), axis=1)
        self.arc_lengths: np.ndarray = np.concatenate([[0.0], np.cumsum(step_lens[:-1])])
        self.total_length: float = float(self.arc_lengths[-1] + step_lens[-1])

        def _segs(pts: np.ndarray) -> np.ndarray:
            return np.stack([pts, np.roll(pts, -1, axis=0)], axis=1)

        self._wall_segs: np.ndarray = np.concatenate(
            [_segs(self.inner), _segs(self.outer)], axis=0
        )

    def nearest_idx(self, pos: np.ndarray) -> int:
        return int(np.argmin(np.linalg.norm(self.centerline - pos, axis=1)))

    def get_track_state(self, pos: np.ndarray) -> tuple:
        """Single nearest-point lookup returning (progress, lateral, track_heading, on_track)."""
        idx = self.nearest_idx(pos)
        delta = pos - self.centerline[idx]
        lateral = float(np.dot(delta, self.normals[idx]))
        arc = self.arc_lengths[idx] + float(np.dot(delta, self.tangents[idx]))
        progress = (arc % self.total_length) / self.total_length
        heading = float(np.arctan2(self.tangents[idx, 1], self.tangents[idx, 0]))
        on_track = abs(lateral) <= self.half_width
        return progress, lateral, heading, on_track

    def ray_distances(
        self, pos: np.ndarray, heading: float, max_dist: float = 200.0
    ) -> np.ndarray:
        return np.array([self._cast_ray(pos, heading + a, max_dist) for a in RAY_ANGLES])

    def _cast_ray(self, pos: np.ndarray, angle: float, max_dist: float) -> float:
        d = np.array([np.cos(angle), np.sin(angle)])
        A = self._wall_segs[:, 0]
        B = self._wall_segs[:, 1]
        r = B - A
        AP = A - pos

        dxr = d[0] * r[:, 1] - d[1] * r[:, 0]
        APxr = AP[:, 0] * r[:, 1] - AP[:, 1] * r[:, 0]
        APxd = AP[:, 0] * d[1] - AP[:, 1] * d[0]

        valid = np.abs(dxr) > 1e-10
        safe = np.where(valid, dxr, 1.0)
        t = np.where(valid, APxr / safe, np.inf)
        s = np.where(valid, APxd / safe, -1.0)

        mask = valid & (t > 1e-3) & (s >= 0.0) & (s <= 1.0)
        return float(np.min(np.where(mask, t, max_dist)))
