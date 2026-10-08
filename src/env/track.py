from typing import NamedTuple

import numpy as np
from scipy.interpolate import splprep, splev

TRACK_WIDTH = 44.0
LEGACY_TRACK_WIDTH = 70.0
N_SAMPLES = 2000
LEGACY_N_SAMPLES = 800
SHANGHAI_CIRCUIT_LENGTH_MIN = 3000.0
RAY_ANGLES = np.deg2rad([-90.0, -45.0, 0.0, 45.0, 90.0])

_CONTROL_POINTS = np.array([
    [200, 530],
    [600, 530],
    [720, 480],
    [720, 380],
    [450, 380],
    [350, 320],
    [450, 220],
    [650, 180],
    [650, 100],
    [450,  80],
    [250, 120],
    [150, 250],
    [ 80, 380],
    [120, 480],
], dtype=float)

SHANGHAI_CONTROL_POINTS = np.array([
    [364.0, 354.0],
    [373.0, 330.0],
    [383.0, 306.0],
    [392.0, 282.0],
    [400.0, 258.0],
    [408.0, 234.0],
    [416.0, 210.0],
    [426.0, 186.0],
    [436.0, 162.0],
    [446.0, 138.0],
    [466.0, 117.0],
    [490.0, 110.0],
    [514.0, 112.0],
    [538.0, 126.0],
    [549.0, 150.0],
    [542.0, 174.0],
    [519.0, 188.0],
    [498.0, 169.0],
    [474.0, 158.0],
    [461.0, 182.0],
    [473.0, 206.0],
    [497.0, 223.0],
    [521.0, 225.0],
    [545.0, 221.0],
    [569.0, 217.0],
    [593.0, 213.0],
    [617.0, 209.0],
    [641.0, 207.0],
    [665.0, 204.0],
    [689.0, 200.0],
    [713.0, 202.0],
    [737.0, 208.0],
    [761.0, 214.0],
    [785.0, 221.0],
    [809.0, 228.0],
    [833.0, 234.0],
    [837.0, 258.0],
    [813.0, 266.0],
    [789.0, 273.0],
    [765.0, 275.0],
    [741.0, 273.0],
    [717.0, 271.0],
    [693.0, 269.0],
    [669.0, 268.0],
    [645.0, 268.0],
    [621.0, 270.0],
    [597.0, 283.0],
    [577.0, 306.0],
    [568.0, 330.0],
    [567.0, 354.0],
    [572.0, 378.0],
    [586.0, 402.0],
    [600.0, 426.0],
    [612.0, 450.0],
    [611.0, 474.0],
    [596.0, 498.0],
    [572.0, 511.0],
    [548.0, 512.0],
    [524.0, 508.0],
    [500.0, 506.0],
    [486.0, 528.0],
    [494.0, 552.0],
    [504.0, 576.0],
    [528.0, 583.0],
    [552.0, 583.0],
    [576.0, 583.0],
    [600.0, 585.0],
    [624.0, 585.0],
    [648.0, 585.0],
    [672.0, 587.0],
    [696.0, 588.0],
    [720.0, 588.0],
    [744.0, 588.0],
    [768.0, 590.0],
    [792.0, 590.0],
    [812.0, 578.0],
    [805.0, 554.0],
    [816.0, 530.0],
    [840.0, 527.0],
    [864.0, 535.0],
    [885.0, 555.0],
    [891.0, 579.0],
    [886.0, 603.0],
    [871.0, 627.0],
    [847.0, 643.0],
    [823.0, 650.0],
    [799.0, 650.0],
    [775.0, 650.0],
    [751.0, 650.0],
    [727.0, 650.0],
    [703.0, 650.0],
    [679.0, 650.0],
    [655.0, 650.0],
    [631.0, 650.0],
    [607.0, 650.0],
    [583.0, 650.0],
    [559.0, 650.0],
    [535.0, 650.0],
    [511.0, 652.0],
    [487.0, 652.0],
    [463.0, 652.0],
    [439.0, 652.0],
    [415.0, 652.0],
    [391.0, 652.0],
    [367.0, 652.0],
    [343.0, 652.0],
    [319.0, 652.0],
    [295.0, 652.0],
    [271.0, 652.0],
    [247.0, 652.0],
    [223.0, 652.0],
    [199.0, 652.0],
    [175.0, 652.0],
    [151.0, 652.0],
    [127.0, 652.0],
    [103.0, 654.0],
    [79.0, 655.0],
    [55.0, 653.0],
    [58.0, 633.0],
    [82.0, 621.0],
    [106.0, 618.0],
    [130.0, 618.0],
    [154.0, 617.0],
    [178.0, 616.0],
    [202.0, 615.0],
    [226.0, 613.0],
    [250.0, 612.0],
    [274.0, 606.0],
    [282.0, 582.0],
    [290.0, 558.0],
    [299.0, 534.0],
    [309.0, 510.0],
    [318.0, 486.0],
    [326.0, 462.0],
    [334.0, 438.0],
    [342.0, 414.0],
    [352.0, 390.0],
    [362.0, 366.0],
], dtype=float)


class TrackState(NamedTuple):
    progress: float
    lateral: float
    track_heading: float
    on_track: bool
    arc_length: float


class Track:
    def __init__(self, width: float = TRACK_WIDTH, circuit: str = "shanghai", n_samples: int | None = None) -> None:
        self.circuit = circuit
        self.half_width = width / 2.0
        if n_samples is not None:
            self.n_samples = n_samples
        else:
            self.n_samples = N_SAMPLES if circuit == "shanghai" else LEGACY_N_SAMPLES
        self._build()

    def _build(self) -> None:
        if self.circuit == "shanghai":
            pts = SHANGHAI_CONTROL_POINTS
        else:
            pts = _CONTROL_POINTS

        tck, _ = splprep([pts[:, 0], pts[:, 1]], s=0, per=True)
        u = np.linspace(0, 1, self.n_samples, endpoint=False)
        cx, cy = splev(u, tck)
        dx, dy = splev(u, tck, der=1)

        mag = np.hypot(dx, dy)
        tx, ty = dx / mag, dy / mag
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

        def _segs(pts_arr: np.ndarray) -> np.ndarray:
            return np.stack([pts_arr, np.roll(pts_arr, -1, axis=0)], axis=1)

        self._wall_segs: np.ndarray = np.concatenate(
            [_segs(self.inner), _segs(self.outer)], axis=0
        )

    def nearest_idx(self, pos: np.ndarray) -> int:
        return int(np.argmin(np.linalg.norm(self.centerline - pos, axis=1)))

    def get_track_state(self, pos: np.ndarray) -> TrackState:
        idx = self.nearest_idx(pos)
        delta = pos - self.centerline[idx]
        lateral = float(np.dot(delta, self.normals[idx]))
        arc = self.arc_lengths[idx] + float(np.dot(delta, self.tangents[idx]))
        if abs(arc) < 1e-10:
            arc = 0.0
        progress = (arc % self.total_length) / self.total_length
        heading = float(np.arctan2(self.tangents[idx, 1], self.tangents[idx, 0]))
        on_track = abs(lateral) <= self.half_width
        return TrackState(progress, lateral, heading, on_track, arc)

    def get_starting_grid(self) -> dict[str, tuple[np.ndarray, float]]:
        idx_p1 = int(len(self.centerline) * 0.985)
        idx_p2 = int(len(self.centerline) * 0.970)
        heading_p1 = float(np.arctan2(self.tangents[idx_p1, 1], self.tangents[idx_p1, 0]))
        heading_p2 = float(np.arctan2(self.tangents[idx_p2, 1], self.tangents[idx_p2, 0]))
        pos_p1 = self.centerline[idx_p1] - self.normals[idx_p1] * (self.half_width * 0.35)
        pos_p2 = self.centerline[idx_p2] + self.normals[idx_p2] * (self.half_width * 0.35)
        return {
            "agent_0": (pos_p1, heading_p1),
            "agent_1": (pos_p2, heading_p2),
        }

    def ray_distances(
        self, pos: np.ndarray, heading: float, max_dist: float = 200.0
    ) -> np.ndarray:
        return np.array([self._cast_ray(pos, heading + a, max_dist) for a in RAY_ANGLES])

    def _cast_ray(self, pos: np.ndarray, angle: float, max_dist: float) -> float:
        d = np.array([np.cos(angle), np.sin(angle)])
        A = self._wall_segs[:, 0]
        B = self._wall_segs[:, 1]
        
        dists_sq = (A[:, 0] - pos[0]) ** 2 + (A[:, 1] - pos[1]) ** 2
        nearby = dists_sq < (max_dist + 50.0) ** 2
        if not np.any(nearby):
            return float(max_dist)
        A = A[nearby]
        B = B[nearby]

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
