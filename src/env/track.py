from __future__ import annotations

from typing import NamedTuple
import numpy as np
from scipy.interpolate import splprep, splev, CubicSpline

TRACK_WIDTH = 70.0
SHANGHAI_TRACK_WIDTH = 44.0
LEGACY_TRACK_WIDTH = 70.0
N_SAMPLES = 2000
LEGACY_N_SAMPLES = 800
SHANGHAI_CIRCUIT_LENGTH_MIN = 3000.0
RAY_ANGLES = np.deg2rad([-90.0, -45.0, 0.0, 45.0, 90.0])

_CONTROL_POINTS = np.array([
    [200.0, 530.0],
    [600.0, 530.0],
    [720.0, 480.0],
    [720.0, 380.0],
    [450.0, 380.0],
    [350.0, 320.0],
    [450.0, 220.0],
    [650.0, 180.0],
    [650.0, 100.0],
    [450.0,  80.0],
    [250.0, 120.0],
    [150.0, 250.0],
    [ 80.0, 380.0],
    [120.0, 480.0],
], dtype=float)

SHANGHAI_CONTROL_POINTS = np.array([
    [472.0, 800.0],
    [703.0, 140.0],
    [740.0, 95.0], [800.0, 75.0], [875.0, 95.0], [915.0, 145.0], [895.0, 185.0], [860.0, 195.0],
    [820.0, 180.0], [790.0, 155.0], [765.0, 165.0], [760.0, 200.0], [785.0, 245.0], [820.0, 270.0],
    [840.0, 275.0],
    [1060.0, 230.0], [1085.0, 230.0], [1110.0, 230.0],
    [1270.0, 268.0], [1305.0, 275.0], [1325.0, 300.0], [1295.0, 330.0], [1210.0, 335.0],
    [995.0, 325.0],
    [940.0, 340.0], [905.0, 380.0], [893.0, 430.0], [905.0, 480.0], [935.0, 530.0],
    [960.0, 575.0], [965.0, 610.0], [945.0, 645.0], [905.0, 668.0], [860.0, 672.0],
    [815.0, 662.0], [785.0, 675.0], [780.0, 710.0], [800.0, 760.0], [815.0, 775.0],
    [830.0, 778.0],
    [1220.0, 790.0],
    [1255.0, 785.0], [1270.0, 760.0], [1270.0, 725.0], [1285.0, 700.0], [1325.0, 695.0],
    [1365.0, 720.0], [1385.0, 765.0], [1365.0, 820.0], [1335.0, 860.0], [1290.0, 880.0],
    [1250.0, 882.0],
    [150.0, 882.0],
    [115.0, 876.0], [95.0, 845.0], [115.0, 815.0], [165.0, 826.0],
    [230.0, 826.0],
    [440.0, 826.0],
    [458.0, 822.0], [466.0, 812.0],
    [472.0, 800.0]
], dtype=float)

_SHANGHAI_CACHE: dict[float, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]] = {}


class TrackState(NamedTuple):
    progress: float
    lateral: float
    track_heading: float
    on_track: bool
    arc_length: float
    curvature: float = 0.0


def _build_shanghai_geometry(actual_width: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    if actual_width in _SHANGHAI_CACHE:
        cached = _SHANGHAI_CACHE[actual_width]
        return (
            cached[0].copy(),
            cached[1].copy(),
            cached[2].copy(),
            cached[3].copy(),
            cached[4].copy(),
            cached[5],
        )

    p_s0_start = np.array([472.0, 800.0])
    p_s0_end = np.array([703.0, 140.0])
    p_sf = p_s0_start + 0.55 * (p_s0_end - p_s0_start)

    straights_def = [
        (p_sf, p_s0_end),
        (np.array([840.0, 275.0]), np.array([1060.0, 230.0])),
        (np.array([1110.0, 230.0]), np.array([1270.0, 268.0])),
        (np.array([1210.0, 335.0]), np.array([995.0, 325.0])),
        (np.array([830.0, 778.0]), np.array([1220.0, 790.0])),
        (np.array([1250.0, 882.0]), np.array([150.0, 882.0])),
        (np.array([230.0, 826.0]), np.array([440.0, 826.0])),
        (p_s0_start, p_sf),
    ]

    corners_def = [
        np.array([
            [703.0, 140.0],
            [740.0, 95.0], [800.0, 75.0], [875.0, 95.0], [915.0, 145.0], [895.0, 185.0], [860.0, 195.0],
            [820.0, 180.0], [790.0, 155.0], [765.0, 165.0], [760.0, 200.0], [785.0, 245.0], [820.0, 270.0],
            [840.0, 275.0],
        ]),
        np.array([[1060.0, 230.0], [1085.0, 230.0], [1110.0, 230.0]]),
        np.array([[1270.0, 268.0], [1305.0, 275.0], [1325.0, 300.0], [1295.0, 330.0], [1210.0, 335.0]]),
        np.array([
            [995.0, 325.0],
            [940.0, 340.0], [905.0, 380.0], [893.0, 430.0], [905.0, 480.0], [935.0, 530.0],
            [960.0, 575.0], [965.0, 610.0], [945.0, 645.0], [905.0, 668.0], [860.0, 672.0],
            [815.0, 662.0], [785.0, 675.0], [780.0, 710.0], [800.0, 760.0], [815.0, 775.0],
            [830.0, 778.0],
        ]),
        np.array([
            [1220.0, 790.0],
            [1255.0, 785.0], [1270.0, 760.0], [1270.0, 725.0], [1285.0, 700.0], [1325.0, 695.0],
            [1365.0, 720.0], [1385.0, 765.0], [1365.0, 820.0], [1335.0, 860.0], [1290.0, 880.0],
            [1250.0, 882.0],
        ]),
        np.array([
            [150.0, 882.0],
            [115.0, 876.0], [95.0, 845.0], [115.0, 815.0], [165.0, 826.0],
            [230.0, 826.0],
        ]),
        np.array([
            [440.0, 826.0],
            [458.0, 822.0], [466.0, 812.0],
            [472.0, 800.0],
        ]),
    ]

    def _sample(scale: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        s_scaled = [(s0 * scale, s1 * scale) for s0, s1 in straights_def]
        c_scaled = [c * scale for c in corners_def]
        raw_pts: list[np.ndarray] = []
        raw_tang: list[np.ndarray] = []
        raw_curv: list[np.ndarray] = []

        for i in range(len(c_scaled)):
            s0, s1 = s_scaled[i]
            l_s = float(np.linalg.norm(s1 - s0))
            t_s = (s1 - s0) / l_s
            n_s = max(2, int(np.ceil(l_s * 4)))
            u_s = np.linspace(0.0, l_s, n_s, endpoint=False)
            raw_pts.append(s0 + np.outer(u_s, t_s))
            raw_tang.append(np.tile(t_s, (n_s, 1)))
            raw_curv.append(np.zeros(n_s))

            pts_c = c_scaled[i]
            next_s0, next_s1 = s_scaled[(i + 1) % len(s_scaled)]
            t_next = (next_s1 - next_s0) / np.linalg.norm(next_s1 - next_s0)
            chords = np.linalg.norm(np.diff(pts_c, axis=0), axis=1)
            u_c = np.concatenate([[0.0], np.cumsum(chords)])
            cs = CubicSpline(u_c, pts_c, bc_type=((1, t_s), (1, t_next)))

            u_dense = np.linspace(0.0, u_c[-1], max(20, int(np.ceil(u_c[-1] * 10))))
            dense_pts = cs(u_dense)
            step_lens = np.linalg.norm(np.diff(dense_pts, axis=0), axis=1)
            l_approx = float(np.sum(step_lens))

            n_c = max(2, int(np.ceil(l_approx * 4)))
            u_eval = np.linspace(0.0, u_c[-1], n_c, endpoint=False)
            d1 = cs(u_eval, 1)
            d2 = cs(u_eval, 2)
            mag = np.hypot(d1[:, 0], d1[:, 1])
            tang_c = d1 / mag[:, None]
            curv_c = (d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]) / (mag ** 3 + 1e-12)
            raw_pts.append(cs(u_eval))
            raw_tang.append(tang_c)
            raw_curv.append(curv_c)

        s0, s1 = s_scaled[-1]
        l_s = float(np.linalg.norm(s1 - s0))
        t_s = (s1 - s0) / l_s
        n_s = max(2, int(np.ceil(l_s * 4)))
        u_s = np.linspace(0.0, l_s, n_s, endpoint=False)
        raw_pts.append(s0 + np.outer(u_s, t_s))
        raw_tang.append(np.tile(t_s, (n_s, 1)))
        raw_curv.append(np.zeros(n_s))

        all_pts = np.vstack(raw_pts)
        all_tang = np.vstack(raw_tang)
        all_curv = np.concatenate(raw_curv)

        dists = np.linalg.norm(np.diff(all_pts, axis=0, append=all_pts[:1]), axis=1)
        cum_s = np.concatenate([[0.0], np.cumsum(dists[:-1])])
        total_l = float(np.sum(dists))

        target_s = np.arange(0.0, total_l, 2.0)
        cl = np.column_stack([
            np.interp(target_s, cum_s, all_pts[:, 0]),
            np.interp(target_s, cum_s, all_pts[:, 1]),
        ])
        tx = np.interp(target_s, cum_s, all_tang[:, 0])
        ty = np.interp(target_s, cum_s, all_tang[:, 1])
        t_mag = np.hypot(tx, ty)
        tang = np.column_stack([tx / t_mag, ty / t_mag])
        norm = np.column_stack([-tang[:, 1], tang[:, 0]])
        curv = np.interp(target_s, cum_s, all_curv)
        return cl, tang, norm, curv, target_s, total_l

    def _check(cl: np.ndarray, arc: np.ndarray, l_circ: float, w_track: float) -> float:
        min_dist_sq = 1e12
        chunk_size = 500
        n_pts = len(cl)
        for i in range(0, n_pts, chunk_size):
            chunk_pts = cl[i : i + chunk_size]
            chunk_arc = arc[i : i + chunk_size]
            d_arc = np.abs(chunk_arc[:, None] - arc[None, :])
            d_arc = np.minimum(d_arc, l_circ - d_arc)
            mask = d_arc > 2.0 * w_track
            diff = chunk_pts[:, None, :] - cl[None, :, :]
            dist_sq = np.sum(diff ** 2, axis=-1)
            dist_sq[~mask] = 1e12
            cur_min = float(np.min(dist_sq))
            if cur_min < min_dist_sq:
                min_dist_sq = cur_min
        return float(np.sqrt(min_dist_sq))

    cl, tang, norm, curv, arc, total_l = _sample(1.0)
    d_min = _check(cl, arc, total_l, actual_width)
    r_min = float(1.0 / (np.max(np.abs(curv)) + 1e-12))

    scale_d = (1.4 * actual_width) / d_min if d_min < 1.4 * actual_width else 1.0
    scale_r = (0.8 * actual_width) / r_min if r_min < 0.8 * actual_width else 1.0
    final_scale = max(1.0, scale_d, scale_r)

    if final_scale > 1.0:
        cl, tang, norm, curv, arc, total_l = _sample(final_scale)

    _SHANGHAI_CACHE[actual_width] = (cl, tang, norm, curv, arc, total_l)
    return cl, tang, norm, curv, arc, total_l


class Track:
    def __init__(self, width: float | None = None, circuit: str = "shanghai", n_samples: int | None = None) -> None:
        self.circuit = circuit
        actual_width = width if width is not None else (SHANGHAI_TRACK_WIDTH if circuit == "shanghai" else TRACK_WIDTH)
        self.half_width = actual_width / 2.0
        if n_samples is not None:
            self.n_samples = n_samples
        else:
            self.n_samples = N_SAMPLES if circuit == "shanghai" else LEGACY_N_SAMPLES
        self._build(actual_width)

    def _build(self, actual_width: float) -> None:
        if self.circuit == "shanghai":
            cl, tang, norm, curv, arc_lens, total_l = _build_shanghai_geometry(actual_width)
            self.centerline = cl
            self.tangents = tang
            self.normals = norm
            self.curvatures = curv
            self.arc_lengths = arc_lens
            self.total_length = total_l
            self.n_samples = len(cl)
        else:
            pts = _CONTROL_POINTS
            tck, _ = splprep([pts[:, 0], pts[:, 1]], s=0, per=True)
            u = np.linspace(0, 1, self.n_samples, endpoint=False)
            cx, cy = splev(u, tck)
            dx, dy = splev(u, tck, der=1)
            ddx, ddy = splev(u, tck, der=2)

            mag = np.hypot(dx, dy)
            tx, ty = dx / mag, dy / mag
            nx, ny = -ty, tx
            kappa = (dx * ddy - dy * ddx) / (mag ** 3 + 1e-9)

            self.centerline = np.column_stack([cx, cy])
            self.tangents = np.column_stack([tx, ty])
            self.normals = np.column_stack([nx, ny])
            self.curvatures = kappa

            step_lens = np.linalg.norm(np.diff(self.centerline, axis=0, append=self.centerline[:1]), axis=1)
            self.arc_lengths = np.concatenate([[0.0], np.cumsum(step_lens[:-1])])
            self.total_length = float(self.arc_lengths[-1] + step_lens[-1])

        hw = self.half_width
        self.inner = self.centerline + self.normals * hw
        self.outer = self.centerline - self.normals * hw

        def _segs(pts_arr: np.ndarray) -> np.ndarray:
            return np.stack([pts_arr, np.roll(pts_arr, -1, axis=0)], axis=1)

        self._wall_segs = np.concatenate(
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
        curv = float(self.curvatures[idx])
        return TrackState(progress, lateral, heading, on_track, arc, curv)

    def get_starting_grid(self) -> dict[str, tuple[np.ndarray, float]]:
        w = self.half_width * 2.0
        s_p1 = (self.total_length - 1.0 * w) % self.total_length
        s_p2 = (s_p1 - 0.57 * w) % self.total_length
        idx_p1 = int(np.clip(s_p1 / 2.0, 0, len(self.centerline) - 1))
        idx_p2 = int(np.clip(s_p2 / 2.0, 0, len(self.centerline) - 1))
        heading_p1 = float(np.arctan2(self.tangents[idx_p1, 1], self.tangents[idx_p1, 0]))
        heading_p2 = float(np.arctan2(self.tangents[idx_p2, 1], self.tangents[idx_p2, 0]))
        pos_p1 = self.centerline[idx_p1] + self.normals[idx_p1] * (0.22 * w)
        pos_p2 = self.centerline[idx_p2] - self.normals[idx_p2] * (0.22 * w)
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
        a_pts = self._wall_segs[:, 0]
        b_pts = self._wall_segs[:, 1]

        dists_sq = (a_pts[:, 0] - pos[0]) ** 2 + (a_pts[:, 1] - pos[1]) ** 2
        nearby = dists_sq < (max_dist + 50.0) ** 2
        if not np.any(nearby):
            return float(max_dist)
        a_pts = a_pts[nearby]
        b_pts = b_pts[nearby]

        r = b_pts - a_pts
        ap = a_pts - pos

        dxr = d[0] * r[:, 1] - d[1] * r[:, 0]
        apxr = ap[:, 0] * r[:, 1] - ap[:, 1] * r[:, 0]
        apxd = ap[:, 0] * d[1] - ap[:, 1] * d[0]

        valid = np.abs(dxr) > 1e-10
        safe = np.where(valid, dxr, 1.0)
        t = np.where(valid, apxr / safe, np.inf)
        s = np.where(valid, apxd / safe, -1.0)

        mask = valid & (t > 1e-3) & (s >= 0.0) & (s <= 1.0)
        return float(np.min(np.where(mask, t, max_dist)))
