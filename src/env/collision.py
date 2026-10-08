from __future__ import annotations

import math
import numpy as np


def _obb_axes(heading: float) -> np.ndarray:
    c, s = math.cos(heading), math.sin(heading)
    return np.array([[c, s], [-s, c]], dtype=np.float64)


def _project(vertices: np.ndarray, axis: np.ndarray) -> tuple[float, float]:
    dots = vertices @ axis
    return float(dots.min()), float(dots.max())


def obb_vertices(pos: np.ndarray, heading: float, half_len: float, half_width: float) -> np.ndarray:
    axes = _obb_axes(heading)
    fwd = axes[0]
    left = axes[1]
    return np.array([
        pos + fwd * half_len + left * half_width,
        pos + fwd * half_len - left * half_width,
        pos - fwd * half_len - left * half_width,
        pos - fwd * half_len + left * half_width,
    ], dtype=np.float64)


def obb_overlap(
    pos_a: np.ndarray,
    heading_a: float,
    pos_b: np.ndarray,
    heading_b: float,
    half_len: float,
    half_width: float,
) -> tuple[bool, np.ndarray, float]:
    """SAT overlap test for two identical OBBs.

    Returns (overlapping, contact_normal_pointing_from_b_to_a, penetration_depth).
    contact_normal and penetration_depth are meaningful only when overlapping is True.
    """
    verts_a = obb_vertices(pos_a, heading_a, half_len, half_width)
    verts_b = obb_vertices(pos_b, heading_b, half_len, half_width)

    axes_a = _obb_axes(heading_a)
    axes_b = _obb_axes(heading_b)
    test_axes = np.vstack([axes_a, axes_b])

    min_pen = math.inf
    best_axis = test_axes[0]

    for axis in test_axes:
        min_a, max_a = _project(verts_a, axis)
        min_b, max_b = _project(verts_b, axis)
        overlap = min(max_a, max_b) - max(min_a, min_b)
        if overlap <= 0.0:
            return False, np.zeros(2), 0.0
        if overlap < min_pen:
            min_pen = overlap
            best_axis = axis.copy()

    # Ensure normal points from b toward a
    d = pos_a - pos_b
    if np.dot(d, best_axis) < 0.0:
        best_axis = -best_axis

    return True, best_axis, min_pen


def resolve_collision(
    pos_a: np.ndarray,
    vel_a: np.ndarray,
    pos_b: np.ndarray,
    vel_b: np.ndarray,
    normal: np.ndarray,
    restitution: float = 0.3,
) -> tuple[np.ndarray, np.ndarray]:
    """Impulse-based inelastic collision resolution for equal-mass bodies."""
    rel_vel = vel_a - vel_b
    vel_along_normal = float(np.dot(rel_vel, normal))
    # Positive means separating along the normal; no impulse needed.
    if vel_along_normal >= 0.0:
        return vel_a, vel_b
    impulse = -(1.0 + restitution) * vel_along_normal / 2.0
    delta = impulse * normal
    return vel_a + delta, vel_b - delta


def resolve_penetration(
    pos_a: np.ndarray,
    pos_b: np.ndarray,
    normal: np.ndarray,
    penetration: float,
    epsilon: float = 0.02,
) -> tuple[np.ndarray, np.ndarray]:
    excess = max(0.0, float(penetration) - epsilon)
    if excess <= 0.0:
        return pos_a.copy(), pos_b.copy()
    norm = float(np.linalg.norm(normal))
    n = normal / norm if norm > 1e-9 else np.array([1.0, 0.0], dtype=np.float64)
    shift = 0.5 * excess * n
    return pos_a + shift, pos_b - shift



def classify_contact(
    normal: np.ndarray,
    heading_leader: float,
    vel_a: np.ndarray,
    vel_b: np.ndarray,
    leader_id: int,
    lateral_history_leader: list[float],
    follower_lateral: float,
    lookback: int = 10,
    rear_cone_deg: float = 45.0,
    lateral_move_threshold: float = 0.1,
) -> str:
    """Classify a collision event as 'follower_fault', 'leader_fault', or 'neutral'.

    Args:
        normal: Contact normal pointing from follower toward leader.
        heading_leader: Leader's current heading in radians.
        vel_a, vel_b: Velocity vectors (world frame) of agent_0, agent_1.
        leader_id: 0 or 1 indicating which agent is the leader.
        lateral_history_leader: Recent lateral track positions of the leader.
        follower_lateral: Current lateral track position of the follower.
        lookback: Number of steps in the leader's lateral history to inspect.
        rear_cone_deg: Half-angle (degrees) of the rear cone defining a rear-end.
        lateral_move_threshold: Fraction of track width the leader must move toward
            the follower within the lookback window to be considered at fault.
    """
    rear_cone_rad = math.radians(rear_cone_deg)
    leader_fwd = np.array([math.cos(heading_leader), math.sin(heading_leader)])
    rear_dir = -leader_fwd

    cos_angle = float(np.clip(np.dot(normal, rear_dir) / (np.linalg.norm(normal) + 1e-9), -1.0, 1.0))
    angle = math.acos(cos_angle)

    if angle <= rear_cone_rad:
        return "follower_fault"

    # Leader fault: one-move lookback - did leader move toward follower?
    if len(lateral_history_leader) >= lookback:
        # Compare the oldest step in the lookback window to the most recent
        window_start = lateral_history_leader[-(lookback)]
        window_end = lateral_history_leader[-1]
        lateral_delta = window_end - window_start
        moved_toward = lateral_delta * (follower_lateral - window_end) > 0.0
        if moved_toward and abs(lateral_delta) > lateral_move_threshold:
            return "leader_fault"

    return "neutral"
