from __future__ import annotations

import math
import numpy as np
import pygame

CORNER_LABELS = (
    (1, (790, 62)),
    (2, (835, 160)),
    (3, (770, 188)),
    (4, (795, 283)),
    (5, (1090, 192)),
    (6, (1328, 283)),
    (7, (935, 402)),
    (8, (922, 628)),
    (9, (807, 693)),
    (10, (790, 797)),
    (11, (1283, 797)),
    (12, (1288, 672)),
    (13, (1403, 793)),
    (14, (97, 884)),
    (15, (183, 808)),
    (16, (490, 835)),
)

TEAM_COLORS = (
    (195, 25, 30),
    (15, 25, 75),
    (0, 160, 140),
    (255, 135, 0),
    (0, 110, 75),
    (30, 65, 180),
    (180, 20, 40),
    (80, 140, 200),
    (220, 220, 225),
    (70, 75, 85),
)

def get_font(size: int) -> pygame.font.Font:
    if not pygame.font.get_init():
        pygame.font.init()
    return pygame.font.Font(None, size)


def draw_static_world(
    surf: pygame.Surface,
    track,
    world_rect: tuple[float, float, float, float],
    scale: float,
    sample_indices: np.ndarray | list[int] | None = None,
) -> None:
    min_wx, min_wy, max_wx, max_wy = world_rect
    sw, sh = surf.get_size()

    surf.fill((18, 30, 20))

    stripe_w = 40.0
    u0 = (min_wx + min_wy) * scale
    k_min = int(math.floor((min_wx + min_wy) / stripe_w)) - 1
    k_max = int(math.ceil((max_wx + max_wy) / stripe_w)) + 1
    for k in range(k_min, k_max, 2):
        d0 = k * stripe_w * scale - u0
        d1 = (k + 1) * stripe_w * scale - u0
        pts = []
        for x_edge in (0, sw):
            y_val = d0 - x_edge
            if 0 <= y_val <= sh:
                pts.append((x_edge, y_val))
            y_val = d1 - x_edge
            if 0 <= y_val <= sh:
                pts.append((x_edge, y_val))
        for y_edge in (0, sh):
            x_val = d0 - y_edge
            if 0 <= x_val <= sw:
                pts.append((x_val, y_edge))
            x_val = d1 - y_edge
            if 0 <= x_val <= sw:
                pts.append((x_val, y_edge))
        if len(pts) >= 3:
            cx = sum(p[0] for p in pts) / len(pts)
            cy = sum(p[1] for p in pts) / len(pts)
            pts = sorted(set(pts), key=lambda p: math.atan2(p[1] - cy, p[0] - cx))
            if len(pts) >= 3:
                pygame.draw.polygon(surf, (22, 35, 24), [(int(round(p[0])), int(round(p[1]))) for p in pts])

    def w2s(pt):
        return (int(round((pt[0] - min_wx) * scale)), int(round((pt[1] - min_wy) * scale)))

    N = len(track.centerline)
    outer = track.outer
    inner = track.inner
    curv = track.curvatures
    norm = track.normals
    tang = track.tangents
    w_track = track.half_width * 2.0
    margin_w = w_track * 1.5

    if sample_indices is None:
        indices = range(N)
    else:
        indices = sample_indices

    w_pit = 0.5 * w_track
    wall_w = max(2, int(round(2.0 * scale)))
    pit_indices = list(range(5850, N)) + list(range(0, 350))
    for idx in range(len(pit_indices) - 1):
        i = pit_indices[idx]
        nxt = pit_indices[idx + 1]
        p0 = inner[i]
        p1 = inner[nxt]
        if (
            min(p0[0], p1[0]) - w_pit > max_wx
            or max(p0[0], p1[0]) + w_pit < min_wx
            or min(p0[1], p1[1]) - w_pit > max_wy
            or max(p0[1], p1[1]) + w_pit < min_wy
        ):
            continue
        pit0 = p0 - norm[i] * w_pit
        pit1 = p1 - norm[nxt] * w_pit
        pygame.draw.polygon(surf, (27, 28, 31), [w2s(p) for p in (p0, p1, pit1, pit0)])
        pygame.draw.line(surf, (140, 145, 155), w2s(p0), w2s(p1), wall_w)

    for b_i, i in enumerate(pit_indices[20:-20:25]):
        center = inner[i] - norm[i] * (w_pit + 4.0)
        if (
            center[0] + 15.0 < min_wx
            or center[0] - 15.0 > max_wx
            or center[1] + 15.0 < min_wy
            or center[1] - 15.0 > max_wy
        ):
            continue
        f_box = tang[i] * 10.0
        s_box = norm[i] * 4.0
        pts = [center + f_box + s_box, center + f_box - s_box, center - f_box - s_box, center - f_box + s_box]
        col = TEAM_COLORS[b_i % len(TEAM_COLORS)]
        pygame.draw.polygon(surf, col, [w2s(p) for p in pts])

    for i in indices:
        nxt = (i + 1) % N
        c_pt = track.centerline[i]
        if (
            c_pt[0] + margin_w < min_wx
            or c_pt[0] - margin_w > max_wx
            or c_pt[1] + margin_w < min_wy
            or c_pt[1] - margin_w > max_wy
        ):
            continue

        r_i = 1.0 / (abs(curv[i]) + 1e-9)
        r_nxt = 1.0 / (abs(curv[nxt]) + 1e-9)
        t_i = min(1.0, max(0.0, (6.0 * w_track / r_i - 1.0) / 0.3)) if r_i < 6.0 * w_track else 0.0
        t_nxt = min(1.0, max(0.0, (6.0 * w_track / r_nxt - 1.0) / 0.3)) if r_nxt < 6.0 * w_track else 0.0
        if t_i > 0 or t_nxt > 0:
            if curv[i] >= 0:
                p0, p1 = inner[i], inner[nxt]
                ro0 = p0 - norm[i] * (0.35 * w_track * t_i)
                ro1 = p1 - norm[nxt] * (0.35 * w_track * t_nxt)
                gr0 = ro0 - norm[i] * (0.25 * w_track * t_i)
                gr1 = ro1 - norm[nxt] * (0.25 * w_track * t_nxt)
            else:
                p0, p1 = outer[i], outer[nxt]
                ro0 = p0 + norm[i] * (0.35 * w_track * t_i)
                ro1 = p1 + norm[nxt] * (0.35 * w_track * t_nxt)
                gr0 = ro0 + norm[i] * (0.25 * w_track * t_i)
                gr1 = ro1 + norm[nxt] * (0.25 * w_track * t_nxt)
            pygame.draw.polygon(surf, (194, 168, 126), [w2s(p) for p in (ro0, ro1, gr1, gr0)])
            pygame.draw.polygon(surf, (52, 54, 58), [w2s(p) for p in (p0, p1, ro1, ro0)])

        noise = ((i * 73 + 19) % 7) - 3
        col = (32 + noise, 33 + noise, 36 + noise)
        poly = [w2s(outer[i]), w2s(outer[nxt]), w2s(inner[nxt]), w2s(inner[i])]
        pygame.draw.polygon(surf, col, poly)

        t_i_k = min(1.0, max(0.0, (4.0 * w_track / r_i - 1.0) / 0.3)) if r_i < 4.0 * w_track else 0.0
        t_nxt_k = min(1.0, max(0.0, (4.0 * w_track / r_nxt - 1.0) / 0.3)) if r_nxt < 4.0 * w_track else 0.0
        if t_i_k > 0 or t_nxt_k > 0:
            if curv[i] >= 0:
                p0, p1 = outer[i], outer[nxt]
                k0 = p0 + norm[i] * (0.07 * w_track * t_i_k)
                k1 = p1 + norm[nxt] * (0.07 * w_track * t_nxt_k)
            else:
                p0, p1 = inner[i], inner[nxt]
                k0 = p0 - norm[i] * (0.07 * w_track * t_i_k)
                k1 = p1 - norm[nxt] * (0.07 * w_track * t_nxt_k)
            is_red = (i // 4) % 2 == 0
            k_col = (215, 30, 30) if is_red else (245, 245, 248)
            pygame.draw.polygon(surf, k_col, [w2s(p) for p in (p0, p1, k1, k0)])

    edge_w = max(2, int(round(3.0 * scale * 0.5)))
    for i in indices:
        nxt = (i + 1) % N
        c_pt = track.centerline[i]
        if (
            c_pt[0] + margin_w < min_wx
            or c_pt[0] - margin_w > max_wx
            or c_pt[1] + margin_w < min_wy
            or c_pt[1] - margin_w > max_wy
        ):
            continue
        pygame.draw.line(surf, (240, 242, 248), w2s(outer[i]), w2s(outer[nxt]), edge_w)
        pygame.draw.line(surf, (240, 242, 248), w2s(inner[i]), w2s(inner[nxt]), edge_w)

    chev_w = max(2, int(round(2.0 * scale)))
    for idx_range in (range(3980, 5220, 50), range(50, 320, 50)):
        for idx in idx_range:
            c = track.centerline[idx]
            if c[0] + 15.0 < min_wx or c[0] - 15.0 > max_wx or c[1] + 15.0 < min_wy or c[1] - 15.0 > max_wy:
                continue
            t_vec = tang[idx]
            n_vec = norm[idx]
            tip = c + t_vec * 8.0
            pl = c - t_vec * 4.0 - n_vec * 6.0
            pr = c - t_vec * 4.0 + n_vec * 6.0
            pygame.draw.lines(surf, (44, 46, 52), False, [w2s(pl), w2s(tip), w2s(pr)], chev_w)

    p_out = outer[0]
    p_in = inner[0]
    tang0 = tang[0]
    if not (
        max(p_out[0], p_in[0]) + 10.0 < min_wx
        or min(p_out[0], p_in[0]) - 10.0 > max_wx
        or max(p_out[1], p_in[1]) + 10.0 < min_wy
        or min(p_out[1], p_in[1]) - 10.0 > max_wy
    ):
        for col in range(14):
            for row in range(2):
                is_white = (col + row) % 2 == 0
                c = (245, 248, 255) if is_white else (25, 27, 30)
                t0 = col / 14.0
                t1 = (col + 1) / 14.0
                p0a = p_in + (p_out - p_in) * t0 + tang0 * (row * 3.0)
                p1a = p_in + (p_out - p_in) * t1 + tang0 * (row * 3.0)
                p1b = p_in + (p_out - p_in) * t1 + tang0 * ((row + 1) * 3.0)
                p0b = p_in + (p_out - p_in) * t0 + tang0 * ((row + 1) * 3.0)
                pygame.draw.polygon(surf, c, [w2s(p) for p in (p0a, p1a, p1b, p0b)])

    car_len = 0.39 * w_track
    car_wid = 0.14 * w_track
    box_hl = (car_len * 1.15) / 2.0
    box_hw = (car_wid * 1.5) / 2.0
    grid_w = max(1, int(round(1.0 * scale)))
    grid_front_w = max(2, int(round(2.5 * scale)))
    for k in range(20):
        s_slot = (track.total_length - 1.0 * w_track - k * 0.57 * w_track) % track.total_length
        idx_slot = int(np.clip(s_slot / 2.0, 0, N - 1))
        t_slot = tang[idx_slot]
        n_slot = norm[idx_slot]
        lat_sign = 1.0 if (k % 2 == 0) else -1.0
        lat_off = lat_sign * (0.22 * w_track)
        pos_slot = track.centerline[idx_slot] + n_slot * lat_off
        if (
            pos_slot[0] + 20.0 < min_wx
            or pos_slot[0] - 20.0 > max_wx
            or pos_slot[1] + 20.0 < min_wy
            or pos_slot[1] - 20.0 > max_wy
        ):
            continue
        fwd = t_slot * box_hl
        side = n_slot * box_hw
        box_pts = [pos_slot + fwd + side, pos_slot + fwd - side, pos_slot - fwd - side, pos_slot - fwd + side]
        poly = [w2s(p) for p in box_pts]
        pygame.draw.polygon(surf, (230, 235, 245), poly, grid_w)
        pygame.draw.line(surf, (245, 248, 255), poly[0], poly[1], grid_front_w)

    ref_scale = track.centerline[0, 0] / 599.05
    font_cnum = get_font(min(48, max(12, int(round(16 * scale / 0.5)))))
    for num, (rx, ry) in CORNER_LABELS:
        w_pt = np.array([rx * ref_scale, ry * ref_scale])
        if w_pt[0] < min_wx - 30.0 or w_pt[0] > max_wx + 30.0 or w_pt[1] < min_wy - 30.0 or w_pt[1] > max_wy + 30.0:
            continue
        s_pt = w2s(w_pt)
        txt = font_cnum.render(str(num), True, (215, 225, 240))
        rect = txt.get_rect(center=s_pt)
        surf.blit(txt, rect)
