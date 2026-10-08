from __future__ import annotations

import math
import numpy as np
import pygame


class ViewerEffects:
    def __init__(self, window_size: tuple[int, int]):
        self.w, self.h = window_size
        self.fs = self.h / 720.0
        self.view_w = self.w
        self.view_h = self.h - int(round(self.h * 0.18))

        self.overtake_frames = 0
        self.overtake_leader = "agent_0"
        self.prev_leader = "agent_0"

        self.contact_frames = 0
        self.contact_screen_pt = (0, 0)
        self.prev_contact = False

        self._digits_cache: dict[int, pygame.Surface] = {}
        self._init_resources()

    def _init_resources(self) -> None:
        if not pygame.font.get_init():
            pygame.font.init()

        font_family = "ubuntu" if "ubuntu" in pygame.font.get_fonts() else "dejavusans"
        mono_family = "ubuntumono" if "ubuntumono" in pygame.font.get_fonts() else "monospace"

        f_tag = pygame.font.SysFont(font_family, max(13, int(round(13 * self.fs))), bold=True)
        f_banner = pygame.font.SysFont(font_family, max(14, int(round(15 * self.fs))), bold=True)
        self.f_dist = pygame.font.SysFont(mono_family, max(13, int(round(13 * self.fs))), bold=True)

        self.tag_surfs = {}
        for ag, tag_str, bg_col, txt_col in (
            ("agent_1", "HAM", (200, 20, 35), (255, 255, 255)),
            ("agent_0", "VER", (16, 28, 64), (255, 215, 0)),
        ):
            t_surf = f_tag.render(tag_str, True, txt_col)
            tw = t_surf.get_width() + int(round(10 * self.fs))
            th = t_surf.get_height() + int(round(4 * self.fs))
            surf = pygame.Surface((tw, th))
            surf.fill(bg_col)
            pygame.draw.rect(surf, (240, 245, 255), (0, 0, tw, th), 1, border_radius=3)
            surf.blit(t_surf, (int(round(5 * self.fs)), int(round(2 * self.fs))))
            try:
                self.tag_surfs[ag] = surf.convert()
            except Exception:
                self.tag_surfs[ag] = surf

        self.overtake_banners = {}
        for ag, msg, col in (
            ("agent_1", "OVERTAKE: #44 HAM LEADS", (220, 20, 35)),
            ("agent_0", "OVERTAKE: #1 VER LEADS", (0, 200, 120)),
        ):
            t_surf = f_banner.render(msg, True, (255, 255, 255))
            bw = t_surf.get_width() + int(round(32 * self.fs))
            bh = t_surf.get_height() + int(round(12 * self.fs))
            surf = pygame.Surface((bw, bh))
            surf.fill((16, 20, 28))
            pygame.draw.rect(surf, col, (0, 0, bw, bh), 2, border_radius=6)
            surf.blit(t_surf, (int(round(16 * self.fs)), int(round(6 * self.fs))))
            try:
                self.overtake_banners[ag] = surf.convert()
            except Exception:
                self.overtake_banners[ag] = surf

    def on_step(self, current_leader: str, is_contact: bool, contact_screen_pt: tuple[int, int]) -> None:
        if current_leader != self.prev_leader:
            self.overtake_frames = 45
            self.overtake_leader = current_leader
            self.prev_leader = current_leader

        if is_contact and not self.prev_contact:
            self.contact_frames = 8
            self.contact_screen_pt = contact_screen_pt
        elif is_contact:
            self.contact_screen_pt = contact_screen_pt
        self.prev_contact = is_contact

    def draw_driver_tags(
        self,
        screen: pygame.Surface,
        pos_s_ham: np.ndarray,
        pos_s_ver: np.ndarray,
    ) -> None:
        ham_surf = self.tag_surfs["agent_1"]
        ver_surf = self.tag_surfs["agent_0"]

        hw, hh = ham_surf.get_width(), ham_surf.get_height()
        vw, vh = ver_surf.get_width(), ver_surf.get_height()

        hx = int(round(pos_s_ham[0] - hw * 0.5))
        hy = int(round(pos_s_ham[1] - hh - 22 * self.fs))

        vx = int(round(pos_s_ver[0] - vw * 0.5))
        vy = int(round(pos_s_ver[1] - vh - 22 * self.fs))

        dx = abs(hx - vx)
        dy = abs(hy - vy)
        min_dx = max(hw, vw) + 4
        min_dy = max(hh, vh) + 2

        if dx < min_dx and dy < min_dy:
            nudge_y = (min_dy - dy) // 2 + 2
            if hy <= vy:
                hy -= nudge_y
                vy += nudge_y
            else:
                hy += nudge_y
                vy -= nudge_y

        if -hw <= hx <= self.view_w and -hh <= hy <= self.view_h:
            screen.blit(ham_surf, (hx, hy))
        if -vw <= vx <= self.view_w and -vh <= vy <= self.view_h:
            screen.blit(ver_surf, (vx, vy))

    def _get_distance_surf(self, dist_val: int) -> pygame.Surface:
        surf = self._digits_cache.get(dist_val)
        if surf is None:
            if len(self._digits_cache) > 500:
                self._digits_cache.clear()
            surf = self.f_dist.render(f"{dist_val}u", True, (255, 230, 110))
            self._digits_cache[dist_val] = surf
        return surf

    def draw_off_screen_arrows(
        self,
        screen: pygame.Surface,
        pos_s: np.ndarray,
        dist_units: float,
        color: tuple[int, int, int],
    ) -> None:
        px, py = float(pos_s[0]), float(pos_s[1])
        pad = 28 * self.fs
        vw, vh = float(self.view_w), float(self.view_h)

        if pad <= px <= vw - pad and pad <= py <= vh - pad:
            return

        cx, cy = vw * 0.5, vh * 0.5
        dx = px - cx
        dy = py - cy
        angle = math.atan2(dy, dx)

        half_w = cx - pad
        half_h = cy - pad

        if abs(dx) * half_h > abs(dy) * half_w:
            edge_x = cx + math.copysign(half_w, dx)
            edge_y = cy + math.copysign(half_w, dx) * math.tan(angle)
        else:
            edge_y = cy + math.copysign(half_h, dy)
            edge_x = cx + math.copysign(half_h, dy) / (math.tan(angle) + 1e-6)

        edge_x = float(np.clip(edge_x, pad, vw - pad))
        edge_y = float(np.clip(edge_y, pad, vh - pad))

        arrow_len = 14.0 * self.fs
        arrow_w = 8.0 * self.fs
        tip = np.array([edge_x, edge_y])
        back = tip - np.array([math.cos(angle), math.sin(angle)]) * arrow_len
        left_perp = np.array([-math.sin(angle), math.cos(angle)]) * arrow_w

        p1 = tip
        p2 = back + left_perp
        p3 = back - left_perp

        pygame.draw.polygon(screen, color, [(int(round(p[0])), int(round(p[1]))) for p in (p1, p2, p3)])
        pygame.draw.polygon(screen, (255, 255, 255), [(int(round(p[0])), int(round(p[1]))) for p in (p1, p2, p3)], 1)

        d_surf = self._get_distance_surf(int(round(dist_units)))
        dw = d_surf.get_width()
        dh = d_surf.get_height()
        tx = int(round(edge_x - dw * 0.5 - math.cos(angle) * (arrow_len + 12 * self.fs)))
        ty = int(round(edge_y - dh * 0.5 - math.sin(angle) * (arrow_len + 12 * self.fs)))
        tx = int(np.clip(tx, 4, vw - dw - 4))
        ty = int(np.clip(ty, 4, vh - dh - 4))
        screen.blit(d_surf, (tx, ty))

    def draw_banners_and_effects(self, screen: pygame.Surface) -> None:
        if self.overtake_frames > 0:
            surf = self.overtake_banners[self.overtake_leader]
            bx = (self.view_w - surf.get_width()) // 2
            by = int(round(16 * self.fs))
            screen.blit(surf, (bx, by))
            self.overtake_frames -= 1

        if self.contact_frames > 0:
            cx, cy = self.contact_screen_pt
            r_in = 6 * self.fs
            r_out = 18 * self.fs
            for idx in range(6):
                ang = idx * (math.pi / 3.0) + (8 - self.contact_frames) * 0.1
                x0 = cx + math.cos(ang) * r_in
                y0 = cy + math.sin(ang) * r_in
                x1 = cx + math.cos(ang) * r_out
                y1 = cy + math.sin(ang) * r_out
                pygame.draw.line(screen, (255, 235, 40), (int(round(x0)), int(round(y0))), (int(round(x1)), int(round(y1))), 2)
            self.contact_frames -= 1
