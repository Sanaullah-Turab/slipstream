from __future__ import annotations

import math
import numpy as np
import pygame

SECTOR_COLORS = {
    1: (220, 40, 40),
    2: (40, 200, 220),
    3: (240, 210, 40),
}


class BroadcastHUD:
    def __init__(self, track, window_size: tuple[int, int]):
        self.track = track
        self.w, self.h = window_size
        self.font_scale = self.h / 720.0

        self.bottom_h = int(round(self.h * 0.18))
        self.bottom_y = self.h - self.bottom_h
        self.view_w = self.w
        self.view_h = self.bottom_y

        self.show_help = False
        self._text_cache: dict[tuple[str, str, tuple[int, int, int]], pygame.Surface] = {}

        self._init_fonts()
        self._build_static_bottom_bar()
        self._build_minimap()
        self._build_help_overlay()

    def _init_fonts(self) -> None:
        if not pygame.font.get_init():
            pygame.font.init()

        font_family = "ubuntu" if "ubuntu" in pygame.font.get_fonts() else "dejavusans"
        mono_family = "ubuntumono" if "ubuntumono" in pygame.font.get_fonts() else "monospace"

        fs = self.font_scale
        self.fonts = {
            "small": pygame.font.SysFont(font_family, max(13, int(round(13 * fs))), bold=True),
            "regular": pygame.font.SysFont(font_family, max(13, int(round(14 * fs))), bold=True),
            "mono": pygame.font.SysFont(mono_family, max(13, int(round(14 * fs))), bold=True),
            "speed": pygame.font.SysFont(mono_family, max(24, int(round(28 * fs))), bold=True),
            "speed_unit": pygame.font.SysFont(font_family, max(13, int(round(14 * fs))), bold=True),
            "title": pygame.font.SysFont(font_family, max(15, int(round(16 * fs))), bold=True),
            "badge": pygame.font.SysFont(font_family, max(13, int(round(13 * fs))), bold=True),
            "f1": pygame.font.SysFont(font_family, max(14, int(round(15 * fs))), bold=True),
        }

    def render_text(self, text: str, font_name: str, color: tuple[int, int, int]) -> pygame.Surface:
        key = (text, font_name, color)
        surf = self._text_cache.get(key)
        if surf is None:
            if len(self._text_cache) > 2000:
                self._text_cache.clear()
            surf = self.fonts[font_name].render(text, True, color)
            self._text_cache[key] = surf
        return surf

    def _build_static_bottom_bar(self) -> None:
        self.static_bottom_bar = pygame.Surface((self.w, self.bottom_h))
        self.static_bottom_bar.fill((14, 17, 24))
        pygame.draw.line(self.static_bottom_bar, (36, 44, 60), (0, 0), (self.w, 0), 2)

        col_w = self.w // 3
        pygame.draw.line(self.static_bottom_bar, (36, 44, 60), (col_w, 0), (col_w, self.bottom_h), 2)
        pygame.draw.line(self.static_bottom_bar, (36, 44, 60), (2 * col_w, 0), (2 * col_w, self.bottom_h), 2)

        card_margin = int(round(8 * self.font_scale))
        for col_idx in range(3):
            cx = col_idx * col_w + card_margin
            cw = col_w - 2 * card_margin if col_idx < 2 else (self.w - 2 * col_w) - 2 * card_margin
            ch = self.bottom_h - 2 * card_margin
            cy = card_margin
            pygame.draw.rect(self.static_bottom_bar, (20, 25, 36), (cx, cy, cw, ch), border_radius=6)
            pygame.draw.rect(self.static_bottom_bar, (36, 44, 60), (cx, cy, cw, ch), 1, border_radius=6)

        try:
            self.static_bottom_bar = self.static_bottom_bar.convert()
        except Exception:
            pass

    def _build_minimap(self) -> None:
        mw = int(round(260 * self.font_scale))
        mh = int(round(150 * self.font_scale))
        self.minimap_w = mw
        self.minimap_h = mh
        self.minimap_x = self.w - mw - 16
        self.minimap_y = 16

        self.minimap_base = pygame.Surface((mw, mh))
        self.minimap_base.fill((16, 20, 28))
        pygame.draw.rect(self.minimap_base, (45, 55, 75), (0, 0, mw, mh), 2, border_radius=6)

        cl = self.track.centerline
        min_x, max_x = cl[:, 0].min(), cl[:, 0].max()
        min_y, max_y = cl[:, 1].min(), cl[:, 1].max()

        pad = 12 * self.font_scale
        avail_w = mw - 2 * pad
        avail_h = mh - 2 * pad
        self.minimap_scale = min(avail_w / (max_x - min_x), avail_h / (max_y - min_y))
        self.minimap_off_x = pad + (avail_w - (max_x - min_x) * self.minimap_scale) * 0.5 - min_x * self.minimap_scale
        self.minimap_off_y = pad + (avail_h - (max_y - min_y) * self.minimap_scale) * 0.5 - min_y * self.minimap_scale

        pts = [
            (
                int(round(p[0] * self.minimap_scale + self.minimap_off_x)),
                int(round(p[1] * self.minimap_scale + self.minimap_off_y)),
            )
            for p in cl[::4]
        ]
        if len(pts) > 2:
            pygame.draw.lines(self.minimap_base, (140, 155, 175), True, pts, max(2, int(round(2 * self.font_scale))))

        try:
            self.minimap_base = self.minimap_base.convert()
        except Exception:
            pass

    def _build_help_overlay(self) -> None:
        hw = int(round(460 * self.font_scale))
        hh = int(round(340 * self.font_scale))
        self.help_w = hw
        self.help_h = hh
        self.help_x = (self.w - hw) // 2
        self.help_y = (self.view_h - hh) // 2

        surf = pygame.Surface((hw, hh))
        surf.fill((12, 16, 22))
        pygame.draw.rect(surf, (0, 230, 140), (0, 0, hw, hh), 2, border_radius=8)

        title = self.fonts["title"].render("CONTROLS & SHORTCUTS", True, (0, 230, 140))
        surf.blit(title, ((hw - title.get_width()) // 2, int(round(16 * self.font_scale))))

        shortcuts = [
            ("1", "Lock camera on #44 HAM (Ferrari)"),
            ("2", "Lock camera on #1 VER (Red Bull)"),
            ("A", "Auto-framing follow mode (default)"),
            ("O", "Toggle full circuit overview"),
            ("+ / -", "Zoom in / Zoom out (2.4, 3.2, 4.0)"),
            ("H", "Toggle help overlay"),
            ("R", "Restart race grid"),
            ("ESC / Q", "Exit viewer"),
        ]

        y_offset = int(round(56 * self.font_scale))
        line_spacing = int(round(32 * self.font_scale))
        for key_str, desc_str in shortcuts:
            k_surf = self.fonts["badge"].render(key_str, True, (255, 220, 100))
            d_surf = self.fonts["small"].render(desc_str, True, (220, 230, 245))
            pygame.draw.rect(
                surf,
                (24, 30, 42),
                (int(round(24 * self.font_scale)), y_offset - 2, int(round(72 * self.font_scale)), int(round(24 * self.font_scale))),
                border_radius=4,
            )
            surf.blit(k_surf, (int(round(32 * self.font_scale)), y_offset + 2))
            surf.blit(d_surf, (int(round(108 * self.font_scale)), y_offset + 2))
            y_offset += line_spacing

        try:
            self.help_overlay = surf.convert()
        except Exception:
            self.help_overlay = surf

    def draw_minimap(self, screen: pygame.Surface, pos_ham: np.ndarray, pos_ver: np.ndarray) -> None:
        screen.blit(self.minimap_base, (self.minimap_x, self.minimap_y))

        for pos, col in ((pos_ham, (220, 20, 35)), (pos_ver, (30, 140, 255))):
            mx = self.minimap_x + int(round(pos[0] * self.minimap_scale + self.minimap_off_x))
            my = self.minimap_y + int(round(pos[1] * self.minimap_scale + self.minimap_off_y))
            pygame.draw.circle(screen, (255, 255, 255), (mx, my), max(4, int(round(4 * self.font_scale))))
            pygame.draw.circle(screen, col, (mx, my), max(3, int(round(3 * self.font_scale))))

    def draw_top_chips(
        self,
        screen: pygame.Surface,
        lap: int,
        progress: float,
        is_contact: bool,
    ) -> None:
        sec = 1 if progress < 0.33 else (2 if progress < 0.67 else 3)
        chip_x = int(round(16 * self.font_scale))
        chip_y = int(round(16 * self.font_scale))
        fs = self.font_scale

        lap_txt = f"LAP {lap + 1}"
        sec_txt = f"S{sec}"
        lap_surf = self.render_text(lap_txt, "badge", (255, 255, 255))
        sec_surf = self.render_text(sec_txt, "badge", (10, 10, 15) if sec != 1 else (255, 255, 255))

        w_lap = lap_surf.get_width() + int(round(16 * fs))
        w_sec = sec_surf.get_width() + int(round(16 * fs))
        h_chip = int(round(28 * fs))

        pygame.draw.rect(screen, (16, 22, 32), (chip_x, chip_y, w_lap, h_chip), border_radius=4)
        screen.blit(lap_surf, (chip_x + int(round(8 * fs)), chip_y + int(round(5 * fs))))

        pygame.draw.rect(screen, SECTOR_COLORS[sec], (chip_x + w_lap + 4, chip_y, w_sec, h_chip), border_radius=4)
        screen.blit(sec_surf, (chip_x + w_lap + 4 + int(round(8 * fs)), chip_y + int(round(5 * fs))))

        flag_x = chip_x + w_lap + w_sec + 12
        if is_contact:
            f_col = (200, 30, 40)
            f_txt = "! CONTACT !"
        else:
            f_col = (16, 160, 80)
            f_txt = "GREEN FLAG"
        flag_surf = self.render_text(f_txt, "badge", (255, 255, 255))
        w_flag = flag_surf.get_width() + int(round(16 * fs))
        pygame.draw.rect(screen, f_col, (flag_x, chip_y, w_flag, h_chip), border_radius=4)
        screen.blit(flag_surf, (flag_x + int(round(8 * fs)), chip_y + int(round(5 * fs))))

    def draw_bottom_bar(
        self,
        screen: pygame.Surface,
        state_ham: dict,
        state_ver: dict,
        current_leader: str,
        step_count: int,
        dt: float,
        position_swaps: int,
        is_contact: bool,
    ) -> None:
        screen.blit(self.static_bottom_bar, (0, self.bottom_y))

        col_w = self.w // 3
        card_m = int(round(8 * self.font_scale))
        fs = self.font_scale

        def _fmt(sec: float | None) -> str:
            if sec is None:
                return "--:--.---"
            m = int(sec // 60)
            s_rem = sec % 60
            return f"{m:02d}:{s_rem:06.3f}"

        drivers = [
            ("agent_1", 0, state_ham, "#44 HAM", "SCUDERIA FERRARI", (220, 20, 35), (255, 220, 0)),
            ("agent_0", 2, state_ver, "#1 VER", "RED BULL RACING", (11, 24, 60), (255, 215, 0)),
        ]

        leader_is_ham = current_leader == "agent_1"

        for ag, col_idx, st, car_num, team_name, primary_col, detail_col in drivers:
            cx = col_idx * col_w + card_m
            cw = col_w - 2 * card_m if col_idx < 2 else (self.w - 2 * col_w) - 2 * card_m
            cy = self.bottom_y + card_m
            ch = self.bottom_h - 2 * card_m

            is_p1 = (ag == current_leader)
            rank_str = "P1" if is_p1 else "P2"
            rank_col = (0, 230, 140) if is_p1 else (220, 225, 235)

            if is_p1:
                pygame.draw.rect(screen, (0, 230, 140) if ag == "agent_0" else (220, 20, 35), (cx, cy, cw, ch), 2, border_radius=6)

            screen.blit(self.render_text(rank_str, "badge", rank_col), (cx + int(round(12 * fs)), cy + int(round(8 * fs))))
            screen.blit(self.render_text(f"{car_num}  {team_name}", "regular", (255, 255, 255)), (cx + int(round(42 * fs)), cy + int(round(7 * fs))))

            spd = st["speed"]
            spd_str = f"{spd:5.1f}"
            s_surf = self.render_text(spd_str, "speed", (245, 248, 255))
            screen.blit(s_surf, (cx + int(round(12 * fs)), cy + int(round(28 * fs))))
            screen.blit(self.render_text("u/s", "speed_unit", (160, 180, 205)), (cx + int(round(12 * fs)) + s_surf.get_width() + 4, cy + int(round(38 * fs))))

            bar_x = cx + int(round(12 * fs))
            bar_y = cy + int(round(68 * fs))
            bar_w = int(round(cw - 24 * fs))
            bar_h = max(4, int(round(6 * fs)))
            pygame.draw.rect(screen, (32, 40, 56), (bar_x, bar_y, bar_w, bar_h), border_radius=2)
            fill_w = int(min(max(spd, 0.0) / 160.0, 1.0) * bar_w)
            if fill_w > 0:
                pygame.draw.rect(screen, detail_col, (bar_x, bar_y, fill_w, bar_h), border_radius=2)

            lap_str = f"L{st['laps']}  {st['progress'] * 100:4.1f}%"
            screen.blit(self.render_text(lap_str, "mono", (210, 225, 245)), (cx + int(round(12 * fs)), cy + int(round(80 * fs))))

            sec = 1 if st["progress"] < 0.33 else (2 if st["progress"] < 0.67 else 3)
            sec_x = cx + int(round(cw - 86 * fs))
            sec_y = cy + int(round(80 * fs))
            for s_idx in (1, 2, 3):
                bx = sec_x + (s_idx - 1) * int(round(24 * fs))
                s_col = SECTOR_COLORS[s_idx] if sec == s_idx else (32, 38, 50)
                t_col = (10, 10, 15) if (sec == s_idx and s_idx != 1) else (255, 255, 255) if sec == s_idx else (120, 135, 155)
                pygame.draw.rect(screen, s_col, (bx, sec_y, int(round(20 * fs)), int(round(14 * fs))), border_radius=2)
                screen.blit(self.render_text(f"S{s_idx}", "small", t_col), (bx + 2, sec_y))

            last_t = _fmt(st.get("last_lap_time"))
            best_t = _fmt(st.get("best_lap_time"))
            time_str = f"L: {last_t}  B: {best_t}"
            screen.blit(self.render_text(time_str, "small", (170, 195, 225)), (cx + int(round(12 * fs)), cy + int(round(98 * fs))))

        mc_x = col_w + card_m
        mc_w = col_w - 2 * card_m
        mc_y = self.bottom_y + card_m
        screen.blit(self.render_text("RACE CONTROL & TELEMETRY", "title", (180, 210, 245)), (mc_x + int(round(12 * fs)), mc_y + int(round(6 * fs))))

        time_line = f"STEP: {step_count:04d}   TIME: {step_count * dt:5.1f}s"
        screen.blit(self.render_text(time_line, "mono", (190, 215, 250)), (mc_x + int(round(12 * fs)), mc_y + int(round(28 * fs))))

        follower_s = state_ver if leader_is_ham else state_ham
        gap_sec = follower_s.get("gap_to_leader_seconds", 0.0)
        dist_between = float(np.linalg.norm(state_ham["pos"] - state_ver["pos"]))
        gap_line = f"Gap: +{gap_sec:5.3f}s  ({dist_between:5.1f}u)"
        screen.blit(self.render_text(gap_line, "mono", (255, 220, 120)), (mc_x + int(round(12 * fs)), mc_y + int(round(48 * fs))))

        drs_x = mc_x + int(round(12 * fs))
        drs_y = mc_y + int(round(70 * fs))
        if 0.0 < gap_sec < 1.0:
            drs_bg = (15, 60, 30)
            drs_txt = "DRS ENABLED (<1s)"
            drs_col = (80, 245, 140)
        else:
            drs_bg = (35, 42, 52)
            drs_txt = "DRS DISABLED"
            drs_col = (160, 175, 195)
        drs_surf = self.render_text(drs_txt, "badge", drs_col)
        pygame.draw.rect(screen, drs_bg, (drs_x, drs_y, drs_surf.get_width() + 12, int(round(20 * fs))), border_radius=3)
        screen.blit(drs_surf, (drs_x + 6, drs_y + 2))

        contact_steps = state_ham.get("steps_in_contact", 0)
        stat_line = f"Contact: {contact_steps} stp   Swaps: {position_swaps}"
        screen.blit(self.render_text(stat_line, "small", (170, 195, 225)), (mc_x + int(round(12 * fs)), mc_y + int(round(98 * fs))))

        if self.show_help:
            screen.blit(self.help_overlay, (self.help_x, self.help_y))
