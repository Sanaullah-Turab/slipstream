from __future__ import annotations

import collections
import math
import numpy as np
import pygame

from .world import draw_static_world

TILE_SIZE = 512
ZOOM_LEVELS = (2.4, 3.2, 4.0)
DEFAULT_ZOOM = 3.2
MAX_CACHE_TILES = 40
MAX_RUBBER_PER_TILE = 500


class TileCache:
    def __init__(self, track):
        self.track = track
        self.zoom = DEFAULT_ZOOM
        self.cache: collections.OrderedDict[tuple[float, int, int], pygame.Surface] = collections.OrderedDict()
        self.tile_indices: dict[float, dict[tuple[int, int], np.ndarray]] = {}
        self.rubber_segments: dict[tuple[float, int, int], list[tuple[np.ndarray, np.ndarray]]] = {}
        self._build_spatial_indices()

    def _build_spatial_indices(self) -> None:
        N = len(self.track.centerline)
        margin = self.track.half_width * 2.0 * 1.5
        cl = self.track.centerline

        for z in ZOOM_LEVELS:
            tile_w = TILE_SIZE / z
            tile_h = TILE_SIZE / z
            self.tile_indices[z] = {}

            min_tx = int(math.floor((cl[:, 0].min() - margin) / tile_w)) - 1
            max_tx = int(math.ceil((cl[:, 0].max() + margin) / tile_w)) + 1
            min_ty = int(math.floor((cl[:, 1].min() - margin) / tile_h)) - 1
            max_ty = int(math.ceil((cl[:, 1].max() + margin) / tile_h)) + 1

            for tx in range(min_tx, max_tx + 1):
                wx0 = tx * tile_w
                wx1 = (tx + 1) * tile_w
                mask_x = (cl[:, 0] >= wx0 - margin) & (cl[:, 0] <= wx1 + margin)
                if not np.any(mask_x):
                    continue
                for ty in range(min_ty, max_ty + 1):
                    wy0 = ty * tile_h
                    wy1 = (ty + 1) * tile_h
                    mask = mask_x & (cl[:, 1] >= wy0 - margin) & (cl[:, 1] <= wy1 + margin)
                    idx = np.where(mask)[0]
                    if len(idx) > 0:
                        self.tile_indices[z][(tx, ty)] = idx

    def set_zoom(self, z: float) -> None:
        if z in ZOOM_LEVELS and z != self.zoom:
            self.zoom = z
            self.cache.clear()

    def zoom_in(self) -> float:
        idx = ZOOM_LEVELS.index(self.zoom)
        if idx < len(ZOOM_LEVELS) - 1:
            self.set_zoom(ZOOM_LEVELS[idx + 1])
        return self.zoom

    def zoom_out(self) -> float:
        idx = ZOOM_LEVELS.index(self.zoom)
        if idx > 0:
            self.set_zoom(ZOOM_LEVELS[idx - 1])
        return self.zoom

    def reset_rubber(self) -> None:
        self.rubber_segments.clear()
        self.cache.clear()

    def get_tile_bounds(self, tx: int, ty: int, z: float | None = None) -> tuple[float, float, float, float]:
        if z is None:
            z = self.zoom
        tw = TILE_SIZE / z
        th = TILE_SIZE / z
        return (tx * tw, ty * th, (tx + 1) * tw, (ty + 1) * th)

    def _bake_tile(self, key: tuple[float, int, int]) -> pygame.Surface:
        z, tx, ty = key
        world_rect = self.get_tile_bounds(tx, ty, z)
        sample_indices = self.tile_indices.get(z, {}).get((tx, ty), None)

        surf_2x = pygame.Surface((TILE_SIZE * 2, TILE_SIZE * 2))
        draw_static_world(surf_2x, self.track, world_rect, z * 2.0, sample_indices)
        scaled = pygame.transform.smoothscale(surf_2x, (TILE_SIZE, TILE_SIZE))
        try:
            tile_surf = scaled.convert()
        except Exception:
            tile_surf = scaled

        if key in self.rubber_segments:
            min_wx, min_wy = world_rect[0], world_rect[1]
            for p0, p1 in self.rubber_segments[key]:
                s0 = (int(round((p0[0] - min_wx) * z)), int(round((p0[1] - min_wy) * z)))
                s1 = (int(round((p1[0] - min_wx) * z)), int(round((p1[1] - min_wy) * z)))
                pygame.draw.line(tile_surf, (16, 17, 19), s0, s1, 2)

        self.cache[key] = tile_surf
        self.cache.move_to_end(key)
        if len(self.cache) > MAX_CACHE_TILES:
            self.cache.popitem(last=False)
        return tile_surf

    def get_tile(self, tx: int, ty: int) -> pygame.Surface:
        key = (self.zoom, tx, ty)
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return self._bake_tile(key)

    def add_rubber_segment(self, p0: np.ndarray, p1: np.ndarray) -> None:
        tw = TILE_SIZE / self.zoom
        th = TILE_SIZE / self.zoom
        tx0 = int(math.floor(min(p0[0], p1[0]) / tw))
        tx1 = int(math.floor(max(p0[0], p1[0]) / tw))
        ty0 = int(math.floor(min(p0[1], p1[1]) / th))
        ty1 = int(math.floor(max(p0[1], p1[1]) / th))

        for tx in range(tx0, tx1 + 1):
            for ty in range(ty0, ty1 + 1):
                key = (self.zoom, tx, ty)
                if key not in self.rubber_segments:
                    self.rubber_segments[key] = []
                segs = self.rubber_segments[key]
                if len(segs) < MAX_RUBBER_PER_TILE:
                    segs.append((p0.copy(), p1.copy()))
                if key in self.cache:
                    surf = self.cache[key]
                    min_wx, min_wy, _, _ = self.get_tile_bounds(tx, ty)
                    s0 = (int(round((p0[0] - min_wx) * self.zoom)), int(round((p0[1] - min_wy) * self.zoom)))
                    s1 = (int(round((p1[0] - min_wx) * self.zoom)), int(round((p1[1] - min_wy) * self.zoom)))
                    pygame.draw.line(surf, (16, 17, 19), s0, s1, 2)

    def prebake_spawn_ring(self, spawn_pos: np.ndarray, view_size: tuple[int, int]) -> None:
        vw, vh = view_size
        tw = TILE_SIZE / self.zoom
        th = TILE_SIZE / self.zoom
        min_wx = spawn_pos[0] - (vw * 0.5) / self.zoom
        max_wx = spawn_pos[0] + (vw * 0.5) / self.zoom
        min_wy = spawn_pos[1] - (vh * 0.5) / self.zoom
        max_wy = spawn_pos[1] + (vh * 0.5) / self.zoom

        min_tx = int(math.floor(min_wx / tw)) - 1
        max_tx = int(math.floor(max_wx / tw)) + 1
        min_ty = int(math.floor(min_wy / th)) - 1
        max_ty = int(math.floor(max_wy / th)) + 1

        for tx in range(min_tx, max_tx + 1):
            for ty in range(min_ty, max_ty + 1):
                key = (self.zoom, tx, ty)
                if key not in self.cache:
                    self._bake_tile(key)

    def update_frame(
        self,
        cam_pos: np.ndarray,
        view_size: tuple[int, int],
        leader_s: float,
    ) -> None:
        vw, vh = view_size
        tw = TILE_SIZE / self.zoom
        th = TILE_SIZE / self.zoom
        min_wx = cam_pos[0] - (vw * 0.5) / self.zoom
        max_wx = cam_pos[0] + (vw * 0.5) / self.zoom
        min_wy = cam_pos[1] - (vh * 0.5) / self.zoom
        max_wy = cam_pos[1] + (vh * 0.5) / self.zoom

        min_tx = int(math.floor(min_wx / tw))
        max_tx = int(math.floor(max_wx / tw))
        min_ty = int(math.floor(min_wy / th))
        max_ty = int(math.floor(max_wy / th))

        needed_urgent = []
        for ty in range(min_ty, max_ty + 1):
            for tx in range(min_tx, max_tx + 1):
                key = (self.zoom, tx, ty)
                if key not in self.cache:
                    needed_urgent.append(key)

        for key in needed_urgent:
            self._bake_tile(key)

        bakes_left = 2
        ring_candidates = []
        for ty in range(min_ty - 1, max_ty + 2):
            for tx in range(min_tx - 1, max_tx + 2):
                if min_tx <= tx <= max_tx and min_ty <= ty <= max_ty:
                    continue
                key = (self.zoom, tx, ty)
                if key not in self.cache:
                    ring_candidates.append(key)

        for key in ring_candidates:
            if bakes_left <= 0:
                break
            self._bake_tile(key)
            bakes_left -= 1

        if bakes_left > 0:
            lap_len = self.track.total_length
            cl = self.track.centerline
            for ds in range(-60, 520, 60):
                if bakes_left <= 0:
                    break
                s_probe = (leader_s + ds) % lap_len
                idx = int(np.clip(s_probe / 2.0, 0, len(cl) - 1))
                p = cl[idx]
                tx = int(math.floor(p[0] / tw))
                ty = int(math.floor(p[1] / th))
                key = (self.zoom, tx, ty)
                if key not in self.cache:
                    self._bake_tile(key)
                    bakes_left -= 1

    def render_tiles(
        self,
        target_surf: pygame.Surface,
        cam_pos: np.ndarray,
        view_size: tuple[int, int],
    ) -> None:
        vw, vh = view_size
        tw = TILE_SIZE / self.zoom
        th = TILE_SIZE / self.zoom
        vc_x = vw * 0.5
        vc_y = vh * 0.5

        min_wx = cam_pos[0] - vc_x / self.zoom
        max_wx = cam_pos[0] + vc_x / self.zoom
        min_wy = cam_pos[1] - vc_y / self.zoom
        max_wy = cam_pos[1] + vc_y / self.zoom

        min_tx = int(math.floor(min_wx / tw))
        max_tx = int(math.floor(max_wx / tw))
        min_ty = int(math.floor(min_wy / th))
        max_ty = int(math.floor(max_wy / th))

        for ty in range(min_ty, max_ty + 1):
            for tx in range(min_tx, max_tx + 1):
                tile = self.get_tile(tx, ty)
                t_wx = tx * tw
                t_wy = ty * th
                sx = int(round((t_wx - cam_pos[0]) * self.zoom + vc_x))
                sy = int(round((t_wy - cam_pos[1]) * self.zoom + vc_y))
                target_surf.blit(tile, (sx, sy))
