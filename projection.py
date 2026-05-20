"""
projection.py - 3D to 2D projection engine.

Supports Top-Down (plan), Front-Elevation, and Isometric projections
for converting 3D scene geometry to 2D drawing coordinates.
"""

import math
import numpy as np
from typing import Tuple, List
from scene_parser import Vector3


class ProjectionEngine:
    """Handles 3D-to-2D coordinate projection for technical drawings."""

    def __init__(self, view: str = "top", scale: float = 1.0,
                 canvas_width: float = 800, canvas_height: float = 600,
                 margin: float = 80):
        self.view = view
        self.scale = scale
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.margin = margin
        self._offset_x = 0.0
        self._offset_y = 0.0

    def set_auto_fit(self, scene_bounds: Tuple[Vector3, Vector3]):
        """Calculate scale and offset to fit scene into canvas."""
        bmin, bmax = scene_bounds
        draw_w = self.canvas_width - 2 * self.margin
        draw_h = self.canvas_height - 2 * self.margin

        if self.view == "top":
            scene_w = bmax.x - bmin.x
            scene_h = bmax.y - bmin.y
        elif self.view == "front":
            scene_w = bmax.x - bmin.x
            scene_h = bmax.z - bmin.z
        elif self.view == "iso":
            # Isometric bounding — approximate
            scene_w = (bmax.x - bmin.x) + (bmax.y - bmin.y)
            scene_h = (bmax.x - bmin.x) * 0.5 + (bmax.y - bmin.y) * 0.5 + (bmax.z - bmin.z)
        else:
            scene_w = bmax.x - bmin.x
            scene_h = bmax.y - bmin.y

        if scene_w <= 0:
            scene_w = 1
        if scene_h <= 0:
            scene_h = 1

        sx = draw_w / scene_w
        sy = draw_h / scene_h
        self.scale = min(sx, sy)

        if self.view == "top":
            self._offset_x = self.margin - bmin.x * self.scale
            self._offset_y = self.margin - bmin.y * self.scale
        elif self.view == "front":
            self._offset_x = self.margin - bmin.x * self.scale
            self._offset_y = self.margin - bmin.z * self.scale
        elif self.view == "iso":
            iso_min = self._iso_project_raw(bmin.x, bmin.y, bmin.z)
            iso_max = self._iso_project_raw(bmax.x, bmax.y, bmax.z)
            all_corners = [
                self._iso_project_raw(bmin.x, bmin.y, bmin.z),
                self._iso_project_raw(bmax.x, bmin.y, bmin.z),
                self._iso_project_raw(bmin.x, bmax.y, bmin.z),
                self._iso_project_raw(bmax.x, bmax.y, bmin.z),
                self._iso_project_raw(bmin.x, bmin.y, bmax.z),
                self._iso_project_raw(bmax.x, bmin.y, bmax.z),
                self._iso_project_raw(bmin.x, bmax.y, bmax.z),
                self._iso_project_raw(bmax.x, bmax.y, bmax.z),
            ]
            xs = [c[0] for c in all_corners]
            ys = [c[1] for c in all_corners]
            raw_min_x, raw_max_x = min(xs), max(xs)
            raw_min_y, raw_max_y = min(ys), max(ys)
            raw_w = raw_max_x - raw_min_x
            raw_h = raw_max_y - raw_min_y
            if raw_w <= 0:
                raw_w = 1
            if raw_h <= 0:
                raw_h = 1
            sx = draw_w / raw_w
            sy = draw_h / raw_h
            self.scale = min(sx, sy)
            self._offset_x = self.margin - raw_min_x * self.scale
            self._offset_y = self.margin - raw_min_y * self.scale

    def _iso_project_raw(self, x: float, y: float, z: float) -> Tuple[float, float]:
        """Raw isometric projection (30-degree) without scale/offset."""
        angle = math.radians(30)
        ix = (x - y) * math.cos(angle)
        iy = (x + y) * math.sin(angle) - z
        return (ix, iy)

    def project(self, point: Vector3) -> Tuple[float, float]:
        """Project a 3D point to 2D canvas coordinates."""
        if self.view == "top":
            return self._project_top(point)
        elif self.view == "front":
            return self._project_front(point)
        elif self.view == "iso":
            return self._project_iso(point)
        else:
            return self._project_top(point)

    def _project_top(self, p: Vector3) -> Tuple[float, float]:
        """Top-down: X→right, Y→down (flipped for screen coords)."""
        sx = p.x * self.scale + self._offset_x
        sy = self.canvas_height - (p.y * self.scale + self._offset_y)
        return (sx, sy)

    def _project_front(self, p: Vector3) -> Tuple[float, float]:
        """Front elevation: X→right, Z→up."""
        sx = p.x * self.scale + self._offset_x
        sy = self.canvas_height - (p.z * self.scale + self._offset_y)
        return (sx, sy)

    def _project_iso(self, p: Vector3) -> Tuple[float, float]:
        """Isometric projection (30-degree axonometric)."""
        ix, iy = self._iso_project_raw(p.x, p.y, p.z)
        sx = ix * self.scale + self._offset_x
        sy = self.canvas_height - (iy * self.scale + self._offset_y)
        return (sx, sy)

    def project_line(self, start: Vector3, end: Vector3) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        return (self.project(start), self.project(end))

    def project_rect(self, position: Vector3, size: Vector3, rotation: float = 0) -> List[Tuple[float, float]]:
        """Project a 3D axis-aligned box footprint to 2D polygon."""
        cx, cy = position.x, position.y
        hw, hd = size.x / 2, size.y / 2

        corners_local = [
            (-hw, -hd),
            (hw, -hd),
            (hw, hd),
            (-hw, hd),
        ]

        rad = math.radians(rotation)
        cos_r, sin_r = math.cos(rad), math.sin(rad)

        corners_3d = []
        for lx, ly in corners_local:
            rx = lx * cos_r - ly * sin_r + cx
            ry = lx * sin_r + ly * cos_r + cy
            corners_3d.append(Vector3(rx, ry, position.z))

        return [self.project(c) for c in corners_3d]

    def project_rect_elevation(self, position: Vector3, size: Vector3) -> List[Tuple[float, float]]:
        """Project a box as a front-elevation rectangle (X × Z)."""
        cx, cz = position.x, position.z
        hw, hh = size.x / 2, size.z / 2

        corners = [
            Vector3(cx - hw, position.y, cz),
            Vector3(cx + hw, position.y, cz),
            Vector3(cx + hw, position.y, cz + hh),
            Vector3(cx - hw, position.y, cz + hh),
        ]
        return [self.project(c) for c in corners]


def compute_scene_bounds(scene) -> Tuple[Vector3, Vector3]:
    """Compute the axis-aligned bounding box of the entire scene."""
    all_points: List[Vector3] = []

    for w in scene.walls:
        all_points.append(w.start)
        all_points.append(w.end)
        all_points.append(Vector3(w.end.x, w.end.y, w.height))

    for fl in scene.floors:
        all_points.extend(fl.corners)

    for d in scene.doors:
        all_points.append(d.position)

    for s in scene.track_spotlights:
        all_points.append(s.position)

    for l in scene.led_strips:
        all_points.append(l.start)
        all_points.append(l.end)

    for t in scene.truss_segments:
        all_points.append(t.start)
        all_points.append(t.end)

    for f in scene.furniture:
        p = f.position
        s = f.size
        all_points.append(Vector3(p.x - s.x/2, p.y - s.y/2, p.z))
        all_points.append(Vector3(p.x + s.x/2, p.y + s.y/2, p.z + s.z))

    for a in scene.assets:
        p = a.position
        s = a.size
        all_points.append(Vector3(p.x - s.x/2, p.y - s.y/2, p.z))
        all_points.append(Vector3(p.x + s.x/2, p.y + s.y/2, p.z + s.z))

    if not all_points:
        return (Vector3(0, 0, 0), Vector3(10, 10, 5))

    xs = [p.x for p in all_points]
    ys = [p.y for p in all_points]
    zs = [p.z for p in all_points]

    return (
        Vector3(min(xs), min(ys), min(zs)),
        Vector3(max(xs), max(ys), max(zs)),
    )
