"""
dimensioning.py - Dimension lines, arrowheads, and annotation placement.

Generates professional-grade dimension lines with:
- Extension lines and tick marks / arrowheads
- Text labels with anti-overlap logic
- Configurable offsets to avoid geometry clutter
"""

import math
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class DimensionLine:
    """A fully computed dimension line ready for rendering."""
    p1: Tuple[float, float]          # Start of dimension (on geometry)
    p2: Tuple[float, float]          # End of dimension (on geometry)
    ext1_start: Tuple[float, float]  # Extension line 1 start (on geometry)
    ext1_end: Tuple[float, float]    # Extension line 1 end (offset)
    ext2_start: Tuple[float, float]  # Extension line 2 start (on geometry)
    ext2_end: Tuple[float, float]    # Extension line 2 end (offset)
    dim_start: Tuple[float, float]   # Dimension line start (between ext lines)
    dim_end: Tuple[float, float]     # Dimension line end
    text_pos: Tuple[float, float]    # Label position
    text_angle: float                # Text rotation in degrees
    label: str                       # Dimension text (e.g., "2.50m")
    arrow1: List[Tuple[float, float]]  # Arrowhead polygon at start
    arrow2: List[Tuple[float, float]]  # Arrowhead polygon at end


@dataclass
class AnnotationLabel:
    """A text label placed in the drawing."""
    position: Tuple[float, float]
    text: str
    angle: float = 0.0
    font_size: float = 7.0
    anchor: str = "center"


class DimensioningEngine:
    """Computes dimension lines and annotations with anti-overlap."""

    def __init__(self, offset: float = 20.0, arrow_size: float = 6.0,
                 font_size: float = 8.0, min_gap: float = 14.0):
        self.offset = offset
        self.arrow_size = arrow_size
        self.font_size = font_size
        self.min_gap = min_gap
        self._placed_labels: List[Tuple[float, float, float, float]] = []  # (x, y, w, h)
        self._dim_count = 0

    def reset(self):
        self._placed_labels.clear()
        self._dim_count = 0

    def create_dimension(self, p1: Tuple[float, float], p2: Tuple[float, float],
                         label: str, side: str = "auto",
                         offset_multiplier: float = 1.0) -> DimensionLine:
        """
        Create a dimension line between two 2D points.

        Args:
            p1, p2: The two geometry points to dimension
            label: Text to display (e.g., "5.00m")
            side: "left", "right", "auto" — which side to place dimension
            offset_multiplier: Multiply the base offset for stacking
        """
        self._dim_count += 1
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        length = math.sqrt(dx**2 + dy**2)
        if length < 1e-6:
            length = 1.0

        # Unit direction along dimension
        ux = dx / length
        uy = dy / length

        # Normal (perpendicular) direction
        if side == "right":
            nx, ny = uy, -ux
        elif side == "left":
            nx, ny = -uy, ux
        else:
            nx, ny = -uy, ux  # Default: left side

        total_offset = self.offset * offset_multiplier

        # Extension lines
        ext_gap = 3.0  # Small gap between geometry and extension line start
        ext1_start = (p1[0] + nx * ext_gap, p1[1] + ny * ext_gap)
        ext1_end = (p1[0] + nx * (total_offset + 4), p1[1] + ny * (total_offset + 4))
        ext2_start = (p2[0] + nx * ext_gap, p2[1] + ny * ext_gap)
        ext2_end = (p2[0] + nx * (total_offset + 4), p2[1] + ny * (total_offset + 4))

        # Dimension line (at offset distance)
        dim_start = (p1[0] + nx * total_offset, p1[1] + ny * total_offset)
        dim_end = (p2[0] + nx * total_offset, p2[1] + ny * total_offset)

        # Text position (midpoint of dimension line)
        text_x = (dim_start[0] + dim_end[0]) / 2
        text_y = (dim_start[1] + dim_end[1]) / 2

        # Text angle (parallel to dimension line, always readable)
        text_angle = math.degrees(math.atan2(dy, dx))
        if text_angle > 90:
            text_angle -= 180
        elif text_angle < -90:
            text_angle += 180

        # Offset text slightly from dimension line for readability
        text_offset = self.font_size * 0.8
        text_pos = (text_x + nx * text_offset, text_y + ny * text_offset)

        # Anti-overlap check
        text_pos = self._resolve_overlap(text_pos, label)

        # Arrowheads
        arrow1 = self._make_arrowhead(dim_start, ux, uy, towards_end=True)
        arrow2 = self._make_arrowhead(dim_end, ux, uy, towards_end=False)

        return DimensionLine(
            p1=p1, p2=p2,
            ext1_start=ext1_start, ext1_end=ext1_end,
            ext2_start=ext2_start, ext2_end=ext2_end,
            dim_start=dim_start, dim_end=dim_end,
            text_pos=text_pos, text_angle=text_angle,
            label=label,
            arrow1=arrow1, arrow2=arrow2,
        )

    def _make_arrowhead(self, tip: Tuple[float, float],
                        ux: float, uy: float,
                        towards_end: bool) -> List[Tuple[float, float]]:
        """Create a filled arrowhead triangle at the given tip."""
        s = self.arrow_size
        if not towards_end:
            ux, uy = -ux, -uy

        # Tip
        tx, ty = tip
        # Back-left and back-right of arrowhead
        bx = tx - ux * s
        by = ty - uy * s
        nx, ny = -uy, ux
        left = (bx + nx * s * 0.3, by + ny * s * 0.3)
        right = (bx - nx * s * 0.3, by - ny * s * 0.3)

        return [tip, left, right]

    def _resolve_overlap(self, pos: Tuple[float, float],
                         label: str) -> Tuple[float, float]:
        """Nudge label position to avoid overlapping existing labels."""
        est_w = len(label) * self.font_size * 0.6
        est_h = self.font_size * 1.4
        x, y = pos
        original_y = y

        for attempt in range(8):
            overlaps = False
            for ox, oy, ow, oh in self._placed_labels:
                if (abs(x - ox) < (est_w + ow) / 2 + self.min_gap and
                        abs(y - oy) < (est_h + oh) / 2 + self.min_gap):
                    overlaps = True
                    break
            if not overlaps:
                break
            # Alternate up/down nudging rather than always down
            shift = (est_h + self.min_gap)
            if attempt % 2 == 0:
                y = original_y + (attempt // 2 + 1) * shift
            else:
                y = original_y - (attempt // 2 + 1) * shift

        self._placed_labels.append((x, y, est_w, est_h))
        return (x, y)

    def create_annotation(self, position: Tuple[float, float],
                          text: str, angle: float = 0.0) -> AnnotationLabel:
        """Create a text annotation with anti-overlap."""
        pos = self._resolve_overlap(position, text)
        return AnnotationLabel(position=pos, text=text, angle=angle,
                               font_size=self.font_size - 1)


def compute_wall_dimensions(scene, projector, dim_engine, units="meters"):
    """Generate dimension lines for all walls."""
    from scene_parser import format_dimension
    dimensions = []

    for i, wall in enumerate(scene.walls):
        p1_2d = projector.project(wall.start)
        p2_2d = projector.project(wall.end)

        # Wall length
        label = format_dimension(wall.length_2d, units)
        offset_mult = 1.0 + (i % 3) * 0.5
        dim = dim_engine.create_dimension(p1_2d, p2_2d, label,
                                          offset_multiplier=offset_mult)
        dimensions.append(dim)

    return dimensions


def compute_wall_height_dimensions(scene, projector, dim_engine, units="meters"):
    """Generate height dimension lines for walls (front elevation view)."""
    from scene_parser import Vector3, format_dimension
    dimensions = []

    for i, wall in enumerate(scene.walls):
        # Vertical dimension: from base to top
        base = projector.project(wall.start)
        top = projector.project(Vector3(wall.start.x, wall.start.y, wall.height))

        label = format_dimension(wall.height, units)
        dim = dim_engine.create_dimension(base, top, label, side="right",
                                          offset_multiplier=1.0 + i * 0.4)
        dimensions.append(dim)

    return dimensions


def compute_asset_dimensions(scene, projector, dim_engine, units="meters"):
    """Generate dimension lines for furniture and assets."""
    from scene_parser import format_dimension
    dimensions = []

    for item in scene.furniture + scene.assets:
        p = projector.project(item.position)
        # Width dimension
        from scene_parser import Vector3
        left = Vector3(item.position.x - item.size.x/2, item.position.y, item.position.z)
        right = Vector3(item.position.x + item.size.x/2, item.position.y, item.position.z)
        p_left = projector.project(left)
        p_right = projector.project(right)

        label = format_dimension(item.size.x, units)
        dim = dim_engine.create_dimension(p_left, p_right, label, side="right",
                                          offset_multiplier=0.8)
        dimensions.append(dim)

    return dimensions


def compute_annotations(scene, projector, dim_engine):
    """Generate text annotations for all labeled objects."""
    annotations = []

    for wall in scene.walls:
        mid = projector.project(
            type(wall.start)(
                (wall.start.x + wall.end.x) / 2,
                (wall.start.y + wall.end.y) / 2,
                (wall.start.z + wall.end.z) / 2,
            )
        )
        annotations.append(dim_engine.create_annotation(mid, wall.label))

    for item in scene.furniture:
        p = projector.project(item.position)
        annotations.append(dim_engine.create_annotation(p, item.label))

    for item in scene.assets:
        p = projector.project(item.position)
        annotations.append(dim_engine.create_annotation(p, item.label))

    for spot in scene.track_spotlights:
        p = projector.project(spot.position)
        annotations.append(dim_engine.create_annotation(p, spot.label))

    for truss in scene.truss_segments:
        mid_pos = type(truss.start)(
            (truss.start.x + truss.end.x) / 2,
            (truss.start.y + truss.end.y) / 2,
            (truss.start.z + truss.end.z) / 2,
        )
        p = projector.project(mid_pos)
        annotations.append(dim_engine.create_annotation(p, truss.label))

    return annotations
