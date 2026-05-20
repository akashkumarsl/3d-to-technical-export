"""
renderer_dxf.py - DXF rendering engine with strictly managed named layers.

Generates a multi-layer DXF file with:
- Layer 1: ARCHITECTURE (walls, floors, doors, furniture, assets)
- Layer 2: LIGHTING_RIGGING (truss, spotlights, LED strips)
- Layer 3: ANNOTATIONS (dimensions, labels, legend)
"""

import math
import ezdxf
from ezdxf.enums import TextEntityAlignment
from ezdxf.math import Vec3 as DxfVec3

from scene_parser import Scene, Vector3, format_dimension
from projection import ProjectionEngine, compute_scene_bounds
from dimensioning import (
    DimensioningEngine,
    compute_wall_dimensions, compute_wall_height_dimensions,
    compute_asset_dimensions, compute_annotations,
)


# DXF color indices (ACI)
ACI_WHITE = 7
ACI_RED = 1
ACI_YELLOW = 2
ACI_GREEN = 3
ACI_CYAN = 4
ACI_BLUE = 5
ACI_MAGENTA = 6
ACI_GRAY = 8
ACI_LIGHT_GRAY = 9

# Layer names
LAYER_ARCH = "1-ARCHITECTURE"
LAYER_LIGHTING = "2-LIGHTING_RIGGING"
LAYER_ANNOT = "3-ANNOTATIONS"


class DXFRenderer:
    """Renders a Scene to a multi-layered DXF file."""

    def __init__(self, output_path: str, units: str = "meters"):
        self.output_path = output_path
        self.units = units

    def render(self, scene: Scene, views: list = None):
        if views is None:
            views = ["top"]

        doc = ezdxf.new("R2010")
        self._setup_layers(doc)

        for view in views:
            layout_name = f"{view.upper()}_VIEW"
            if view == views[0]:
                msp = doc.modelspace()
                self._render_view(msp, scene, view)
            else:
                layout = doc.layouts.new(layout_name)
                self._render_view(layout, scene, view)

        doc.saveas(self.output_path)

    def _setup_layers(self, doc):
        """Create strictly named layers with distinct colors and line weights."""
        layers = doc.layers

        layers.add(LAYER_ARCH, color=ACI_WHITE)
        arch_layer = layers.get(LAYER_ARCH)
        arch_layer.dxf.lineweight = 35  # 0.35mm

        layers.add(LAYER_LIGHTING, color=ACI_CYAN)
        light_layer = layers.get(LAYER_LIGHTING)
        light_layer.dxf.lineweight = 25  # 0.25mm

        layers.add(LAYER_ANNOT, color=ACI_GREEN)
        annot_layer = layers.get(LAYER_ANNOT)
        annot_layer.dxf.lineweight = 13  # 0.13mm

    def _render_view(self, msp, scene: Scene, view: str):
        """Render all scene elements into a layout."""
        bounds = compute_scene_bounds(scene)

        # For DXF we use real-world coordinates (no pixel scaling)
        # Scale factor converts to drawing units (cm for detail)
        scale = 100.0 if self.units == "meters" else 1.0

        proj = ProjectionEngine(
            view=view, scale=scale,
            canvas_width=10000, canvas_height=10000, margin=0
        )
        # For DXF, don't auto-fit — use real coordinates
        proj.scale = scale
        proj._offset_x = 0
        proj._offset_y = 0

        # --- Architecture Layer ---
        self._draw_floors_dxf(msp, scene, proj)
        self._draw_walls_dxf(msp, scene, proj, view)
        self._draw_doors_dxf(msp, scene, proj, view)
        self._draw_furniture_dxf(msp, scene, proj, view)
        self._draw_assets_dxf(msp, scene, proj, view)

        # --- Lighting & Rigging Layer ---
        self._draw_truss_dxf(msp, scene, proj)
        self._draw_spotlights_dxf(msp, scene, proj)
        self._draw_led_strips_dxf(msp, scene, proj)

        # --- Annotations Layer ---
        dim_engine = DimensioningEngine(offset=15 * scale / 100,
                                        arrow_size=4 * scale / 100,
                                        font_size=6)
        if view == "top":
            dims = compute_wall_dimensions(scene, proj, dim_engine, self.units)
            dims += compute_asset_dimensions(scene, proj, dim_engine, self.units)
        elif view == "front":
            dims = compute_wall_dimensions(scene, proj, dim_engine, self.units)
            dims += compute_wall_height_dimensions(scene, proj, dim_engine, self.units)
        else:
            dims = compute_wall_dimensions(scene, proj, dim_engine, self.units)

        for dim in dims:
            self._draw_dimension_dxf(msp, dim)

        annotations = compute_annotations(scene, proj, dim_engine)
        for ann in annotations:
            self._draw_annotation_dxf(msp, ann)

        # Legend
        self._draw_legend_dxf(msp, bounds, scale)

    # ---------------------------------------------------------------
    # Architecture
    # ---------------------------------------------------------------

    def _draw_floors_dxf(self, msp, scene, proj):
        for floor in scene.floors:
            pts = [proj.project(c) for c in floor.corners]
            pts_3d = [(p[0], p[1], 0) for p in pts]
            if len(pts_3d) >= 3:
                pts_3d.append(pts_3d[0])  # Close
                msp.add_lwpolyline(pts_3d, dxfattribs={
                    "layer": LAYER_ARCH, "color": ACI_LIGHT_GRAY
                })

    def _draw_walls_dxf(self, msp, scene, proj, view):
        for wall in scene.walls:
            if view == "top":
                dx = wall.end.x - wall.start.x
                dy = wall.end.y - wall.start.y
                length = math.sqrt(dx**2 + dy**2)
                if length < 1e-6:
                    continue
                nx = -dy / length * wall.thickness / 2
                ny = dx / length * wall.thickness / 2

                corners = [
                    proj.project(Vector3(wall.start.x + nx, wall.start.y + ny, 0)),
                    proj.project(Vector3(wall.end.x + nx, wall.end.y + ny, 0)),
                    proj.project(Vector3(wall.end.x - nx, wall.end.y - ny, 0)),
                    proj.project(Vector3(wall.start.x - nx, wall.start.y - ny, 0)),
                ]
                pts = [(c[0], c[1], 0) for c in corners]
                pts.append(pts[0])
                msp.add_lwpolyline(pts, dxfattribs={
                    "layer": LAYER_ARCH, "color": ACI_WHITE
                })
            elif view == "front":
                p1 = proj.project(wall.start)
                p2 = proj.project(wall.end)
                p3 = proj.project(Vector3(wall.end.x, wall.end.y, wall.height))
                p4 = proj.project(Vector3(wall.start.x, wall.start.y, wall.height))
                pts = [(p[0], p[1], 0) for p in [p1, p2, p3, p4, p1]]
                msp.add_lwpolyline(pts, dxfattribs={
                    "layer": LAYER_ARCH, "color": ACI_WHITE
                })
            else:
                # Isometric — front face
                s, e = wall.start, wall.end
                h = wall.height
                pts_raw = [
                    proj.project(s),
                    proj.project(e),
                    proj.project(Vector3(e.x, e.y, h)),
                    proj.project(Vector3(s.x, s.y, h)),
                ]
                pts = [(p[0], p[1], 0) for p in pts_raw]
                pts.append(pts[0])
                msp.add_lwpolyline(pts, dxfattribs={
                    "layer": LAYER_ARCH, "color": ACI_WHITE
                })

    def _draw_doors_dxf(self, msp, scene, proj, view):
        for door in scene.doors:
            pos = door.position
            hw = door.width / 2
            left = proj.project(Vector3(pos.x - hw, pos.y, pos.z))
            right = proj.project(Vector3(pos.x + hw, pos.y, pos.z))
            msp.add_line(
                (left[0], left[1]), (right[0], right[1]),
                dxfattribs={"layer": LAYER_ARCH, "color": ACI_GRAY}
            )
            # Swing arc
            if view == "top":
                radius = math.sqrt((right[0]-left[0])**2 + (right[1]-left[1])**2)
                if radius > 0:
                    msp.add_arc(
                        center=(left[0], left[1]),
                        radius=radius,
                        start_angle=0, end_angle=90,
                        dxfattribs={"layer": LAYER_ARCH, "color": ACI_GRAY,
                                    "linetype": "DASHED"}
                    )

    def _draw_furniture_dxf(self, msp, scene, proj, view):
        for item in scene.furniture:
            pts = proj.project_rect(item.position, item.size, item.rotation)
            pts_3d = [(p[0], p[1], 0) for p in pts]
            pts_3d.append(pts_3d[0])
            color = ACI_GRAY if item.type == "chair" else ACI_LIGHT_GRAY
            msp.add_lwpolyline(pts_3d, dxfattribs={
                "layer": LAYER_ARCH, "color": color
            })

    def _draw_assets_dxf(self, msp, scene, proj, view):
        for asset in scene.assets:
            if asset.type == "socket":
                center = proj.project(asset.position)
                msp.add_circle(
                    (center[0], center[1]), radius=3,
                    dxfattribs={"layer": LAYER_ARCH, "color": ACI_GREEN}
                )
            else:
                pts = proj.project_rect(asset.position, asset.size, asset.rotation)
                pts_3d = [(p[0], p[1], 0) for p in pts]
                pts_3d.append(pts_3d[0])
                msp.add_lwpolyline(pts_3d, dxfattribs={
                    "layer": LAYER_ARCH, "color": ACI_MAGENTA
                })

    # ---------------------------------------------------------------
    # Lighting & Rigging
    # ---------------------------------------------------------------

    def _draw_truss_dxf(self, msp, scene, proj):
        for truss in scene.truss_segments:
            p1 = proj.project(truss.start)
            p2 = proj.project(truss.end)

            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            length = math.sqrt(dx**2 + dy**2)
            if length < 1e-6:
                continue
            nx = -dy / length * 3
            ny = dx / length * 3

            # Double line
            msp.add_line(
                (p1[0]+nx, p1[1]+ny), (p2[0]+nx, p2[1]+ny),
                dxfattribs={"layer": LAYER_LIGHTING, "color": ACI_BLUE}
            )
            msp.add_line(
                (p1[0]-nx, p1[1]-ny), (p2[0]-nx, p2[1]-ny),
                dxfattribs={"layer": LAYER_LIGHTING, "color": ACI_BLUE}
            )

            # Cross bracing
            segments = max(int(length / 20), 2)
            for i in range(segments):
                t1 = i / segments
                t2 = (i + 1) / segments
                x1 = p1[0] + dx * t1
                y1 = p1[1] + dy * t1
                x2 = p1[0] + dx * t2
                y2 = p1[1] + dy * t2
                if i % 2 == 0:
                    msp.add_line((x1+nx, y1+ny), (x2-nx, y2-ny),
                                 dxfattribs={"layer": LAYER_LIGHTING, "color": ACI_BLUE})
                else:
                    msp.add_line((x1-nx, y1-ny), (x2+nx, y2+ny),
                                 dxfattribs={"layer": LAYER_LIGHTING, "color": ACI_BLUE})

    def _draw_spotlights_dxf(self, msp, scene, proj):
        for spot in scene.track_spotlights:
            center = proj.project(spot.position)
            r = 5
            msp.add_circle(
                (center[0], center[1]), radius=r,
                dxfattribs={"layer": LAYER_LIGHTING, "color": ACI_YELLOW}
            )
            # Cross
            msp.add_line((center[0]-r, center[1]), (center[0]+r, center[1]),
                         dxfattribs={"layer": LAYER_LIGHTING, "color": ACI_YELLOW})
            msp.add_line((center[0], center[1]-r), (center[0], center[1]+r),
                         dxfattribs={"layer": LAYER_LIGHTING, "color": ACI_YELLOW})

    def _draw_led_strips_dxf(self, msp, scene, proj):
        for led in scene.led_strips:
            p1 = proj.project(led.start)
            p2 = proj.project(led.end)
            msp.add_line(
                (p1[0], p1[1]), (p2[0], p2[1]),
                dxfattribs={"layer": LAYER_LIGHTING, "color": ACI_YELLOW,
                            "linetype": "DASHED"}
            )

    # ---------------------------------------------------------------
    # Annotations
    # ---------------------------------------------------------------

    def _draw_dimension_dxf(self, msp, dim):
        """Draw dimension using DXF entities."""
        attrs = {"layer": LAYER_ANNOT, "color": ACI_GREEN}

        # Extension lines
        msp.add_line(
            (dim.ext1_start[0], dim.ext1_start[1]),
            (dim.ext1_end[0], dim.ext1_end[1]),
            dxfattribs=attrs
        )
        msp.add_line(
            (dim.ext2_start[0], dim.ext2_start[1]),
            (dim.ext2_end[0], dim.ext2_end[1]),
            dxfattribs=attrs
        )

        # Dimension line
        msp.add_line(
            (dim.dim_start[0], dim.dim_start[1]),
            (dim.dim_end[0], dim.dim_end[1]),
            dxfattribs=attrs
        )

        # Arrowheads as solid fills
        for arrow in [dim.arrow1, dim.arrow2]:
            if len(arrow) >= 3:
                pts = [(p[0], p[1]) for p in arrow]
                hatch = msp.add_hatch(color=ACI_GREEN, dxfattribs={"layer": LAYER_ANNOT})
                hatch.paths.add_polyline_path(pts, is_closed=True)

        # Text
        msp.add_text(
            dim.label,
            height=3,
            rotation=dim.text_angle,
            dxfattribs={
                "layer": LAYER_ANNOT,
                "color": ACI_GREEN,
                "insert": (dim.text_pos[0], dim.text_pos[1]),
            }
        )

    def _draw_annotation_dxf(self, msp, ann):
        """Draw annotation label."""
        msp.add_text(
            ann.text,
            height=2.5,
            rotation=ann.angle,
            dxfattribs={
                "layer": LAYER_ANNOT,
                "color": ACI_GREEN,
                "insert": (ann.position[0], ann.position[1]),
            }
        )

    def _draw_legend_dxf(self, msp, bounds, scale):
        """Draw legend block in DXF."""
        bmin, bmax = bounds
        lx = -50
        ly = -50
        lw = 200
        lh = 40

        # Border
        msp.add_lwpolyline(
            [(lx, ly, 0), (lx+lw, ly, 0), (lx+lw, ly+lh, 0), (lx, ly+lh, 0), (lx, ly, 0)],
            dxfattribs={"layer": LAYER_ANNOT, "color": ACI_GREEN}
        )

        labels = [
            "Track ceiling spotlight",
            "Socket",
            "Led strip",
            "Lightbox",
            "Light inside",
            "Neon Sign",
        ]

        y_pos = ly + lh - 8
        x_pos = lx + 5
        for i, label in enumerate(labels):
            if i == 3:
                y_pos -= 14
                x_pos = lx + 5
            msp.add_text(
                label,
                height=2.5,
                dxfattribs={
                    "layer": LAYER_ANNOT,
                    "color": ACI_GREEN,
                    "insert": (x_pos, y_pos),
                }
            )
            x_pos += 65
