"""
renderer_pdf.py - PDF rendering engine with layered Optional Content Groups.

Generates multi-page, multi-layer technical PDFs with:
- Layer 1: Architecture (walls, floors, doors)
- Layer 2: Lighting & Rigging (truss, spotlights, LED strips)
- Layer 3: Annotations (dimensions, labels, legend)
- Title block with metadata
"""

import math
from reportlab.lib.pagesizes import A3, A4, landscape
from reportlab.lib.units import mm, cm
from reportlab.lib.colors import Color, black, white, gray, lightgrey
from reportlab.pdfgen import canvas as pdf_canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.rl_accel import fp_str

from scene_parser import Scene, Vector3, format_dimension
from projection import ProjectionEngine, compute_scene_bounds
from dimensioning import (
    DimensioningEngine, DimensionLine, AnnotationLabel,
    compute_wall_dimensions, compute_wall_height_dimensions,
    compute_asset_dimensions, compute_annotations,
)


class _OCGHelper:
    """Manages PDF Optional Content Groups (layers) via low-level PDF ops."""

    def __init__(self, canvas):
        from reportlab.pdfbase.pdfdoc import PDFDictionary, PDFArray, PDFString, PDFName
        self._canvas = canvas
        self._ocgs = []
        self._ocg_refs = {}
        self._PDFDictionary = PDFDictionary
        self._PDFArray = PDFArray
        self._PDFString = PDFString
        self._PDFName = PDFName

    def add_ocg(self, name: str, visible: bool = True):
        """Register an OCG and return its internal name."""
        doc = self._canvas._doc
        ocg_dict = self._PDFDictionary({
            "Type": self._PDFName("OCG"),
            "Name": self._PDFString(name),
        })
        ref = doc.Reference(ocg_dict)
        internal_name = f"oc{len(self._ocgs)}"
        self._ocgs.append((name, ref, visible, internal_name))
        self._ocg_refs[name] = (ref, internal_name)
        return name

    def begin_ocg(self, name: str):
        """Mark the start of content belonging to this OCG."""
        ref, internal_name = self._ocg_refs[name]
        if not hasattr(self._canvas, '_ocg_page_resources'):
            self._canvas._ocg_page_resources = {}
        self._canvas._ocg_page_resources[internal_name] = ref
        self._canvas._code.append(f"/{internal_name} BDC")

    def end_ocg(self):
        """Mark the end of OCG content."""
        self._canvas._code.append("EMC")

    def finalize(self):
        """Write OCG catalog entries after all pages are rendered."""
        doc = self._canvas._doc
        if not self._ocgs:
            return

        ocg_refs = [entry[1] for entry in self._ocgs]
        on_refs = [entry[1] for entry in self._ocgs if entry[2]]

        ocg_array = self._PDFArray(ocg_refs)
        on_array = self._PDFArray(on_refs)

        d_dict = self._PDFDictionary({
            "ON": on_array,
            "OFF": self._PDFArray([]),
            "BaseState": self._PDFName("ON"),
            "Order": ocg_array,
        })
        ocprops = self._PDFDictionary({
            "OCGs": ocg_array,
            "D": d_dict,
        })

        cat = doc.Catalog
        # Add OCProperties to the NoDefault list so it gets serialized
        if "OCProperties" not in cat.__NoDefault__:
            cat.__NoDefault__ = list(cat.__NoDefault__) + ["OCProperties"]
        cat.OCProperties = ocprops

    def inject_page_resources(self):
        """Inject OCG references into current page's Properties resource dict."""
        if not hasattr(self._canvas, '_ocg_page_resources'):
            return
        props = self._canvas._ocg_page_resources
        if not props:
            return
        props_dict = self._PDFDictionary(props)
        # Inject into page extra dict for properties resources
        page = self._canvas._doc.thisPageRef()
        if hasattr(page, 'Properties'):
            page.Properties = props_dict
        self._canvas._ocg_page_resources = {}


# Drafting colors
COLOR_WALL = Color(0, 0, 0, 1)
COLOR_WALL_FILL = Color(0.85, 0.85, 0.85, 1)
COLOR_FLOOR = Color(0.92, 0.92, 0.92, 1)
COLOR_DOOR = Color(0.3, 0.3, 0.3, 1)
COLOR_FURNITURE = Color(0.4, 0.4, 0.4, 1)
COLOR_LIGHTBOX = Color(0.5, 0.5, 0.5, 1)
COLOR_LIGHTBOX_HATCH = Color(0.6, 0.6, 0.6, 1)
COLOR_TRUSS = Color(0.2, 0.2, 0.6, 1)
COLOR_SPOTLIGHT = Color(0.8, 0.6, 0.0, 1)
COLOR_LED = Color(0.9, 0.8, 0.2, 1)
COLOR_DIM = Color(0.15, 0.15, 0.15, 1)
COLOR_LABEL = Color(0.1, 0.1, 0.1, 1)
COLOR_TITLE = Color(0, 0, 0, 1)
COLOR_SOCKET = Color(0.3, 0.6, 0.3, 1)
COLOR_NEON = Color(0.7, 0.2, 0.2, 1)

# Line weights (in points)
LW_WALL = 1.5
LW_WALL_THIN = 0.5
LW_FURNITURE = 0.8
LW_TRUSS = 1.2
LW_DIM = 0.3
LW_EXT = 0.2
LW_BORDER = 2.0


class PDFRenderer:
    """Renders a Scene to a multi-layered technical PDF."""

    def __init__(self, output_path: str, page_size=None, units: str = "meters"):
        self.output_path = output_path
        self.page_size = page_size or landscape(A3)
        self.units = units
        self.page_w = self.page_size[0]
        self.page_h = self.page_size[1]

    def render(self, scene: Scene, views: list = None):
        """Render the full technical drawing."""
        if views is None:
            views = ["top", "front", "iso"]

        c = pdf_canvas.Canvas(self.output_path, pagesize=self.page_size)

        # Create Optional Content Groups (layers) via helper
        ocg = _OCGHelper(c)
        layer_arch = ocg.add_ocg("Layer 1 - Architecture", visible=True)
        layer_lighting = ocg.add_ocg("Layer 2 - Lighting & Rigging", visible=True)
        layer_annot = ocg.add_ocg("Layer 3 - Annotations", visible=True)

        for view in views:
            self._render_page(c, scene, view, ocg, layer_arch, layer_lighting, layer_annot)
            c.showPage()

        ocg.finalize()
        c.save()

    def _render_page(self, c, scene: Scene, view: str,
                     ocg, layer_arch, layer_lighting, layer_annot):
        """Render one page (one view) of the technical drawing."""
        # Drawing area (inside title block)
        margin = 25 * mm
        title_block_h = 20 * mm
        draw_w = self.page_w - 2 * margin
        draw_h = self.page_h - 2 * margin - title_block_h

        # Set up projection
        bounds = compute_scene_bounds(scene)
        projector = ProjectionEngine(
            view=view,
            canvas_width=draw_w,
            canvas_height=draw_h,
            margin=40,
        )
        projector.set_auto_fit(bounds)

        # Translate canvas so drawing area starts at margin
        c.saveState()
        c.translate(margin, margin + title_block_h)

        # --- Layer 1: Architecture ---
        ocg.begin_ocg(layer_arch)
        self._draw_floors(c, scene, projector)
        self._draw_walls(c, scene, projector, view)
        self._draw_doors(c, scene, projector, view)
        self._draw_furniture(c, scene, projector, view)
        self._draw_assets(c, scene, projector, view)
        ocg.end_ocg()

        # --- Layer 2: Lighting & Rigging ---
        ocg.begin_ocg(layer_lighting)
        self._draw_truss(c, scene, projector)
        self._draw_spotlights(c, scene, projector)
        self._draw_led_strips(c, scene, projector)
        ocg.end_ocg()

        # --- Layer 3: Annotations ---
        ocg.begin_ocg(layer_annot)
        dim_engine = DimensioningEngine(offset=18, arrow_size=5, font_size=7)

        # Dimension lines
        if view == "top":
            dims = compute_wall_dimensions(scene, projector, dim_engine, self.units)
            dims += compute_asset_dimensions(scene, projector, dim_engine, self.units)
        elif view == "front":
            dims = compute_wall_dimensions(scene, projector, dim_engine, self.units)
            dims += compute_wall_height_dimensions(scene, projector, dim_engine, self.units)
        else:
            dims = compute_wall_dimensions(scene, projector, dim_engine, self.units)

        for dim in dims:
            self._draw_dimension(c, dim)

        # Text annotations
        annotations = compute_annotations(scene, projector, dim_engine)
        for ann in annotations:
            self._draw_annotation(c, ann)

        # Legend
        self._draw_legend(c, draw_w, draw_h)
        ocg.end_ocg()

        c.restoreState()

        # Inject OCG page resources
        ocg.inject_page_resources()

        # Title block (outside drawing area)
        self._draw_title_block(c, scene, view, margin, title_block_h)

        # Border
        c.setStrokeColor(black)
        c.setLineWidth(LW_BORDER)
        c.rect(margin, margin, self.page_w - 2*margin, self.page_h - 2*margin)

    # ---------------------------------------------------------------
    # Layer 1: Architecture Drawing Methods
    # ---------------------------------------------------------------

    def _draw_floors(self, c, scene: Scene, proj: ProjectionEngine):
        for floor in scene.floors:
            pts = [proj.project(corner) for corner in floor.corners]
            if len(pts) < 3:
                continue
            path = c.beginPath()
            path.moveTo(pts[0][0], pts[0][1])
            for pt in pts[1:]:
                path.lineTo(pt[0], pt[1])
            path.close()
            c.setFillColor(COLOR_FLOOR)
            c.setStrokeColor(COLOR_WALL)
            c.setLineWidth(LW_WALL_THIN)
            c.drawPath(path, stroke=1, fill=1)

    def _draw_walls(self, c, scene: Scene, proj: ProjectionEngine, view: str):
        for wall in scene.walls:
            if view == "top":
                self._draw_wall_top(c, wall, proj)
            elif view == "front":
                self._draw_wall_front(c, wall, proj)
            else:
                self._draw_wall_iso(c, wall, proj)

    def _draw_wall_top(self, c, wall, proj: ProjectionEngine):
        """Draw wall as a thick line with thickness in top view."""
        dx = wall.end.x - wall.start.x
        dy = wall.end.y - wall.start.y
        length = math.sqrt(dx**2 + dy**2)
        if length < 1e-6:
            return

        nx = -dy / length * wall.thickness / 2
        ny = dx / length * wall.thickness / 2

        corners = [
            proj.project(Vector3(wall.start.x + nx, wall.start.y + ny, 0)),
            proj.project(Vector3(wall.end.x + nx, wall.end.y + ny, 0)),
            proj.project(Vector3(wall.end.x - nx, wall.end.y - ny, 0)),
            proj.project(Vector3(wall.start.x - nx, wall.start.y - ny, 0)),
        ]

        path = c.beginPath()
        path.moveTo(corners[0][0], corners[0][1])
        for pt in corners[1:]:
            path.lineTo(pt[0], pt[1])
        path.close()
        c.setFillColor(COLOR_WALL_FILL)
        c.setStrokeColor(COLOR_WALL)
        c.setLineWidth(LW_WALL)
        c.drawPath(path, stroke=1, fill=1)

    def _draw_wall_front(self, c, wall, proj: ProjectionEngine):
        """Draw wall as a rectangle in front elevation."""
        p1 = proj.project(wall.start)
        p2 = proj.project(wall.end)
        p3 = proj.project(Vector3(wall.end.x, wall.end.y, wall.height))
        p4 = proj.project(Vector3(wall.start.x, wall.start.y, wall.height))

        path = c.beginPath()
        path.moveTo(p1[0], p1[1])
        path.lineTo(p2[0], p2[1])
        path.lineTo(p3[0], p3[1])
        path.lineTo(p4[0], p4[1])
        path.close()
        c.setFillColor(COLOR_WALL_FILL)
        c.setStrokeColor(COLOR_WALL)
        c.setLineWidth(LW_WALL)
        c.drawPath(path, stroke=1, fill=1)

    def _draw_wall_iso(self, c, wall, proj: ProjectionEngine):
        """Draw wall as 3D box in isometric view."""
        s, e = wall.start, wall.end
        h = wall.height

        # Front face
        pts_front = [
            proj.project(s),
            proj.project(e),
            proj.project(Vector3(e.x, e.y, h)),
            proj.project(Vector3(s.x, s.y, h)),
        ]

        path = c.beginPath()
        path.moveTo(pts_front[0][0], pts_front[0][1])
        for pt in pts_front[1:]:
            path.lineTo(pt[0], pt[1])
        path.close()
        c.setFillColor(COLOR_WALL_FILL)
        c.setStrokeColor(COLOR_WALL)
        c.setLineWidth(LW_WALL)
        c.drawPath(path, stroke=1, fill=1)

    def _draw_doors(self, c, scene: Scene, proj: ProjectionEngine, view: str):
        for door in scene.doors:
            if view == "top":
                self._draw_door_top(c, door, proj)
            elif view == "front":
                self._draw_door_front(c, door, proj)

    def _draw_door_top(self, c, door, proj: ProjectionEngine):
        """Draw door with swing arc in top view."""
        pos = door.position
        hw = door.width / 2

        # Door opening (gap in wall)
        left = proj.project(Vector3(pos.x - hw, pos.y, pos.z))
        right = proj.project(Vector3(pos.x + hw, pos.y, pos.z))

        c.setStrokeColor(COLOR_DOOR)
        c.setLineWidth(LW_FURNITURE)

        # Door leaf
        c.line(left[0], left[1], right[0], right[1])

        # Swing arc
        center = left
        radius = abs(right[0] - left[0])
        if radius > 0:
            c.setDash([2, 2])
            c.arc(center[0] - radius, center[1] - radius,
                  center[0] + radius, center[1] + radius,
                  startAng=0, extent=90)
            c.setDash([])

    def _draw_door_front(self, c, door, proj: ProjectionEngine):
        """Draw door in front elevation."""
        pos = door.position
        hw = door.width / 2

        corners = [
            proj.project(Vector3(pos.x - hw, pos.y, 0)),
            proj.project(Vector3(pos.x + hw, pos.y, 0)),
            proj.project(Vector3(pos.x + hw, pos.y, door.height)),
            proj.project(Vector3(pos.x - hw, pos.y, door.height)),
        ]

        path = c.beginPath()
        path.moveTo(corners[0][0], corners[0][1])
        for pt in corners[1:]:
            path.lineTo(pt[0], pt[1])
        path.close()
        c.setStrokeColor(COLOR_DOOR)
        c.setLineWidth(LW_FURNITURE)
        c.drawPath(path, stroke=1, fill=0)

    def _draw_furniture(self, c, scene: Scene, proj: ProjectionEngine, view: str):
        for item in scene.furniture:
            pts = proj.project_rect(item.position, item.size, item.rotation)
            if len(pts) < 3:
                continue

            path = c.beginPath()
            path.moveTo(pts[0][0], pts[0][1])
            for pt in pts[1:]:
                path.lineTo(pt[0], pt[1])
            path.close()

            c.setStrokeColor(COLOR_FURNITURE)
            c.setLineWidth(LW_FURNITURE)
            if item.type == "table":
                c.setFillColor(Color(0.9, 0.88, 0.82, 1))
            else:
                c.setFillColor(Color(0.8, 0.8, 0.8, 1))
            c.drawPath(path, stroke=1, fill=1)

            # Chair circle marker
            if item.type == "chair":
                center = proj.project(item.position)
                r = min(abs(pts[1][0] - pts[0][0]), abs(pts[2][1] - pts[1][1])) * 0.3
                if r > 1:
                    c.setFillColor(COLOR_FURNITURE)
                    c.circle(center[0], center[1], r, stroke=0, fill=1)

    def _draw_assets(self, c, scene: Scene, proj: ProjectionEngine, view: str):
        for asset in scene.assets:
            pts = proj.project_rect(asset.position, asset.size, asset.rotation)
            if len(pts) < 3:
                continue

            if asset.type == "lightbox":
                self._draw_lightbox(c, pts)
            elif asset.type == "neon_sign":
                self._draw_neon_sign(c, pts, asset)
            elif asset.type == "socket":
                center = proj.project(asset.position)
                c.setStrokeColor(COLOR_SOCKET)
                c.setLineWidth(0.5)
                c.circle(center[0], center[1], 3, stroke=1, fill=0)
                # Small dot
                c.setFillColor(COLOR_SOCKET)
                c.circle(center[0], center[1], 1, stroke=0, fill=1)
            else:
                path = c.beginPath()
                path.moveTo(pts[0][0], pts[0][1])
                for pt in pts[1:]:
                    path.lineTo(pt[0], pt[1])
                path.close()
                c.setStrokeColor(COLOR_FURNITURE)
                c.setLineWidth(LW_FURNITURE)
                c.setFillColor(Color(0.95, 0.95, 0.95, 1))
                c.drawPath(path, stroke=1, fill=1)

    def _draw_lightbox(self, c, pts):
        """Draw lightbox with hatching pattern."""
        path = c.beginPath()
        path.moveTo(pts[0][0], pts[0][1])
        for pt in pts[1:]:
            path.lineTo(pt[0], pt[1])
        path.close()
        c.setFillColor(Color(0.95, 0.95, 0.95, 1))
        c.setStrokeColor(COLOR_LIGHTBOX)
        c.setLineWidth(LW_FURNITURE)
        c.drawPath(path, stroke=1, fill=1)

        # Diagonal hatching
        c.setStrokeColor(COLOR_LIGHTBOX_HATCH)
        c.setLineWidth(0.2)
        min_x = min(p[0] for p in pts)
        max_x = max(p[0] for p in pts)
        min_y = min(p[1] for p in pts)
        max_y = max(p[1] for p in pts)
        step = 4
        x = min_x
        while x < max_x:
            c.line(x, min_y, x + (max_y - min_y), max_y)
            x += step

    def _draw_neon_sign(self, c, pts, asset):
        """Draw neon sign with distinctive marker."""
        center_x = sum(p[0] for p in pts) / len(pts)
        center_y = sum(p[1] for p in pts) / len(pts)

        c.setStrokeColor(COLOR_NEON)
        c.setLineWidth(1.0)
        # Vertical line with S symbol
        c.line(center_x, center_y - 8, center_x, center_y + 8)
        c.setFont("Helvetica-Bold", 8)
        c.setFillColor(COLOR_NEON)
        c.drawCentredString(center_x, center_y - 3, "S")

    # ---------------------------------------------------------------
    # Layer 2: Lighting & Rigging
    # ---------------------------------------------------------------

    def _draw_truss(self, c, scene: Scene, proj: ProjectionEngine):
        for truss in scene.truss_segments:
            p1 = proj.project(truss.start)
            p2 = proj.project(truss.end)

            # Double line for truss
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            length = math.sqrt(dx**2 + dy**2)
            if length < 1e-6:
                continue
            nx = -dy / length * 3
            ny = dx / length * 3

            c.setStrokeColor(COLOR_TRUSS)
            c.setLineWidth(LW_TRUSS)
            c.line(p1[0] + nx, p1[1] + ny, p2[0] + nx, p2[1] + ny)
            c.line(p1[0] - nx, p1[1] - ny, p2[0] - nx, p2[1] - ny)

            # Cross bracing
            c.setLineWidth(0.3)
            segments = max(int(length / 20), 2)
            for i in range(segments):
                t1 = i / segments
                t2 = (i + 1) / segments
                x1 = p1[0] + dx * t1
                y1 = p1[1] + dy * t1
                x2 = p1[0] + dx * t2
                y2 = p1[1] + dy * t2
                if i % 2 == 0:
                    c.line(x1 + nx, y1 + ny, x2 - nx, y2 - ny)
                else:
                    c.line(x1 - nx, y1 - ny, x2 + nx, y2 + ny)

    def _draw_spotlights(self, c, scene: Scene, proj: ProjectionEngine):
        for spot in scene.track_spotlights:
            center = proj.project(spot.position)

            # Spotlight symbol: circle with cross
            r = 5
            c.setStrokeColor(COLOR_SPOTLIGHT)
            c.setLineWidth(0.8)
            c.circle(center[0], center[1], r, stroke=1, fill=0)
            # Cross hair
            c.line(center[0] - r, center[1], center[0] + r, center[1])
            c.line(center[0], center[1] - r, center[0], center[1] + r)
            # Small filled center
            c.setFillColor(COLOR_SPOTLIGHT)
            c.circle(center[0], center[1], 1.5, stroke=0, fill=1)

    def _draw_led_strips(self, c, scene: Scene, proj: ProjectionEngine):
        for led in scene.led_strips:
            p1 = proj.project(led.start)
            p2 = proj.project(led.end)

            c.setStrokeColor(COLOR_LED)
            c.setLineWidth(1.5)
            c.setDash([6, 3])
            c.line(p1[0], p1[1], p2[0], p2[1])
            c.setDash([])

            # Small dots along strip
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            length = math.sqrt(dx**2 + dy**2)
            n_dots = max(int(length / 12), 2)
            c.setFillColor(COLOR_LED)
            for i in range(n_dots + 1):
                t = i / n_dots
                x = p1[0] + dx * t
                y = p1[1] + dy * t
                c.circle(x, y, 1, stroke=0, fill=1)

    # ---------------------------------------------------------------
    # Layer 3: Annotations
    # ---------------------------------------------------------------

    def _draw_dimension(self, c, dim: DimensionLine):
        """Draw a complete dimension line with extensions and arrowheads."""
        c.setStrokeColor(COLOR_DIM)
        c.setFillColor(COLOR_DIM)

        # Extension lines
        c.setLineWidth(LW_EXT)
        c.line(dim.ext1_start[0], dim.ext1_start[1],
               dim.ext1_end[0], dim.ext1_end[1])
        c.line(dim.ext2_start[0], dim.ext2_start[1],
               dim.ext2_end[0], dim.ext2_end[1])

        # Dimension line
        c.setLineWidth(LW_DIM)
        c.line(dim.dim_start[0], dim.dim_start[1],
               dim.dim_end[0], dim.dim_end[1])

        # Arrowheads
        for arrow in [dim.arrow1, dim.arrow2]:
            if len(arrow) >= 3:
                path = c.beginPath()
                path.moveTo(arrow[0][0], arrow[0][1])
                path.lineTo(arrow[1][0], arrow[1][1])
                path.lineTo(arrow[2][0], arrow[2][1])
                path.close()
                c.drawPath(path, stroke=0, fill=1)

        # Text label
        c.saveState()
        c.translate(dim.text_pos[0], dim.text_pos[1])
        c.rotate(dim.text_angle)
        c.setFont("Helvetica", 7)
        c.setFillColor(COLOR_DIM)
        c.drawCentredString(0, 0, dim.label)
        c.restoreState()

    def _draw_annotation(self, c, ann: AnnotationLabel):
        """Draw a text annotation."""
        c.saveState()
        c.translate(ann.position[0], ann.position[1])
        c.rotate(ann.angle)
        c.setFont("Helvetica", ann.font_size)
        c.setFillColor(COLOR_LABEL)
        c.drawCentredString(0, 0, ann.text)
        c.restoreState()

    def _draw_legend(self, c, draw_w: float, draw_h: float):
        """Draw legend box in bottom-left corner."""
        lx = 10
        ly = 10
        lw = 280
        lh = 55
        padding = 6

        # Background
        c.setFillColor(white)
        c.setStrokeColor(black)
        c.setLineWidth(0.5)
        c.rect(lx, ly, lw, lh, stroke=1, fill=1)

        c.setFont("Helvetica-Bold", 7)
        c.setFillColor(black)
        y = ly + lh - padding - 6

        # Row 1
        items_row1 = [
            ("Track ceiling spotlight", self._legend_spotlight),
            ("Led strip", self._legend_led),
            ("Light inside", self._legend_light_inside),
        ]
        x = lx + padding
        for label, draw_fn in items_row1:
            draw_fn(c, x, y)
            c.setFont("Helvetica", 6)
            c.drawString(x + 18, y - 2, label)
            x += 90

        # Row 2
        y -= 18
        items_row2 = [
            ("Socket", self._legend_socket),
            ("Lightbox", self._legend_lightbox),
            ("Neon Sign", self._legend_neon),
        ]
        x = lx + padding
        for label, draw_fn in items_row2:
            draw_fn(c, x, y)
            c.setFont("Helvetica", 6)
            c.drawString(x + 18, y - 2, label)
            x += 90

    def _legend_spotlight(self, c, x, y):
        c.setStrokeColor(COLOR_SPOTLIGHT)
        c.setLineWidth(0.6)
        c.circle(x + 6, y, 4, stroke=1, fill=0)
        c.line(x + 2, y, x + 10, y)
        c.line(x + 6, y - 4, x + 6, y + 4)

    def _legend_led(self, c, x, y):
        c.setStrokeColor(COLOR_LED)
        c.setLineWidth(1.0)
        c.setDash([4, 2])
        c.line(x, y, x + 14, y)
        c.setDash([])
        c.setFillColor(COLOR_LED)
        c.circle(x + 3, y, 1, stroke=0, fill=1)
        c.circle(x + 7, y, 1, stroke=0, fill=1)
        c.circle(x + 11, y, 1, stroke=0, fill=1)

    def _legend_light_inside(self, c, x, y):
        c.setStrokeColor(COLOR_SPOTLIGHT)
        c.setLineWidth(0.6)
        c.circle(x + 6, y, 4, stroke=1, fill=0)
        # Rays
        for angle in range(0, 360, 45):
            rad = math.radians(angle)
            c.line(x + 6 + 4 * math.cos(rad), y + 4 * math.sin(rad),
                   x + 6 + 6 * math.cos(rad), y + 6 * math.sin(rad))

    def _legend_socket(self, c, x, y):
        c.setStrokeColor(COLOR_SOCKET)
        c.setLineWidth(0.5)
        c.circle(x + 6, y, 3, stroke=1, fill=0)
        c.setFillColor(COLOR_SOCKET)
        c.circle(x + 6, y, 1, stroke=0, fill=1)

    def _legend_lightbox(self, c, x, y):
        c.setFillColor(Color(0.95, 0.95, 0.95, 1))
        c.setStrokeColor(COLOR_LIGHTBOX)
        c.setLineWidth(0.5)
        c.rect(x, y - 4, 14, 8, stroke=1, fill=1)
        # Hatching
        c.setStrokeColor(COLOR_LIGHTBOX_HATCH)
        c.setLineWidth(0.2)
        for i in range(4):
            c.line(x + i * 4, y - 4, x + i * 4 + 8, y + 4)

    def _legend_neon(self, c, x, y):
        c.setStrokeColor(COLOR_NEON)
        c.setLineWidth(0.8)
        c.line(x + 6, y - 6, x + 6, y + 6)
        c.setFont("Helvetica-Bold", 7)
        c.setFillColor(COLOR_NEON)
        c.drawCentredString(x + 6, y - 3, "S")

    def _draw_title_block(self, c, scene: Scene, view: str,
                          margin: float, tb_height: float):
        """Draw the title block at the bottom of the page."""
        x = margin
        y = margin
        w = self.page_w - 2 * margin
        h = tb_height

        c.setStrokeColor(black)
        c.setLineWidth(1.0)
        c.rect(x, y, w, h)

        # Dividers
        c.line(x + w * 0.5, y, x + w * 0.5, y + h)
        c.line(x + w * 0.75, y, x + w * 0.75, y + h)

        # Title
        c.setFont("Helvetica-Bold", 10)
        c.setFillColor(COLOR_TITLE)
        c.drawString(x + 8, y + h - 14, scene.metadata.title)

        # View name
        view_names = {"top": "Top-Down Plan View", "front": "Front Elevation",
                      "iso": "Isometric View"}
        c.setFont("Helvetica", 8)
        c.drawString(x + 8, y + 4, view_names.get(view, view))

        # Scale
        c.setFont("Helvetica", 9)
        c.drawString(x + w * 0.5 + 8, y + h - 14, f"Scale: {scene.metadata.scale}")
        c.drawString(x + w * 0.5 + 8, y + 4, f"Units: {scene.metadata.units}")

        # Page size label
        c.setFont("Helvetica", 12)
        view_letter = {"top": "A3", "front": "A3", "iso": "A3"}
        c.drawString(x + w * 0.75 + 8, y + h - 14, view_letter.get(view, "A3"))

        # Scale ratio
        c.setFont("Helvetica", 14)
        c.drawRightString(x + w - 8, y + 4, scene.metadata.scale)
