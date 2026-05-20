"""
glb_parser.py - Parse a GLB/glTF file and convert it into a Scene object.

Uses trimesh to load the GLB, extracts world-space bounding boxes for each
mesh instance, and classifies them into walls / furniture / lighting / assets
based on name patterns and geometric heuristics.

Coordinate Remapping
--------------------
glTF convention: Y-up, -Z forward.
Our Scene convention: Z-up (X horizontal, Y depth, Z height).
Mapping: scene.x = glb.x,  scene.y = -glb.z,  scene.z = glb.y
"""

import os
import re
import math
import numpy as np
import trimesh
from typing import List, Tuple

from scene_parser import (
    Scene, SceneMetadata, Vector3,
    Wall, Floor, Door,
    TrackSpotlight, LEDStrip, TrussSegment,
    Furniture, Asset,
)


# ---------- Coordinate remap ----------

def _remap(point: np.ndarray) -> Vector3:
    """glTF (Y-up, -Z forward) → Scene (Z-up, +Y depth)."""
    x, y, z = float(point[0]), float(point[1]), float(point[2])
    return Vector3(x, -z, y)


def _remap_aabb(bmin: np.ndarray, bmax: np.ndarray) -> Tuple[Vector3, Vector3]:
    """Remap an AABB and return (min, max) in scene coords."""
    # Remap each corner first
    corners_glb = [
        np.array([bmin[0], bmin[1], bmin[2]]),
        np.array([bmax[0], bmin[1], bmin[2]]),
        np.array([bmin[0], bmax[1], bmin[2]]),
        np.array([bmax[0], bmax[1], bmin[2]]),
        np.array([bmin[0], bmin[1], bmax[2]]),
        np.array([bmax[0], bmin[1], bmax[2]]),
        np.array([bmin[0], bmax[1], bmax[2]]),
        np.array([bmax[0], bmax[1], bmax[2]]),
    ]
    remapped = [_remap(c) for c in corners_glb]
    xs = [p.x for p in remapped]
    ys = [p.y for p in remapped]
    zs = [p.z for p in remapped]
    return (
        Vector3(min(xs), min(ys), min(zs)),
        Vector3(max(xs), max(ys), max(zs)),
    )


# ---------- Mesh instance extraction ----------

def _extract_instances(scene: trimesh.Scene):
    """
    Extract every mesh instance with its world-space transform applied.

    Returns: list of (name, world_bounds_min, world_bounds_max, world_centroid, mesh)
    """
    instances = []

    # trimesh Scene.dump() returns world-transformed meshes
    # but we also want the original node names. Use the graph.
    graph = scene.graph

    for node_name in graph.nodes_geometry:
        try:
            transform, geom_name = graph.get(node_name)
        except Exception:
            continue

        if geom_name is None or geom_name not in scene.geometry:
            continue

        mesh = scene.geometry[geom_name]
        verts = mesh.vertices

        if len(verts) == 0:
            continue

        # Apply world transform
        homog = np.ones((len(verts), 4))
        homog[:, :3] = verts
        world_verts = (transform @ homog.T).T[:, :3]

        bmin = world_verts.min(axis=0)
        bmax = world_verts.max(axis=0)
        centroid = (bmin + bmax) / 2.0

        instances.append({
            "node_name": node_name,
            "geom_name": geom_name,
            "bmin_glb": bmin,
            "bmax_glb": bmax,
            "centroid_glb": centroid,
            "vertex_count": len(verts),
        })

    return instances


# ---------- Classification heuristics ----------

def _classify(name_node: str, name_geom: str,
              bmin_s: Vector3, bmax_s: Vector3) -> str:
    """
    Classify a mesh instance. Name-based rules first, then size heuristics.
    """
    name = f"{name_node} {name_geom}".lower()

    size_x = bmax_s.x - bmin_s.x
    size_y = bmax_s.y - bmin_s.y
    size_z = bmax_s.z - bmin_s.z

    footprint = size_x * size_y
    max_dim = max(size_x, size_y, size_z)
    min_dim = min(size_x, size_y, size_z)

    # === Hard skips ===
    if footprint < 1e-6:
        return "skip"
    if max_dim > 15.0:
        return "skip"           # backdrop / skybox
    if "table_leg" in name or "table leg" in name:
        return "skip"

    # === Name-based classification (takes priority) ===
    if "floor" in name or "wood" in name:
        return "floor"
    if "table_top" in name or "table top" in name or "ingo_table" in name:
        return "table"
    if "bench" in name or "chair" in name or "ios bench" in name:
        return "chair"
    if any(k in name for k in ["light source", "spotlight", "pil_guell",
                                "pil guell", "ceiling lamp", "lamp"]):
        return "spotlight"
    if "omni" in name and size_z > 1.0:
        return "spotlight"      # vertical light fixture
    if "led" in name or "strip" in name:
        return "led_strip"
    if "truss" in name or "rig" in name:
        return "truss"
    if any(k in name for k in ["tv", "screen", "lightbox", "display", "lcd"]):
        return "lightbox"
    if "neon" in name or "sign" in name:
        return "neon_sign"
    if "socket" in name or "outlet" in name or "plug" in name:
        return "socket"
    if "door" in name:
        return "skip"

    # === Size-based heuristics for unnamed objects ===

    # Skip planar decals
    if min_dim < 0.02:
        return "skip"

    # Skip very tiny decorative objects
    if max_dim < 0.08:
        return "skip"

    # Floor (large flat)
    if size_z < 0.05 and footprint > 5.0:
        return "floor"

    # Walls: tall and thin
    if size_z > 1.5:
        thinness = (min(size_x, size_y) / max(size_x, size_y)
                    if max(size_x, size_y) > 0 else 1)
        if thinness < 0.15 and max(size_x, size_y) > 0.5:
            return "wall"
        if size_x < 0.3 and size_y < 0.3 and size_z > 2.0:
            return "wall"  # pillar

    # Skip small decorative items
    if max_dim < 0.4:
        return "skip"

    return "asset"


# ---------- Main converter ----------

def scene_from_glb(glb_path: str, units: str = "meters") -> Scene:
    """Load a GLB file and convert it into a Scene object."""
    print(f"  Loading GLB: {glb_path}")
    tm_scene = trimesh.load(glb_path, force="scene")

    if not isinstance(tm_scene, trimesh.Scene):
        # Single-mesh case: wrap in a Scene
        tm_scene = trimesh.Scene(tm_scene)

    instances = _extract_instances(tm_scene)
    print(f"  Extracted {len(instances)} mesh instances")

    walls: List[Wall] = []
    floors: List[Floor] = []
    doors: List[Door] = []
    spots: List[TrackSpotlight] = []
    leds: List[LEDStrip] = []
    trusses: List[TrussSegment] = []
    furniture: List[Furniture] = []
    assets: List[Asset] = []

    counts = {}

    for inst in instances:
        node = inst["node_name"]
        geom = inst["geom_name"]
        bmin_s, bmax_s = _remap_aabb(inst["bmin_glb"], inst["bmax_glb"])

        category = _classify(node, geom, bmin_s, bmax_s)
        counts[category] = counts.get(category, 0) + 1

        if category == "skip":
            continue

        center = Vector3(
            (bmin_s.x + bmax_s.x) / 2,
            (bmin_s.y + bmax_s.y) / 2,
            (bmin_s.z + bmax_s.z) / 2,
        )
        size = Vector3(
            bmax_s.x - bmin_s.x,
            bmax_s.y - bmin_s.y,
            bmax_s.z - bmin_s.z,
        )

        # Ensure non-zero size for rendering (planar decals get a min thickness)
        size = Vector3(
            max(size.x, 0.05),
            max(size.y, 0.05),
            max(size.z, 0.05),
        )

        # Use a friendly label from the node name
        label = _clean_label(node)

        if category == "floor":
            corners = [
                Vector3(bmin_s.x, bmin_s.y, bmin_s.z),
                Vector3(bmax_s.x, bmin_s.y, bmin_s.z),
                Vector3(bmax_s.x, bmax_s.y, bmin_s.z),
                Vector3(bmin_s.x, bmax_s.y, bmin_s.z),
            ]
            floors.append(Floor(
                id=f"floor_{len(floors)+1}",
                label=label or "Floor",
                corners=corners,
            ))

        elif category == "wall":
            # Approximate wall: long axis = start→end
            if size.x >= size.y:
                start = Vector3(bmin_s.x, center.y, bmin_s.z)
                end = Vector3(bmax_s.x, center.y, bmin_s.z)
                thickness = max(size.y, 0.05)
            else:
                start = Vector3(center.x, bmin_s.y, bmin_s.z)
                end = Vector3(center.x, bmax_s.y, bmin_s.z)
                thickness = max(size.x, 0.05)
            height = max(size.z, 0.5)
            walls.append(Wall(
                id=f"wall_{len(walls)+1}",
                label=label or f"Wall {len(walls)+1}",
                start=start,
                end=end,
                height=height,
                thickness=thickness,
            ))

        elif category == "table":
            furniture.append(Furniture(
                id=f"table_{len(furniture)+1}",
                type="table",
                label=label or "Table",
                position=Vector3(center.x, center.y, bmin_s.z),
                size=size,
                rotation=0,
            ))

        elif category == "chair":
            furniture.append(Furniture(
                id=f"chair_{len(furniture)+1}",
                type="chair",
                label=label or "Chair",
                position=Vector3(center.x, center.y, bmin_s.z),
                size=size,
                rotation=0,
            ))

        elif category == "spotlight":
            spots.append(TrackSpotlight(
                id=f"spot_{len(spots)+1}",
                label=label or "Spotlight",
                position=center,
                target=Vector3(center.x, center.y, 0),
                wattage=35,
            ))

        elif category == "led_strip":
            # Long axis determines start/end
            if size.x >= size.y:
                start = Vector3(bmin_s.x, center.y, center.z)
                end = Vector3(bmax_s.x, center.y, center.z)
            else:
                start = Vector3(center.x, bmin_s.y, center.z)
                end = Vector3(center.x, bmax_s.y, center.z)
            leds.append(LEDStrip(
                id=f"led_{len(leds)+1}",
                label=label or "LED Strip",
                start=start, end=end,
                color="warm_white",
            ))

        elif category == "truss":
            if size.x >= size.y:
                start = Vector3(bmin_s.x, center.y, center.z)
                end = Vector3(bmax_s.x, center.y, center.z)
            else:
                start = Vector3(center.x, bmin_s.y, center.z)
                end = Vector3(center.x, bmax_s.y, center.z)
            trusses.append(TrussSegment(
                id=f"truss_{len(trusses)+1}",
                label=label or "Truss",
                start=start, end=end,
                profile="box_300",
            ))

        elif category == "lightbox":
            assets.append(Asset(
                id=f"lightbox_{len([a for a in assets if a.type=='lightbox'])+1}",
                type="lightbox",
                label=label or "Lightbox",
                position=Vector3(center.x, center.y, bmin_s.z),
                size=size,
                rotation=0,
                content="",
            ))

        elif category == "neon_sign":
            assets.append(Asset(
                id=f"neon_{len([a for a in assets if a.type=='neon_sign'])+1}",
                type="neon_sign",
                label=label or "Neon Sign",
                position=center,
                size=size,
                rotation=0,
                content=label,
            ))

        elif category == "socket":
            assets.append(Asset(
                id=f"socket_{len([a for a in assets if a.type=='socket'])+1}",
                type="socket",
                label=label or "Socket",
                position=center,
                size=size,
                rotation=0,
            ))

        else:  # generic asset
            assets.append(Asset(
                id=f"asset_{len(assets)+1}",
                type="generic",
                label=label or geom[:30],
                position=Vector3(center.x, center.y, bmin_s.z),
                size=size,
                rotation=0,
            ))

    # If no floor was found, synthesize one from the overall bounds
    if not floors:
        bounds = tm_scene.bounds
        bmin_s, bmax_s = _remap_aabb(bounds[0], bounds[1])
        floors.append(Floor(
            id="floor_main",
            label="Main Floor",
            corners=[
                Vector3(bmin_s.x, bmin_s.y, 0),
                Vector3(bmax_s.x, bmin_s.y, 0),
                Vector3(bmax_s.x, bmax_s.y, 0),
                Vector3(bmin_s.x, bmax_s.y, 0),
            ],
        ))

    print(f"  Classification: {counts}")
    print(f"  Walls: {len(walls)}, Floors: {len(floors)}, Furniture: {len(furniture)}, "
          f"Spotlights: {len(spots)}, Assets: {len(assets)}")

    metadata = SceneMetadata(
        title=f"Exhibition Booth — {os.path.basename(glb_path)}",
        scale="1:50",
        units=units,
        author="3D-to-Technical Export (GLB import)",
    )

    return Scene(
        metadata=metadata,
        walls=walls,
        floors=floors,
        doors=doors,
        track_spotlights=spots,
        led_strips=leds,
        truss_segments=trusses,
        furniture=furniture,
        assets=assets,
    )


def _clean_label(name: str) -> str:
    """Turn a node/geometry name into a human-readable label."""
    if not name:
        return ""
    # Drop common suffixes
    s = re.sub(r"_\d+$", "", name)
    s = re.sub(r"_\[.*?\]", "", s)
    s = re.sub(r"_GR\d+", "", s)
    s = re.sub(r"_Material.*$", "", s)
    s = re.sub(r"_[\w\-]+_0$", "", s)
    # Replace separators
    s = s.replace("_", " ").replace(".", " ")
    s = re.sub(r"\s+", " ", s).strip()
    # Truncate
    if len(s) > 30:
        s = s[:27] + "..."
    return s
