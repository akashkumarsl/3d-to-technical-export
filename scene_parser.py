"""
scene_parser.py - Parse and validate 3D scene data from JSON input.

Converts raw JSON scene definitions into structured dataclasses for
use by the projection and rendering engines.
"""

import json
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class Vector3:
    x: float
    y: float
    z: float

    def to_tuple(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)

    def __sub__(self, other: "Vector3") -> "Vector3":
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __add__(self, other: "Vector3") -> "Vector3":
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)

    def length(self) -> float:
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)


@dataclass
class Wall:
    id: str
    label: str
    start: Vector3
    end: Vector3
    height: float
    thickness: float

    @property
    def length_2d(self) -> float:
        dx = self.end.x - self.start.x
        dy = self.end.y - self.start.y
        return math.sqrt(dx**2 + dy**2)


@dataclass
class Floor:
    id: str
    label: str
    corners: List[Vector3]


@dataclass
class Door:
    id: str
    label: str
    position: Vector3
    width: float
    height: float
    swing: str


@dataclass
class TrackSpotlight:
    id: str
    label: str
    position: Vector3
    target: Vector3
    wattage: float


@dataclass
class LEDStrip:
    id: str
    label: str
    start: Vector3
    end: Vector3
    color: str


@dataclass
class TrussSegment:
    id: str
    label: str
    start: Vector3
    end: Vector3
    profile: str


@dataclass
class Furniture:
    id: str
    type: str
    label: str
    position: Vector3
    size: Vector3
    rotation: float


@dataclass
class Asset:
    id: str
    type: str
    label: str
    position: Vector3
    size: Vector3
    rotation: float = 0.0
    content: str = ""


@dataclass
class SceneMetadata:
    title: str
    scale: str
    units: str
    author: str


@dataclass
class Scene:
    metadata: SceneMetadata
    walls: List[Wall]
    floors: List[Floor]
    doors: List[Door]
    track_spotlights: List[TrackSpotlight]
    led_strips: List[LEDStrip]
    truss_segments: List[TrussSegment]
    furniture: List[Furniture]
    assets: List[Asset]


def _parse_vec3(data) -> Vector3:
    return Vector3(float(data[0]), float(data[1]), float(data[2]))


def parse_scene(json_path: str) -> Scene:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    meta = data.get("metadata", {})
    metadata = SceneMetadata(
        title=meta.get("title", "Untitled"),
        scale=meta.get("scale", "1:50"),
        units=meta.get("units", "meters"),
        author=meta.get("author", ""),
    )

    walls = []
    for w in data.get("walls", []):
        walls.append(Wall(
            id=w["id"],
            label=w.get("label", w["id"]),
            start=_parse_vec3(w["start"]),
            end=_parse_vec3(w["end"]),
            height=float(w["height"]),
            thickness=float(w["thickness"]),
        ))

    floors = []
    for fl in data.get("floors", []):
        corners = [_parse_vec3(c) for c in fl["corners"]]
        floors.append(Floor(id=fl["id"], label=fl.get("label", fl["id"]), corners=corners))

    doors = []
    for d in data.get("doors", []):
        doors.append(Door(
            id=d["id"],
            label=d.get("label", d["id"]),
            position=_parse_vec3(d["position"]),
            width=float(d["width"]),
            height=float(d["height"]),
            swing=d.get("swing", "inward"),
        ))

    spots = []
    for s in data.get("lighting", {}).get("track_spotlights", []):
        spots.append(TrackSpotlight(
            id=s["id"],
            label=s.get("label", s["id"]),
            position=_parse_vec3(s["position"]),
            target=_parse_vec3(s["target"]),
            wattage=float(s.get("wattage", 0)),
        ))

    leds = []
    for l in data.get("lighting", {}).get("led_strips", []):
        leds.append(LEDStrip(
            id=l["id"],
            label=l.get("label", l["id"]),
            start=_parse_vec3(l["start"]),
            end=_parse_vec3(l["end"]),
            color=l.get("color", "white"),
        ))

    trusses = []
    for t in data.get("rigging", {}).get("truss_segments", []):
        trusses.append(TrussSegment(
            id=t["id"],
            label=t.get("label", t["id"]),
            start=_parse_vec3(t["start"]),
            end=_parse_vec3(t["end"]),
            profile=t.get("profile", "box_300"),
        ))

    furniture = []
    for f in data.get("furniture", []):
        furniture.append(Furniture(
            id=f["id"],
            type=f["type"],
            label=f.get("label", f["id"]),
            position=_parse_vec3(f["position"]),
            size=_parse_vec3(f["size"]),
            rotation=float(f.get("rotation", 0)),
        ))

    assets = []
    for a in data.get("assets", []):
        assets.append(Asset(
            id=a["id"],
            type=a["type"],
            label=a.get("label", a["id"]),
            position=_parse_vec3(a["position"]),
            size=_parse_vec3(a["size"]),
            rotation=float(a.get("rotation", 0)),
            content=a.get("content", ""),
        ))

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


def meters_to_feet(value_m: float) -> float:
    return value_m * 3.28084


def feet_to_meters(value_ft: float) -> float:
    return value_ft / 3.28084


def format_dimension(value_m: float, units: str = "meters") -> str:
    if units == "feet":
        ft = meters_to_feet(value_m)
        return f"{ft:.1f}'"
    else:
        cm = value_m * 100
        if cm >= 100:
            return f"{value_m:.2f}m"
        return f"{cm:.1f}cm"
