"""
Smart Semsar - Furniture catalog + staging schema (staging.json)
العقد الموحد للفرش: الكتالوج (القطع المتاحة) والـ staging (مكان كل قطعة في المخطط).

Rules:
  - A staging belongs to one plan.json and never changes it; one plan can have
    several stagings (different styles).
  - Items use the plan's frame: meters, per floor, x to the right, y up.
  - (x, y) is the centre of the item's footprint.
  - rotation_deg is counter-clockwise. At 0 the width runs along x, the back
    edge is on the +y side and the item faces -y.
  - Sizes are real-world sizes from the catalog; they are NOT scaled with a
    look-alike plan.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field

from backend.schema.plan import Point, RoomType

SCHEMA_VERSION = 1

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CATALOG = REPO_ROOT / "data" / "catalog" / "furniture.json"


# ---------- catalog ----------
class CatalogItem(BaseModel):
    id: str                                 # e.g. "bed_double"
    kind: str = ""                          # what a user would call it: bed / sofa / tv / desk ...
    category: str                           # bed / sofa / table / storage ...
    name: str
    width_m: float                          # along the back edge
    depth_m: float                          # back to front
    height_m: float
    room_types: list[RoomType]              # rooms this item may go in
    against_wall: bool = True               # back edge sits on a wall
    clearance_front_m: float = 0.0          # free strip needed in front (to use / walk past it)
    styles: list[str] = Field(default_factory=list)   # empty = fits any style
    color: tuple[int, int, int] = (170, 160, 150)
    mesh: Optional[str] = None              # ready-to-place GLB from repo root; None or missing -> box

    def is_a(self, names) -> bool:
        """True if any of the names is this item's id, kind or category."""
        return bool({self.id, self.kind, self.category} & set(names))


class Catalog(BaseModel):
    schema_version: Literal[1] = SCHEMA_VERSION
    items: list[CatalogItem]

    def get(self, item_id: str) -> CatalogItem:
        for item in self.items:
            if item.id == item_id:
                return item
        raise KeyError(f"Unknown catalog item: {item_id}")

    def vocabulary(self) -> set[str]:
        """Every name preferences may use in must_have / exclude (ids, kinds, categories)."""
        return {n for i in self.items for n in (i.id, i.kind, i.category) if n}

    def for_room(self, room_type: str, style: Optional[str] = None) -> list[CatalogItem]:
        return [i for i in self.items if room_type in i.room_types
                and (style is None or not i.styles or style in i.styles)]


def load_catalog(path: str | Path = DEFAULT_CATALOG) -> Catalog:
    return Catalog.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))


# ---------- staging ----------
class PlacedItem(BaseModel):
    id: str                                 # unique inside the staging
    catalog_id: str
    room_id: str                            # Room.id in the plan
    level: int                              # Floor.level in the plan
    x: float
    y: float
    rotation_deg: float = 0.0
    width_m: float                          # copied from the catalog so the file stands alone
    depth_m: float
    height_m: float

    @classmethod
    def from_catalog(cls, item: CatalogItem, id: str, room_id: str, level: int,
                     x: float, y: float, rotation_deg: float = 0.0) -> "PlacedItem":
        return cls(id=id, catalog_id=item.id, room_id=room_id, level=level, x=x, y=y,
                   rotation_deg=rotation_deg, width_m=item.width_m, depth_m=item.depth_m,
                   height_m=item.height_m)

    def to_plan(self, lx: float, ly: float) -> Point:
        """Item-local point (x along width, +y toward the back) -> plan meters."""
        a = math.radians(self.rotation_deg)
        return (round(self.x + lx * math.cos(a) - ly * math.sin(a), 4),
                round(self.y + lx * math.sin(a) + ly * math.cos(a), 4))

    def footprint(self) -> list[Point]:
        """Rotated rectangle on the floor, starting at the front-left corner."""
        w, d = self.width_m / 2, self.depth_m / 2
        return [self.to_plan(-w, -d), self.to_plan(w, -d), self.to_plan(w, d), self.to_plan(-w, d)]

    def front_zone(self, clearance_m: float) -> list[Point]:
        """Strip in front of the item that must stay free (empty list if no clearance)."""
        if clearance_m <= 0:
            return []
        w, d = self.width_m / 2, self.depth_m / 2
        return [self.to_plan(-w, -d - clearance_m), self.to_plan(w, -d - clearance_m),
                self.to_plan(w, -d), self.to_plan(-w, -d)]


class Staging(BaseModel):
    schema_version: Literal[1] = SCHEMA_VERSION
    staging_id: str
    plan_id: str                            # Plan.plan_id this staging was built for
    style: Optional[str] = None
    preferences: dict = Field(default_factory=dict)
    items: list[PlacedItem] = Field(default_factory=list)
    unplaced: list[str] = Field(default_factory=list)   # requested (must_have / dining size) but did not fit

    def in_room(self, room_id: str) -> list[PlacedItem]:
        return [i for i in self.items if i.room_id == room_id]


# ---------- load / save ----------
def save_staging(staging: Staging, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(staging.model_dump_json(indent=1), encoding="utf-8")
    return path


def load_staging(path: str | Path) -> Staging:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported staging schema_version: {data.get('schema_version')}")
    return Staging.model_validate(data)
