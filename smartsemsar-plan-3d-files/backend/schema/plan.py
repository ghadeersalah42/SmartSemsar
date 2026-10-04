"""
Smart Semsar - Canonical floor plan schema (plan.json)
العقد الموحد للمخطط: مصدر الحقيقة الوحيد للـ 2D والـ 3D.

Rules:
  - All geometry is in meters, per floor, in a floor-local frame:
    x to the right, y up (SVG y is flipped), origin at the floor's min corner.
  - The 3D model is built from this file deterministically (no diffusion geometry).
  - Every plan.json carries schema_version = 1.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field

SCHEMA_VERSION = 1

PlanSource = Literal["user_upload", "cubicasa_lookalike", "generated"]

# الرسالة اللي بتظهر للمستخدم حسب مصدر المخطط
SOURCE_WARNINGS: dict[str, str] = {
    "user_upload": "Built from your uploaded floor plan.",
    "cubicasa_lookalike": (
        "Representative layout: same rooms and size as this listing, "
        "not its actual floor plan."
    ),
    "generated": "AI-generated concept, not based on any real plan.",
}

RoomType = Literal[
    "bedroom", "bathroom", "living", "kitchen", "dining", "entry", "hallway",
    "storage", "closet", "laundry", "sauna", "technical", "room",
    "outdoor", "garage", "other",
]

# Room types that do not count toward indoor area / المساحات الخارجية
OUTDOOR_TYPES = {"outdoor", "garage"}

FixtureType = Literal[
    "toilet", "sink", "shower", "bathtub", "base_cabinet", "wall_cabinet",
    "appliance", "closet", "fireplace", "chimney", "sauna_bench",
    "stairs", "column", "other",
]

Point = tuple[float, float]


class Room(BaseModel):
    id: str
    type: RoomType
    raw_type: Optional[str] = None          # e.g. CubiCasa class "Space Bath Shower"
    polygon: list[Point]                    # meters
    area_sqm: float

    @property
    def is_indoor(self) -> bool:
        return self.type not in OUTDOOR_TYPES


class Wall(BaseModel):
    id: str
    polygon: list[Point]                    # wall footprint, meters
    thickness_m: float
    exterior: bool = False
    height_m: float = 2.7


class Door(BaseModel):
    id: str
    wall_id: str
    polygon: list[Point]                    # opening footprint inside the wall
    width_m: float
    kind: str = "swing"                     # swing / slide / zfold / none
    height_m: float = 2.1


class Window(BaseModel):
    id: str
    wall_id: str
    polygon: list[Point]
    width_m: float
    sill_height_m: float = 0.9
    height_m: float = 1.2


class Fixture(BaseModel):
    """Built-in element already drawn on the plan (الحاجات الثابتة: حمام، مطبخ، سلم، عمود)."""
    id: str
    type: FixtureType
    raw_type: Optional[str] = None          # e.g. CubiCasa class "FixedFurniture Toilet"
    room_id: Optional[str] = None           # None = not inside any room (e.g. column in a wall)
    polygon: list[Point]                    # footprint, meters
    height_m: float = 1.0
    elevation_m: float = 0.0                # bottom above the floor (wall cabinets)

    @property
    def on_floor(self) -> bool:
        """False for wall-hung fixtures: furniture may stand under them."""
        return self.elevation_m == 0.0


class Floor(BaseModel):
    level: int                              # 0 = ground, 1 = first, -1 = basement
    name: str = ""
    rooms: list[Room] = Field(default_factory=list)
    walls: list[Wall] = Field(default_factory=list)
    doors: list[Door] = Field(default_factory=list)
    windows: list[Window] = Field(default_factory=list)
    fixtures: list[Fixture] = Field(default_factory=list)

    @property
    def indoor_area_sqm(self) -> float:
        return sum(r.area_sqm for r in self.rooms if r.is_indoor)


class ScaleInfo(BaseModel):
    # method: "native" = drawing's own scale; "fit_listing_area" = scaled to the listing area
    method: Literal["native", "fit_listing_area"]
    native_px_per_m: Optional[float] = None     # measured from the source drawing
    factor: float = 1.0                          # linear factor applied on top of native meters
    native_area_sqm: Optional[float] = None      # indoor area before fitting
    target_area_sqm: Optional[float] = None      # listing pf_area_sqm (if any)


class Plan(BaseModel):
    schema_version: Literal[1] = SCHEMA_VERSION
    plan_id: str
    source: PlanSource
    source_ref: Optional[str] = None             # e.g. "cubicasa:4770"
    listing_id: Optional[str] = None             # e.g. "PROP_1001"
    scale: ScaleInfo
    floors: list[Floor]
    total_area_sqm: float                        # indoor room area, all floors

    @property
    def num_floors(self) -> int:
        return len(self.floors)

    @property
    def source_warning(self) -> str:
        return SOURCE_WARNINGS[self.source]

    def rooms(self, room_type: Optional[str] = None) -> list[Room]:
        return [r for f in self.floors for r in f.rooms
                if room_type is None or r.type == room_type]

    def count(self, room_type: str) -> int:
        return len(self.rooms(room_type))

    def fixtures(self, room_id: Optional[str] = None) -> list[Fixture]:
        return [x for f in self.floors for x in f.fixtures
                if room_id is None or x.room_id == room_id]

    def recompute_total_area(self) -> float:
        self.total_area_sqm = round(sum(f.indoor_area_sqm for f in self.floors), 2)
        return self.total_area_sqm


# ---------- load / save ----------
def save_plan(plan: Plan, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(plan.model_dump_json(indent=1), encoding="utf-8")
    return path


def load_plan(path: str | Path) -> Plan:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported plan schema_version: {data.get('schema_version')}")
    return Plan.model_validate(data)
