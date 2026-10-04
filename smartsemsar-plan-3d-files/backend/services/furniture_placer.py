"""
plan.json + catalog (+ style / preferences) -> staging.json  (deterministic rules, no LLM)
توزيع الفرش على الغرف بقواعد ثابتة: نوع الغرفة، الحوائط، الأبواب، الشبابيك، والحاجات الثابتة.

Style and preferences only choose WHICH catalog items are used; the rules decide where.
An item that does not fit is skipped, so a small room simply gets less furniture.
Preferences (backend/schema/design.py) steer the rules:
  density      minimal = essentials only, full = every optional piece that fits
  must_have    forces an item in every room it can go in, whatever the room size
  exclude      leaves items out
  dining_seats 4 or 6 instead of the placer's own choice
What was asked for but did not fit is listed in staging.unplaced.
After a room is furnished its doors must still be reachable on foot (same check as
staging_validator); if not, the last placed items are dropped until they are.

    staging = stage_plan(plan, preferences={"density": "full", "must_have": ["desk"], "exclude": ["tv"]})

Usage:
    python -m backend.services.furniture_placer PROP_1002
    -> data/stagings/PROP_1002.json
"""
import math
import sys
from typing import Callable, Iterator, NamedTuple, Optional, Union

from shapely.geometry import LineString, MultiPolygon, Polygon
from shapely.geometry import Point as ShPoint
from shapely.geometry.polygon import orient
from shapely.ops import unary_union

from backend.schema.design import DesignPreferences
from backend.schema.plan import Floor, Plan, Room, load_plan
from backend.schema.staging import (REPO_ROOT, Catalog, CatalogItem, PlacedItem,
                                    Staging, load_catalog, save_staging)
from backend.services.staging_validator import cut_off_doors

STEP_M = 0.1                # slide step along a wall
GRID_M = 0.25               # grid for free-standing items
WALL_GAP_M = 0.02           # gap between an item's back and the wall face
WALL_SNAP_M = 0.15          # a back edge this close to a wall counts as "against the wall"
DOOR_CLEAR_M = (0.7, 1.0)   # clear depth around a door = its width, kept inside this range
FIXTURE_CLEAR_M = 0.5       # room to use a built-in closet / cabinet / toilet
WINDOW_CLEAR_M = 0.25
LOW_ITEM_M = 0.9            # items up to the window sill may stand under a window
EPS_AREA = 1e-3             # overlaps smaller than this are just touching edges

SOFA_TABLE_GAP_M = 0.45
ARMCHAIR_GAP_M = 0.5
NIGHTSTAND_GAP_M = 0.05
CHAIR_GAP_M = 0.05
CHAIR_PULL_M = 0.3          # room to pull a dining chair out
TV_DISTANCE_M = (1.8, 5.5)
TV_MAX_OFFSET_M = 1.0       # how far off the sofa's axis the TV may sit
TV_VIEW_M = 3.0             # sofa-to-TV distance when the sofa stands free in a deep room


class Spot(NamedTuple):
    item: PlacedItem
    offset_m: float         # distance from the middle of the wall
    corner_m: float         # distance from the nearest end of the wall
    wall_m: float           # wall length
    wall: int               # index of the wall in the room outline


def _largest(geom) -> Polygon:
    if isinstance(geom, Polygon):
        return geom
    if isinstance(geom, MultiPolygon):
        return max(geom.geoms, key=lambda p: p.area)
    return Polygon()


def _hits(a, b) -> bool:
    return not b.is_empty and a.intersects(b) and a.intersection(b).area > EPS_AREA


def _facing(item: PlacedItem) -> tuple[float, float]:
    a = math.radians(item.rotation_deg)
    return math.sin(a), -math.cos(a)


def anchored(anchor: PlacedItem, cat: CatalogItem, lx: float, ly: float, turn_deg: float = 0.0) -> PlacedItem:
    """Item placed relative to another one, in the anchor's own frame."""
    x, y = anchor.to_plan(lx, ly)
    return PlacedItem.from_catalog(cat, id="", room_id=anchor.room_id, level=anchor.level,
                                   x=x, y=y, rotation_deg=(anchor.rotation_deg + turn_deg) % 360)


class RoomSpace:
    """Free space of one room while it is being furnished."""

    def __init__(self, floor: Floor, room: Room):
        self.floor, self.room, self.level = floor, room, floor.level
        walls = unary_union([Polygon(w.polygon).buffer(0) for w in floor.walls])
        # usable floor = room minus walls (door openings are part of the walls)
        # simplify: wall corners cut one straight side into several short edges
        self.poly = _largest(Polygon(room.polygon).buffer(0).difference(walls)).simplify(0.02)
        self.inside = self.poly.buffer(0.002)       # rounding slack only
        near = self.poly.buffer(1.5)
        self.walls = walls.buffer(WALL_SNAP_M).intersection(near)
        # مفيش اتجاه فتح للباب في البيانات، فبنسيب مساحة فاضية على الناحيتين
        # Only doors of this room count: a door on the far side of a wall needs no room here.
        doors = [(Polygon(d.polygon).buffer(0), min(max(d.width_m, DOOR_CLEAR_M[0]), DOOR_CLEAR_M[1]))
                 for d in floor.doors]
        self.doors = unary_union([g.buffer(depth) for g, depth in doors
                                  if g.distance(self.poly) < 0.05]).intersection(near)
        self.windows = unary_union([Polygon(w.polygon).buffer(0).buffer(WINDOW_CLEAR_M)
                                    for w in floor.windows]).intersection(near)
        solid = [x for x in floor.fixtures if x.on_floor]
        self.fixture_types = {x.type for x in floor.fixtures if x.room_id == room.id}
        # hard = things an item or its front strip may never overlap
        self.hard = unary_union([Polygon(x.polygon).buffer(0) for x in solid]).intersection(near)
        # clear = floor that stays empty but that front strips may share (same rule as doors)
        reachable = [Polygon(x.polygon).buffer(0) for x in solid if x.type not in ("column", "chimney")]
        self.clear = unary_union([self.doors] + [
            g.buffer(FIXTURE_CLEAR_M) for g in reachable if g.distance(self.poly) < 0.05]).intersection(near)
        self.front = Polygon()      # front strips / view corridors of placed items
        self.items: list[PlacedItem] = []

    @property
    def usable(self) -> bool:
        return self.poly.area >= 1.0

    # ----- checks -----
    def fits(self, item: PlacedItem, cat: CatalogItem, companion: bool = False) -> bool:
        """companion=True: the item belongs to a group (coffee table in front of the sofa),
        so it may stand inside another item's front strip."""
        fp = Polygon(item.footprint())
        if not self.inside.contains(fp) or _hits(fp, self.hard) or _hits(fp, self.clear):
            return False
        if not companion and _hits(fp, self.front):
            return False
        if cat.height_m > LOW_ITEM_M and _hits(fp, self.windows):
            return False
        zone = item.front_zone(cat.clearance_front_m)
        if zone:
            z = Polygon(zone)
            if not self.inside.contains(z) or _hits(z, self.hard):
                return False
        return True

    def place(self, item: PlacedItem, cat: CatalogItem) -> PlacedItem:
        item.id = f"{self.room.id}_I{len(self.items)}"
        self.items.append(item)
        self.hard = unary_union([self.hard, Polygon(item.footprint())])
        zone = item.front_zone(cat.clearance_front_m)
        if zone:
            self.keep_free(Polygon(zone))
        return item

    def keep_free(self, geom) -> None:
        self.front = unary_union([self.front, geom])

    def snapshot(self):
        return self.hard, self.front, len(self.items)

    def restore(self, snap) -> None:
        self.hard, self.front, n = snap
        del self.items[n:]

    # ----- candidates -----
    def wall_spots(self, cat: CatalogItem) -> Iterator[Spot]:
        """Every valid position with the item's back on a wall of this room."""
        ring = list(orient(self.poly, 1.0).exterior.coords)     # counter-clockwise: room is on the left
        for wall, ((x0, y0), (x1, y1)) in enumerate(zip(ring, ring[1:])):
            length = math.hypot(x1 - x0, y1 - y0)
            if length < cat.width_m:
                continue
            ux, uy = (x1 - x0) / length, (y1 - y0) / length
            nx, ny = -uy, ux                                    # into the room
            rotation = math.degrees(math.atan2(-uy, -ux)) % 360  # back toward the wall
            inset = cat.depth_m / 2 + WALL_GAP_M
            t = cat.width_m / 2
            while t <= length - cat.width_m / 2 + 1e-9:
                item = PlacedItem.from_catalog(
                    cat, id="", room_id=self.room.id, level=self.level,
                    x=round(x0 + ux * t + nx * inset, 4), y=round(y0 + uy * t + ny * inset, 4),
                    rotation_deg=round(rotation, 2))
                fp = item.footprint()
                # edges with no wall behind them are openings to the next room
                if self.walls.contains(LineString([fp[2], fp[3]])) and self.fits(item, cat):
                    yield Spot(item, abs(t - length / 2), min(t, length - t) - cat.width_m / 2,
                               length, wall)
                t += STEP_M

    def best_wall_spot(self, cat: Optional[CatalogItem], score: Callable[[Spot], float]) -> Optional[PlacedItem]:
        if cat is None:
            return None
        best = max(self.wall_spots(cat), key=score, default=None)
        return self.place(best.item, cat) if best else None

    def place_companion(self, anchor: PlacedItem, cat: Optional[CatalogItem],
                        lx: float, ly: float, turn_deg: float = 0.0) -> Optional[PlacedItem]:
        if cat is None:
            return None
        item = anchored(anchor, cat, lx, ly, turn_deg)
        return self.place(item, cat) if self.fits(item, cat, companion=True) else None

    def door_distance(self, item: PlacedItem, cap: float = 3.0) -> float:
        return cap if self.doors.is_empty else min(self.doors.distance(ShPoint(item.x, item.y)), cap)

    def main_axis_deg(self) -> float:
        """Direction of the room's long side."""
        box = list(self.poly.minimum_rotated_rectangle.exterior.coords)
        (x0, y0), (x1, y1) = max(zip(box, box[1:]), key=lambda e: math.dist(*e))
        return math.degrees(math.atan2(y1 - y0, x1 - x0)) % 180


class Wishes:
    """What the preferences ask of the room rules."""

    def __init__(self, prefs: Optional[DesignPreferences] = None):
        prefs = prefs or DesignPreferences()
        self.density, self.must = prefs.density, set(prefs.must_have)

    def wants(self, item: Optional[CatalogItem], level: str = "normal", when: bool = True) -> bool:
        """Should the rules try to place this item?
        level: "essential" (placed even in a minimal room), "normal", or "full" (extras).
        when:  the rule's own condition, e.g. the room is big enough."""
        if item is None:
            return False
        if item.is_a(self.must):                # asked for by name: try it whatever the room
            return True
        if level == "essential":
            return when
        if self.density == "minimal":
            return False
        if self.density == "full":              # every optional piece that fits
            return True
        return level == "normal" and when


# ---------- room rules ----------
def furnish_bedroom(space: RoomSpace, cat: dict[str, CatalogItem], master: bool,
                    wishes: Optional[Wishes] = None) -> None:
    wishes = wishes or Wishes()
    night = cat.get("nightstand")

    def nightstands(bed: PlacedItem) -> list[PlacedItem]:
        if night is None:
            return []
        lx = bed.width_m / 2 + night.width_m / 2 + NIGHTSTAND_GAP_M
        ly = bed.depth_m / 2 - night.depth_m / 2            # backs in one line
        return [anchored(bed, night, side * lx, ly) for side in (1, -1)]

    def bed_score(spot: Spot) -> float:
        # بعيد عن الباب، في نص الحيطة، مش تحت الشباك، وفيه مكان للكومودينو
        sides = sum(space.fits(n, night, companion=True) for n in nightstands(spot.item))
        under_window = _hits(Polygon(spot.item.footprint()), space.windows)
        return space.door_distance(spot.item) - 0.5 * spot.offset_m + 1.0 * sides - 2.0 * under_window

    big = master or space.room.area_sqm >= 12
    beds = [cat[i] for i in (["bed_double", "bed_single"] if big else ["bed_single"]) if i in cat]
    # أوضة صغيرة: نقبل السرير من غير مساحة قدامه بدل ما الأوضة تفضل من غير سرير
    beds += [b.model_copy(update={"clearance_front_m": 0.0}) for b in beds[-1:]]
    bed = None
    for bed_cat in beds:
        bed = space.best_wall_spot(bed_cat, bed_score)
        if bed:
            break
    if bed is None:
        return      # too small for a bed: leave it empty rather than half furnished
    if wishes.wants(night):
        for n in nightstands(bed):
            if space.fits(n, night, companion=True):
                space.place(n, night)

    def corner(spot: Spot) -> float:
        return -spot.corner_m

    wardrobe, dresser, desk, shelf = (cat.get(i) for i in ("wardrobe", "dresser", "desk", "bookshelf"))
    if wishes.wants(wardrobe, "essential", when="closet" not in space.fixture_types):  # no built-in closet
        space.best_wall_spot(wardrobe, corner)
    def place_dresser() -> None:
        if wishes.wants(dresser, when=master):
            space.best_wall_spot(dresser, corner)

    def place_desk() -> None:
        if wishes.wants(desk, when=not master and space.room.area_sqm >= 10):
            # desk likes daylight
            space.best_wall_spot(desk, lambda s: 2.0 * _hits(Polygon(s.item.footprint()), space.windows)
                                 - 0.2 * s.corner_m)

    # the room's usual piece goes first, so a fuller room never loses it to an extra one
    for place in ((place_dresser, place_desk) if master else (place_desk, place_dresser)):
        place()
    if wishes.wants(shelf, "full"):
        space.best_wall_spot(shelf, corner)


def furnish_living(space: RoomSpace, cat: dict[str, CatalogItem], wishes: Optional[Wishes] = None) -> None:
    wishes = wishes or Wishes()
    tv_cat = cat.get("tv_unit")

    def sofa_score(spot: Spot) -> float:
        return 0.2 * spot.wall_m - 0.5 * spot.offset_m + 0.5 * space.door_distance(spot.item)

    def tv_score(sofa: PlacedItem) -> Callable[[Spot], float]:
        fx, fy = _facing(sofa)

        def score(spot: Spot) -> float:
            tx, ty = _facing(spot.item)
            vx, vy = spot.item.x - sofa.x, spot.item.y - sofa.y
            ahead, aside = vx * fx + vy * fy, abs(vx * fy - vy * fx)
            ok = (fx * tx + fy * ty < -0.9 and TV_DISTANCE_M[0] <= ahead <= TV_DISTANCE_M[1]
                  and aside <= TV_MAX_OFFSET_M)
            return -aside - 0.1 * abs(ahead - 3.0) if ok else -math.inf
        return score

    def place_tv(sofa: PlacedItem) -> Optional[PlacedItem]:
        if tv_cat is None:
            return None
        score = tv_score(sofa)
        best = max(space.wall_spots(tv_cat), key=score, default=None)
        return space.place(best.item, tv_cat) if best and score(best) > -math.inf else None

    sofa = tv = sofa_cat = None
    for sofa_id in ("sofa_3_seat", "sofa_2_seat"):
        sofa_cat = cat.get(sofa_id)
        if sofa_cat is None:
            continue
        spots = sorted(space.wall_spots(sofa_cat), key=sofa_score, reverse=True)
        per_wall = list({s.wall: s for s in reversed(spots)}.values())[::-1]    # best spot of each wall
        # الكنبة لازم يكون قدامها مكان للتلفزيون، فبنجرب أحسن مكان على كل حيطة بالترتيب
        for spot in per_wall:
            snap = space.snapshot()
            sofa = space.place(spot.item.model_copy(), sofa_cat)
            tv = place_tv(sofa)
            if tv:
                break
            space.restore(snap)
            sofa = None
        if sofa is None and tv_cat is not None:
            # deep room, no wall opposite within range: TV on a wall, sofa standing free facing it
            ly = -(tv_cat.depth_m / 2 + TV_VIEW_M + sofa_cat.depth_m / 2)
            tv_spots = sorted(space.wall_spots(tv_cat), key=sofa_score, reverse=True)
            for spot in list({s.wall: s for s in reversed(tv_spots)}.values())[::-1]:
                snap = space.snapshot()
                tv = space.place(spot.item.model_copy(), tv_cat)
                sofa = space.place_companion(tv, sofa_cat, 0.0, ly, turn_deg=180)
                if sofa:
                    break
                space.restore(snap)
                tv = None
        if sofa is None and spots:
            sofa = space.place(spots[0].item, sofa_cat)     # no TV possible: keep the best sofa spot
        if sofa:
            break
    if sofa is None:
        return
    if tv:  # nothing else may stand between the sofa and the TV
        space.keep_free(Polygon(sofa.footprint()).union(Polygon(tv.footprint())).convex_hull)

    table = cat.get("coffee_table")
    ly = -(sofa.depth_m / 2 + SOFA_TABLE_GAP_M + (table.depth_m / 2 if table else 0.3))
    placed_table = space.place_companion(sofa, table, 0.0, ly)
    arm, shelf = cat.get("armchair"), cat.get("bookshelf")
    if wishes.wants(arm, when=space.room.area_sqm >= 18):
        half = (placed_table.width_m / 2) if placed_table else 0.55
        lx = half + ARMCHAIR_GAP_M + arm.depth_m / 2
        space.place_companion(sofa, arm, lx, ly, turn_deg=-90)     # faces the table
        space.place_companion(sofa, arm, -lx, ly, turn_deg=90)
    if wishes.wants(shelf):
        space.best_wall_spot(shelf, lambda s: -s.corner_m)


def furnish_dining(space: RoomSpace, cat: dict[str, CatalogItem], seats: int = 6) -> bool:
    """Free-standing table + chairs in the most open part of the room. True if a table was placed."""
    chair = cat.get("dining_chair")
    reach = (chair.depth_m + CHAIR_GAP_M + CHAIR_PULL_M) if chair else CHAIR_PULL_M
    blocked = unary_union([space.hard, space.clear, space.front])
    min_x, min_y, max_x, max_y = space.poly.bounds
    axis = space.main_axis_deg()

    for table_id, per_side in (("dining_table_6", 3), ("dining_table_4", 2)):
        table = cat.get(table_id)
        if table is None or per_side * 2 > seats:
            continue
        # table + pulled-out chairs must fit as one block
        block = table.model_copy(update={"width_m": table.width_m + 2 * CHAIR_PULL_M,
                                         "depth_m": table.depth_m + 2 * reach})
        best, best_open = None, -1.0
        for rotation in (axis, axis + 90):
            y = min_y
            while y <= max_y:
                x = min_x
                while x <= max_x:
                    probe = PlacedItem.from_catalog(block, id="", room_id=space.room.id, level=space.level,
                                                    x=round(x, 4), y=round(y, 4), rotation_deg=round(rotation, 2))
                    fp = Polygon(probe.footprint())
                    if space.inside.contains(fp) and not _hits(fp, blocked):
                        centre = ShPoint(x, y)
                        open_m = space.poly.exterior.distance(centre)
                        if not blocked.is_empty:
                            open_m = min(open_m, blocked.distance(centre))
                        if open_m > best_open + 1e-9:
                            best, best_open = probe, open_m
                    x += GRID_M
                y += GRID_M
        if best is None:
            continue
        placed = space.place(PlacedItem.from_catalog(table, id="", room_id=space.room.id, level=space.level,
                                                     x=best.x, y=best.y, rotation_deg=best.rotation_deg), table)
        if chair:
            ly = table.depth_m / 2 + chair.depth_m / 2 + CHAIR_GAP_M
            for side, turn in ((1, 0), (-1, 180)):          # chairs face the table
                for k in range(per_side):
                    lx = (k - (per_side - 1) / 2) * (table.width_m / per_side)
                    space.place_companion(placed, chair, lx, side * ly, turn)
        return True
    return False


def furnish_entry(space: RoomSpace, cat: dict[str, CatalogItem], wishes: Optional[Wishes] = None) -> None:
    wishes = wishes or Wishes()
    console, shoes = cat.get("console_table"), cat.get("shoe_cabinet")
    for item, big_enough in ((console, space.room.area_sqm >= 8), (shoes, True)):   # one piece per entry
        if wishes.wants(item, when=big_enough) and space.best_wall_spot(
                item, lambda s: space.door_distance(s.item) - 0.2 * s.corner_m):
            return


def keep_paths_open(space: RoomSpace) -> int:
    """Drop the last placed items until every door of the room is reachable again.
    Companions are always placed after their anchor, so none is left orphaned.
    Returns how many items were dropped."""
    dropped = 0
    while space.items and cut_off_doors(space.floor, space.room,
                                        [Polygon(i.footprint()) for i in space.items]):
        space.items.pop()
        dropped += 1
    return dropped


# ---------- plan -> staging ----------
def stage_plan(plan: Plan, catalog: Optional[Catalog] = None, style: Optional[str] = None,
               preferences: Union[DesignPreferences, dict, None] = None) -> Staging:
    """preferences: a DesignPreferences or a dict with its fields (see backend/schema/design.py)."""
    catalog = catalog or load_catalog()
    prefs = preferences if isinstance(preferences, DesignPreferences) \
        else DesignPreferences.model_validate(preferences or {})
    style = style or (prefs.style if prefs.style != "unknown" else None)
    wishes = Wishes(prefs)
    exclude = set(prefs.exclude)

    def available(room_type: str) -> dict[str, CatalogItem]:
        return {i.id: i for i in catalog.for_room(room_type, style) if not i.is_a(exclude)}

    bedrooms = plan.rooms("bedroom")
    master_id = max(bedrooms, key=lambda r: r.area_sqm).id if bedrooms else None
    has_dining_room = plan.count("dining") > 0
    spaces: list[RoomSpace] = []
    dining_done = False

    for floor in plan.floors:
        for room in floor.rooms:
            space = RoomSpace(floor, room)
            if not space.usable:
                continue
            cat = available(room.type)
            if room.type == "bedroom":
                furnish_bedroom(space, cat, master=room.id == master_id, wishes=wishes)
            elif room.type == "living":
                furnish_living(space, cat, wishes)
                if not has_dining_room and not dining_done:     # السفرة في الريسبشن لو مفيش أوضة سفرة
                    seats = prefs.dining_seats or (6 if room.area_sqm >= 30 else 4)
                    dining_done = furnish_dining(space, cat, seats=seats)
            elif room.type == "dining":
                dining_done = furnish_dining(space, cat, seats=prefs.dining_seats or 6) or dining_done
            elif room.type in ("entry", "hallway"):
                furnish_entry(space, cat, wishes)
            keep_paths_open(space)
            spaces.append(space)

    if not dining_done:     # last resort: a small table in the kitchen
        for space in spaces:
            if space.room.type == "kitchen" and furnish_dining(space, available("kitchen"),
                                                               seats=prefs.dining_seats or 4):
                keep_paths_open(space)
                break

    items = [i for s in spaces for i in s.items]
    # asked for by name, but nowhere in the result (no room, excluded, or not a catalog name)
    placed = [catalog.get(i.catalog_id) for i in items]
    unplaced = [name for name in prefs.must_have if not any(c.is_a([name]) for c in placed)]
    if prefs.dining_seats and not any(c.id == f"dining_table_{prefs.dining_seats}" for c in placed):
        unplaced.append(f"dining_table_{prefs.dining_seats}")
    return Staging(staging_id=f"{plan.plan_id}_{style or 'default'}", plan_id=plan.plan_id, style=style,
                   preferences=prefs.model_dump(exclude_unset=True), items=items, unplaced=unplaced)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python -m backend.services.furniture_placer <property_id> [style]")
    pid = sys.argv[1]
    style = sys.argv[2] if len(sys.argv) > 2 else None
    staging = stage_plan(load_plan(REPO_ROOT / "data" / "plans" / f"{pid}.json"), style=style)
    out = save_staging(staging, REPO_ROOT / "data" / "stagings" / f"{pid}.json")
    print(f"{len(staging.items)} items -> {out}")
