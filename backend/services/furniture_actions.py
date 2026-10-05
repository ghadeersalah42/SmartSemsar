"""
Placement instructions (from the agent's LLM / VLM) -> PlacedItems on the plan.
الموديل بيقول "السرير على الحيطة B"، والكود هنا بيحسب المكان بالظبط ويرفض أي حاجة مش سليمة.

The model decides the design (what goes where, in words); this module decides the geometry.
Every placement passes the same rules as furniture_placer, and staging_validator has the final
say after each instruction. What fails comes back as a short sentence the model can act on.

Room description (what the model reads):
    describe_room(plan, room_id, staging)  ->  "Room F0_R4 (bedroom, 21.4 m2 ...)\n  wall A ..."

Instructions (one dict each; `name` labels a piece so later instructions can refer to it):
    {"item": "bed_double",   "place": "wall", "wall": "B", "align": "center|start|end", "name": "bed"}
    {"item": "tv_unit",      "place": "wall", "wall": "D", "align_to": "sofa"}
    {"item": "nightstand",   "place": "beside", "ref": "bed", "side": "both|left|right"}
    {"item": "coffee_table", "place": "in_front_of", "ref": "sofa", "gap_m": 0.45, "face_ref": false}
    {"item": "dining_table_4", "place": "center" | "free"}
    {"remove": "<item id, label or catalog id>"}

    staging, errors = apply_actions(plan, staging, {"F0_R4": [...]})
"""
import math
from typing import Optional

from shapely.geometry import LineString, Point, Polygon
from shapely.geometry.polygon import orient
from shapely.ops import unary_union

from backend.schema.plan import Plan
from backend.schema.staging import Catalog, CatalogItem, PlacedItem, Staging, load_catalog
from backend.services.staging_validator import (DOOR_CLEAR_M, WINDOW_SILL_M, cut_off_doors,
                                                room_doors, validate_staging)

WALL_GAP_M = 0.02           # back edge this far off the wall face
DOOR_MARGIN_M = 0.1         # free wall beside a door, along the wall
OPEN_SIDE_COVERAGE = 0.5    # a side with less wall than this is open (open-plan kitchen)
SLIDE_STEP_M = 0.1
FREE_GRID_M = 0.2
EPS_AREA = 1e-3


def _hits(a, b) -> bool:
    return not b.is_empty and a.intersects(b) and a.intersection(b).area > EPS_AREA


def _poly(points) -> Polygon:
    p = Polygon(points).buffer(0)
    return max(getattr(p, "geoms", [p]), key=lambda g: g.area)


def _unit(dx, dy):
    n = math.hypot(dx, dy)
    return dx / n, dy / n


def _side_name(n):
    # where the wall is = opposite of the inward normal
    ang = math.degrees(math.atan2(-n[1], -n[0])) % 360
    return ["east", "north", "west", "south"][int(((ang + 45) % 360) // 90)]


def _interval(poly, p0, t, length):
    s = [(x - p0[0]) * t[0] + (y - p0[1]) * t[1] for x, y in poly.exterior.coords]
    return max(0.0, min(s)), min(length, max(s))


def _subtract(spans, cuts):
    out = list(spans)
    for c0, c1 in cuts:
        nxt = []
        for a, b in out:
            if c1 <= a or c0 >= b:
                nxt.append((a, b))
                continue
            if c0 > a:
                nxt.append((a, c0))
            if c1 < b:
                nxt.append((c1, b))
        out = nxt
    return [(a, b) for a, b in out if b - a > 1e-3]


class WallSide:
    def __init__(self, label, p0, p1, n, is_open):
        self.label, self.p0, self.p1, self.n, self.is_open = label, p0, p1, n, is_open
        self.length = math.dist(p0, p1)
        self.t = _unit(p1[0] - p0[0], p1[1] - p0[1])
        self.side = _side_name(n)
        self.doors, self.windows, self.fixtures = [], [], []   # [(s0, s1)] / [(s0, s1, type)]

    def at(self, s, inward=0.0):
        return (self.p0[0] + self.t[0] * s + self.n[0] * inward,
                self.p0[1] + self.t[1] * s + self.n[1] * inward)

    def segment(self):
        return LineString([self.p0, self.p1])


class RoomView:
    """One room as the agent sees it: lettered walls with their doors, windows and built-ins."""

    def __init__(self, plan: Plan, room_id: str):
        self.floor, self.room = next((f, r) for f in plan.floors for r in f.rooms if r.id == room_id)
        self.poly = orient(_poly(self.room.polygon), sign=1.0)     # CCW: inside is on the left
        solid = unary_union([_poly(w.polygon) for w in self.floor.walls]).buffer(0.06)
        self.walls: list[WallSide] = []
        coords = list(self.poly.simplify(0.03).exterior.coords)[:-1]
        for i in range(len(coords)):
            p0, p1 = coords[i], coords[(i + 1) % len(coords)]
            if math.dist(p0, p1) < 0.5:
                continue
            t = _unit(p1[0] - p0[0], p1[1] - p0[1])
            seg = LineString([p0, p1])
            is_open = seg.intersection(solid).length < OPEN_SIDE_COVERAGE * seg.length
            self.walls.append(WallSide(chr(65 + len(self.walls)), p0, p1, (-t[1], t[0]), is_open))

        self.doors = room_doors(self.floor, self.room)                   # [(door id, polygon)]
        self.door_zone = unary_union([g.buffer(DOOR_CLEAR_M) for _, g in self.doors])
        self.fixtures = [x for x in self.floor.fixtures if x.on_floor]
        self.fixture_shape = unary_union([_poly(x.polygon) for x in self.fixtures])
        self.walls_shape = unary_union([_poly(w.polygon) for w in self.floor.walls])
        self.windows_shape = unary_union([_poly(w.polygon).buffer(0.2) for w in self.floor.windows])
        for w in self.walls:
            seg = w.segment()
            for _, g in self.doors:
                if g.distance(seg) < 0.08:
                    s0, s1 = _interval(g, w.p0, w.t, w.length)
                    if s1 - s0 > 0.3:
                        w.doors.append((s0, s1))
            for win in self.floor.windows:
                g = _poly(win.polygon)
                if g.distance(seg) < 0.08:
                    s0, s1 = _interval(g, w.p0, w.t, w.length)
                    if s1 - s0 > 0.3:
                        w.windows.append((s0, s1))
            for x in self.fixtures:
                g = _poly(x.polygon)
                if g.distance(seg) < 0.15 and g.intersection(self.poly.buffer(0.05)).area > EPS_AREA:
                    s0, s1 = _interval(g, w.p0, w.t, w.length)
                    if s1 - s0 > 0.1:
                        w.fixtures.append((s0, s1, x.type))

    def wall(self, label) -> Optional[WallSide]:
        return next((w for w in self.walls if w.label == str(label).strip().upper()), None)

    def free_spans(self, wall: WallSide, tall: bool):
        if wall.is_open:
            return []
        cuts = [(a - DOOR_MARGIN_M - DOOR_CLEAR_M, b + DOOR_MARGIN_M + DOOR_CLEAR_M) for a, b in wall.doors]
        cuts += [(a, b) for a, b, _ in wall.fixtures]
        if tall:
            cuts += wall.windows
        return _subtract([(0.0, wall.length)], cuts)

    def longest_wall(self) -> WallSide:
        return max(self.walls, key=lambda w: (not w.is_open, w.length))

    def describe(self, items: list[PlacedItem] = (), catalog: Optional[Catalog] = None) -> str:
        r = self.room
        x0, y0, x1, y1 = self.poly.bounds
        lines = [f"Room {r.id} ({r.type}, {r.area_sqm:.1f} m2, about {x1 - x0:.1f} x {y1 - y0:.1f} m)"]
        for w in self.walls:
            if w.is_open:
                lines.append(f"  side {w.label} ({w.side}, {w.length:.1f} m): open to the next room, no wall")
                continue
            feats = [f"door at {a:.1f}-{b:.1f} m" for a, b in w.doors]
            feats += [f"window at {a:.1f}-{b:.1f} m" for a, b in w.windows]
            feats += [f"built-in {t.replace('_', ' ')} at {a:.1f}-{b:.1f} m" for a, b, t in w.fixtures]
            free = ", ".join(f"{b - a:.1f} m" for a, b in self.free_spans(w, tall=False)) or "none"
            lines.append(f"  wall {w.label} ({w.side}, {w.length:.1f} m): {'; '.join(feats) or 'plain wall'}"
                         f"; free wall: {free}")
        loose = [x.type.replace("_", " ") for x in self.fixtures if x.room_id == r.id
                 and all(_poly(x.polygon).distance(w.segment()) >= 0.15 for w in self.walls)]
        if loose:
            lines.append(f"  free-standing built-ins: {', '.join(loose)}")
        for i in items:
            name = catalog.get(i.catalog_id).name if catalog else i.catalog_id
            lines.append(f"  placed: {i.id} = {name} at ({i.x:.1f}, {i.y:.1f})")
        return "\n".join(lines)


def describe_room(plan: Plan, room_id: str, staging: Optional[Staging] = None,
                  catalog: Optional[Catalog] = None) -> str:
    return RoomView(plan, room_id).describe(staging.in_room(room_id) if staging else [], catalog)


# ---------- placing ----------
def _rotation_facing(n) -> float:
    """rotation_deg so the item faces direction n (the schema: at 0 it faces -y)."""
    return math.degrees(math.atan2(n[0], -n[1])) % 360


class RoomPlacer:
    """Places catalog items in one room of a staging; every candidate is checked first."""

    def __init__(self, plan: Plan, staging: Staging, room_id: str, catalog: Catalog):
        self.plan, self.catalog = plan, catalog
        self.view = RoomView(plan, room_id)
        self.items = list(staging.items)                     # whole staging (open-plan overlaps)
        self.labels: dict[str, str] = {}                     # label -> item id
        self._n = sum(1 for i in staging.items if i.room_id == room_id)

    # ----- lookup -----
    def find(self, ref) -> Optional[PlacedItem]:
        ref = str(ref or "")
        key = self.labels.get(ref, ref)
        for i in reversed(self.items):
            if i.room_id == self.view.room.id and (i.id == key or i.catalog_id == ref):
                return i
        return None

    def _new_id(self, catalog_id):
        self._n += 1
        return f"{self.view.room.id}_{catalog_id}_{self._n}"

    # ----- checks (same rules as furniture_placer / staging_validator) -----
    def problem(self, item: PlacedItem, c: CatalogItem, ignore=()) -> Optional[str]:
        v = self.view
        fp = _poly(item.footprint())
        if not v.poly.buffer(0.02).contains(fp) or _hits(fp, v.walls_shape):
            return "goes outside the room or into a wall"
        if _hits(fp, v.fixture_shape):
            return "overlaps a built-in (kitchen, bathroom or closet)"
        if _hits(fp, v.door_zone):
            return "blocks a door"
        if c.height_m > WINDOW_SILL_M and _hits(fp, v.windows_shape):
            return "is too tall to stand in front of a window"
        for o in self.items:
            if o.level == item.level and o.id not in ignore and _hits(fp, _poly(o.footprint())):
                return f"overlaps the {self.catalog.get(o.catalog_id).name.lower()} ({o.id})"
        zone = item.front_zone(c.clearance_front_m)
        if zone:
            z = _poly(zone)
            if _hits(z, v.walls_shape) or _hits(z, v.fixture_shape) or any(
                    o.level == item.level and o.id not in ignore and _hits(z, _poly(o.footprint()))
                    for o in self.items):
                return f"has no free space in front of it ({c.clearance_front_m} m needed to use it)"
        return None

    def doors_ok(self) -> bool:
        fps = [_poly(i.footprint()) for i in self.items if i.room_id == self.view.room.id]
        return not cut_off_doors(self.view.floor, self.view.room, fps)

    def _take(self, c, x, y, rot, label):
        item = PlacedItem.from_catalog(c, self._new_id(c.id), self.view.room.id, self.view.floor.level,
                                       round(x, 3), round(y, 3), round(rot % 360, 1))
        why = self.problem(item, c)
        if why:
            return None, why
        self.items.append(item)
        if not self.doors_ok():
            self.items.pop()
            return None, "cuts off the way to a door"
        if label:
            self.labels[str(label)] = item.id
        return item, None

    # ----- actions -----
    def on_wall(self, c: CatalogItem, label, align="center", name=None, align_to=None):
        wall = self.view.wall(label)
        if wall is None:
            return None, f"wall {label} does not exist in {self.view.room.id}"
        if wall.is_open:
            return None, f"side {wall.label} is open (no wall) - nothing can stand against it"
        tall = c.height_m > WINDOW_SILL_M
        spans = [(a, b) for a, b in self.view.free_spans(wall, tall) if b - a >= c.width_m]
        if not spans:
            free = ", ".join(f"{b - a:.1f}" for a, b in self.view.free_spans(wall, tall)) or "none"
            return None, f"{c.name} ({c.width_m} m wide) does not fit on wall {wall.label} (free wall: {free} m)"
        ref = self.find(align_to) if align_to else None
        if ref is not None:
            want = (ref.x - wall.p0[0]) * wall.t[0] + (ref.y - wall.p0[1]) * wall.t[1]
        else:
            want = {"start": c.width_m / 2, "end": wall.length - c.width_m / 2}.get(align, wall.length / 2)
        spots = []
        for a, b in spans:
            lo, hi = a + c.width_m / 2, b - c.width_m / 2
            k = max(1, int((hi - lo) / SLIDE_STEP_M))
            spots += [lo + (hi - lo) * i / k for i in range(k + 1)]
        spots.sort(key=lambda s: abs(s - want))
        rot = _rotation_facing(wall.n)
        why = None
        for s in spots:
            x, y = wall.at(s, c.depth_m / 2 + WALL_GAP_M)
            item, why = self._take(c, x, y, rot, name)
            if item:
                return item, None
        return None, f"{c.name} on wall {wall.label} {why}"

    def beside(self, c: CatalogItem, ref, side="both", gap=0.05):
        base = self.find(ref)
        if base is None:
            return [], f"{c.name}: nothing called '{ref}' in {self.view.room.id} to stand beside"
        placed, reasons = [], []
        for sign in ([-1, 1] if side == "both" else [-1 if side == "left" else 1]):
            lx = sign * (base.width_m / 2 + gap + c.width_m / 2)
            ly = base.depth_m / 2 - c.depth_m / 2                # backs in line (+y is the back)
            x, y = base.to_plan(lx, ly)
            item, why = self._take(c, x, y, base.rotation_deg, None)
            (placed.append(item) if item else reasons.append(why))
        return placed, (None if placed else f"{c.name} beside the {ref}: {reasons[0]}")

    def in_front_of(self, c: CatalogItem, ref, gap=0.45, name=None, face_ref=None):
        base = self.find(ref)
        if base is None:
            return None, f"{c.name}: nothing called '{ref}' in {self.view.room.id} to stand in front of"
        x, y = base.to_plan(0.0, -(base.depth_m / 2 + gap + c.depth_m / 2))     # -y is the front
        turn = face_ref if face_ref is not None else c.category == "chair"
        item, why = self._take(c, x, y, base.rotation_deg + (180 if turn else 0), name)
        return (item, None) if item else (None, f"{c.name} in front of the {ref} {why}")

    def middle(self, c: CatalogItem, name=None, at=None):
        if at is None:
            pt = self.view.poly.centroid
            at = (pt.x, pt.y) if self.view.poly.contains(pt) else self.view.poly.representative_point().coords[0]
        w = self.view.longest_wall()
        base = math.degrees(math.atan2(w.t[1], w.t[0]))
        why = None
        for r in (0.0, 0.3, 0.6, 0.9):
            for k in range(8 if r else 1):
                a = k * math.pi / 4
                for rot in (base, base + 90):
                    item, why = self._take(c, at[0] + r * math.cos(a), at[1] + r * math.sin(a), rot, name)
                    if item:
                        return item, None
        return None, f"{c.name} does not fit in the middle of the room ({why})"

    def free_spot(self, c: CatalogItem, name=None, clearance=0.5):
        v = self.view
        x0, y0, x1, y1 = v.poly.bounds
        w = v.longest_wall()
        base = math.degrees(math.atan2(w.t[1], w.t[0]))
        blocked = unary_union([v.poly.exterior.buffer(0.01), v.fixture_shape] +
                              [_poly(o.footprint()) for o in self.items if o.level == v.floor.level])
        best, score = None, -1.0
        y = y0
        while y <= y1:
            x = x0
            while x <= x1:
                for rot in (base, base + 90):
                    probe = PlacedItem.from_catalog(c, "probe", v.room.id, v.floor.level, x, y, rot)
                    if self.problem(probe, c) is None:
                        d = min(_poly(probe.footprint()).distance(blocked), 1.2)
                        if d > score:
                            best, score = (x, y, rot), d
                x += FREE_GRID_M
            y += FREE_GRID_M
        if best is None or score < clearance - 1e-6:
            return None, f"{c.name} needs open floor with {clearance} m around it; none is left"
        return self._take(c, *best, name)

    def remove(self, ref):
        item = self.find(ref)
        if item is None:
            return f"nothing called '{ref}' in {self.view.room.id} to remove"
        self.items = [i for i in self.items if i.id != item.id]
        self.labels = {k: v for k, v in self.labels.items() if v != item.id}
        return None


def run_action(p: RoomPlacer, a: dict):
    """-> (placed items, error or None)"""
    if "remove" in a:
        err = p.remove(a["remove"])
        return [], err
    try:
        c = p.catalog.get(str(a.get("item")))
    except KeyError:
        return [], f"'{a.get('item')}' is not in the catalog"
    if p.view.room.type not in c.room_types:
        return [], f"{c.name} is not meant for a {p.view.room.type} (catalog lists {', '.join(c.room_types)})"
    place, name = a.get("place", "wall"), a.get("name")
    if place == "wall":
        item, err = p.on_wall(c, a.get("wall", "A"), a.get("align", "center"), name, a.get("align_to"))
    elif place == "beside":
        return p.beside(c, a.get("ref"), a.get("side", "both"))
    elif place == "in_front_of":
        item, err = p.in_front_of(c, a.get("ref"), float(a.get("gap_m", 0.45)), name, a.get("face_ref"))
    elif place == "under":
        base = p.find(a.get("ref"))
        if base is None:
            return [], f"{c.name}: nothing called '{a.get('ref')}' to go under"
        item, err = p.middle(c, name, at=(base.x, base.y))
    elif place == "center":
        item, err = p.middle(c, name)
    elif place == "free":
        item, err = p.free_spot(c, name, float(a.get("min_clearance_m", 0.5)))
    else:
        return [], f"unknown place '{place}' (use wall, beside, in_front_of, center, free)"
    return ([item] if item else []), err


def apply_actions(plan: Plan, staging: Staging, actions_by_room: dict[str, list[dict]],
                  catalog: Optional[Catalog] = None) -> tuple[Staging, dict[str, list[str]]]:
    """Run the instructions room by room on a copy of the staging.
    Each instruction must leave the staging valid (staging_validator); otherwise it is undone.
    -> (new staging, {room id: [what failed, in words]})"""
    catalog = catalog or load_catalog()
    current = staging.model_copy(deep=True)
    errors: dict[str, list[str]] = {}
    known = {r.id for f in plan.floors for r in f.rooms}
    for room_id, actions in actions_by_room.items():
        if room_id not in known:
            errors.setdefault(room_id, []).append(f"room {room_id} is not in this plan")
            continue
        p = RoomPlacer(plan, current, room_id, catalog)
        for a in actions if isinstance(actions, list) else []:
            if not isinstance(a, dict):
                continue
            before = list(p.items)
            placed, err = run_action(p, a)
            if not err:
                trial = current.model_copy(update={"items": p.items})
                result = validate_staging(plan, trial, catalog)
                failed = [ck["name"] for ck in result["checks"] if not ck["passed"]]
                if failed:
                    p.items = before
                    err = f"{a.get('item', 'that change')} was undone: validator failed {', '.join(failed)}"
            if err:
                errors.setdefault(room_id, []).append(err)
        current = current.model_copy(update={"items": p.items})
    return current, errors
