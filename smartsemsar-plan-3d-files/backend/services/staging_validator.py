"""
Deterministic staging validator (no LLM).
بيراجع الـ staging على المخطط: كل قطعة جوه أوضتها، مفيش تداخل، الأبواب فاضية، وفيه ممر للمشي.

    result = validate_staging(plan, staging)
    -> {approved, checks[], warnings[]}
Each failed check lists the item / door ids at fault in `actual`.
If approved is False the staging must not be shown; the placer drops items and tries again.
"""
import sys
from typing import Optional

from pydantic import BaseModel, Field
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union

from backend.schema.plan import Floor, Plan, Room, load_plan
from backend.schema.staging import REPO_ROOT, Catalog, Staging, load_catalog, load_staging
from backend.services.plan_validator import Check

EPS_AREA = 1e-3             # overlaps smaller than this are just touching edges
ROOM_TOLERANCE_M = 0.02
DOOR_CLEAR_M = 0.6          # minimum free depth on both sides of a door
PATH_WIDTH_M = 0.6          # a person must be able to walk through
DOOR_REACH_M = 0.15         # how far past the path half-width a door opening counts as reached
WINDOW_SILL_M = 0.9
SIZE_TOLERANCE_M = 0.005


class StagingValidation(BaseModel):
    approved: bool
    checks: list[Check] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


def _poly(points) -> Polygon:
    return Polygon(points).buffer(0)


def _hits(a, b) -> bool:
    return not b.is_empty and a.intersects(b) and a.intersection(b).area > EPS_AREA


def _parts(geom) -> list[Polygon]:
    if isinstance(geom, Polygon):
        return [] if geom.is_empty else [geom]
    if isinstance(geom, MultiPolygon):
        return list(geom.geoms)
    return [g for g in getattr(geom, "geoms", []) if isinstance(g, Polygon) and not g.is_empty]


def room_doors(floor: Floor, room: Room) -> list[tuple[str, Polygon]]:
    """Doors whose opening touches this room."""
    room_poly = _poly(room.polygon)
    doors = [(d.id, _poly(d.polygon)) for d in floor.doors]
    return [(i, g) for i, g in doors if g.distance(room_poly) < 0.05]


def cut_off_doors(floor: Floor, room: Room, footprints: list[Polygon]) -> list[str]:
    """Doors of the room that furniture has cut off (ids, sorted).

    The empty room is the reference: a door counts only if a person could reach it
    before the furniture went in, and two doors must stay connected only if they
    were connected then. A corridor that is too narrow anyway is not the staging's fault.
    """
    doors = room_doors(floor, room)
    if not doors or not footprints:
        return []
    fixed = unary_union([_poly(w.polygon) for w in floor.walls] +
                        [_poly(x.polygon) for x in floor.fixtures if x.on_floor])
    room_free = _poly(room.polygon).difference(fixed)
    reach = PATH_WIDTH_M / 2 + DOOR_REACH_M

    def reached(free) -> dict[str, set[int]]:
        # shrink by half a body width: what is left is where a person's centre can be
        areas = _parts(free.buffer(-PATH_WIDTH_M / 2))
        return {i: {k for k, a in enumerate(areas) if a.intersects(g.buffer(reach))} for i, g in doors}

    empty = reached(room_free)
    staged = reached(room_free.difference(unary_union(footprints)))
    bad = {i for i, _ in doors if empty[i] and not staged[i]}
    ids = [i for i, _ in doors]
    for n, a in enumerate(ids):
        for b in ids[n + 1:]:
            if empty[a] & empty[b] and staged[a] and staged[b] and not staged[a] & staged[b]:
                bad |= {a, b}
    return sorted(bad)


def validate_staging(plan: Plan, staging: Staging, catalog: Optional[Catalog] = None) -> dict:
    catalog = catalog or load_catalog()
    warnings: list[str] = []
    known_ids = {c.id: c for c in catalog.items}
    rooms = {r.id: (f, r) for f in plan.floors for r in f.rooms}

    # 1) references: plan, room, level, catalog item
    bad_refs = [] if staging.plan_id == plan.plan_id else [f"plan_id:{staging.plan_id}"]
    duplicates = {i.id for i in staging.items if sum(j.id == i.id for j in staging.items) > 1}
    bad_refs += sorted(duplicates)
    items = []
    for i in staging.items:
        if i.room_id not in rooms or rooms[i.room_id][0].level != i.level or i.catalog_id not in known_ids:
            bad_refs.append(i.id)
        else:
            items.append(i)     # only items we can locate go through the geometry checks

    # 2) real size: staged sizes must be the catalog's, never scaled with the plan
    wrong_size = [i.id for i in items
                  if any(abs(a - b) > SIZE_TOLERANCE_M for a, b in zip(
                      (i.width_m, i.depth_m, i.height_m),
                      (known_ids[i.catalog_id].width_m, known_ids[i.catalog_id].depth_m,
                       known_ids[i.catalog_id].height_m)))]
    wrong_room_type = [i.id for i in items
                       if rooms[i.room_id][1].type not in known_ids[i.catalog_id].room_types]
    if wrong_room_type:
        warnings.append(f"Items in a room type the catalog does not list for them: {', '.join(wrong_room_type)}.")

    outside, on_fixture, at_door, overlaps, blocked_paths, at_window = [], [], [], [], [], []
    fp = {i.id: _poly(i.footprint()) for i in items}

    for floor in plan.floors:
        walls = unary_union([_poly(w.polygon) for w in floor.walls])
        fixtures = unary_union([_poly(x.polygon) for x in floor.fixtures if x.on_floor])
        windows = unary_union([_poly(w.polygon).buffer(0.2) for w in floor.windows])
        on_level = [i for i in items if i.level == floor.level]

        for room in floor.rooms:
            in_room = [i for i in on_level if i.room_id == room.id]
            if not in_room:
                continue
            room_poly = _poly(room.polygon).buffer(ROOM_TOLERANCE_M)
            door_zone = unary_union([g.buffer(DOOR_CLEAR_M) for _, g in room_doors(floor, room)])
            for i in in_room:
                # 3) inside the room and not in a wall
                if not room_poly.contains(fp[i.id]) or _hits(fp[i.id], walls):
                    outside.append(i.id)
                # 4) not on a toilet / cabinet / staircase / column
                if _hits(fp[i.id], fixtures):
                    on_fixture.append(i.id)
                # 5) doors can open and be walked through
                if _hits(fp[i.id], door_zone):
                    at_door.append(i.id)
                if i.height_m > WINDOW_SILL_M and _hits(fp[i.id], windows):
                    at_window.append(i.id)
            # 7) every door still reachable on foot
            blocked_paths += [f"{room.id}:{d}" for d in cut_off_doors(floor, room, [fp[i.id] for i in in_room])]

        # 6) no two items overlap (across rooms too, for open-plan spaces)
        for n, a in enumerate(on_level):
            for b in on_level[n + 1:]:
                if _hits(fp[a.id], fp[b.id]):
                    overlaps.append(f"{a.id}|{b.id}")

    if at_window:
        warnings.append(f"Tall items in front of a window: {', '.join(at_window)}.")
    if not staging.items:
        warnings.append("Staging has no items.")
    if staging.unplaced:
        warnings.append(f"Requested but not placed (no room for it): {', '.join(staging.unplaced)}.")

    def check(name: str, bad: list[str], message: str) -> Check:
        return Check(name=name, passed=not bad, expected=[], actual=bad,
                     message="" if not bad else f"{len(bad)} {message}")

    checks = [
        check("references", bad_refs, "bad reference(s): unknown plan, room, level, catalog item or repeated id."),
        check("real_size", wrong_size, "item(s) whose size differs from the catalog."),
        check("inside_room", outside, "item(s) outside their room or inside a wall."),
        check("fixtures_clear", on_fixture, "item(s) standing on a built-in fixture, stairs or a column."),
        check("doors_clear", at_door, f"item(s) within {DOOR_CLEAR_M} m of a door."),
        check("no_overlaps", overlaps, "overlapping pair(s) of items."),
        check("paths", blocked_paths, f"door(s) no longer reachable through a {PATH_WIDTH_M} m wide path."),
    ]
    return StagingValidation(approved=all(c.passed for c in checks), checks=checks,
                             warnings=warnings).model_dump()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python -m backend.services.staging_validator <property_id>")
    pid = sys.argv[1]
    result = validate_staging(load_plan(REPO_ROOT / "data" / "plans" / f"{pid}.json"),
                              load_staging(REPO_ROOT / "data" / "stagings" / f"{pid}.json"))
    print("APPROVED" if result["approved"] else "REJECTED")
    for c in result["checks"]:
        print(f"  {'ok  ' if c['passed'] else 'FAIL'} {c['name']}  {c['message']} {c['actual'] or ''}")
    for w in result["warnings"]:
        print(f"  warning: {w}")
