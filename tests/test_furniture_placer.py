import pytest
from shapely.geometry import Polygon

from backend.schema.plan import load_plan
from backend.schema.staging import load_catalog
from backend.services.furniture_placer import DOOR_CLEAR_M, _facing, stage_plan

PLANS = ["PROP_1002", "PROP_1004", "PROP_1007"]     # flat, studio, 2-floor villa


@pytest.fixture(scope="module")
def catalog():
    return load_catalog()


@pytest.fixture(scope="module", params=PLANS)
def staged(request, catalog):
    plan = load_plan(f"data/plans/{request.param}.json")
    return plan, stage_plan(plan, catalog)


def footprints(staging):
    return [(i, Polygon(i.footprint())) for i in staging.items]


def test_items_reference_the_plan_and_catalog(staged, catalog):
    plan, staging = staged
    assert staging.plan_id == plan.plan_id
    level_of = {r.id: f.level for f in plan.floors for r in f.rooms}
    ids = [i.id for i in staging.items]
    assert len(ids) == len(set(ids))
    for i in staging.items:
        c = catalog.get(i.catalog_id)
        assert level_of[i.room_id] == i.level
        assert (i.width_m, i.depth_m, i.height_m) == (c.width_m, c.depth_m, c.height_m)   # real size, not scaled


def test_items_stay_inside_their_room(staged):
    plan, staging = staged
    room = {r.id: Polygon(r.polygon).buffer(0.02) for r in plan.rooms()}
    assert all(room[i.room_id].contains(fp) for i, fp in footprints(staging))


def test_items_go_only_in_allowed_room_types(staged, catalog):
    plan, staging = staged
    room_type = {r.id: r.type for r in plan.rooms()}
    assert all(room_type[i.room_id] in catalog.get(i.catalog_id).room_types for i in staging.items)


def test_no_overlaps(staged):
    _, staging = staged
    fps = footprints(staging)
    for a in range(len(fps)):
        for b in range(a + 1, len(fps)):
            if fps[a][0].level == fps[b][0].level:
                assert fps[a][1].intersection(fps[b][1]).area < 1e-3, (fps[a][0].id, fps[b][0].id)


def test_fixtures_and_doors_stay_clear(staged):
    plan, staging = staged
    for floor in plan.floors:
        # (shape, keep-clear depth): fixtures themselves, doors plus the minimum depth on both sides
        blocked = [(Polygon(x.polygon).buffer(0), 0.0) for x in floor.fixtures if x.on_floor]
        blocked += [(Polygon(d.polygon).buffer(0), DOOR_CLEAR_M[0] - 0.05) for d in floor.doors]
        rooms = {r.id: Polygon(r.polygon).buffer(0) for r in floor.rooms}
        for i, fp in footprints(staging):
            if i.level != floor.level:
                continue
            for shape, depth in blocked:
                if shape.distance(rooms[i.room_id]) < 0.05:     # only what touches the item's room
                    assert fp.intersection(shape.buffer(depth)).area < 1e-3, i.id


def test_tall_items_do_not_cover_windows(staged, catalog):
    plan, staging = staged
    for floor in plan.floors:
        windows = [Polygon(w.polygon).buffer(0).buffer(0.2) for w in floor.windows]
        for i, fp in footprints(staging):
            if i.level == floor.level and i.height_m > 0.9:
                assert not any(fp.intersection(w).area > 1e-3 for w in windows), i.id


def test_tv_faces_the_sofa(staged):
    _, staging = staged
    for tv in (i for i in staging.items if i.catalog_id == "tv_unit"):
        sofa = next(i for i in staging.in_room(tv.room_id) if i.catalog_id.startswith("sofa"))
        (sx, sy), (tx, ty) = _facing(sofa), _facing(tv)
        assert sx * tx + sy * ty < -0.9                                     # opposite directions
        assert (tv.x - sofa.x) * sx + (tv.y - sofa.y) * sy > 1.5            # TV is in front of the sofa


def test_every_plan_gets_the_basics(staged):
    plan, staging = staged
    placed = [i.catalog_id for i in staging.items]
    assert any(c.startswith("bed_") for c in placed)
    assert any(c.startswith("sofa") for c in placed)
    master = max(plan.rooms("bedroom"), key=lambda r: r.area_sqm)
    assert any(i.catalog_id.startswith("bed_") for i in staging.in_room(master.id))


def test_bathrooms_and_outdoor_stay_empty(staged):
    plan, staging = staged
    room_type = {r.id: r.type for r in plan.rooms()}
    assert not any(room_type[i.room_id] in ("bathroom", "outdoor", "garage", "other") for i in staging.items)


def test_placement_is_deterministic(catalog):
    plan = load_plan("data/plans/PROP_1002.json")
    assert stage_plan(plan, catalog) == stage_plan(plan, catalog)


def test_exclude_preference(catalog):
    plan = load_plan("data/plans/PROP_1002.json")
    staging = stage_plan(plan, catalog, preferences={"exclude": ["tv_unit", "chair"]})
    placed = {i.catalog_id for i in staging.items}
    assert "tv_unit" not in placed and "dining_chair" not in placed     # by id and by category
    assert any(c.startswith("sofa") for c in placed)
    assert staging.preferences == {"exclude": ["tv_unit", "chair"]}


def test_unknown_style_uses_unstyled_items(catalog):
    # catalog items without styles fit any style
    plan = load_plan("data/plans/PROP_1004.json")
    staging = stage_plan(plan, catalog, style="modern")
    assert staging.style == "modern" and staging.staging_id.endswith("_modern")
    assert staging.items == stage_plan(plan, catalog).items
