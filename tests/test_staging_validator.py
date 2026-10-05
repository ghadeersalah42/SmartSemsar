import pytest
from shapely.geometry import Polygon

from backend.schema.plan import Door, Floor, Plan, Room, ScaleInfo, Wall, Window, load_plan
from backend.schema.staging import PlacedItem, Staging, load_catalog
from backend.services.furniture_placer import RoomSpace, keep_paths_open, stage_plan
from backend.services.staging_validator import cut_off_doors, validate_staging


@pytest.fixture(scope="module")
def catalog():
    return load_catalog()


def box_plan():
    """One 4 x 3 m bedroom, a door in the left wall and one in the right wall, a window on top."""
    t = 0.2
    walls = [
        Wall(id="W0", polygon=[(-t, -t), (4 + t, -t), (4 + t, 0), (-t, 0)], thickness_m=t),        # bottom
        Wall(id="W1", polygon=[(-t, 3), (4 + t, 3), (4 + t, 3 + t), (-t, 3 + t)], thickness_m=t),  # top
        Wall(id="W2", polygon=[(-t, 0), (0, 0), (0, 3), (-t, 3)], thickness_m=t),                  # left
        Wall(id="W3", polygon=[(4, 0), (4 + t, 0), (4 + t, 3), (4, 3)], thickness_m=t),            # right
    ]
    doors = [
        Door(id="D_left", wall_id="W2", polygon=[(-t, 1.0), (0, 1.0), (0, 1.9), (-t, 1.9)], width_m=0.9),
        Door(id="D_right", wall_id="W3", polygon=[(4, 1.0), (4 + t, 1.0), (4 + t, 1.9), (4, 1.9)], width_m=0.9),
    ]
    windows = [Window(id="O0", wall_id="W1", polygon=[(1.0, 3), (2.0, 3), (2.0, 3 + t), (1.0, 3 + t)], width_m=1.0)]
    room = Room(id="R0", type="bedroom", polygon=[(0, 0), (4, 0), (4, 3), (0, 3)], area_sqm=12.0)
    return Plan(plan_id="box", source="generated", scale=ScaleInfo(method="native"),
                floors=[Floor(level=0, rooms=[room], walls=walls, doors=doors, windows=windows)],
                total_area_sqm=12.0)


def put(catalog, catalog_id, item_id, x, y, rotation_deg=0.0):
    return PlacedItem.from_catalog(catalog.get(catalog_id), id=item_id, room_id="R0", level=0,
                                   x=x, y=y, rotation_deg=rotation_deg)


def run(catalog, *items):
    result = validate_staging(box_plan(), Staging(staging_id="s", plan_id="box", items=list(items)), catalog)
    return result, {c["name"]: c for c in result["checks"]}


@pytest.mark.parametrize("pid", ["PROP_1002", "PROP_1004", "PROP_1007"])
def test_placer_output_is_approved(pid, catalog):
    plan = load_plan(f"data/plans/{pid}.json")
    result = validate_staging(plan, stage_plan(plan, catalog), catalog)
    assert result["approved"], [c for c in result["checks"] if not c["passed"]]


def test_good_staging_passes(catalog):
    result, checks = run(catalog, put(catalog, "bed_double", "a", 2.5, 1.0, 180))    # back on the bottom wall
    assert result["approved"] and all(c["passed"] for c in checks.values())


def test_overlap_is_reported_as_a_pair(catalog):
    result, checks = run(catalog, put(catalog, "nightstand", "a", 2.0, 0.3), put(catalog, "nightstand", "b", 2.2, 0.3))
    assert not result["approved"]
    assert checks["no_overlaps"]["actual"] == ["a|b"]


def test_touching_items_are_not_an_overlap(catalog):
    side_by_side = 2.0 + catalog.get("nightstand").width_m
    _, checks = run(catalog, put(catalog, "nightstand", "a", 2.0, 0.3), put(catalog, "nightstand", "b", side_by_side, 0.3))
    assert checks["no_overlaps"]["passed"]


def test_item_through_a_wall_fails(catalog):
    _, checks = run(catalog, put(catalog, "nightstand", "a", 2.0, 0.05))     # sticks 15 cm into the bottom wall
    assert checks["inside_room"]["actual"] == ["a"]


def test_item_at_a_door_fails(catalog):
    _, checks = run(catalog, put(catalog, "nightstand", "a", 0.3, 1.45))
    assert checks["doors_clear"]["actual"] == ["a"]
    assert checks["inside_room"]["passed"]


def test_scaled_item_fails(catalog):
    item = put(catalog, "nightstand", "a", 2.0, 0.3)
    item.width_m *= 1.85                                    # stretched with a look-alike plan
    _, checks = run(catalog, item)
    assert checks["real_size"]["actual"] == ["a"]


def test_bad_references_fail(catalog):
    ghost_room = put(catalog, "nightstand", "a", 2.0, 0.3)
    ghost_room.room_id = "R9"
    ghost_item = put(catalog, "nightstand", "b", 3.0, 0.3)
    ghost_item.catalog_id = "hot_tub"
    twin1, twin2 = put(catalog, "nightstand", "c", 1.0, 0.3), put(catalog, "nightstand", "c", 3.5, 2.6)
    result = validate_staging(box_plan(), Staging(staging_id="s", plan_id="other_plan",
                                                  items=[ghost_room, ghost_item, twin1, twin2]), catalog)
    refs = next(c for c in result["checks"] if c["name"] == "references")
    assert not result["approved"]
    assert set(refs["actual"]) == {"plan_id:other_plan", "a", "b", "c"}


def test_item_on_a_fixture_fails(catalog):
    plan = load_plan("data/plans/PROP_1002.json")
    floor = plan.floors[0]
    fixture = next(x for x in floor.fixtures if x.on_floor and x.room_id)
    spot = Polygon(fixture.polygon).buffer(0).representative_point()
    item = PlacedItem.from_catalog(catalog.get("nightstand"), id="a", room_id=fixture.room_id, level=0,
                                   x=spot.x, y=spot.y)
    result = validate_staging(plan, Staging(staging_id="s", plan_id=plan.plan_id, items=[item]), catalog)
    assert "a" in next(c for c in result["checks"] if c["name"] == "fixtures_clear")["actual"]


def test_tall_item_at_a_window_is_only_a_warning(catalog):
    result, checks = run(catalog, put(catalog, "wardrobe", "a", 1.5, 2.68))  # back on the top wall, under the window
    assert result["approved"]
    assert any("window" in w for w in result["warnings"])


# ---------- walkable paths ----------
def barrier(catalog, n):
    """Wardrobes turned sideways in a line across the middle of the room (1.2 m each)."""
    return [put(catalog, "wardrobe", f"w{k}", 2.0, 0.7 + 1.2 * k, 90) for k in range(n)]


def test_one_wardrobe_leaves_a_path(catalog):
    result, checks = run(catalog, *barrier(catalog, 1))
    assert checks["paths"]["passed"] and result["approved"]


def test_wall_of_wardrobes_cuts_the_room_in_two(catalog):
    # 2.4 m of wardrobes in a 3 m room leaves gaps of 0.1 and 0.5 m: a person cannot pass
    result, checks = run(catalog, *barrier(catalog, 2))
    assert not result["approved"]
    assert checks["paths"]["actual"] == ["R0:D_left", "R0:D_right"]
    assert checks["no_overlaps"]["passed"] and checks["doors_clear"]["passed"]     # nothing else is wrong


def test_narrow_room_is_not_blamed_on_the_staging():
    # a 0.5 m corridor cannot be walked even when empty, so its doors are not required
    plan = box_plan()
    floor = plan.floors[0]
    floor.rooms[0].polygon = [(0, 1.2), (4, 1.2), (4, 1.7), (0, 1.7)]
    far_corner = Polygon([(3.0, 1.2), (3.4, 1.2), (3.4, 1.4), (3.0, 1.4)])
    assert cut_off_doors(floor, floor.rooms[0], [far_corner]) == []


def test_placer_drops_items_that_block_a_door(catalog):
    plan = box_plan()
    space = RoomSpace(plan.floors[0], plan.floors[0].rooms[0])
    space.items = [put(catalog, "nightstand", "keep", 0.3, 0.3)] + barrier(catalog, 2)
    assert keep_paths_open(space) == 1                      # dropping the last wardrobe is enough
    assert [i.id for i in space.items] == ["keep", "w0"]
