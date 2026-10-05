import pytest
from shapely.geometry import Polygon

from backend.schema.plan import load_plan
from data.scripts.extract_cubecasa import (_apply, _compose, build_plan,
                                           fixture_spec_from_class,
                                           parse_matrix, parse_svg)


def native_plan(cubicasa_id):
    return build_plan(parse_svg(f"data/cubicasa_svg/{cubicasa_id}.svg"), "t", source="user_upload")


def test_parse_matrix():
    assert parse_matrix("matrix(0,1,-1,0,231.2,282.4)") == (0, 1, -1, 0, 231.2, 282.4)
    assert parse_matrix("translate(3 12)") == (1, 0, 0, 1, 0, 0)
    assert parse_matrix(None) == (1, 0, 0, 1, 0, 0)


def test_compose_applies_child_first():
    move = (1, 0, 0, 1, 10, 0)
    turn = (0, 1, -1, 0, 0, 0)          # 90 degrees
    # turn then move: (1, 0) -> (0, 1) -> (10, 1)
    assert _apply(_compose(move, turn), [(1, 0)]) == [(10, 1)]


def test_fixture_class_mapping():
    assert fixture_spec_from_class("FixedFurniture Toilet")[0] == "toilet"
    assert fixture_spec_from_class("FixedFurniture ShowerScreen")[0] == "other"   # not "shower"
    assert fixture_spec_from_class("FixedFurniture ElectricalAppliance Refrigerator")[0] == "appliance"
    assert fixture_spec_from_class("FixedFurniture Misc")[0] == "other"


def test_wall_cabinets_do_not_stand_on_the_floor():
    fixtures = native_plan(10718).fixtures()
    wall = [x for x in fixtures if x.type == "wall_cabinet"]
    assert wall and not any(x.on_floor for x in wall)
    assert all(x.on_floor for x in fixtures if x.type != "wall_cabinet")


def test_toilets_land_in_bathrooms():
    # the fixture transform is right only if every toilet ends up inside a bathroom
    for cid in (12803, 4770, 10718):
        plan = native_plan(cid)
        room_type = {r.id: r.type for r in plan.rooms()}
        toilets = [x for x in plan.fixtures() if x.type == "toilet"]
        assert toilets
        assert all(room_type.get(x.room_id) == "bathroom" for x in toilets)


def test_native_fixture_sizes_are_realistic():
    for x in native_plan(12803).fixtures():
        if x.type == "toilet":
            assert 0.15 < Polygon(x.polygon).area < 0.6     # ~0.4 x 0.7 m


def test_stairs_kept_on_multi_floor_plans():
    plan = native_plan(4770)
    assert plan.num_floors == 2
    assert any(x.type == "stairs" for x in plan.fixtures())


def test_fixtures_scale_with_the_plan():
    raw = parse_svg("data/cubicasa_svg/12803.svg")
    small = build_plan(raw, "t", target_area_sqm=60.0)
    big = build_plan(raw, "t", target_area_sqm=240.0)
    a = Polygon(small.fixtures()[0].polygon).area
    b = Polygon(big.fixtures()[0].polygon).area
    assert b / a == pytest.approx(4.0, rel=0.02)


def test_saved_plans_carry_fixtures():
    plan = load_plan("data/plans/PROP_1001.json")
    room_ids = {r.id for r in plan.rooms()}
    assert plan.fixtures()
    assert all(x.room_id is None or x.room_id in room_ids for x in plan.fixtures())
