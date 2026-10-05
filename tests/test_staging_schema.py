import json

import pytest
import trimesh
from shapely.geometry import Polygon

from backend.schema.staging import (REPO_ROOT, SCHEMA_VERSION, PlacedItem, Staging,
                                    load_catalog, load_staging, save_staging)


@pytest.fixture(scope="module")
def catalog():
    return load_catalog()


def bed(catalog, rotation_deg=0.0):
    return PlacedItem.from_catalog(catalog.get("bed_double"), id="i0", room_id="F0_R4", level=0,
                                   x=5.0, y=3.0, rotation_deg=rotation_deg)


def test_catalog_ids_are_unique_and_sizes_positive(catalog):
    ids = [i.id for i in catalog.items]
    assert len(ids) == len(set(ids))
    assert all(i.width_m > 0 and i.depth_m > 0 and i.height_m > 0 for i in catalog.items)


def test_models_match_the_catalog_size(catalog):
    # models are used as they are, so each must already be the item's real size:
    # centred on the origin, standing on the floor, width along x and depth along z
    for item in catalog.items:
        assert item.mesh, item.id
        lo, hi = trimesh.load(REPO_ROOT / item.mesh).bounds
        # catalog sizes are rounded to the centimetre
        assert (lo[0], hi[0]) == pytest.approx((-item.width_m / 2, item.width_m / 2), abs=6e-3), item.id
        assert (lo[2], hi[2]) == pytest.approx((-item.depth_m / 2, item.depth_m / 2), abs=6e-3), item.id
        assert lo[1] == pytest.approx(0.0, abs=1e-3), item.id
        assert hi[1] >= item.height_m - 6e-3, item.id       # extras (the TV) may rise above


def test_catalog_lookup(catalog):
    assert catalog.get("bed_double").width_m == 1.6
    with pytest.raises(KeyError):
        catalog.get("no_such_item")
    assert {i.category for i in catalog.for_room("bedroom")} >= {"bed", "storage"}
    assert not catalog.for_room("bathroom")     # bathrooms are covered by plan fixtures
    # items without styles fit any style
    assert catalog.for_room("living", style="modern") == catalog.for_room("living")


def test_placed_item_takes_real_size_from_catalog(catalog):
    item = bed(catalog)
    assert (item.width_m, item.depth_m, item.height_m) == (1.6, 2.0, 0.5)
    assert Polygon(item.footprint()).area == pytest.approx(3.2)


def test_footprint_at_zero_rotation(catalog):
    # centre (5, 3), width along x, back edge on +y
    assert bed(catalog).footprint() == [(4.2, 2.0), (5.8, 2.0), (5.8, 4.0), (4.2, 4.0)]


def test_rotation_is_counter_clockwise(catalog):
    # 90 degrees: width now runs along y, and the front faces +x
    fp = bed(catalog, 90).footprint()
    xs, ys = [p[0] for p in fp], [p[1] for p in fp]
    assert (min(xs), max(xs)) == pytest.approx((4.0, 6.0))
    assert (min(ys), max(ys)) == pytest.approx((2.2, 3.8))
    zone = bed(catalog, 90).front_zone(0.6)
    assert max(p[0] for p in zone) == pytest.approx(6.6)


def test_front_zone_touches_the_front_edge(catalog):
    item = bed(catalog)
    zone = Polygon(item.front_zone(0.6))
    assert zone.area == pytest.approx(1.6 * 0.6)
    assert zone.bounds == pytest.approx((4.2, 1.4, 5.8, 2.0))
    assert item.front_zone(0) == []


def test_round_trip(tmp_path, catalog):
    staging = Staging(staging_id="s1", plan_id="PROP_1002_cubicasa_9623", style="modern",
                      preferences={"budget": "mid"}, items=[bed(catalog)])
    path = save_staging(staging, tmp_path / "staging.json")
    assert load_staging(path) == staging
    assert staging.in_room("F0_R4") == staging.items and staging.in_room("F0_R0") == []


def test_rejects_unknown_schema_version(tmp_path):
    path = save_staging(Staging(staging_id="s1", plan_id="p"), tmp_path / "staging.json")
    data = json.loads(path.read_text())
    assert data["schema_version"] == SCHEMA_VERSION
    data["schema_version"] = 2
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        load_staging(path)
