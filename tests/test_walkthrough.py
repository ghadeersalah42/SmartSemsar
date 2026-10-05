import base64
import json
import re

import pytest
from shapely.geometry import Point, Polygon

from backend.schema.plan import load_plan
from backend.schema.staging import load_catalog
from backend.services.furniture_placer import stage_plan
from backend.services.walkthrough import (SPAWN_CLEAR_M, build_walkthrough,
                                          staging_payload, viewer_hints)


def test_spawn_is_inside_entry_room():
    plan = load_plan("data/plans/PROP_1002.json")
    hints = viewer_hints(plan)
    floor = hints["floors"][0]
    entries = [r for r in plan.floors[0].rooms if r.type == "entry"]
    assert any(Polygon(r.polygon).contains(Point(floor["spawn"])) for r in entries)
    assert hints["source_warning"].startswith("Representative layout")


def test_every_floor_gets_a_spawn_point():
    plan = load_plan("data/plans/PROP_1001.json")   # two-floor plan 4770
    hints = viewer_hints(plan)
    assert [f["level"] for f in hints["floors"]] == [0, 1]
    for f, fh in zip(plan.floors, hints["floors"]):
        assert any(Polygon(r.polygon).buffer(1e-6).contains(Point(fh["spawn"])) for r in f.rooms)


def test_html_is_self_contained(tmp_path):
    out = build_walkthrough("data/plans/PROP_1002.json", tmp_path / "w.html",
                            listing={"property_id": "PROP_1002", "title": "Test </script> title"})
    html = out.read_text(encoding="utf-8")
    assert "__PLAN_JSON__" not in html and "__GLB_B64__" not in html
    assert html.count("</script>") == html.count("<script")       # nothing broke out of a JSON block
    plan_json = re.search(r'id="plan-data">(.*?)</script>', html, re.S).group(1)
    assert json.loads(plan_json)["schema_version"] == 1


# ---------- furnished walkthrough ----------
def staging_json(html):
    return json.loads(re.search(r'id="staging-data">(.*?)</script>', html, re.S).group(1))


def test_unfurnished_page_has_no_staging(tmp_path):
    out = build_walkthrough("data/plans/PROP_1002.json", tmp_path / "w.html")
    html = out.read_text(encoding="utf-8")
    assert "__STAGING_JSON__" not in html
    assert staging_json(html) is None


def test_furnished_page_embeds_every_item(tmp_path):
    plan = load_plan("data/plans/PROP_1002.json")
    staging = stage_plan(plan)
    out = build_walkthrough("data/plans/PROP_1002.json", tmp_path / "w.html", staging=staging)
    embedded = staging_json(out.read_text(encoding="utf-8"))
    assert [i["id"] for i in embedded["items"]] == [i.id for i in staging.items]
    bed = next(i for i in embedded["items"] if i["catalog_id"] == "bed_double")
    # the viewer needs position, real size and the catalog's look
    assert {"x", "y", "rotation_deg", "level", "room_id", "width_m", "depth_m", "height_m"} <= bed.keys()
    assert bed["name"] == "Double bed" and len(bed["color"]) == 3


def test_furnished_page_embeds_each_model_once(tmp_path):
    plan = load_plan("data/plans/PROP_1002.json")
    staging = stage_plan(plan)
    out = build_walkthrough("data/plans/PROP_1002.json", tmp_path / "w.html", staging=staging)
    embedded = staging_json(out.read_text(encoding="utf-8"))
    used = {i.catalog_id for i in staging.items}
    assert set(embedded["models"]) == used                  # 6 chairs -> one chair model
    assert all(base64.b64decode(glb)[:4] == b"glTF" for glb in embedded["models"].values())


def test_item_without_a_model_file_stays_a_box():
    plan = load_plan("data/plans/PROP_1002.json")
    catalog = load_catalog()
    catalog.get("bed_double").mesh = "data/catalog/models/not_there.glb"
    catalog.get("nightstand").mesh = None
    payload = staging_payload(stage_plan(plan, catalog), catalog)
    assert "bed_double" not in payload["models"] and "nightstand" not in payload["models"]
    assert any(i["catalog_id"] == "bed_double" for i in payload["items"])   # still placed and drawn
    assert "sofa_3_seat" in payload["models"]


def test_rejected_staging_is_not_rendered(tmp_path):
    plan = load_plan("data/plans/PROP_1002.json")
    staging = stage_plan(plan)
    staging.items.append(staging.items[0].model_copy(update={"id": "copy"}))     # two items in one spot
    with pytest.raises(ValueError, match="no_overlaps"):
        build_walkthrough("data/plans/PROP_1002.json", tmp_path / "w.html", staging=staging)


@pytest.mark.parametrize("pid", ["PROP_1002", "PROP_1004", "PROP_1007"])
def test_visitor_does_not_start_inside_furniture(pid):
    plan = load_plan(f"data/plans/{pid}.json")
    staging = stage_plan(plan)
    for fh in viewer_hints(plan, staging=staging)["floors"]:
        spawn = Point(fh["spawn"])
        items = [i for i in staging.items if i.level == fh["level"]]
        assert all(Polygon(i.footprint()).distance(spawn) >= SPAWN_CLEAR_M - 1e-6 for i in items)
