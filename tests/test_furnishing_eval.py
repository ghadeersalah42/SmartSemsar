"""Round 2 furnishing / 3D metrics: the log, the summaries and the benchmark helpers."""
import json

from backend.eval import layout_benchmark, photo_benchmark
from backend.schema.plan import load_plan
from backend.schema.staging import REPO_ROOT, load_catalog
from backend.services import furnishing_metrics, multi_furnish

PLAN = REPO_ROOT / "data" / "plans" / "PROP_1002.json"


def test_metrics_off_under_pytest_unless_asked(tmp_path, monkeypatch):
    monkeypatch.delenv(furnishing_metrics.PATH_ENV, raising=False)
    assert furnishing_metrics.log_path() is None
    furnishing_metrics.record("model_3d", seconds=1.0)          # must not raise or write

    log = tmp_path / "m.jsonl"
    monkeypatch.setenv(furnishing_metrics.PATH_ENV, str(log))
    furnishing_metrics.record("model_3d", generator="colab", cache_hit=False, seconds=200.0, ok=True)
    assert json.loads(log.read_text(encoding="utf-8"))["seconds"] == 200.0


def test_summarize_3d_times_only_generated_pieces():
    events = [
        {"event": "model_3d", "generator": "colab", "cache_hit": False, "seconds": 100.0, "ok": True},
        {"event": "model_3d", "generator": "colab", "cache_hit": False, "seconds": 300.0, "ok": True},
        {"event": "model_3d", "generator": "colab", "cache_hit": True, "seconds": 0.0, "ok": True},
        {"event": "model_3d", "generator": "colab", "cache_hit": False, "seconds": 600.0, "ok": False},
        {"event": "layout", "approved": True},
    ]
    s = furnishing_metrics.summarize_3d(events)["colab"]
    assert s["calls"] == 4 and s["cache_hits"] == 1 and s["cache_hit_rate"] == 0.25
    assert s["generated"] == 2 and s["failed"] == 1
    assert s["seconds_mean"] == 200.0 and s["seconds_p95"] == 300.0


def test_raw_model_logs_miss_then_hit(tmp_path, monkeypatch):
    log = tmp_path / "m.jsonl"
    monkeypatch.setenv(furnishing_metrics.PATH_ENV, str(log))
    photo = tmp_path / "sofa.jpg"
    photo.write_bytes(b"not really a jpeg")
    calls = []

    class FakeGenerator:
        @staticmethod
        def generate_model(src, out):
            calls.append(src)
            out.write_bytes(b"glTF")

    monkeypatch.setitem(multi_furnish.GENERATORS, "fake", FakeGenerator)
    first = multi_furnish.raw_model(photo, "fake", tmp_path / "raw", say=lambda _: None)
    second = multi_furnish.raw_model(photo, "fake", tmp_path / "raw", say=lambda _: None)
    assert first == second and len(calls) == 1
    s = furnishing_metrics.summarize_3d(furnishing_metrics.load(log))["fake"]
    assert s["calls"] == 2 and s["cache_hits"] == 1 and s["generated"] == 1


def test_rejection_reasons_from_error_text():
    assert layout_benchmark.reasons_of("sofa_3_seat was undone: validator failed doors_clear, paths") == \
        ["doors_clear", "paths"]
    assert layout_benchmark.reasons_of("Sofa (2.2 m wide) does not fit on wall B (free wall: 1.1 m)") == \
        ["no_free_wall"]
    assert layout_benchmark.reasons_of("Bed on wall A blocks a door") == ["doors_clear"]
    assert layout_benchmark.reasons_of("'spaceship' is not in the catalog") == ["unknown_item"]
    assert layout_benchmark.reasons_of("something new") == ["other"]


def test_rules_layout_run_and_summary():
    r = layout_benchmark.run_rules(load_plan(PLAN), load_catalog(), {})
    assert r["final"] is True and r["items"] > 0
    s = layout_benchmark.summarize([r, {**r, "first_try": False, "reasons": ["paths", "paths"]},
                                    {**r, "first_try": None, "reasons": []}])
    assert s["layouts"] == 3 and s["scored"] == 2 and s["first_try_pass"] == (1 if r["first_try"] else 0)
    assert s["rejection_reasons"][0]["reason"] == "paths"


def test_photo_summary_by_condition():
    results = [
        {"expected_kinds": ["sofa"], "conditions": ["low_light", "clutter"], "success": True, "stage": "recognised",
         "vision_s": 3.0},
        {"expected_kinds": ["bed"], "conditions": ["low_light"], "success": False, "stage": "wrong_kind",
         "vision_s": 5.0},
        {"expected_kinds": [], "conditions": ["normal"], "success": True, "stage": "refused", "vision_s": 4.0},
    ]
    s = photo_benchmark.summarize(results)
    assert s["overall"] == {"photos": 3, "success": 2, "rate": 0.667}
    assert s["by_condition"]["low_light"]["rate"] == 0.5
    assert s["must_refuse"]["success"] == 1 and s["outcomes"]["wrong_kind"] == 1


def test_labelled_photo_set_is_well_formed():
    rows = photo_benchmark.read_labels()
    kinds = {i.kind for i in load_catalog().items}
    assert len(rows) >= 30
    for r in rows:
        assert set(photo_benchmark.split(r["expected_kinds"])) <= kinds, r
        assert photo_benchmark.split(r["conditions"]), r


def test_dark_photos_are_brightened_bright_ones_untouched():
    from PIL import Image, ImageStat
    from backend.services.vision_service import brighten
    dark = Image.linear_gradient("L").resize((40, 40)).point(lambda v: 20 + v * 60 // 255).convert("RGB")
    assert ImageStat.Stat(brighten(dark).convert("L")).mean[0] > ImageStat.Stat(dark.convert("L")).mean[0] + 30
    bright = Image.new("RGB", (40, 40), (180, 170, 160))
    assert brighten(bright) is bright


def test_phone_rotation_is_applied_before_the_model_sees_the_photo(tmp_path):
    import base64
    import io
    from PIL import Image
    from backend.services.vision_service import _image_part
    photo = tmp_path / "sideways.jpg"
    exif = Image.Exif()
    exif[0x0112] = 6                                   # "rotate 90" as phones write it
    Image.new("RGB", (200, 100), "white").save(photo, exif=exif)
    sent = Image.open(io.BytesIO(base64.b64decode(_image_part(photo)["inline_data"]["data"])))
    assert sent.size == (100, 200)


def test_room_description_counts_placed_pieces_and_windows():
    from backend.services.furniture_actions import describe_room
    from backend.services.furniture_placer import stage_plan
    plan = load_plan(PLAN)
    room = next(r for r in plan.rooms() if r.type == "bedroom")
    empty = describe_room(plan, room.id)
    furnished = describe_room(plan, room.id, stage_plan(plan))
    assert "tall pieces:" in empty                       # a window wall tells tall pieces apart
    assert empty != furnished.split("\n  placed:")[0]    # the furnished room has less free wall
    assert " 0.0 m" not in furnished and " 0.1 m" not in furnished
