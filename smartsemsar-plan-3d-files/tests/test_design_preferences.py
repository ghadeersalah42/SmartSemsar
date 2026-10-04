from collections import Counter

import pytest

from backend.schema.design import DesignPreferences
from backend.schema.plan import load_plan
from backend.schema.staging import load_catalog
from backend.services.furniture_placer import Wishes, stage_plan
from backend.services.staging_validator import validate_staging

FLAT, STUDIO, VILLA = "PROP_1002", "PROP_1004", "PROP_1007"


@pytest.fixture(scope="module")
def catalog():
    return load_catalog()


def staged(pid, catalog, **prefs):
    plan = load_plan(f"data/plans/{pid}.json")
    staging = stage_plan(plan, catalog, preferences=prefs)
    assert validate_staging(plan, staging, catalog)["approved"], prefs      # preferences never break the layout
    return staging, Counter(i.catalog_id for i in staging.items)


# ---------- the schema ----------
def test_defaults_ask_for_nothing_special():
    prefs = DesignPreferences()
    assert (prefs.style, prefs.density, prefs.dining_seats) == ("unknown", "normal", None)
    assert prefs.must_have == [] and prefs.exclude == []


def test_llm_output_with_unusable_values_falls_back():
    # "cozy" is not a style and 5 is not a table size: both are dropped and reported, the rest is kept
    prefs = DesignPreferences.from_loose({"style": "cozy", "density": "full", "dining_seats": 5,
                                          "must_have": ["desk"], "style_brief": "warm and calm", "junk": 1})
    assert prefs.style == "unknown" and prefs.dining_seats is None
    assert prefs.density == "full" and prefs.must_have == ["desk"] and prefs.style_brief == "warm and calm"
    assert prefs.missing_fields == ["dining_seats", "style"]


def test_catalog_vocabulary(catalog):
    words = catalog.vocabulary()
    assert {"tv", "desk", "sofa", "dining_table", "bed_double", "storage"} <= words
    prefs = DesignPreferences(must_have=["desk", "piano"], exclude=["tv", "jacuzzi"])
    assert prefs.unknown_names(words) == ["jacuzzi", "piano"]


def test_every_catalog_item_has_a_kind(catalog):
    assert all(i.kind for i in catalog.items)
    assert catalog.get("tv_unit").is_a(["tv"]) and catalog.get("tv_unit").is_a(["storage"])
    assert not catalog.get("wardrobe").is_a(["tv"])


def test_wishes_levels(catalog):
    desk, bed = catalog.get("desk"), catalog.get("bed_double")
    assert Wishes().wants(desk) and not Wishes().wants(desk, when=False)
    assert not Wishes().wants(desk, "full")                                         # extras need density=full
    assert not Wishes(DesignPreferences(density="minimal")).wants(desk)
    assert Wishes(DesignPreferences(density="minimal")).wants(bed, "essential")
    assert Wishes(DesignPreferences(density="full")).wants(desk, when=False)        # room size no longer matters
    assert Wishes(DesignPreferences(density="minimal", must_have=["desk"])).wants(desk, when=False)
    assert not Wishes().wants(None)


# ---------- density ----------
@pytest.mark.parametrize("pid", [FLAT, VILLA])
def test_density_only_adds_furniture(pid, catalog):
    _, minimal = staged(pid, catalog, density="minimal")
    _, normal = staged(pid, catalog)
    _, full = staged(pid, catalog, density="full")
    assert sum(minimal.values()) < sum(normal.values()) <= sum(full.values())
    assert all(normal[k] >= n for k, n in minimal.items())      # nothing essential disappears


def test_minimal_keeps_only_the_essentials(catalog):
    _, items = staged(VILLA, catalog, density="minimal")
    assert items["bed_double"] == 5 and items["sofa_3_seat"] == 1 and items["dining_table_6"] == 1
    assert not set(items) & {"nightstand", "armchair", "bookshelf", "dresser", "desk", "console_table", "shoe_cabinet"}


def test_full_fills_every_bedroom(catalog):
    _, normal = staged(VILLA, catalog)
    _, full = staged(VILLA, catalog, density="full")
    assert full["dresser"] > normal["dresser"] and full["bookshelf"] > normal["bookshelf"]
    assert full["desk"] >= normal["desk"]


# ---------- must_have / exclude ----------
def test_must_have_adds_the_item_where_the_rules_would_not(catalog):
    _, normal = staged(VILLA, catalog)
    staging, forced = staged(VILLA, catalog, must_have=["desk"])
    assert forced["desk"] == normal["desk"] + 1                 # the master bedroom gets one too
    assert staging.unplaced == []


def test_must_have_beats_minimal(catalog):
    _, items = staged(FLAT, catalog, density="minimal", must_have=["nightstand"])
    assert items["nightstand"] == 4 and "bookshelf" not in items


def test_must_have_that_does_not_fit_is_reported(catalog):
    staging, items = staged(STUDIO, catalog, must_have=["desk"])
    assert "desk" not in items and staging.unplaced == ["desk"]
    plan = load_plan(f"data/plans/{STUDIO}.json")
    assert any("desk" in w for w in validate_staging(plan, staging, catalog)["warnings"])


def test_unknown_must_have_is_reported_not_invented(catalog):
    staging, _ = staged(FLAT, catalog, must_have=["piano"])
    assert staging.unplaced == ["piano"]


def test_exclude_by_kind_leaves_the_rest_of_the_category(catalog):
    _, items = staged(FLAT, catalog, exclude=["tv"])
    assert "tv_unit" not in items
    assert items["wardrobe"] and items["nightstand"]            # also "storage", but not excluded


def test_exclude_wins_over_must_have(catalog):
    staging, items = staged(FLAT, catalog, exclude=["tv"], must_have=["tv"])
    assert "tv_unit" not in items and staging.unplaced == ["tv"]


# ---------- dining seats ----------
def test_dining_seats_overrides_the_room_size_rule(catalog):
    _, default = staged(FLAT, catalog)
    _, four = staged(FLAT, catalog, dining_seats=4)
    assert default["dining_table_6"] == 1 and default["dining_chair"] == 6
    assert four["dining_table_4"] == 1 and four["dining_chair"] == 4 and "dining_table_6" not in four


def test_dining_size_that_does_not_fit_is_reported(catalog):
    staging, items = staged(STUDIO, catalog, dining_seats=6)
    assert "dining_table_6" not in items and staging.unplaced == ["dining_table_6"]


# ---------- how preferences travel ----------
def test_object_and_dict_give_the_same_staging(catalog):
    plan = load_plan(f"data/plans/{FLAT}.json")
    as_dict = stage_plan(plan, catalog, preferences={"density": "full", "exclude": ["tv"]})
    as_object = stage_plan(plan, catalog, preferences=DesignPreferences(density="full", exclude=["tv"]))
    assert as_dict.items == as_object.items
    assert as_dict.preferences == {"density": "full", "exclude": ["tv"]}      # only what was asked is stored


def test_style_from_preferences_names_the_staging(catalog):
    plan = load_plan(f"data/plans/{STUDIO}.json")
    staging = stage_plan(plan, catalog, preferences={"style": "modern", "style_brief": "calm, light wood"})
    assert staging.style == "modern" and staging.staging_id.endswith("_modern")
    assert staging.preferences["style_brief"] == "calm, light wood"
    assert stage_plan(plan, catalog, preferences={"style": "unknown"}).style is None
