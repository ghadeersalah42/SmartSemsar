"""LLM furnishing: instructions -> positions, feedback loop, fallbacks (scripted fake model)."""
import pytest
from shapely.geometry import Point

from backend.schema.plan import load_plan
from backend.schema.staging import Staging, load_catalog
from backend.services.furniture_actions import RoomView, apply_actions, describe_room
from backend.services.llm_furnisher import furnish
from backend.services.staging_validator import validate_staging

PLAN = "data/plans/PROP_1002.json"


@pytest.fixture(scope="module")
def plan():
    return load_plan(PLAN)


@pytest.fixture(scope="module")
def catalog():
    return load_catalog()


def _room(plan, rtype, min_area=0):
    return next(r for r in plan.rooms() if r.type == rtype and r.area_sqm > min_area)


def _plain_wall(plan, room_id):
    v = RoomView(plan, room_id)
    return next(w.label for w in v.walls if not (w.is_open or w.doors or w.windows) and w.length > 2.5)


def test_room_description_lists_walls_doors_and_open_sides(plan):
    text = describe_room(plan, _room(plan, "living").id)
    assert "wall A" in text and "door at" in text and "window at" in text
    assert "open to the next room" in text                       # living room open to the kitchen


def test_instructions_become_valid_positions(plan, catalog):
    bed, liv = _room(plan, "bedroom", 20), _room(plan, "living")
    empty = Staging(staging_id="t", plan_id=plan.plan_id)
    st, errors = apply_actions(plan, empty, {
        bed.id: [{"item": "bed_double", "place": "wall", "wall": _plain_wall(plan, bed.id), "name": "bed"},
                 {"item": "nightstand", "place": "beside", "ref": "bed"}],
        liv.id: [{"item": "sofa_3_seat", "place": "wall", "wall": "B", "name": "sofa"},
                 {"item": "coffee_table", "place": "in_front_of", "ref": "sofa"}]}, catalog)
    assert not errors
    assert [i.catalog_id for i in st.in_room(bed.id)] == ["bed_double", "nightstand", "nightstand"]
    assert validate_staging(plan, st, catalog)["approved"]
    # the bed's back is against the wall: it faces into the room
    b = st.in_room(bed.id)[0]
    v = RoomView(plan, bed.id)
    assert v.poly.contains(Point(b.to_plan(0, -b.depth_m)))      # one depth in front: inside the room


def test_bad_instructions_are_explained_and_change_nothing(plan, catalog):
    bed = _room(plan, "bedroom", 20)
    empty = Staging(staging_id="t", plan_id=plan.plan_id)
    st, errors = apply_actions(plan, empty, {bed.id: [
        {"item": "wardrobe", "place": "wall", "wall": "Z"},
        {"item": "sofa_3_seat", "place": "wall", "wall": "A"},
        {"item": "nightstand", "place": "beside", "ref": "bed"},
        {"item": "spaceship", "place": "center"}]}, catalog)
    msgs = " | ".join(errors[bed.id])
    assert "wall Z does not exist" in msgs and "not meant for a bedroom" in msgs
    assert "nothing called 'bed'" in msgs and "not in the catalog" in msgs
    assert not st.items


class FakeLLM:
    name = "fake/test"

    def __init__(self, answers):
        self.answers, self.calls = list(answers), []

    def chat_json(self, messages):
        self.calls.append([dict(m) for m in messages])
        return self.answers.pop(0)


def test_failures_go_back_to_the_model_and_get_fixed(plan, catalog):
    bed = _room(plan, "bedroom", 20)
    wall = _plain_wall(plan, bed.id)
    bad = {"rooms": {bed.id: [{"item": "bed_double", "place": "wall", "wall": "Z", "name": "bed"}]},
           "summary": "first try"}
    good = {"rooms": {bed.id: [{"item": "bed_double", "place": "wall", "wall": wall, "name": "bed"},
                               {"item": "nightstand", "place": "beside", "ref": "bed"}]},
            "summary": "A calm bedroom with the bed facing the window."}
    llm = FakeLLM([bad, good])
    st, report = furnish(plan, {"style": "modern"}, catalog, llm=llm)
    assert len(llm.calls) == 2
    feedback = llm.calls[1][-1]["content"]
    assert "wall Z does not exist" in feedback and bed.id in feedback
    assert {i.catalog_id for i in st.in_room(bed.id)} == {"bed_double", "nightstand"}
    assert report.summary.startswith("A calm bedroom") and report.rounds == 2
    assert report.planner == "llm+rules:fake/test" and report.rule_rooms      # other rooms by rules
    assert validate_staging(plan, st, catalog)["approved"]


def test_no_model_or_broken_model_falls_back_to_rules(plan, catalog):
    st, report = furnish(plan, None, catalog, use_llm=False)
    assert report.planner == "rules" and st.items

    class Broken:
        name = "broken"

        def chat_json(self, messages):
            raise ConnectionError("offline")
    st, report = furnish(plan, None, catalog, llm=Broken())
    assert report.planner == "rules" and st.items and "failed in round 1" in report.warnings[0]
    assert validate_staging(plan, st, catalog)["approved"]
