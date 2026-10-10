"""Several photos -> several pieces: which catalog item each replaces, the 3D cache, the whole apartment."""
import trimesh

from backend.schema.design import DesignPreferences
from backend.schema.staging import load_catalog
from backend.services import multi_furnish
from backend.services.multi_furnish import Piece, assign_slots, furnish_with_pieces, raw_model


def test_each_photo_takes_the_next_free_item_of_its_kind():
    said = []
    out = assign_slots([{"label": "a", "kind": "sofa"}, {"label": "b", "kind": "Sofa"},
                        {"label": "c", "kind": "sofa"}, {"label": "d", "kind": "Console table"},
                        {"label": "e", "kind": "spaceship"}, {"label": "f", "kind": ""}],
                       load_catalog(), said.append)
    assert [(p["label"], p["slot"]) for p in out] == [("a", "sofa_3_seat"), ("b", "sofa_2_seat"),
                                                      ("d", "console_table")]
    assert any("c: skipped (all sofa models" in s for s in said)
    assert any("e: skipped ('spaceship'" in s for s in said) and any("f: skipped (say what" in s for s in said)


def test_a_catalog_id_picks_the_item_itself():
    out = assign_slots([{"label": "a", "kind": "sofa_2_seat"}], load_catalog(), lambda s: None)
    assert out[0]["slot"] == "sofa_2_seat"


def test_the_3d_model_is_made_once_per_photo(tmp_path, monkeypatch):
    calls = []

    class FakeServer:
        @staticmethod
        def generate_model(photo, out):
            calls.append(photo)
            trimesh.creation.box().export(out)
            return out

    monkeypatch.setitem(multi_furnish.GENERATORS, "colab", FakeServer)
    photo = tmp_path / "sofa.jpg"
    photo.write_bytes(b"not really a jpeg, only hashed")
    first = raw_model(photo, "colab", tmp_path / "raw", lambda s: None)
    said = []
    second = raw_model(photo, "colab", tmp_path / "raw", said.append)
    assert first == second and len(calls) == 1 and "reused" in said[0]


def test_whole_apartment_with_two_own_pieces(tmp_path, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    sofa, bed = tmp_path / "sofa.glb", tmp_path / "bed.glb"
    trimesh.creation.box(extents=(2.0, 0.8, 0.9)).export(sofa)
    trimesh.creation.box(extents=(1.6, 0.5, 2.0)).export(bed)
    result = furnish_with_pieces("data/plans/PROP_1002.json",
                                 [Piece(str(sofa), "sofa_3_seat", 2.2, label="sofa.jpg"),
                                  Piece(str(bed), "bed_double", 1.6, label="bed.jpg")],
                                 tmp_path / "out", DesignPreferences(density="full"), say=lambda s: None)
    placed = [i.catalog_id for i in result.staging.items]
    assert result.approved and result.walkthrough.exists()
    assert placed.count("bed_double") >= 1 and "sofa_3_seat" in placed
    assert set(result.mine) == {"sofa.jpg", "bed.jpg"} and result.mine["sofa.jpg"].width_m == 2.2
    assert {"sofa_3_seat", "bed_double"} <= set(result.prefs.must_have)
