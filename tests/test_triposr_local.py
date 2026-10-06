"""triposr_local without downloading TripoSR: checkpoint names, photo preparation, the turn to +z, dispatch."""
import numpy as np
import pytest
import trimesh
from PIL import Image

from backend.services import custom_furniture, triposr_local
from backend.services.triposr_local import TripoSRError


def test_transformers_4_checkpoint_names_are_renamed_for_transformers_5():
    old = {"image_tokenizer.model.encoder.layer.3.attention.attention.query.weight": 1,
           "image_tokenizer.model.encoder.layer.3.attention.output.dense.bias": 2,
           "image_tokenizer.model.encoder.layer.3.intermediate.dense.weight": 3,
           "image_tokenizer.model.encoder.layer.3.output.dense.weight": 4,
           "decoder.layers.0.weight": 5}
    new = triposr_local._renamed(old, wanted={"decoder.layers.0.weight"})
    assert set(new) == {"image_tokenizer.model.layers.3.attention.q_proj.weight",
                        "image_tokenizer.model.layers.3.attention.o_proj.bias",
                        "image_tokenizer.model.layers.3.mlp.fc1.weight",
                        "image_tokenizer.model.layers.3.mlp.fc2.weight",
                        "decoder.layers.0.weight"}
    # names the installed transformers already uses are kept (transformers 4)
    assert triposr_local._renamed(old, wanted=set(old)) == old


def studio_photo(path, background=(235, 235, 232)):
    """A brown chair-like shape (seat, back, two legs with a gap between them) on a plain background."""
    rgb = np.full((300, 240, 3), background, np.uint8)
    rgb[60:200, 60:180] = (110, 70, 40)       # back + seat
    rgb[200:280, 60:80] = (110, 70, 40)       # left leg
    rgb[200:280, 160:180] = (110, 70, 40)     # right leg
    Image.fromarray(rgb).save(path)
    return path


def test_plain_background_is_cut_out_without_rembg(tmp_path, monkeypatch):
    monkeypatch.setitem(triposr_local._loaded, "rembg", None)
    cut = np.asarray(triposr_local.cut_out(Image.open(studio_photo(tmp_path / "chair.png"))))
    alpha = cut[..., 3]
    assert alpha[100, 100] == 255 and alpha[250, 70] == 255          # body and a leg kept
    assert alpha[5, 5] == 0 and alpha[250, 120] == 0                  # background and the gap between legs removed


def test_busy_background_without_rembg_is_a_clear_error(tmp_path, monkeypatch):
    monkeypatch.setitem(triposr_local._loaded, "rembg", None)
    noise = np.random.default_rng(0).integers(0, 255, (200, 200, 3), dtype=np.uint8)
    Image.fromarray(noise).save(tmp_path / "room.png")
    with pytest.raises(TripoSRError, match="plain background"):
        triposr_local.cut_out(Image.open(tmp_path / "room.png"))


def test_prepare_centres_the_piece_on_grey(tmp_path):
    rgba = np.zeros((400, 300, 4), np.uint8)
    rgba[100:300, 50:150] = (200, 30, 30, 255)    # a 200 x 100 px piece, off-centre, transparent around it
    Image.fromarray(rgba).save(tmp_path / "piece.png")
    image = np.asarray(triposr_local.prepare(tmp_path / "piece.png"))
    side = image.shape[0]
    assert image.shape == (side, side, 3) and side == int(200 / triposr_local.FOREGROUND_RATIO)
    assert tuple(image[0, 0]) == (127, 127, 127)                       # grey behind the piece
    assert tuple(image[side // 2, side // 2]) == (200, 30, 30)          # piece in the middle


class FakeTripoSR:
    """Answers like TripoSR: z up, the photographed side toward +x, faces wound inside out
    (as scikit-image's marching cubes leaves them), sRGB vertex colours in 0..1."""
    def __call__(self, images, device):
        assert images[0].size[0] == images[0].size[1]
        return [0]

    def extract_mesh(self, codes, resolution):
        seat = trimesh.creation.box(extents=(0.5, 0.5, 0.1))
        back = trimesh.creation.box(extents=(0.05, 0.5, 0.8))
        back.apply_translation((-0.25, 0, 0.35))      # backrest away from the camera
        mesh = trimesh.util.concatenate([seat, back])
        mesh.invert()
        mesh.visual.vertex_colors = np.tile([0.5, 0.5, 0.5, 1.0], (len(mesh.vertices), 1))
        return [mesh]


def test_generated_model_faces_front_is_upright_and_has_linear_colours(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    monkeypatch.setitem(triposr_local._loaded, "model", FakeTripoSR())
    rgba = np.zeros((64, 64, 4), np.uint8)
    rgba[10:50, 20:40] = 255
    Image.fromarray(rgba).save(tmp_path / "chair.png")
    out = triposr_local.generate_model(tmp_path / "chair.png", tmp_path / "chair_raw.glb", resolution=96)
    mesh = trimesh.load(out, force="mesh")
    assert mesh.volume > 0                                            # faces point outward
    lo, hi = mesh.bounds
    assert hi[1] - lo[1] == pytest.approx(0.8, abs=1e-3)              # y is up
    top = mesh.vertices[mesh.vertices[:, 1] > hi[1] - 0.1]
    assert top[:, 2].mean() < 0                                       # backrest at -z: the front faces +z
    assert abs(int(mesh.visual.vertex_colors[0][0]) - 55) <= 1        # sRGB 0.5 stored as linear 0.21


def test_missing_packages_are_named(monkeypatch):
    monkeypatch.setattr(triposr_local, "NEEDS", ("no_such_module_xyz",))
    monkeypatch.delitem(triposr_local._loaded, "model", raising=False)
    with pytest.raises(TripoSRError, match="no_such_module_xyz.*requirements-photo3d"):
        triposr_local.load_model()


def test_custom_furniture_uses_local_by_default_and_turns_colab_models(tmp_path, monkeypatch):
    calls, fitted = [], []

    def fake_generate(name):
        def generate(image, raw, **kw):
            calls.append(name)
        return generate
    monkeypatch.setattr(triposr_local, "generate_model", fake_generate("local"))
    monkeypatch.setattr(custom_furniture.colab_service, "generate_model", fake_generate("colab"))
    monkeypatch.setattr(custom_furniture, "add_furniture_model",
                        lambda raw, kind, folder, catalog, width_m, yaw_deg, name: fitted.append(yaw_deg))
    custom_furniture.add_furniture_from_photo(tmp_path / "x.jpg", "armchair", tmp_path, yaw_deg=90)
    custom_furniture.add_furniture_from_photo(tmp_path / "x.jpg", "armchair", tmp_path, yaw_deg=270,
                                              generator="colab")
    assert calls == ["local", "colab"]
    assert fitted == [90, 90]        # TripoSR through the Colab server faces -z: half a turn more
