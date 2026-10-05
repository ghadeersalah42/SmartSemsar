"""trellis_service with a fake Space client (no network): call order, file handling, errors."""
import pytest

from backend.services import trellis_service
from backend.services.trellis_service import TrellisError, generate_model


class FakeSpace:
    def __init__(self, tmp_path, glb=b"glTF\x02\x00\x00\x00rest", fail=None):
        self.calls, self.tmp, self.glb, self.fail = [], tmp_path, glb, fail

    def predict(self, *args, api_name):
        self.calls.append(api_name)
        if api_name == self.fail:
            raise RuntimeError("You have exceeded your ZeroGPU quota")
        if api_name == "/preprocess_image":
            (self.tmp / "clean.png").write_bytes(b"png")
            return {"path": str(self.tmp / "clean.png")}
        if api_name == "/get_seed":
            return 42
        if api_name == "/extract_glb":
            f = self.tmp / "space_out.glb"
            f.write_bytes(self.glb)
            return str(f), str(f)
        return None


def test_generate_model_runs_the_space_steps_in_order(tmp_path):
    photo = tmp_path / "chair.jpg"
    photo.write_bytes(b"jpg")
    space = FakeSpace(tmp_path)
    out = generate_model(photo, tmp_path / "out" / "chair_raw.glb", client=space)
    assert space.calls == ["/start_session", "/preprocess_image", "/get_seed", "/image_to_3d", "/extract_glb"]
    assert out.read_bytes()[:4] == b"glTF"


def test_quota_and_bad_output_are_explained(tmp_path):
    photo = tmp_path / "chair.jpg"
    photo.write_bytes(b"jpg")
    with pytest.raises(TrellisError, match="quota"):
        generate_model(photo, tmp_path / "a.glb", client=FakeSpace(tmp_path, fail="/image_to_3d"))
    with pytest.raises(TrellisError, match="did not return a GLB"):
        generate_model(photo, tmp_path / "b.glb", client=FakeSpace(tmp_path, glb=b"<html>error</html>"))
    with pytest.raises(ValueError):
        generate_model(photo, tmp_path / "c.glb", resolution="999", client=FakeSpace(tmp_path))


def test_custom_furniture_can_use_trellis(tmp_path, monkeypatch):
    from backend.services import custom_furniture
    called = {}

    def fake_generate(image, raw, **kw):
        called["raw"] = raw
        raise TrellisError("stop here")          # the fit step is covered by test_custom_furniture
    monkeypatch.setattr(trellis_service, "generate_model", fake_generate)
    with pytest.raises(TrellisError):
        custom_furniture.add_furniture_from_photo(tmp_path / "x.jpg", "armchair", tmp_path, generator="trellis")
    assert called["raw"].name == "armchair_raw.glb"
    with pytest.raises(ValueError):
        custom_furniture.add_furniture_from_photo(tmp_path / "x.jpg", "armchair", tmp_path, generator="dalle")
