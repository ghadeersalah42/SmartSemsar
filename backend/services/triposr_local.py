"""
Photo -> 3D model with TripoSR, run right here: on the GPU if there is one, else on the CPU.
صورة قطعة فرش -> موديل GLB على نفس الجهاز، من غير حساب ولا quota ولا سيرفر.

Same contract as trellis_service / colab_service.generate_model(): one photo of one piece in, a GLB
out, front toward +z (custom_furniture.fit_model() then sets the real size).

Nothing here needs an account: the TripoSR code (from its Hugging Face Space) and weights
(stabilityai/TripoSR, 1.7 GB) are public files, downloaded once and cached. Downloading files is not
GPU time, so the ZeroGPU quota that stops TRELLIS.2 does not apply. Measured on a 4-core CPU:
~25 s to load, ~15 s per photo at resolution 128; a Colab T4 takes a few seconds.

Differences from the TripoSR repo, so it installs anywhere with pip only:
  - marching cubes: scikit-image when torchmcubes (a C++/CUDA build) is not installed
  - checkpoints load with transformers 4 and 5 (5 renamed the ViT weights)
  - background: u2net (the model rembg uses) run directly with onnxruntime, else a cut-out of a plain
    background. rembg itself is not used: its recent versions require numpy >= 2.3, and installing
    them upgrades numpy under a running notebook.

    pip install -r requirements-photo3d.txt
    glb = generate_model("chair.jpg", "out/chair_raw.glb")

Usage:
    python -m backend.services.triposr_local <photo> <out.glb> [resolution]
"""
import hashlib
import importlib.util
import re
import sys
import threading
import types
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import trimesh
from PIL import Image, ImageOps

from backend.config import mask_secrets

CODE_REPO, CODE_REVISION = "stabilityai/TripoSR", "f84354eb350eb07a108faf33a6bc564d455f9764"   # the Space
WEIGHTS_REPO = "stabilityai/TripoSR"
NEEDS = ("torch", "omegaconf", "einops", "transformers", "huggingface_hub")
DEFAULT_RESOLUTION = 192        # marching-cubes grid: 128 is coarser and faster, 256 finer and larger
MAX_PHOTO_PX = 1024
FOREGROUND_RATIO = 0.85         # how much of the square the piece fills, as in TripoSR's demo
ALPHA_MIN = 16                  # fainter mask values are background noise, not the piece's outline
# background remover: u2net (Apache-2.0, 170 MB), the one TripoSR's demo uses through rembg
U2NET_URL = "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx"
U2NET_MD5 = "60024c5c889badc19c04ad937298a77b"
CACHE_DIR = Path.home() / ".cache" / "smartsemsar"
# transformers 5 renamed the ViT weights that the checkpoint stores under their version-4 names
VIT_RENAMES = [(r"\.encoder\.layer\.(\d+)\.", r".layers.\1."),
               (r"\.attention\.attention\.query\.", ".attention.q_proj."),
               (r"\.attention\.attention\.key\.", ".attention.k_proj."),
               (r"\.attention\.attention\.value\.", ".attention.v_proj."),
               (r"\.attention\.output\.dense\.", ".attention.o_proj."),
               (r"\.intermediate\.dense\.", ".mlp.fc1."),
               (r"\.output\.dense\.", ".mlp.fc2.")]

_loaded: dict = {}
_lock = threading.Lock()


class TripoSRError(RuntimeError):
    """A package is missing, the download failed, or the photo could not be used."""


def missing_packages() -> list[str]:
    return [m for m in NEEDS if importlib.util.find_spec(m) is None]


def is_available() -> bool:
    return not missing_packages()


def is_loaded() -> bool:
    return "model" in _loaded


def device() -> str:
    """cuda:0 when PyTorch sees a GPU, else cpu."""
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda:0" if torch.cuda.is_available() else "cpu"


# ---------- model ----------
def _marching_cubes(level, threshold):
    """torchmcubes.marching_cubes with scikit-image: same vertex order ((x, y, z) = (k, j, i))."""
    import torch
    from skimage.measure import marching_cubes
    verts, faces, _normals, _values = marching_cubes(level.float().cpu().numpy(), threshold)
    return torch.from_numpy(verts[:, ::-1].copy()).float(), torch.from_numpy(faces.astype(np.int64))


@contextmanager
def _stand_ins():
    """While TripoSR's code is imported: stand-ins for modules it imports but never runs here
    (rembg: backgrounds are removed by cut_out(); imageio: only for videos) and for torchmcubes
    when it is not installed. Real modules that are already imported are left alone."""
    stand_ins = {"rembg": {}, "imageio": {}}
    if importlib.util.find_spec("torchmcubes") is None:
        stand_ins["torchmcubes"] = {"marching_cubes": _marching_cubes}
    added = [name for name in stand_ins if name not in sys.modules]
    for name in added:
        sys.modules[name] = types.SimpleNamespace(**stand_ins[name])
    try:
        yield
    finally:
        for name in added:
            sys.modules.pop(name, None)


def _code_dir() -> str:
    from huggingface_hub import snapshot_download
    return snapshot_download(CODE_REPO, repo_type="space", revision=CODE_REVISION,
                             allow_patterns=["tsr/*.py"], token=False)


def _renamed(state: dict, wanted: set) -> dict:
    out = {}
    for key, value in state.items():
        if key not in wanted:
            for old, new in VIT_RENAMES:
                key = re.sub(old, new, key)
        out[key] = value
    return out


def load_model():
    """TripoSR on device(), loaded once per process."""
    with _lock:
        if "model" in _loaded:
            return _loaded["model"]
        missing = missing_packages()
        if missing:
            raise TripoSRError(f"Missing packages: {', '.join(missing)} "
                               "(pip install -r requirements-photo3d.txt).")
        if importlib.util.find_spec("torchmcubes") is None and importlib.util.find_spec("skimage") is None:
            raise TripoSRError("Install scikit-image (or torchmcubes) for the mesh step.")
        import torch
        from huggingface_hub import hf_hub_download
        from omegaconf import OmegaConf

        try:
            code = _code_dir()
            if code not in sys.path:
                sys.path.insert(0, code)
            with _stand_ins():
                from tsr.models.tokenizers import image as tokenizer
                from tsr.system import TSR
                # public files: never send a token (a wrong HF_TOKEN would make the download fail)
                tokenizer.hf_hub_download = lambda *a, **k: hf_hub_download(*a, **{**k, "token": False})
                cfg = OmegaConf.load(hf_hub_download(WEIGHTS_REPO, "config.yaml", token=False))
                OmegaConf.resolve(cfg)
                model = TSR(cfg)
            state = torch.load(hf_hub_download(WEIGHTS_REPO, "model.ckpt", token=False), map_location="cpu")
            model.load_state_dict(_renamed(state, set(model.state_dict())))
        except Exception as e:      # network, disk, an incompatible package version
            raise TripoSRError(f"Could not load TripoSR ({type(e).__name__}: {mask_secrets(e)[:300]}).") from e
        model.renderer.set_chunk_size(8192)
        model.to(device()).eval()
        _loaded["model"] = model
        return model


# ---------- photo ----------
def _plain_background_cut(rgb: np.ndarray) -> np.ndarray:
    """Alpha mask for a piece on a plain background (studio / catalog photos)."""
    from scipy import ndimage
    border = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]]).astype(np.float32)
    background = np.median(border, axis=0)
    spread = np.percentile(np.linalg.norm(border - background, axis=1), 90)
    if spread > 40:
        raise TripoSRError("The photo's background is not plain and the background remover is not "
                           "available. Use a photo on a plain background, or pip install onnxruntime.")
    piece = np.linalg.norm(rgb.astype(np.float32) - background, axis=2) > max(30.0, 2.5 * spread)
    piece = ndimage.binary_opening(piece, iterations=2)
    labels, n = ndimage.label(piece)
    if n == 0:
        raise TripoSRError("No piece of furniture found in the photo.")
    sizes = np.bincount(labels.ravel())[1:]
    keep = np.isin(labels, 1 + np.flatnonzero(sizes >= 0.02 * sizes.max()))
    return (keep * 255).astype(np.uint8)


def _u2net_session():
    """u2net on onnxruntime; the model is downloaded once into CACHE_DIR and checked."""
    import onnxruntime
    path = CACHE_DIR / "u2net.onnx"
    if not path.exists():
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        part = path.with_suffix(".part")
        urllib.request.urlretrieve(U2NET_URL, part)
        if hashlib.md5(part.read_bytes()).hexdigest() != U2NET_MD5:
            part.unlink()
            raise TripoSRError("The background remover's model download was damaged; try again.")
        part.rename(path)
    return onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def _u2net_mask(session, rgb: Image.Image) -> Image.Image:
    """Alpha mask, as rembg's u2net computes it."""
    x = np.asarray(rgb.resize((320, 320), Image.Resampling.LANCZOS), np.float32)
    x = (x / max(x.max(), 1e-6) - (0.485, 0.456, 0.406)) / (0.229, 0.224, 0.225)
    pred = session.run(None, {session.get_inputs()[0].name: x.transpose(2, 0, 1)[None].astype(np.float32)})[0][0, 0]
    pred = (pred - pred.min()) / max(pred.max() - pred.min(), 1e-6)
    mask = Image.fromarray((pred.clip(0, 1) * 255).astype(np.uint8), "L")
    return mask.resize(rgb.size, Image.Resampling.LANCZOS)


def cut_out(image: Image.Image, say=None) -> Image.Image:
    """RGBA with the background transparent: the photo's own alpha, else u2net, else a plain-background cut."""
    if image.mode == "RGBA" and image.getextrema()[3][0] < 255:
        return image
    rgb = image.convert("RGB")
    if "u2net" not in _loaded:
        try:
            _loaded["u2net"] = _u2net_session()
        except Exception as e:      # onnxruntime missing, or the model download failed
            _loaded["u2net"] = None
            if say:
                say(f"background remover not available ({type(e).__name__}); cutting out a plain background instead")
    out = rgb.convert("RGBA")
    if _loaded["u2net"]:
        out.putalpha(_u2net_mask(_loaded["u2net"], rgb))
    else:
        out.putalpha(Image.fromarray(_plain_background_cut(np.asarray(rgb))))
    return out


def prepare(photo, say=None) -> Image.Image:
    """Photo -> what TripoSR expects: the piece centred on a grey square."""
    image = ImageOps.exif_transpose(Image.open(photo))
    image.thumbnail((MAX_PHOTO_PX, MAX_PHOTO_PX))
    rgba = np.asarray(cut_out(image, say))
    ys, xs = np.nonzero(rgba[..., 3] > ALPHA_MIN)
    if len(xs) < 100:
        raise TripoSRError("No piece of furniture found in the photo.")
    piece = rgba[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    side = int(max(piece.shape[:2]) / FOREGROUND_RATIO)
    square = np.zeros((side, side, 4), np.uint8)
    y0, x0 = (side - piece.shape[0]) // 2, (side - piece.shape[1]) // 2
    square[y0:y0 + piece.shape[0], x0:x0 + piece.shape[1]] = piece
    a = square[..., 3:4].astype(np.float32) / 255
    rgb = square[..., :3].astype(np.float32) * a + 127.5 * (1 - a)
    return Image.fromarray(rgb.astype(np.uint8))


# ---------- photo -> GLB ----------
def _srgb_to_linear(c: np.ndarray) -> np.ndarray:
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def photo_to_mesh(photo, resolution: int = DEFAULT_RESOLUTION, say=None) -> trimesh.Trimesh:
    import torch
    model = load_model()
    image = prepare(photo, say)
    with torch.no_grad():
        codes = model([image], device=device())
        mesh = model.extract_mesh(codes, resolution=resolution)[0]
    if len(mesh.faces) == 0:
        raise TripoSRError("TripoSR made an empty model from this photo.")
    if mesh.volume < 0:         # scikit-image winds the faces the other way round than torchmcubes
        mesh.invert()
    # TripoSR's frame -> glTF (y up), as its demo does; then a half turn so the side seen in the
    # photo is the front (+z), like TRELLIS.2 models and the catalog
    mesh.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))
    mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
    mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi, [0, 1, 0]))
    # glTF vertex colours are linear; TripoSR's are like photo pixels (sRGB)
    colors = mesh.visual.vertex_colors.astype(np.float32) / 255
    colors[:, :3] = _srgb_to_linear(colors[:, :3])
    mesh.visual.vertex_colors = (colors * 255).round().astype(np.uint8)
    return mesh


def generate_model(image_path, out_glb, resolution: int = DEFAULT_RESOLUTION, say=None) -> Path:
    """One photo of one piece of furniture -> GLB saved at out_glb."""
    if not 64 <= int(resolution) <= 320:
        raise ValueError("resolution must be between 64 and 320")
    try:
        mesh = photo_to_mesh(image_path, int(resolution), say)
    except TripoSRError:
        raise
    except Exception as e:      # out of memory, a broken photo, ...
        raise TripoSRError(f"TripoSR failed ({type(e).__name__}: {mask_secrets(e)[:300]}).") from e
    out_glb = Path(out_glb)
    out_glb.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(out_glb)
    return out_glb


if __name__ == "__main__":
    if len(sys.argv) not in (3, 4):
        sys.exit("usage: python -m backend.services.triposr_local <photo> <out.glb> [resolution]")
    import time
    t0 = time.time()
    try:
        out = generate_model(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) == 4 else DEFAULT_RESOLUTION,
                             say=print)
    except TripoSRError as e:
        sys.exit(f"error: {e}")
    print(f"{out} ({out.stat().st_size // 1024} KB, {time.time() - t0:.0f} s on {device()})")
