"""
A user's own furniture: raw GLB (from a photo) -> catalog-ready model -> catalog item.
قطعة فرش من صورة المستخدم: بنظبط مقاسها واتجاهها وبنحطها في الكتالوج بدل القطعة الأصلية.

A generated model has no real size and may face any way. fit_model():
  - turns it (yaw_deg) so its front faces +z
  - scales it uniformly so its WIDTH is the real width; depth and height follow the model
  - centres it on the origin and stands it on y = 0
Geometry, colours and textures are untouched: the fit is one transform on a new root node.

with_custom_item() then returns a catalog where that piece replaces the stock item of the same
kind (same id, so the placement rules keep working), with the fitted model's real size.

    catalog, item = add_furniture_from_photo("sofa.jpg", "sofa", folder="data/custom/user42")
    staging = stage_plan(plan, catalog)
"""
import json
import math
import struct
from pathlib import Path
from typing import Optional

import numpy as np
import trimesh

from backend.schema.staging import Catalog, CatalogItem, load_catalog
from backend.services import colab_service, trellis_service, triposr_local

SIZE_LIMITS_M = (0.15, 4.0)     # a fitted piece outside this range is a bad model or a wrong width
GENERATORS = ("local", "colab", "trellis")
# the Colab server turns TripoSR models like TripoSR's demo does, which leaves the side seen in the
# photo facing -z; a half turn makes it the front (+z). triposr_local and TRELLIS.2 already face +z.
FRONT_TURN_DEG = {"colab": 180.0}


# ---------- GLB in / out ----------
def read_glb(path) -> tuple[dict, bytes]:
    data = Path(path).read_bytes()
    magic, version, _ = struct.unpack("<4sII", data[:12])
    if magic != b"glTF" or version != 2:
        raise ValueError(f"{path} is not a GLB (glTF 2) file")
    gltf, binary, pos = None, b"", 12
    while pos < len(data):
        length, kind = struct.unpack("<I4s", data[pos:pos + 8])
        chunk = data[pos + 8:pos + 8 + length]
        if kind == b"JSON":
            gltf = json.loads(chunk)
        elif kind == b"BIN\x00":
            binary = chunk
        pos += 8 + length
    if gltf is None:
        raise ValueError(f"{path} has no JSON chunk")
    return gltf, binary


def write_glb(path, gltf: dict, binary: bytes) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    text += b" " * (-len(text) % 4)
    binary += b"\x00" * (-len(binary) % 4)
    total = 12 + 8 + len(text) + (8 + len(binary) if binary else 0)
    with open(path, "wb") as f:
        f.write(struct.pack("<4sII", b"glTF", 2, total))
        f.write(struct.pack("<I4s", len(text), b"JSON") + text)
        if binary:
            f.write(struct.pack("<I4s", len(binary), b"BIN\x00") + binary)
    return path


# ---------- fit ----------
def fit_model(raw_glb, out_glb, width_m: float, yaw_deg: float = 0.0) -> tuple[float, float, float]:
    """Raw GLB -> catalog-ready GLB. Returns (width, depth, height) in meters."""
    lo, hi = trimesh.load(raw_glb).bounds
    a = math.radians(yaw_deg)
    turn = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]) @ turn.T
    lo, hi = corners.min(0), corners.max(0)
    if hi[0] - lo[0] <= 0:
        raise ValueError("The model has no width")
    scale = width_m / (hi[0] - lo[0])
    width, height, depth = ((hi - lo) * scale).tolist()
    for name, value in (("width", width), ("depth", depth), ("height", height)):
        if not SIZE_LIMITS_M[0] <= value <= SIZE_LIMITS_M[1]:
            raise ValueError(f"Fitted {name} is {value:.2f} m. Try turning the model 90 degrees "
                             f"or giving a different width.")

    gltf, binary = read_glb(raw_glb)
    matrix = np.eye(4)
    matrix[:3, :3] = turn * scale
    matrix[:3, 3] = [-(lo[0] + hi[0]) / 2 * scale, -lo[1] * scale, -(lo[2] + hi[2]) / 2 * scale]
    scene = gltf["scenes"][gltf.get("scene", 0)]
    gltf["nodes"].append({"name": "fit", "children": list(scene["nodes"]), "matrix": matrix.T.flatten().tolist()})
    scene["nodes"] = [len(gltf["nodes"]) - 1]
    write_glb(out_glb, gltf, binary)
    return round(width, 2), round(depth, 2), round(height, 2)


# ---------- catalog ----------
def stock_item(catalog: Catalog, kind: str) -> CatalogItem:
    """The stock item a custom piece of this kind replaces (the first one of that kind)."""
    for item in catalog.items:
        if item.kind == kind or item.id == kind:
            return item
    raise KeyError(f"No catalog item of kind '{kind}'. Known kinds: {sorted({i.kind for i in catalog.items})}")


def with_custom_item(catalog: Catalog, kind: str, mesh_path, size_m: tuple[float, float, float],
                     name: Optional[str] = None) -> tuple[Catalog, CatalogItem]:
    """A copy of the catalog where the user's piece stands in for the stock item of that kind.
    The original catalog is not changed."""
    old = stock_item(catalog, kind)
    width, depth, height = size_m
    new = old.model_copy(update={"width_m": width, "depth_m": depth, "height_m": height,
                                 "mesh": Path(mesh_path).as_posix(), "name": name or f"Your {old.kind.replace('_', ' ')}"})
    items = [new if i.id == old.id else i for i in catalog.items]
    return catalog.model_copy(update={"items": items}), new


def add_furniture_model(raw_glb, kind: str, folder, catalog: Optional[Catalog] = None,
                        width_m: Optional[float] = None, yaw_deg: float = 0.0,
                        name: Optional[str] = None) -> tuple[Catalog, CatalogItem]:
    """Fit a raw GLB and put it in the catalog. width_m defaults to the stock item's width."""
    catalog = catalog or load_catalog()
    old = stock_item(catalog, kind)
    fitted = Path(folder) / f"{old.id}.glb"
    size = fit_model(raw_glb, fitted, width_m or old.width_m, yaw_deg)
    return with_custom_item(catalog, kind, fitted.resolve(), size, name)


def add_furniture_from_photo(image_path, kind: str, folder, catalog: Optional[Catalog] = None,
                             width_m: Optional[float] = None, yaw_deg: float = 0.0,
                             name: Optional[str] = None, generator: str = "local",
                             **options) -> tuple[Catalog, CatalogItem]:
    """Photo -> image-to-3D -> fit -> catalog. The raw model is kept next to the fitted one,
    so a wrong direction can be fixed with add_furniture_model(raw, ..., yaw_deg=90) without the GPU.
    generator: "local" = TripoSR on this machine, GPU or CPU, no account (triposr_local),
               "colab" = TripoSR on our Colab GPU server (colab_service),
               "trellis" = TRELLIS.2 on its free Hugging Face Space (trellis_service), textured."""
    if generator not in GENERATORS:
        raise ValueError(f"Unknown generator '{generator}' (use one of {', '.join(GENERATORS)})")
    catalog = catalog or load_catalog()
    raw = Path(folder) / f"{stock_item(catalog, kind).id}_raw.glb"
    {"local": triposr_local, "colab": colab_service, "trellis": trellis_service}[generator].generate_model(
        image_path, raw, **options)
    yaw_deg = (yaw_deg + FRONT_TURN_DEG.get(generator, 0.0)) % 360
    return add_furniture_model(raw, kind, folder, catalog, width_m, yaw_deg, name)
