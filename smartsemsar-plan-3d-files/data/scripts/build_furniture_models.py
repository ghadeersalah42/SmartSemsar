"""
Kenney Furniture Kit (CC0)  ->  data/catalog/models/<catalog id>.glb
تجهيز موديلات الفرش: كل موديل بيتظبط على مقاس القطعة الحقيقي في الكتالوج.

Every output model is ready to drop into the viewer with no scaling:
  - meters, y up, standing on y = 0
  - footprint centred on the origin, width along x, depth along z
  - the front faces +z (same as a catalog item at rotation 0, seen in the viewer)
  - width and depth are exactly the catalog's; extras may rise above the catalog
    height (the TV on the TV unit)

The kit is not stored in the repo. Download it from https://kenney.nl/assets/furniture-kit
and unzip it, then (from repo root):
    python data/scripts/build_furniture_models.py "<kit>/Models/GLTF format"
"""
import argparse
import os
import sys

import numpy as np
import trimesh

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)

from backend.schema.staging import load_catalog  # noqa: E402

OUT_DIR = os.path.join(REPO_ROOT, "data", "catalog", "models")

# catalog id -> kit model(s). Several names = copies set side by side along the width.
# Only the items Poly Haven has no good model for; the rest come from build_polyhaven_models.py.
SOURCES = {
    "bed_double": ["bedDouble"],
    "bed_single": ["bedSingle"],
}
# catalog id -> (kit model, width m) stood on top of the fitted item, keeping its own proportions
ON_TOP = {}

# الألوان الأصلية فاقعة (كنب وسراير حمرا)، فبنبدلها بألوان هادية. sRGB 0-255 per kit material name.
PALETTE = {
    "wood": (158, 120, 86),
    "carpet": (112, 128, 146),
    "carpetWhite": (236, 232, 222),
    "metal": (184, 188, 190),
    "metalDark": (34, 36, 38),
    "_defaultMat": (224, 219, 209),
}


def _linear(c: int) -> float:
    s = c / 255
    return s / 12.92 if s <= 0.04045 else ((s + 0.055) / 1.055) ** 2.4


def load_parts(kit_dir: str, name: str) -> list[trimesh.Trimesh]:
    """Meshes of one kit model in its own frame, recoloured."""
    scene = trimesh.load(os.path.join(kit_dir, name + ".glb"))
    parts = scene.dump(concatenate=False)
    for m in parts:
        material = m.visual.material
        rgb = PALETTE.get(material.name)
        if rgb:
            material.baseColorFactor = [_linear(c) for c in rgb] + [1.0]
    return parts


def bounds(parts) -> tuple[np.ndarray, np.ndarray]:
    lo = np.min([m.bounds[0] for m in parts], axis=0)
    hi = np.max([m.bounds[1] for m in parts], axis=0)
    return lo, hi


def moved(parts, matrix) -> list[trimesh.Trimesh]:
    out = []
    for m in parts:
        m = m.copy()
        m.apply_transform(matrix)
        out.append(m)
    return out


def fit(parts, width: float, height: float, depth: float, base_y: float = 0.0):
    """Stretch to exactly width x height x depth, centred on the origin, standing on base_y."""
    lo, hi = bounds(parts)
    scale = np.array([width, height, depth]) / (hi - lo)
    matrix = np.eye(4)
    matrix[:3, :3] = np.diag(scale)
    centre = (lo + hi) / 2
    matrix[:3, 3] = [-centre[0] * scale[0], base_y - lo[1] * scale[1], -centre[2] * scale[2]]
    return moved(parts, matrix)


def build_model(kit_dir: str, item) -> trimesh.Scene:
    names = SOURCES[item.id]
    parts, x = [], 0.0
    for name in names:                          # side by side along the width
        piece = load_parts(kit_dir, name)
        lo, hi = bounds(piece)
        shift = np.eye(4)
        shift[0, 3] = x - lo[0]
        parts += moved(piece, shift)
        x += hi[0] - lo[0]
    parts = fit(parts, item.width_m, item.height_m, item.depth_m)

    if item.id in ON_TOP:
        name, top_width = ON_TOP[item.id]
        top = load_parts(kit_dir, name)
        lo, hi = bounds(top)
        k = top_width / (hi[0] - lo[0])
        parts += fit(top, top_width, (hi[1] - lo[1]) * k, (hi[2] - lo[2]) * k, base_y=item.height_m)

    scene = trimesh.Scene()
    for n, m in enumerate(parts):
        scene.add_geometry(m, node_name=f"{item.id}_{n}", geom_name=f"{item.id}_{n}")
    return scene


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kit_dir", help='the kit\'s "Models/GLTF format" folder')
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    for item in load_catalog().items:
        if item.id not in SOURCES:
            print(f"-  {item.id}: not from this kit")
            continue
        out = os.path.join(OUT_DIR, f"{item.id}.glb")
        build_model(args.kit_dir, item).export(out)
        print(f"ok {item.id}: {os.path.getsize(out) // 1024} KB")
