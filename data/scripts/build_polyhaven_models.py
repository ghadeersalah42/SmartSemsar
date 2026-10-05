"""
Poly Haven models (CC0)  ->  data/catalog/models/<catalog id>.glb  + real sizes in the catalog
موديلات حقيقية بخامات (خشب، جلد، قماش) من Poly Haven بدل الموديلات البسيطة.

Each model is downloaded as glTF with 1k textures, then packed into ONE small GLB:
  - geometry and materials untouched, textures shrunk to TEXTURE_PX and stored as JPEG
  - kept at its TRUE size (never stretched); the catalog item takes the model's size
  - turned so the front faces +z, centred on the origin, standing on y = 0
    (same convention as build_furniture_models.py, so the viewer needs no scaling)

Usage (from repo root; needs internet the first time, downloads are cached):
    python data/scripts/build_polyhaven_models.py [--cache <folder>]
"""
import argparse
import io
import json
import math
import os
import re
import struct
import sys
import tempfile
import urllib.request

import numpy as np
import trimesh
from PIL import Image

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT_DIR = os.path.join(REPO_ROOT, "data", "catalog", "models")
CATALOG = os.path.join(REPO_ROOT, "data", "catalog", "furniture.json")
API = "https://api.polyhaven.com/files/"
TEXTURE_PX = 512        # longest side of every texture in the packed GLB
JPEG_QUALITY = 82

# catalog id -> (Poly Haven asset id, yaw in degrees that turns its front to +z, display name)
SOURCES = {
    "sofa_3_seat": ("sofa_02", 0, "Leather sofa"),
    "sofa_2_seat": ("Sofa_01", 0, "Fabric sofa"),
    "armchair": ("modern_arm_chair_01", 0, "Armchair"),
    "coffee_table": ("modern_coffee_table_01", 90, "Coffee table"),
    "tv_unit": ("modern_wooden_cabinet", 0, "TV unit"),
    "bookshelf": ("wooden_display_shelves_01", 90, "Display shelves"),
    "dining_table_6": ("dining_table", 0, "Dining table for 6"),
    "dining_table_4": ("round_wooden_table_01", 0, "Round dining table for 4"),
    "dining_chair": ("painted_wooden_chair_01", 0, "Dining chair"),
    "nightstand": ("side_table_01", 0, "Nightstand"),
    "wardrobe": ("drawer_cabinet", 0, "Storage unit"),
    "dresser": ("WoodenTable_03", 0, "Dresser"),
    "desk": ("metal_office_desk", 0, "Desk"),
    "console_table": ("chinese_console_table", 0, "Console table"),
    "shoe_cabinet": ("vintage_wooden_drawer_01", 0, "Shoe cabinet"),
}
# Poly Haven has no flat TV, so a plain one is added on the TV unit: (width, height, thickness) m
FLAT_TV = {"tv_unit": (1.25, 0.72, 0.04)}


# ---------- download ----------
def fetch(url: str, path: str) -> str:
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        req = urllib.request.Request(url, headers={"User-Agent": "SmartSemsar-research"})
        with urllib.request.urlopen(req, timeout=120) as r, open(path, "wb") as f:
            f.write(r.read())
    return path


def download_gltf(asset: str, cache: str) -> str:
    """Download <asset>.gltf with its .bin and 1k textures; return the .gltf path."""
    folder = os.path.join(cache, asset)
    info_path = fetch(API + asset, os.path.join(folder, "files.json"))
    main = json.load(open(info_path, encoding="utf-8"))["gltf"]["1k"]["gltf"]
    gltf_path = fetch(main["url"], os.path.join(folder, f"{asset}.gltf"))
    for rel, f in main.get("include", {}).items():
        fetch(f["url"], os.path.join(folder, *rel.split("/")))
    return gltf_path


# ---------- pack ----------
def _pad(data: bytes, fill: bytes = b"\x00") -> bytes:
    return data + fill * (-len(data) % 4)


def _shrunk_jpeg(path: str) -> bytes:
    im = Image.open(path)
    if im.mode != "RGB":
        im = im.convert("RGB")
    im.thumbnail((TEXTURE_PX, TEXTURE_PX), Image.LANCZOS)
    out = io.BytesIO()
    im.save(out, "JPEG", quality=JPEG_QUALITY, optimize=True)
    return out.getvalue()


def _add_box(gltf: dict, chunks: list[bytes], offset: int, centre, size, rgb, name: str) -> int:
    """Append a plain coloured box as a new mesh node; returns the new byte offset."""
    box = trimesh.creation.box(extents=size)
    box.apply_translation(centre)
    pos = np.asarray(box.vertices[box.faces].reshape(-1, 3), dtype="<f4")       # flat shaded: no shared vertices
    nor = np.asarray(np.repeat(box.face_normals, 3, axis=0), dtype="<f4")
    views = []
    for arr in (pos, nor):
        data = _pad(arr.tobytes())
        gltf["bufferViews"].append({"buffer": 0, "byteOffset": offset, "byteLength": arr.nbytes, "target": 34962})
        views.append(len(gltf["bufferViews"]) - 1)
        chunks.append(data)
        offset += len(data)
    gltf["accessors"].append({"bufferView": views[0], "componentType": 5126, "count": len(pos), "type": "VEC3",
                              "min": pos.min(0).tolist(), "max": pos.max(0).tolist()})
    gltf["accessors"].append({"bufferView": views[1], "componentType": 5126, "count": len(nor), "type": "VEC3"})
    gltf.setdefault("materials", []).append({"name": name, "pbrMetallicRoughness": {
        "baseColorFactor": [*rgb, 1.0], "metallicFactor": 0.0, "roughnessFactor": 0.35}})
    gltf["meshes"].append({"name": name, "primitives": [{
        "attributes": {"POSITION": len(gltf["accessors"]) - 2, "NORMAL": len(gltf["accessors"]) - 1},
        "material": len(gltf["materials"]) - 1}]})
    gltf["nodes"].append({"name": name, "mesh": len(gltf["meshes"]) - 1})
    return offset


def pack_glb(gltf_path: str, out_path: str, yaw_deg: float, flat_tv=None) -> tuple[float, float, float]:
    """glTF + .bin + textures -> one normalised GLB. Returns (width, depth, height) in meters."""
    folder = os.path.dirname(gltf_path)
    gltf = json.load(open(gltf_path, encoding="utf-8"))
    assert len(gltf["buffers"]) == 1, "expected a single .bin"
    chunks = [_pad(open(os.path.join(folder, gltf["buffers"][0]["uri"]), "rb").read())]
    offset = len(chunks[0])

    for image in gltf.get("images", []):        # textures move inside the GLB
        data = _shrunk_jpeg(os.path.join(folder, *image.pop("uri").split("/")))
        gltf["bufferViews"].append({"buffer": 0, "byteOffset": offset, "byteLength": len(data)})
        image.update(bufferView=len(gltf["bufferViews"]) - 1, mimeType="image/jpeg")
        chunks.append(_pad(data))
        offset += len(chunks[-1])

    # model bounds in its own frame, after the yaw that brings its front to +z
    a = math.radians(yaw_deg)
    turn = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
    lo, hi = trimesh.load(gltf_path).bounds
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]) @ turn.T
    lo, hi = corners.min(0), corners.max(0)
    width, height, depth = (hi - lo).tolist()

    scene = gltf["scenes"][gltf.get("scene", 0)]
    children = list(scene["nodes"])
    if flat_tv:                                  # screen + foot, on top, toward the back
        tv_w, tv_h, tv_t = flat_tv
        inv = turn.T                             # box positions are given in the turned frame
        top, back = hi[1], lo[2] + 0.12
        mid_x = (lo[0] + hi[0]) / 2
        for centre, size, name in (((mid_x, top + 0.06 + tv_h / 2, back), (tv_w, tv_h, tv_t), "tv_screen"),
                                   ((mid_x, top + 0.03, back), (0.45, 0.06, 0.2), "tv_foot")):
            c = (np.array(centre) @ inv.T).tolist()
            s = np.abs(np.array(size) @ inv.T).tolist()
            offset = _add_box(gltf, chunks, offset, c, s, (0.015, 0.016, 0.018), name)
            children.append(len(gltf["nodes"]) - 1)

    # one root node: turn, then centre on the origin and stand on the floor
    matrix = np.eye(4)
    matrix[:3, :3] = turn
    matrix[:3, 3] = [-(lo[0] + hi[0]) / 2, -lo[1], -(lo[2] + hi[2]) / 2]
    gltf["nodes"].append({"name": "root", "children": children, "matrix": matrix.T.flatten().tolist()})
    scene["nodes"] = [len(gltf["nodes"]) - 1]

    binary = b"".join(chunks)
    gltf["buffers"] = [{"byteLength": len(binary)}]
    text = _pad(json.dumps(gltf, separators=(",", ":")).encode("utf-8"), b" ")
    with open(out_path, "wb") as f:
        f.write(struct.pack("<4sII", b"glTF", 2, 12 + 8 + len(text) + 8 + len(binary)))
        f.write(struct.pack("<I4s", len(text), b"JSON") + text)
        f.write(struct.pack("<I4s", len(binary), b"BIN\x00") + binary)
    return round(width, 2), round(depth, 2), round(height, 2)


# ---------- catalog ----------
def update_catalog(sizes: dict[str, tuple[float, float, float]]) -> None:
    """Write each model's true size and name into its catalog line (the rest of the line is kept)."""
    lines = open(CATALOG, encoding="utf-8").read().split("\n")
    for n, line in enumerate(lines):
        m = re.search(r'"id": "([a-z0-9_]+)"', line)
        if not m or m.group(1) not in sizes:
            continue
        w, d, h = sizes[m.group(1)]
        for key, value in (("width_m", w), ("depth_m", d), ("height_m", h)):
            line = re.sub(rf'"{key}": [0-9.]+', f'"{key}": {value}', line)
        line = re.sub(r'"name": "[^"]*"', f'"name": "{SOURCES[m.group(1)][2]}"', line)
        lines[n] = line
    open(CATALOG, "w", encoding="utf-8", newline="\n").write("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", default=os.path.join(tempfile.gettempdir(), "polyhaven_cache"))
    ap.add_argument("--only", nargs="*", help="catalog ids to build (default: all)")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    sizes = {}
    for item_id, (asset, yaw, _) in SOURCES.items():
        if args.only and item_id not in args.only:
            continue
        out = os.path.join(OUT_DIR, f"{item_id}.glb")
        sizes[item_id] = pack_glb(download_gltf(asset, args.cache), out, yaw, FLAT_TV.get(item_id))
        w, d, h = sizes[item_id]
        print(f"ok {item_id:15s} <- {asset:28s} {w:.2f} x {d:.2f} x {h:.2f} m   {os.path.getsize(out) // 1024} KB")
    update_catalog(sizes)
    print(f"catalog sizes updated: {CATALOG}")
    sys.exit(0)
