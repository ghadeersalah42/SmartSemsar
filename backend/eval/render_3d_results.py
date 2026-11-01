"""
Before / after pictures and a video of the photo benchmark: each photo next to the 3D piece made from it.
صورة قبل وبعد لكل صورة اتجربت (الصورة الأصلية جنب الموديل الـ 3D)، وفيديو بيعرضهم كلهم.

Reads the photo_benchmark JSON and the fitted models it left in data/eval/raw_3d/fitted/<photo name>/.
Renders without OpenGL: the mesh is turned, shaded and painted face by face with PIL.

Usage:
    python -m backend.eval.render_3d_results --photos-root "D:/.../ترابيزات" \
        --results reports/furnishing/photo_3d_run1/photo_benchmark.json --out reports/furnishing/before_after
    -> <out>/NN_<photo>.jpg for every photo, <out>/all.jpg (contact sheet) and <out>/photo_to_3d.mp4
"""
import argparse
import json
import math
from pathlib import Path
from typing import Optional

import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFont, ImageOps

from backend.eval.photo_benchmark import RAW_DIR

W, H = 1280, 720
PANEL = 600                     # each half of a frame (photo | 3D)
BG = (246, 244, 240)
INK = (30, 30, 30)
OK_COLOUR, FAIL_COLOUR = (40, 140, 70), (190, 50, 40)
LIGHT = np.array([0.4, 0.8, 0.6]) / np.linalg.norm([0.4, 0.8, 0.6])
FPS = 15


def font(size: int):
    for name in ("arial.ttf", "segoeui.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def load_mesh(glb) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """GLB -> vertices (y up, metres), faces, per-face RGB 0..1. Textures are sampled to vertex colours."""
    scene = trimesh.load(glb, force="scene")
    verts, faces, cols, base = [], [], [], 0
    for node in scene.graph.nodes_geometry:
        transform, name = scene.graph[node]
        mesh = scene.geometry[name].copy()
        mesh.apply_transform(transform)
        try:
            vc = mesh.visual.to_color().vertex_colors[:, :3] / 255.0
            fc = vc[mesh.faces].mean(axis=1)
        except Exception:                       # no colour at all: plain grey
            fc = np.full((len(mesh.faces), 3), 0.7)
        verts.append(mesh.vertices)
        faces.append(mesh.faces + base)
        cols.append(fc)
        base += len(mesh.vertices)
    return np.concatenate(verts), np.concatenate(faces), np.concatenate(cols)


def render(mesh, yaw_deg: float, size: int = PANEL, pitch_deg: float = 20.0) -> Image.Image:
    """Painter's algorithm: turn about y, tilt, sort faces far to near, shade by the light."""
    v, f, c = mesh
    a, p = math.radians(yaw_deg), math.radians(pitch_deg)
    turn = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
    tilt = np.array([[1, 0, 0], [0, math.cos(p), -math.sin(p)], [0, math.sin(p), math.cos(p)]])
    r = (v - (v.max(0) + v.min(0)) / 2) @ turn.T @ tilt.T
    tri = r[f]
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12
    shade = 0.45 + 0.55 * np.abs(normals @ LIGHT)
    colours = (np.clip(c * shade[:, None], 0, 1) * 255).astype(int)
    span = max(np.ptp(r[:, 0]), np.ptp(r[:, 1])) or 1.0
    scale = size * 0.82 / span
    xy = np.stack([size / 2 + tri[..., 0] * scale, size / 2 - tri[..., 1] * scale], axis=-1)
    img = Image.new("RGB", (size, size), BG)
    draw = ImageDraw.Draw(img)
    for k in np.argsort(tri[..., 2].mean(axis=1)):         # far (negative z) first, the camera looks down -z
        draw.polygon([tuple(pt) for pt in xy[k]], fill=tuple(colours[k]))
    return img


def photo_panel(path: Path, size: int = PANEL) -> Image.Image:
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        im.thumbnail((size, size))
    panel = Image.new("RGB", (size, size), BG)
    panel.paste(im, ((size - im.width) // 2, (size - im.height) // 2))
    return panel


def text_panel(lines: list[str], colour, size: int = PANEL) -> Image.Image:
    panel = Image.new("RGB", (size, size), (235, 232, 226))
    d = ImageDraw.Draw(panel)
    y = size // 2 - 30 * len(lines) // 2
    for n, line in enumerate(lines):
        d.text((size // 2, y + 34 * n), line, fill=colour if n == 0 else INK, font=font(30 if n == 0 else 22),
               anchor="mm")
    return panel


def outcome_lines(r: dict) -> tuple[list[str], tuple]:
    stage = r["stage"]
    if stage == "3d_ok":
        w, d, h = r["size_m"]
        return [f"{r['kind'].replace('_', ' ')}: {w} x {d} x {h} m"], OK_COLOUR
    if stage == "refused":
        return ["Correctly refused", (r.get("reason") or "")[:48]], OK_COLOUR
    words = {"wrongly_accepted": "Accepted, should be refused", "wrong_kind": f"Wrong kind: {r.get('kind')}",
             "bad_3d_size": "3D model has an absurd size", "3d_error": "3D server error",
             "vision_error": "Vision model error"}
    if stage.startswith("refused:"):
        return ["Refused, should be accepted", (r.get("reason") or "")[:48]], FAIL_COLOUR
    return [words.get(stage, stage), (r.get("error") or "")[:48]], FAIL_COLOUR


def frame(n: int, total: int, r: dict, left: Image.Image, right: Image.Image) -> Image.Image:
    lines, colour = outcome_lines(r)
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.text((40, 28), f"{n} / {total}   " + ", ".join(r["conditions"]), fill=INK, font=font(26))
    d.text((W - 40, 28), "OK" if r["success"] else "FAIL", fill=colour, font=font(30), anchor="ra")
    img.paste(left, (30, 80))
    img.paste(right, (W - 30 - PANEL, 80))
    d.text((30 + PANEL // 2, H - 22), "Photo", fill=INK, font=font(20), anchor="ms")
    label = lines[0] + (f"   ({r['model_3d_s']:.0f} s)" if (r.get("model_3d_s") or 0) >= 1 else "")
    d.text((W - 30 - PANEL // 2, H - 22), label, fill=colour, font=font(20), anchor="ms")
    return img


def fitted_model(photo: str) -> Optional[Path]:
    folder = RAW_DIR / "fitted" / Path(photo).stem
    found = sorted(folder.glob("*.glb")) if folder.exists() else []
    return found[0] if found else None


def title_card(summary: dict, model_3d: dict) -> Image.Image:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    o = summary["overall"]
    d.text((W // 2, 200), "Smart Semsar: real photos -> 3D furniture", fill=INK, font=font(44), anchor="mm")
    d.text((W // 2, 290), f"{o['photos']} workshop photos: poor light, odd angles, clutter",
           fill=INK, font=font(28), anchor="mm")
    d.text((W // 2, 360), f"Success: {o['success']} / {o['photos']}", fill=OK_COLOUR, font=font(36), anchor="mm")
    for k, (gen, s) in enumerate(model_3d.items()):
        d.text((W // 2, 430 + 40 * k), f"3D ({gen}): median {s['seconds_median']} s per piece, "
               f"{s['generated']} made, {s['failed']} failed", fill=INK, font=font(24), anchor="mm")
    return img


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--photos-root", required=True, type=Path)
    ap.add_argument("--results", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--times-from", type=Path, default=None,
                    help="an earlier photo_benchmark JSON whose 3D times to show (a cached run has none)")
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args(argv)

    data = json.loads(args.results.read_text(encoding="utf-8"))
    results = data["results"]
    if args.times_from:
        times = {r["photo"]: r["model_3d_s"] for r in json.loads(args.times_from.read_text(encoding="utf-8"))["results"]
                 if r.get("model_3d_s")}
        results = [{**r, "model_3d_s": r.get("model_3d_s") or times.get(r["photo"])} for r in results]
    args.out.mkdir(parents=True, exist_ok=True)
    writer = None
    if not args.no_video:
        import imageio.v2 as imageio
        writer = imageio.get_writer(args.out / "photo_to_3d.mp4", fps=FPS, codec="libx264", quality=7,
                                    macro_block_size=8)
        card = np.asarray(title_card(data["summary"], data.get("model_3d") or {}))
        for _ in range(FPS * 3):
            writer.append_data(card)

    stills = []
    for n, r in enumerate(results, 1):
        photo = photo_panel(args.photos_root / r["photo"])
        glb = fitted_model(r["photo"]) if r["stage"] == "3d_ok" else None
        mesh = load_mesh(glb) if glb else None
        right = render(mesh, 30) if mesh else text_panel(*outcome_lines(r))
        still = frame(n, len(results), r, photo, right)
        still.save(args.out / f"{n:02d}_{Path(r['photo']).stem}.jpg", quality=88)
        stills.append(still)
        if writer:
            for _ in range(FPS):                               # the photo alone, then the result
                writer.append_data(np.asarray(frame(n, len(results), r, photo, text_panel(["...", ""], INK))))
            if mesh:
                for k in range(FPS * 4):                       # one full turn in 4 s
                    writer.append_data(np.asarray(frame(n, len(results), r, photo, render(mesh, 30 + 360 * k / (FPS * 4)))))
            else:
                for _ in range(FPS * 2):
                    writer.append_data(np.asarray(still))
        print(f"{n:2d}. {r['stage']:20s} {r['photo']}")
    if writer:
        writer.close()

    cols = 3
    thumb = [s.resize((W // 2, H // 2)) for s in stills]
    sheet = Image.new("RGB", (cols * W // 2, math.ceil(len(thumb) / cols) * H // 2), BG)
    for k, t in enumerate(thumb):
        sheet.paste(t, ((k % cols) * W // 2, (k // cols) * H // 2))
    sheet.save(args.out / "all.jpg", quality=85)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
