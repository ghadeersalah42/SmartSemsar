"""
Render a .glb (from plan_to_glb) to PNG previews: top-down + isometric.
معاينة سريعة للموديل من غير OpenGL (matplotlib بس).

Usage (from repo root):
    python data/scripts/render_glb_preview.py data/models/PROP_1002.glb data/models/PROP_1002_preview.png
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import trimesh  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402


def load_triangles(glb_path):
    """GLB (y-up) -> triangles (N,3,3) in z-up coords + RGBA colors (N,4) in 0..1."""
    scene = trimesh.load(glb_path, force="scene")
    tris, cols = [], []
    for node in scene.graph.nodes_geometry:
        transform, geom_name = scene.graph[node]
        mesh = scene.geometry[geom_name].copy()
        mesh.apply_transform(transform)
        tris.append(mesh.triangles)
        visual = mesh.visual.to_color() if hasattr(mesh.visual, "to_color") else mesh.visual
        cols.append(visual.face_colors / 255.0)
    tris, cols = np.concatenate(tris), np.concatenate(cols)
    # y-up -> z-up: (x, y, z) -> (x, -z, y)
    return np.stack([tris[..., 0], -tris[..., 2], tris[..., 1]], axis=-1), cols


def render(glb_path, out_png, title=None):
    tris, cols = load_triangles(glb_path)
    fig = plt.figure(figsize=(14, 7))

    # 1) top-down: painter's algorithm, lowest faces first
    ax = fig.add_subplot(1, 2, 1)
    order = np.argsort(tris[..., 2].mean(axis=1))
    shade = cols[order].copy()
    top_z = tris[order][..., 2].mean(axis=1)
    shade[:, :3] *= (0.75 + 0.25 * (top_z - top_z.min()) / max(np.ptp(top_z), 1e-6))[:, None]
    ax.add_collection(PolyCollection(tris[order][..., :2], facecolors=shade, edgecolors="none"))
    ax.autoscale()
    ax.set_aspect("equal")
    ax.set_title("Top-down")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")

    # 2) isometric with simple directional shading
    ax3 = fig.add_subplot(1, 2, 2, projection="3d")
    normals = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12
    light = np.array([0.4, -0.6, 0.7]) / np.linalg.norm([0.4, -0.6, 0.7])
    shaded = cols.copy()
    shaded[:, :3] *= (0.55 + 0.45 * np.abs(normals @ light))[:, None]
    ax3.add_collection3d(Poly3DCollection(tris, facecolors=shaded, edgecolors="none"))
    mins, maxs = tris.reshape(-1, 3).min(0), tris.reshape(-1, 3).max(0)
    span = (maxs - mins).max()
    for setter, lo in zip((ax3.set_xlim, ax3.set_ylim, ax3.set_zlim), mins):
        setter(lo, lo + span)
    ax3.view_init(elev=35, azim=-60)
    ax3.set_box_aspect((1, 1, 1))
    ax3.set_axis_off()
    ax3.set_title("Isometric")

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    return out_png


# if __name__ == "__main__":
#     if len(sys.argv) < 3:
#         sys.exit("usage: render_glb_preview.py <in.glb> <out.png> [title]")
#     print(render(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None))
