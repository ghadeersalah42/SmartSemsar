"""
plan.json -> .glb  (deterministic extrusion, no AI geometry)
تحويل المخطط لموديل 3D: حوائط بارتفاع ~2.7م بفتحات للأبواب والشبابيك + أرضية لكل غرفة.

Usage:
    python -m backend.services.plan_to_glb data/plans/PROP_1002.json out.glb
"""
import sys

import numpy as np
import trimesh
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union

from backend.schema.plan import Plan, load_plan

FLOOR_THICKNESS_M = 0.05
EPS = 1e-3

ROOM_COLORS = {
    "bedroom": (168, 198, 230), "bathroom": (150, 215, 210), "living": (240, 214, 160),
    "kitchen": (240, 180, 150), "dining": (235, 200, 170), "entry": (210, 210, 210),
    "outdoor": (180, 215, 160), "garage": (190, 190, 190), "sauna": (215, 180, 140),
}
DEFAULT_ROOM_COLOR = (225, 220, 210)
WALL_COLOR = (245, 245, 240)
WALL_EXTERIOR_COLOR = (200, 196, 188)


def _polys(geom):
    """Shapely geometry -> list of valid non-empty Polygons."""
    if geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom] if geom.area > EPS * EPS else []
    if isinstance(geom, MultiPolygon):
        return [p for p in geom.geoms if p.area > EPS * EPS]
    return [g for g in getattr(geom, "geoms", []) if isinstance(g, Polygon) and g.area > EPS * EPS]


def _prism(poly, z0, z1, color):
    """Extrude a 2D polygon between z0 and z1."""
    if z1 - z0 <= EPS:
        return None
    mesh = trimesh.creation.extrude_polygon(poly, z1 - z0)
    mesh.apply_translation((0, 0, z0))
    mesh.visual.face_colors = (*color, 255)
    return mesh


def floor_meshes(floor, base_z=0.0):
    """All meshes for one floor, keyed by name."""
    meshes = {}

    # أرضية كل غرفة
    for room in floor.rooms:
        poly = Polygon(room.polygon).buffer(0)
        for i, p in enumerate(_polys(poly)):
            m = _prism(p, base_z - FLOOR_THICKNESS_M, base_z,
                       ROOM_COLORS.get(room.type, DEFAULT_ROOM_COLOR))
            if m is not None:
                meshes[f"floor_{room.id}_{i}"] = m

    openings = {}
    for d in floor.doors:
        openings.setdefault(d.wall_id, []).append((Polygon(d.polygon).buffer(0), [(d.height_m, None)]))
    for w in floor.windows:
        openings.setdefault(w.wall_id, []).append(
            (Polygon(w.polygon).buffer(0), [(0.0, w.sill_height_m), (w.sill_height_m + w.height_m, None)]))

    # الحوائط: الجزء المصمت بالارتفاع الكامل + أجزاء فوق/تحت الفتحات
    for wall in floor.walls:
        wpoly = Polygon(wall.polygon).buffer(0)
        color = WALL_EXTERIOR_COLOR if wall.exterior else WALL_COLOR
        H = wall.height_m
        ops = openings.get(wall.id, [])
        solid = wpoly.difference(unary_union([o for o, _ in ops])) if ops else wpoly
        parts = [_prism(p, base_z, base_z + H, color) for p in _polys(solid)]
        for opoly, bands in ops:
            cut = opoly.intersection(wpoly)
            for z0, z1 in bands:
                z1 = H if z1 is None else min(z1, H)
                parts += [_prism(p, base_z + z0, base_z + z1, color) for p in _polys(cut)]
        parts = [p for p in parts if p is not None]
        if parts:
            meshes[f"wall_{wall.id}"] = trimesh.util.concatenate(parts)
    return meshes


def plan_to_scene(plan: Plan, storey_height_m=None) -> trimesh.Scene:
    """Stack floors by level (basement below 0)."""
    scene = trimesh.Scene()
    for floor in plan.floors:
        wall_h = max((w.height_m for w in floor.walls), default=2.7)
        step = storey_height_m or (wall_h + FLOOR_THICKNESS_M)
        for name, mesh in floor_meshes(floor, base_z=floor.level * step).items():
            scene.add_geometry(mesh, node_name=f"L{floor.level}_{name}", geom_name=f"L{floor.level}_{name}")
    return scene


def export_glb(plan: Plan, out_path):
    scene = plan_to_scene(plan)
    # glTF is y-up: rotate our z-up model
    scene.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, (1, 0, 0)))
    scene.export(out_path)
    return out_path


# if __name__ == "__main__":
#     if len(sys.argv) != 3:
#         sys.exit("usage: python -m backend.services.plan_to_glb <plan.json> <out.glb>")
#     print(export_glb(load_plan(sys.argv[1]), sys.argv[2]))
