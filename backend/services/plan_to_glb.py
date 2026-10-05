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
FIXTURE_COLORS = {
    "closet": (214, 200, 178), "base_cabinet": (232, 228, 220), "wall_cabinet": (232, 228, 220),
    "sink": (205, 214, 222), "appliance": (226, 230, 232), "toilet": (250, 250, 250),
    "shower": (200, 214, 222), "bathtub": (250, 250, 250), "fireplace": (150, 120, 105),
    "chimney": (150, 120, 105), "sauna_bench": (196, 160, 118), "stairs": (190, 170, 145),
    "column": (236, 236, 232), "other": (215, 210, 200),
}
STEP_RISE_M = 0.17        # one stair step
STEP_RUN_M = 0.28
MIN_FLIGHT_WIDTH_M = 0.6  # straight stair pieces narrower than this are steps, not a flight
MIN_FLIGHT_RUN_M = 1.2    # ... and so are shorter ones (terrace steps, a step between two levels)
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


def _sides(poly):
    """(short side, long side) of a piece's minimum rectangle, meters."""
    pts = np.array(poly.minimum_rotated_rectangle.exterior.coords[:3])
    a, b = np.linalg.norm(pts[1] - pts[0]), np.linalg.norm(pts[2] - pts[1])
    return min(a, b), max(a, b)


def _steps(poly, z0, z1, start, color, along_short=False):
    """Steps on one stair piece, rising from z0 to z1 along its long side (short side for porch
    steps), starting at the end nearest to `start` (a point) or at the first end if None."""
    rect = poly.minimum_rotated_rectangle
    pts = np.array(rect.exterior.coords[:4])
    e1, e2 = pts[1] - pts[0], pts[3] - pts[0]
    run, side = (e1, e2) if np.linalg.norm(e1) >= np.linalg.norm(e2) else (e2, e1)
    if along_short:
        run, side = side, run
    origin = pts[0]
    if start is not None:
        far = origin + run
        if np.linalg.norm(far + side / 2 - start) < np.linalg.norm(origin + side / 2 - start):
            origin, run = far, -run
    n = max(1, int(round(np.linalg.norm(run) / STEP_RUN_M)))
    parts = []
    for i in range(n):
        a = origin + run * i / n
        step = Polygon([a, a + run / n, a + run / n + side, a + side]).intersection(poly)
        for p in _polys(step):
            m = _prism(p, z0, z0 + (z1 - z0) * (i + 1) / n, color)
            if m is not None:
                parts.append(m)
    end = origin + run + side / 2
    return parts, end


def _stair_runs(fixtures):
    """Group touching stair pieces (flight, turn, flight ...) in drawing order."""
    runs = []
    for fx in fixtures:
        poly = Polygon(fx.polygon).buffer(0)
        if runs and runs[-1][-1][1].distance(poly) < 0.05:
            runs[-1].append((fx, poly))
        else:
            runs.append([(fx, poly)])
    return runs


def fixture_meshes(floor, base_z=0.0, storey_height_m=2.75):
    """Built-ins from the drawing: boxes at their real height; stairs as steps."""
    meshes = {}
    for fx in floor.fixtures:
        if fx.type == "stairs":
            continue
        poly = Polygon(fx.polygon).buffer(0)
        z0 = base_z + fx.elevation_m
        parts = [m for m in (_prism(p, z0, z0 + fx.height_m, FIXTURE_COLORS.get(fx.type, FIXTURE_COLORS["other"]))
                             for p in _polys(poly)) if m is not None]
        if parts:
            meshes[f"fixture_{fx.id}"] = trimesh.util.concatenate(parts)

    color = FIXTURE_COLORS["stairs"]
    room_type = {r.id: r.type for r in floor.rooms}
    stairs = [fx for fx in floor.fixtures if fx.type == "stairs"]
    flights = []
    for run in _stair_runs(stairs):
        has_turn = any((fx.raw_type or "").startswith("Winding") for fx, _p in run)
        indoor = all(room_type.get(fx.room_id) not in ("outdoor", "garage") for fx, _p in run)
        wide = min(_sides(p)[0] for _fx, p in run) >= MIN_FLIGHT_WIDTH_M
        long_ = sum(_sides(p)[1] for _fx, p in run) >= MIN_FLIGHT_RUN_M
        if has_turn or (indoor and wide and long_):
            flights.append(run)
            continue
        for fx, poly in run:   # terrace / porch steps or a single step: climbed along the short side
            n = max(1, round(_sides(poly)[0] / STEP_RUN_M))
            parts, _end = _steps(poly, base_z, base_z + STEP_RISE_M * n, None, color, along_short=True)
            if parts:
                meshes[f"fixture_{fx.id}"] = trimesh.util.concatenate(parts)
    for run in flights:     # a flight always reaches the next floor (the source has no rise data)
        height = storey_height_m
        start = None
        if len(run) > 1:     # begin at the end of the first piece away from the second one
            c2 = np.array(run[1][1].centroid.coords[0])
            rect = np.array(run[0][1].minimum_rotated_rectangle.exterior.coords[:4])
            start = max(rect, key=lambda q: np.linalg.norm(q - c2))
        for k, (fx, poly) in enumerate(run):
            z0 = base_z + height * k / len(run)
            z1 = base_z + height * (k + 1) / len(run)
            parts, start = _steps(poly, z0, z1, start, color)
            if parts:
                meshes[f"fixture_{fx.id}"] = trimesh.util.concatenate(parts)
    return meshes


def plan_to_scene(plan: Plan, storey_height_m=None) -> trimesh.Scene:
    """Stack floors by level (basement below 0)."""
    scene = trimesh.Scene()
    for floor in plan.floors:
        wall_h = max((w.height_m for w in floor.walls), default=2.7)
        step = storey_height_m or (wall_h + FLOOR_THICKNESS_M)
        meshes = {**floor_meshes(floor, base_z=floor.level * step),
                  **fixture_meshes(floor, base_z=floor.level * step, storey_height_m=step)}
        for name, mesh in meshes.items():
            scene.add_geometry(mesh, node_name=f"L{floor.level}_{name}", geom_name=f"L{floor.level}_{name}")
    return scene


def export_glb(plan: Plan, out_path):
    scene = plan_to_scene(plan)
    # glTF is y-up: rotate our z-up model
    scene.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, (1, 0, 0)))
    scene.export(out_path)
    return out_path


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: python -m backend.services.plan_to_glb <plan.json> <out.glb>")
    print(export_glb(load_plan(sys.argv[1]), sys.argv[2]))
