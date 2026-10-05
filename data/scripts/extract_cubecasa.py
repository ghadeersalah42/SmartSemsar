"""
CubiCasa5k model.svg  ->  plan.json (backend/schema/plan.py)

- يقرأ الغرف والحوائط والأبواب والشبابيك من الـ SVG بالأبعاد الحقيقية.
- Also keeps the built-in fixtures (toilets, sinks, kitchen cabinets, closets),
  stairs and columns, so the furniture stage can place around them.
- Native scale is measured per plan from the room dimension labels
  (e.g. "8'4\" x 12'10\"") instead of a fixed pixel factor.
- Look-alike plans are then scaled uniformly so their indoor area equals the
  listing's pf_area_sqm.
- Also rewrites the svg_* area columns of final_merged_dataset.csv.

Usage (from repo root):
    python data/scripts/extract_cubecasa.py
"""
import argparse
import math
import os
import re
import statistics
import sys
import xml.etree.ElementTree as ET

import pandas as pd
from shapely.geometry import MultiPoint, Polygon
from shapely.ops import unary_union

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)

from backend.schema.plan import (  # noqa: E402
    Door, Fixture, Floor, Plan, Room, ScaleInfo, Wall, Window, save_plan,
)

NS = "{http://www.w3.org/2000/svg}"
DEFAULT_PX_PER_M = 100.0  # CubiCasa drawings are ~1 px = 1 cm

# CubiCasa "Space ..." class -> our room type (أول كلمة مطابقة هي اللي بتتاخد)
ROOM_TYPE_MAP = [
    ("Outdoor", "outdoor"),
    ("CarPort", "garage"),
    ("Bedroom", "bedroom"),
    ("Bath", "bathroom"),          # Bath, Bath Shower (KPH / WC / PH)
    ("Sauna", "sauna"),
    ("LivingRoom", "living"),
    ("Kitchen", "kitchen"),
    ("Dining", "dining"),
    ("DraughtLobby", "entry"),
    ("Entry", "entry"),
    ("Storage", "storage"),
    ("Closet", "closet"),
    ("DressingRoom", "closet"),
    ("Utility", "laundry"),
    ("TechnicalRoom", "technical"),
    ("UserDefined", "room"),
    ("Room", "room"),
]

# CubiCasa "FixedFurniture ..." class -> (fixture type, height m, elevation m)
# أول كلمة مطابقة هي اللي بتتاخد (ShowerScreen قبل Shower)
FIXTURE_TYPE_MAP = [
    ("Toilet", "toilet", 0.8, 0.0),
    ("DoubleSink", "sink", 0.85, 0.0),
    ("RoundSink", "sink", 0.85, 0.0),
    ("Sink", "sink", 0.85, 0.0),
    ("ShowerScreen", "other", 2.0, 0.0),
    ("Shower", "shower", 0.05, 0.0),        # tray only
    ("Bathtub", "bathtub", 0.55, 0.0),
    ("BaseCabinet", "base_cabinet", 0.9, 0.0),
    ("WallCabinet", "wall_cabinet", 0.7, 1.4),
    ("ElectricalAppliance", "appliance", 0.85, 0.0),
    ("CoatCloset", "closet", 2.1, 0.0),
    ("Closet", "closet", 2.1, 0.0),
    ("Fireplace", "fireplace", 1.2, 0.0),
    ("Chimney", "chimney", 2.7, 0.0),
    ("SaunaBench", "sauna_bench", 0.5, 0.0),
]
DEFAULT_FIXTURE = ("other", 1.0, 0.0)
FULL_HEIGHT_FIXTURE_M = 2.7     # stairs and columns run floor to ceiling
MIN_FIXTURE_AREA_PX = 25.0      # drop symbols smaller than ~5 x 5 cm (taps, marks)
IDENTITY = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)


# ---------- helpers ----------
def parse_points(points_str):
    """'x,y x,y' or 'x,y,x,y' -> [(x, y), ...]"""
    nums = [float(v) for v in re.findall(r"-?\d+(?:\.\d+)?(?:[eE]-?\d+)?", points_str or "")]
    return list(zip(nums[0::2], nums[1::2]))


def feet_inches_to_m(text):
    """'8\\'4" x 12\\'10"' -> [2.54, 3.91]"""
    parts = re.findall(r"(\d+)'(?:\s*(\d+)\")?", text or "")
    return [int(ft) * 0.3048 + int(inch or 0) * 0.0254 for ft, inch in parts]


def room_type_from_class(cls):
    tokens = cls.split()[1:]  # drop "Space"
    for key, rtype in ROOM_TYPE_MAP:
        if any(t.startswith(key) for t in tokens):
            return rtype
    return "other"


def floor_level(floorplan_class):
    """'Floorplan Floor-2' -> 1 ; 'Floorplan Basement-1' -> -1"""
    m = re.search(r"(Floor|Basement)-(\d+)", floorplan_class or "")
    if not m:
        return 0
    n = int(m.group(2))
    return n - 1 if m.group(1) == "Floor" else -n


def _poly_px(elem):
    poly = elem.find(NS + "polygon")
    return parse_points(poly.attrib.get("points")) if poly is not None else []


def _long_side(pts):
    return max(((pts[i][0] - pts[i - 1][0]) ** 2 + (pts[i][1] - pts[i - 1][1]) ** 2) ** 0.5
               for i in range(len(pts)))


def fixture_spec_from_class(cls):
    """'FixedFurniture ElectricalAppliance Refrigerator' -> ('appliance', 0.85, 0.0)"""
    tokens = cls.split()[1:]  # drop "FixedFurniture"
    for key, ftype, height, elevation in FIXTURE_TYPE_MAP:
        if any(t.startswith(key) for t in tokens):
            return ftype, height, elevation
    return DEFAULT_FIXTURE


def parse_matrix(transform):
    """'matrix(a,b,c,d,e,f)' -> (a, b, c, d, e, f); anything else -> identity."""
    m = re.match(r"\s*matrix\(([^)]*)\)", transform or "")
    nums = [float(v) for v in re.split(r"[,\s]+", m.group(1).strip())] if m else []
    return tuple(nums) if len(nums) == 6 else IDENTITY


def _compose(parent, child):
    """Matrix that applies `child` first, then `parent`."""
    a1, b1, c1, d1, e1, f1 = parent
    a2, b2, c2, d2, e2, f2 = child
    return (a1 * a2 + c1 * b2, b1 * a2 + d1 * b2, a1 * c2 + c1 * d2, b1 * c2 + d1 * d2,
            a1 * e2 + c1 * f2 + e1, b1 * e2 + d1 * f2 + f1)


def _apply(matrix, pts):
    a, b, c, d, e, f = matrix
    return [(a * x + c * y + e, b * x + d * y + f) for x, y in pts]


def _walk(elem, matrix=IDENTITY):
    """Yield (element, matrix to floor pixels) for elem and everything under it."""
    matrix = _compose(matrix, parse_matrix(elem.attrib.get("transform")))
    yield elem, matrix
    for child in elem:
        yield from _walk(child, matrix)


def _shape_px(elem):
    """Footprint of a fixture / column / stair part in its own frame.
    Fixtures keep theirs in a BoundaryPolygon group; the first shape found wins."""
    holder = next((c for c in elem if c.attrib.get("class") == "BoundaryPolygon"), elem)
    for s in holder.iter():
        tag, a = s.tag.replace(NS, ""), s.attrib
        if tag == "polygon":
            return parse_points(a.get("points"))
        if tag == "rect":
            x, y, w, h = (float(a.get(k, 0)) for k in ("x", "y", "width", "height"))
            return [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
        if tag == "circle":
            cx, cy, r = (float(a.get(k, 0)) for k in ("cx", "cy", "r"))
            return [(cx + r * math.cos(i * math.pi / 6), cy + r * math.sin(i * math.pi / 6))
                    for i in range(12)]
        if tag == "path":  # M/L outlines, sometimes with a repeated diagonal -> use the hull
            hull = MultiPoint(parse_points(a.get("d"))).convex_hull
            return list(hull.exterior.coords)[:-1] if isinstance(hull, Polygon) else []
    return []


# ---------- 1) SVG -> raw geometry in pixels ----------
def parse_svg(svg_path):
    """Return {'px_per_m', 'floors': [{level, name, rooms, walls, fixtures}]} in SVG pixels."""
    root = ET.parse(svg_path).getroot()
    floors, ratios = [], []

    for floor_g in (e for e in root.iter() if e.attrib.get("class") == "Floor"):
        plan_g = next((c for c in floor_g if "Floorplan" in c.attrib.get("class", "")), floor_g)
        fclass = plan_g.attrib.get("class", "")
        floor = {"level": floor_level(fclass), "name": fclass.replace("Floorplan", "").strip(),
                 "rooms": [], "walls": [], "fixtures": []}

        for e, matrix in _walk(floor_g):
            cls = e.attrib.get("class", "")
            head = cls.split()[:1]
            # الحاجات الثابتة مرسومة بإحداثيات محلية + transform، فبنحوّلها لإحداثيات الدور
            if cls.startswith("FixedFurniture ") or head == ["Column"] or head in (["Flight"], ["Winding"]):
                if cls.startswith("FixedFurniture "):
                    spec = fixture_spec_from_class(cls)
                else:
                    spec = ("column" if head == ["Column"] else "stairs", FULL_HEIGHT_FIXTURE_M, 0.0)
                pts = _apply(matrix, _shape_px(e))
                if len(pts) >= 3 and Polygon(pts).area >= MIN_FIXTURE_AREA_PX:
                    floor["fixtures"].append({"raw_type": cls.strip(), "type": spec[0], "height_m": spec[1],
                                              "elevation_m": spec[2], "pts": pts})
                continue
            if cls.startswith("Space "):
                pts = _poly_px(e)
                if len(pts) < 3:
                    continue
                floor["rooms"].append({"raw_type": cls, "type": room_type_from_class(cls), "pts": pts})
                # native scale from the dimension label (عرض/طول الغرفة الحقيقي)
                label = next((t for t in e.iter()
                              if t.attrib.get("class") == "TextLabel DimensionMeasureLabel"), None)
                dims = feet_inches_to_m("".join(label.itertext())) if label is not None else []
                if len(dims) == 2 and min(dims) > 0.5:
                    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
                    pw, ph = sorted([max(xs) - min(xs), max(ys) - min(ys)])
                    a, b = sorted(dims)
                    ratios += [pw / a, ph / b]
            elif cls.split()[:1] == ["Wall"]:
                wall = {"pts": _poly_px(e), "exterior": "External" in cls, "doors": [], "windows": []}
                for o in e:
                    ocls = o.attrib.get("class", "")
                    if ocls.startswith("Door"):
                        kind = (ocls.split() + ["swing"])[1].lower()
                        wall["doors"].append({"pts": _poly_px(o), "kind": kind})
                    elif ocls.startswith("Window"):
                        wall["windows"].append({"pts": _poly_px(o)})
                if len(wall["pts"]) >= 3:
                    floor["walls"].append(wall)

        if floor["rooms"]:
            floors.append(floor)

    px_per_m = statistics.median(ratios) if ratios else DEFAULT_PX_PER_M
    floors.sort(key=lambda f: f["level"])
    return {"px_per_m": px_per_m, "floors": floors}


def native_indoor_area_sqm(raw):
    ppm2 = raw["px_per_m"] ** 2
    return sum(Polygon(r["pts"]).area for f in raw["floors"] for r in f["rooms"]
               if r["type"] not in ("outdoor", "garage")) / ppm2


ANCHOR_M = 0.25        # a fixture group this close to a room side stays glued to it when stretched
GROUP_GAP_M = 0.05     # fixtures closer than this move together (kitchen run, wardrobe row, stair flights)


def fixture_polygons_m(fixtures, rooms_px, min_x, max_y, ppm, factor):
    """Fixture footprints in plan meters, at REAL size.
    On a stretched look-alike (factor != 1) only positions change: touching fixtures form a group
    that moves rigidly. Along each axis a group within ANCHOR_M of one of its room's wall segments
    keeps that distance to the segment (a wardrobe stays against its wall, also in L-shaped rooms);
    otherwise it keeps its relative place. A group outside every room (a column in a wall)
    simply follows the stretch."""
    def native(pts):
        return [((x - min_x) / ppm, (max_y - y) / ppm) for x, y in pts]

    polys = [Polygon(native(x["pts"])).buffer(0) for x in fixtures]
    if factor == 1.0 or not polys:
        return [[(round(px, 4), round(py, 4)) for px, py in _outline(p)] for p in polys]
    rooms = [Polygon(native(r["pts"])).buffer(0) for r in rooms_px]

    parent = list(range(len(polys)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            if polys[i].distance(polys[j]) < GROUP_GAP_M:
                parent[find(i)] = find(j)
    groups = {}
    for i in range(len(polys)):
        groups.setdefault(find(i), []).append(i)

    shift = [None] * len(polys)
    for members in groups.values():
        shape = unary_union([polys[i] for i in members])
        c = shape.centroid
        room = min((r for r in rooms if r.contains(c)), key=lambda r: r.area, default=None)
        if room is None and not all(fixtures[i]["type"] == "column" for i in members):
            # in a niche or doorway: use the nearest room (a column in a wall just follows the stretch)
            room = min(rooms, key=lambda r: r.distance(shape), default=None)
            if room is not None and room.distance(shape) > ANCHOR_M:
                room = None
        if room is None:
            dx, dy = (factor - 1) * c.x, (factor - 1) * c.y
        else:
            dx = _axis_shift(shape, room, 0, factor)
            dy = _axis_shift(shape, room, 1, factor)
        for i in members:
            shift[i] = (dx, dy)
    return [[(round(px + shift[k][0], 4), round(py + shift[k][1], 4)) for px, py in _outline(p)]
            for k, p in enumerate(polys)]


def _axis_shift(shape, room, axis, factor):
    """Shift along x (axis 0) or y (axis 1) for a fixture group in a stretched room."""
    g0, g1 = shape.bounds[axis], shape.bounds[axis + 2]
    o0, o1 = shape.bounds[1 - axis], shape.bounds[3 - axis]          # extent on the other axis
    room = max(getattr(room, "geoms", [room]), key=lambda g: g.area)
    pts = list(room.exterior.coords)
    best = None                                                      # (gap, wall coordinate)
    for (ax, ay), (bx, by) in zip(pts, pts[1:]):
        a, b = (ax, ay), (bx, by)
        if abs(a[axis] - b[axis]) > 0.05:                             # not perpendicular to this axis
            continue
        lo, hi = sorted((a[1 - axis], b[1 - axis]))
        if hi < o0 - 0.05 or lo > o1 + 0.05:                          # segment not beside the group
            continue
        w = (a[axis] + b[axis]) / 2
        gap = min(abs(g0 - w), abs(w - g1))
        if gap < ANCHOR_M and (best is None or gap < best[0]):
            best = (gap, w)
    if best is not None:
        return (factor - 1) * best[1]
    return (factor - 1) * (g0 + g1) / 2


def _outline(poly):
    poly = max(getattr(poly, "geoms", [poly]), key=lambda g: g.area)
    return list(poly.exterior.coords)[:-1]


# ---------- 2) raw geometry -> Plan (meters) ----------
def build_plan(raw, plan_id, source="cubicasa_lookalike", source_ref=None,
               listing_id=None, target_area_sqm=None):
    """
    source == "user_upload"  -> keep the drawing's own scale (listing area is only a check).
    otherwise with target_area_sqm -> scale so indoor area == target_area_sqm.
    """
    ppm = raw["px_per_m"]
    native_area = native_indoor_area_sqm(raw)
    fit = source != "user_upload" and target_area_sqm and native_area > 0
    factor = (target_area_sqm / native_area) ** 0.5 if fit else 1.0
    m_per_px = factor / ppm

    floors = []
    for f in raw["floors"]:
        all_pts = [p for r in f["rooms"] for p in r["pts"]] + [p for w in f["walls"] for p in w["pts"]]
        min_x = min(p[0] for p in all_pts)
        max_y = max(p[1] for p in all_pts)

        def tf(pts):  # px -> meters, flip y so y points up
            return [(round((x - min_x) * m_per_px, 4), round((max_y - y) * m_per_px, 4)) for x, y in pts]

        fl = Floor(level=f["level"], name=f["name"])
        for i, r in enumerate(f["rooms"]):
            poly = tf(r["pts"])
            fl.rooms.append(Room(id=f"F{f['level']}_R{i}", type=r["type"], raw_type=r["raw_type"],
                                 polygon=poly, area_sqm=round(Polygon(poly).area, 2)))
        for i, w in enumerate(f["walls"]):
            wid = f"F{f['level']}_W{i}"
            poly = tf(w["pts"])
            thickness = Polygon(poly).area / _long_side(poly)
            fl.walls.append(Wall(id=wid, polygon=poly, thickness_m=round(thickness, 3), exterior=w["exterior"]))
            for j, d in enumerate(w["doors"]):
                if len(d["pts"]) < 3:
                    continue
                dp = tf(d["pts"])
                fl.doors.append(Door(id=f"{wid}_D{j}", wall_id=wid, polygon=dp,
                                     width_m=round(_long_side(dp), 3), kind=d["kind"]))
            for j, o in enumerate(w["windows"]):
                if len(o["pts"]) < 3:
                    continue
                op = tf(o["pts"])
                fl.windows.append(Window(id=f"{wid}_O{j}", wall_id=wid, polygon=op,
                                         width_m=round(_long_side(op), 3)))
        # fixtures: real size (never stretched), owned by the smallest room that contains them
        room_polys = sorted(((Polygon(r.polygon).buffer(0), r.id) for r in fl.rooms), key=lambda t: t[0].area)
        placed = fixture_polygons_m(f.get("fixtures", []), f["rooms"], min_x, max_y, ppm, factor)
        for i, (x, poly) in enumerate(zip(f.get("fixtures", []), placed)):
            inside = Polygon(poly).buffer(0).representative_point()
            room_id = next((rid for rp, rid in room_polys if rp.contains(inside)), None)
            fl.fixtures.append(Fixture(id=f"F{f['level']}_X{i}", type=x["type"], raw_type=x["raw_type"],
                                       room_id=room_id, polygon=poly, height_m=x["height_m"],
                                       elevation_m=x["elevation_m"]))
        floors.append(fl)

    plan = Plan(
        plan_id=plan_id, source=source, source_ref=source_ref, listing_id=listing_id,
        scale=ScaleInfo(method="fit_listing_area" if fit else "native", native_px_per_m=round(ppm, 2),
                        factor=round(factor, 4), native_area_sqm=round(native_area, 2),
                        target_area_sqm=target_area_sqm),
        floors=floors, total_area_sqm=0.0,
    )
    plan.recompute_total_area()
    return plan


# ---------- 3) dataset -> data/plans/<property_id>.json + corrected CSV ----------
def process_dataset(csv_path, svg_dir, out_dir):
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    cache = {}
    new_cols = {"svg_bedrooms": [], "svg_bathrooms": [], "svg_floors": [], "svg_area_sqm": [],
                "area_diff_sqm": [], "scale_factor": [], "plan_path": []}

    for _, row in df.iterrows():
        cid = str(int(row["cubicasa_id"]))
        if cid not in cache:
            cache[cid] = parse_svg(os.path.join(svg_dir, f"{cid}.svg"))
        raw = cache[cid]
        plan = build_plan(raw, plan_id=f"{row['property_id']}_cubicasa_{cid}",
                          source="cubicasa_lookalike", source_ref=f"cubicasa:{cid}",
                          listing_id=row["property_id"], target_area_sqm=float(row["pf_area_sqm"]))
        out_path = os.path.join(out_dir, f"{row['property_id']}.json")
        save_plan(plan, out_path)

        native = plan.scale.native_area_sqm
        new_cols["svg_bedrooms"].append(plan.count("bedroom"))
        new_cols["svg_bathrooms"].append(plan.count("bathroom"))
        new_cols["svg_floors"].append(plan.num_floors)
        new_cols["svg_area_sqm"].append(native)                      # real plan area, m²
        new_cols["area_diff_sqm"].append(round(native - row["pf_area_sqm"], 2))
        new_cols["scale_factor"].append(plan.scale.factor)
        new_cols["plan_path"].append(os.path.relpath(out_path, REPO_ROOT).replace(os.sep, "/"))

    for col, values in new_cols.items():
        df[col] = values
    # keep image_path as the last column like before
    cols = [c for c in df.columns if c != "image_path"] + ["image_path"]
    df[cols].to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"✅ {len(df)} plans written to {out_dir}; CSV updated: {csv_path}")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=os.path.join(REPO_ROOT, "data", "final_merged_dataset.csv"))
    ap.add_argument("--svg-dir", default=os.path.join(REPO_ROOT, "data", "cubicasa_svg"))
    ap.add_argument("--out-dir", default=os.path.join(REPO_ROOT, "data", "plans"))
    args = ap.parse_args()
    process_dataset(args.csv, args.svg_dir, args.out_dir)
