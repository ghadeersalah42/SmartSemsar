"""
CubiCasa5k model.svg  ->  plan.json (backend/schema/plan.py)

- يقرأ الغرف والحوائط والأبواب والشبابيك من الـ SVG بالأبعاد الحقيقية.
- Native scale is measured per plan from the room dimension labels
  (e.g. "8'4\" x 12'10\"") instead of a fixed pixel factor.
- Look-alike plans are then scaled uniformly so their indoor area equals the
  listing's pf_area_sqm.
- Also rewrites the svg_* area columns of final_merged_dataset.csv.

Usage (from repo root):
    python data/scripts/extract_cubecasa.py
"""
import argparse
import os
import re
import statistics
import sys
import xml.etree.ElementTree as ET

import pandas as pd
from shapely.geometry import Polygon

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)

from backend.schema.plan import (  # noqa: E402
    Door, Floor, Plan, Room, ScaleInfo, Wall, Window, save_plan,
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


# ---------- 1) SVG -> raw geometry in pixels ----------
def parse_svg(svg_path):
    """Return {'px_per_m', 'floors': [{level, name, rooms, walls}]} in SVG pixels."""
    root = ET.parse(svg_path).getroot()
    floors, ratios = [], []

    for floor_g in (e for e in root.iter() if e.attrib.get("class") == "Floor"):
        plan_g = next((c for c in floor_g if "Floorplan" in c.attrib.get("class", "")), floor_g)
        fclass = plan_g.attrib.get("class", "")
        floor = {"level": floor_level(fclass), "name": fclass.replace("Floorplan", "").strip(),
                 "rooms": [], "walls": []}

        for e in floor_g.iter():
            cls = e.attrib.get("class", "")
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


# if __name__ == "__main__":
#     ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
#     ap.add_argument("--csv", default=os.path.join(REPO_ROOT, "data", "final_merged_dataset.csv"))
#     ap.add_argument("--svg-dir", default=os.path.join(REPO_ROOT, "data", "cubicasa_svg"))
#     ap.add_argument("--out-dir", default=os.path.join(REPO_ROOT, "data", "plans"))
#     args = ap.parse_args()
#     process_dataset(args.csv, args.svg_dir, args.out_dir)
