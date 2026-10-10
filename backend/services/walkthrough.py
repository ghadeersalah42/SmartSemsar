"""
plan.json + .glb -> self-contained walkthrough HTML (three.js, first-person).
جولة داخل الموديل: المشي بالكيبورد/اللمس، الحوائط بتمنع المرور والأبواب مفتوحة.

The page embeds the plan, the GLB (base64) and small viewer hints, so it opens
offline from disk or inside the app (e.g. gr.HTML / st.components.v1.html).

Usage:
    python -m backend.services.walkthrough PROP_1002
    -> data/models/PROP_1002.glb + data/models/PROP_1002_walkthrough.html
"""
import base64
import json
import math
import sys
from pathlib import Path
from typing import Mapping, Optional

from shapely.geometry import Polygon

from backend.schema.plan import Plan, load_plan
from backend.services.plan_to_glb import FLOOR_THICKNESS_M, export_glb

REPO_ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = REPO_ROOT / "frontend" / "walkthrough" / "viewer_template.html"
SPAWN_PREFERENCE = ["entry", "living", "hallway", "other", "kitchen", "dining"]


def viewer_hints(plan: Plan, listing: Optional[Mapping] = None) -> dict:
    """Spawn point / label points / floor extents the viewer needs (2D, plan meters)."""
    wall_h = max((w.height_m for f in plan.floors for w in f.walls), default=2.7)
    floors = []
    for f in plan.floors:
        polys = {r.id: Polygon(r.polygon).buffer(0) for r in f.rooms}
        indoor = [r for r in f.rooms if r.is_indoor] or f.rooms

        def rank(r):
            t = SPAWN_PREFERENCE.index(r.type) if r.type in SPAWN_PREFERENCE else len(SPAWN_PREFERENCE)
            return (t, -r.area_sqm)

        start = min(indoor, key=rank)
        # representative_point is always inside the polygon (works for L-shaped rooms)
        sp = polys[start.id].representative_point()
        # look toward the biggest room on this floor
        big = max(indoor, key=lambda r: r.area_sqm)
        bp = polys[big.id].representative_point()
        yaw = math.atan2(-(bp.x - sp.x), (bp.y - sp.y)) if big.id != start.id else 0.0

        xs = [p[0] for r in f.rooms for p in r.polygon]
        ys = [p[1] for r in f.rooms for p in r.polygon]
        floors.append({
            "level": f.level,
            "spawn": [round(sp.x, 3), round(sp.y, 3)],
            "spawn_yaw": round(yaw, 3),
            "center": [round((min(xs) + max(xs)) / 2, 3), round((min(ys) + max(ys)) / 2, 3)],
            "extent": round(max(max(xs) - min(xs), max(ys) - min(ys)), 3),
            "rooms": [{"id": r.id, "type": r.type, "area_sqm": r.area_sqm,
                       "label": [round(polys[r.id].representative_point().x, 3),
                                 round(polys[r.id].representative_point().y, 3)]}
                      for r in f.rooms],
        })

    listing = dict(listing or {})
    start_level = 0 if any(f.level == 0 for f in plan.floors) else plan.floors[0].level
    return {
        "storey_height_m": wall_h + FLOOR_THICKNESS_M,   # must match plan_to_glb stacking
        "wall_height_m": wall_h,
        "start_level": start_level,
        "source_warning": plan.source_warning,
        "listing": {
            "property_id": listing.get("property_id") or plan.listing_id,
            "title": listing.get("title"),
            "location": listing.get("location"),
            "area_sqm": listing.get("pf_area_sqm") or listing.get("area_sqm"),
            "bedrooms": listing.get("pf_bedrooms") or listing.get("bedrooms"),
            "bathrooms": listing.get("pf_bathrooms") or listing.get("bathrooms"),
        },
        "floors": floors,
    }


def _json_for_script(obj) -> str:
    # safe inside <script type="application/json">
    return json.dumps(obj, ensure_ascii=False, default=str).replace("</", "<\\/")


def render_walkthrough_html(plan: Plan, glb_bytes: bytes, listing: Optional[Mapping] = None,
                            standalone: bool = True) -> str:
    """standalone=True wraps the page in <!doctype html>; False returns the body fragment."""
    hints = viewer_hints(plan, listing)
    title = f"Walkthrough {hints['listing']['property_id'] or plan.plan_id}"
    html = (TEMPLATE.read_text(encoding="utf-8")
            .replace("__PAGE_TITLE__", title)
            .replace("__PLAN_JSON__", _json_for_script(plan.model_dump()))
            .replace("__VIEWER_JSON__", _json_for_script(hints))
            .replace("__GLB_B64__", base64.b64encode(glb_bytes).decode("ascii")))
    if standalone:
        html = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
                '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
                '</head>\n<body>\n' + html + "\n</body>\n</html>\n")
    return html


def build_walkthrough(plan_path, out_html, glb_path=None, listing: Optional[Mapping] = None,
                      standalone: bool = True) -> Path:
    plan = load_plan(plan_path)
    glb_path = Path(glb_path or Path(out_html).with_suffix(".glb"))
    if not glb_path.exists():
        export_glb(plan, glb_path)
    out_html = Path(out_html)
    out_html.write_text(render_walkthrough_html(plan, glb_path.read_bytes(), listing, standalone),
                        encoding="utf-8")
    return out_html


# if __name__ == "__main__":
#     import pandas as pd

#     if len(sys.argv) < 2:
#         sys.exit("usage: python -m backend.services.walkthrough <property_id> [--fragment]")
#     pid = sys.argv[1]
#     df = pd.read_csv(REPO_ROOT / "data" / "final_merged_dataset.csv", encoding="utf-8-sig")
#     row = df[df["property_id"] == pid].iloc[0].to_dict()
#     models = REPO_ROOT / "data" / "models"
#     models.mkdir(parents=True, exist_ok=True)
#     out = build_walkthrough(REPO_ROOT / row["plan_path"], models / f"{pid}_walkthrough.html",
#                             glb_path=models / f"{pid}.glb", listing=row,
#                             standalone="--fragment" not in sys.argv)
#     print(out)
