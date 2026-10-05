"""
plan.json + .glb -> self-contained walkthrough HTML (three.js, first-person).
جولة داخل الموديل: المشي بالكيبورد/اللمس، الحوائط بتمنع المرور والأبواب مفتوحة.

The page embeds the plan, the GLB (base64) and small viewer hints, so it opens
offline from disk or inside the app (e.g. gr.HTML / st.components.v1.html).
With a staging it also embeds the furniture; the viewer draws each item in its room
and stops the visitor walking through it. A staging the validator rejects is refused.
Each catalog model (GLB) the staging uses is embedded once; an item whose model file
is missing is drawn as a box of its size.

Usage:
    python -m backend.services.walkthrough PROP_1002            # empty rooms
    python -m backend.services.walkthrough PROP_1002 --staged   # furnished
    -> data/models/PROP_1002.glb + data/models/PROP_1002_walkthrough.html
       (--staged also writes data/stagings/PROP_1002.json)
"""
import base64
import json
import math
import sys
from pathlib import Path
from typing import Mapping, Optional

from shapely.geometry import Polygon
from shapely.ops import unary_union

from backend.schema.plan import Plan, load_plan
from backend.schema.staging import Catalog, Staging, load_catalog, save_staging
from backend.services.plan_to_glb import FLOOR_THICKNESS_M, export_glb
from backend.services.staging_validator import validate_staging

REPO_ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = REPO_ROOT / "frontend" / "walkthrough" / "viewer_template.html"
SPAWN_PREFERENCE = ["entry", "living", "hallway", "other", "kitchen", "dining"]
SPAWN_CLEAR_M = 0.4     # the visitor never starts closer than this to a piece of furniture


def viewer_hints(plan: Plan, listing: Optional[Mapping] = None, staging: Optional[Staging] = None) -> dict:
    """Spawn point / label points / floor extents the viewer needs (2D, plan meters)."""
    wall_h = max((w.height_m for f in plan.floors for w in f.walls), default=2.7)
    floors = []
    for f in plan.floors:
        polys = {r.id: Polygon(r.polygon).buffer(0) for r in f.rooms}
        furniture = unary_union([Polygon(i.footprint()).buffer(SPAWN_CLEAR_M)
                                 for i in (staging.items if staging else []) if i.level == f.level])
        indoor = [r for r in f.rooms if r.is_indoor] or f.rooms

        def rank(r):
            t = SPAWN_PREFERENCE.index(r.type) if r.type in SPAWN_PREFERENCE else len(SPAWN_PREFERENCE)
            return (t, -r.area_sqm)

        start = min(indoor, key=rank)
        # representative_point is always inside the polygon (works for L-shaped rooms)
        free = polys[start.id].difference(furniture)        # do not start inside a sofa
        sp = (polys[start.id] if free.is_empty else free).representative_point()
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


def staging_payload(staging: Staging, catalog: Optional[Catalog] = None) -> dict:
    """staging.json plus what the viewer needs from the catalog (name, colour, model).
    models: {catalog id: GLB as base64}, one entry per model used."""
    catalog = catalog or load_catalog()
    items, models = [], {}
    for i in staging.items:
        c = catalog.get(i.catalog_id)
        mesh_file = REPO_ROOT / c.mesh if c.mesh else None
        if mesh_file and mesh_file.exists() and c.id not in models:
            models[c.id] = base64.b64encode(mesh_file.read_bytes()).decode("ascii")
        items.append({**i.model_dump(), "name": c.name, "category": c.category,
                      "color": list(c.color), "mesh": c.mesh})
    return {"staging_id": staging.staging_id, "style": staging.style, "items": items, "models": models}


def render_walkthrough_html(plan: Plan, glb_bytes: bytes, listing: Optional[Mapping] = None,
                            standalone: bool = True, staging: Optional[Staging] = None,
                            catalog: Optional[Catalog] = None) -> str:
    """standalone=True wraps the page in <!doctype html>; False returns the body fragment.
    staging=None gives the empty rooms."""
    if staging is not None:
        result = validate_staging(plan, staging, catalog)
        if not result["approved"]:
            failed = "; ".join(f"{c['name']}: {c['actual']}" for c in result["checks"] if not c["passed"])
            raise ValueError(f"Staging {staging.staging_id} was rejected by the validator ({failed})")
    hints = viewer_hints(plan, listing, staging)
    title = f"Walkthrough {hints['listing']['property_id'] or plan.plan_id}"
    html = (TEMPLATE.read_text(encoding="utf-8")
            .replace("__PAGE_TITLE__", title)
            .replace("__PLAN_JSON__", _json_for_script(plan.model_dump()))
            .replace("__VIEWER_JSON__", _json_for_script(hints))
            .replace("__STAGING_JSON__", _json_for_script(staging_payload(staging, catalog) if staging else None))
            .replace("__GLB_B64__", base64.b64encode(glb_bytes).decode("ascii")))
    if standalone:
        html = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
                '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
                '</head>\n<body>\n' + html + "\n</body>\n</html>\n")
    return html


def build_walkthrough(plan_path, out_html, glb_path=None, listing: Optional[Mapping] = None,
                      standalone: bool = True, staging: Optional[Staging] = None,
                      catalog: Optional[Catalog] = None) -> Path:
    """catalog: the one the staging was made with (needed when it holds a user's own furniture)."""
    plan = load_plan(plan_path)
    glb_path = Path(glb_path or Path(out_html).with_suffix(".glb"))
    export_glb(plan, glb_path)        # always rebuild: plan.json is the source of truth
    out_html = Path(out_html)
    out_html.write_text(render_walkthrough_html(plan, glb_path.read_bytes(), listing, standalone, staging,
                                                catalog), encoding="utf-8")
    return out_html


if __name__ == "__main__":
    import pandas as pd

    if len(sys.argv) < 2:
        sys.exit("usage: python -m backend.services.walkthrough <property_id> [--fragment] [--staged]")
    pid = sys.argv[1]
    df = pd.read_csv(REPO_ROOT / "data" / "final_merged_dataset.csv", encoding="utf-8-sig")
    row = df[df["property_id"] == pid].iloc[0].to_dict()
    models = REPO_ROOT / "data" / "models"
    models.mkdir(parents=True, exist_ok=True)
    staging = None
    if "--staged" in sys.argv:
        from backend.services.furniture_placer import stage_plan
        staging = stage_plan(load_plan(REPO_ROOT / row["plan_path"]))
        save_staging(staging, REPO_ROOT / "data" / "stagings" / f"{pid}.json")
    out = build_walkthrough(REPO_ROOT / row["plan_path"], models / f"{pid}_walkthrough.html",
                            glb_path=models / f"{pid}.glb", listing=row,
                            standalone="--fragment" not in sys.argv, staging=staging)
    print(out)
