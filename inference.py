"""
Smart Semsar - run the whole furniture pipeline on one listing and open the result.
تجربة كل اللي اتعمل لحد دلوقتي على إعلان واحد: الطلبات -> توزيع الفرش -> المراجعة -> الجولة 3D.

    plan.json + preferences (+ your own furniture) -> placer -> validator -> walkthrough.html

Examples (from this folder):
    python inference.py PROP_1002
    python inference.py PROP_1007 --density full --must-have desk --exclude tv --dining-seats 4
    python inference.py PROP_1002 --prefs my_prefs.json          # e.g. an LLM's JSON answer
    python inference.py PROP_1002 --model chair.glb --kind armchair --width 0.9
    python inference.py PROP_1002 --photo sofa.jpg --kind sofa   # TripoSR on this machine (GPU or CPU, no account)
    python inference.py PROP_1002 --photo sofa.jpg --kind sofa --generator colab   # the Colab server
    python inference.py PROP_1002 --photo chair.jpg              # kind and width read from the photo (Gemini)
    python inference.py PROP_1002 --photo chair.jpg --kind armchair --generator trellis   # TRELLIS.2 (HF_TOKEN)
    python inference.py PROP_1002 --planner llm --style-brief "warm, light wood"         # Gemini places furniture

Everything is written to out/<property id>/ ; nothing under data/ is touched.
The same run() is used by the demo page (app.py).
"""
import argparse
import json
import sys
import time
import webbrowser
from collections import Counter
from pathlib import Path
from typing import Callable, Optional

import pandas as pd

from backend.schema.design import DesignPreferences
from backend.schema.plan import load_plan
from backend.schema.staging import load_catalog, save_staging
from backend.services import vision_service
from backend.services.colab_service import ColabError
from backend.services.custom_furniture import add_furniture_from_photo, add_furniture_model
from backend.services.furniture_actions import apply_actions
from backend.services.furniture_placer import stage_plan
from backend.services.llm_furnisher import furnish
from backend.services.trellis_service import TrellisError
from backend.services.triposr_local import TripoSRError
from backend.services.staging_validator import validate_staging
from backend.services.walkthrough import build_walkthrough

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "out"


class PipelineError(RuntimeError):
    """The run cannot continue; the message is meant for the user."""


def load_listings() -> pd.DataFrame:
    return pd.read_csv(ROOT / "data" / "final_merged_dataset.csv", encoding="utf-8-sig")


def place_unplaced(plan, staging, catalog):
    """Pieces that were asked for but did not fit the rules (often the user's own piece):
    try once more with placement instructions - beside the main piece of the room, else in
    open floor. -> (new staging, names placed now)"""
    placed = []
    for name in list(staging.unplaced):
        item = next((i for i in catalog.items if i.is_a([name])), None)
        if item is None:
            continue
        for room in (r for r in plan.rooms() if r.type in item.room_types):
            anchors = [i for i in staging.in_room(room.id) if i.catalog_id.split("_")[0] in ("sofa", "bed")]
            tries = [[{"item": item.id, "place": "beside", "ref": a.id, "side": side}]
                     for a in anchors for side in ("right", "left")]
            tries.append([{"item": item.id, "place": "free", "min_clearance_m": 0.4}])
            for actions in tries:
                new, errors = apply_actions(plan, staging, {room.id: actions}, catalog)
                if not errors:
                    staging = new.model_copy(update={"unplaced": [u for u in new.unplaced if u != name]})
                    placed.append(name)
                    break
            if name in placed:
                break
    return staging, placed


def run(property_id: str, prefs: Optional[DesignPreferences] = None, photo=None, model=None,
        kind: Optional[str] = None, width_m: Optional[float] = None, yaw_deg: float = 0.0,
        out_dir=OUT_DIR, say: Callable[[str], None] = print, generator: str = "local",
        planner: str = "rules") -> Optional[Path]:
    """One full run. Reports each step through say(); returns the walkthrough page,
    or None if the validator rejected the staging."""
    started = time.time()
    prefs = prefs or DesignPreferences()

    # 1) the listing and its plan
    listings = load_listings()
    rows = listings[listings["property_id"] == property_id]
    if rows.empty:
        raise PipelineError(f"No listing {property_id}.")
    listing = rows.iloc[0].to_dict()
    plan_path = ROOT / listing["plan_path"]
    plan = load_plan(plan_path)
    out_dir = Path(out_dir) / property_id
    say(f"1. plan      {listing['title'].strip()}")
    say(f"             {plan.total_area_sqm:.0f} m2, {plan.num_floors} floor(s), {len(plan.rooms())} rooms, "
        f"{len(plan.fixtures())} built-in fixtures")

    # 2) what the user asked for
    catalog = load_catalog()
    both = sorted(set(prefs.must_have) & set(prefs.exclude))
    if both:      # ticked as "must have" and "leave out": must have wins
        prefs = prefs.model_copy(update={"exclude": [n for n in prefs.exclude if n not in both]})
    say(f"2. wishes    {prefs.model_dump(exclude_defaults=True) or 'none (defaults)'}")
    if both:
        say(f"             {both} was both 'must have' and 'leave out'; kept it as must have")
    unknown = prefs.unknown_names(catalog.vocabulary())
    if unknown:
        say(f"             not catalog names, will be reported as unplaced: {unknown}")

    # 3) the user's own furniture (optional)
    if photo and not (kind and width_m):
        # the vision model says what the photo is; anything the user stated wins over its answer
        if vision_service.is_configured():
            try:
                seen = vision_service.analyze_photo(photo, catalog)
            except vision_service.VisionError as e:
                if not kind:
                    raise PipelineError(f"Could not read the photo: {e} Choose what it is yourself "
                                        "('What is it?' / --kind) and run again.") from e
                seen = None
                say(f"3. photo     could not be read ({str(e)[:90]}); using your choice: {kind}")
            if seen is not None:
                say(f"3. photo     {seen.name or seen.scene}: kind {seen.kind}, about {seen.width_m} m wide, "
                    f"style {seen.style}")
                if not seen.usable and not (kind and seen.scene == "single_item"):
                    raise PipelineError(seen.reason)
                kind, width_m = kind or seen.kind, width_m or seen.width_m
        elif not kind:
            raise PipelineError(f"Say what the photo is (kind), or set {vision_service.KEY_ENV} "
                                "so it is recognised from the photo.")
    if photo or model:
        if not kind:
            raise PipelineError("Say what the model is (kind), e.g. sofa.")
        try:
            add = add_furniture_from_photo if photo else add_furniture_model
            extra = {"generator": generator} if photo else {}
            if photo and generator == "local":
                from backend.services import triposr_local
                if triposr_local.is_available() and not triposr_local.is_loaded():
                    say(f"3. 3D model  TripoSR on this machine ({triposr_local.device()}); the first photo also "
                        "downloads and loads the model (1-2 min)")
                extra["say"] = lambda m: say(f"             {m}")
            catalog, item = add(photo or model, kind, out_dir / "custom", catalog, width_m=width_m,
                                yaw_deg=yaw_deg, **extra)
        except (ColabError, TrellisError, TripoSRError, KeyError, ValueError) as e:
            raise PipelineError(str(e)) from e
        say(f"3. your item {item.name}: {item.width_m} x {item.depth_m} x {item.height_m} m "
            f"(replaces {item.id}) from {'photo via ' + generator if photo else 'model'}")
        if item.id not in prefs.must_have:      # the user's own piece should appear in the walkthrough
            prefs = prefs.model_copy(update={"must_have": prefs.must_have + [item.id]})
        if item.is_a(prefs.exclude):            # ... and "leave out" never removes it
            prefs = prefs.model_copy(update={"exclude": [n for n in prefs.exclude if not item.is_a([n])]})
    else:
        say("3. your item none (stock catalog)")

    # 4) place and check
    if planner == "llm":
        staging, report = furnish(plan, prefs, catalog)
        say(f"4. arranged  by {report.planner}" + (f" ({report.rounds} round(s))" if report.rounds else ""))
        for w in report.warnings:
            say(f"             {w}")
        if report.summary:
            say(f"             model says: {report.summary}")
    else:
        staging = stage_plan(plan, catalog, preferences=prefs)
    if staging.unplaced:
        staging, rescued = place_unplaced(plan, staging, catalog)
        if rescued:
            say(f"             placed with instructions after the rules: {rescued}")
    result = validate_staging(plan, staging, catalog)
    room_type = {r.id: r.type for r in plan.rooms()}
    say(f"4. placed    {len(staging.items)} items")
    for rtype in sorted({room_type[i.room_id] for i in staging.items}):
        counts = Counter(i.catalog_id for i in staging.items if room_type[i.room_id] == rtype)
        say(f"             {rtype:8s} " + ", ".join(f"{n} x {k}" if n > 1 else k for k, n in sorted(counts.items())))
    if staging.unplaced:
        say(f"             asked for but did not fit: {staging.unplaced}")
    say(f"5. checked   {'APPROVED' if result['approved'] else 'REJECTED'}")
    for check in result["checks"]:
        if not check["passed"]:
            say(f"             FAIL {check['name']}: {check['message']} {check['actual']}")
    for warning in result["warnings"]:
        say(f"             warning: {warning}")
    save_staging(staging, out_dir / "staging.json")
    if not result["approved"]:
        say(f"   staging saved for inspection: {out_dir / 'staging.json'} (no walkthrough built)")
        return None

    # 5) the 3D walkthrough
    html = build_walkthrough(plan_path, out_dir / "walkthrough.html", glb_path=out_dir / "architecture.glb",
                             listing=listing, staging=staging, catalog=catalog)
    say(f"6. built     {html}  ({html.stat().st_size / 1e6:.1f} MB, {time.time() - started:.1f} s)")
    return html


def read_preferences(args) -> DesignPreferences:
    """--prefs file first, then the single flags on top of it."""
    data = json.loads(Path(args.prefs).read_text(encoding="utf-8")) if args.prefs else {}
    for name in ("style", "density", "dining_seats", "style_brief"):
        if getattr(args, name) is not None:
            data[name] = getattr(args, name)
    for name in ("must_have", "exclude"):
        if getattr(args, name):
            data[name] = list(data.get(name, [])) + getattr(args, name)
    return DesignPreferences.from_loose(data)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("property_id", help="e.g. PROP_1002 (see data/final_merged_dataset.csv)")
    ap.add_argument("--prefs", help="JSON file with DesignPreferences fields")
    ap.add_argument("--style", help="modern / classic / minimal / industrial")
    ap.add_argument("--style-brief", dest="style_brief", help="free text describing the wanted look")
    ap.add_argument("--density", help="minimal / normal / full")
    ap.add_argument("--must-have", dest="must_have", nargs="*", help="catalog names to force, e.g. desk armchair")
    ap.add_argument("--exclude", nargs="*", help="catalog names to leave out, e.g. tv")
    ap.add_argument("--dining-seats", dest="dining_seats", type=int, help="4 or 6")
    ap.add_argument("--photo", help="photo of ONE piece of furniture (turned into 3D by --generator)")
    ap.add_argument("--generator", default="local", choices=["local", "colab", "trellis"],
                    help="photo -> 3D: local = TripoSR on this machine (GPU or CPU, no account), "
                         "colab = TripoSR on our Colab server, trellis = TRELLIS.2 on Hugging Face (GPU quota)")
    ap.add_argument("--planner", default="rules", choices=["rules", "llm"],
                    help="rules = furniture_placer; llm = Gemini/Groq chooses and arranges (falls back to rules)")
    ap.add_argument("--model", help="a GLB to use as your own furniture instead of a photo")
    ap.add_argument("--kind", help="what the photo / model is: sofa, armchair, bed, desk, ...")
    ap.add_argument("--width", type=float, help="real width of your piece in meters (default: the stock item's)")
    ap.add_argument("--yaw", type=float, default=0.0, help="turn your piece by this many degrees if it faces sideways")
    ap.add_argument("--out", default=str(OUT_DIR), help="output folder")
    ap.add_argument("--no-open", action="store_true", help="do not open the browser")
    args = ap.parse_args()
    try:
        html = run(args.property_id, read_preferences(args), photo=args.photo, model=args.model, kind=args.kind,
                   width_m=args.width, yaw_deg=args.yaw, out_dir=args.out, generator=args.generator,
                   planner=args.planner)
    except PipelineError as e:
        print(f"error: {e}")
        return 1
    if html is None:
        return 1
    if not args.no_open:
        webbrowser.open(html.resolve().as_uri())
    return 0


if __name__ == "__main__":
    sys.exit(main())
