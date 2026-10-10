"""
Furnish a whole apartment with the user's own furniture: several photos -> several 3D pieces.
فرش الشقة كلها بعفش المستخدم: كل صورة بتبقى قطعة 3D وبتاخد مكان قطعة من الكتالوج.

    pieces = assign_slots([{"label": "sofa.jpg", "kind": "sofa"}, ...], catalog)   # which catalog item each replaces
    raw = raw_model("sofa.jpg", "colab", RAW_DIR)                                  # photo -> GLB, cached per photo
    result = furnish_with_pieces(plan_path, [Piece(raw, "sofa_3_seat", 2.2)], out_dir, prefs)

Each piece takes the place of one catalog item of its kind (the first sofa photo replaces the 3-seat
sofa, a second sofa photo the 2-seat one), so every copy of that item in the apartment becomes the
user's. The rules (furniture_placer) or the LLM (llm_furnisher) then place everything,
staging_validator checks it, and the walkthrough is built.

Used by the furniture agent (backend/agents/furniture_node.py) and SmartSemsar_multi_photo.ipynb.
"""
import hashlib
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Optional

from backend.schema.design import DesignPreferences
from backend.schema.plan import load_plan
from backend.schema.staging import Catalog, CatalogItem, Staging, load_catalog, save_staging
from backend.services import colab_service, trellis_service
from backend.services.custom_furniture import FRONT_TURN_DEG, add_furniture_model
from backend.services.furniture_actions import place_unplaced
from backend.services.furniture_placer import stage_plan
from backend.services.llm_furnisher import furnish as llm_furnish
from backend.services.staging_validator import validate_staging
from backend.services.walkthrough import build_walkthrough

GENERATORS = {"colab": colab_service, "trellis": trellis_service}


@dataclass
class Piece:
    """One of the user's own pieces, already a raw GLB."""
    raw: str                            # raw GLB from image-to-3D or an upload
    slot: str                           # catalog id or kind it replaces, e.g. "sofa_3_seat" / "sofa"
    width_m: Optional[float] = None     # real width; None = the stock item's width
    yaw_deg: float = 0.0                # turn it if it faces sideways
    label: str = ""                     # for the report, e.g. the photo's file name


@dataclass
class FurnishResult:
    staging: Staging
    catalog: Catalog                    # holds the user's pieces; needed to show the staging
    approved: bool
    failed_checks: list[str]
    staging_path: Path
    walkthrough: Optional[Path]         # None when the validator rejected the staging
    mine: dict[str, CatalogItem] = field(default_factory=dict)    # piece label -> its fitted catalog item
    prefs: DesignPreferences = field(default_factory=DesignPreferences)


def raw_model(photo, generator: str, raw_dir, say: Callable[[str], None] = print) -> Path:
    """Photo -> raw GLB with "colab" (the diffusion server) or "trellis". Made once per photo and
    generator: running again with the same photo (e.g. after changing a width) reuses it."""
    digest = hashlib.sha1(Path(photo).read_bytes()).hexdigest()[:12]
    raw = Path(raw_dir) / f"{generator}_{digest}.glb"
    if raw.exists():
        say("     3D model reused (same photo as before)")
        return raw
    raw.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    GENERATORS[generator].generate_model(photo, raw)
    say(f"     3D model made in {time.time() - started:.0f} s")
    return raw


def slot_name(text: str) -> str:
    """"Console table" / "console-table" -> console_table"""
    return "_".join((text or "").strip().lower().replace("-", " ").split())


def assign_slots(pieces: list[dict], catalog: Catalog, say: Callable[[str], None] = print) -> list[dict]:
    """pieces: [{"label", "kind", ...}] -> the same dicts plus "slot", the catalog item each one replaces.
    The first piece of a kind takes the first catalog item of that kind, the next one the next item.
    "kind" may also be a catalog id (sofa_2_seat) to choose the item. Pieces that fit nowhere are
    skipped and reported."""
    kinds, ids = {i.kind for i in catalog.items}, {i.id for i in catalog.items}
    used, out = set(), []
    for n, p in enumerate(pieces, 1):
        want = slot_name(p.get("kind") or "")
        if not want:
            say(f"  {n}. {p.get('label', '')}: skipped (say what it is)")
            continue
        if want not in kinds and want not in ids:
            say(f"  {n}. {p.get('label', '')}: skipped ('{want}' is not one of {sorted(kinds)})")
            continue
        free = [i for i in catalog.items if (i.kind == want or i.id == want) and i.id not in used]
        if not free:
            say(f"  {n}. {p.get('label', '')}: skipped (all {want} models are already taken by other photos)")
            continue
        used.add(free[0].id)
        out.append({**p, "slot": free[0].id})
    return out


def furnish_with_pieces(plan_path, pieces: list[Piece], out_dir, prefs: Optional[DesignPreferences] = None,
                        planner: str = "rules", generator: Optional[str] = None, listing: Optional[Mapping] = None,
                        glb_path=None, say: Callable[[str], None] = print) -> FurnishResult:
    """Put the user's pieces in the catalog, furnish the plan, check it, save it and build the walkthrough.
    generator: what made the raw GLBs ("colab" models face backwards, so they are turned to match).
    planner: "rules" or "llm" (the LLM falls back to the rules when no model answers)."""
    started = time.time()
    plan = load_plan(plan_path)
    out_dir = Path(out_dir)
    catalog = load_catalog()
    prefs = prefs or DesignPreferences()

    mine: dict[str, CatalogItem] = {}
    for p in pieces:
        yaw = (p.yaw_deg + FRONT_TURN_DEG.get(generator or "", 0.0)) % 360
        label = p.label or Path(p.raw).name
        try:
            catalog, item = add_furniture_model(p.raw, p.slot, out_dir / "custom", catalog, width_m=p.width_m,
                                                yaw_deg=yaw, name=f"Your {p.slot.replace('_', ' ')}")
        except (KeyError, ValueError) as e:
            say(f"  {label}: not used ({e})")
            continue
        mine[label] = item
        say(f"  {label} -> {item.id}: {item.width_m} x {item.depth_m} x {item.height_m} m")

    # the user's pieces always appear, and "leave out" never removes them; ticked twice: must have wins
    both = set(prefs.must_have) & set(prefs.exclude)
    if both:
        say(f"{sorted(both)} was both 'must have' and 'leave out': kept it as must have")
    items = list(mine.values())
    prefs = prefs.model_copy(update={
        "must_have": prefs.must_have + [i.id for i in items if i.id not in prefs.must_have],
        "exclude": [n for n in prefs.exclude if n not in both and not any(i.is_a([n]) for i in items)]})
    say(f"Wishes: {prefs.model_dump(exclude_defaults=True) or 'none (defaults)'}")
    unknown = prefs.unknown_names(catalog.vocabulary())
    if unknown:
        say(f"  not catalog names, will be reported as unplaced: {unknown}")

    if planner == "llm":
        staging, report = llm_furnish(plan, prefs, catalog)
        say(f"Arranged by {report.planner}" + (f" ({report.rounds} round(s))" if report.rounds else ""))
        for w in report.warnings:
            say(f"  {w}")
        if report.summary:
            say(f"  model says: {report.summary}")
    else:
        staging = stage_plan(plan, catalog, preferences=prefs)
    if staging.unplaced:
        staging, rescued = place_unplaced(plan, staging, catalog)
        if rescued:
            say(f"Placed with extra rules: {rescued}")

    ids = {i.id for i in items}
    room_type = {r.id: r.type for r in plan.rooms()}
    say(f"Placed {len(staging.items)} items, {sum(i.catalog_id in ids for i in staging.items)} of them yours:")
    for rtype in sorted({room_type[i.room_id] for i in staging.items}):
        counts = Counter(i.catalog_id for i in staging.items if room_type[i.room_id] == rtype)
        say(f"  {rtype:8s} " + ", ".join((f"{n} x " if n > 1 else "") + (f"YOUR {k}" if k in ids else k)
                                          for k, n in sorted(counts.items())))
    if staging.unplaced:
        say(f"Did not fit: {staging.unplaced}")

    result = validate_staging(plan, staging, catalog)
    failed = [f"{c['name']}: {c['message']} {c['actual']}" for c in result["checks"] if not c["passed"]]
    say("Check: " + ("APPROVED" if result["approved"] else "REJECTED"))
    for f in failed:
        say(f"  FAIL {f}")
    staging_path = save_staging(staging, out_dir / "staging.json")
    walkthrough = None
    if result["approved"]:
        walkthrough = build_walkthrough(plan_path, out_dir / "walkthrough.html",
                                        glb_path=glb_path or out_dir / "architecture.glb",
                                        listing=listing, staging=staging, catalog=catalog)
        say(f"Walkthrough built: {walkthrough.stat().st_size / 1e6:.1f} MB in {time.time() - started:.0f} s")
    return FurnishResult(staging, catalog, result["approved"], failed, staging_path, walkthrough, mine, prefs)
