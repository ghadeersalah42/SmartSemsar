"""
Photo benchmark: real furniture photos (poor lighting, odd angles, clutter) -> does the pipeline get them right?
بيجرب الـ pipeline على صور عفش حقيقية (إضاءة وحشة، زوايا غريبة، زحمة) ويحسب نسبة النجاح.

The labelled set is data/eval/furniture_photos.csv (photos stay outside the repo, under --photos-root):
    photo            path under the photos root
    expected_kinds   catalog kinds that count as right, "|"-separated; empty = the photo must be refused
                     (a whole room, several pieces, a built-in kitchen ...)
    conditions       "|"-separated: low_light, odd_angle, clutter, cut_off, glare, normal
    notes            what a person sees in the photo

A photo succeeds when:
    expected_kinds set   -> vision_service says usable with one of those kinds, and (with --generator)
                            the 3D model is made and fit_model gives a sane size
    expected_kinds empty -> vision_service refuses it (usable = False)

Steps:
    python -m backend.eval.photo_benchmark select --photos-root "D:/.../ترابيزات" --n 30
        -> a draft CSV with the darkest / lowest-contrast photos measured; check every row by eye
    python -m backend.eval.photo_benchmark run --photos-root "D:/.../ترابيزات" [--generator trellis]
        -> reports/furnishing/photo_benchmark.json and .md
Run "run" twice with a generator: the second run shows the 3D cache (same photo -> same model).
"""
import argparse
import csv
import json
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

from PIL import Image, ImageStat

from backend.schema.staging import REPO_ROOT, load_catalog
from backend.services import furnishing_metrics

LABELS = REPO_ROOT / "data" / "eval" / "furniture_photos.csv"
REPORT_DIR = REPO_ROOT / "reports" / "furnishing"
RAW_DIR = REPO_ROOT / "data" / "eval" / "raw_3d"        # 3D cache for the benchmark (not committed)
FIELDS = ["photo", "expected_kinds", "conditions", "notes"]
LOW_LIGHT_LUMA = 85         # mean brightness (0-255) below this is a dark photo
LOW_CONTRAST_STD = 45
# the photo folders on the team drive: the catalog kinds their pieces can be, and how many to pick of 30
FOLDER_KINDS = {
    "ترابيزات": ("coffee_table|dining_table|console_table|desk", 6),
    "ركنة و انتريه": ("sofa|armchair", 6),
    "كراسي": ("dining_chair|armchair", 6),
    "غرفة نوم": ("bed|wardrobe|dresser|nightstand", 5),
    "غرفة أطفال": ("bed|wardrobe|desk|bookshelf|dresser", 4),
    "مطابخ": ("", 3),
}
IMAGE_TYPES = {".jpg", ".jpeg", ".png", ".webp"}


def light(path) -> tuple[float, float]:
    """(mean brightness, contrast) of the photo, 0-255."""
    with Image.open(path) as im:
        im = im.convert("L")
        im.thumbnail((256, 256))
        stat = ImageStat.Stat(im)
        return stat.mean[0], stat.stddev[0]


def read_labels(path=LABELS) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as f:
        return [r for r in csv.DictReader(f) if r.get("photo") and not r["photo"].startswith("#")]


def split(value: Optional[str]) -> list[str]:
    return [v.strip() for v in (value or "").split("|") if v.strip()]


def select(root: Path, n: int, seed: int = 7) -> list[dict]:
    """Draft rows: about half the darkest / flattest photos, the rest at random, spread over the folders."""
    rows = []
    quota_total = sum(q for _, q in FOLDER_KINDS.values())
    for folder, (kinds, quota) in FOLDER_KINDS.items():
        photos = sorted(p for p in (root / folder).glob("*") if p.suffix.lower() in IMAGE_TYPES)
        rng = random.Random(f"{seed}{folder}")
        measured = [(p, *light(p)) for p in photos]
        dark = sorted(measured, key=lambda m: m[1])
        share = max(1, round(quota * n / quota_total))
        picked = dark[:share // 2 + share % 2]
        rest = [m for m in measured if m not in picked]
        picked += rng.sample(rest, min(len(rest), share - len(picked)))
        for p, luma, std in picked:
            tags = (["low_light"] if luma < LOW_LIGHT_LUMA else []) + (["low_contrast"] if std < LOW_CONTRAST_STD else [])
            rows.append({"photo": p.relative_to(root).as_posix(), "expected_kinds": kinds,
                         "conditions": "|".join(tags), "notes": f"CHECK BY EYE; luma {luma:.0f}, contrast {std:.0f}"})
    return rows[:n]


def evaluate(row: dict, root: Path, generator: Optional[str], catalog) -> dict:
    from backend.services.custom_furniture import FRONT_TURN_DEG, add_furniture_model
    from backend.services.multi_furnish import raw_model
    from backend.services.vision_service import VisionError, analyze_photo

    photo = root / row["photo"]
    expected = split(row["expected_kinds"])
    out = {"photo": row["photo"], "expected_kinds": expected, "conditions": split(row["conditions"]) or ["normal"]}
    started = time.perf_counter()
    try:
        seen = analyze_photo(photo, catalog)
    except VisionError as e:
        return {**out, "success": False, "stage": "vision_error", "error": str(e)[:200],
                "vision_s": round(time.perf_counter() - started, 1)}
    out.update(vision_s=round(time.perf_counter() - started, 1), usable=seen.usable, kind=seen.kind,
               scene=seen.scene, reason=seen.reason, width_m=seen.width_m)
    if not expected:
        return {**out, "success": not seen.usable, "stage": "refused" if not seen.usable else "wrongly_accepted"}
    if not seen.usable:
        return {**out, "success": False, "stage": f"refused:{seen.scene}"}
    if seen.kind not in expected:
        return {**out, "success": False, "stage": "wrong_kind"}
    if not generator:
        return {**out, "success": True, "stage": "recognised"}

    started = time.perf_counter()
    try:
        raw = raw_model(photo, generator, RAW_DIR, say=lambda _: None)
        out["model_3d_s"] = round(time.perf_counter() - started, 1)
        _, item = add_furniture_model(raw, seen.kind, RAW_DIR / "fitted" / Path(row["photo"]).stem, catalog,
                                      width_m=seen.width_m, yaw_deg=FRONT_TURN_DEG.get(generator, 0.0))
    except ValueError as e:              # fit_model: the model came out with an absurd size
        return {**out, "success": False, "stage": "bad_3d_size", "error": str(e)[:200]}
    except Exception as e:               # server down, quota, timeout
        return {**out, "success": False, "stage": "3d_error", "error": f"{type(e).__name__}: {str(e)[:200]}",
                "model_3d_s": round(time.perf_counter() - started, 1)}
    return {**out, "success": True, "stage": "3d_ok", "size_m": [item.width_m, item.depth_m, item.height_m]}


def vision_model() -> str:
    from backend.config import setting
    from backend.services.vision_service import DEFAULT_MODEL, MODEL_ENV
    return setting(MODEL_ENV, DEFAULT_MODEL)


def summarize(results: list[dict]) -> dict:
    def rate(rs):
        return {"photos": len(rs), "success": sum(r["success"] for r in rs),
                "rate": round(sum(r["success"] for r in rs) / len(rs), 3) if rs else None}
    by_condition = defaultdict(list)
    for r in results:
        for c in r["conditions"]:
            by_condition[c].append(r)
    stages = defaultdict(int)
    for r in results:
        stages[r["stage"]] += 1
    vision = [r["vision_s"] for r in results if "vision_s" in r]
    return {"overall": rate(results),
            "pieces": rate([r for r in results if r["expected_kinds"]]),
            "must_refuse": rate([r for r in results if not r["expected_kinds"]]),
            "by_condition": {c: rate(rs) for c, rs in sorted(by_condition.items())},
            "outcomes": dict(sorted(stages.items(), key=lambda kv: -kv[1])),
            "vision_s_median": sorted(vision)[len(vision) // 2] if vision else None}


def to_markdown(s: dict, model_3d: dict, generator: Optional[str]) -> str:
    pct = lambda d: f"{d['success']} / {d['photos']} ({(d['rate'] or 0) * 100:.0f} %)" if d["photos"] else "-"
    lines = ["# Furnishing: real furniture photos", "",
             f"- Success (whole set): **{pct(s['overall'])}**"
             + ("" if generator else "  - recognition only, no 3D generator in this run"),
             f"- Pieces recognised{' and turned into 3D' if generator else ''}: {pct(s['pieces'])}",
             f"- Photos correctly refused (room, several pieces, kitchen): {pct(s['must_refuse'])}",
             f"- Vision model: {vision_model()}; median time per photo: {s['vision_s_median']} s", "",
             "| Condition | Success |", "|---|---:|"]
    lines += [f"| {c} | {pct(d)} |" for c, d in s["by_condition"].items()]
    lines += ["", "| Outcome | Photos |", "|---|---:|"] + [f"| {k} | {n} |" for k, n in s["outcomes"].items()]
    if model_3d:
        lines += ["", "## 3D pieces", "", "| Generator | Calls | Generated | Mean s | Median s | P95 s | Cache hit rate |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        lines += [f"| {g} | {d['calls']} | {d['generated']} | {d['seconds_mean']} | {d['seconds_median']} | "
                  f"{d['seconds_p95']} | {(d['cache_hit_rate'] or 0) * 100:.0f} % |" for g, d in model_3d.items()]
    return "\n".join(lines) + "\n"


def main(argv: Optional[list[str]] = None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sel = sub.add_parser("select", help="draft a labelled set from the photo folders")
    run = sub.add_parser("run", help="evaluate the labelled set")
    for p in (sel, run):
        p.add_argument("--photos-root", required=True, type=Path)
        p.add_argument("--labels", type=Path, default=LABELS)
    sel.add_argument("--n", type=int, default=30)
    run.add_argument("--generator", choices=["colab", "trellis"], default=None,
                     help="also build the 3D model of every recognised piece")
    run.add_argument("--only", default=None, help="comma-separated row numbers (1-based), e.g. 1,7,14")
    run.add_argument("--out", type=Path, default=REPORT_DIR)
    args = ap.parse_args(argv)

    if args.cmd == "select":
        rows = select(args.photos_root, args.n)
        args.labels.parent.mkdir(parents=True, exist_ok=True)
        with open(args.labels, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, FIELDS)
            w.writeheader()
            w.writerows(rows)
        print(f"{len(rows)} rows -> {args.labels}. Check every row by eye before running.")
        return rows

    catalog = load_catalog()
    started = time.time()
    results = []
    only = {int(n) for n in args.only.split(",")} if args.only else None
    for n, row in enumerate(read_labels(args.labels), 1):
        if only and n not in only:
            continue
        r = evaluate(row, args.photos_root, args.generator, catalog)
        results.append(r)
        print(f"{n:2d}. {'ok  ' if r['success'] else 'FAIL'} {r['stage']:20s} {r.get('kind') or '-':14s} {row['photo']}")
    summary = summarize(results)
    model_3d = furnishing_metrics.summarize_3d(e for e in furnishing_metrics.load() if e.get("ts", 0) >= started)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "photo_benchmark.json").write_text(json.dumps(
        {"summary": summary, "model_3d": model_3d, "generator": args.generator, "vision_model": vision_model(),
         "results": results},
        indent=1, ensure_ascii=False), encoding="utf-8")
    (args.out / "photo_benchmark.md").write_text(to_markdown(summary, model_3d, args.generator), encoding="utf-8")
    print(to_markdown(summary, model_3d, args.generator))
    return summary


if __name__ == "__main__":
    main()
