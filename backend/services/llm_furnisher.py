"""
Furnish a plan with an LLM: the model chooses and arranges, the code measures and checks.
الموديل بيختار العفش ويقول مكانه بالكلام؛ الكود بيحسب المكان، والـ validator بيراجع، والغلط بيرجع للموديل يصلحه.

    staging, report = furnish(plan, preferences={"style": "modern", "must_have": ["desk"]})

Flow per call:
  1. every furnishable room is described in words (furniture_actions.describe_room)
  2. the model answers with placement instructions (JSON)
  3. furniture_actions.apply_actions places them; each one must pass staging_validator
  4. what failed goes back to the model, up to MAX_ROUNDS times
  5. rooms the model left empty are furnished by the rule placer (furniture_placer)
No model configured or reachable -> the rule placer does everything (report says so).

Model (OpenAI-compatible chat API), first one configured wins:
  GEMINI_API_KEY   -> Gemini (SMARTSEMSAR_GEMINI_MODEL, default gemini-flash-latest)
  GROQ_API_KEY     -> Groq   (SMARTSEMSAR_GROQ_MODEL, default llama-3.3-70b-versatile)
  OLLAMA_BASE_URL  -> local Ollama (SMARTSEMSAR_OLLAMA_MODEL, default qwen2.5)

Usage:
    python -m backend.services.llm_furnisher PROP_1002 [--notes "home office, light wood"]
"""
import json
import sys
from dataclasses import dataclass, field
from typing import Optional, Union

import requests

from backend.config import mask_secrets, setting
from backend.schema.design import DesignPreferences
from backend.schema.plan import Plan, load_plan
from backend.schema.staging import REPO_ROOT, Catalog, Staging, load_catalog, save_staging
from backend.services.furniture_actions import apply_actions, describe_room
from backend.services.furniture_placer import stage_plan
from backend.services.staging_validator import validate_staging

MAX_ROUNDS = 3

SYSTEM_PROMPT = """You are an interior designer furnishing an empty apartment for a real-estate preview.
Each room is described with lettered walls (A, B, ...): their length, doors, windows, built-ins and how
much free wall is left. Use only catalog items, and say where each piece goes with these instructions:

  {"item": "<catalog id>", "place": "wall", "wall": "<letter>", "align": "center|start|end", "name": "<label>"}
  {"item": "<catalog id>", "place": "wall", "wall": "<letter>", "align_to": "<label>"}
  {"item": "<catalog id>", "place": "beside", "ref": "<label>", "side": "both|left|right"}
  {"item": "<catalog id>", "place": "in_front_of", "ref": "<label>", "gap_m": 0.45, "face_ref": false}
  {"item": "<catalog id>", "place": "center"}   or   {"item": "<catalog id>", "place": "free"}
  {"remove": "<placed item id or label>"}

"name" labels a piece so later instructions can refer to it. face_ref true turns a piece to face the
reference (a free-standing TV unit facing the sofa).

Design rules: beds, sofas and storage stand with their back against a wall; a piece only goes on a
wall whose free wall is at least its width; nothing tall in front of a window; never block a door
or the way between doors; kitchens and bathrooms already have their built-ins; sides marked "open"
have no wall. Follow the customer's style and wishes. A few well-placed pieces beat a crowded room.

Answer with JSON only: {"rooms": {"<room id>": [<instruction>, ...]}, "summary": "<one sentence for the customer>"}"""


@dataclass
class LLMClient:
    name: str
    base_url: str
    model: str
    api_key: str = ""
    timeout: float = 90.0

    def chat_json(self, messages: list[dict]) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        body = {"model": self.model, "messages": messages, "temperature": 0.2,
                "response_format": {"type": "json_object"}}
        r = requests.post(f"{self.base_url}/chat/completions", headers=headers, json=body, timeout=self.timeout)
        r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"]
        return json.loads(text[text.find("{"):text.rfind("}") + 1])


def detect_llms() -> list[LLMClient]:
    """Every configured model, best first: Gemini, Groq, local Ollama."""
    found = []
    if setting("GEMINI_API_KEY"):
        model = setting("SMARTSEMSAR_GEMINI_MODEL", "gemini-flash-latest")
        found.append(LLMClient(f"gemini/{model}", "https://generativelanguage.googleapis.com/v1beta/openai",
                               model, setting("GEMINI_API_KEY")))
    if setting("GROQ_API_KEY"):
        model = setting("SMARTSEMSAR_GROQ_MODEL", "llama-3.3-70b-versatile")
        found.append(LLMClient(f"groq/{model}", "https://api.groq.com/openai/v1", model, setting("GROQ_API_KEY")))
    base = setting("OLLAMA_BASE_URL", "http://localhost:11434/v1").rstrip("/")
    try:
        requests.get(f"{base}/models", timeout=1.5).raise_for_status()
        model = setting("SMARTSEMSAR_OLLAMA_MODEL", "qwen2.5")
        found.append(LLMClient(f"ollama/{model}", base, model, timeout=240.0))
    except requests.RequestException:
        pass
    return found


def detect_llm() -> Optional[LLMClient]:
    found = detect_llms()
    return found[0] if found else None


@dataclass
class FurnishReport:
    planner: str = "rules"                      # "llm:<model>", "llm+rules:<model>" or "rules"
    summary: Optional[str] = None               # the model's sentence for the customer
    rounds: int = 0
    unresolved: dict = field(default_factory=dict)   # room id -> what still failed after the last round
    rule_rooms: list = field(default_factory=list)   # rooms furnished by the rule placer
    warnings: list = field(default_factory=list)


def _catalog_text(catalog: Catalog, style: Optional[str], exclude: list[str]) -> str:
    lines = []
    for i in catalog.items:
        if (style and i.styles and style not in i.styles) or i.is_a(exclude):
            continue
        lines.append(f"- {i.id}: {i.name}, {i.width_m} x {i.depth_m} m, h {i.height_m} m; "
                     f"rooms: {', '.join(i.room_types)}{'; against a wall' if i.against_wall else ''}")
    return "\n".join(lines)


def _prefs_text(p: DesignPreferences, notes: Optional[str]) -> str:
    parts = [f"style: {p.style}", f"density: {p.density}"]
    if p.style_brief:
        parts.append(f"look: {p.style_brief}")
    if p.palette:
        parts.append(f"colours: {', '.join(p.palette)}")
    if p.materials:
        parts.append(f"materials: {', '.join(p.materials)}")
    if p.must_have:
        parts.append(f"must have: {', '.join(p.must_have)}")
    if p.exclude:
        parts.append(f"leave out: {', '.join(p.exclude)}")
    if p.dining_seats:
        parts.append(f"dining seats: {p.dining_seats}")
    if notes or p.notes:
        parts.append(f"customer notes: {notes or p.notes}")
    return "; ".join(parts)


def furnish(plan: Plan, preferences: Union[DesignPreferences, dict, None] = None,
            catalog: Optional[Catalog] = None, llm: Optional[LLMClient] = None, use_llm: bool = True,
            notes: Optional[str] = None, base: Optional[Staging] = None,
            staging_id: Optional[str] = None) -> tuple[Staging, FurnishReport]:
    catalog = catalog or load_catalog()
    prefs = preferences if isinstance(preferences, DesignPreferences) \
        else DesignPreferences.from_loose(preferences or {})
    style = prefs.style if prefs.style != "unknown" else None
    staging_id = staging_id or f"{plan.listing_id or plan.plan_id}_{style or 'auto'}"
    report = FurnishReport()
    backups: list = []
    if llm is None and use_llm:
        found = detect_llms()
        llm, backups = (found[0], found[1:]) if found else (None, [])

    room_types = {t for i in catalog.items for t in i.room_types}
    rooms = [r for r in plan.rooms() if r.type in room_types]
    staging = base.model_copy(deep=True) if base else Staging(
        staging_id=staging_id, plan_id=plan.plan_id, style=style, preferences=prefs.model_dump())

    if llm is None:
        if use_llm:
            report.warnings.append("No LLM configured (GEMINI_API_KEY / GROQ_API_KEY / Ollama): rules only.")
        rules = stage_plan(plan, catalog, style, prefs)
        rules.staging_id = staging_id
        report.rule_rooms = [r.id for r in rooms]
        return rules, report

    user = (f"Customer wishes: {_prefs_text(prefs, notes)}\n\nCatalog:\n{_catalog_text(catalog, style, prefs.exclude)}"
            "\n\nRooms:\n" + "\n\n".join(describe_room(plan, r.id, staging, catalog) for r in rooms))
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]
    pending = {r.id for r in rooms}
    for round_no in range(1, MAX_ROUNDS + 1):
        answer = None
        while answer is None:
            try:
                answer = llm.chat_json(messages)
            except Exception as e:  # busy, quota, network, bad JSON: next model, else rules do the rest
                report.warnings.append(f"{llm.name} failed in round {round_no} "
                                       f"({type(e).__name__}: {mask_secrets(e)[:120]}).")
                if not backups:
                    break
                llm = backups.pop(0)
                report.warnings.append(f"Switched to {llm.name}.")
        if answer is None:
            break
        report.rounds = round_no
        report.summary = answer.get("summary") or report.summary
        todo = {rid: acts for rid, acts in (answer.get("rooms") or {}).items() if rid in pending}
        if not todo:
            break
        staging, errors = apply_actions(plan, staging, todo, catalog)
        report.unresolved = {rid: errs for rid, errs in errors.items()}
        pending = set(errors)
        if not errors:
            break
        messages.append({"role": "assistant", "content": json.dumps(answer)})
        messages.append({"role": "user", "content":
                         "These instructions failed; the rest is placed. Current state of those rooms:\n\n" +
                         "\n\n".join(describe_room(plan, rid, staging, catalog) + "\n  failed: " + "; ".join(errs)
                                     for rid, errs in errors.items() if rid in {r.id for r in rooms}) +
                         "\n\nAnswer with corrected instructions for these rooms only (same JSON)."})

    # rooms the model left empty: take the rule placer's furniture for them, if it still fits
    empty = [r.id for r in rooms if not staging.in_room(r.id)]
    if empty:
        rules = stage_plan(plan, catalog, style, prefs)
        for item in rules.items:
            if item.room_id in empty and all(i.id != item.id for i in staging.items):
                trial = staging.model_copy(update={"items": staging.items + [item]})
                if validate_staging(plan, trial, catalog)["approved"]:
                    staging = trial
        report.rule_rooms = [rid for rid in empty if staging.in_room(rid)]
    used_llm = any(i.room_id not in report.rule_rooms for i in staging.items)
    report.planner = (f"llm+rules:{llm.name}" if report.rule_rooms else f"llm:{llm.name}") if used_llm else "rules"
    staging.staging_id = staging_id
    return staging, report


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("property_id")
    ap.add_argument("--style", default=None)
    ap.add_argument("--notes", default=None)
    ap.add_argument("--rules", action="store_true", help="skip the LLM")
    args = ap.parse_args()
    plan = load_plan(REPO_ROOT / "data" / "plans" / f"{args.property_id}.json")
    staging, report = furnish(plan, {"style": args.style} if args.style else None,
                              use_llm=not args.rules, notes=args.notes)
    out = save_staging(staging, REPO_ROOT / "data" / "stagings" / f"{args.property_id}.json")
    print(f"{len(staging.items)} items -> {out}  ({report.planner}, rounds={report.rounds})")
    for w in report.warnings:
        print("  !", w)
    for rid, errs in report.unresolved.items():
        print(f"  {rid}: {'; '.join(errs)}")
    if report.summary:
        print("  summary:", report.summary)
    sys.exit(0)
