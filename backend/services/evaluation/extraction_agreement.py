"""Controlled Field-by-Field Requirements Extraction Agreement Evaluation (Machine vs. Human GT).

Runs SmartSemsar's intake extractor (`backend.services.user_input_text.extract_requirements`,
CustomerRequirements schema, Groq LLM at temperature=0) identically on BOTH:
  - Latest machine-generated conversation (`conversations/<call_id>_conversation.json`,
    produced by local `faster-whisper large-v3-turbo` + `pyannote/speaker-diarization-community-1`
    + speaker role assignment)
  - Manually prepared human reference transcript (`ground_truth/transcripts/<call_id>_gt_transcript.json`)

Controlled experiment design:
  - Both transcript sources are formatted identically as `Role: text` dialogue turns
    (merging consecutive turns by the same role) so the ONLY variable is the transcript source.
  - Only recordings with manually prepared human transcripts (`smartsemsar_rec1`,
    `smartsemsar_rec8`, identified via `manifest.json` `annotation_source ==
    "manual_human_transcript_with_waveform_timestamps"`) are included by default.
  - Absent fields (`None` / `""` / `"unknown"` in BOTH extractions) are tracked separately as
    `both_absent` and NEVER counted in the Requirement Match Rate denominator.

Metric definitions (denominator H = total human-reference requirements present):
  - Requirement Match Rate = (exact_matches + semantic_matches) / H * 100
  - Exact Match Rate       = exact_matches / H * 100
  - Semantic Match Rate    = semantic_matches / H * 100
  - Missing Requirement Rate = missing_requirements / H * 100
  - Extra Requirement Rate (vs. human ref)   = extra_requirements / H * 100
  - Extra Requirement Rate (vs. machine ext) = extra_requirements / M * 100
  - Conflicting Value Rate = conflicting_values / H * 100

Usage:
    .venv/Scripts/python.exe backend/services/evaluation/extraction_agreement.py
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(REPO_ROOT / ".env", override=False)

from backend.services.evaluation.run_evaluation import (  # noqa: E402
    GT_DIR,
    REPORTS_DIR,
    _is_pending_or_template,
    _load_json_strict,
    _load_manifest,
    _resolve_gt_path,
    _validate_transcript_gt,
)
from backend.services.pipeline import (  # noqa: E402
    CONVERSATION_DIR,
    TRANSCRIPT_DIR,
    WHISPER_MODEL,
    configure_utf8_console,
    normalize_arabic_text,
    save_json,
)
from backend.services.user_input_text import (  # noqa: E402
    MODEL_NAME as OLLAMA_FALLBACK_MODEL,
    SYSTEM_PROMPT as EXTRACTION_SYSTEM_PROMPT,
    extract_requirements,
)

MANUAL_HUMAN_ANNOTATION_SOURCE = "manual_human_transcript_with_waveform_timestamps"
# Whisper drafts that a person listened to and corrected (rec2-rec7, rec9-rec10)
HUMAN_REVIEWED_ANNOTATION_SOURCE = "human_reviewed_whisper_multipass_waveform_aligned"

COMPARED_FIELDS = [
    "location",
    "budget_min",
    "budget_max",
    "area_sqm",
    "bedrooms",
    "property_type",
    "purpose",
    "deal_type",
    "notes",
]

ENUM_FIELDS = {"property_type", "purpose", "deal_type"}
NUMERIC_FIELDS = {"budget_min", "budget_max", "area_sqm", "bedrooms"}

_LOCATION_SYNONYMS: dict[str, str] = {
    "التجمع الخامس": "fifth_settlement",
    "تجمع خامس": "fifth_settlement",
    "fifth settlement": "fifth_settlement",
    "5th settlement": "fifth_settlement",
    "new cairo fifth settlement": "fifth_settlement",
    "القاهرة الجديدة التجمع الخامس": "fifth_settlement",
    "التجمع الخامس القاهرة الجديدة": "fifth_settlement",
    "مصر الجديدة": "heliopolis",
    "مصر الجديده": "heliopolis",
    "heliopolis": "heliopolis",
    "الشيخ زايد": "sheikh_zayed",
    "زايد": "sheikh_zayed",
    "sheikh zayed": "sheikh_zayed",
    "مدينة نصر": "nasr_city",
    "مدينه نصر": "nasr_city",
    "nasr city": "nasr_city",
    "6 اكتوبر": "sixth_of_october",
    "اكتوبر": "sixth_of_october",
    "october": "sixth_of_october",
    "السادس من اكتوبر": "sixth_of_october",
    "6th of october": "sixth_of_october",
    "sixth of october": "sixth_of_october",
    "العاصمة الادارية": "new_administrative_capital",
    "العاصمه الاداريه": "new_administrative_capital",
    "العاصمة الادارية الجديدة": "new_administrative_capital",
    "new administrative capital": "new_administrative_capital",
    "التجمع": "new_cairo_tagamoa",
    "new cairo": "new_cairo_tagamoa",
    "القاهرة الجديدة": "new_cairo_tagamoa",
}

_PROPERTY_TYPE_SYNONYMS: dict[str, str] = {
    "apartment": "apartment",
    "flat": "apartment",
    "شقه": "apartment",
    "شقة": "apartment",
    "villa": "villa",
    "فيلا": "villa",
    "فيله": "villa",
    "duplex": "duplex",
    "دوبلكس": "duplex",
    "studio": "studio",
    "استوديو": "studio",
    "ستوديو": "studio",
    "townhouse": "townhouse",
    "تاون هاوس": "townhouse",
}

_DEAL_TYPE_SYNONYMS: dict[str, str] = {
    "rent": "rent",
    "rental": "rent",
    "ايجار": "rent",
    "تاجير": "rent",
    "buy": "buy",
    "purchase": "buy",
    "ownership": "buy",
    "تمليك": "buy",
    "شراء": "buy",
}

_PURPOSE_SYNONYMS: dict[str, str] = {
    "living": "living",
    "residential": "living",
    "سكن": "living",
    "سكني": "living",
    "للسكن": "living",
    "investment": "investment",
    "استثمار": "investment",
    "استثماري": "investment",
    "للاستثمار": "investment",
}

_NOTE_CANONICAL_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"^(?:مش\s+صغير[هة]?|ليست\s+صغير[هة]?|ليس\s+صغير[هة]?|not\s+small|not\s+too\s+small|spacious|واسع[هة]?|كبير[هة]?)$", "not_small"),
)

_ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


def _is_absent(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        s = value.strip().lower()
        return s == "" or s in ("unknown", "none", "null", "n/a")
    return False


def _clean_arabic_canonical(text: str) -> str:
    s = str(text).translate(_ARABIC_DIGITS)
    s = normalize_arabic_text(s).lower()
    s = s.replace("ة", "ه")
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def parse_numeric_requirement(value: Any, field: str) -> Optional[float]:
    """Parse equivalent numeric formats (e.g. '12 million', '12,000,000', '180 sqm', '15 الف')."""
    if _is_absent(value):
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if not isinstance(value, str):
        return None

    s = _clean_arabic_canonical(value).replace(",", "")
    if field == "area_sqm":
        s = re.sub(r"\b(?:sqm|sq\s*m|m2|متر|متر\s*مربع|م2)\b", "", s).strip()
    m = re.search(r"(\d+(?:\.\d+)?)", s)
    if not m:
        return None
    num = float(m.group(1))
    if field in ("budget_min", "budget_max"):
        if re.search(r"(?:مليون|million|\bm\b)", s):
            if num < 1000:
                num *= 1_000_000.0
        elif re.search(r"(?:الف|الاف|تلاف|thousand|\bk\b)", s):
            if num < 1000:
                num *= 1_000.0
    return num


def canonicalize_field_value(field: str, value: Any) -> Any:
    """Return a semantically normalized canonical representation for comparison."""
    if _is_absent(value):
        return None

    if field in NUMERIC_FIELDS:
        parsed = parse_numeric_requirement(value, field)
        if parsed is not None:
            return int(round(parsed)) if field == "bedrooms" else round(parsed, 4)
        return _clean_arabic_canonical(str(value))

    clean = _clean_arabic_canonical(str(value))

    if field == "property_type":
        return _PROPERTY_TYPE_SYNONYMS.get(clean, clean)
    if field == "deal_type":
        return _DEAL_TYPE_SYNONYMS.get(clean, clean)
    if field == "purpose":
        return _PURPOSE_SYNONYMS.get(clean, clean)
    if field == "location":
        # Support multi-location alternatives/lists (e.g., "الشيخ زايد أو أكتوبر" == "أكتوبر, الشيخ زايد")
        parts = [
            p.strip()
            for p in re.split(r"\s*(?:,|،|/|\b(?:او|أو|or|and)\b|\bو(?=\s))\s*", clean)
            if p.strip()
        ]
        canon_parts: list[str] = []
        for part in (parts or [clean]):
            stripped = re.sub(r"^(?:في|منطقة|منطقه|in|at)\s+", "", part).strip()
            mapped = None
            for candidate in (part, stripped):
                if candidate in _LOCATION_SYNONYMS:
                    mapped = _LOCATION_SYNONYMS[candidate]
                    break
                if candidate.replace("ه", "ة") in _LOCATION_SYNONYMS:
                    mapped = _LOCATION_SYNONYMS[candidate.replace("ه", "ة")]
                    break
            canon_parts.append(mapped or stripped)
        unique_sorted = sorted(set(canon_parts))
        return unique_sorted[0] if len(unique_sorted) == 1 else " | ".join(unique_sorted)

    if field == "notes":
        for pat, canon in _NOTE_CANONICAL_PATTERNS:
            if re.match(pat, clean):
                return canon
        return clean

    return clean


def _exact_values_equal(field: str, machine_val: Any, human_val: Any) -> bool:
    """True only when present values match identically (ignoring trivial int/float or whitespace)."""
    if _is_absent(machine_val) or _is_absent(human_val):
        return False
    if field in NUMERIC_FIELDS:
        if isinstance(machine_val, (int, float)) and isinstance(human_val, (int, float)):
            return abs(float(machine_val) - float(human_val)) < 1e-6
    if isinstance(machine_val, str) and isinstance(human_val, str):
        if field in ENUM_FIELDS:
            return machine_val.strip().lower() == human_val.strip().lower()
        return re.sub(r"\s+", " ", machine_val.strip()) == re.sub(r"\s+", " ", human_val.strip())
    return machine_val == human_val


def _merge_same_role_turns(turns: list[tuple[str, str]]) -> str:
    """Format (role, text) turns identically, merging adjacent turns from the same role."""
    merged: list[list[str]] = []
    for role, text in turns:
        t = re.sub(r"\s+", " ", str(text or "")).strip()
        if not t:
            continue
        r = str(role or "Speaker").strip().capitalize()
        if merged and merged[-1][0] == r:
            merged[-1][1] = f"{merged[-1][1]} {t}"
        else:
            merged.append([r, t])
    return "\n".join(f"{r}: {t}" for r, t in merged)


def format_human_transcript(gt_payload: dict[str, Any]) -> str:
    """Format manually transcribed human GT using canonical speaker roles (`Client:` / `Broker:`)."""
    segments = gt_payload.get("segments")
    if isinstance(segments, list) and segments:
        turns = [
            (
                seg.get("role") or seg.get("speaker_id") or "Speaker",
                seg.get("text", ""),
            )
            for seg in segments
        ]
        formatted = _merge_same_role_turns(turns)
        if formatted:
            return formatted
    return str(gt_payload.get("reference_transcript", "")).strip()


def format_machine_conversation(
    conv_payload: Optional[dict[str, Any]],
    transcript_payload: Optional[dict[str, Any]] = None,
) -> tuple[str, str]:
    """Format latest machine conversation identically to human GT using predicted speaker roles."""
    if conv_payload and isinstance(conv_payload.get("conversation"), list) and conv_payload["conversation"]:
        turns = []
        for turn in conv_payload["conversation"]:
            role = turn.get("role")
            if not role or str(role).strip().lower() == "unassigned":
                role = turn.get("speaker") or "Speaker"
            text = turn.get("raw_text") or turn.get("text") or ""
            turns.append((role, text))
        formatted = _merge_same_role_turns(turns)
        if formatted:
            return formatted, "conversation_with_predicted_roles"

    if transcript_payload:
        segments = transcript_payload.get("segments")
        if isinstance(segments, list) and segments:
            lines = [str(seg.get("text", "")).strip() for seg in segments if str(seg.get("text", "")).strip()]
            if lines:
                return "\n".join(lines), "transcript_without_roles"
        return str(transcript_payload.get("full_transcript", "")).strip(), "full_transcript_without_roles"

    return "", "empty"


def _compute_rate(num: int, denom: int) -> Optional[float]:
    return round((num / denom) * 100.0, 2) if denom > 0 else None


def summarize_counts(counts: dict[str, int]) -> dict[str, Any]:
    """Compute all required rates from raw match-category counts without inflating absent fields."""
    exact = counts["exact_match"]
    semantic = counts["semantic_match"]
    missing = counts["missing_requirement"]
    extra = counts["extra_requirement"]
    conflicting = counts["conflicting_value"]
    both_absent = counts["both_absent"]

    human_total = exact + semantic + missing + conflicting
    machine_total = exact + semantic + extra + conflicting
    matched = exact + semantic

    return {
        "total_human_requirements": human_total,
        "total_machine_requirements": machine_total,
        "matched_requirements": matched,
        "exact_matches": exact,
        "semantic_matches": semantic,
        "missing_requirements": missing,
        "extra_requirements": extra,
        "conflicting_values": conflicting,
        "both_absent_fields_excluded": both_absent,
        "requirement_match_rate_pct": _compute_rate(matched, human_total),
        "exact_match_rate_pct": _compute_rate(exact, human_total),
        "semantic_match_rate_pct": _compute_rate(semantic, human_total),
        "missing_requirement_rate_pct": _compute_rate(missing, human_total),
        "extra_requirement_rate_vs_human_pct": _compute_rate(extra, human_total),
        "extra_requirement_rate_vs_machine_pct": _compute_rate(extra, machine_total),
        "conflicting_value_rate_pct": _compute_rate(conflicting, human_total),
    }


def compare_extractions(machine_dict: dict[str, Any], human_dict: dict[str, Any]) -> dict[str, Any]:
    """Compare machine vs. human CustomerRequirements field by field with strict taxonomy."""
    counts = {
        "exact_match": 0,
        "semantic_match": 0,
        "missing_requirement": 0,
        "extra_requirement": 0,
        "conflicting_value": 0,
        "both_absent": 0,
    }
    per_field: dict[str, Any] = {}

    for field in COMPARED_FIELDS:
        m_raw = machine_dict.get(field)
        h_raw = human_dict.get(field)
        m_absent = _is_absent(m_raw)
        h_absent = _is_absent(h_raw)
        m_canon = canonicalize_field_value(field, m_raw)
        h_canon = canonicalize_field_value(field, h_raw)

        if h_absent and m_absent:
            category = "both_absent"
        elif not h_absent and m_absent:
            category = "missing_requirement"
        elif h_absent and not m_absent:
            category = "extra_requirement"
        elif _exact_values_equal(field, m_raw, h_raw):
            category = "exact_match"
        elif m_canon == h_canon:
            category = "semantic_match"
        else:
            category = "conflicting_value"

        counts[category] += 1
        per_field[field] = {
            "human_value": h_raw,
            "machine_value": m_raw,
            "human_canonical": h_canon,
            "machine_canonical": m_canon,
            "category": category,
            "matched": category in ("exact_match", "semantic_match"),
        }

    return {
        "metrics": summarize_counts(counts),
        "per_field": per_field,
    }


def _active_extraction_config() -> dict[str, Any]:
    use_groq = bool(os.getenv("GROQ_API_KEY"))
    return {
        "extractor_function": "backend.services.user_input_text.extract_requirements",
        "output_schema": "backend.schema.state.CustomerRequirements",
        "provider": "groq" if use_groq else "ollama",
        "model": (
            os.getenv("SMARTSEMSAR_GROQ_MODEL", "openai/gpt-oss-120b")
            if use_groq
            else os.getenv("SMARTSEMSAR_OLLAMA_MODEL", OLLAMA_FALLBACK_MODEL)
        ),
        "temperature": 0,
        "response_format": "json_object",
        "prompt_source": "backend.services.user_input_text.SYSTEM_PROMPT",
        "prompt_char_length": len(EXTRACTION_SYSTEM_PROMPT),
        "input_formatting": "Identical 'Role: text' dialogue lines with consecutive same-role turns merged",
        "compared_fields": COMPARED_FIELDS,
    }


def run_extraction_agreement(
    call_ids: Optional[list[str]] = None,
    *,
    manual_human_only: bool = True,
    pause_s: float = 0.0,
    gt_dir: Path = GT_DIR,
    transcript_dir: Path = TRANSCRIPT_DIR,
    conversation_dir: Path = CONVERSATION_DIR,
    reports_dir: Path = REPORTS_DIR,
) -> dict[str, Any]:
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    manifest_map, _ = _load_manifest(gt_dir)

    excluded_non_manual: list[dict[str, str]] = []
    if call_ids is not None:
        targets = list(call_ids)
    elif manifest_map is not None:
        targets = []
        for cid, entry in manifest_map.items():
            src = entry.get("annotation_source", "")
            if manual_human_only and src != MANUAL_HUMAN_ANNOTATION_SOURCE:
                excluded_non_manual.append({
                    "call_id": cid,
                    "annotation_source": src,
                    "reason": "Excluded from manual human-transcript agreement evaluation (not manually transcribed from scratch).",
                })
                continue
            targets.append(cid)
    else:
        targets = ["smartsemsar_rec1", "smartsemsar_rec8"]

    overall_counts = {
        "exact_match": 0,
        "semantic_match": 0,
        "missing_requirement": 0,
        "extra_requirement": 0,
        "conflicting_value": 0,
        "both_absent": 0,
    }
    subgroup_counts: dict[str, dict[str, int]] = {
        "manual_human_transcripts": {k: 0 for k in overall_counts},
        "user_provided_multipass_reference": {k: 0 for k in overall_counts},
        "human_reviewed_multipass_reference": {k: 0 for k in overall_counts},
    }
    subgroup_call_ids: dict[str, list[str]] = {
        "manual_human_transcripts": [],
        "user_provided_multipass_reference": [],
        "human_reviewed_multipass_reference": [],
    }
    field_counts = {
        f: {k: 0 for k in overall_counts}
        for f in COMPARED_FIELDS
    }
    per_call_results: list[dict[str, Any]] = []

    for cid in targets:
        manifest_entry = manifest_map.get(cid) if manifest_map else None
        gt_path = _resolve_gt_path(
            gt_dir, "transcripts", f"{cid}_gt_transcript.json", manifest_entry, "ground_truth_transcript"
        )
        transcript_path = Path(transcript_dir) / f"{cid}_transcript.json"
        conv_path = Path(conversation_dir) / f"{cid}_conversation.json"

        try:
            raw_gt = _load_json_strict(gt_path)
            if _is_pending_or_template(raw_gt):
                per_call_results.append({"call_id": cid, "status": "PENDING_GROUND_TRUTH"})
                continue
            gt_payload = _validate_transcript_gt(raw_gt, cid)
        except Exception as exc:
            per_call_results.append({
                "call_id": cid,
                "status": "ERROR",
                "error": f"GT load/validation failed: {type(exc).__name__}: {exc}",
            })
            continue

        if not transcript_path.exists() or not conv_path.exists():
            per_call_results.append({
                "call_id": cid,
                "status": "MISSING_PIPELINE_OUTPUT",
                "missing": [str(p) for p in (transcript_path, conv_path) if not p.exists()],
            })
            continue

        try:
            tr_payload = _load_json_strict(transcript_path)
            conv_payload = _load_json_strict(conv_path)

            human_input = format_human_transcript(gt_payload)
            machine_input, machine_format_mode = format_machine_conversation(conv_payload, tr_payload)

            time.sleep(pause_s)   # Groq free tier: 8000 tokens/min, so keep every call on Groq (no Ollama fallback)
            print(f"[{cid}] Running extract_requirements() on HUMAN transcript...", flush=True)
            human_req = extract_requirements(call_transcript=human_input, raw_text=None).model_dump()
            time.sleep(pause_s)
            print(f"[{cid}] Running extract_requirements() on MACHINE transcript ({tr_payload.get('whisper_model')})...", flush=True)
            machine_req = extract_requirements(call_transcript=machine_input, raw_text=None).model_dump()

            comparison = compare_extractions(machine_req, human_req)
            annot_src = (
                (manifest_entry.get("annotation_source") if manifest_entry else None)
                or gt_payload.get("annotator")
                or ""
            )
            subgroup_key = (
                "manual_human_transcripts"
                if annot_src == MANUAL_HUMAN_ANNOTATION_SOURCE
                else "human_reviewed_multipass_reference"
                if annot_src == HUMAN_REVIEWED_ANNOTATION_SOURCE
                else "user_provided_multipass_reference"
            )
            subgroup_call_ids[subgroup_key].append(cid)
            for f, details in comparison["per_field"].items():
                cat = details["category"]
                overall_counts[cat] += 1
                subgroup_counts[subgroup_key][cat] += 1
                field_counts[f][cat] += 1

            per_call_results.append({
                "call_id": cid,
                "status": "EVALUATED",
                "reference_provenance_group": subgroup_key,
                "human_annotation_source": gt_payload.get("annotator"),
                "human_transcript_path": str(gt_path),
                "machine_transcript_path": str(transcript_path),
                "machine_conversation_path": str(conv_path),
                "machine_asr_model": tr_payload.get("whisper_model", WHISPER_MODEL),
                "machine_asr_backend": tr_payload.get("asr_backend", "local_faster_whisper"),
                "machine_speaker_role_mapping": conv_payload.get("speaker_role_mapping"),
                "machine_input_format": machine_format_mode,
                "formatted_human_input": human_input,
                "formatted_machine_input": machine_input,
                "human_extraction": human_req,
                "machine_extraction": machine_req,
                "comparison": comparison,
            })
        except Exception as exc:
            per_call_results.append({
                "call_id": cid,
                "status": "ERROR",
                "error": f"{type(exc).__name__}: {exc}",
            })

    eval_count = sum(1 for r in per_call_results if r["status"] == "EVALUATED")
    error_count = sum(1 for r in per_call_results if r["status"] == "ERROR")
    status = "EVALUATED" if (eval_count == len(targets) and error_count == 0 and len(targets) > 0) else (
        "PARTIALLY_EVALUATED" if eval_count > 0 else "ERROR"
    )

    legacy_archive = reports_dir / "archive" / "extraction_agreement_report_legacy_groq_20261121.json"

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_scope": (
            "all_10_manifest_calls_with_provenance_breakdown"
            if not manual_human_only and call_ids is None
            else ("manual_human_transcripts_only" if manual_human_only and call_ids is None else "custom_call_selection")
        ),
        "status": status,
        "extraction_configuration": _active_extraction_config(),
        "metric_definitions": {
            "denominator_H": "Total human-reference requirements present (non-null, non-empty, non-'unknown') across evaluated fields; fields absent in both extractions ('both_absent') are excluded.",
            "denominator_M": "Total machine-extracted requirements present (non-null, non-empty, non-'unknown').",
            "requirement_match_rate_pct": "(exact_matches + semantic_matches) / H * 100",
            "exact_match_rate_pct": "exact_matches / H * 100",
            "semantic_match_rate_pct": "semantic_matches / H * 100",
            "missing_requirement_rate_pct": "missing_requirements / H * 100",
            "extra_requirement_rate_vs_human_pct": "extra_requirements / H * 100",
            "extra_requirement_rate_vs_machine_pct": "extra_requirements / M * 100",
            "conflicting_value_rate_pct": "conflicting_values / H * 100",
        },
        "previous_report_preserved_at": str(legacy_archive) if legacy_archive.exists() else None,
        "total_target_calls": len(targets),
        "evaluated_calls": eval_count,
        "error_calls": error_count,
        "excluded_non_manual_calls": excluded_non_manual,
        "overall_summary": summarize_counts(overall_counts),
        "provenance_subgroup_summary": {
            k: {
                "call_ids": subgroup_call_ids[k],
                "num_calls": len(subgroup_call_ids[k]),
                **summarize_counts(subgroup_counts[k]),
            }
            for k in ("manual_human_transcripts", "human_reviewed_multipass_reference", "user_provided_multipass_reference")
            if subgroup_call_ids[k]
        },
        "per_field_summary": {f: summarize_counts(field_counts[f]) for f in COMPARED_FIELDS},
        "calls": per_call_results,
    }
    save_json(reports_dir / "extraction_agreement_report.json", report)
    return report


def main(argv: Optional[list[str]] = None) -> int:
    configure_utf8_console()
    parser = argparse.ArgumentParser(
        description="Controlled machine-vs-human requirements extraction agreement evaluation."
    )
    parser.add_argument(
        "--calls",
        nargs="+",
        default=None,
        help="Optional explicit list of call IDs (default: manual human transcripts only, i.e. rec1 & rec8).",
    )
    parser.add_argument(
        "--all-manifest-calls",
        action="store_true",
        help="Include all manifest calls instead of restricting to manually prepared human transcripts.",
    )
    parser.add_argument(
        "--pause",
        type=float,
        default=0.0,
        help="Seconds to wait before each LLM call, to stay under the Groq tokens-per-minute limit (e.g. 25).",
    )
    args = parser.parse_args(argv)

    report = run_extraction_agreement(
        call_ids=args.calls,
        manual_human_only=not args.all_manifest_calls,
        pause_s=args.pause,
    )
    print(json.dumps({
        "status": report["status"],
        "evaluation_scope": report["evaluation_scope"],
        "evaluated_calls": report["evaluated_calls"],
        "overall_summary": report["overall_summary"],
    }, ensure_ascii=False, indent=2))
    return 0 if report["error_calls"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
