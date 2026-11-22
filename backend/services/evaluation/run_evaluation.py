"""Speech Evaluation runner for SmartSemsar.

Safely evaluates WER, CER, DER, Broker/Client role accuracy, and objective Audio Quality
across recorded phone calls with strict Ground Truth validation and per-call error isolation.

Strict integrity rules:
- Never fabricates ground-truth annotations or metric values.
- Rejects template files (`_is_template: true`), pending files (`status: "pending"`),
  and placeholder role values (e.g., `broker_or_client`, `client_or_broker`).
- Never reports WER, CER, DER, or role accuracy until verified human ground truth exists.
- Isolates malformed JSON and schema/validation errors per call so remaining calls continue.

Exit codes:
- 0: Completed with zero evaluation errors (calls may be EVALUATED, PARTIALLY_EVALUATED,
     PENDING_GROUND_TRUTH, or PENDING_ROLE_PREDICTIONS).
- 1: One or more calls (or manifest.json) encountered an ERROR (malformed JSON, schema
     violation, speaker ID mismatch, invalid timestamps, or invalid arguments).
- 2: `--strict` was requested and zero calls were evaluated across transcript/diarization/role metrics.

Usage:
    python backend/services/evaluation/run_evaluation.py
    python backend/services/evaluation/run_evaluation.py --calls smartsemsar_rec1 smartsemsar_rec2
    python backend/services/evaluation/run_evaluation.py --collar 0.25
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.services.evaluation.metrics import (  # noqa: E402
    FORBIDDEN_ROLE_PLACEHOLDERS,
    VALID_ROLES,
    _validate_turns,
    analyze_audio_quality,
    compute_cer,
    compute_corpus_cer,
    compute_corpus_der,
    compute_corpus_role_accuracy,
    compute_corpus_wer,
    compute_der,
    compute_role_accuracy,
    compute_wer,
)
from backend.services.pipeline import (  # noqa: E402
    CONVERSATION_DIR,
    DIARIZATION_DIR,
    EVALUATION_DIR,
    PROCESSED_AUDIO_DIR,
    QC_DIR,
    RAW_AUDIO_DIR,
    TRANSCRIPT_DIR,
    VAD_DIR,
    configure_utf8_console,
    discover_calls,
    save_json,
)

GT_DIR = EVALUATION_DIR / "ground_truth"
GT_TRANSCRIPTS_DIR = GT_DIR / "transcripts"
GT_DIARIZATION_DIR = GT_DIR / "diarization"
GT_ROLES_DIR = GT_DIR / "roles"
REPORTS_DIR = EVALUATION_DIR / "reports"

EVAL_EXIT_OK = 0
EVAL_EXIT_ERROR = 1
EVAL_EXIT_NO_EVALUATED = 2


class GroundTruthValidationError(ValueError):
    """Raised when a non-pending Ground Truth file violates required schema rules."""


def _load_json_strict(path: Path) -> Optional[dict[str, Any]]:
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise GroundTruthValidationError(f"Malformed JSON in '{path}': {exc}") from exc
    if not isinstance(data, dict):
        raise GroundTruthValidationError(
            f"Expected JSON object in '{path}', got {type(data).__name__}."
        )
    return data


def _is_pending_or_template(payload: Optional[dict[str, Any]]) -> bool:
    if payload is None:
        return True
    if payload.get("_is_template") is True:
        return True
    status = payload.get("status")
    if isinstance(status, str) and status.strip().lower() == "pending":
        return True
    return False


def _validate_role_value(role_val: Any, context: str) -> str:
    if not isinstance(role_val, str) or not role_val.strip():
        raise GroundTruthValidationError(
            f"Invalid role in {context}: expected 'broker' or 'client', got {role_val!r}."
        )
    norm = role_val.strip().lower()
    if norm in FORBIDDEN_ROLE_PLACEHOLDERS or norm not in VALID_ROLES:
        raise GroundTruthValidationError(
            f"Invalid role '{role_val}' in {context}: only {sorted(VALID_ROLES)} are allowed "
            f"(placeholder values are rejected)."
        )
    return norm


def _validate_transcript_gt(payload: dict[str, Any], call_id: str) -> dict[str, Any]:
    if payload.get("call_id") and payload["call_id"] != call_id:
        raise GroundTruthValidationError(
            f"Transcript GT call_id mismatch: expected '{call_id}', got '{payload['call_id']}'."
        )
    ref_text = payload.get("reference_transcript")
    if not isinstance(ref_text, str) or not ref_text.strip():
        raise GroundTruthValidationError(
            f"Transcript GT for '{call_id}' has empty or invalid 'reference_transcript'."
        )
    if "segments" in payload and payload["segments"] is not None:
        segs = payload["segments"]
        if not isinstance(segs, list):
            raise GroundTruthValidationError(
                f"Transcript GT 'segments' for '{call_id}' must be a list."
            )
        for idx, seg in enumerate(segs):
            if not isinstance(seg, dict):
                raise GroundTruthValidationError(
                    f"Transcript GT segment[{idx}] for '{call_id}' must be an object."
                )
            if "start" in seg or "end" in seg:
                _validate_turns([seg], f"gt_transcript[{call_id}].segments[{idx}]")
            if seg.get("role") is not None:
                _validate_role_value(seg["role"], f"gt_transcript[{call_id}].segments[{idx}].role")
    return payload


def _validate_diarization_gt(payload: dict[str, Any], call_id: str) -> dict[str, Any]:
    if payload.get("call_id") and payload["call_id"] != call_id:
        raise GroundTruthValidationError(
            f"Diarization GT call_id mismatch: expected '{call_id}', got '{payload['call_id']}'."
        )
    turns = payload.get("turns")
    if not isinstance(turns, list) or not turns:
        raise GroundTruthValidationError(
            f"Diarization GT for '{call_id}' must contain a non-empty 'turns' list."
        )
    _validate_turns(turns, f"gt_diarization[{call_id}]")
    for idx, t in enumerate(turns):
        if t.get("role") is not None:
            _validate_role_value(t["role"], f"gt_diarization[{call_id}].turns[{idx}].role")
    return payload


def _validate_roles_gt(payload: dict[str, Any], call_id: str) -> dict[str, Any]:
    if payload.get("call_id") and payload["call_id"] != call_id:
        raise GroundTruthValidationError(
            f"Roles GT call_id mismatch: expected '{call_id}', got '{payload['call_id']}'."
        )
    roles = payload.get("speaker_roles")
    if not isinstance(roles, dict) or not roles:
        raise GroundTruthValidationError(
            f"Roles GT for '{call_id}' must contain a non-empty 'speaker_roles' object."
        )
    normalized_roles = {}
    for spk, role in roles.items():
        if not isinstance(spk, str) or not spk.strip():
            raise GroundTruthValidationError(
                f"Roles GT for '{call_id}' contains an empty speaker ID key."
            )
        normalized_roles[spk.strip()] = _validate_role_value(
            role, f"gt_roles[{call_id}].speaker_roles['{spk}']"
        )

    pipe_map = payload.get("pipeline_speaker_to_reference_speaker")
    if pipe_map is not None:
        if not isinstance(pipe_map, dict):
            raise GroundTruthValidationError(
                f"Roles GT 'pipeline_speaker_to_reference_speaker' for '{call_id}' must be a dict."
            )
        for pipe_spk, ref_spk in pipe_map.items():
            if not isinstance(pipe_spk, str) or not pipe_spk.strip():
                raise GroundTruthValidationError(
                    f"Roles GT for '{call_id}' has invalid pipeline speaker key {pipe_spk!r}."
                )
            if ref_spk not in normalized_roles:
                raise GroundTruthValidationError(
                    f"Roles GT for '{call_id}' maps '{pipe_spk}' -> '{ref_spk}', "
                    f"which is not present in speaker_roles {sorted(normalized_roles.keys())}."
                )

    return {**payload, "speaker_roles": normalized_roles}


def _validate_cross_file_speaker_consistency(
    call_id: str,
    gt_tr: Optional[dict[str, Any]],
    gt_diar: Optional[dict[str, Any]],
    gt_role: Optional[dict[str, Any]],
) -> None:
    """Verify speaker IDs and roles are consistent across completed Ground Truth files."""
    if gt_role is None:
        return
    role_map: dict[str, str] = gt_role["speaker_roles"]
    pipe_map: dict[str, str] = gt_role.get("pipeline_speaker_to_reference_speaker") or {}

    def _check_items(items: list[dict[str, Any]], source_name: str):
        for idx, item in enumerate(items):
            spk = item.get("speaker_id") if item.get("speaker_id") is not None else item.get("speaker")
            if spk is None:
                continue
            spk_str = str(spk).strip()
            # Allow canonical reference speaker IDs directly in speaker_roles
            if spk_str not in role_map and spk_str not in pipe_map.values():
                raise GroundTruthValidationError(
                    f"Speaker ID mismatch for '{call_id}': {source_name}[{idx}] uses speaker "
                    f"'{spk_str}', which is missing from gt_roles speaker_roles "
                    f"{sorted(role_map.keys())}."
                )
            if item.get("role") is not None and spk_str in role_map:
                item_role = _validate_role_value(item["role"], f"{source_name}[{idx}].role")
                if item_role != role_map[spk_str]:
                    raise GroundTruthValidationError(
                        f"Role mismatch for '{call_id}': {source_name}[{idx}] assigns "
                        f"'{spk_str}' -> '{item_role}', but gt_roles assigns '{role_map[spk_str]}'."
                    )

    if gt_diar is not None and isinstance(gt_diar.get("turns"), list):
        _check_items(gt_diar["turns"], "gt_diarization.turns")
    if gt_tr is not None and isinstance(gt_tr.get("segments"), list):
        _check_items(gt_tr["segments"], "gt_transcript.segments")


def _load_manifest(gt_dir: Path) -> tuple[Optional[dict[str, dict[str, Any]]], Optional[str]]:
    """Load and validate ground_truth/manifest.json if present."""
    manifest_path = Path(gt_dir) / "manifest.json"
    if not manifest_path.exists():
        return None, None
    try:
        data = _load_json_strict(manifest_path)
        calls = data.get("calls")
        if not isinstance(calls, list):
            raise GroundTruthValidationError("manifest.json must contain a 'calls' list.")
        by_id: dict[str, dict[str, Any]] = {}
        for idx, entry in enumerate(calls):
            if not isinstance(entry, dict) or not entry.get("call_id"):
                raise GroundTruthValidationError(
                    f"manifest.json calls[{idx}] must be an object with 'call_id'."
                )
            cid = str(entry["call_id"]).strip()
            # Validate naming convention consistency for manifest GT paths
            expected_names = {
                "ground_truth_transcript": f"{cid}_gt_transcript.json",
                "ground_truth_diarization": f"{cid}_gt_diarization.json",
                "ground_truth_roles": f"{cid}_gt_roles.json",
            }
            for field, exp_name in expected_names.items():
                if field in entry and entry[field]:
                    actual_name = Path(entry[field]).name
                    if actual_name != exp_name:
                        raise GroundTruthValidationError(
                            f"manifest.json path naming mismatch for '{cid}.{field}': "
                            f"expected filename '{exp_name}', got '{actual_name}'."
                        )
            by_id[cid] = entry
        return by_id, None
    except Exception as exc:
        return None, str(exc)


def _resolve_gt_path(
    gt_dir: Path,
    subdir: str,
    default_filename: str,
    manifest_entry: Optional[dict[str, Any]],
    manifest_field: str,
) -> Path:
    """Resolve GT path consistently using manifest.json when aligned with gt_dir."""
    local_candidate = Path(gt_dir) / subdir / default_filename
    if manifest_entry and manifest_entry.get(manifest_field):
        raw_p = Path(manifest_entry[manifest_field])
        # Prefer gt_dir-relative path so custom/test gt_dir overrides work cleanly
        rel_in_gt_dir = Path(gt_dir) / subdir / raw_p.name
        if rel_in_gt_dir.exists() or Path(gt_dir).resolve() != GT_DIR.resolve():
            return rel_in_gt_dir
        repo_p = raw_p if raw_p.is_absolute() else (REPO_ROOT / raw_p)
        return repo_p
    return local_candidate


def _aggregate_section_status(
    total_calls: int,
    evaluated_calls: int,
    error_calls: int,
    *,
    pending_role_prediction_calls: int = 0,
    empty_status: str = "PENDING_GROUND_TRUTH",
) -> str:
    """Determine report-level status across calls using the explicit evaluation state taxonomy."""
    if total_calls == 0:
        return empty_status
    if evaluated_calls == total_calls and error_calls == 0:
        return "EVALUATED"
    if evaluated_calls > 0:
        return "PARTIALLY_EVALUATED"
    if error_calls > 0:
        return "ERROR"
    if pending_role_prediction_calls > 0:
        return "PENDING_ROLE_PREDICTIONS"
    return empty_status


def evaluate_calls(
    call_ids: Optional[list[str]] = None,
    *,
    collar: float = 0.25,
    processed_audio_dir: Path = PROCESSED_AUDIO_DIR,
    vad_dir: Path = VAD_DIR,
    transcript_dir: Path = TRANSCRIPT_DIR,
    diarization_dir: Path = DIARIZATION_DIR,
    conversation_dir: Path = CONVERSATION_DIR,
    qc_dir: Path = QC_DIR,
    gt_dir: Path = GT_DIR,
    reports_dir: Path = REPORTS_DIR,
) -> dict[str, Any]:
    """Run all ready evaluations with manifest integration, strict validation, and per-call error isolation."""
    if not isinstance(collar, (int, float)) or isinstance(collar, bool) or collar < 0:
        raise ValueError(f"Invalid collar value: {collar!r}. Collar must be >= 0.")

    gt_dir = Path(gt_dir)
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    manifest_map, manifest_error = _load_manifest(gt_dir)
    if call_ids is not None:
        targets = list(call_ids)
    elif manifest_map is not None:
        targets = list(manifest_map.keys())
    else:
        targets = discover_calls(raw_audio_dir=RAW_AUDIO_DIR, ext="mp4")

    wer_cer_results: list[dict[str, Any]] = []
    der_results: list[dict[str, Any]] = []
    role_results: list[dict[str, Any]] = []
    audio_quality_results: list[dict[str, Any]] = []

    for cid in targets:
        manifest_entry = manifest_map.get(cid) if manifest_map else None

        # Pre-load & validate GT files for cross-file speaker consistency (each isolated)
        gt_tr_path = _resolve_gt_path(
            gt_dir, "transcripts", f"{cid}_gt_transcript.json", manifest_entry, "ground_truth_transcript"
        )
        gt_diar_path = _resolve_gt_path(
            gt_dir, "diarization", f"{cid}_gt_diarization.json", manifest_entry, "ground_truth_diarization"
        )
        gt_role_path = _resolve_gt_path(
            gt_dir, "roles", f"{cid}_gt_roles.json", manifest_entry, "ground_truth_roles"
        )

        validated_gt_tr: Optional[dict[str, Any]] = None
        validated_gt_diar: Optional[dict[str, Any]] = None
        validated_gt_role: Optional[dict[str, Any]] = None
        der_optimal_mapping: Optional[dict[str, str]] = None

        # 1. WER / CER
        try:
            raw_gt_tr = _load_json_strict(gt_tr_path)
            if _is_pending_or_template(raw_gt_tr):
                wer_cer_results.append({
                    "call_id": cid,
                    "status": "PENDING_GROUND_TRUTH",
                    "wer": None,
                    "cer": None,
                })
            else:
                validated_gt_tr = _validate_transcript_gt(raw_gt_tr, cid)
                hyp_tr = _load_json_strict(Path(transcript_dir) / f"{cid}_transcript.json")
                if hyp_tr is None:
                    wer_cer_results.append({
                        "call_id": cid,
                        "status": "MISSING_PIPELINE_OUTPUT",
                        "wer": None,
                        "cer": None,
                    })
                else:
                    ref_text = validated_gt_tr["reference_transcript"]
                    hyp_text = hyp_tr.get("full_transcript", "")
                    wer_m = compute_wer(ref_text, hyp_text, normalize=True)
                    cer_m = compute_cer(ref_text, hyp_text, normalize=True)
                    wer_cer_results.append({
                        "call_id": cid,
                        "status": "EVALUATED",
                        "wer": wer_m["wer"],
                        "cer": cer_m["cer"],
                        "wer_details": wer_m,
                        "cer_details": cer_m,
                    })
        except Exception as exc:
            wer_cer_results.append({
                "call_id": cid,
                "status": "ERROR",
                "wer": None,
                "cer": None,
                "error_type": type(exc).__name__,
                "error": str(exc),
            })

        # 2. DER
        try:
            raw_gt_diar = _load_json_strict(gt_diar_path)
            if _is_pending_or_template(raw_gt_diar):
                der_results.append({
                    "call_id": cid,
                    "status": "PENDING_GROUND_TRUTH",
                    "der": None,
                })
            else:
                validated_gt_diar = _validate_diarization_gt(raw_gt_diar, cid)
                hyp_diar = _load_json_strict(Path(diarization_dir) / f"{cid}_diarization.json")
                if hyp_diar is None:
                    der_results.append({
                        "call_id": cid,
                        "status": "MISSING_PIPELINE_OUTPUT",
                        "der": None,
                    })
                else:
                    der_m = compute_der(validated_gt_diar["turns"], hyp_diar.get("turns", []), collar=collar)
                    der_optimal_mapping = der_m.get("optimal_speaker_mapping")
                    der_results.append({
                        "call_id": cid,
                        "status": "EVALUATED",
                        **der_m,
                    })
        except Exception as exc:
            der_results.append({
                "call_id": cid,
                "status": "ERROR",
                "der": None,
                "error_type": type(exc).__name__,
                "error": str(exc),
            })

        # 3. Broker / Client Role Accuracy (with cross-file speaker consistency validation)
        try:
            raw_gt_role = _load_json_strict(gt_role_path)
            if _is_pending_or_template(raw_gt_role):
                role_results.append({
                    "call_id": cid,
                    "status": "PENDING_GROUND_TRUTH",
                    "role_accuracy": None,
                })
            else:
                validated_gt_role = _validate_roles_gt(raw_gt_role, cid)
                _validate_cross_file_speaker_consistency(
                    cid, validated_gt_tr, validated_gt_diar, validated_gt_role
                )
                hyp_conv = _load_json_strict(Path(conversation_dir) / f"{cid}_conversation.json")
                if hyp_conv is None:
                    role_results.append({
                        "call_id": cid,
                        "status": "MISSING_PIPELINE_OUTPUT",
                        "role_accuracy": None,
                    })
                else:
                    pred_map = hyp_conv.get("speaker_role_mapping", {})
                    if not isinstance(pred_map, dict):
                        raise GroundTruthValidationError(
                            f"Pipeline conversation '{cid}' has invalid 'speaker_role_mapping'."
                        )

                    ref_roles_raw: dict[str, str] = validated_gt_role["speaker_roles"]
                    explicit_pipe_map: dict[str, str] = (
                        validated_gt_role.get("pipeline_speaker_to_reference_speaker") or {}
                    )

                    if explicit_pipe_map:
                        aligned_ref_roles = {
                            pipe_spk: ref_roles_raw[ref_spk]
                            for pipe_spk, ref_spk in explicit_pipe_map.items()
                        }
                    elif set(ref_roles_raw.keys()) & set(pred_map.keys()):
                        aligned_ref_roles = ref_roles_raw
                    elif der_optimal_mapping:
                        aligned_ref_roles = {
                            pipe_spk: ref_roles_raw[ref_spk]
                            for pipe_spk, ref_spk in der_optimal_mapping.items()
                            if ref_spk in ref_roles_raw
                        }
                    else:
                        has_any_valid_pred = any(
                            isinstance(v, str) and v.strip().lower() in VALID_ROLES
                            for v in pred_map.values()
                        )
                        if not has_any_valid_pred:
                            aligned_ref_roles = ref_roles_raw
                        else:
                            raise GroundTruthValidationError(
                                f"Cannot align gt_roles speaker IDs {sorted(ref_roles_raw.keys())} "
                                f"to pipeline speaker IDs {sorted(pred_map.keys())} for '{cid}'. "
                                f"Provide 'pipeline_speaker_to_reference_speaker' in gt_roles or "
                                f"completed gt_diarization turns for overlap-based alignment."
                            )

                    role_m = compute_role_accuracy(aligned_ref_roles, pred_map)
                    role_results.append({
                        "call_id": cid,
                        **role_m,
                    })
        except Exception as exc:
            role_results.append({
                "call_id": cid,
                "status": "ERROR",
                "role_accuracy": None,
                "error_type": type(exc).__name__,
                "error": str(exc),
            })

        # 4. Objective Audio Quality & Noise Analysis
        try:
            wav_path = Path(processed_audio_dir) / f"{cid}_standardized.wav"
            vad_payload = _load_json_strict(Path(vad_dir) / f"{cid}_vad.json")
            qc_payload = _load_json_strict(Path(qc_dir) / f"{cid}_qc.json")
            if not wav_path.exists():
                audio_quality_results.append({
                    "call_id": cid,
                    "status": "PENDING_PIPELINE_OUTPUTS",
                })
            else:
                vad_segs = vad_payload.get("speech_segments") if vad_payload else None
                aq = analyze_audio_quality(wav_path, vad_segments=vad_segs)
                audio_quality_results.append({
                    "call_id": cid,
                    "status": "EVALUATED",
                    "qc_status": qc_payload.get("status") if qc_payload else None,
                    **aq,
                })
        except Exception as exc:
            audio_quality_results.append({
                "call_id": cid,
                "status": "ERROR",
                "error_type": type(exc).__name__,
                "error": str(exc),
            })

    def _counts(items: list[dict[str, Any]]) -> dict[str, int]:
        return {
            "total_calls": len(items),
            "evaluated_calls": sum(1 for x in items if x["status"] == "EVALUATED"),
            "pending_calls": sum(
                1 for x in items if x["status"] in ("PENDING_GROUND_TRUTH", "PENDING_PIPELINE_OUTPUTS")
            ),
            "pending_role_prediction_calls": sum(
                1 for x in items if x["status"] == "PENDING_ROLE_PREDICTIONS"
            ),
            "missing_pipeline_output_calls": sum(
                1 for x in items if x["status"] == "MISSING_PIPELINE_OUTPUT"
            ),
            "error_calls": sum(1 for x in items if x["status"] == "ERROR"),
        }

    wer_counts = _counts(wer_cer_results)
    der_counts = _counts(der_results)
    role_counts = _counts(role_results)
    aq_counts = _counts(audio_quality_results)

    wer_agg = compute_corpus_wer([x["wer_details"] for x in wer_cer_results if x["status"] == "EVALUATED"])
    cer_agg = compute_corpus_cer([x["cer_details"] for x in wer_cer_results if x["status"] == "EVALUATED"])
    der_agg = compute_corpus_der([x for x in der_results if x["status"] == "EVALUATED"])
    role_agg = compute_corpus_role_accuracy([x for x in role_results if x["status"] == "EVALUATED"])

    wer_status = _aggregate_section_status(
        wer_counts["total_calls"], wer_counts["evaluated_calls"], wer_counts["error_calls"]
    )
    der_status = _aggregate_section_status(
        der_counts["total_calls"], der_counts["evaluated_calls"], der_counts["error_calls"]
    )
    role_status = _aggregate_section_status(
        role_counts["total_calls"],
        role_counts["evaluated_calls"],
        role_counts["error_calls"],
        pending_role_prediction_calls=role_counts["pending_role_prediction_calls"],
    )
    aq_status = _aggregate_section_status(
        aq_counts["total_calls"],
        aq_counts["evaluated_calls"],
        aq_counts["error_calls"],
        empty_status="PENDING_PIPELINE_OUTPUTS",
    )

    now = datetime.now(timezone.utc).isoformat()
    wer_cer_report = {
        "generated_at_utc": now,
        "status": wer_status,
        **wer_counts,
        "macro_wer": wer_agg["macro_wer"],
        "micro_wer": wer_agg["micro_wer"],
        "macro_cer": cer_agg["macro_cer"],
        "micro_cer": cer_agg["micro_cer"],
        "corpus_wer_totals": wer_agg,
        "corpus_cer_totals": cer_agg,
        "calls": wer_cer_results,
    }
    der_report = {
        "generated_at_utc": now,
        "collar_sec": collar,
        "status": der_status,
        **der_counts,
        "macro_der": der_agg["macro_der"],
        "micro_der": der_agg["micro_der"],
        "corpus_der_totals": der_agg,
        "calls": der_results,
    }
    role_report = {
        "generated_at_utc": now,
        "status": role_status,
        **role_counts,
        "macro_role_accuracy": role_agg["macro_role_accuracy"],
        "micro_role_accuracy": role_agg["micro_role_accuracy"],
        "corpus_role_totals": role_agg,
        "calls": role_results,
    }
    audio_quality_report = {
        "generated_at_utc": now,
        "status": aq_status,
        **aq_counts,
        "snr_note": (
            "rough_snr_estimate_db is a rough VAD speech-vs-nonspeech RMS ratio estimate, "
            "not a calibrated or reliable measured SNR."
        ),
        "calls": audio_quality_results,
    }

    save_json(reports_dir / "wer_cer_report.json", wer_cer_report)
    save_json(reports_dir / "der_report.json", der_report)
    save_json(reports_dir / "role_accuracy_report.json", role_report)
    save_json(reports_dir / "audio_quality_report.json", audio_quality_report)

    total_errors = (
        wer_counts["error_calls"]
        + der_counts["error_calls"]
        + role_counts["error_calls"]
        + aq_counts["error_calls"]
        + (1 if manifest_error else 0)
    )
    any_evaluated = any(
        c["evaluated_calls"] > 0 for c in (wer_counts, der_counts, role_counts)
    )
    all_core_evaluated = (
        len(targets) > 0
        and wer_status == "EVALUATED"
        and der_status == "EVALUATED"
        and role_status == "EVALUATED"
        and total_errors == 0
    )

    if all_core_evaluated:
        overall_status = "EVALUATED"
    elif any_evaluated:
        overall_status = "PARTIALLY_EVALUATED"
    elif total_errors > 0:
        overall_status = "ERROR"
    elif role_status == "PENDING_ROLE_PREDICTIONS":
        overall_status = "PENDING_ROLE_PREDICTIONS"
    else:
        overall_status = "PENDING_GROUND_TRUTH"

    summary = {
        "generated_at_utc": now,
        "status": overall_status,
        "manifest_loaded": manifest_map is not None,
        "manifest_error": manifest_error,
        "total_calls": len(targets),
        "total_errors": total_errors,
        "call_ids": targets,
        "wer_cer": {
            "status": wer_status,
            "evaluated_calls": wer_counts["evaluated_calls"],
            "pending_calls": wer_counts["pending_calls"],
            "missing_pipeline_output_calls": wer_counts["missing_pipeline_output_calls"],
            "error_calls": wer_counts["error_calls"],
            "macro_wer": wer_agg["macro_wer"],
            "micro_wer": wer_agg["micro_wer"],
            "macro_cer": cer_agg["macro_cer"],
            "micro_cer": cer_agg["micro_cer"],
        },
        "der": {
            "status": der_status,
            "evaluated_calls": der_counts["evaluated_calls"],
            "pending_calls": der_counts["pending_calls"],
            "missing_pipeline_output_calls": der_counts["missing_pipeline_output_calls"],
            "error_calls": der_counts["error_calls"],
            "macro_der": der_agg["macro_der"],
            "micro_der": der_agg["micro_der"],
        },
        "role_accuracy": {
            "status": role_status,
            "evaluated_calls": role_counts["evaluated_calls"],
            "pending_calls": role_counts["pending_calls"],
            "pending_role_prediction_calls": role_counts["pending_role_prediction_calls"],
            "missing_pipeline_output_calls": role_counts["missing_pipeline_output_calls"],
            "error_calls": role_counts["error_calls"],
            "macro_role_accuracy": role_agg["macro_role_accuracy"],
            "micro_role_accuracy": role_agg["micro_role_accuracy"],
        },
        "audio_quality": {
            "status": aq_status,
            "evaluated_calls": aq_counts["evaluated_calls"],
            "pending_calls": aq_counts["pending_calls"],
            "error_calls": aq_counts["error_calls"],
        },
    }
    save_json(reports_dir / "evaluation_summary.json", summary)
    return summary


def get_evaluation_exit_code(summary: dict[str, Any], *, strict: bool = False) -> int:
    """Return process exit code for an evaluation summary."""
    if summary.get("total_errors", 0) > 0:
        return EVAL_EXIT_ERROR
    if strict:
        core_eval = (
            summary.get("wer_cer", {}).get("evaluated_calls", 0)
            + summary.get("der", {}).get("evaluated_calls", 0)
            + summary.get("role_accuracy", {}).get("evaluated_calls", 0)
        )
        if core_eval == 0:
            return EVAL_EXIT_NO_EVALUATED
    return EVAL_EXIT_OK


def main(argv: Optional[list[str]] = None) -> int:
    configure_utf8_console()

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate SmartSemsar Speech Pipeline outputs against verified ground truth. "
            "Exit codes: 0=no errors, 1=evaluation/validation error, 2=strict mode with 0 evaluated calls."
        )
    )
    parser.add_argument(
        "--calls",
        nargs="+",
        default=None,
        help="Optional subset of call IDs (default: calls listed in manifest.json or raw_calls).",
    )
    parser.add_argument(
        "--collar",
        type=float,
        default=0.25,
        help="DER forgiveness collar in seconds around reference turn boundaries (default: 0.25).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return exit code 2 if zero calls were evaluated across WER/CER, DER, and role accuracy.",
    )
    args = parser.parse_args(argv)

    try:
        summary = evaluate_calls(call_ids=args.calls, collar=args.collar)
    except (ValueError, RuntimeError) as exc:
        print(f"\n[EVALUATION ERROR] {exc}", file=sys.stderr)
        return EVAL_EXIT_ERROR

    print("\nEvaluation Summary:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return get_evaluation_exit_code(summary, strict=args.strict)


if __name__ == "__main__":
    sys.exit(main())
