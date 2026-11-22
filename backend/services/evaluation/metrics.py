"""Objective Speech Evaluation metrics (WER, CER, DER, Role Accuracy, Audio Quality).

All transcript metrics reuse `normalize_arabic_text` from `backend.services.pipeline`
to stay strictly consistent with the existing speech pipeline normalization.
"""
from __future__ import annotations

from itertools import permutations
import math
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from backend.services.pipeline import normalize_arabic_text

VALID_ROLES = {"broker", "client"}
FORBIDDEN_ROLE_PLACEHOLDERS = {
    "broker_or_client",
    "client_or_broker",
    "tbd",
    "todo",
    "placeholder",
    "unknown",
    "unassigned",
}


def _validate_finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Invalid {name}: expected a finite number, got {value!r}.")
    num = float(value)
    if not math.isfinite(num):
        raise ValueError(f"Invalid {name}: value must be finite, got {value!r}.")
    return num


def _validate_turns(turns: list[dict[str, Any]], label: str) -> list[dict[str, Any]]:
    """Validate diarization turn timestamps and speaker labels strictly."""
    if not isinstance(turns, list):
        raise ValueError(f"Invalid {label} turns: expected a list, got {type(turns).__name__}.")
    validated: list[dict[str, Any]] = []
    for idx, t in enumerate(turns):
        if not isinstance(t, dict):
            raise ValueError(f"Invalid {label} turn at index {idx}: expected dict, got {type(t).__name__}.")
        if "start" not in t or "end" not in t:
            raise ValueError(f"Invalid {label} turn at index {idx}: missing 'start' or 'end'.")
        start = _validate_finite_number(t["start"], f"{label}[{idx}].start")
        end = _validate_finite_number(t["end"], f"{label}[{idx}].end")
        if start < 0.0:
            raise ValueError(f"Invalid {label} turn at index {idx}: negative start timestamp ({start}).")
        if end <= start:
            raise ValueError(
                f"Invalid {label} turn at index {idx}: end ({end}) must be strictly greater than start ({start})."
            )
        raw_speaker = t.get("speaker_id") if t.get("speaker_id") is not None else t.get("speaker")
        if not isinstance(raw_speaker, str) or not raw_speaker.strip():
            raise ValueError(
                f"Invalid {label} turn at index {idx}: speaker label must be a non-empty string."
            )
        validated.append({
            "start": start,
            "end": end,
            "speaker": raw_speaker.strip(),
        })
    return validated


def _levenshtein_counts(ref: Sequence[str], hyp: Sequence[str]) -> tuple[int, int, int, int]:
    """Compute edit distance and (substitutions, deletions, insertions) via DP."""
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    ops = [[(0, 0, 0)] * (m + 1) for _ in range(n + 1)]  # (sub, del, ins)

    for i in range(1, n + 1):
        dp[i][0] = i
        ops[i][0] = (0, i, 0)
    for j in range(1, m + 1):
        dp[0][j] = j
        ops[0][j] = (0, 0, j)

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
                ops[i][j] = ops[i - 1][j - 1]
            else:
                sub_cost = dp[i - 1][j - 1] + 1
                del_cost = dp[i - 1][j] + 1
                ins_cost = dp[i][j - 1] + 1
                best = min(sub_cost, del_cost, ins_cost)
                dp[i][j] = best
                if best == sub_cost:
                    s, d, ins = ops[i - 1][j - 1]
                    ops[i][j] = (s + 1, d, ins)
                elif best == del_cost:
                    s, d, ins = ops[i - 1][j]
                    ops[i][j] = (s, d + 1, ins)
                else:
                    s, d, ins = ops[i][j - 1]
                    ops[i][j] = (s, d, ins + 1)

    s, d, ins = ops[n][m]
    return dp[n][m], s, d, ins


def compute_wer(
    reference_text: str,
    hypothesis_text: str,
    *,
    normalize: bool = True,
) -> dict[str, Any]:
    """Compute Word Error Rate (WER) between reference and hypothesis Arabic text."""
    if not isinstance(reference_text, str):
        raise ValueError("Reference transcript must be a string.")
    if not isinstance(hypothesis_text, str):
        raise ValueError("Hypothesis transcript must be a string.")

    ref_str = normalize_arabic_text(reference_text) if normalize else reference_text.strip()
    hyp_str = normalize_arabic_text(hypothesis_text) if normalize else hypothesis_text.strip()

    ref_words = ref_str.split() if ref_str else []
    hyp_words = hyp_str.split() if hyp_str else []

    if not ref_words:
        raise ValueError("Reference transcript is empty; cannot compute WER.")

    edits, subs, dels, ins = _levenshtein_counts(ref_words, hyp_words)
    return {
        "wer": round(edits / len(ref_words), 6),
        "reference_words": len(ref_words),
        "hypothesis_words": len(hyp_words),
        "edit_distance": edits,
        "substitutions": subs,
        "deletions": dels,
        "insertions": ins,
        "normalized": normalize,
    }


def compute_cer(
    reference_text: str,
    hypothesis_text: str,
    *,
    normalize: bool = True,
    ignore_spaces: bool = False,
) -> dict[str, Any]:
    """Compute Character Error Rate (CER) between reference and hypothesis text."""
    if not isinstance(reference_text, str):
        raise ValueError("Reference transcript must be a string.")
    if not isinstance(hypothesis_text, str):
        raise ValueError("Hypothesis transcript must be a string.")

    ref_str = normalize_arabic_text(reference_text) if normalize else reference_text.strip()
    hyp_str = normalize_arabic_text(hypothesis_text) if normalize else hypothesis_text.strip()

    if ignore_spaces:
        ref_str = ref_str.replace(" ", "")
        hyp_str = hyp_str.replace(" ", "")

    ref_chars = list(ref_str)
    hyp_chars = list(hyp_str)

    if not ref_chars:
        raise ValueError("Reference transcript is empty; cannot compute CER.")

    edits, subs, dels, ins = _levenshtein_counts(ref_chars, hyp_chars)
    return {
        "cer": round(edits / len(ref_chars), 6),
        "reference_characters": len(ref_chars),
        "hypothesis_characters": len(hyp_chars),
        "edit_distance": edits,
        "substitutions": subs,
        "deletions": dels,
        "insertions": ins,
        "normalized": normalize,
        "ignore_spaces": ignore_spaces,
    }


def compute_corpus_wer(per_call_wer: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-call WER into macro (mean of call WERs) and micro (corpus-level) WER."""
    if not per_call_wer:
        return {
            "macro_wer": None,
            "micro_wer": None,
            "total_reference_words": 0,
            "total_hypothesis_words": 0,
            "total_edit_distance": 0,
            "total_substitutions": 0,
            "total_deletions": 0,
            "total_insertions": 0,
        }
    total_ref = sum(int(m["reference_words"]) for m in per_call_wer)
    total_hyp = sum(int(m["hypothesis_words"]) for m in per_call_wer)
    total_edits = sum(int(m["edit_distance"]) for m in per_call_wer)
    total_subs = sum(int(m["substitutions"]) for m in per_call_wer)
    total_dels = sum(int(m["deletions"]) for m in per_call_wer)
    total_ins = sum(int(m["insertions"]) for m in per_call_wer)
    macro_wer = sum(float(m["wer"]) for m in per_call_wer) / len(per_call_wer)
    micro_wer = (total_edits / total_ref) if total_ref > 0 else None
    return {
        "macro_wer": round(macro_wer, 6),
        "micro_wer": round(micro_wer, 6) if micro_wer is not None else None,
        "total_reference_words": total_ref,
        "total_hypothesis_words": total_hyp,
        "total_edit_distance": total_edits,
        "total_substitutions": total_subs,
        "total_deletions": total_dels,
        "total_insertions": total_ins,
    }


def compute_corpus_cer(per_call_cer: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-call CER into macro (mean of call CERs) and micro (corpus-level) CER."""
    if not per_call_cer:
        return {
            "macro_cer": None,
            "micro_cer": None,
            "total_reference_characters": 0,
            "total_hypothesis_characters": 0,
            "total_edit_distance": 0,
            "total_substitutions": 0,
            "total_deletions": 0,
            "total_insertions": 0,
        }
    total_ref = sum(int(m["reference_characters"]) for m in per_call_cer)
    total_hyp = sum(int(m["hypothesis_characters"]) for m in per_call_cer)
    total_edits = sum(int(m["edit_distance"]) for m in per_call_cer)
    total_subs = sum(int(m["substitutions"]) for m in per_call_cer)
    total_dels = sum(int(m["deletions"]) for m in per_call_cer)
    total_ins = sum(int(m["insertions"]) for m in per_call_cer)
    macro_cer = sum(float(m["cer"]) for m in per_call_cer) / len(per_call_cer)
    micro_cer = (total_edits / total_ref) if total_ref > 0 else None
    return {
        "macro_cer": round(macro_cer, 6),
        "micro_cer": round(micro_cer, 6) if micro_cer is not None else None,
        "total_reference_characters": total_ref,
        "total_hypothesis_characters": total_hyp,
        "total_edit_distance": total_edits,
        "total_substitutions": total_subs,
        "total_deletions": total_dels,
        "total_insertions": total_ins,
    }


def _apply_collar_exclusions(ref_turns: list[dict[str, Any]], collar: float) -> list[tuple[float, float]]:
    """Build merged exclusion intervals of +/- collar around reference boundaries."""
    if collar <= 0:
        return []
    raw = []
    for t in ref_turns:
        for boundary in (float(t["start"]), float(t["end"])):
            raw.append((max(0.0, boundary - collar), boundary + collar))
    raw.sort()
    merged: list[tuple[float, float]] = []
    for s, e in raw:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def _subtract_intervals(
    intervals: list[tuple[float, float]],
    exclusions: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    if not exclusions:
        return intervals
    out = []
    for s, e in intervals:
        cur = [(s, e)]
        for ex_s, ex_e in exclusions:
            nxt = []
            for a, b in cur:
                if ex_e <= a or ex_s >= b:
                    nxt.append((a, b))
                else:
                    if ex_s > a:
                        nxt.append((a, ex_s))
                    if ex_e < b:
                        nxt.append((ex_e, b))
            cur = nxt
        out.extend(cur)
    return [(a, b) for a, b in out if b > a]


def compute_der(
    reference_turns: list[dict[str, Any]],
    hypothesis_turns: list[dict[str, Any]],
    *,
    collar: float = 0.0,
) -> dict[str, Any]:
    """Compute Diarization Error Rate (DER) with optimal 1-to-1 speaker label mapping.

    Neutral cluster IDs (e.g., SPEAKER_00 / SPEAKER_01) are optimally mapped to
    reference speaker IDs so arbitrary label numbering never biases DER.
    Validates collar >= 0, finite non-negative start < end timestamps, and
    non-empty speaker labels.
    """
    collar_val = _validate_finite_number(collar, "collar")
    if collar_val < 0.0:
        raise ValueError(f"Invalid collar: must be >= 0.0 seconds, got {collar_val}.")

    valid_ref = _validate_turns(reference_turns, "reference")
    valid_hyp = _validate_turns(hypothesis_turns, "hypothesis")

    if not valid_ref:
        raise ValueError("Reference diarization turns are empty; cannot compute DER.")

    exclusions = _apply_collar_exclusions(valid_ref, collar_val)
    boundaries = sorted({
        pt
        for items in (valid_ref, valid_hyp)
        for t in items
        for pt in (t["start"], t["end"])
    } | {pt for ex in exclusions for pt in ex})

    ref_speakers = sorted({t["speaker"] for t in valid_ref})
    hyp_speakers = sorted({t["speaker"] for t in valid_hyp})

    # Collect elementary scored intervals and active speaker sets
    segments: list[tuple[float, set[str], set[str]]] = []
    for i in range(len(boundaries) - 1):
        a, b = boundaries[i], boundaries[i + 1]
        if b <= a:
            continue
        if _subtract_intervals([(a, b)], exclusions) == []:
            continue
        mid = 0.5 * (a + b)
        r_active = {t["speaker"] for t in valid_ref if t["start"] <= mid <= t["end"]}
        h_active = {t["speaker"] for t in valid_hyp if t["start"] <= mid <= t["end"]}
        segments.append((b - a, r_active, h_active))

    total_ref_speech = sum(dur * len(r_act) for dur, r_act, _ in segments)
    if total_ref_speech <= 0:
        raise ValueError("Total scored reference speech time is zero; cannot compute DER.")

    # Compute speaker co-occurrence duration matrix to find optimal 1-to-1 mapping
    pair_overlap: dict[tuple[str, str], float] = {}
    for dur, r_act, h_act in segments:
        for r in r_act:
            for h in h_act:
                pair_overlap[(r, h)] = pair_overlap.get((r, h), 0.0) + dur

    # Exact optimal 1-to-1 matching (exact permutation search; speaker count is small, typically 2)
    best_mapping: dict[str, str] = {}
    if ref_speakers and hyp_speakers:
        k = min(len(ref_speakers), len(hyp_speakers))
        best_score = -1.0
        if len(ref_speakers) >= len(hyp_speakers):
            for perm in permutations(ref_speakers, k):
                mapping = {h: r for h, r in zip(hyp_speakers, perm)}
                score = sum(pair_overlap.get((r, h), 0.0) for h, r in mapping.items())
                if score > best_score:
                    best_score = score
                    best_mapping = mapping
        else:
            for perm in permutations(hyp_speakers, k):
                mapping = {h: r for h, r in zip(perm, ref_speakers)}
                score = sum(pair_overlap.get((r, h), 0.0) for h, r in mapping.items())
                if score > best_score:
                    best_score = score
                    best_mapping = mapping

    missed_speech = 0.0
    false_alarm = 0.0
    speaker_confusion = 0.0

    for dur, r_act, h_act in segments:
        n_ref = len(r_act)
        n_hyp = len(h_act)
        mapped_hyp = {best_mapping.get(h, f"__unmapped_{h}") for h in h_act}
        n_correct = len(r_act & mapped_hyp)

        missed_speech += dur * max(0, n_ref - n_hyp)
        false_alarm += dur * max(0, n_hyp - n_ref)
        speaker_confusion += dur * (min(n_ref, n_hyp) - n_correct)

    der = (missed_speech + false_alarm + speaker_confusion) / total_ref_speech
    return {
        "der": round(der, 6),
        "collar_sec": collar_val,
        "total_reference_speech_sec": round(total_ref_speech, 3),
        "total_error_sec": round(missed_speech + false_alarm + speaker_confusion, 3),
        "missed_speech_sec": round(missed_speech, 3),
        "false_alarm_sec": round(false_alarm, 3),
        "speaker_confusion_sec": round(speaker_confusion, 3),
        "missed_speech_rate": round(missed_speech / total_ref_speech, 6),
        "false_alarm_rate": round(false_alarm / total_ref_speech, 6),
        "speaker_confusion_rate": round(speaker_confusion / total_ref_speech, 6),
        "optimal_speaker_mapping": best_mapping,
        "reference_speakers": ref_speakers,
        "hypothesis_speakers": hyp_speakers,
    }


def compute_corpus_der(per_call_der: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-call DER into macro (mean) and micro (speech-duration weighted) DER."""
    if not per_call_der:
        return {
            "macro_der": None,
            "micro_der": None,
            "total_reference_speech_sec": 0.0,
            "total_error_sec": 0.0,
            "total_missed_speech_sec": 0.0,
            "total_false_alarm_sec": 0.0,
            "total_speaker_confusion_sec": 0.0,
            "micro_missed_speech_rate": None,
            "micro_false_alarm_rate": None,
            "micro_speaker_confusion_rate": None,
        }
    total_ref = sum(float(m["total_reference_speech_sec"]) for m in per_call_der)
    total_miss = sum(float(m["missed_speech_sec"]) for m in per_call_der)
    total_fa = sum(float(m["false_alarm_sec"]) for m in per_call_der)
    total_conf = sum(float(m["speaker_confusion_sec"]) for m in per_call_der)
    total_err = total_miss + total_fa + total_conf
    macro_der = sum(float(m["der"]) for m in per_call_der) / len(per_call_der)
    micro_der = (total_err / total_ref) if total_ref > 0 else None
    return {
        "macro_der": round(macro_der, 6),
        "micro_der": round(micro_der, 6) if micro_der is not None else None,
        "total_reference_speech_sec": round(total_ref, 3),
        "total_error_sec": round(total_err, 3),
        "total_missed_speech_sec": round(total_miss, 3),
        "total_false_alarm_sec": round(total_fa, 3),
        "total_speaker_confusion_sec": round(total_conf, 3),
        "micro_missed_speech_rate": round(total_miss / total_ref, 6) if total_ref > 0 else None,
        "micro_false_alarm_rate": round(total_fa / total_ref, 6) if total_ref > 0 else None,
        "micro_speaker_confusion_rate": round(total_conf / total_ref, 6) if total_ref > 0 else None,
    }


def compute_role_accuracy(
    reference_roles: dict[str, str],
    predicted_roles: dict[str, str],
) -> dict[str, Any]:
    """Compute Broker/Client role classification accuracy.

    Expects dicts mapping speaker ID or turn ID -> 'broker' or 'client'.
    Rejects template placeholders (e.g., 'broker_or_client', 'client_or_broker')
    and never assumes SPEAKER_00 is client or SPEAKER_01 is broker.
    """
    if not isinstance(reference_roles, dict) or not reference_roles:
        raise ValueError("Reference role annotations are empty; cannot compute role accuracy.")
    if not isinstance(predicted_roles, dict):
        raise ValueError("Predicted role mapping must be a dict.")

    total = len(reference_roles)
    evaluated = 0
    correct = 0
    unassigned = 0
    mismatches = []

    for key, ref_role_raw in reference_roles.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError("Reference role keys must be non-empty speaker IDs.")
        ref_role = str(ref_role_raw).strip().lower() if ref_role_raw is not None else ""
        if ref_role in FORBIDDEN_ROLE_PLACEHOLDERS or ref_role not in VALID_ROLES:
            raise ValueError(
                f"Invalid reference role '{ref_role_raw}' for '{key}'. "
                f"Final role annotations must be one of {sorted(VALID_ROLES)} (placeholders are rejected)."
            )

        pred_raw = predicted_roles.get(key)
        pred_role = str(pred_raw).strip().lower() if pred_raw is not None else ""
        if pred_role not in VALID_ROLES:
            unassigned += 1
            mismatches.append({"id": key, "reference": ref_role, "predicted": pred_raw or "unassigned"})
            continue

        evaluated += 1
        if pred_role == ref_role:
            correct += 1
        else:
            mismatches.append({"id": key, "reference": ref_role, "predicted": pred_role})

    if evaluated == 0:
        return {
            "status": "PENDING_ROLE_PREDICTIONS",
            "role_accuracy": None,
            "total_items": total,
            "evaluated_items": 0,
            "unassigned_items": unassigned,
            "correct": 0,
            "mismatches": mismatches,
        }

    return {
        "status": "EVALUATED",
        "role_accuracy": round(correct / total, 6),
        "evaluated_only_accuracy": round(correct / evaluated, 6),
        "total_items": total,
        "evaluated_items": evaluated,
        "unassigned_items": unassigned,
        "correct": correct,
        "mismatches": mismatches,
    }


def compute_corpus_role_accuracy(per_call_roles: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-call role accuracy into macro (mean) and micro (item-weighted) accuracy."""
    if not per_call_roles:
        return {
            "macro_role_accuracy": None,
            "micro_role_accuracy": None,
            "total_items": 0,
            "total_evaluated_items": 0,
            "total_correct": 0,
        }
    total_items = sum(int(m["total_items"]) for m in per_call_roles)
    total_eval = sum(int(m["evaluated_items"]) for m in per_call_roles)
    total_correct = sum(int(m["correct"]) for m in per_call_roles)
    macro_acc = sum(float(m["role_accuracy"]) for m in per_call_roles) / len(per_call_roles)
    micro_acc = (total_correct / total_items) if total_items > 0 else None
    return {
        "macro_role_accuracy": round(macro_acc, 6),
        "micro_role_accuracy": round(micro_acc, 6) if micro_acc is not None else None,
        "total_items": total_items,
        "total_evaluated_items": total_eval,
        "total_correct": total_correct,
    }


def _read_wav_float32(wav_path: Path) -> tuple[np.ndarray, int, int]:
    """Read WAV samples as mono float32 along with sample_rate and channel count."""
    try:
        import soundfile as sf
        data, sr = sf.read(str(wav_path), dtype="float32", always_2d=True)
        channels = int(data.shape[1])
        return data.mean(axis=1), int(sr), channels
    except ImportError:
        import wave
        with wave.open(str(wav_path), "rb") as wf:
            sr = wf.getframerate()
            channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            frames = wf.readframes(wf.getnframes())
        if sampwidth != 2:
            raise RuntimeError(
                "Install 'soundfile' to analyze non-16-bit WAV audio: pip install soundfile"
            )
        pcm = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        if channels > 1:
            pcm = pcm.reshape(-1, channels).mean(axis=1)
        return pcm, int(sr), int(channels)


def analyze_audio_quality(
    wav_path: Path | str,
    vad_segments: Optional[list[dict[str, Any]]] = None,
) -> dict[str, Any]:
    """Compute objective audio quality & noise metrics from a standardized WAV file."""
    wav_path = Path(wav_path)
    if not wav_path.exists():
        raise FileNotFoundError(f"WAV file not found for audio analysis: {wav_path}")

    audio, sr, channels = _read_wav_float32(wav_path)
    n_samples = len(audio)
    duration = n_samples / sr if sr > 0 else 0.0

    rms = float(np.sqrt(np.mean(audio ** 2))) if n_samples > 0 else 0.0
    peak = float(np.max(np.abs(audio))) if n_samples > 0 else 0.0
    rms_dbfs = round(20.0 * math.log10(max(rms, 1e-12)), 2)
    peak_dbfs = round(20.0 * math.log10(max(peak, 1e-12)), 2)
    crest_factor_db = round(peak_dbfs - rms_dbfs, 2)
    clipped = int(np.sum(np.abs(audio) >= 0.99))
    clipping_ratio = clipped / n_samples if n_samples > 0 else 0.0

    speech_rms = None
    noise_rms = None
    estimated_snr_db = None
    speech_ratio = None

    if vad_segments is not None and n_samples > 0:
        mask = np.zeros(n_samples, dtype=bool)
        for seg in vad_segments:
            s_idx = max(0, min(n_samples, int(round(float(seg["start"]) * sr))))
            e_idx = max(0, min(n_samples, int(round(float(seg["end"]) * sr))))
            if e_idx > s_idx:
                mask[s_idx:e_idx] = True

        speech_samples = audio[mask]
        noise_samples = audio[~mask]
        speech_ratio = round(float(mask.mean()), 4)

        if len(speech_samples) > 0:
            speech_rms = float(np.sqrt(np.mean(speech_samples ** 2)))
        if len(noise_samples) > 0:
            noise_rms = float(np.sqrt(np.mean(noise_samples ** 2)))
        if speech_rms is not None and noise_rms is not None and noise_rms > 1e-12 and speech_rms > 1e-12:
            estimated_snr_db = round(20.0 * math.log10(speech_rms / noise_rms), 2)

    return {
        "audio_file": wav_path.name,
        "sample_rate": int(sr),
        "channels": channels,
        "duration_sec": round(duration, 3),
        "rms_energy": round(rms, 6),
        "rms_dbfs": rms_dbfs,
        "peak_amplitude": round(peak, 6),
        "peak_dbfs": peak_dbfs,
        "crest_factor_db": crest_factor_db,
        "clipped_samples": clipped,
        "clipping_ratio": round(clipping_ratio, 8),
        "speech_ratio": speech_ratio,
        "speech_rms": round(speech_rms, 6) if speech_rms is not None else None,
        "noise_floor_rms": round(noise_rms, 6) if noise_rms is not None else None,
        "rough_snr_estimate_db": estimated_snr_db,
        "snr_is_rough_estimate": True,
        "snr_estimation_method": "rough_vad_speech_vs_nonspeech_rms_ratio",
        "snr_disclaimer": (
            "Rough estimate derived from VAD speech vs. non-speech RMS energy ratio; "
            "not a calibrated or reliable measured SNR."
        ),
    }
