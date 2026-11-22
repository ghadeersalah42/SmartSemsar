"""Run audio standardization, Silero VAD, objective audio quality audit, and
machine ASR transcription for all 10 SmartSemsar recordings.

- Preserves original MP4 recordings in `backend/services/data/raw_calls/` (read-only).
- Never touches Human Ground Truth files in `backend/services/data/evaluation/ground_truth/`.
- Saves primary `whisper-large-v3-turbo` (via Groq API) machine transcripts to
  `backend/services/data/transcripts/<call_id>_transcript.json` (pipeline schema).
- Saves offline local `faster-whisper-small` transcripts separately to
  `backend/services/data/evaluation/machine_transcripts/faster_whisper_small/`.
- Does NOT fabricate diarization outputs if pyannote is unavailable.

Usage:
    .venv/Scripts/python.exe backend/services/evaluation/run_asr_and_audio_batch.py
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
import traceback
from typing import Any, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(REPO_ROOT / ".env", override=False)

from backend.services.evaluation.metrics import analyze_audio_quality  # noqa: E402
from backend.services.pipeline import (  # noqa: E402
    EVALUATION_DIR,
    PROCESSED_AUDIO_DIR,
    RAW_AUDIO_DIR,
    SPEECH_SEGMENTS_DIR,
    TRANSCRIPT_DIR,
    VAD_DIR,
    discover_calls,
    ensure_output_dirs,
    inspect_audio,
    load_vad_model,
    normalize_arabic_text,
    resolve_audio_path,
    run_vad,
    save_json,
    setup_ffmpeg,
    standardize_audio,
    transcribe as pipeline_transcribe,
)

MACHINE_TRANSCRIPTS_DIR = EVALUATION_DIR / "machine_transcripts"
SMALL_TRANSCRIPTS_DIR = MACHINE_TRANSCRIPTS_DIR / "faster_whisper_small"
GROQ_TRANSCRIPTS_DIR = MACHINE_TRANSCRIPTS_DIR / "groq_whisper_large_v3_turbo"
REPORTS_DIR = EVALUATION_DIR / "reports"


def transcribe_with_groq_whisper(
    wav_path: Path,
    model_name: str = "whisper-large-v3-turbo",
) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    """Transcribe standardized WAV via Groq Whisper API with segment timestamps."""
    from groq import Groq

    if not os.environ.get("GROQ_API_KEY"):
        raise RuntimeError("GROQ_API_KEY is missing in environment/.env.")

    client = Groq()
    with open(wav_path, "rb") as f:
        resp = client.audio.transcriptions.create(
            file=(wav_path.name, f.read()),
            model=model_name,
            language="ar",
            response_format="verbose_json",
            temperature=0.0,
        )

    raw_segments = getattr(resp, "segments", None) or []
    segments: list[dict[str, Any]] = []
    for idx, seg in enumerate(raw_segments, start=1):
        if isinstance(seg, dict):
            s_start = float(seg.get("start", 0.0))
            s_end = float(seg.get("end", 0.0))
            s_text = str(seg.get("text", "")).strip()
            avg_logprob = seg.get("avg_logprob")
            no_speech_prob = seg.get("no_speech_prob")
        else:
            s_start = float(getattr(seg, "start", 0.0))
            s_end = float(getattr(seg, "end", 0.0))
            s_text = str(getattr(seg, "text", "")).strip()
            avg_logprob = getattr(seg, "avg_logprob", None)
            no_speech_prob = getattr(seg, "no_speech_prob", None)

        seg_item: dict[str, Any] = {
            "segment_id": idx,
            "start": round(s_start, 3),
            "end": round(s_end, 3),
            "text": s_text,
            "normalized_text": normalize_arabic_text(s_text),
        }
        if avg_logprob is not None:
            seg_item["avg_logprob"] = round(float(avg_logprob), 4)
        if no_speech_prob is not None:
            seg_item["no_speech_prob"] = round(float(no_speech_prob), 4)
        segments.append(seg_item)

    full_raw = str(getattr(resp, "text", "") or "").strip()
    full_norm = " ".join(s["normalized_text"] for s in segments if s["normalized_text"]).strip()
    if not full_norm and full_raw:
        full_norm = normalize_arabic_text(full_raw)

    meta = {
        "language": getattr(resp, "language", "ar") or "ar",
        "duration": getattr(resp, "duration", None),
        "raw_full_text": full_raw,
        "backend": "groq_api",
        "model": model_name,
    }
    return meta, segments, full_norm


def compute_spectral_profile(wav_path: Path) -> dict[str, Any]:
    """Compute band-energy distribution (sub-300Hz, 300-3400Hz telephony, >3400Hz) for audio audit."""
    import soundfile as sf

    data, sr = sf.read(str(wav_path), dtype="float32", always_2d=True)
    audio = data.mean(axis=1)
    if len(audio) == 0 or sr <= 0:
        return {}

    freqs = np.fft.rfftfreq(len(audio), d=1.0 / sr)
    power = np.abs(np.fft.rfft(audio)) ** 2
    total_power = float(np.sum(power)) + 1e-18

    low_band = float(np.sum(power[freqs < 300.0]) / total_power)
    phone_band = float(np.sum(power[(freqs >= 300.0) & (freqs <= 3400.0)]) / total_power)
    high_band = float(np.sum(power[freqs > 3400.0]) / total_power)
    spectral_centroid_hz = float(np.sum(freqs * power) / total_power)

    return {
        "energy_below_300hz_ratio": round(low_band, 4),
        "energy_phone_band_300_3400hz_ratio": round(phone_band, 4),
        "energy_above_3400hz_ratio": round(high_band, 4),
        "spectral_centroid_hz": round(spectral_centroid_hz, 1),
    }


def run_batch(
    call_ids: Optional[list[str]] = None,
    *,
    run_groq: bool = True,
    run_local_small: bool = True,
    ext: str = "mp4",
) -> dict[str, Any]:
    ensure_output_dirs()
    SMALL_TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    GROQ_TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    targets = call_ids or discover_calls(raw_audio_dir=RAW_AUDIO_DIR, ext=ext)
    if not targets:
        raise FileNotFoundError(f"No .{ext} recordings found in {RAW_AUDIO_DIR}.")

    ffmpeg_path = setup_ffmpeg()
    device = "cpu"
    print(f"Loading Silero VAD once on {device}...")
    vad_bundle = load_vad_model(device)

    small_model = None
    if run_local_small:
        from faster_whisper import WhisperModel

        print("Loading cached local faster-whisper 'small' (CPU int8) once...")
        small_model = WhisperModel("small", device="cpu", compute_type="int8", local_files_only=True)

    call_records: list[dict[str, Any]] = []
    total_raw_sec = 0.0
    total_proc_sec = 0.0
    total_vad_speech_sec = 0.0

    for idx, cid in enumerate(targets, start=1):
        print(f"\n[{idx}/{len(targets)}] === {cid} ===")
        rec: dict[str, Any] = {
            "call_id": cid,
            "audio_origin": "provided_dataset_recording",
            "is_simulated": False,
            "errors": [],
        }
        try:
            clean_id, audio_path = resolve_audio_path(cid, ext=ext, raw_audio_dir=RAW_AUDIO_DIR)
            raw_dur, raw_info = inspect_audio(
                audio_path, clean_id, plot=False, ffmpeg_path=ffmpeg_path, return_details=True
            )
            wav_path = PROCESSED_AUDIO_DIR / f"{clean_id}_standardized.wav"
            proc_dur, wav_info = standardize_audio(
                ffmpeg_path, audio_path, wav_path, return_details=True
            )
            total_raw_sec += raw_dur
            total_proc_sec += proc_dur

            # Silero VAD
            vad_path = VAD_DIR / f"{clean_id}_vad.json"
            vad_result = run_vad(
                wav_path,
                clean_id,
                device,
                vad_bundle=vad_bundle,
                speech_segments_dir=SPEECH_SEGMENTS_DIR,
            )
            save_json(vad_path, vad_result)
            total_vad_speech_sec += float(vad_result.get("total_speech_duration", 0.0))

            # Objective audio quality & spectral profile
            aq = analyze_audio_quality(wav_path, vad_segments=vad_result.get("speech_segments"))
            spectral = compute_spectral_profile(wav_path)

            rec.update({
                "source_audio_file": audio_path.name,
                "standardized_wav": str(wav_path),
                "raw_audio_info": raw_info,
                "standardized_wav_info": wav_info,
                "vad_num_segments": vad_result["num_speech_segments"],
                "vad_speech_duration_sec": vad_result["total_speech_duration"],
                "vad_speech_ratio": vad_result["speech_ratio"],
                "audio_quality": aq,
                "spectral_profile": spectral,
            })

            # Primary ASR: Groq whisper-large-v3-turbo (saved to official pipeline transcript path)
            if run_groq:
                try:
                    meta, segs, full_norm = transcribe_with_groq_whisper(
                        wav_path, model_name="whisper-large-v3-turbo"
                    )
                    payload = {
                        "call_id": clean_id,
                        "source_audio_file": audio_path.name,
                        "audio_file": wav_path.name,
                        "whisper_model": "whisper-large-v3-turbo",
                        "asr_backend": "groq_api",
                        "language": meta["language"],
                        "language_probability": None,
                        "speaker_diarization_applied": False,
                        "diarization_note": (
                            "pyannote/speaker-diarization-community-1 is blocked by Hugging Face "
                            "403 GatedRepoError; no speaker labels fabricated."
                        ),
                        "segments": segs,
                        "full_transcript": full_norm,
                        "raw_full_transcript": meta["raw_full_text"],
                    }
                    # Save to both official transcripts dir (for run_evaluation.py) and machine_transcripts archive
                    save_json(TRANSCRIPT_DIR / f"{clean_id}_transcript.json", payload)
                    save_json(GROQ_TRANSCRIPTS_DIR / f"{clean_id}_transcript.json", payload)
                    rec["groq_whisper_large_v3_turbo"] = {
                        "status": "SUCCESS",
                        "num_segments": len(segs),
                        "word_count": len(full_norm.split()) if full_norm else 0,
                        "transcript_path": str(TRANSCRIPT_DIR / f"{clean_id}_transcript.json"),
                    }
                except Exception as exc:
                    rec["errors"].append(f"groq_asr: {type(exc).__name__}: {exc}")
                    rec["groq_whisper_large_v3_turbo"] = {"status": "ERROR", "error": str(exc)}

            # Secondary ASR: Cached local faster-whisper-small (offline baseline)
            if run_local_small and small_model is not None:
                try:
                    info_s, segs_s, full_s = pipeline_transcribe(
                        wav_path, "cpu", whisper_model=small_model
                    )
                    small_payload = {
                        "call_id": clean_id,
                        "source_audio_file": audio_path.name,
                        "audio_file": wav_path.name,
                        "whisper_model": "faster-whisper-small",
                        "asr_backend": "faster_whisper_local_cpu_int8",
                        "language": info_s.language,
                        "language_probability": round(float(info_s.language_probability), 4),
                        "speaker_diarization_applied": False,
                        "segments": segs_s,
                        "full_transcript": full_s,
                    }
                    out_small = SMALL_TRANSCRIPTS_DIR / f"{clean_id}_transcript.json"
                    save_json(out_small, small_payload)
                    rec["faster_whisper_small"] = {
                        "status": "SUCCESS",
                        "num_segments": len(segs_s),
                        "word_count": len(full_s.split()) if full_s else 0,
                        "transcript_path": str(out_small),
                    }
                except Exception as exc:
                    rec["errors"].append(f"small_asr: {type(exc).__name__}: {exc}")
                    rec["faster_whisper_small"] = {"status": "ERROR", "error": str(exc)}

            rec["status"] = "SUCCESS" if not rec["errors"] else "PARTIAL_ERROR"
        except Exception as exc:
            rec["status"] = "ERROR"
            rec["errors"].append(f"{type(exc).__name__}: {exc}")
            traceback.print_exc()

        call_records.append(rec)

    # Rank recordings by rough SNR estimate to empirically identify noisier vs cleaner recordings
    valid_snrs = [
        (r["call_id"], r["audio_quality"]["rough_snr_estimate_db"])
        for r in call_records
        if r.get("audio_quality", {}).get("rough_snr_estimate_db") is not None
    ]
    valid_snrs_sorted = sorted(valid_snrs, key=lambda x: x[1])

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "total_calls": len(call_records),
        "total_raw_duration_sec": round(total_raw_sec, 3),
        "total_raw_duration_min": round(total_raw_sec / 60.0, 3),
        "total_standardized_duration_sec": round(total_proc_sec, 3),
        "total_standardized_duration_min": round(total_proc_sec / 60.0, 3),
        "total_vad_speech_duration_sec": round(total_vad_speech_sec, 3),
        "total_vad_speech_duration_min": round(total_vad_speech_sec / 60.0, 3),
        "calls_ranked_by_rough_snr_lowest_first": [
            {"call_id": cid, "rough_snr_estimate_db": snr} for cid, snr in valid_snrs_sorted
        ],
        "snr_disclaimer": (
            "rough_snr_estimate_db is a rough VAD speech-vs-nonspeech RMS energy ratio, "
            "not a calibrated SNR measurement. Relative ranking helps identify recordings "
            "with higher background energy/noise floors."
        ),
        "diarization_blocker": (
            "pyannote/speaker-diarization-community-1 returned HTTP 403 GatedRepoError with "
            "current HF_TOKEN and is absent from local cache. No diarization outputs fabricated."
        ),
        "calls": call_records,
    }
    save_json(REPORTS_DIR / "dataset_audio_audit.json", summary)
    return summary


def main(argv: Optional[list[str]] = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Standardize, VAD, audit, and transcribe all 10 calls.")
    parser.add_argument("--calls", nargs="+", default=None)
    parser.add_argument("--skip-groq", action="store_true")
    parser.add_argument("--skip-small", action="store_true")
    args = parser.parse_args(argv)

    summary = run_batch(
        call_ids=args.calls,
        run_groq=not args.skip_groq,
        run_local_small=not args.skip_small,
    )
    print(
        f"\nCompleted {summary['total_calls']} calls | "
        f"Total duration: {summary['total_standardized_duration_sec']}s "
        f"({summary['total_standardized_duration_min']} min) | "
        f"VAD speech: {summary['total_vad_speech_duration_sec']}s "
        f"({summary['total_vad_speech_duration_min']} min)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
