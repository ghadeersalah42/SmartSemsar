"""Call Intelligence Pipeline (Speech Processing & Evaluation-Ready Outputs).

Usage (single call):
    python backend/services/pipeline.py --call-id smartsemsar_rec1 --ext mp4 --skip-profile
    python backend/services/pipeline.py --call-id call_00 --ext ogg --plot --skip-profile

Usage (batch of recordings):
    python backend/services/pipeline.py --batch --ext mp4 --skip-profile
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import traceback
from typing import Any, Optional

import numpy as np
from pydantic import BaseModel, Field

from backend.guardrails.pii import safe_print  # Privacy: names and phone numbers masked in logs

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional if python-dotenv is not installed yet
    def load_dotenv(*_args, **_kwargs):
        return False

# ------------------------------------------------------------------ config
PROJECT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_ROOT.parents[1]

# Load service-local .env first (without overriding shell env), then repo-root .env
load_dotenv(PROJECT_ROOT / ".env", override=False)
load_dotenv(REPO_ROOT / ".env", override=False)

DATA_DIR = PROJECT_ROOT / "data"
RAW_AUDIO_DIR = DATA_DIR / "raw_calls"
PROCESSED_AUDIO_DIR = DATA_DIR / "processed_audio"
SPEECH_SEGMENTS_DIR = PROCESSED_AUDIO_DIR / "speech_segments"
TRANSCRIPT_DIR = DATA_DIR / "transcripts"
VAD_DIR = TRANSCRIPT_DIR / "vad"
DIARIZATION_DIR = TRANSCRIPT_DIR / "diarization"
CONVERSATION_DIR = TRANSCRIPT_DIR / "conversations"
QC_DIR = TRANSCRIPT_DIR / "qc"
PROFILES_DIR = TRANSCRIPT_DIR / "profiles"
BATCH_LOGS_DIR = DATA_DIR / "batch_logs"
EVALUATION_DIR = DATA_DIR / "evaluation"

HF_TOKEN = os.environ.get("HF_TOKEN")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL = os.environ.get("SMARTSEMSAR_GROQ_MODEL", "openai/gpt-oss-120b")

WHISPER_MODEL = "large-v3-turbo"
# Expected real-estate words for Whisper (initial_prompt). Tested on the 10 real calls
# (evaluation/vocab_prompt_experiment.py): "تمليك" 13/17 -> 17/17 and "التجمع" 5/8 -> 8/8 heard
# correctly, extraction agreement +2 requirements. Set SMARTSEMSAR_WHISPER_PROMPT="" to turn it off.
WHISPER_INITIAL_PROMPT = os.environ.get(
    "SMARTSEMSAR_WHISPER_PROMPT",
    "مكالمة عقارية بالعامية المصرية: شقة، فيلا، دوبلكس، أوض نوم، ميزانية، مليون، متر، "
    "التجمع، نيو كايرو، قريبة من المدارس، تشطيب، تقسيط، سكن، استثمار، إيجار، تمليك، غرف، ألف، "
    "يا فندم، مكتب عقارات، التجمع الخامس، العاصمة الإدارية، 6 أكتوبر، الشيخ زايد، مصر الجديدة، "
    "المعادي، مدينة نصر.",
) or None
DIARIZATION_MODEL = "pyannote/speaker-diarization-community-1"
EXPECTED_SPEAKERS = 2
TARGET_SAMPLE_RATE = 16000
TARGET_CHANNELS = 1
UNASSIGNED_SPEAKER = "UNKNOWN"
VALID_SPEAKER_ROLES = ("broker", "client")
SUPPORTED_AUDIO_EXTENSIONS = ("mp4", "ogg", "wav", "m4a", "mp3", "flac")

# Speaker diarization cleanup, alignment & QC policy thresholds
DEFAULT_MIN_TURN_DURATION_SEC = 0.10
DEFAULT_SAME_SPEAKER_MERGE_GAP_SEC = 0.05
DEFAULT_MIN_OVERLAP_SEC = 0.05
DEFAULT_MIN_OVERLAP_RATIO = 0.10
DEFAULT_MIN_SPEAKER_DOMINANCE = 0.65
DEFAULT_MAX_AMBIGUOUS_OVERLAP_RATIO = 0.50
MINOR_TIMESTAMP_OVERSHOOT_TOLERANCE_SEC = 0.10
SEVERE_TIMESTAMP_OVERSHOOT_THRESHOLD_SEC = 1.00

# Process exit codes (applied consistently to single-call and batch runs):
#   0 -> Completed with QC PASS or WARNING (no processing errors, no QC FAIL)
#   1 -> Runtime/processing error (missing file, missing env, exception)
#   2 -> Processing completed, but one or more calls failed QC (qc_status == "FAIL")
EXIT_OK = 0
EXIT_PROCESSING_ERROR = 1
EXIT_QC_FAILED = 2


def ensure_output_dirs(base_dirs: Optional[dict[str, Path]] = None) -> dict[str, Path]:
    """Create all pipeline directories without touching raw audio files."""
    dirs = base_dirs or {
        "data": DATA_DIR,
        "raw_audio": RAW_AUDIO_DIR,
        "processed_audio": PROCESSED_AUDIO_DIR,
        "speech_segments": SPEECH_SEGMENTS_DIR,
        "transcripts": TRANSCRIPT_DIR,
        "vad": VAD_DIR,
        "diarization": DIARIZATION_DIR,
        "conversations": CONVERSATION_DIR,
        "qc": QC_DIR,
        "profiles": PROFILES_DIR,
        "batch_logs": BATCH_LOGS_DIR,
        "evaluation": EVALUATION_DIR,
    }
    for d in dirs.values():
        Path(d).mkdir(parents=True, exist_ok=True)
    return dirs


ensure_output_dirs()


# ------------------------------------------------------------------ dependency & env helpers
def _require_module(import_name: str, pip_name: Optional[str] = None):
    """Import an optional/heavy dependency with an actionable error message."""
    try:
        return __import__(import_name, fromlist=["*"])
    except ImportError as exc:
        pkg = pip_name or import_name
        raise RuntimeError(
            f"Missing required dependency '{import_name}'. "
            f"Install it with: pip install {pkg}"
        ) from exc


def get_hf_token() -> Optional[str]:
    """Return HF_TOKEN from environment (falling back to module-level value)."""
    val = os.environ.get("HF_TOKEN")
    if val is not None:
        return val.strip() or None
    return HF_TOKEN.strip() if isinstance(HF_TOKEN, str) and HF_TOKEN.strip() else None


def get_gemini_api_key() -> Optional[str]:
    """Return GEMINI_API_KEY from environment (falling back to module-level value)."""
    val = os.environ.get("GEMINI_API_KEY")
    if val is not None:
        return val.strip() or None
    return (
        GEMINI_API_KEY.strip()
        if isinstance(GEMINI_API_KEY, str) and GEMINI_API_KEY.strip()
        else None
    )


def get_groq_api_key() -> Optional[str]:
    """Return GROQ_API_KEY from environment (falling back to module-level value)."""
    val = os.environ.get("GROQ_API_KEY")
    if val is not None:
        return val.strip() or None
    return (
        GROQ_API_KEY.strip()
        if isinstance(GROQ_API_KEY, str) and GROQ_API_KEY.strip()
        else None
    )


def configure_utf8_console() -> None:
    """Ensure stdout and stderr emit UTF-8 safely (especially on Windows consoles)."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def validate_environment(require_hf: bool = True, require_gemini: bool = False) -> None:
    """Validate required environment variables up front with clear guidance."""
    missing = []
    if require_hf and not get_hf_token():
        missing.append(
            "HF_TOKEN is missing. Set HF_TOKEN in your environment or in "
            f"'{REPO_ROOT / '.env'}' (or '{PROJECT_ROOT / '.env'}') and ensure your "
            f"Hugging Face account has accepted access terms for '{DIARIZATION_MODEL}'."
        )
    if require_gemini and not get_gemini_api_key():
        missing.append(
            "GEMINI_API_KEY is missing. Set GEMINI_API_KEY in your environment or .env "
            "file, or pass --skip-profile to skip customer profile extraction."
        )
    if missing:
        raise RuntimeError("Missing required environment configuration:\n- " + "\n- ".join(missing))


def load_json(path: Path | str) -> Any:
    """Read a JSON file using UTF-8 (tolerating UTF-8 BOM if present)."""
    with open(Path(path), "r", encoding="utf-8-sig") as f:
        return json.load(f)


def save_json(path: Path | str, data: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print("Saved:", path, flush=True)
    return path


def _natural_sort_key(text: str):
    return [int(tok) if tok.isdigit() else tok.lower() for tok in re.split(r"(\d+)", str(text))]


def normalize_call_id_and_ext(
    call_id: str,
    ext: Optional[str] = None,
    default_ext: str = "ogg",
) -> tuple[str, str]:
    """Normalize a call identifier and audio extension without silently rewriting mismatches.

    Rules:
    - If `call_id` includes a recognized audio extension (e.g., `smartsemsar_rec1.mp4`):
      - When `ext` is None/empty, preserve the filename's extension (`mp4`).
      - When `ext` is explicitly provided and matches, return `(stem, ext)`.
      - When `ext` is explicitly provided and conflicts with the filename's extension,
        raise `ValueError` instead of silently substituting a different extension.
    - If `call_id` has no recognized audio extension, use `ext` or `default_ext`.
    """
    explicit_ext = ext.lstrip(".").lower() if ext else None
    raw_name = Path(call_id).name
    suffix = Path(raw_name).suffix.lstrip(".").lower()

    if suffix and suffix in SUPPORTED_AUDIO_EXTENSIONS:
        if explicit_ext is not None and explicit_ext != suffix:
            raise ValueError(
                f"Audio extension mismatch: call_id '{call_id}' has extension '.{suffix}', "
                f"which conflicts with --ext '{explicit_ext}'. Pass matching --ext '{suffix}' "
                f"or omit --ext when providing a filename with an extension."
            )
        return Path(raw_name).stem, suffix

    resolved_ext = explicit_ext or default_ext.lstrip(".").lower()
    return raw_name, resolved_ext


def resolve_audio_path(
    call_id: str,
    ext: Optional[str] = None,
    raw_audio_dir: Path = RAW_AUDIO_DIR,
    default_ext: str = "ogg",
) -> tuple[str, Path]:
    """Resolve and validate the input audio path for a given call_id and extension."""
    candidate = Path(call_id)
    if candidate.parent != Path(".") and candidate.is_file():
        clean_call_id, clean_ext = normalize_call_id_and_ext(
            candidate.name, ext=ext, default_ext=default_ext
        )
        return clean_call_id, candidate

    clean_call_id, clean_ext = normalize_call_id_and_ext(
        call_id, ext=ext, default_ext=default_ext
    )
    raw_dir = Path(raw_audio_dir)
    audio_path = raw_dir / f"{clean_call_id}.{clean_ext}"

    if not audio_path.exists():
        available = (
            sorted([p.name for p in raw_dir.iterdir() if p.is_file()], key=_natural_sort_key)
            if raw_dir.exists()
            else []
        )
        hint = (
            f"Available files in {raw_dir}: {', '.join(available)}"
            if available
            else f"Directory '{raw_dir}' has no audio files."
        )
        raise FileNotFoundError(
            f"Audio file not found: {audio_path}\n"
            f"Expected filename '{clean_call_id}.{clean_ext}' (call_id='{clean_call_id}', ext='{clean_ext}').\n"
            f"{hint}"
        )

    return clean_call_id, audio_path


# ------------------------------------------------------------------ model container
@dataclass
class PipelineModels:
    """Reusable speech models container to avoid reloading between batch calls."""

    device: str
    vad_bundle: Optional[tuple[Any, Any]] = None
    whisper_model: Optional[Any] = None
    diarization_pipeline: Optional[Any] = None


def load_vad_model(device: str):
    torch = _require_module("torch")
    vad_model, vad_utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad", model="silero_vad", trust_repo=True
    )
    vad_model = vad_model.to(device)
    return vad_model, vad_utils


def load_whisper_model(device: str):
    fw = _require_module("faster_whisper", "faster-whisper")
    compute_type = "float16" if device == "cuda" else "int8"
    print("Whisper device:", device, "| compute type:", compute_type, "| model:", WHISPER_MODEL)
    return fw.WhisperModel(WHISPER_MODEL, device=device, compute_type=compute_type)


def load_diarization_pipeline(device: str):
    torch = _require_module("torch")
    pyannote_audio = _require_module("pyannote.audio", "pyannote.audio")
    token = get_hf_token()
    if not token:
        validate_environment(require_hf=True, require_gemini=False)
    diar_pipeline = pyannote_audio.Pipeline.from_pretrained(DIARIZATION_MODEL, token=token)
    diar_pipeline.to(torch.device(device))
    return diar_pipeline


def load_pipeline_models(device: str) -> PipelineModels:
    """Pre-load VAD, Whisper, and pyannote models once for batch processing."""
    validate_environment(require_hf=True, require_gemini=False)
    print("=" * 60)
    print("LOADING SHARED PIPELINE MODELS")
    print("=" * 60)
    return PipelineModels(
        device=device,
        vad_bundle=load_vad_model(device),
        whisper_model=load_whisper_model(device),
        diarization_pipeline=load_diarization_pipeline(device),
    )


# ------------------------------------------------------------------ runtime
def get_device() -> str:
    torch = _require_module("torch")
    print("=" * 50)
    print("RUNTIME")
    print("=" * 50)
    print("PyTorch version :", torch.__version__)
    print("CUDA available  :", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU             :", torch.cuda.get_device_name(0))
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print("VRAM            :", round(vram, 2), "GB")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Selected device :", device)
    return device


# ------------------------------------------------------------------ audio
def setup_ffmpeg() -> str:
    ffmpeg_path = None
    try:
        import imageio_ffmpeg
        ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        ffmpeg_path = shutil.which("ffmpeg")

    if not ffmpeg_path:
        raise RuntimeError(
            "FFmpeg not found. Run: pip install imageio-ffmpeg "
            "(or install FFmpeg and add it to PATH)."
        )

    os.environ["PATH"] = os.path.dirname(ffmpeg_path) + os.pathsep + os.environ["PATH"]
    result = subprocess.run([ffmpeg_path, "-version"], capture_output=True, text=True)
    lines = result.stdout.splitlines()
    print("FFmpeg:", lines[0] if lines else "version unknown")
    return ffmpeg_path


def _load_raw_audio_for_inspection(
    audio_path: Path,
    ffmpeg_path: Optional[str] = None,
) -> tuple[np.ndarray, int, int]:
    """Load raw audio preserving native sample rate and channel count.

    Falls back to FFmpeg decoding for containers like MP4 when librosa/audioread
    cannot directly invoke a versioned imageio-ffmpeg executable.
    """
    audio_path = Path(audio_path)
    librosa_err = None
    try:
        librosa = _require_module("librosa")
        raw, sr = librosa.load(audio_path, sr=None, mono=False)
        if raw.ndim == 1:
            channels = 1
            mono = raw
        else:
            channels = int(raw.shape[0])
            mono = np.mean(raw, axis=0)
        return mono.astype(np.float32), int(sr), channels
    except Exception as exc:
        librosa_err = exc

    sf = _require_module("soundfile")
    ff = ffmpeg_path or setup_ffmpeg()
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_wav = Path(tmpdir) / "decoded_native.wav"
        cmd = [ff, "-y", "-i", str(audio_path), "-vn", "-c:a", "pcm_f32le", str(tmp_wav)]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0 or not tmp_wav.exists():
            raise RuntimeError(
                f"Failed to decode audio file '{audio_path}'.\n"
                f"librosa error: {librosa_err}\n"
                f"ffmpeg stderr: {res.stderr}"
            ) from librosa_err
        data, sr = sf.read(tmp_wav, dtype="float32", always_2d=True)
        channels = int(data.shape[1])
        mono = data.mean(axis=1)
        return mono.astype(np.float32), int(sr), channels


def inspect_audio(
    audio_path: Path | str,
    call_id: str,
    plot: bool = False,
    ffmpeg_path: Optional[str] = None,
    return_details: bool = False,
):
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    size_mb = round(audio_path.stat().st_size / 1024**2, 3)
    print("=" * 50)
    print("CALL INPUT")
    print("=" * 50)
    print("Call ID :", call_id)
    print("File    :", audio_path)
    print("Size MB :", size_mb)

    audio, sr, channels = _load_raw_audio_for_inspection(audio_path, ffmpeg_path=ffmpeg_path)
    if len(audio) == 0 or sr <= 0:
        raise RuntimeError(f"Audio file '{audio_path}' contains no readable samples.")
    duration = len(audio) / sr

    print("\n" + "=" * 50)
    print("RAW AUDIO INFO")
    print("=" * 50)
    print("Format        :", audio_path.suffix)
    print("Sample rate   :", sr)
    print("Channels      :", channels)
    print("Duration      :", round(duration, 2), "seconds")
    print("Samples       :", len(audio))
    print("Min / Max amp :", round(float(audio.min()), 4), "/", round(float(audio.max()), 4))

    rms = float(np.sqrt(np.mean(audio ** 2)))
    peak = float(np.max(np.abs(audio)))
    clipped = int(np.sum(np.abs(audio) >= 0.99))
    ratio = clipped / len(audio)

    if ratio == 0:
        clipping_status = "No clipping"
    elif ratio < 0.001:
        clipping_status = "Very small clipping"
    else:
        clipping_status = "Significant clipping"

    print("\n" + "=" * 50)
    print("AUDIO QUALITY")
    print("=" * 50)
    print("RMS energy      :", f"{rms:.6f}")
    print("Peak amplitude  :", f"{peak:.6f}")
    print("Clipped samples :", clipped)
    print("Clipping ratio  :", f"{ratio:.6%}")
    print("Status:", clipping_status)

    if plot:
        import matplotlib.pyplot as plt
        import librosa.display
        plt.figure(figsize=(15, 4))
        librosa.display.waveshow(audio, sr=sr)
        plt.title(f"Raw Call Waveform — {call_id}")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Amplitude")
        plt.tight_layout()
        plt.show()

    details = {
        "format": audio_path.suffix.lower(),
        "size_mb": size_mb,
        "sample_rate": int(sr),
        "channels": int(channels),
        "duration": round(float(duration), 3),
        "samples": int(len(audio)),
        "min_amplitude": round(float(audio.min()), 6),
        "max_amplitude": round(float(audio.max()), 6),
        "rms_energy": round(rms, 6),
        "peak_amplitude": round(peak, 6),
        "clipped_samples": clipped,
        "clipping_ratio": round(float(ratio), 8),
        "clipping_status": clipping_status,
    }
    if return_details:
        return duration, details
    return duration


def inspect_wav_properties(wav_path: Path | str) -> dict[str, Any]:
    """Read actual sample rate, channel count, and duration from a WAV file."""
    sf = _require_module("soundfile")
    wav_path = Path(wav_path)
    if not wav_path.exists():
        raise FileNotFoundError(f"WAV file not found: {wav_path}")
    info = sf.info(str(wav_path))
    duration = float(info.frames) / float(info.samplerate) if info.samplerate > 0 else 0.0
    return {
        "file": wav_path.name,
        "path": str(wav_path),
        "sample_rate": int(info.samplerate),
        "channels": int(info.channels),
        "frames": int(info.frames),
        "duration": round(duration, 3),
        "subtype": info.subtype,
    }


def standardize_audio(
    ffmpeg_path: str,
    src: Path | str,
    dst: Path | str,
    return_details: bool = False,
):
    src = Path(src)
    dst = Path(dst)
    if not src.exists():
        raise FileNotFoundError(f"Source audio file not found: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)

    # -vn ensures MP4 video tracks or cover art are ignored cleanly
    command = [
        ffmpeg_path,
        "-y",
        "-i",
        str(src),
        "-vn",
        "-ar",
        str(TARGET_SAMPLE_RATE),
        "-ac",
        str(TARGET_CHANNELS),
        "-c:a",
        "pcm_s16le",
        str(dst),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg standardization failed for '{src}':\n{result.stderr}")

    wav_info = inspect_wav_properties(dst)
    duration = wav_info["duration"]
    print("\n" + "=" * 50)
    print("STANDARDIZED AUDIO")
    print("=" * 50)
    print("File        :", dst.name)
    print("Sample rate :", wav_info["sample_rate"])
    print("Channels    :", wav_info["channels"])
    print("Duration    :", round(duration, 2))
    if return_details:
        return duration, wav_info
    return duration


# ------------------------------------------------------------------ VAD
def run_vad(
    wav_path: Path | str,
    call_id: str,
    device: str,
    vad_bundle: Optional[tuple[Any, Any]] = None,
    speech_segments_dir: Optional[Path] = None,
):
    torch = _require_module("torch")
    sf = _require_module("soundfile")
    wav_path = Path(wav_path)

    if vad_bundle is None:
        vad_model, vad_utils = load_vad_model(device)
    else:
        vad_model, vad_utils = vad_bundle
    get_speech_timestamps = vad_utils[0]

    vad_audio, sample_rate = sf.read(wav_path)
    if vad_audio.ndim > 1:
        vad_audio = vad_audio.mean(axis=1)
    tensor = torch.from_numpy(vad_audio.astype(np.float32)).to(device)

    timestamps = get_speech_timestamps(tensor, vad_model, sampling_rate=sample_rate)

    seg_dir = (Path(speech_segments_dir) if speech_segments_dir else SPEECH_SEGMENTS_DIR) / call_id
    seg_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 50)
    print("VAD RESULT")
    print("=" * 50)
    print("Speech segments:", len(timestamps))
    speech_segments = []
    for i, seg in enumerate(timestamps, start=1):
        s, e = seg["start"] / sample_rate, seg["end"] / sample_rate
        print(f"{i:02d}. {s:.2f}s -> {e:.2f}s ({e - s:.2f}s)")
        out = seg_dir / f"{call_id}_speech_{i:02d}.wav"
        sf.write(out, vad_audio[seg["start"]:seg["end"]], sample_rate)
        speech_segments.append({
            "id": i,
            "start": round(s, 3),
            "end": round(e, 3),
            "duration": round(e - s, 3),
            "audio_segment_file": out.name,
        })

    total_duration = len(vad_audio) / sample_rate if sample_rate > 0 else 0.0
    total_speech = sum(seg["duration"] for seg in speech_segments)
    speech_ratio = round(total_speech / total_duration, 4) if total_duration > 0 else 0.0

    return {
        "call_id": call_id,
        "audio_file": wav_path.name,
        "sample_rate": int(sample_rate),
        "duration": round(total_duration, 3),
        "num_speech_segments": len(speech_segments),
        "total_speech_duration": round(total_speech, 3),
        "speech_ratio": speech_ratio,
        "speech_segments": speech_segments,
    }


# ------------------------------------------------------------------ STT
def normalize_arabic_text(text):
    text = str(text)
    text = text.replace("ـ", "")
    for variant in ("أ", "إ", "آ"):
        text = text.replace(variant, "ا")
    text = text.replace("ى", "ي")
    text = re.sub(r"[\u0617-\u061A\u064B-\u0652]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def transcribe(wav_path: Path | str, device: str, whisper_model: Optional[Any] = None):
    model = whisper_model if whisper_model is not None else load_whisper_model(device)
    segments, info = model.transcribe(
        str(wav_path),
        language="ar",
        task="transcribe",
        beam_size=5,
        vad_filter=False,
        initial_prompt=WHISPER_INITIAL_PROMPT,
    )
    segments = list(segments)
    audio_duration = float(getattr(info, "duration", 0.0) or 0.0)

    print("=" * 60)
    print("WHISPER TRANSCRIPTION")
    print("=" * 60)
    print("Detected language   :", info.language)
    print("Language probability:", round(info.language_probability, 4))
    print("Raw segment count   :", len(segments))

    result = []
    dropped_overflow = 0
    for s in segments:
        raw_start = float(s.start)
        raw_end = float(s.end)
        original = s.text.strip()
        if not original:
            continue
        if audio_duration > 0.0:
            # Drop trailing 30-second-window Whisper silence hallucinations that overflow past EOF
            if raw_start >= audio_duration:
                dropped_overflow += 1
                continue
            if raw_end > audio_duration + 5.0 and raw_start >= max(0.0, audio_duration - 6.0):
                dropped_overflow += 1
                print(
                    f"  [dropped EOF hallucination overflow: {raw_start:.2f}s -> {raw_end:.2f}s "
                    f"(audio duration {audio_duration:.2f}s)]"
                )
                continue
            start_s = round(max(0.0, raw_start), 3)
            end_s = round(min(raw_end, audio_duration), 3)
        else:
            start_s = round(max(0.0, raw_start), 3)
            end_s = round(raw_end, 3)

        if end_s <= start_s:
            continue

        seg_idx = len(result) + 1
        safe_print(f"{seg_idx:02d}. [{start_s:.2f}s -> {end_s:.2f}s] {original}")
        result.append({
            "segment_id": seg_idx,
            "start": start_s,
            "end": end_s,
            "text": original,
            "normalized_text": normalize_arabic_text(original),
        })

    if dropped_overflow:
        print(f"Filtered overflow segments: {dropped_overflow}")
    full_text = " ".join(x["normalized_text"] for x in result if x["normalized_text"]).strip()
    return info, result, full_text


# ------------------------------------------------------------------ diarization
def clean_diarization_turns(
    turns: list[dict[str, Any]],
    *,
    min_turn_duration_sec: float = DEFAULT_MIN_TURN_DURATION_SEC,
    merge_gap_sec: float = DEFAULT_SAME_SPEAKER_MERGE_GAP_SEC,
    audio_duration: Optional[float] = None,
) -> list[dict[str, Any]]:
    """Remove sub-threshold frame-boundary glitch slivers and merge adjacent same-speaker turns.

    Never drops all turns belonging to a speaker that has no longer turns, so detected
    speaker count is preserved while 17ms-80ms boundary artifacts are cleaned up.
    """
    normalized: list[dict[str, Any]] = []
    for t in turns:
        s = float(t["start"])
        e = float(t["end"])
        spk = str(t["speaker"])
        if audio_duration is not None and audio_duration > 0.0:
            s = max(0.0, min(s, audio_duration))
            e = max(0.0, min(e, audio_duration))
        else:
            s = max(0.0, s)
        if e <= s:
            continue
        normalized.append({
            "start": round(s, 3),
            "end": round(e, 3),
            "duration": round(e - s, 3),
            "speaker": spk,
        })

    normalized.sort(key=lambda item: (item["start"], item["end"], item["speaker"]))
    if not normalized:
        return []

    # Track which speakers have at least one turn >= min_turn_duration_sec
    speakers_with_valid = {
        t["speaker"] for t in normalized if (t["end"] - t["start"]) >= min_turn_duration_sec
    }
    filtered = [
        t
        for t in normalized
        if (t["end"] - t["start"]) >= min_turn_duration_sec
        or t["speaker"] not in speakers_with_valid
    ]

    merged: list[dict[str, Any]] = []
    for t in filtered:
        if (
            merged
            and merged[-1]["speaker"] == t["speaker"]
            and (t["start"] - merged[-1]["end"]) <= merge_gap_sec
        ):
            merged[-1]["end"] = max(merged[-1]["end"], t["end"])
            merged[-1]["duration"] = round(merged[-1]["end"] - merged[-1]["start"], 3)
        else:
            merged.append(dict(t))

    return merged


def run_diarization(
    wav_path: Path | str,
    device: str,
    num_speakers: int = EXPECTED_SPEAKERS,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
    diarization_pipeline: Optional[Any] = None,
    *,
    min_turn_duration_sec: float = DEFAULT_MIN_TURN_DURATION_SEC,
    merge_gap_sec: float = DEFAULT_SAME_SPEAKER_MERGE_GAP_SEC,
):
    torch = _require_module("torch")
    sf = _require_module("soundfile")

    min_spk = num_speakers if min_speakers is None else min_speakers
    max_spk = num_speakers if max_speakers is None else max_speakers
    if min_spk < 1 or max_spk < min_spk:
        raise ValueError(
            f"Invalid speaker bounds: min_speakers={min_spk}, max_speakers={max_spk}."
        )

    pipeline = (
        diarization_pipeline
        if diarization_pipeline is not None
        else load_diarization_pipeline(device)
    )

    data, sr = sf.read(str(wav_path), dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    audio_duration = (len(data) / float(sr)) if sr > 0 else None
    waveform = torch.from_numpy(data).unsqueeze(0)  # (1, samples)

    output = pipeline(
        {"waveform": waveform, "sample_rate": sr},
        min_speakers=min_spk,
        max_speakers=max_spk,
    )
    exclusive = getattr(output, "exclusive_speaker_diarization", None)
    if exclusive is None:
        exclusive = getattr(output, "speaker_diarization", output)

    raw_turns = []
    for t, _, spk in exclusive.itertracks(yield_label=True):
        start_s = round(float(t.start), 3)
        end_s = round(float(t.end), 3)
        raw_turns.append({
            "start": start_s,
            "end": end_s,
            "duration": round(max(0.0, end_s - start_s), 3),
            "speaker": str(spk),
        })
    turns = clean_diarization_turns(
        raw_turns,
        min_turn_duration_sec=min_turn_duration_sec,
        merge_gap_sec=merge_gap_sec,
        audio_duration=audio_duration,
    )
    speakers = sorted({t["speaker"] for t in turns})

    print("=" * 60)
    print("SPEAKER DIARIZATION")
    print("=" * 60)
    print(f"Configured speakers: min={min_spk}, max={max_spk}")
    print(f"Turns (raw -> clean): {len(raw_turns)} -> {len(turns)}")
    print("Detected speakers  :", speakers)
    for i, t in enumerate(turns, start=1):
        print(f"{i:02d}. [{t['start']:.2f}s -> {t['end']:.2f}s] {t['speaker']}")
    return turns, speakers


# ------------------------------------------------------------------ alignment
def overlap_duration(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def assign_speaker(
    start: float,
    end: float,
    turns: list[dict[str, Any]],
    return_details: bool = False,
    *,
    min_overlap_sec: float = DEFAULT_MIN_OVERLAP_SEC,
    min_overlap_ratio: float = DEFAULT_MIN_OVERLAP_RATIO,
    min_speaker_dominance: float = DEFAULT_MIN_SPEAKER_DOMINANCE,
):
    """Match a transcript segment [start, end] to speaker diarization turns.

    Accumulates total overlap duration per speaker across all overlapping turns,
    requires overlap to meet configurable `min_overlap_sec` and `min_overlap_ratio`
    thresholds, and flags ambiguous multi-speaker assignments when the top speaker's
    share of overlapping speech falls below `min_speaker_dominance`.
    """
    if start is None or end is None or end <= start or start < 0:
        if return_details:
            return None, 0.0, {
                "speaker_overlaps": {},
                "overlapping_speakers": [],
                "overlap_ratio": 0.0,
                "speaker_dominance": 0.0,
                "is_ambiguous": False,
                "assignment_status": "invalid_timestamp",
            }
        return None, 0.0

    speaker_overlaps: dict[str, float] = {}
    for t in turns:
        t_start = t.get("start")
        t_end = t.get("end")
        t_speaker = t.get("speaker")
        if (
            t_start is None
            or t_end is None
            or t_end <= t_start
            or not t_speaker
        ):
            continue
        ov = overlap_duration(start, end, t_start, t_end)
        if ov > 0.0:
            speaker_overlaps[t_speaker] = speaker_overlaps.get(t_speaker, 0.0) + ov

    if not speaker_overlaps:
        if return_details:
            return None, 0.0, {
                "speaker_overlaps": {},
                "overlapping_speakers": [],
                "overlap_ratio": 0.0,
                "speaker_dominance": 0.0,
                "is_ambiguous": False,
                "assignment_status": "unassigned_no_overlap",
            }
        return None, 0.0

    # Deterministic selection: highest accumulated overlap, tie-break by speaker ID
    best_speaker, best_overlap = max(
        sorted(speaker_overlaps.items(), key=lambda kv: kv[0]),
        key=lambda kv: kv[1],
    )
    seg_duration = end - start
    total_overlap = sum(speaker_overlaps.values())
    overlap_ratio = min(1.0, best_overlap / seg_duration) if seg_duration > 0 else 0.0
    speaker_dominance = (best_overlap / total_overlap) if total_overlap > 0 else 0.0
    overlapping_speakers = sorted(speaker_overlaps.keys())

    # Reject assignments that fall below the minimum overlap confidence thresholds
    if best_overlap < min_overlap_sec or overlap_ratio < min_overlap_ratio:
        if return_details:
            return None, best_overlap, {
                "speaker_overlaps": {k: round(v, 3) for k, v in sorted(speaker_overlaps.items())},
                "overlapping_speakers": overlapping_speakers,
                "overlap_ratio": round(overlap_ratio, 4),
                "speaker_dominance": round(speaker_dominance, 4),
                "is_ambiguous": len(overlapping_speakers) > 1,
                "assignment_status": "unassigned_below_threshold",
            }
        return None, best_overlap

    is_multi = len(overlapping_speakers) > 1
    is_ambiguous = is_multi and (speaker_dominance < min_speaker_dominance)
    if is_ambiguous:
        status = "assigned_ambiguous_multi_speaker"
    elif is_multi:
        status = "assigned_multi_speaker_overlap"
    else:
        status = "assigned"

    if return_details:
        return best_speaker, best_overlap, {
            "speaker_overlaps": {k: round(v, 3) for k, v in sorted(speaker_overlaps.items())},
            "overlapping_speakers": overlapping_speakers,
            "overlap_ratio": round(overlap_ratio, 4),
            "speaker_dominance": round(speaker_dominance, 4),
            "is_ambiguous": is_ambiguous,
            "assignment_status": status,
        }
    return best_speaker, best_overlap


def build_conversation(
    transcript_segments: list[dict[str, Any]],
    speaker_turns: list[dict[str, Any]],
    *,
    min_overlap_sec: float = DEFAULT_MIN_OVERLAP_SEC,
    min_overlap_ratio: float = DEFAULT_MIN_OVERLAP_RATIO,
    min_speaker_dominance: float = DEFAULT_MIN_SPEAKER_DOMINANCE,
) -> list[dict[str, Any]]:
    """Build speaker-labeled conversation preserving timestamps and neutral speaker IDs.

    Segments with zero overlap or overlap below threshold against diarization turns
    are explicitly marked as UNASSIGNED_SPEAKER ('UNKNOWN') with speaker_assigned=False
    instead of silently guessing a speaker. Neutral pyannote speaker labels are
    preserved without assuming which speaker is client or broker.
    """
    conversation = []
    for seg in transcript_segments:
        speaker, best_overlap, details = assign_speaker(
            seg["start"],
            seg["end"],
            speaker_turns,
            return_details=True,
            min_overlap_sec=min_overlap_sec,
            min_overlap_ratio=min_overlap_ratio,
            min_speaker_dominance=min_speaker_dominance,
        )
        speaker_assigned = speaker is not None
        label = speaker if speaker_assigned else UNASSIGNED_SPEAKER

        conversation.append({
            "turn_id": len(conversation) + 1,
            "start": seg["start"],
            "end": seg["end"],
            "speaker": label,
            "speaker_assigned": speaker_assigned,
            "overlap_duration": round(best_overlap, 3),
            "overlap_ratio": details["overlap_ratio"],
            "speaker_dominance": details["speaker_dominance"],
            "is_ambiguous": details["is_ambiguous"],
            "overlapping_speakers": details["overlapping_speakers"],
            "assignment_status": details["assignment_status"],
            "raw_text": seg.get("text", ""),
            "text": seg["normalized_text"],
        })

    print("=" * 60)
    print("FINAL CONVERSATION")
    print("=" * 60)
    for t in conversation:
        safe_print(f"[{t['start']:.2f}s -> {t['end']:.2f}s] {t['speaker']}: {t['text']}")
    return conversation


# ------------------------------------------------------------------ speaker role assignment
ROLE_ASSIGNMENT_SYSTEM_PROMPT = """You are a conversation analyst for SmartSemsar, an Egyptian Arabic real-estate phone call intake system.

You receive a diarized phone call transcript with NEUTRAL speaker labels (e.g., SPEAKER_00, SPEAKER_01).
Determine which speaker ID is the real-estate "broker" and which speaker ID is the "client" based strictly on conversational content.

Role definitions:
- "broker": The real-estate agent/office representative who answers or opens the intake call ("مكتب عقارات", "مع حضرتك", "يا فندم", "اتفضل/اتفضلي"), asks qualifying questions (location, number of rooms, rent vs buy / إيجار ولا تمليك, budget / الميزانية في حدود كام), summarizes the customer's requirements back for confirmation, or closes the call ("شكرا لاتصال حضرتك", "هشوفلك اختيارات", "ابعتلك التفاصيل").
- "client": The customer calling/speaking to request a property ("انا بدور علي...", "محتاج/محتاجة...", "عايز/عايزة..."), answering the broker's questions with their desired area, room count, rent/buy preference, or budget.

Strict rules:
1. NEVER assume SPEAKER_00 is the broker or client; judge strictly from what each speaker says.
2. In a two-speaker call where one speaker clearly asks broker intake questions / greets as the real-estate office / summarizes and thanks the caller, assign that speaker "broker" and the other speaker "client".
3. Transcripts may contain ASR typos (e.g., "عقاعد/عقراط/عكرات" for "عقارات", "قوضة/اوضة" for "غرفة", "دمليك/تمليل" for "تمليك", "بيلا/بلا" for "فيلا"). Use context to understand intent.
4. Only return "unassigned" if there is genuinely no conversational evidence (for example, an empty transcript or completely unintelligible text).
5. Respond ONLY with a valid JSON object matching this exact schema (no markdown fences):
{
  "speaker_role_mapping": {
    "<SPEAKER_ID>": "broker" | "client" | "unassigned"
  },
  "confidence": "high" | "medium" | "low",
  "reasoning": "Brief factual explanation referencing conversational cues"
}
"""

# Weighted Egyptian Arabic real-estate dialogue cues (normalized text) for deterministic fallback
_BROKER_CUE_PATTERNS: tuple[tuple[str, float], ...] = (
    (r"مكتب\s*(?:عقار|عقارات|عقاعد|عقراط|عكرات|عطارات)", 4.0),
    (r"مع\s+حضرتك", 3.0),
    (r"شكرا\s+(?:لاتصال|لاتصار|لك)", 3.5),
    (r"في\s+امان\s+الله", 2.0),
    (r"(?:هاشوفلك|هشوفلك|اشوفلك|ازبطلك|ابعتلك|احددلك)", 3.0),
    (r"اختيارات\s+مناسبة", 2.5),
    (r"تحتاج\s+المساعدة", 3.0),
    (r"كنت\s+(?:حابب|احبب)\s+اعرف", 3.0),
    (r"(?:كنت|حضرتك)\s+(?:بتدور|بتدوري|بالدوري)", 3.0),
    (r"(?:ايجار|بيجار)\s+(?:ولا|او)\s+(?:تمليك|تمليل|تمليئ)", 3.0),
    (r"(?:تمليك|دمليك|شراء)\s+(?:ولا|او)\s+ايجار", 3.0),
    (r"شقة\s+ولا\s+(?:فيلا|بيلا)", 3.0),
    (r"(?:كام|كم)\s+(?:غرف|غرفة|اوض|اوضة|قوض|قوضة)", 2.5),
    (r"عدد\s+الغرف", 2.0),
    (r"(?:في\s+حدود|محدود)\s+(?:كام|كم)", 2.5),
    (r"ميزانيتك", 2.5),
    (r"والميزانية", 1.8),
    (r"الميزانية\s+(?:تقريبا|اللي\s+في\s+بالك)", 2.5),
    (r"منطقة\s+معينة", 2.0),
    (r"يا\s*فندم", 1.8),
    (r"اتفضل|اتفضلي|اتفضالي", 1.8),
    (r"حضرتك", 1.2),
    (r"تمام\s+(?:يا\s*فندم|العاصمة|مصر|شقة)", 1.5),
    (r"(?:يبقي|ماشي|طيب|خلاص)\s+(?:شقة|فيلا|بيلا|بلا)", 2.0),
    (r"صح\s*[؟?]", 1.5),
    (r"[؟?]", 0.6),
)

_CLIENT_CUE_PATTERNS: tuple[tuple[str, float], ...] = (
    (r"(?:انا\s+)?بدور\s+علي", 3.5),
    (r"(?:انا\s+)?(?:محتاج|محتاجة)", 3.5),
    (r"(?:انا\s+)?(?:عايز|عايزة|هايزة)", 3.5),
    (r"(?:معايا|معي)\s+ميزانية", 3.0),
    (r"شراء\s+طبعا", 2.5),
    (r"مش\s+(?:عايز|عايزة|هايزة|عارفة|عارف)", 2.0),
    (r"ما\s+عرفش|معرفش", 2.0),
    (r"بفضل\s+يعني", 2.0),
    (r"اه\s+بدور", 3.0),
)


def _infer_speaker_roles_heuristic(
    conversation: list[dict[str, Any]],
    speakers: list[str],
) -> tuple[dict[str, str], dict[str, Any]]:
    """Deterministic role classifier using Egyptian Arabic broker/client conversational cues."""
    mapping: dict[str, str] = {spk: "unassigned" for spk in speakers}
    if not speakers or not conversation:
        return mapping, {"method": "heuristic", "confidence": "low", "scores": {}, "reasoning": "No speakers or turns."}

    speaker_texts: dict[str, list[str]] = {spk: [] for spk in speakers}
    for turn in conversation:
        spk = turn.get("speaker")
        if spk in speaker_texts:
            txt = normalize_arabic_text(turn.get("text") or turn.get("raw_text") or "")
            if txt:
                speaker_texts[spk].append(txt)

    scores: dict[str, dict[str, float]] = {}
    total_evidence = 0.0
    for spk in speakers:
        joined = " ".join(speaker_texts[spk])
        b_score = 0.0
        c_score = 0.0
        for pat, weight in _BROKER_CUE_PATTERNS:
            hits = len(re.findall(pat, joined))
            if hits:
                b_score += weight * hits
        for pat, weight in _CLIENT_CUE_PATTERNS:
            hits = len(re.findall(pat, joined))
            if hits:
                c_score += weight * hits
        total_evidence += b_score + c_score
        scores[spk] = {
            "broker_score": round(b_score, 3),
            "client_score": round(c_score, 3),
            "net_broker_score": round(b_score - c_score, 3),
            "word_count": float(len(joined.split())) if joined else 0.0,
        }

    if total_evidence <= 0.0:
        return mapping, {
            "method": "heuristic",
            "confidence": "low",
            "scores": scores,
            "reasoning": "No broker/client conversational cues detected.",
        }

    if len(speakers) == 2:
        s0, s1 = speakers[0], speakers[1]
        n0 = scores[s0]["net_broker_score"]
        n1 = scores[s1]["net_broker_score"]
        if n0 != n1:
            broker_spk, client_spk = (s0, s1) if n0 > n1 else (s1, s0)
            mapping[broker_spk] = "broker"
            mapping[client_spk] = "client"
            margin = abs(n0 - n1)
            conf = "high" if margin >= 3.0 else ("medium" if margin >= 1.5 else "low")
            return mapping, {
                "method": "heuristic",
                "confidence": conf,
                "scores": scores,
                "reasoning": f"Assigned {broker_spk}=broker (net={scores[broker_spk]['net_broker_score']}) vs {client_spk}=client (net={scores[client_spk]['net_broker_score']}).",
            }
        return mapping, {
            "method": "heuristic",
            "confidence": "low",
            "scores": scores,
            "reasoning": "Tied broker/client cue scores between speakers.",
        }

    # General N-speaker fallback (N != 2)
    for spk in speakers:
        net = scores[spk]["net_broker_score"]
        if net > 0:
            mapping[spk] = "broker"
        elif net < 0:
            mapping[spk] = "client"
    return mapping, {
        "method": "heuristic",
        "confidence": "medium",
        "scores": scores,
        "reasoning": "Assigned roles individually by net conversational cue score.",
    }


def _call_role_assignment_llm(prompt_text: str) -> tuple[Optional[dict[str, Any]], Optional[str], Optional[str]]:
    """Invoke Groq (preferred, matching user_input_text.py) or Gemini for role assignment."""
    groq_key = get_groq_api_key()
    if groq_key:
        try:
            from groq import Groq

            model_name = os.environ.get("SMARTSEMSAR_GROQ_MODEL", GROQ_MODEL)
            client = Groq(api_key=groq_key)
            resp = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": ROLE_ASSIGNMENT_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt_text},
                ],
                response_format={"type": "json_object"},
                temperature=0,
            )
            raw = (resp.choices[0].message.content or "").strip()
            return json.loads(raw), f"groq:{model_name}", None
        except Exception as exc:
            groq_err = f"Groq failed ({type(exc).__name__}: {exc})"
    else:
        groq_err = None

    gemini_key = get_gemini_api_key()
    if gemini_key:
        try:
            from langchain_core.messages import HumanMessage, SystemMessage
            from langchain_google_genai import ChatGoogleGenerativeAI

            model_name = os.environ.get("GEMINI_MODEL", GEMINI_MODEL)
            llm = ChatGoogleGenerativeAI(
                model=model_name, google_api_key=gemini_key, temperature=0, request_timeout=20, retries=0
            )
            resp = llm.invoke([
                SystemMessage(content=ROLE_ASSIGNMENT_SYSTEM_PROMPT),
                HumanMessage(content=prompt_text),
            ])
            raw = str(resp.content or "").strip()
            if raw.startswith("```"):
                raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.IGNORECASE | re.DOTALL).strip()
            return json.loads(raw), f"gemini:{model_name}", None
        except Exception as exc:
            gem_err = f"Gemini failed ({type(exc).__name__}: {exc})"
            err = "; ".join(e for e in (groq_err, gem_err) if e)
            return None, None, err

    return None, None, groq_err or "No LLM API key configured (GROQ_API_KEY / GEMINI_API_KEY)."


def infer_speaker_roles(
    conversation: list[dict[str, Any]],
    speakers: Optional[list[str]] = None,
    *,
    use_llm: bool = True,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Infer 'broker' / 'client' roles for neutral diarization speaker IDs from conversation context.

    Uses LLM contextual analysis (Groq / Gemini, matching existing project architecture) when
    available, with a deterministic Egyptian Arabic conversational heuristic fallback.
    Never hardcodes speaker IDs and never accesses Ground Truth files.
    """
    detected_speakers = sorted({
        *(speakers or []),
        *(
            t["speaker"]
            for t in conversation
            if t.get("speaker") and t["speaker"] != UNASSIGNED_SPEAKER
        ),
    })
    if not detected_speakers:
        return {}, {
            "method": "none",
            "confidence": "low",
            "reasoning": "No assigned speakers found in conversation.",
        }

    transcript_lines = [
        f"[{t.get('start', 0.0):.2f}s -> {t.get('end', 0.0):.2f}s] {t.get('speaker')}: {t.get('text') or t.get('raw_text') or ''}"
        for t in conversation
        if (t.get("text") or t.get("raw_text") or "").strip()
    ]
    if not transcript_lines:
        return {spk: "unassigned" for spk in detected_speakers}, {
            "method": "none",
            "confidence": "low",
            "reasoning": "Conversation contains no spoken text.",
        }

    llm_error = None
    if use_llm:
        prompt_text = (
            f"Speakers detected: {', '.join(detected_speakers)}\n\n"
            "Conversation transcript:\n" + "\n".join(transcript_lines)
        )
        parsed, provider, llm_error = _call_role_assignment_llm(prompt_text)
        if isinstance(parsed, dict) and isinstance(parsed.get("speaker_role_mapping"), dict):
            raw_map = parsed["speaker_role_mapping"]
            cleaned_map: dict[str, str] = {}
            for spk in detected_speakers:
                val = str(raw_map.get(spk, "unassigned")).strip().lower()
                cleaned_map[spk] = val if val in VALID_SPEAKER_ROLES else "unassigned"

            # Accept LLM prediction if at least one speaker received a valid role
            if any(v in VALID_SPEAKER_ROLES for v in cleaned_map.values()):
                print("=" * 60)
                print("SPEAKER ROLE ASSIGNMENT")
                print("=" * 60)
                print("Method  :", provider)
                print("Mapping :", cleaned_map)
                return cleaned_map, {
                    "method": provider,
                    "confidence": str(parsed.get("confidence", "medium")).strip().lower(),
                    "reasoning": str(parsed.get("reasoning", "")).strip(),
                }
            llm_error = "LLM returned only 'unassigned' roles; falling back to heuristic."

    heur_map, heur_details = _infer_speaker_roles_heuristic(conversation, detected_speakers)
    if llm_error:
        heur_details["llm_fallback_reason"] = llm_error
    print("=" * 60)
    print("SPEAKER ROLE ASSIGNMENT")
    print("=" * 60)
    print("Method  :", heur_details["method"])
    print("Mapping :", heur_map)
    return heur_map, heur_details


# ------------------------------------------------------------------ QC
def _classify_invalid_timestamps(
    items: list[dict[str, Any]],
    processed_duration: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Classify invalid timestamps into (severe_errors, minor_overshoots).

    Severity policy:
    - SEVERE (causes QC FAIL): missing start/end, negative start, inverted or
      zero-length range (start >= end), or end exceeding processed_duration by
      more than SEVERE_TIMESTAMP_OVERSHOOT_THRESHOLD_SEC (1.0s).
    - MINOR (causes QC WARNING): end exceeding processed_duration by between
      MINOR_TIMESTAMP_OVERSHOOT_TOLERANCE_SEC (0.1s) and 1.0s (typical frame rounding).
    """
    severe: list[dict[str, Any]] = []
    minor: list[dict[str, Any]] = []
    for item in items:
        s = item.get("start")
        e = item.get("end")
        if s is None or e is None or s < 0 or s >= e:
            severe.append(item)
        elif processed_duration > 0 and e > processed_duration + SEVERE_TIMESTAMP_OVERSHOOT_THRESHOLD_SEC:
            severe.append(item)
        elif processed_duration > 0 and e > processed_duration + MINOR_TIMESTAMP_OVERSHOOT_TOLERANCE_SEC:
            minor.append(item)
    return severe, minor


def _verify_output_files(expected_output_files: Optional[dict[str, Path | str]]) -> list[str]:
    missing: list[str] = []
    if expected_output_files:
        for label, path_val in expected_output_files.items():
            p = Path(path_val)
            if not p.exists() or p.stat().st_size == 0:
                missing.append(f"{label}: {p}")
    return missing


def run_qc(
    call_id: str,
    raw_duration: float,
    processed_duration: float,
    speakers: list[str],
    conversation: list[dict[str, Any]],
    *,
    processed_sample_rate: Optional[int] = None,
    processed_channels: Optional[int] = None,
    raw_audio_info: Optional[dict[str, Any]] = None,
    diarization_turns: Optional[list[dict[str, Any]]] = None,
    expected_output_files: Optional[dict[str, Path | str]] = None,
    expected_speakers: int = EXPECTED_SPEAKERS,
    expected_min_speakers: Optional[int] = None,
    expected_max_speakers: Optional[int] = None,
    vad_result: Optional[dict[str, Any]] = None,
    vad_speech_duration: Optional[float] = None,
    max_ambiguous_overlap_ratio: float = DEFAULT_MAX_AMBIGUOUS_OVERLAP_RATIO,
) -> dict[str, Any]:
    diff = abs(raw_duration - processed_duration)
    exp_min = expected_speakers if expected_min_speakers is None else expected_min_speakers
    exp_max = expected_speakers if expected_max_speakers is None else expected_max_speakers

    empty = [t for t in conversation if not str(t.get("text", "")).strip()]
    severe_conv_ts, minor_conv_ts = _classify_invalid_timestamps(conversation, processed_duration)
    if diarization_turns is not None:
        severe_diar_ts, minor_diar_ts = _classify_invalid_timestamps(diarization_turns, processed_duration)
    else:
        severe_diar_ts, minor_diar_ts = [], []

    invalid_conv_count = len(severe_conv_ts) + len(minor_conv_ts)
    invalid_diar_count = len(severe_diar_ts) + len(minor_diar_ts)
    total_severe_ts = len(severe_conv_ts) + len(severe_diar_ts)
    total_minor_ts = len(minor_conv_ts) + len(minor_diar_ts)

    unassigned = [
        t
        for t in conversation
        if t.get("speaker") in (None, "", UNASSIGNED_SPEAKER)
        or t.get("speaker_assigned") is False
    ]
    multi_overlap = [
        t
        for t in conversation
        if len(t.get("overlapping_speakers", [])) > 1
    ]
    ambiguous_overlap = [
        t
        for t in conversation
        if t.get("is_ambiguous") is True
        or t.get("assignment_status") == "assigned_ambiguous_multi_speaker"
    ]

    n_turns = len(conversation)
    multi_overlap_ratio = round(len(multi_overlap) / n_turns, 4) if n_turns > 0 else 0.0
    ambiguous_overlap_ratio = round(len(ambiguous_overlap) / n_turns, 4) if n_turns > 0 else 0.0

    sample_rate_valid = (
        None if processed_sample_rate is None else (int(processed_sample_rate) == TARGET_SAMPLE_RATE)
    )
    is_mono = (
        None if processed_channels is None else (int(processed_channels) == TARGET_CHANNELS)
    )

    missing_output_files = _verify_output_files(expected_output_files)

    full_text = " ".join(str(t.get("text", "")) for t in conversation).strip()
    words = len(full_text.split()) if full_text else 0
    transcription_segment_duration = sum(
        max(0.0, float(t["end"]) - float(t["start"]))
        for t in conversation
        if t.get("start") is not None and t.get("end") is not None
    )

    resolved_vad_duration = vad_speech_duration
    vad_speech_ratio = None
    if vad_result is not None:
        if resolved_vad_duration is None and vad_result.get("total_speech_duration") is not None:
            resolved_vad_duration = float(vad_result["total_speech_duration"])
        if vad_result.get("speech_ratio") is not None:
            vad_speech_ratio = float(vad_result["speech_ratio"])

    if total_severe_ts > 0:
        timestamps_status = "FAIL"
    elif total_minor_ts > 0:
        timestamps_status = "WARNING"
    else:
        timestamps_status = "PASS"

    # Multi-speaker overlap QC policy:
    #   PASS    -> 0 multi-speaker overlap turns and 0 ambiguous assignments
    #   FAIL    -> ambiguous multi-speaker turns >= max_ambiguous_overlap_ratio (default 50%)
    #   WARNING -> any multi-speaker overlap or ambiguous assignment below fail threshold
    if n_turns > 0 and ambiguous_overlap_ratio >= max_ambiguous_overlap_ratio and len(ambiguous_overlap) > 0:
        multi_overlap_status = "FAIL"
    elif multi_overlap or ambiguous_overlap:
        multi_overlap_status = "WARNING"
    else:
        multi_overlap_status = "PASS"

    checks = {
        "duration": (
            "FAIL"
            if processed_duration <= 0
            else ("PASS" if diff < 0.1 else "WARNING")
        ),
        "sample_rate_16khz": (
            "SKIPPED"
            if sample_rate_valid is None
            else ("PASS" if sample_rate_valid else "FAIL")
        ),
        "mono_audio": (
            "SKIPPED"
            if is_mono is None
            else ("PASS" if is_mono else "FAIL")
        ),
        "speakers": (
            "PASS"
            if exp_min <= len(speakers) <= exp_max
            else ("FAIL" if len(speakers) == 0 and words > 0 else "WARNING")
        ),
        "empty_segments": "PASS" if not empty else "WARNING",
        "timestamps": timestamps_status,
        "unassigned_speakers": "PASS" if not unassigned else "WARNING",
        "multi_speaker_overlap": multi_overlap_status,
        "transcript": "PASS" if words > 0 else "FAIL",
        "output_files": "PASS" if not missing_output_files else "FAIL",
    }

    if "FAIL" in checks.values():
        overall_status = "FAIL"
    elif "WARNING" in checks.values():
        overall_status = "WARNING"
    else:
        overall_status = "PASS"

    expected_spk_label = f"{exp_min}" if exp_min == exp_max else f"{exp_min}-{exp_max}"
    print("=" * 50)
    print("QC")
    print("=" * 50)
    print("Duration           :", checks["duration"], f"(raw {raw_duration:.3f}s -> proc {processed_duration:.3f}s, diff {diff:.3f}s)")
    if processed_sample_rate is not None:
        print("Sample rate 16 kHz :", checks["sample_rate_16khz"], f"({processed_sample_rate} Hz)")
    if processed_channels is not None:
        print("Mono audio         :", checks["mono_audio"], f"({processed_channels} ch)")
    print("Speakers           :", checks["speakers"], f"({len(speakers)} detected, expected {expected_spk_label})")
    print("Empty segments     :", checks["empty_segments"], f"({len(empty)})")
    print("Timestamps         :", checks["timestamps"], f"(severe {total_severe_ts}, minor overshoot {total_minor_ts})")
    print("Unassigned speakers:", checks["unassigned_speakers"], f"({len(unassigned)})")
    print(
        "Multi-spk overlap  :",
        checks["multi_speaker_overlap"],
        f"(overlap {len(multi_overlap)}, ambiguous {len(ambiguous_overlap)}/{n_turns})",
    )
    print("Transcript         :", checks["transcript"], f"({words} words)")
    if expected_output_files:
        print("Output files       :", checks["output_files"], f"({len(missing_output_files)} missing)")
    print("Overall QC status  :", overall_status)

    return {
        "call_id": call_id,
        "raw_duration": round(float(raw_duration), 3),
        "processed_duration": round(float(processed_duration), 3),
        "duration_difference": round(float(diff), 3),
        "processed_sample_rate": processed_sample_rate,
        "processed_channels": processed_channels,
        "sample_rate_valid": sample_rate_valid,
        "is_mono": is_mono,
        "expected_speakers": expected_speakers,
        "expected_min_speakers": exp_min,
        "expected_max_speakers": exp_max,
        "num_speakers": len(speakers),
        "speakers": speakers,
        "conversation_turns": n_turns,
        "empty_segments": len(empty),
        "invalid_timestamps": invalid_conv_count,
        "severe_invalid_timestamps": len(severe_conv_ts),
        "minor_timestamp_overshoots": len(minor_conv_ts),
        "invalid_diarization_timestamps": invalid_diar_count,
        "severe_invalid_diarization_timestamps": len(severe_diar_ts),
        "minor_diarization_overshoots": len(minor_diar_ts),
        "unassigned_speaker_segments": len(unassigned),
        "unassigned_speaker_turn_ids": [t.get("turn_id") for t in unassigned],
        "multi_speaker_overlap_segments": len(multi_overlap),
        "multi_speaker_overlap_turn_ids": [t.get("turn_id") for t in multi_overlap],
        "multi_speaker_overlap_ratio": multi_overlap_ratio,
        "ambiguous_speaker_segments": len(ambiguous_overlap),
        "ambiguous_speaker_turn_ids": [t.get("turn_id") for t in ambiguous_overlap],
        "ambiguous_speaker_ratio": ambiguous_overlap_ratio,
        "word_count": words,
        "character_count": len(full_text),
        "transcription_segment_duration": round(float(transcription_segment_duration), 3),
        "vad_speech_duration": (
            round(float(resolved_vad_duration), 3) if resolved_vad_duration is not None else None
        ),
        "vad_speech_ratio": round(float(vad_speech_ratio), 4) if vad_speech_ratio is not None else None,
        # Backward-compatible alias for transcription_segment_duration
        "speech_duration": round(float(transcription_segment_duration), 3),
        "checked_output_files": sorted(expected_output_files.keys()) if expected_output_files else [],
        "missing_output_files": missing_output_files,
        "raw_audio_info": raw_audio_info,
        "checks": checks,
        "status": overall_status,
    }


# ------------------------------------------------------------------ customer profile
class CustomerProfile(BaseModel):
    budget: Optional[float] = Field(default=None, description="Maximum customer budget in Egyptian pounds.")
    location: Optional[str] = Field(default=None, description="Preferred property location.")
    property_type: Optional[str] = Field(default=None, description="Requested property type.")
    rooms: Optional[int] = Field(default=None, description="Number of bedrooms or rooms.")
    area_sqm: Optional[float] = Field(default=None, description="Requested area in square meters.")
    finishing: Optional[str] = Field(default=None, description="Preferred finishing status.")
    payment_plan: Optional[str] = Field(default=None, description="Preferred payment plan.")
    purpose: Optional[str] = Field(default=None, description="Purpose of buying.")


SYSTEM_PROMPT = """
You are an AI assistant extracting customer requirements from Egyptian Arabic real-estate phone calls.

Extract ONLY information explicitly stated by the customer or clearly confirmed by the customer.

Strict rules:
1. Never guess, assume, or invent a requirement.
2. Broker questions, suggestions, and statements are NOT customer requirements unless the customer explicitly confirms them.
3. If the customer hesitates, contradicts themselves, or gives an ambiguous answer, return null for the affected field.
4. Correct ASR errors only when the intended words are unambiguous from the conversation.
5. Do not infer finishing status, payment plan, property type, or purpose from context alone.
6. Return null for every field that is not explicitly mentioned or reliably confirmed.
7. Convert amounts expressed in millions to Egyptian pounds.
8. Preserve the specific location name as spoken. Do not shorten or replace it with a broader location.
9. A property type such as apartment must be extracted only when the customer explicitly mentions it.
10. Output only the structured fields defined by the schema.

Field-specific rules:
- budget: Extract the customer's stated budget.
- location: Extract the exact requested location.
- property_type: Extract the property type explicitly requested by the customer.
- rooms: Extract the explicitly requested number of rooms.
- area_sqm: Extract only an explicitly stated area in square meters.
- finishing: Extract only an explicitly stated finishing preference.
- payment_plan: Extract only an explicitly stated payment method or installment plan.
- purpose: Extract rent or purchase only when the customer's intention is clear and unambiguous. If the customer hesitates or contradicts themselves, return null.
"""


def extract_customer_profile(conversation):
    api_key = get_gemini_api_key()
    if not api_key:
        validate_environment(require_hf=False, require_gemini=True)

    from langchain_core.prompts import ChatPromptTemplate
    from langchain_google_genai import ChatGoogleGenerativeAI

    llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, google_api_key=api_key, request_timeout=20, retries=0)
    structured_llm = llm.with_structured_output(CustomerProfile)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "Customer conversation:\n\n{conversation}"),
    ])

    text = "\n".join(f"{t['speaker']}: {t['text']}" for t in conversation)
    safe_print(text)
    return (prompt | structured_llm).invoke({"conversation": text})


# ------------------------------------------------------------------ single & batch orchestration
def get_call_exit_code(call_result: dict[str, Any]) -> int:
    """Return process exit code for a single-call result.

    Policy:
    - EXIT_OK (0): QC PASS or WARNING
    - EXIT_QC_FAILED (2): Processing finished, but qc_status == "FAIL"
    - EXIT_PROCESSING_ERROR (1): Runtime/processing error
    """
    status = call_result.get("status")
    qc_status = call_result.get("qc_status")
    if status == "ERROR":
        return EXIT_PROCESSING_ERROR
    if status == "QC_FAILED" or qc_status == "FAIL":
        return EXIT_QC_FAILED
    return EXIT_OK


def get_batch_exit_code(batch_summary: dict[str, Any]) -> int:
    """Return process exit code for a batch summary.

    Policy:
    - EXIT_PROCESSING_ERROR (1): One or more calls raised a processing exception
    - EXIT_QC_FAILED (2): No processing exceptions, but one or more calls failed QC
    - EXIT_OK (0): All calls completed with QC PASS or WARNING
    """
    if batch_summary.get("processing_errors", 0) > 0:
        return EXIT_PROCESSING_ERROR
    if batch_summary.get("qc_fail", 0) > 0:
        return EXIT_QC_FAILED
    return EXIT_OK


def process_call(
    call_id: str,
    ext: Optional[str] = None,
    *,
    plot: bool = False,
    skip_profile: bool = False,
    infer_roles: bool = True,
    use_llm_for_roles: bool = True,
    device: Optional[str] = None,
    ffmpeg_path: Optional[str] = None,
    models: Optional[PipelineModels] = None,
    raw_audio_dir: Path = RAW_AUDIO_DIR,
    output_dirs: Optional[dict[str, Path]] = None,
    min_speakers: int = EXPECTED_SPEAKERS,
    max_speakers: int = EXPECTED_SPEAKERS,
    min_overlap_sec: float = DEFAULT_MIN_OVERLAP_SEC,
    min_overlap_ratio: float = DEFAULT_MIN_OVERLAP_RATIO,
    min_speaker_dominance: float = DEFAULT_MIN_SPEAKER_DOMINANCE,
    max_ambiguous_overlap_ratio: float = DEFAULT_MAX_AMBIGUOUS_OVERLAP_RATIO,
) -> dict[str, Any]:
    """Process one call end-to-end, save all artifacts, validate outputs, and propagate QC status."""
    dirs = ensure_output_dirs(output_dirs)
    clean_call_id, audio_path = resolve_audio_path(call_id, ext=ext, raw_audio_dir=raw_audio_dir)

    validate_environment(
        require_hf=(models is None or models.diarization_pipeline is None),
        require_gemini=(not skip_profile),
    )

    selected_device = models.device if models is not None else (device or get_device())
    ff = ffmpeg_path or setup_ffmpeg()

    print(f"[{clean_call_id}] Stage 1/7: Audio inspection & standardization", flush=True)
    raw_duration, raw_audio_info = inspect_audio(
        audio_path, clean_call_id, plot=plot, ffmpeg_path=ff, return_details=True
    )

    wav_path = dirs["processed_audio"] / f"{clean_call_id}_standardized.wav"
    processed_duration, wav_info = standardize_audio(
        ff, audio_path, wav_path, return_details=True
    )

    print(f"[{clean_call_id}] Stage 2/7: Voice Activity Detection (Silero VAD)", flush=True)
    vad_path = dirs["vad"] / f"{clean_call_id}_vad.json"
    vad_result = run_vad(
        wav_path,
        clean_call_id,
        selected_device,
        vad_bundle=models.vad_bundle if models else None,
        speech_segments_dir=dirs["speech_segments"],
    )
    save_json(vad_path, vad_result)

    print(f"[{clean_call_id}] Stage 3/7: Speech-to-Text (Whisper {WHISPER_MODEL})", flush=True)
    transcript_path = dirs["transcripts"] / f"{clean_call_id}_transcript.json"
    info, segments, full_text = transcribe(
        wav_path,
        selected_device,
        whisper_model=models.whisper_model if models else None,
    )
    save_json(transcript_path, {
        "call_id": clean_call_id,
        "source_audio_file": audio_path.name,
        "audio_file": wav_path.name,
        "whisper_model": WHISPER_MODEL,
        "language": info.language,
        "language_probability": round(float(info.language_probability), 4),
        "segments": segments,
        "full_transcript": full_text,
    })

    print(f"[{clean_call_id}] Stage 4/7: Speaker Diarization ({DIARIZATION_MODEL})", flush=True)
    diarization_path = dirs["diarization"] / f"{clean_call_id}_diarization.json"
    turns, speakers = run_diarization(
        wav_path,
        selected_device,
        num_speakers=max_speakers,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        diarization_pipeline=models.diarization_pipeline if models else None,
    )

    print(f"[{clean_call_id}] Stage 5/7: Transcript-Diarization Alignment", flush=True)
    conversation_path = dirs["conversations"] / f"{clean_call_id}_conversation.json"
    conversation = build_conversation(
        segments,
        turns,
        min_overlap_sec=min_overlap_sec,
        min_overlap_ratio=min_overlap_ratio,
        min_speaker_dominance=min_speaker_dominance,
    )
    conv_speakers = sorted({
        t["speaker"]
        for t in conversation
        if t.get("speaker") and t["speaker"] != UNASSIGNED_SPEAKER
    })
    all_speakers = sorted(set(speakers) | set(conv_speakers))

    print(f"[{clean_call_id}] Stage 6/7: Speaker Role Assignment (infer_roles={infer_roles})", flush=True)
    if infer_roles:
        speaker_role_mapping, role_inference_details = infer_speaker_roles(
            conversation,
            speakers=all_speakers,
            use_llm=use_llm_for_roles,
        )
    else:
        speaker_role_mapping = {spk: "unassigned" for spk in all_speakers}
        role_inference_details = {
            "method": "skipped",
            "confidence": "none",
            "reasoning": "Role inference skipped by caller configuration.",
        }

    for turn in conversation:
        spk = turn.get("speaker")
        turn["role"] = (
            speaker_role_mapping.get(spk, "unassigned")
            if spk and spk != UNASSIGNED_SPEAKER
            else "unassigned"
        )

    save_json(diarization_path, {
        "call_id": clean_call_id,
        "source_audio_file": audio_path.name,
        "audio_file": wav_path.name,
        "diarization_model": DIARIZATION_MODEL,
        "min_speakers": min_speakers,
        "max_speakers": max_speakers,
        "num_speakers": len(speakers),
        "speakers": speakers,
        "speaker_role_mapping": {spk: speaker_role_mapping.get(spk, "unassigned") for spk in speakers},
        "speaker_role_inference": role_inference_details,
        "turns": turns,
    })

    unassigned_count = sum(1 for t in conversation if not t.get("speaker_assigned", False))
    ambiguous_count = sum(1 for t in conversation if t.get("is_ambiguous", False))
    save_json(conversation_path, {
        "call_id": clean_call_id,
        "source_audio_file": audio_path.name,
        "audio_file": wav_path.name,
        "language": info.language,
        "duration": round(float(processed_duration), 3),
        "num_speakers": len(conv_speakers),
        "speakers": conv_speakers,
        "speaker_role_mapping": {spk: speaker_role_mapping.get(spk, "unassigned") for spk in conv_speakers},
        "speaker_role_inference": role_inference_details,
        "unassigned_speaker_segments": unassigned_count,
        "ambiguous_speaker_segments": ambiguous_count,
        "conversation": conversation,
    })

    expected_outputs: dict[str, Path] = {
        "standardized_wav": wav_path,
        "vad_json": vad_path,
        "transcript_json": transcript_path,
        "diarization_json": diarization_path,
        "conversation_json": conversation_path,
    }

    if skip_profile:
        print("Skipping customer profile extraction.")
    else:
        profile = extract_customer_profile(conversation)
        profile_path = dirs["profiles"] / f"{clean_call_id}_customer_profile.json"
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        profile_path.write_text(
            profile.model_dump_json(indent=2, exclude_none=False), encoding="utf-8"
        )
        print("\nCustomer Profile:")
        print(profile.model_dump_json(indent=2))
        print("Saved:", profile_path)
        expected_outputs["customer_profile_json"] = profile_path

    print(f"[{clean_call_id}] Stage 7/7: Quality Control & Output Validation", flush=True)
    qc_path = dirs["qc"] / f"{clean_call_id}_qc.json"
    qc = run_qc(
        clean_call_id,
        raw_duration,
        processed_duration,
        speakers,
        conversation,
        processed_sample_rate=wav_info["sample_rate"],
        processed_channels=wav_info["channels"],
        raw_audio_info=raw_audio_info,
        diarization_turns=turns,
        expected_output_files=expected_outputs,
        expected_speakers=max_speakers,
        expected_min_speakers=min_speakers,
        expected_max_speakers=max_speakers,
        vad_result=vad_result,
        max_ambiguous_overlap_ratio=max_ambiguous_overlap_ratio,
    )
    # Include qc_json in expected artifacts list, save it, then verify ALL final outputs including qc_json
    expected_outputs["qc_json"] = qc_path
    qc["checked_output_files"] = sorted(expected_outputs.keys())
    save_json(qc_path, qc)

    final_missing = _verify_output_files(expected_outputs)
    if final_missing != qc["missing_output_files"]:
        qc["missing_output_files"] = final_missing
        qc["checks"]["output_files"] = "PASS" if not final_missing else "FAIL"
        if "FAIL" in qc["checks"].values():
            qc["status"] = "FAIL"
        elif "WARNING" in qc["checks"].values():
            qc["status"] = "WARNING"
        else:
            qc["status"] = "PASS"
        if qc_path.exists():
            save_json(qc_path, qc)

    if qc["status"] == "FAIL":
        call_status = "QC_FAILED"
    elif qc["status"] == "WARNING":
        call_status = "SUCCESS_WITH_WARNINGS"
    else:
        call_status = "SUCCESS"

    return {
        "call_id": clean_call_id,
        "source_audio": str(audio_path),
        "status": call_status,
        "qc_status": qc["status"],
        "qc": qc,
        "outputs": {k: str(v) for k, v in expected_outputs.items()},
    }


def discover_calls(
    raw_audio_dir: Path = RAW_AUDIO_DIR,
    ext: str = "mp4",
    call_ids: Optional[list[str]] = None,
) -> list[str]:
    """Return a naturally sorted list of call IDs to process."""
    clean_ext = ext.lstrip(".").lower()
    if call_ids:
        return [normalize_call_id_and_ext(cid, ext=clean_ext, default_ext=clean_ext)[0] for cid in call_ids]
    raw_dir = Path(raw_audio_dir)
    if not raw_dir.exists():
        return []
    matches = [
        p.stem
        for p in raw_dir.iterdir()
        if p.is_file() and p.suffix.lstrip(".").lower() == clean_ext
    ]
    return sorted(matches, key=_natural_sort_key)


def process_batch(
    call_ids: Optional[list[str]] = None,
    ext: str = "mp4",
    *,
    skip_profile: bool = True,
    infer_roles: bool = True,
    use_llm_for_roles: bool = True,
    plot: bool = False,
    raw_audio_dir: Path = RAW_AUDIO_DIR,
    output_dirs: Optional[dict[str, Path]] = None,
    models: Optional[PipelineModels] = None,
    ffmpeg_path: Optional[str] = None,
    min_speakers: int = EXPECTED_SPEAKERS,
    max_speakers: int = EXPECTED_SPEAKERS,
    min_overlap_sec: float = DEFAULT_MIN_OVERLAP_SEC,
    min_overlap_ratio: float = DEFAULT_MIN_OVERLAP_RATIO,
    min_speaker_dominance: float = DEFAULT_MIN_SPEAKER_DOMINANCE,
    max_ambiguous_overlap_ratio: float = DEFAULT_MAX_AMBIGUOUS_OVERLAP_RATIO,
) -> dict[str, Any]:
    """Process multiple calls independently, reusing loaded models and persisting progress incrementally."""
    dirs = ensure_output_dirs(output_dirs)
    target_calls = discover_calls(raw_audio_dir=raw_audio_dir, ext=ext, call_ids=call_ids)
    if not target_calls:
        raise FileNotFoundError(
            f"No '.{ext.lstrip('.')}' recordings found in '{raw_audio_dir}'."
        )

    validate_environment(
        require_hf=(models is None or models.diarization_pipeline is None),
        require_gemini=(not skip_profile),
    )

    shared_ffmpeg = ffmpeg_path or setup_ffmpeg()
    shared_models = models if models is not None else load_pipeline_models(get_device())

    run_started = datetime.now(timezone.utc)
    run_id = run_started.strftime("%Y%m%d_%H%M%S")
    error_log_path = dirs["batch_logs"] / f"batch_errors_{run_id}.log"
    summary_latest = dirs["batch_logs"] / "batch_summary.json"
    summary_run = dirs["batch_logs"] / f"batch_summary_{run_id}.json"

    results = []
    qc_pass = 0
    qc_warning = 0
    qc_fail = 0
    processing_errors = 0
    error_entries: list[str] = []

    def _build_and_persist_summary(batch_status: str) -> dict[str, Any]:
        succeeded_now = qc_pass + qc_warning
        failed_now = processing_errors + qc_fail
        if error_entries:
            error_log_path.parent.mkdir(parents=True, exist_ok=True)
            error_log_path.write_text("\n".join(error_entries), encoding="utf-8")
        summary_payload = {
            "run_id": run_id,
            "batch_status": batch_status,
            "started_at_utc": run_started.isoformat(),
            "completed_at_utc": (
                datetime.now(timezone.utc).isoformat() if batch_status == "COMPLETED" else None
            ),
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "raw_audio_dir": str(raw_audio_dir),
            "extension": ext.lstrip(".").lower(),
            "whisper_model": WHISPER_MODEL,
            "diarization_model": DIARIZATION_MODEL,
            "min_speakers": min_speakers,
            "max_speakers": max_speakers,
            "skip_profile": skip_profile,
            "infer_roles": infer_roles,
            "total_calls": len(target_calls),
            "completed_calls": len(results),
            "succeeded": succeeded_now,
            "failed": failed_now,
            "processing_errors": processing_errors,
            "qc_pass": qc_pass,
            "qc_warning": qc_warning,
            "qc_fail": qc_fail,
            "qc_summary": {
                "PASS": qc_pass,
                "WARNING": qc_warning,
                "FAIL": qc_fail,
                "ERROR": processing_errors,
            },
            "error_log": str(error_log_path) if error_entries else None,
            "calls": results,
        }
        save_json(summary_latest, summary_payload)
        save_json(summary_run, summary_payload)
        return summary_payload

    print("=" * 60, flush=True)
    print(
        f"BATCH RUN ({len(target_calls)} calls) | ext=.{ext.lstrip('.')} "
        f"| skip_profile={skip_profile} | infer_roles={infer_roles}",
        flush=True,
    )
    print("=" * 60, flush=True)

    for idx, cid in enumerate(target_calls, start=1):
        print(f"\n[{idx}/{len(target_calls)}] Processing call: {cid}", flush=True)
        try:
            call_result = process_call(
                cid,
                ext=ext,
                plot=plot,
                skip_profile=skip_profile,
                infer_roles=infer_roles,
                use_llm_for_roles=use_llm_for_roles,
                ffmpeg_path=shared_ffmpeg,
                models=shared_models,
                raw_audio_dir=raw_audio_dir,
                output_dirs=dirs,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
                min_overlap_sec=min_overlap_sec,
                min_overlap_ratio=min_overlap_ratio,
                min_speaker_dominance=min_speaker_dominance,
                max_ambiguous_overlap_ratio=max_ambiguous_overlap_ratio,
            )
            results.append(call_result)
            qc_st = call_result.get("qc_status")
            if qc_st == "FAIL" or call_result.get("status") == "QC_FAILED":
                qc_fail += 1
                failed_checks = {
                    k: v for k, v in call_result.get("qc", {}).get("checks", {}).items() if v == "FAIL"
                }
                error_entries.append(
                    f"=== CALL: {cid} (QC_FAILED) ===\nFailed QC checks: {failed_checks}\n"
                )
            elif qc_st == "WARNING":
                qc_warning += 1
            else:
                qc_pass += 1
        except Exception as exc:
            processing_errors += 1
            tb = traceback.format_exc()
            print(f"[ERROR] Call '{cid}' raised processing error: {exc}", file=sys.stderr, flush=True)
            error_entries.append(
                f"=== CALL: {cid} (PROCESSING_ERROR) ===\nError: {type(exc).__name__}: {exc}\n{tb}\n"
            )
            results.append({
                "call_id": cid,
                "status": "ERROR",
                "qc_status": None,
                "error_type": type(exc).__name__,
                "error": str(exc),
            })

        # Persist progress after every call so mid-batch interruptions retain full diagnostics
        _build_and_persist_summary("IN_PROGRESS" if idx < len(target_calls) else "COMPLETED")

    summary = _build_and_persist_summary("COMPLETED")
    if error_entries:
        print("Saved error log:", error_log_path, flush=True)

    succeeded = qc_pass + qc_warning
    print("\n" + "=" * 60, flush=True)
    print(
        f"BATCH COMPLETE: {succeeded}/{len(target_calls)} passed QC "
        f"(PASS={qc_pass}, WARNING={qc_warning}, QC_FAIL={qc_fail}, PROCESSING_ERRORS={processing_errors})",
        flush=True,
    )
    print("=" * 60, flush=True)
    return summary


# ------------------------------------------------------------------ main
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "SmartSemsar Speech Pipeline (single call or batch processing). "
            "Exit codes: 0=PASS/WARNING, 1=processing error, 2=QC FAIL."
        )
    )
    parser.add_argument(
        "--call-id",
        default="call_00",
        help="Call ID or recording filename (e.g., smartsemsar_rec1 or smartsemsar_rec1.mp4).",
    )
    parser.add_argument(
        "--ext",
        default=None,
        help="Audio file extension without leading dot (e.g., mp4, ogg). Defaults to ogg when --call-id has no extension.",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Process all recordings with --ext in raw_calls (or those listed in --calls).",
    )
    parser.add_argument(
        "--calls",
        nargs="+",
        default=None,
        help="Optional explicit list of call IDs to process with --batch.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=RAW_AUDIO_DIR,
        help=f"Directory containing raw recordings. Default: {RAW_AUDIO_DIR}",
    )
    parser.add_argument(
        "--min-speakers",
        type=int,
        default=EXPECTED_SPEAKERS,
        help=f"Minimum expected speakers for diarization and QC (default: {EXPECTED_SPEAKERS}).",
    )
    parser.add_argument(
        "--max-speakers",
        type=int,
        default=EXPECTED_SPEAKERS,
        help=f"Maximum expected speakers for diarization and QC (default: {EXPECTED_SPEAKERS}).",
    )
    parser.add_argument(
        "--min-overlap-sec",
        type=float,
        default=DEFAULT_MIN_OVERLAP_SEC,
        help=f"Minimum speaker overlap in seconds to assign a segment (default: {DEFAULT_MIN_OVERLAP_SEC}).",
    )
    parser.add_argument(
        "--min-overlap-ratio",
        type=float,
        default=DEFAULT_MIN_OVERLAP_RATIO,
        help=f"Minimum fraction of segment duration covered by assigned speaker (default: {DEFAULT_MIN_OVERLAP_RATIO}).",
    )
    parser.add_argument(
        "--min-speaker-dominance",
        type=float,
        default=DEFAULT_MIN_SPEAKER_DOMINANCE,
        help=f"Minimum top-speaker share of overlapping speech before flagging ambiguous (default: {DEFAULT_MIN_SPEAKER_DOMINANCE}).",
    )
    parser.add_argument("--plot", action="store_true", help="Plot raw audio waveform.")
    parser.add_argument(
        "--skip-profile",
        action="store_true",
        help="Skip Gemini customer profile extraction.",
    )
    parser.add_argument(
        "--skip-roles",
        action="store_true",
        help="Skip broker/client speaker role assignment (leave roles unassigned).",
    )
    parser.add_argument(
        "--heuristic-roles",
        action="store_true",
        help="Use deterministic offline Arabic heuristic for role assignment instead of LLM.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    configure_utf8_console()

    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        if args.batch:
            summary = process_batch(
                call_ids=args.calls,
                ext=args.ext or "ogg",
                skip_profile=args.skip_profile,
                infer_roles=not args.skip_roles,
                use_llm_for_roles=not args.heuristic_roles,
                plot=args.plot,
                raw_audio_dir=args.input_dir,
                min_speakers=args.min_speakers,
                max_speakers=args.max_speakers,
                min_overlap_sec=args.min_overlap_sec,
                min_overlap_ratio=args.min_overlap_ratio,
                min_speaker_dominance=args.min_speaker_dominance,
            )
            return get_batch_exit_code(summary)

        result = process_call(
            args.call_id,
            ext=args.ext,
            plot=args.plot,
            skip_profile=args.skip_profile,
            infer_roles=not args.skip_roles,
            use_llm_for_roles=not args.heuristic_roles,
            raw_audio_dir=args.input_dir,
            min_speakers=args.min_speakers,
            max_speakers=args.max_speakers,
            min_overlap_sec=args.min_overlap_sec,
            min_overlap_ratio=args.min_overlap_ratio,
            min_speaker_dominance=args.min_speaker_dominance,
        )
        return get_call_exit_code(result)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"\n[PIPELINE ERROR] {exc}", file=sys.stderr)
        return EXIT_PROCESSING_ERROR


if __name__ == "__main__":
    sys.exit(main())
