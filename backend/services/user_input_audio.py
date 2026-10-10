"""Call Intelligence Pipeline (single file).

Usage:
    python pipeline.py --call-id call_00
    python pipeline.py --call-id call_00 --plot --skip-profile
"""
import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import soundfile as sf
import torch
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from backend.schema.state import CustomerRequirements

# ------------------------------------------------------------------ config
PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

DATA_DIR = PROJECT_ROOT / "data"
RAW_AUDIO_DIR = DATA_DIR / "raw_calls"
PROCESSED_AUDIO_DIR = DATA_DIR / "processed_audio"
SPEECH_SEGMENTS_DIR = PROCESSED_AUDIO_DIR / "speech_segments"
TRANSCRIPT_DIR = DATA_DIR / "transcripts"
DIARIZATION_DIR = TRANSCRIPT_DIR / "diarization"

HF_TOKEN = os.environ.get("HF_TOKEN")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")

WHISPER_MODEL = "large-v3"
DIARIZATION_MODEL = "pyannote/speaker-diarization-community-1"
EXPECTED_SPEAKERS = 2

for d in (DATA_DIR, RAW_AUDIO_DIR, PROCESSED_AUDIO_DIR,
          SPEECH_SEGMENTS_DIR, TRANSCRIPT_DIR, DIARIZATION_DIR):
    d.mkdir(parents=True, exist_ok=True)


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print("Saved:", path)


# ------------------------------------------------------------------ runtime
def get_device():
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
def setup_ffmpeg():
    import imageio_ffmpeg
    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    os.environ["PATH"] = os.path.dirname(ffmpeg_path) + os.pathsep + os.environ["PATH"]
    result = subprocess.run([ffmpeg_path, "-version"], capture_output=True, text=True)
    print("FFmpeg:", result.stdout.splitlines()[0])
    return ffmpeg_path


def inspect_audio(audio_path, call_id, plot=False):
    print("=" * 50)
    print("CALL INPUT")
    print("=" * 50)
    print("Call ID :", call_id)
    print("File    :", audio_path)
    print("Size MB :", round(audio_path.stat().st_size / 1024**2, 3))

    audio, sr = librosa.load(audio_path, sr=None, mono=True)
    duration = len(audio) / sr

    print("\n" + "=" * 50)
    print("RAW AUDIO INFO")
    print("=" * 50)
    print("Format        :", audio_path.suffix)
    print("Sample rate   :", sr)
    print("Duration      :", round(duration, 2), "seconds")
    print("Samples       :", len(audio))
    print("Min / Max amp :", round(float(audio.min()), 4), "/", round(float(audio.max()), 4))

    rms = np.sqrt(np.mean(audio ** 2))
    peak = np.max(np.abs(audio))
    clipped = int(np.sum(np.abs(audio) >= 0.99))
    ratio = clipped / len(audio)

    print("\n" + "=" * 50)
    print("AUDIO QUALITY")
    print("=" * 50)
    print("RMS energy      :", f"{rms:.6f}")
    print("Peak amplitude  :", f"{peak:.6f}")
    print("Clipped samples :", clipped)
    print("Clipping ratio  :", f"{ratio:.6%}")
    if ratio == 0:
        print("Status: No clipping")
    elif ratio < 0.001:
        print("Status: Very small clipping")
    else:
        print("Status: Significant clipping")

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

    return duration


def standardize_audio(ffmpeg_path, src, dst):
    command = [ffmpeg_path, "-y", "-i", str(src),
               "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(dst)]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr)

    audio, sr = librosa.load(dst, sr=None, mono=True)
    duration = len(audio) / sr
    print("\n" + "=" * 50)
    print("STANDARDIZED AUDIO")
    print("=" * 50)
    print("File        :", dst.name)
    print("Sample rate :", sr)
    print("Duration    :", round(duration, 2))
    return duration


# ------------------------------------------------------------------ VAD
def run_vad(wav_path, call_id, device):
    vad_model, vad_utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad", model="silero_vad", trust_repo=True
    )
    vad_model = vad_model.to(device)
    get_speech_timestamps = vad_utils[0]

    vad_audio, sample_rate = sf.read(wav_path)
    if vad_audio.ndim > 1:
        vad_audio = vad_audio.mean(axis=1)
    tensor = torch.from_numpy(vad_audio.astype(np.float32)).to(device)

    timestamps = get_speech_timestamps(tensor, vad_model, sampling_rate=sample_rate)

    print("=" * 50)
    print("VAD RESULT")
    print("=" * 50)
    print("Speech segments:", len(timestamps))
    for i, seg in enumerate(timestamps, start=1):
        s, e = seg["start"] / sample_rate, seg["end"] / sample_rate
        print(f"{i:02d}. {s:.2f}s → {e:.2f}s ({e - s:.2f}s)")
        out = SPEECH_SEGMENTS_DIR / f"{call_id}_speech_{i:02d}.wav"
        sf.write(out, vad_audio[seg["start"]:seg["end"]], sample_rate)

    return {
        "call_id": call_id,
        "audio_file": wav_path.name,
        "sample_rate": sample_rate,
        "duration": round(len(vad_audio) / sample_rate, 3),
        "speech_segments": [
            {
                "id": i,
                "start": round(seg["start"] / sample_rate, 3),
                "end": round(seg["end"] / sample_rate, 3),
                "duration": round((seg["end"] - seg["start"]) / sample_rate, 3),
            }
            for i, seg in enumerate(timestamps, start=1)
        ],
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


def transcribe(wav_path, device):
    from faster_whisper import WhisperModel

    compute_type = "float16" if device == "cuda" else "int8"
    print("Whisper device:", device, "| compute type:", compute_type)

    model = WhisperModel(WHISPER_MODEL, device=device, compute_type=compute_type)
    segments, info = model.transcribe(
      str(wav_path), language="ar", task="transcribe",
      beam_size=5, vad_filter=False,
      condition_on_previous_text=False,
      initial_prompt="مكالمة عقارية بالعامية المصرية: شقة، فيلا، دوبلكس، أوض نوم، ميزانية، مليون، متر، التجمع، نيو كايرو، قريبة من المدارس، تشطيب، تقسيط، سكن، استثمار.",
  )
    segments = list(segments)

    print("=" * 60)
    print("WHISPER TRANSCRIPTION")
    print("=" * 60)
    print("Detected language   :", info.language)
    print("Language probability:", round(info.language_probability, 4))
    print("Number of segments  :", len(segments))
    for i, s in enumerate(segments, start=1):
        print(f"{i:02d}. [{s.start:.2f}s → {s.end:.2f}s] {s.text.strip()}")

    result = []
    for s in segments:
        original = s.text.strip()
        result.append({
            "start": round(s.start, 3),
            "end": round(s.end, 3),
            "text": original,
            "normalized_text": normalize_arabic_text(original),
        })
    full_text = " ".join(x["normalized_text"] for x in result).strip()
    return info, result, full_text


# ------------------------------------------------------------------ diarization
def run_diarization(wav_path, device, num_speakers=EXPECTED_SPEAKERS):
    from pyannote.audio import Pipeline

    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN is missing. Put it in the .env file.")

    pipeline = Pipeline.from_pretrained(DIARIZATION_MODEL, token=HF_TOKEN)
    pipeline.to(torch.device(device))

    output = pipeline(str(wav_path), num_speakers=num_speakers)
    exclusive = output.exclusive_speaker_diarization

    turns = [
        {"start": round(t.start, 3), "end": round(t.end, 3), "speaker": spk}
        for t, _, spk in exclusive.itertracks(yield_label=True)
    ]
    speakers = sorted({t["speaker"] for t in turns})

    print("=" * 60)
    print("SPEAKER DIARIZATION")
    print("=" * 60)
    print("Detected speakers:", speakers)
    for i, t in enumerate(turns, start=1):
        print(f"{i:02d}. [{t['start']:.2f}s → {t['end']:.2f}s] {t['speaker']}")
    return turns, speakers


# ------------------------------------------------------------------ alignment
def overlap_duration(a_start, a_end, b_start, b_end):
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def assign_speaker(start, end, turns):
    best_speaker, best_overlap = None, 0.0
    for t in turns:
        ov = overlap_duration(start, end, t["start"], t["end"])
        if ov > best_overlap:
            best_overlap, best_speaker = ov, t["speaker"]
    return best_speaker, best_overlap


def build_conversation(transcript_segments, speaker_turns):
    conversation = []
    for seg in transcript_segments:
        speaker, _ = assign_speaker(seg["start"], seg["end"], speaker_turns)
        conversation.append({
            "turn_id": len(conversation) + 1,
            "start": seg["start"],
            "end": seg["end"],
            "speaker": speaker,
            "text": seg["normalized_text"],
        })

    print("=" * 60)
    print("FINAL CONVERSATION")
    print("=" * 60)
    for t in conversation:
        print(f"[{t['start']:.2f}s → {t['end']:.2f}s] {t['speaker']}: {t['text']}")
    return conversation


# ------------------------------------------------------------------ QC
def run_qc(call_id, raw_duration, processed_duration, speakers, conversation):
    diff = abs(raw_duration - processed_duration)
    empty = [t for t in conversation if not t["text"].strip()]
    invalid = [t for t in conversation
               if t["start"] >= t["end"] or t["start"] < 0
               or t["end"] > processed_duration + 0.1]
    full_text = " ".join(t["text"] for t in conversation)
    words = len(full_text.split())
    speech_duration = sum(t["end"] - t["start"] for t in conversation)

    print("=" * 50)
    print("QC")
    print("=" * 50)
    print("Duration   :", "PASS" if diff < 0.1 else "WARNING", f"(diff {diff:.3f}s)")
    print("Speakers   :", "PASS" if len(speakers) == EXPECTED_SPEAKERS else "WARNING",
          f"({len(speakers)} detected)")
    print("Empty segs :", "PASS" if not empty else "WARNING", f"({len(empty)})")
    print("Timestamps :", "PASS" if not invalid else "WARNING", f"({len(invalid)} invalid)")
    print("Transcript :", "PASS" if words > 0 else "FAIL", f"({words} words)")

    return {
        "call_id": call_id,
        "raw_duration": round(raw_duration, 3),
        "processed_duration": round(processed_duration, 3),
        "duration_difference": round(diff, 3),
        "num_speakers": len(speakers),
        "speakers": speakers,
        "conversation_turns": len(conversation),
        "empty_segments": len(empty),
        "invalid_timestamps": len(invalid),
        "word_count": words,
        "character_count": len(full_text),
        "speech_duration": round(speech_duration, 3),
        "status": "PASS" if (not empty and not invalid and words > 0) else "WARNING",
    }


# ------------------------------------------------------------------ customer profile
# class CustomerProfile(BaseModel):
#     budget: Optional[float] = Field(default=None, description="Maximum customer budget in Egyptian pounds.")
#     location: Optional[str] = Field(default=None, description="Preferred property location.")
#     property_type: Optional[str] = Field(default=None, description="Requested property type.")
#     rooms: Optional[int] = Field(default=None, description="Number of bedrooms or rooms.")
#     area_sqm: Optional[float] = Field(default=None, description="Requested area in square meters.")
#     finishing: Optional[str] = Field(default=None, description="Preferred finishing status.")
#     payment_plan: Optional[str] = Field(default=None, description="Preferred payment plan.")
#     purpose: Optional[str] = Field(default=None, description="Purpose of buying.")


SYSTEM_PROMPT = """
You are an AI real-estate sales assistant.

Extract structured customer requirements
from an Arabic real-estate phone call.

The conversation may contain:

- Egyptian Arabic
- Arabic dialect
- ASR spelling mistakes
- Informal real-estate terminology

Rules:

1. Understand intended meaning from context.

2. Correct obvious ASR errors only when
   the intended meaning is clear.

3. Normalize real-estate terminology.

4. Convert amounts expressed in millions
   to Egyptian pounds.

5. Do NOT invent information.

6. If a field is not mentioned or cannot
   be reliably inferred, return null.

7. Extract only customer property requirements.
"""


def extract_customer_profile(conversation):
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is missing. Put it in the .env file.")

    from langchain_core.prompts import ChatPromptTemplate
    from langchain_google_genai import ChatGoogleGenerativeAI

    llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, google_api_key=GEMINI_API_KEY)
    structured_llm = llm.with_structured_output(CustomerRequirements)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "Customer conversation:\n\n{conversation}"),
    ])

    text = "\n".join(f"{t['speaker']}: {t['text']}" for t in conversation)
    print(text)
    return (prompt | structured_llm).invoke({"conversation": text})


# ------------------------------------------------------------------ main
def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser()
    parser.add_argument("--call-id", default="call_00")
    parser.add_argument("--ext", default="ogg")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--skip-profile", action="store_true")
    args = parser.parse_args()

    call_id = args.call_id
    device = get_device()

    audio_path = RAW_AUDIO_DIR / f"{call_id}.{args.ext}"
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found:\n{audio_path}")

    raw_duration = inspect_audio(audio_path, call_id, plot=args.plot)
    ffmpeg_path = setup_ffmpeg()
    wav_path = PROCESSED_AUDIO_DIR / f"{call_id}_standardized.wav"
    processed_duration = standardize_audio(ffmpeg_path, audio_path, wav_path)

    save_json(TRANSCRIPT_DIR / f"{call_id}_vad.json", run_vad(wav_path, call_id, device))

    info, segments, full_text = transcribe(wav_path, device)
    save_json(TRANSCRIPT_DIR / f"{call_id}_transcript.json", {
        "call_id": call_id,
        "audio_file": wav_path.name,
        "language": info.language,
        "language_probability": round(info.language_probability, 4),
        "segments": segments,
        "full_transcript": full_text,
    })

    turns, speakers = run_diarization(wav_path, device)
    save_json(DIARIZATION_DIR / f"{call_id}_diarization.json", {
        "call_id": call_id,
        "audio_file": wav_path.name,
        "num_speakers": len(speakers),
        "speakers": speakers,
        "turns": turns,
    })

    conversation = build_conversation(segments, turns)
    conv_speakers = sorted({t["speaker"] for t in conversation if t["speaker"]})
    save_json(TRANSCRIPT_DIR / f"{call_id}_conversation.json", {
        "call_id": call_id,
        "audio_file": wav_path.name,
        "language": info.language,
        "duration": round(processed_duration, 3),
        "num_speakers": len(conv_speakers),
        "speakers": conv_speakers,
        "conversation": conversation,
    })

    qc = run_qc(call_id, raw_duration, processed_duration, speakers, conversation)
    save_json(TRANSCRIPT_DIR / f"{call_id}_qc.json", qc)

    if args.skip_profile:
        print("Skipping customer profile extraction.")
        return

    profile = extract_customer_profile(conversation)
    out = TRANSCRIPT_DIR / f"{call_id}_customer_profile.json"
    out.write_text(profile.model_dump_json(indent=2, exclude_none=False), encoding="utf-8")
    print("\nCustomer Profile:")
    print(profile.model_dump_json(indent=2))
    print("Saved:", out)


# if __name__ == "__main__":
#     main()