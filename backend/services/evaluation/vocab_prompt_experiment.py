"""Whisper vocabulary-prompt experiment (initial_prompt) on the 10 real calls.

Question: does giving faster-whisper large-v3-turbo a list of expected real-estate words
(`initial_prompt`) fix the errors that break extraction ("فيلا" heard as "بيلا"/"بلا",
"التجمع" as "التجامعة", "تمليك" as "تمليئك")?

Only the prompt changes between variants; everything else is `pipeline.transcribe()`'s
setup (same model, int8 CPU, language="ar", beam_size=5, vad_filter=False, same
standardized 16 kHz WAV, same EOF-hallucination filter).

Variants:
  baseline   - no prompt (control: should reproduce the official transcripts)
  app_prompt - the exact prompt the app already uses in user_input_audio.py
  extended   - app prompt + deal words and common Cairo areas. Some of these areas also
               occur in the 10 test calls, so this variant is optimistic (possible test leakage).

Speaker roles: pyannote is not re-run. Each new Whisper segment takes the speaker/role of
the official conversation turn it overlaps most in time (official
`transcripts/conversations/<call>_conversation.json`), so diarization stays fixed and the
prompt is the only variable.

Outputs (never touches the official transcripts/reports):
  data/evaluation/experiments/vocab_prompt/<variant>/transcripts/<call>_transcript.json
  data/evaluation/experiments/vocab_prompt/<variant>/conversations/<call>_conversation.json
  data/evaluation/experiments/vocab_prompt/vocab_prompt_results.json

Usage:
  python backend/services/evaluation/vocab_prompt_experiment.py
  python backend/services/evaluation/vocab_prompt_experiment.py --variants baseline app_prompt
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.services import pipeline as pl  # noqa: E402
from backend.services.evaluation.metrics import (  # noqa: E402
    compute_cer, compute_corpus_cer, compute_corpus_wer, compute_wer,
)

DATA_DIR = REPO_ROOT / "backend" / "services" / "data"
RAW_DIR = DATA_DIR / "raw_calls"
OFFICIAL_CONV_DIR = DATA_DIR / "transcripts" / "conversations"
GT_DIR = DATA_DIR / "evaluation" / "ground_truth" / "transcripts"
OUT_DIR = DATA_DIR / "evaluation" / "experiments" / "vocab_prompt"
WAV_DIR = OUT_DIR / "_wav"

APP_PROMPT = ("مكالمة عقارية بالعامية المصرية: شقة، فيلا، دوبلكس، أوض نوم، ميزانية، مليون، متر، "
              "التجمع، نيو كايرو، قريبة من المدارس، تشطيب، تقسيط، سكن، استثمار.")
EXTENDED_PROMPT = (APP_PROMPT[:-1] + "، إيجار، تمليك، غرف، ألف، يا فندم، مكتب عقارات، "
                   "التجمع الخامس، العاصمة الإدارية، 6 أكتوبر، الشيخ زايد، مصر الجديدة، المعادي، مدينة نصر.")
# "extended" is pipeline.REAL_ESTATE_WORD_LIST (off by default; a test keeps them equal)
VARIANTS = {"baseline": None, "app_prompt": APP_PROMPT, "extended": EXTENDED_PROMPT}

# words whose mishearing broke extraction in the official run, and the wrong forms seen
KEY_WORDS = {
    "فيلا": ["بيلا", "بلا"],
    "التجمع": ["التجامعة"],
    "تمليك": ["تمليئك", "تمليئ"],
    "شقة": [],
}
CALLS = [f"smartsemsar_rec{i}" for i in range(1, 11)]


def transcribe(model, wav: Path, prompt):
    """pipeline.transcribe() with an optional initial_prompt (same filtering of EOF overflow)."""
    segments, info = model.transcribe(str(wav), language="ar", task="transcribe",
                                      beam_size=5, vad_filter=False, initial_prompt=prompt)
    segments = list(segments)
    dur = float(getattr(info, "duration", 0.0) or 0.0)
    out = []
    for s in segments:
        start, end, text = float(s.start), float(s.end), s.text.strip()
        if not text or (dur and start >= dur):
            continue
        if dur and end > dur + 5.0 and start >= max(0.0, dur - 6.0):
            continue
        start, end = round(max(0.0, start), 3), round(min(end, dur) if dur else end, 3)
        if end <= start:
            continue
        out.append({"segment_id": len(out) + 1, "start": start, "end": end, "text": text,
                    "normalized_text": pl.normalize_arabic_text(text)})
    full = " ".join(x["normalized_text"] for x in out if x["normalized_text"]).strip()
    return out, full


def assign_roles(segments, official_conv):
    """Each segment takes the speaker/role of the official turn it overlaps most."""
    turns = official_conv["conversation"]
    conv = []
    for seg in segments:
        best, best_ov = None, 0.0
        for t in turns:
            ov = max(0.0, min(seg["end"], t["end"]) - max(seg["start"], t["start"]))
            if ov > best_ov:
                best, best_ov = t, ov
        if best is None:  # no overlap: nearest turn by midpoint
            mid = (seg["start"] + seg["end"]) / 2
            best = min(turns, key=lambda t: abs((t["start"] + t["end"]) / 2 - mid))
        conv.append({"turn_id": seg["segment_id"], "start": seg["start"], "end": seg["end"],
                     "speaker": best.get("speaker"), "role": best.get("role"),
                     "raw_text": seg["text"], "text": seg["text"]})
    payload = {k: v for k, v in official_conv.items() if k != "conversation"}
    payload["conversation"] = conv
    payload["role_source"] = "copied from the official conversation by time overlap (pyannote not re-run)"
    return payload


def key_word_counts(text: str) -> dict:
    return {w: {"correct": text.count(w), "wrong": sum(text.count(x) for x in bad)}
            for w, bad in KEY_WORDS.items()}


def main(argv=None) -> int:
    pl.configure_utf8_console()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    args = ap.parse_args(argv)

    ffmpeg = pl.setup_ffmpeg()
    WAV_DIR.mkdir(parents=True, exist_ok=True)
    wavs = {}
    for cid in CALLS:
        wav = WAV_DIR / f"{cid}_standardized.wav"
        if not wav.exists():
            pl.standardize_audio(ffmpeg, RAW_DIR / f"{cid}.mp4", wav)
        wavs[cid] = wav

    model = pl.load_whisper_model("cpu")
    results = {"generated_at_utc": datetime.now(timezone.utc).isoformat(),
               "model": pl.WHISPER_MODEL, "prompts": {k: VARIANTS[k] for k in args.variants},
               "variants": {}}
    for name in args.variants:
        prompt = VARIANTS[name]
        tdir, cdir = OUT_DIR / name / "transcripts", OUT_DIR / name / "conversations"
        tdir.mkdir(parents=True, exist_ok=True); cdir.mkdir(parents=True, exist_ok=True)
        wer_rows, cer_rows, per_call = [], [], []
        hyp_all, ref_all = "", ""
        t0 = time.time()
        for cid in CALLS:
            print(f"[{name}] {cid} ...", flush=True)
            segs, full = transcribe(model, wavs[cid], prompt)
            pl.save_json(tdir / f"{cid}_transcript.json", {
                "call_id": cid, "source_audio_file": f"{cid}.mp4", "audio_file": wavs[cid].name,
                "whisper_model": pl.WHISPER_MODEL, "asr_backend": "faster-whisper", "language": "ar",
                "initial_prompt": prompt, "segments": segs, "full_transcript": full})
            official = json.loads((OFFICIAL_CONV_DIR / f"{cid}_conversation.json").read_text(encoding="utf-8"))
            pl.save_json(cdir / f"{cid}_conversation.json", assign_roles(segs, official))
            ref = json.loads((GT_DIR / f"{cid}_gt_transcript.json").read_text(encoding="utf-8"))["reference_transcript"]
            w, c = compute_wer(ref, full), compute_cer(ref, full)
            wer_rows.append(w); cer_rows.append(c)
            hyp_all += " " + full; ref_all += " " + pl.normalize_arabic_text(ref)
            per_call.append({"call_id": cid, "wer": w["wer"], "cer": c["cer"], "key_words": key_word_counts(full)})
        results["variants"][name] = {
            "seconds": round(time.time() - t0, 1),
            **{k: v for k, v in compute_corpus_wer(wer_rows).items() if k in ("macro_wer", "micro_wer")},
            **{k: v for k, v in compute_corpus_cer(cer_rows).items() if k in ("macro_cer", "micro_cer")},
            "key_words_total": key_word_counts(hyp_all),
            "key_words_in_reference": key_word_counts(ref_all),
            "calls": per_call,
        }
        v = results["variants"][name]
        print(f"[{name}] micro WER {v['micro_wer']:.4f} | micro CER {v['micro_cer']:.4f} | {v['key_words_total']}", flush=True)
    pl.save_json(OUT_DIR / "vocab_prompt_results.json", results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
