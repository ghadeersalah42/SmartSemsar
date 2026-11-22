"""Batch Speech Pipeline runner for SmartSemsar recorded phone calls.

Processes all 10 MP4 calls in `backend/services/data/raw_calls/` independently
while loading Silero VAD, Whisper (`large-v3-turbo`), and pyannote
(`pyannote/speaker-diarization-community-1`) only once across the batch.

Usage:
    python backend/services/batch_pipeline.py
    python backend/services/batch_pipeline.py --calls smartsemsar_rec1 smartsemsar_rec2
    python backend/services/batch_pipeline.py --with-profile
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Optional

# Allow direct execution via `python backend/services/batch_pipeline.py`
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.services.pipeline import (  # noqa: E402
    DEFAULT_MIN_OVERLAP_RATIO,
    DEFAULT_MIN_OVERLAP_SEC,
    DEFAULT_MIN_SPEAKER_DOMINANCE,
    EXIT_PROCESSING_ERROR,
    EXPECTED_SPEAKERS,
    RAW_AUDIO_DIR,
    configure_utf8_console,
    get_batch_exit_code,
    process_batch,
)

DEFAULT_EVAL_CALLS = [f"smartsemsar_rec{i}" for i in range(1, 11)]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Batch-process SmartSemsar real-estate phone recordings. "
            "Exit codes: 0=all PASS/WARNING, 1=processing error, 2=QC FAIL."
        )
    )
    parser.add_argument(
        "--calls",
        nargs="+",
        default=None,
        help=(
            "Specific call IDs to process. If omitted, processes all matching files "
            "in --input-dir (e.g., smartsemsar_rec1..smartsemsar_rec10)."
        ),
    )
    parser.add_argument(
        "--ext",
        default="mp4",
        help="Recording file extension (default: mp4; ogg also supported).",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=RAW_AUDIO_DIR,
        help=f"Directory containing raw recordings (default: {RAW_AUDIO_DIR}).",
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
    parser.add_argument(
        "--with-profile",
        action="store_true",
        help="Also run Gemini customer profile extraction (skipped by default for batch speech evaluation).",
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
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot waveform for each call.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    configure_utf8_console()

    args = build_arg_parser().parse_args(argv)
    try:
        summary = process_batch(
            call_ids=args.calls,
            ext=args.ext,
            skip_profile=not args.with_profile,
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
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"\n[BATCH PIPELINE ERROR] {exc}", file=sys.stderr)
        return EXIT_PROCESSING_ERROR


if __name__ == "__main__":
    sys.exit(main())
