"""
MagicHub Egyptian Arabic Conversational Speech Corpus (ASR-EgArbCSC) Loader
Handles dataset loading, transcription parsing, and speaker turn extraction.
Dataset reference: https://magichub.com/datasets/egyptian-arabic-conversational-speech-corpus/
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import json


class MagicHubDatasetLoader:
    """Loader and parser for the MagicHub Egyptian Arabic Conversational Corpus."""

    def __init__(self, dataset_dir: Path):
        self.dataset_dir = Path(dataset_dir)
        self.audio_dir = self.dataset_dir / "audio" if (self.dataset_dir / "audio").exists() else self.dataset_dir
        self.transcripts_dir = self.dataset_dir / "transcripts" if (self.dataset_dir / "transcripts").exists() else self.dataset_dir

    def list_audio_files(self) -> List[Path]:
        """Lists all WAV audio files in the dataset directory."""
        if not self.audio_dir.exists():
            return []
        files = list(self.audio_dir.glob("*.wav")) + list(self.audio_dir.glob("*.ogg")) + list(self.audio_dir.glob("*.mp3"))
        return sorted(files)

    def load_call_data(self, call_id: str) -> Optional[Dict[str, Any]]:
        """
        Loads audio path, reference transcript, and speaker turns for a given call ID.
        """
        # Look for audio file
        audio_candidates = list(self.audio_dir.glob(f"{call_id}.*"))
        if not audio_candidates:
            return None
        audio_path = audio_candidates[0]

        # Look for transcript file (JSON, TSV, or TXT)
        json_path = self.transcripts_dir / f"{call_id}.json"
        txt_path = self.transcripts_dir / f"{call_id}.txt"
        tsv_path = self.transcripts_dir / f"{call_id}.tsv"

        turns = []
        full_transcript = ""

        if json_path.exists():
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                turns = data.get("turns", data.get("conversation", []))
                full_transcript = data.get("full_transcript", "")
                if not full_transcript and turns:
                    full_transcript = " ".join(t.get("text", "") for t in turns)
        elif tsv_path.exists():
            turns, full_transcript = self._parse_tsv(tsv_path)
        elif txt_path.exists():
            turns, full_transcript = self._parse_txt(txt_path)

        return {
            "call_id": call_id,
            "audio_path": audio_path,
            "turns": turns,
            "full_transcript": full_transcript
        }

    def _parse_tsv(self, tsv_path: Path) -> Tuple[List[Dict[str, Any]], str]:
        """Parses a TSV transcript file with columns: start \t end \t speaker \t text [ \t role ]."""
        turns = []
        texts = []
        with open(tsv_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 4:
                    try:
                        start = float(parts[0])
                        end = float(parts[1])
                        speaker = parts[2]
                        text = parts[3]
                        role = parts[4] if len(parts) >= 5 else None
                        turn = {
                            "start": start,
                            "end": end,
                            "speaker": speaker,
                            "text": text
                        }
                        if role:
                            turn["role"] = role
                        turns.append(turn)
                        texts.append(text)
                    except ValueError:
                        continue
        return turns, " ".join(texts)

    def _parse_txt(self, txt_path: Path) -> Tuple[List[Dict[str, Any]], str]:
        """Parses a plain TXT transcript file formatted with speaker prefixes."""
        turns = []
        texts = []
        with open(txt_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                # Example: SPEAKER_00 [0.0 -> 4.5]: نص الحوار
                if ":" in line:
                    prefix, text = line.split(":", 1)
                    speaker = prefix.split("[")[0].strip()
                    turns.append({
                        "start": 0.0,
                        "end": 0.0,
                        "speaker": speaker,
                        "text": text.strip()
                    })
                    texts.append(text.strip())
                else:
                    texts.append(line)
        return turns, " ".join(texts)

    def load_all_calls(self) -> List[Dict[str, Any]]:
        """Loads all available calls in the dataset directory."""
        audio_files = self.list_audio_files()
        calls = []
        for af in audio_files:
            call_id = af.stem
            # Clean common suffixes like _standardized
            clean_id = call_id.replace("_standardized", "")
            data = self.load_call_data(clean_id) or self.load_call_data(call_id)
            if data:
                calls.append(data)
        return calls
