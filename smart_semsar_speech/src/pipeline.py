"""
End-to-End Call Intelligence Pipeline for Smart Semsar
Integrates Audio Standardization, VAD, ASR, Diarization, Role Separation,
QC, and Structured Customer Intelligence Extraction.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np

from .audio_processing import (
    standardize_audio,
    load_wav_audio,
    compute_audio_quality_metrics,
)
from .text_normalization import normalize_arabic_text
from .vad_engine import VADEngine
from .diarization_engine import DiarizationEngine, assign_speaker
from .role_separator import RoleSeparator


class CallIntelligencePipeline:
    """Master pipeline for real estate phone call speech analytics."""

    def __init__(
        self,
        hf_token: Optional[str] = None,
        whisper_model_name: str = "large-v3-turbo",
        device: str = "cpu",
        gemini_api_key: Optional[str] = None
    ):
        self.device = device
        self.hf_token = hf_token
        self.whisper_model_name = whisper_model_name
        self.gemini_api_key = gemini_api_key
        
        # Engines
        self.vad_engine = VADEngine(device=self.device)
        self.diarization_engine = DiarizationEngine(hf_token=self.hf_token)
        self.role_separator = RoleSeparator()
        self.whisper_model = None

    def _get_whisper_model(self):
        """Lazy-loads Whisper model to save memory."""
        if self.whisper_model is None:
            try:
                from faster_whisper import WhisperModel
                compute_type = "float16" if self.device == "cuda" else "int8"
                self.whisper_model = WhisperModel(
                    self.whisper_model_name,
                    device=self.device,
                    compute_type=compute_type
                )
            except Exception:
                self.whisper_model = None
        return self.whisper_model

    def process_call(
        self,
        audio_path: Path,
        call_id: Optional[str] = None,
        output_dir: Optional[Path] = None,
        offline_transcript: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Executes full speech intelligence pipeline on a phone call.
        """
        call_id = call_id or audio_path.stem
        out_dir = Path(output_dir) if output_dir else audio_path.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Standardize Audio
        standardized_wav = out_dir / f"{call_id}_standardized.wav"
        if not standardized_wav.exists() or standardized_wav != audio_path:
            try:
                standardize_audio(audio_path, standardized_wav)
            except Exception:
                standardized_wav = audio_path
                
        # 2. Load & Quality Check
        audio_data, sr = load_wav_audio(standardized_wav)
        qc_audio = compute_audio_quality_metrics(audio_data, sr)
        
        # 3. Voice Activity Detection (VAD)
        speech_timestamps = self.vad_engine.detect_speech_timestamps(audio_data, sample_rate=sr)
        speech_segments = []
        for i, seg in enumerate(speech_timestamps, start=1):
            start_sec = round(seg["start"] / sr, 3)
            end_sec = round(seg["end"] / sr, 3)
            speech_segments.append({
                "id": i,
                "start": start_sec,
                "end": end_sec,
                "duration": round(end_sec - start_sec, 3)
            })
            
        # 4. Automatic Speech Recognition (ASR)
        model = self._get_whisper_model()
        transcript_segments = []
        
        if model is not None:
            try:
                segments, info = model.transcribe(
                    str(standardized_wav),
                    language="ar",
                    task="transcribe",
                    beam_size=5,
                    vad_filter=False
                )
                for seg in segments:
                    orig_text = seg.text.strip()
                    transcript_segments.append({
                        "start": round(seg.start, 3),
                        "end": round(seg.end, 3),
                        "text": orig_text,
                        "normalized_text": normalize_arabic_text(orig_text)
                    })
            except Exception:
                pass
                
        # If ASR was offline or empty, use fallback/ground truth if provided
        if not transcript_segments and offline_transcript:
            norm_off = normalize_arabic_text(offline_transcript)
            transcript_segments.append({
                "start": 0.0,
                "end": qc_audio["duration_seconds"],
                "text": offline_transcript,
                "normalized_text": norm_off
            })
            
        full_transcript = " ".join(s["normalized_text"] for s in transcript_segments)
        
        # 5. Diarization
        speaker_turns = self.diarization_engine.diarize_audio(
            audio_path=standardized_wav,
            audio_array=audio_data,
            sample_rate=sr
        )
        detected_speakers = sorted(list(set(t["speaker"] for t in speaker_turns)))
        
        # 6. Alignment (ASR Segments -> Speakers)
        aligned_conversation = []
        for seg in transcript_segments:
            spk, overlap = assign_speaker(seg["start"], seg["end"], speaker_turns)
            aligned_conversation.append({
                "start": seg["start"],
                "end": seg["end"],
                "speaker": spk,
                "text": seg["normalized_text"]
            })
            
        # 7. Role Separation (Broker vs. Client)
        role_result = self.role_separator.separate_roles(aligned_conversation)
        final_turns = role_result["assigned_turns"]
        
        # 8. QC Validations
        empty_segments = [t for t in final_turns if not t.get("text", "").strip()]
        invalid_ts = [t for t in final_turns if t["start"] >= t["end"] or t["start"] < 0]
        words = len(full_transcript.split())
        qc_status = "PASS" if (len(empty_segments) == 0 and len(invalid_ts) == 0 and words > 0) else "WARNING"
        
        pipeline_output = {
            "call_id": call_id,
            "audio_file": standardized_wav.name,
            "duration": qc_audio["duration_seconds"],
            "qc_metrics": qc_audio,
            "num_speakers": len(detected_speakers),
            "speakers": detected_speakers,
            "role_mapping": role_result["role_mapping"],
            "role_separation_confidence": role_result["confidence"],
            "conversation": final_turns,
            "full_transcript": full_transcript,
            "vad_segments_count": len(speech_segments),
            "qc_status": qc_status
        }
        
        # Save output JSON
        json_out = out_dir / f"{call_id}_intelligence.json"
        with open(json_out, "w", encoding="utf-8") as f:
            json.dump(pipeline_output, f, ensure_ascii=False, indent=2)
            
        return pipeline_output
