"""
Voice Activity Detection (VAD) Engine
Supports Silero VAD with robust local energy-based acoustic fallback.
"""

from typing import List, Dict, Any, Optional
import numpy as np


class VADEngine:
    """Voice Activity Detector with Silero and Energy-based fallback."""
    
    def __init__(self, mode: str = "auto", device: str = "cpu"):
        self.mode = mode
        self.device = device
        self.silero_model = None
        self.silero_utils = None
        
        if self.mode in ["auto", "silero"]:
            self._init_silero()
            
    def _init_silero(self):
        """Attempts to load Silero VAD from torch hub."""
        try:
            import torch
            model, utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                trust_repo=True,
                verbose=False
            )
            self.silero_model = model.to(self.device)
            self.silero_utils = utils
        except Exception:
            self.silero_model = None
            
    def detect_speech_timestamps(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        threshold: float = 0.5,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 100,
    ) -> List[Dict[str, int]]:
        """
        Detects speech timestamps in samples.
        
        Returns list of dicts: [{'start': int_sample, 'end': int_sample}]
        """
        # If Silero is available, use it
        if self.silero_model is not None:
            try:
                import torch
                get_speech_timestamps = self.silero_utils[0]
                audio_tensor = torch.from_numpy(audio.astype(np.float32)).to(self.device)
                timestamps = get_speech_timestamps(
                    audio_tensor,
                    self.silero_model,
                    sampling_rate=sample_rate,
                    threshold=threshold,
                    min_speech_duration_ms=min_speech_duration_ms,
                    min_silence_duration_ms=min_silence_duration_ms,
                )
                return [{"start": int(s["start"]), "end": int(s["end"])} for s in timestamps]
            except Exception:
                pass
                
        # Acoustic energy-based fallback VAD
        return self._energy_vad(
            audio=audio,
            sample_rate=sample_rate,
            min_speech_ms=min_speech_duration_ms,
            min_silence_ms=min_silence_duration_ms
        )
        
    def _energy_vad(
        self,
        audio: np.ndarray,
        sample_rate: int,
        frame_ms: int = 30,
        energy_threshold_ratio: float = 0.05,
        min_speech_ms: int = 250,
        min_silence_ms: int = 150
    ) -> List[Dict[str, int]]:
        """A robust local energy-based VAD implementation."""
        frame_size = int(sample_rate * (frame_ms / 1000.0))
        if frame_size <= 0:
            frame_size = 480
            
        num_frames = len(audio) // frame_size
        if num_frames == 0:
            return [{"start": 0, "end": len(audio)}]
            
        frames = audio[:num_frames * frame_size].reshape((num_frames, frame_size))
        energies = np.sqrt(np.mean(frames ** 2, axis=1))
        
        max_energy = np.max(energies) if len(energies) > 0 else 0.0
        if max_energy == 0:
            return []
            
        threshold = max(0.005, max_energy * energy_threshold_ratio)
        is_speech = energies >= threshold
        
        # Smooth and group segments
        segments = []
        in_speech = False
        start_frame = 0
        silence_count = 0
        min_silence_frames = int(min_silence_ms / frame_ms)
        min_speech_frames = int(min_speech_ms / frame_ms)
        
        for idx, speech in enumerate(is_speech):
            if speech:
                if not in_speech:
                    in_speech = True
                    start_frame = idx
                silence_count = 0
            else:
                if in_speech:
                    silence_count += 1
                    if silence_count >= min_silence_frames:
                        end_frame = idx - silence_count
                        if (end_frame - start_frame) >= min_speech_frames:
                            segments.append({
                                "start": start_frame * frame_size,
                                "end": min(len(audio), end_frame * frame_size)
                            })
                        in_speech = False
                        silence_count = 0
                        
        if in_speech:
            end_frame = num_frames
            if (end_frame - start_frame) >= min_speech_frames:
                segments.append({
                    "start": start_frame * frame_size,
                    "end": len(audio)
                })
                
        if not segments and len(audio) > 0:
            segments.append({"start": 0, "end": len(audio)})
            
        return segments
