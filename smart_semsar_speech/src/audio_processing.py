"""
Audio Processing & Telephony Simulation Module
Handles standardization, quality assessment, ITU-T G.712 telephone filtering,
and noise simulation at configurable SNR levels.
"""

import os
import shutil
import subprocess
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

import numpy as np
from scipy import signal
import scipy.io.wavfile as wavfile


def get_ffmpeg_executable() -> str:
    """Finds ffmpeg executable from PATH or imageio-ffmpeg."""
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        return ffmpeg_path
    
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    
    return "ffmpeg"


def standardize_audio(
    input_path: Path,
    output_path: Path,
    target_sample_rate: int = 16000,
    force_mono: bool = True
) -> Path:
    """
    Standardizes any audio format into 16kHz mono 16-bit PCM WAV using FFmpeg.
    """
    ffmpeg_exe = get_ffmpeg_executable()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        ffmpeg_exe,
        "-y",
        "-i", str(input_path),
        "-ar", str(target_sample_rate),
        "-c:a", "pcm_s16le"
    ]
    if force_mono:
        cmd.extend(["-ac", "1"])
        
    cmd.append(str(output_path))
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg audio standardization failed:\n{result.stderr}")
        
    return output_path


def load_wav_audio(wav_path: Path) -> Tuple[np.ndarray, int]:
    """
    Loads a WAV file into a normalized float32 numpy array and sample rate.
    """
    sr, data = wavfile.read(str(wav_path))
    
    # Convert multi-channel to mono
    if data.ndim > 1:
        data = data.mean(axis=1)
        
    # Normalize to float32 [-1.0, 1.0]
    if data.dtype == np.int16:
        float_data = (data / 32768.0).astype(np.float32)
    elif data.dtype == np.int32:
        float_data = (data / 2147483648.0).astype(np.float32)
    elif data.dtype == np.uint8:
        float_data = ((data - 128) / 128.0).astype(np.float32)
    else:
        float_data = data.astype(np.float32)
        
    return float_data, sr


def save_wav_audio(
    audio: np.ndarray,
    sr: int,
    output_path: Path
) -> Path:
    """
    Saves a float32 numpy array as a 16-bit PCM WAV file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Clip and convert to int16
    clipped = np.clip(audio, -1.0, 1.0)
    int16_data = (clipped * 32767.0).astype(np.int16)
    wavfile.write(str(output_path), sr, int16_data)
    return output_path


def apply_telephony_bandpass(
    audio: np.ndarray,
    sr: int,
    lowcut: float = 300.0,
    highcut: float = 3400.0,
    order: int = 5
) -> np.ndarray:
    """
    Applies an ITU-T G.712 bandpass filter (300 Hz - 3400 Hz) to simulate real telephone channel audio.
    """
    nyquist = 0.5 * sr
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(order, [low, high], btype="band")
    filtered = signal.filtfilt(b, a, audio)
    return filtered.astype(np.float32)


def add_acoustic_noise(
    audio: np.ndarray,
    snr_db: float,
    noise_type: str = "white"
) -> np.ndarray:
    """
    Adds noise to audio at a specific Signal-to-Noise Ratio (SNR) in dB.
    
    Args:
        audio: 1D float32 audio array.
        snr_db: Target SNR in decibels (e.g., 20.0, 10.0, 5.0).
        noise_type: 'white' or 'pink'.
    """
    signal_power = np.mean(audio ** 2)
    if signal_power == 0:
        return audio
        
    snr_linear = 10.0 ** (snr_db / 10.0)
    noise_power = signal_power / snr_linear
    
    if noise_type == "white":
        noise = np.random.normal(0, np.sqrt(noise_power), len(audio))
    else:
        # Simple pink-like filtered noise
        white = np.random.normal(0, 1.0, len(audio))
        b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
        a = [1.0, -2.494956002, 2.017265875, -0.522189400]
        pink = signal.lfilter(b, a, white)
        pink_power = np.mean(pink ** 2)
        if pink_power > 0:
            noise = pink * np.sqrt(noise_power / pink_power)
        else:
            noise = white * np.sqrt(noise_power)
            
    noisy_audio = audio + noise
    # Prevent hard clipping distortion
    max_val = np.max(np.abs(noisy_audio))
    if max_val > 1.0:
        noisy_audio = noisy_audio / max_val
        
    return noisy_audio.astype(np.float32)


def compute_audio_quality_metrics(audio: np.ndarray, sr: int) -> Dict[str, Any]:
    """
    Calculates audio quality statistics: duration, RMS, peak amplitude, and clipping ratio.
    """
    duration = len(audio) / sr if sr > 0 else 0.0
    rms = float(np.sqrt(np.mean(audio ** 2)))
    peak = float(np.max(np.abs(audio))) if len(audio) > 0 else 0.0
    
    clipping_thresh = 0.99
    clipped_samples = int(np.sum(np.abs(audio) >= clipping_thresh))
    clipping_ratio = float(clipped_samples / len(audio)) if len(audio) > 0 else 0.0
    
    return {
        "duration_seconds": round(duration, 3),
        "sample_rate": sr,
        "samples": len(audio),
        "mean_amplitude": float(np.mean(audio)),
        "std_amplitude": float(np.std(audio)),
        "rms_energy": round(rms, 6),
        "peak_amplitude": round(peak, 6),
        "clipped_samples": clipped_samples,
        "clipping_ratio_pct": f"{clipping_ratio * 100:.4f}%",
        "quality_status": "Clean" if clipping_ratio < 0.001 else "Clipped"
    }
