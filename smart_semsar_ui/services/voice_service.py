"""Future integration boundary for audio processing.

This module intentionally contains no model/API calls in the UI-only version.
Implement preprocessing/transcription here when backend integration begins.
"""


def analyze_audio_demo(filename: str) -> dict:
    """Return a predictable demo result without processing audio."""
    return {
        "filename": filename,
        "status": "demo",
        "message": "This is a UI-only preview. No audio was processed.",
        "steps": [
            {"name": "Audio Standardization", "status": "pending"},
            {"name": "Voice Activity Detection", "status": "pending"},
            {"name": "Speech Recognition", "status": "pending"},
            {"name": "Speaker Diarization", "status": "pending"},
            {"name": "Arabic Normalization", "status": "pending"},
            {"name": "Intent Detection", "status": "pending"},
            {"name": "Customer Profile", "status": "pending"},
        ],
    }
