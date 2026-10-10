"""app.py: links pipeline.py (audio -> speaker-labeled conversation)
with user_input_text.py (requirements extraction via Ollama)."""
from datetime import datetime
import sys
from pathlib import Path
from backend.services.user_input_text import extract_requirements

import gradio as gr

from backend.services.pipeline import (
    PROCESSED_AUDIO_DIR,
    build_conversation,
    get_device,
    run_diarization,
    setup_ffmpeg,
    standardize_audio,
    transcribe,
    DATA_DIR
) 

DEVICE = get_device()
OUTPUT_DIR = Path(r"backend\services\data\Json_Files")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def merge_turns(conversation):
    merged = []
    for t in conversation:
        if merged and merged[-1]["speaker"] == t["speaker"]:
            merged[-1]["text"] += " " + t["text"]
            merged[-1]["end"] = t["end"]
        else:
            merged.append(dict(t))
    return merged

def audio_to_conversation(audio_file):
    """Audio file -> speaker-labeled conversation text."""
    call_id = Path(audio_file).stem
    wav_path = PROCESSED_AUDIO_DIR / f"{call_id}_standardized.wav"

    standardize_audio(setup_ffmpeg(), Path(audio_file), wav_path)
    _, segments, _ = transcribe(wav_path, DEVICE)
    turns, _ = run_diarization(wav_path, DEVICE)
    conversation = merge_turns(build_conversation(segments, turns))

    return "\n".join(f"{t['speaker']}: {t['text']}" for t in conversation)


def link_call_to_requirements(audio_file, text_input):
    raw_text = text_input.strip() if text_input else None
    conversation_text = audio_to_conversation(audio_file) if audio_file is not None else None

    if not any([conversation_text, raw_text]):
        return (
            gr.Textbox(visible=False),
            "Please provide at least one input: a recorded call or text.",
            None,
        )

    result = extract_requirements(call_transcript=conversation_text, raw_text=raw_text)
    json_text = result.model_dump_json(indent=2)

    name = Path(audio_file).stem if audio_file else "text"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = OUTPUT_DIR / f"{name}_{stamp}.json"
    json_path.write_text(json_text, encoding="utf-8")

    # Show the conversation box only when a call was provided
    conversation_box = gr.Textbox(value=conversation_text, visible=conversation_text is not None)

    return conversation_box, json_text, str(json_path)


demo = gr.Interface(
    fn=link_call_to_requirements,
    inputs=[
        gr.Audio(sources=["upload", "microphone"], type="filepath", label="Recorded Call (optional)"),
        gr.Textbox(lines=3, label="Text (Arabic or English, optional)",
                   placeholder="e.g. I want a 150 sqm apartment in New Cairo..."),
    ],
    outputs=[
        gr.Textbox(label="Call Conversation (speakers)", lines=8, visible=False),
        gr.Textbox(label="Extracted Requirements (JSON)", lines=14),
        gr.File(label="Download JSON"),
    ],
    title="Smart Semsar: Tell us what you're looking for",
    description="Upload a call recording and/or type your requirements. You can fill in one or both.",
)

# if __name__ == "__main__":
#     if hasattr(sys.stdout, "reconfigure"):
#         sys.stdout.reconfigure(encoding="utf-8")
#     demo.launch(allowed_paths=[str(OUTPUT_DIR)])