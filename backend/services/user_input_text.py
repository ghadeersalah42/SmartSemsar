"""
Smart Semsar - Customer intake (VS Code version)
Inputs: recorded call and/or text.

Before running:
  1) pip install -r requirements.txt
  2) Install Ollama from https://ollama.com/download  then run:  ollama pull qwen2.5
"""
import json
from typing import Optional, Literal

from dotenv import main
# import ollama
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
import gradio as gr
from pydantic import BaseModel, Field
from faster_whisper import WhisperModel
from backend.schema.state import CustomerRequirements


# ---------- 1) Schema ----------
# class CustomerRequirements(BaseModel):
#     location: Optional[str] = None
#     budget_min: Optional[float] = None
#     budget_max: Optional[float] = None
#     area_sqm: Optional[float] = None
#     bedrooms: Optional[int] = None
#     property_type: Optional[Literal["apartment", "villa", "duplex", "studio", "townhouse", "unknown"]] = "unknown"
#     purpose: Optional[Literal["living", "investment", "unknown"]] = "unknown"
#     notes: Optional[str] = None
#     missing_fields: list[str] = Field(default_factory=list)


# ---------- 2) Speech to text ----------
# Use "small" if your laptop is slow; "medium" is more accurate for Arabic.
_whisper_model = WhisperModel("medium", device="auto", compute_type="int8")


def transcribe_audio(file_path: str) -> str:
    segments, info = _whisper_model.transcribe(
        file_path,
        language=None,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
    )
    print(f"Detected language: {info.language} (confidence: {info.language_probability:.2f})")
    return " ".join(s.text.strip() for s in segments).strip()


# ---------- 3) LLM extraction (Ollama) ----------
# MODEL_NAME = "qwen2.5"

SYSTEM_PROMPT = '''You are a real-estate intake assistant.
You receive raw text gathered from a phone call transcript and/or free text written
by a customer. The input may be in Arabic, English, or a mix of both
(including Egyptian dialect and code-switching).

Extract the customer's property requirements and respond with ONLY a single valid
JSON object matching this exact schema, no prose, no markdown fences:

{
  "location": string or null,
  "budget_min": number or null,
  "budget_max": number or null,
  "area_sqm": number or null,
  "bedrooms": integer or null,
  "property_type": one of ["apartment","villa","duplex","studio","townhouse","unknown"],
  "purpose": one of ["living","investment","unknown"],
  "notes": string or null,
  "missing_fields": array of field names you could not determine
}

Rules:
- Never invent values. If a field is not mentioned, set it to null and add its name to missing_fields.
- Numbers must be plain numbers (no currency symbols, no commas).
- Keep the "location" and "notes" values in their original language as stated by the customer.
- Respond in JSON only.
- Use "notes" ONLY for extra customer requirements that do not fit any other field (for example: near schools, quiet area, high floor).
- Never copy budget, area, bedrooms, or location text into "notes". If there is nothing extra, set "notes" to null.
- Do NOT guess "property_type" or "purpose". Use "unknown" unless the customer clearly states them.
'''


def build_user_prompt(call_transcript=None, raw_text=None) -> str:
    parts = []
    if call_transcript:
        parts.append(f"--- CALL TRANSCRIPT ---\n{call_transcript}")
    if raw_text:
        parts.append(f"--- CUSTOMER TEXT ---\n{raw_text}")
    return "\n\n".join(parts) if parts else "No input provided."

llm = ChatGroq(model_name="openai/gpt-oss-20b", temperature=0)
structured_llm = llm.with_structured_output(CustomerRequirements)
                                            
def extract_requirements(call_transcript=None, raw_text=None) -> CustomerRequirements:
    user_prompt = build_user_prompt(call_transcript, raw_text)
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_prompt)
    ]
    req = structured_llm.invoke(messages)

    fields = ["location", "budget_min", "budget_max", "area_sqm",
              "bedrooms", "property_type", "purpose", "notes"]
    req.missing_fields = [f for f in fields if getattr(req, f) in (None, "unknown")]

    return req
    # response = ollama.chat(
    #     model=MODEL_NAME,
    #     format="json",
    #     messages=[
    #         {"role": "system", "content": SYSTEM_PROMPT},
    #         {"role": "user", "content": build_user_prompt(call_transcript, raw_text)},
    #     ],
    #     options={"temperature": 0},
    # )
    # raw_content = response["message"]["content"]
    # try:
    #     parsed = json.loads(raw_content)
    # except json.JSONDecodeError:
    #     cleaned = raw_content.strip().strip("`").replace("json\n", "", 1)
    #     parsed = json.loads(cleaned)

    # req = CustomerRequirements(**parsed)

    # fields = ["location", "budget_min", "budget_max", "area_sqm",
    #           "bedrooms", "property_type", "purpose", "notes"]
    # req.missing_fields = [f for f in fields if getattr(req, f) in (None, "unknown")]

    # return req

# if __name__ == "__main__":
#    main()