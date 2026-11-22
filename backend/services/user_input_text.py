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
import ollama
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
# _whisper_model = WhisperModel("medium", device="auto", compute_type="int8")
_whisper_model = None

def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        # ممكن كمان نستخدم "base" أو "small" لو الـ medium بيستهلك الرامات عندك
        _whisper_model = WhisperModel("small", device="cpu", compute_type="int8")
    return _whisper_model

def transcribe_audio(file_path: str) -> str:
    _whisper_model=get_whisper_model()
    segments, info = _whisper_model.transcribe(
        file_path,
        language=None,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
    )
    print(f"Detected language: {info.language} (confidence: {info.language_probability:.2f})")
    return " ".join(s.text.strip() for s in segments).strip()


VOICE_PROMPT = "طلب عقاري بالعامية المصرية: شقة، فيلا، دوبلكس، أوض نوم، ميزانية، مليون، متر، التجمع، الشيخ زايد، تشطيب، تقسيط."


def transcribe_voice_note(file_path: str) -> str:
    """تسجيل المستخدم من الشات -> نص. Groq Whisper الأول (أسرع وأدق في العربي)،
    ولو مفيش GROQ_API_KEY أو فشل، faster-whisper على الجهاز."""
    import os
    if os.getenv("GROQ_API_KEY"):
        try:
            from groq import Groq
            with open(file_path, "rb") as f:
                result = Groq().audio.transcriptions.create(
                    file=(os.path.basename(file_path), f.read()),
                    model="whisper-large-v3", prompt=VOICE_PROMPT, temperature=0)
            return (result.text or "").strip()
        except Exception as e:
            print(f"Groq transcription failed, using local Whisper: {e}")
    return transcribe_audio(file_path)


# ---------- 3) LLM extraction (Ollama) ----------
MODEL_NAME = "qwen2.5"      # المتسطب على الجهاز (ollama list)
# "qwen2.5"
# model=
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
  "deal_type": one of ["rent","buy","unknown"],
  "notes": string or null,
  "missing_fields": array of field names you could not determine
}

Rules:
- Never invent values. If a field is not mentioned, set it to null and add its name to missing_fields.
- Numbers must be plain numbers (no currency symbols, no commas).
- Budgets are full Egyptian pounds:
  * "5 مليون" / "5 million" / "5m" -> 5000000
  * "800 ألف" / "800k" / "800 alf" -> 800000
  * "8 آلاف" / "تمن آلاف" / "8 alf" -> 8000 (be careful with zeros: "ألف" = 1,000, NOT 10,000).
- Price per square meter ("سعر المتر", "المتر بـ", "el metr", "per m2", "price per m"):
  * If both a price per m² and area_sqm are present (or inferable), compute total budget = price_per_m2 * area_sqm and put it in budget_max.
  * When a price per m² is given for buying (e.g. "20 ألف" or "30 ألف" للمتر), deal_type is typically "buy" unless "إيجار/rent" is explicitly stated.
- Spoken two-number shorthand like "70 6" in Egyptian real-estate context:
  * If one number is a typical area (e.g. 70 sqm) and the following standalone small number (e.g. 6) is a budget in millions, set area_sqm = 70, budget_max = 6000000, and deal_type = "buy" if not rent.
- A single budget figure ("بميزانية 5 مليون", "في حدود 8 آلاف", "up to 5 million") is budget_max only; set budget_min only for a range or an explicit minimum.
- Keep the "location" and "notes" values in their original language as stated by the customer.
- Respond in JSON only.
- Use "notes" ONLY for extra customer requirements that do not fit any other field (for example: near schools, quiet area, high floor).
- Never copy budget, area, bedrooms, or location text into "notes". If there is nothing extra, set "notes" to null.
- Do NOT guess "property_type" or "purpose". Use "unknown" unless the customer clearly states them.
- "deal_type": "rent" for إيجار / ايجار / أأجر / rent, "buy" for تمليك / شراء / اشتري / أشتري / buy; otherwise "unknown" (a budget alone does not decide it).
'''


def build_user_prompt(call_transcript=None, raw_text=None) -> str:
    parts = []
    if call_transcript:
        parts.append(f"--- CALL TRANSCRIPT ---\n{call_transcript}")
    if raw_text:
        parts.append(f"--- CUSTOMER TEXT ---\n{raw_text}")
    return "\n\n".join(parts) if parts else "No input provided."

# llm = ChatGroq(model_name="openai/gpt-oss-20b", temperature=0)
# structured_llm = llm.with_structured_output(CustomerRequirements)
# llm = ChatGoogleGenerativeAI(
#     model="gemini-3.8-flash",
#     temperature=0
# )

def _llm_json(messages) -> str:
    """رد الموديل كـ JSON: Groq الأول (سريع ومفتاحه في .env)، ولو مش موجود أو فشل، Ollama على الجهاز
    (SMARTSEMSAR_OLLAMA_MODEL لو حابة تغيري الموديل المتسطب)"""
    import os
    if os.getenv("GROQ_API_KEY"):
        try:
            from groq import Groq
            response = Groq().chat.completions.create(
                model=os.getenv("SMARTSEMSAR_GROQ_MODEL", "openai/gpt-oss-120b"),
                messages=messages, response_format={"type": "json_object"}, temperature=0)
            return response.choices[0].message.content
        except Exception as e:
            print(f"Groq extraction failed, using Ollama: {e}")
    response = ollama.chat(
        model=os.getenv("SMARTSEMSAR_OLLAMA_MODEL", MODEL_NAME),
        format="json", messages=messages, options={"temperature": 0})
    return response["message"]["content"]


def extract_requirements(call_transcript=None, raw_text=None) -> CustomerRequirements:
    raw_content = _llm_json([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(call_transcript, raw_text)},
    ])
    try:
        parsed = json.loads(raw_content)
    except json.JSONDecodeError:
        cleaned = raw_content.strip().strip("`").replace("json\n", "", 1)
        parsed = json.loads(cleaned)

    req = CustomerRequirements(**parsed)
    import re
    text_blob = f"{call_transcript or ''} {raw_text or ''}".lower()
    # معالجة سعر المتر لو الموديل لم يحسب الميزانية الكلية
    if req.budget_max is None and req.budget_min is None and req.area_sqm:
        has_meter_price = any(
            k in text_blob
            for k in ["سعر المتر", "المتر ب", "el metr", "metr b", "per m", "price per"]
        )
        if has_meter_price:
            m_alf = re.search(r"(?:بـ?|b|ميزيدش عن|عن)?\s*(\d+(?:\.\d+)?)\s*(?:ألف|الف|alf|k)\b", text_blob)
            m_mill = re.search(r"(?:بـ?|b|ميزيدش عن|عن)?\s*(\d+(?:\.\d+)?)\s*(?:مليون|million|m)\b", text_blob)
            unit_price = None
            if m_mill:
                unit_price = float(m_mill.group(1)) * 1_000_000
            elif m_alf:
                unit_price = float(m_alf.group(1)) * 1_000
            if unit_price and unit_price > 0:
                req.budget_max = float(req.area_sqm) * unit_price
                if req.deal_type in (None, "unknown"):
                    req.deal_type = "buy"

    fields = ["location", "budget_min", "budget_max", "area_sqm",
              "bedrooms", "property_type", "purpose", "deal_type", "notes"]
    req.missing_fields = [f for f in fields if getattr(req, f) in (None, "unknown")]

    return req


# def extract_requirements(call_transcript=None, raw_text=None) -> CustomerRequirements:
#     user_prompt = build_user_prompt(call_transcript, raw_text)
    
#     messages = [
#         SystemMessage(content=SYSTEM_PROMPT + "\n\nCRITICAL: Respond ONLY with a valid JSON object matching the requested schema, no extra text."),
#         HumanMessage(content=user_prompt)
#     ]
    
#     # استدعاء عادي جداً بدون with_structured_output لتجنب مشاكل الـ Tool choice
#     response = llm.invoke(messages)
#     raw_content = response.content
    
#     try:
#         parsed = json.loads(raw_content)
#     except json.JSONDecodeError:
#         # تنظيف الـ markdown لو الموديل حط ````json ... ````
#         cleaned = raw_content.strip().strip("`").replace("json\n", "", 1)
#         parsed = json.loads(cleaned)

#     req = CustomerRequirements(**parsed)

#     fields = ["location", "budget_min", "budget_max", "area_sqm",
#               "bedrooms", "property_type", "purpose", "notes"]
#     req.missing_fields = [f for f in fields if getattr(req, f) in (None, "unknown")]

#     return req

# def extract_requirements(call_transcript=None, raw_text=None) -> CustomerRequirements:
#     user_prompt = build_user_prompt(call_transcript, raw_text)
#     messages = [
#         SystemMessage(content=SYSTEM_PROMPT),
#         HumanMessage(content=user_prompt)
#     ]
#     req = structured_llm.invoke(messages)

#     fields = ["location", "budget_min", "budget_max", "area_sqm",
#               "bedrooms", "property_type", "purpose", "notes"]
#     req.missing_fields = [f for f in fields if getattr(req, f) in (None, "unknown")]

#     return req
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