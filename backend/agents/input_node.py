"""input_node.py: LangGraph node that turns the customer's input
(recorded call and/or free text) into structured requirements.

Reads from state:
    audio_path : str | None   path to a recorded call
    user_text  : str | None   free text (Arabic or English)

Writes to state:
    conversation_text : str | None   speaker-labeled call transcript
    requirements      : dict | None  CustomerRequirements as a dict
    missing_fields    : list[str]
    error             : str | None
"""
from backend.services.user_input import audio_to_conversation
from backend.services.user_input_text import extract_requirements
from backend.schema.state import AgentState, PropertyItem, CustomerRequirements
from backend.guardrails.clarify import (find_unclear, drop_guesses, clarification_question,
                                        with_clarification, FIELD_ORDER)
from backend.guardrails.pii import safe_print
from backend.ai_config import threshold
import re
from typing import Optional, Literal, List, Dict, Any
from langgraph.types import interrupt


def has_requirements(req) -> bool:
    """فيه مواصفة واحدة على الأقل اتفهمت من كلام العميل؟ (نوع العقار لوحده مش كفاية:
    "عايزة أدور على شقة" بيطلع apartment ومفيهوش مواصفات)"""
    if req is None:
        return False
    fields = ["location", "budget_min", "budget_max", "area_sqm", "bedrooms"]
    return any(getattr(req, f, None) not in (None, "unknown") for f in fields)


ASK_SPECS = ("تمام 👌 اكتبلي مواصفات الشقة اللي بتدور عليها: المنطقة، الميزانية، عدد الغرف، "
             "والمساحة لو تحب. وكمان عايزها إيجار ولا تمليك؟")
ASK_SPECS_AGAIN = "مش واضحلي المواصفات، ممكن تكتبلي المنطقة أو الميزانية أو عدد الغرف؟"
ASK_DEAL = "تمام 👌 وعايز الشقة إيجار ولا تمليك؟"
ASK_DEAL_AGAIN = "معلش مش متأكد فهمت 🙏 تقصد إيجار ولا تمليك؟"


def ask_deal_type(req, text: Optional[str]):
    """Ask, don't guess: بنسأل إيجار ولا تمليك، ولو الرد مش مفهوم بنسأل مرة كمان.
    لو لسه مش واضح بيفضل unknown (مش بنحط إيجار من عندنا): البحث بيعرض كل الشقق، والرد بيقول إنها كلها إيجار."""
    for question in (ASK_DEAL, ASK_DEAL_AGAIN):
        answer = interrupt(question)
        answer = answer.strip() if isinstance(answer, str) else ""
        if not answer:
            continue
        text = f"{text} - {answer}" if text else answer
        deal = deal_type_from(answer) or extract_requirements(raw_text=answer).deal_type
        if deal in ("rent", "buy"):
            req.deal_type = deal
            break
    return req, text


def deal_type_from(text: str) -> Optional[str]:
    """رد العميل على "إيجار ولا تمليك؟" من الكلمات نفسها (من غير ما نسأل الموديل)"""
    t = (text or "").replace("إ", "ا").replace("أ", "ا").lower()
    if any(w in t for w in ["ايجار", "اجار", "rent"]):
        return "rent"
    if any(w in t for w in ["تمليك", "شراء", "اشتري", "اشترى", "buy"]):
        return "buy"
    return None


MAX_CLARIFY_QUESTIONS = threshold("conversation.max_clarify_questions")
_HAS_NUMBER = re.compile(r"[0-9٠-٩۰-۹]|مليون|ألف|الف|واحد|اتنين|تلات|اربع|أربع|خمس|ست|سبع|تمان|تسع|عشر|مية|ميت")


def clarify_requirements(text: str, req: Optional[CustomerRequirements], check_text: bool = True):
    """Ask, don't guess: لو فيه حقل مش واضح ("70 6") أو برّه الحدود المنطقية، البوت بيسأل
    سؤال واحد قصير، ويعيد الاستخراج بالرد. لو الحقل لسه مش واضح بعدها، بيفضل فاضي (مش بنخمن)."""
    if req is None or not text:
        return text, req
    unclear = find_unclear(text if check_text else "", req)
    for _ in range(MAX_CLARIFY_QUESTIONS):
        if not unclear:
            break
        req = drop_guesses(req, unclear)
        field = next((f for f in FIELD_ORDER if f in unclear), next(iter(unclear)))
        answer = interrupt(clarification_question(unclear))
        answer = answer.strip() if isinstance(answer, str) else ""
        if not answer:
            break
        text = with_clarification(text, answer)
        new_req = extract_requirements(call_transcript=None, raw_text=text)
        unclear = find_unclear(answer, new_req)      # الرقمين القدام في الكلام الأصلي اتسألنا عنهم خلاص
        if not _HAS_NUMBER.search(answer):           # رد من غير أي رقم: أي قيمة للحقل ده تبقى تخمين
            unclear.setdefault(field, "no_answer:" + answer[:20])
        new_req.deal_type = new_req.deal_type if new_req.deal_type not in (None, "unknown") else req.deal_type
        req = new_req
    if unclear:
        req = drop_guesses(req, unclear)
    return text, req


ASK_BUDGET = "تمام 👌 ميزانيتك في حدود كام؟"


def ask_budget(req: Optional[CustomerRequirements], text: Optional[str]):
    """Ask, don't guess (missing): لو العميل مقالش ميزانية خالص، بنسأل سؤال واحد.
    لو رد برقم بنعيد الاستخراج بالرد؛ ولو مردّش برقم ("مش فارقة") بندوّر من غير ميزانية زي ما كان."""
    if req is None or req.budget_max is not None or req.budget_min is not None:
        return req, text
    answer = interrupt(ASK_BUDGET)
    answer = answer.strip() if isinstance(answer, str) else ""
    if not answer or not _HAS_NUMBER.search(answer):
        return req, text
    text = with_clarification(text or "", answer)
    new_req = extract_requirements(call_transcript=None, raw_text=text)
    if new_req.deal_type in (None, "unknown"):
        new_req.deal_type = req.deal_type
    unclear = find_unclear(answer, new_req)          # الرد نفسه مش واضح ("70 6") أو مش منطقي: منخمنش
    if unclear:
        new_req = drop_guesses(new_req, unclear)
    return new_req, text


def revalidate(req: Optional[CustomerRequirements]) -> Optional[CustomerRequirements]:
    """بعد ما نعرف إيجار ولا تمليك، حدود الميزانية بتتغير: نعدي على الـ schema تاني"""
    return CustomerRequirements(**req.model_dump()) if req is not None else None


def input_node(state: AgentState) -> Dict[str, Any]:
    
    print("🔥 ANA HENNA EL-GADEED 🔥")  # سطر مميز جداً عشان نتأكد إنه شايف الملف
    print("Execute Input Node...")
    
    micro_intent = state.get("micro_intent")
    print(f"Current Micro Intent is: {micro_intent}")
    
    # لو البحث جديد، نجبره يقف هنا وما يكملش تحت
    if micro_intent in ["new_search", "modify_search"]:
        print("🔄 [RESET] Force resetting in input_node...")

        # الرسالة نفسها فيها المواصفات ("عايز شقة 3 غرف في التجمع")؟ نستخدمها على طول،
        # ونسأله بس لو كتب حاجة زي "عايز أدور على شقة" من غير مواصفات (لحد 3 مرات)
        resumed_text = (state.get("user_query") or "").strip() or None
        req = extract_requirements(call_transcript=None, raw_text=resumed_text) if resumed_text else None
        for attempt in range(3):
            if has_requirements(req):
                break
            user_response = interrupt(ASK_SPECS if attempt == 0 else ASK_SPECS_AGAIN)
            resumed_text = user_response.strip() if isinstance(user_response, str) else None
            req = extract_requirements(call_transcript=None, raw_text=resumed_text) if resumed_text else None

        # الميزانية اتقالت بس مش واضحة؟ ده سؤال clarify، مش سؤال "ميزانيتك كام"
        budget_said = req is not None and bool(set(find_unclear(resumed_text or "", req)) & {"budget_min", "budget_max"})

        # فيه حاجة مش واضحة ("70 6") أو مش منطقية؟ نسأل سؤال واحد بدل ما نخمن
        resumed_text, req = clarify_requirements(resumed_text, req)

        # الميزانية ناقصة خالص؟ سؤال واحد، ولو مردّش برقم بندوّر من غير ميزانية
        if not budget_said:
            req, resumed_text = ask_budget(req, resumed_text)

        # والشقق مبتطلعش غير لما نعرف هو عايزها إيجار ولا تمليك
        if req is not None and req.deal_type in (None, "unknown"):
            req, resumed_text = ask_deal_type(req, resumed_text)
            req = revalidate(req)                       # ميزانية الإيجار غير التمليك
            resumed_text, req = clarify_requirements(resumed_text, req, check_text=False)

        return {
            "user_req_text": resumed_text,
            "user_query": resumed_text,
            "customer_reqs": req,
            "top_properties": [],
            "selected_property_id": None,
            "selected_property": None,
            "error_message": None
        }
    if (not state.get("customer_reqs") and not state.get("user_req_text")):
        user_response = interrupt("من فضلك، اكتب مواصفات الشقة المطلوبة (المنطقة، الميزانية، عدد الغرف):")
        
        # بعد الـ interrupt، يستأنف التنفيذ هنا
        resumed_text = user_response.strip() if isinstance(user_response, str) else None
        
        # نقوم بتحديث الـ raw_text مباشرة من النص المستأنف
        # لضمان أنه سيستخدم في بقية الكود أدناه
        raw_text = resumed_text
    else:
        # إذا لم يكن هناك interrupt، نحصل على الـ text العادي
        raw_text = (state.get("user_req_text") or "").strip() or None

    input_type = state.get("user_input_type", "text")
    audio_path = state.get("user_req_audio_path")
    # raw_text = (state.get("user_req_text") or "").strip() or None
    print("Input Node execution")

    conversation_text = audio_to_conversation(audio_path) if audio_path else None

    if not any([conversation_text, raw_text]):
        print("No inputss here")
        return {
            # "conversation_text": None,1
            "customer_reqs": None,
            # "missing_fields": [],
            "error": "Please provide at least one input: a recorded call or text.",
        }
    safe_print(f"📥 Extracting requirements from Text: '{raw_text}' | Audio: {conversation_text is not None}")
    req = extract_requirements(call_transcript=conversation_text, raw_text=raw_text)
    raw_text, req = clarify_requirements(raw_text or conversation_text or "", req)
    print("Here is reqs heereeeee")

    return {
        "user_req_text": raw_text,
        "user_query": raw_text,
        "customer_reqs": req,
        "error_message": None
    }
