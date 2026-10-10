from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from backend.schema.state import AgentState, PropertyItem
from typing import List, Optional
from dotenv import load_dotenv, find_dotenv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

class MacroIntentClassifier(BaseModel):
    macro_intent: str = Field(
        description=(
            "Must be strictly one of: "
            "'search_by_reqs', "
            "'sketch_based_search', "
            "'direct_sketch_staging', "
            "'general_browse', "
            "'request_sketch_upload'"
        )
    )
    reasoning: str = Field(description="سبب اختيار هذه النية باختصار")
    suggested_reply: str = Field(
        description="رد ترحيبي/توجيهي بالعامية المصرية يناسب نية العميل"
    )
    quick_actions: List[str] = Field(
        default_factory=lambda: [
            "🔍 بحث بمواصفات محددة",
            "📐 بحث عن شقة شبه السكيتش",
            "🛋️ فرش وتحويل السكيتش لـ 3D",
            "🏢 تصفح كل الشقق المتاحة"
        ],
        description="الخيارات الأربعة الرئيسية للـ UI"
    )

llm = ChatGroq(model_name="openai/gpt-oss-20b", temperature=0)
structured_orchestrator = llm.with_structured_output(MacroIntentClassifier)

ORCHESTRATOR_PROMPT = """
You are the primary Orchestrator (Supervisor Agent) for "Smart Semsar", an AI-powered PropTech platform in Egypt.
Your primary role is to classify the user's intent from their input (Text/Voice transcript in Egyptian Arabic or English) into ONE of the 4 core user paths, or ask for a sketch upload if missing.

### INPUT DATA:
- User Query: "{query}"
- Uploaded Sketch Image Present?: {has_sketch}

---

### CORE UI PATHS & INTENTS:

1. 'search_by_reqs' (بحث بمواصفات محددة)
   - Description: User specifies property criteria (location, budget, rooms, area, finishes).
   - Egyptian Slang/Phrases:
     * "بدور على شقة 3 غرف في التجمع"
     * "ميزانيتي 6 مليون وعايز حاجة في الشيخ زايد"
     * "عايز شقة تشطيب كامل المساحة مش أقل من 150 متر"
     * "محتاج مكتب إداري أو مقارات تجارية في العاصمة"
     * "عايز شقة رخيصة في الشروق أو التجمع الخامس"
   - Routing: Directs to input_node -> matching_node.

2. 'sketch_based_search' (بحث بشقة شبه السكيتش)
   - Description: User wants to search the property database for listings that match or look similar to a 2D floorplan/sketch.
   - Requirements: REQUIRES `has_sketch` = True.
   - Egyptian Slang/Phrases:
     * "معايا الرسمة دي وعايز شقق بنفس التقسيمة في الداتابيز"
     * "عندكم شقق شبه الكروكي ده؟"
     * "شوفلي شقة في الموقع متقسمة زي البلان ده"
     * "عايز أسرش على شقة نفس شكل الرسمة المرفقة"
   - Routing: Directs to sketch_to_svg_node -> matching_node.

3. 'direct_sketch_staging' (فرش وتحويل السكيتش لـ 3D)
   - Description: User wants to directly render, turn into 3D, and furnish/stage their specific 2D sketch/floorplan without database searching.
   - Requirements: REQUIRES `has_sketch` = True.
   - Egyptian Slang/Phrases:
     * "معايا السكيتش ده وعايز أفرشه وأشوفه ثري دي"
     * "حوللي الرسمة المرفقة لـ 3D staging"
     * "عايز أوزع العفش على البلان ده"
     * "فرشلي الشقة دي من الرسمة"
   - Routing: Directs to sketch_to_svg_node -> staging_3d_node.

4. 'general_browse' (تصفح عام / بيلف وخلاص)
   - Description: User has no specific preferences, wants to explore all listings, browse around, or just said a general greeting.
   - Egyptian Slang/Phrases:
     * "بصراحة بيلف وخلاص"
     * "عايز أتفرج على الشقق المتاحة عندكم"
     * "وريني أحدث العقارات المعروضة"
     * "سلام عليكم.. إيه الأخبار عندك إيه؟"
     * "عايز ألف في الموقع وأشوف الخيارات"
     * "وريني المتاح وخلاص"
   - Routing: Directs to matching_node (returns overall top listings).

---

### SPECIAL EDGE CASE: 'request_sketch_upload'
- TRIGGER CONDITION: User expressed intent for 'sketch_based_search' OR 'direct_sketch_staging' (mentions a sketch, drawing, floorplan, plan, or 3D staging), BUT `has_sketch` is False.
- Egyptian Expressions:
  * "معايا رسمة وعايز أفرشها" (without uploading file)
  * "عندي كروكي شقة عايز أشوف شبهها" (without uploading file)
  * "عايز أرفع سكتش"
- ACTION: Select 'request_sketch_upload'. In `suggested_reply`, warmly encourage them to attach/upload the sketch file now to proceed.

---

### DECISION RULES & PRIORITY:
1. IF the text is empty or generic AND `has_sketch` is False -> Classify as 'general_browse'.
2. IF `has_sketch` is False AND user mentions a sketch/drawing -> Classify as 'request_sketch_upload'.
3. IF `has_sketch` is True AND user mentions searching/matching -> Classify as 'sketch_based_search'.
4. IF `has_sketch` is True AND user mentions 3D/furnishing/staging -> Classify as 'direct_sketch_staging'.
5. Always generate `suggested_reply` in natural, polite Egyptian Arabic.
6. Always include the 4 core UI quick actions in `quick_actions`.

---

### FEW-SHOT EXAMPLES:

User Input: "أنا بيلف وخلاص عايز أشوف المتاح" | Sketch: False
-> Intent: 'general_browse'
-> Suggested Reply: "أهلاً بك! تقدر تتصفح أحدث العقارات المتاحة عندنا فوراً، ولو عجبك أي عقار هساعدك في باقي التفاصيل."

User Input: "معايا رسمة شقتي وعايز أفرشها ثري دي" | Sketch: False
-> Intent: 'request_sketch_upload'
-> Suggested Reply: "فكرة ممتازة! ياريت ترفع صورة السكيتش أو المخطط دلوقتي عشان أبدأ أحولهولك 3D وأفرشهولك فوراً 📐✨"

User Input: "معايا السكيتش ده (صورة مرفقة) وعايز شقة شبهه" | Sketch: True
-> Intent: 'sketch_based_search'
-> Suggested Reply: "تمام جداً! هبدأ أحول السكيتش المرفق وأبحثلك في الداتابيز عن شقق بنفس التقسيمة."

User Input: "بدور على شقة 160 متر في التجمع ميزانيتي 6 مليون" | Sketch: False
-> Intent: 'search_by_reqs'
-> Suggested Reply: "تمام! هجمع لك أفضل الشقق المتاحة في التجمع الخامسة بحدود 6 مليون."
"""

DEFAULT_QUICK_ACTIONS = [
    "🔍 بحث بمواصفات محددة",
    "📐 بحث عن شقة شبه السكيتش",
    "🛋️ فرش وتحويل السكيتش لـ 3D",
    "🏢 تصفح كل الشقق المتاحة"
]



def supervisor_router_node(state: AgentState) -> dict:
    print("\n🧠 [Supervisor Router] Evaluating macro entry point...")
    user_query = state.get("user_query") or ""
    current_state = state.get("current_stage") or ""
    

    if current_state=="property_results_displayed":
        return {
        "macro_intent": "search_by_reqs",
        "micro_intent": "",
        # "agent_response": result.suggested_reply,
    }
        
        
    has_sketch = bool(state.get("uploaded_sketch_path"))
    if not user_query:
        print("📌 Initial load: Fetching default properties catalog.")
        return {
            "macro_intent": "general_browse"
        }
    
    formatted_prompt = ORCHESTRATOR_PROMPT.format(
        query=user_query,
        has_sketch=has_sketch
    )

    result = structured_orchestrator.invoke(formatted_prompt)

    print(f"🎯 Detected Path: {result.macro_intent}")
    print(f"💡 Reason: {result.reasoning}")
    
    if result.macro_intent == "search_by_reqs":
        return {
        "macro_intent": result.macro_intent,
        "micro_intent": "new_search",
        "agent_response": result.suggested_reply,
        "quick_actions": DEFAULT_QUICK_ACTIONS  # إرجاع الأزرار الأربعة للـ UI دائماً
    }

    return {
        "macro_intent": result.macro_intent,
        "agent_response": result.suggested_reply,
        "quick_actions": DEFAULT_QUICK_ACTIONS  # إرجاع الأزرار الأربعة للـ UI دائماً
    }











    # current_stage = state.get("current_stage")
    

    # if state.get("selected_property_id") or state.get("micro_intent") == "select_property":
    #     print("📌 Direct button click: Jumping straight to property selection & 3D staging.")
    #     return {
    #         "macro_intent": "select_property",
    #         "micro_intent": "select_property"
    #     }
    
    

    # if current_stage == "property_results_displayed" and not has_sketch:
    #     print("📌 [Supervisor] User is replying to displayed properties -> Routing directly to Micro Intent.")
    #     return {
    #         "macro_intent": "search_by_reqs"
    #     }
    
    
  
    