import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from backend.agents.semsar_graph import the_Orchestrator
import json
from dotenv import load_dotenv, find_dotenv
import os

# يبحث عن ملف .env تلقائياً في الفولدرات الأب ويحمله
load_dotenv(find_dotenv())

# للتأكد من إن المفتاح اتشد صح
if not os.getenv("GROQ_API_KEY"):
    print("❌ Warning: GROQ_API_KEY is not loaded from .env!")
else:
    print("✅ GROQ_API_KEY successfully loaded.")



def inspect_state_delta(event: dict):
    """طباعة نواتج كل النودز اللي مرت عليها الرسالة"""
    for node_name, output in event.items():
        print(f"\n📍 [Executed Node]: {node_name}")

        # 1. طباعة النوايا المكتشفة
        if "macro_intent" in output and output["macro_intent"]:
            print(f"   🎯 Macro Intent : {output['macro_intent']}")
        if "micro_intent" in output and output["micro_intent"]:
            print(f"   🔬 Micro Intent : {output['micro_intent']}")

        # 2. طباعة الشروط المستخرجة (Customer Reqs)
        if "customer_reqs" in output and output["customer_reqs"]:
            reqs = output["customer_reqs"]
            reqs_dict = (
                reqs.model_dump() if hasattr(reqs, "model_dump") else reqs
            )
            print(
                f"   📝 Customer Reqs: {json.dumps(reqs_dict, ensure_ascii=False)}"
            )

        # 3. طباعة نتائج البحث
        if "top_properties" in output and output["top_properties"]:
            props = output["top_properties"]
            print(f"   🏢 Top Properties Found: {len(props)} items")
            if len(props) > 0:
                first_prop = props[0]
                prop_title = getattr(
                    first_prop, "title", first_prop.title
                )
                prop_id = getattr(
                    first_prop, "property_id", first_prop.property_id
                )
                print(f"      👉 Sample Prop 1: ID [{prop_id}] - {prop_title}")

        # 4. طباعة الشقة المختارة للـ 3D
        if "selected_property" in output and output["selected_property"]:
            selected = output["selected_property"]
            title = getattr(selected, "title", selected.title)
            print(f"   🎯 Selected Property for 3D: {title}")

        # 5. مرحلة الجراف الحالية
        if "current_stage" in output and output["current_stage"]:
            print(f"   📌 Current Stage: {output['current_stage']}")


def start_interactive_session():
    # session_id موحد للحفاظ على الذاكرة والتنقل التراكمي في المحادثة
    session_id = "interactive_cli_session"
    config = {"configurable": {"thread_id": session_id}}

    print("=" * 70)
    print("🚀 LangGraph Interactive Testing CLI")
    print("اكتبي رسالتك واضغطي Enter للتجربة.")
    print("اكتبي 'exit' أو 'quit' في أي وقت لإغلاق الجلسة.")
    print("=" * 70 + "\n")

    while True:
        try:
            # استقبال المدخلات من المستخدم مباشرة في الـ Terminal
            user_input = input("\n👤 You: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ["exit", "quit", "q"]:
                print("\n👋 Bye! Ending test session.")
                break

            # تجهيز الـ State المبدئية للرسالة الحالية
            state_input = {
                "session_id": session_id,
                "user_input_type": "text",
                "user_query": user_input,
                "user_text": user_input,
            }

            print("\n⚙️ Processing Graph execution...")
            # تشغيل الجراف وطباعة مسار النودز خطوة بخطوة
            for event in the_Orchestrator.stream(state_input, config=config):
                inspect_state_delta(event)

        except KeyboardInterrupt:
            print("\n👋 Interrupted. Session ended.")
            break
        except Exception as e:
            print(f"\n❌ Execution Error: {e}")


if __name__ == "__main__":
    start_interactive_session()


# from backend.agents.intent_detector import detect_intent  # (عدلي اسم الملف حسب اسم ملفك الحالي)

# if __name__ == "__main__":
#     print("🧪 بدء اختبار تحليل النوايا (Intent Detection Test)...\n")
    
#     # أمثلة لجمل مختلفة لاختبار النوايا
#     test_queries = [
#         "عايز اشوف شقق تانية",
#         "الشقة حلوة بس عايز سعر أقل شوية",
#         "كام سعر الشقة دي ومساحتها كام؟",
#         "ممكن أحجز زيارة للشقة بكره؟",
#         "عايز ابدأ بحث جديد في التجمع الخامس"
#     ]
    
#     for query in test_queries:
#         print(f"💬 الجملة التجريبية: '{query}'")
#         result = detect_intent(query)
#         print(f"🎯 النتيجة المكتشفة: {result['intent']}")
#         print(f"📊 الثقة (Confidence): {result['confidence']}")
#         print(f"💡 السبب: {result['reason']}")
#         print("-" * 50)