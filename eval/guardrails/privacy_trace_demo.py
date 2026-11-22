"""Privacy: trace واحد قبل الإخفاء وبعده.

    python eval/guardrails/privacy_trace_demo.py

  - من غير LangSmith: بيكتب privacy_before_after.md فيه نفس الـ trace مرة زي ما كان بيتبعت ومرة بعد mask_data.
  - لو LANGSMITH_API_KEY موجود: بيبعت trace-ين حقيقيين على مشروع SmartSemsar-privacy-demo
    ("before-masking" و "after-masking")، فتقدروا تاخدوا screenshot من LangSmith للاتنين.
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from backend.guardrails.pii import mask_data, mask_pii, register_name, _mask_with_path  # noqa: E402

HERE = Path(__file__).resolve().parent

SAMPLE_TRACE = {
    "inputs": {
        "user_query": "أنا اسمي منى السيد ورقمي ٠١٠٢٣٤٥٦٧٨٩، عايزة شقة 3 أوض في التجمع بميزانية 4 مليون",
        "booking": {"customer_name": "منى السيد", "customer_phone": "01023456789",
                    "notes": "كلميني على 0100 234 5678 بعد الساعة 5"},
    },
    "outputs": {
        "customer_reqs": {"location": "التجمع", "bedrooms": 3, "budget_max": 4000000, "deal_type": "unknown"},
        "agent_response": "تمام يا منى، لقيتلك 4 شقق في التجمع 👇",
    },
}


def main():
    register_name("منى السيد")               # زي ما الفورم بيعمل لما العميل يكتب اسمه
    after = mask_data(json.loads(json.dumps(SAMPLE_TRACE, ensure_ascii=False)))
    md = ["# Privacy: one trace before and after masking", "",
          "## Before (what was sent to LangSmith and printed in logs)", "```json",
          json.dumps(SAMPLE_TRACE, ensure_ascii=False, indent=2), "```", "",
          "## After (`backend/guardrails/pii.py`)", "```json",
          json.dumps(after, ensure_ascii=False, indent=2), "```", "",
          "Log line after masking:", "```",
          mask_pii(f"guard blocked message (off_topic): {SAMPLE_TRACE['inputs']['user_query']}"), "```"]
    (HERE / "privacy_before_after.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))

    if os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY"):
        send_to_langsmith()
    else:
        print("\nLANGSMITH_API_KEY not set: wrote the before/after file only.")


def send_to_langsmith():
    import time
    import uuid
    from langsmith import Client, traceable, tracing_context
    from langsmith.anonymizer import create_anonymizer
    project = "SmartSemsar-privacy-demo"
    plain, masked = Client(), Client(anonymizer=create_anonymizer(_mask_with_path))

    def step(inputs):
        return SAMPLE_TRACE["outputs"]

    ids = {"before-masking": uuid.uuid4(), "after-masking": uuid.uuid4()}
    # traceable مبيبعتش حاجة غير لو التتبع شغال (LANGSMITH_TRACING=true)، فبنشغله هنا بنفسنا
    with tracing_context(enabled=True, project_name=project):
        for (name, run_id), client in zip(ids.items(), (plain, masked)):
            traceable(name=name, client=client, project_name=project)(step)(
                SAMPLE_TRACE["inputs"], langsmith_extra={"run_id": run_id})
    plain.flush()
    masked.flush()

    # منقولش "اتبعت" غير لما LangSmith نفسه يأكد إنه استلم الاتنين
    found, last_error = {}, None
    for _ in range(15):
        for name, run_id in ids.items():
            if name not in found:
                try:
                    found[name] = plain.read_run(run_id).inputs
                except Exception as e:   # لسه موصلش، أو المفتاح/السيرفر غلط
                    last_error = e
        if len(found) == len(ids):
            break
        time.sleep(2)

    if len(found) == len(ids):
        print(f"\nLangSmith confirmed both traces in project '{project}':")
        for name, inputs in found.items():
            print(f"  {name}: user_query = {inputs.get('inputs', inputs).get('user_query')}")
    else:
        print(f"\nCould NOT confirm the traces in LangSmith (got {sorted(found) or 'none'}).")
        print(f"Last error: {type(last_error).__name__}: {str(last_error)[:300]}")
        print("Check the API key, and set LANGSMITH_ENDPOINT if your account is on the EU server.")
        sys.exit(1)


if __name__ == "__main__":
    main()
