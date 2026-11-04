# تطبيق LLMOps و Prompt Versioning باستخدام LangSmith
## مشروع سمسار العقارات الذكي (Smart Semsar) — نوفمبر 2026

---

## 🌟 نظرة عامة (Overview)

تم بناء هذه الحزمة خصيصاً لتنفيذ متطلبات الـ **LLMOps و Prompt Versioning** بالاعتماد المباشر على منصة **[LangSmith](https://smith.langchain.com/)** التابعة لمؤسسة **LangChain**، والمتوافقة بصورة طبيعية مع مكتبات `langchain` و `langgraph` المستخدمة في مشروع **Smart Semsar**.

تغطي المنظومة المتطلبين الأساسيين:
1. **إدارة إصدارات البرومبتات ووسم التتبع (Prompt Versioning & Trace Tagging):**
   - إصدار كل برومبت عبر **LangSmith Prompt Hub** (`v1.0.0`, `v1.1.0`) مع حفظ التغييرات والـ Commits.
   - وسم كل استدعاء (`Trace`) في LangSmith بالوسوم الإلزامية: رقم نسخة البرومبت (`prompt_version`) والموديل الذي أجاب (`model_name`) والزمن والتوكنز.
2. **التقييم المعياري الآلي عند كل تعديل (Automated Evals on Prompt Change):**
   - تشغيل دالة التقييم **`langsmith.evaluate()`** آلياً على المجموعات الذهبية (Golden Datasets).
   - مقارنة الإصدار الجديد بالنسخة السابقة على لوحة تحكم LangSmith لكشف أي تراجع في الأداء (Regression Detection).
   - توفير ملف سير عمل مؤتمت لـ **GitHub Actions** لتشغيل الاختبارات قبل دمج أي برومبت.

---

## 📁 هيكل الحزمة البرمجية (Package Architecture)

```text
smart_semsar_langsmith/
├── README.md                      # الدليل الشامل باللغتين العربية والإنجليزية
├── requirements.txt               # الاعتماديات (langsmith, pydantic, tabulate)
├── demo_langsmith_integration.py  # سكريبت عملي يعرض دورة العمل كاملة عبر LangSmith
├── run_langsmith_eval.py          # مشغل التقييم عبر سطر الأوامر واستعراض الـ Traces
│
├── langsmith_ops/                 # الموديول المركزي المتكامل مع LangSmith
│   ├── config.py                  # المتغيرات البيئية (LANGCHAIN_API_KEY, PROJECT)
│   ├── compat.py                  # جسر التوافق السحابي والمحلي (Dual-Mode Bridge)
│   ├── prompt_hub.py              # محرك إدارة وسحب ورفع البرومبتات (LangSmith Prompt Hub)
│   ├── tracer.py                  # وسم التتبع (@langsmith_trace و RunnableConfig)
│   └── eval_pipeline.py           # خط التقييم الآلي الشامل باستخدام langsmith.evaluate()
│
├── langsmith_ops/golden_datasets/ # مجموعات البيانات الذهبية المعتمدة للتقييم
│   ├── smart_semsar_intent_golden_set.json
│   ├── smart_semsar_orchestrator_golden_set.json
│   └── smart_semsar_customer_reqs_golden_set.json
│
├── langsmith_ops/prompts/         # نسخ البرومبتات المحفوظة محلياً والمزامنة مع الـ Hub
│   ├── intent_detector/           # v1.0.0 و v1.1.0
│   ├── orchestrator_router/       # v1.0.0
│   └── customer_requirements/     # v1.0.0
│
├── .github/workflows/             # ملف الأتمتة السحابية (CI/CD)
│   └── langsmith_prompt_eval.yml  # تشغيل الـ Evals تلقائياً عند تعديل أي برومبت
│
└── tests/                         # اختبارات الوحدة الآلية
    └── test_langsmith_llmops.py
```

---

## 🚀 التثبيت والتشغيل السريع (Quickstart)

### 1. تثبيت المتطلبات:
```bash
pip install -r requirements.txt
```

### 2. ضبط مفتاح LangSmith (اختياري للربط السحابي المباشر):
إذا كان لديكِ حساب ومفتاح على [LangSmith](https://smith.langchain.com/)، أضيفي المتغيرات الآتية في الـ Terminal أو داخل ملف `.env`:
```bash
export LANGCHAIN_TRACING_V2="true"
export LANGCHAIN_API_KEY="lsv2_pt_xxxxxxxxxxxxxxxxxxxx"
export LANGCHAIN_PROJECT="smart-semsar"
```
*(ملاحظة ذكية: إذا لم يتوفر مفتاح API، يعمل الكود تلقائياً في وضع Offline Dual-Mode المحلي بدون أي أخطاء أو توقف!)*

### 3. تشغيل المثال التوضيحي المتكامل:
```bash
python demo_langsmith_integration.py
```
يعرض هذا السكريبت:
- سحب برومبت موثق بنسخة `v1.1.0` من الـ Prompt Hub.
- تشغيل الوكيل وتوليد الـ Trace موسوماً بـ `prompt_version` و `model_name`.
- تشغيل التقييم الآلي المعياري عبر `langsmith.evaluate()` بنتيجة `PASS` ودقة 95%.

### 4. تشغيل التقييم المعياري لكافة وكلاء المنظومة:
```bash
python run_langsmith_eval.py
```

### 5. استعراض سجلات التتبع الموسومة بالموديل والنسخة:
```bash
python run_langsmith_eval.py --traces
```

### 6. تشغيل اختبارات الوحدة الآلية (Unit Tests):
```bash
python -m unittest discover tests
```

---

## 💡 كيفية الدمج في وكلاء مشروع Smart Semsar

### الخيار الأول: داخل دوال LangChain و LangGraph
عند استدعاء أي عقدة (Node) أو Chain في LangGraph، مرري الـ `config`:
```python
from langsmith_ops.tracer import build_langchain_config

config = build_langchain_config(
    agent_name="intent_detector",
    prompt_version="v1.1.0",
    model_name="openai/gpt-oss-20b"
)

# سيتم إرسال الـ Trace إلى سحابة LangSmith موسوماً بالنسخة والموديل تلقائياً
result = intent_chain.invoke({"conversation": text}, config=config)
```

### الخيار الثاني: باستخدام الـ Decorator المباشر
```python
from langsmith_ops.tracer import langsmith_trace

@langsmith_trace(agent_name="intent_detector", prompt_version="v1.1.0", default_model="openai/gpt-oss-20b")
def detect_intent(text: str):
    ...
```

---

## ⚙️ أتمتة الـ CI/CD (GitHub Actions)
الملف المرفق `.github/workflows/langsmith_prompt_eval.yml` يقوم تلقائياً بمراقبة أي تعديل يُجرى على ملفات البرومبتات في المستودع، ويشغل الـ Evals الخاصة بها في سحابة LangSmith، مع مقارنة الأداء بالنسخ السابقة لمنع أي تراجع (Regression) في بيئات الإنتاج.
