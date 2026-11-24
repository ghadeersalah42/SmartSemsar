# دليل تشغيل منظومة LangSmith LLMOps وإدارة إصدارات البرومبتات (Prompt Versioning)
## مشروع سمسار العقارات الذكي (Smart Semsar)

---

## 🌟 نظرة عامة على المتطلب (Task Overview)

تم دمج منظومة **LangSmith (التابعة لـ LangChain)** بالكامل في المشروع لتنفيذ متطلبات الـ LLMOps الاحترافية:
1. **إصدار كل برومبت بنظام نسخ معتمد (Semantic Versioning - SemVer)** لجميع وكلاء المشروع (Agents) عبر LangSmith Prompt Hub.
2. **وسم وتتبع كل استدعاء (Trace Tagging)** تلقائياً برقم إصدار البرومبت الفعلي واسم الموديل الذي أجاب (`prompt_version` و `model_name`) ضمن `metadata` و `tags` الخاصة بـ LangChain / LangSmith.
3. **التشغيل الآلي للتقييم المعياري (Automated Eval Set on Prompt Change)** بمجرد تعديل أو تحديث أي برومبت، باستخدام `langsmith.evaluate()`، ومقارنة النتائج بالـ Baseline لكشف التراجع (Regression Detection)، ومنع إطلاق أي برومبت يقلل من دقة النظام.

---

## 📁 هيكل التكامل داخل المشروع

```text
SmartSemsar-main/
├── langsmith_ops/                     # موديول LangSmith الأساسي
│   ├── config.py                      # إعدادات LangSmith والمتغيرات البيئية
│   ├── compat.py                      # جسر Dual-Mode (يعمل سحابياً ومحلياً بدون توقف)
│   ├── prompt_hub.py                  # إدارة وتخزين واستدعاء البرومبتات المصدرة
│   ├── tracer.py                      # Decorator و Tracing Tagging للموديل والنسخة
│   ├── eval_pipeline.py               # خط أنابيب التقييم الآلي عبر langsmith.evaluate()
│   ├── prompts/                       # البرومبتات المصدرة بنظام SemVer لكل وكيل
│   │   ├── intent_detector/           # v1.0.0, v1.1.0
│   │   ├── orchestrator_router/       # v1.0.0
│   │   └── customer_requirements/     # v1.0.0
│   ├── golden_datasets/               # مجموعات التقييم المعتمدة (Golden Datasets)
│   ├── traces/                        # سجلات التتبع المحلية والنسخ الاحتياطي
│   └── reports/                       # تقارير التقييم الناتجة بصيغ JSON و Markdown
├── backend/
│   ├── agents/
│   │   └── intent_detector.py         # وكيل اكتشاف النية متكامل مع التتبع والبرومبت
│   └── langsmith_ops/                 # رابط مباشر للموديول داخل الباك إند
├── .github/workflows/
│   └── langsmith_prompt_eval.yml      # CI/CD Pipeline لتشغيل التقييم تلقائياً عند تغيير البرومبتات
├── tests/
│   └── test_langsmith_llmops.py       # اختبارات الوحدة لمنظومة LangSmith بالكامل
├── demo_langsmith_integration.py      # تجربة عملية توضح التتبع وإدارة النسخ والتقييم
└── run_langsmith_eval.py              # أداة سطر الأوامر (CLI) لتشغيل التقييم واستعراض الـ Traces
```

---

## 🚀 التشغيل السريع (Quickstart)

### 1. المتطلبات والبيئة
تأكد من تثبيت الحزم المطلوبة:
```bash
pip install -r requirements.txt
```

### 2. التفعيل السحابي لـ LangSmith (اختياري / للإنتاج)
افتح ملف `.env` وأضف مفتاح الـ API الخاص بـ LangSmith من [LangSmith Console](https://smith.langchain.com/):
```env
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2_pt_your_actual_api_key_here
LANGCHAIN_PROJECT=smart-semsar
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
```
*ملاحظة هامة:* تم بناء المنظومة بتقنية **Dual-Mode** الذكية؛ بحيث إذا لم تتوفر مفاتيح سحابية أو انقطع الاتصال بالإنترنت، تعمل المنظومة داخلياً بسلاسة تامة 100% دون أي توقف أو استثناءات خطأ.

---

## 🛠️ أوامر التشغيل والتقييم (CLI Commands)

### 1. تشغيل الديمو التفاعلي الشامل
لتجربة سحب البرومبت من الـ Hub، وتنفيذ استدعاء الوكيل مع وسم الـ Traces، وتشغيل التقييم الآلي:
```bash
python demo_langsmith_integration.py
```

### 2. تشغيل التقييم الآلي لجميع الوكلاء (All Agents)
```bash
python run_langsmith_eval.py --all
```

### 3. تقييم وكيل محدد ومقارنته بإصدار سابق (Baseline)
```bash
python run_langsmith_eval.py --agent intent_detector --version v1.1.0 --baseline v1.0.0
```

### 4. استعراض أحدث الـ Traces المسجلة وتأكيد وسوم النسخة والموديل
```bash
python run_langsmith_eval.py --traces
```

### 5. تشغيل اختبارات الوحدة (Unit Tests)
```bash
python -m unittest tests/test_langsmith_llmops.py
```

---

## 💡 كيفية استخدام المنظومة في أي Agent جديد

```python
from langsmith_ops.prompt_hub import get_prompt_hub
from langsmith_ops.tracer import langsmith_trace, build_langchain_config

# 1. سحب البرومبت بنسخته المعتمدة من الـ Hub
hub = get_prompt_hub()
prompt_obj = hub.pull_prompt("my_new_agent", version="v1.0.0")

# 2. وسم الـ Trace تلقائياً برقم النسخة والموديل
@langsmith_trace(name="my_new_agent_call", prompt_version="v1.0.0", model_name="openai/gpt-oss-20b")
def run_my_agent(user_message: str):
    # إعداد LangChain RunnableConfig المتوافق مع LangSmith Cloud
    config = build_langchain_config("my_new_agent", "v1.0.0", "openai/gpt-oss-20b")
    # استدعاء الموديل وتمرير الـ config
    ...
```

---

## 🛡️ أتمتة الـ CI/CD (GitHub Actions)
يحتوي مسار `.github/workflows/langsmith_prompt_eval.yml` على فحص تلقائي يتم تشغيله فورياً عند عمل `push` أو `pull request` يمس مجلد البرومبتات `langsmith_ops/prompts/**`، حيث يُشغل التقييم ويمنع دمج أي كود يتسبب في انخفاض الدقة عن الـ Baseline المعتمد.
