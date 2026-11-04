# مشروع سمسار العقارات الذكي (Smart Semsar)
## تقييم منظومة الصوت في المكالمات الحقيقية (Speech On Real Calls)
**إعداد متطلبات شركة فيكسيد مصر (Fixed Solutions) — نوفمبر 2026**

---

## 🌟 نظرة عامة (Overview)

تم تصميم وتطوير هذه الحزمة البرمجية لتنفيذ واختبار متطلبات شركة **فيكسيد مصر (Fixed Solutions)** على مشروع **سمسار العقارات الذكي (Smart Semsar)**، وتشمل:

1. **حساب معدلات الخطأ على مستوى الكلمات والحروف (WER & CER)** على تسجيلات هاتفية بالعامية المصرية في ظل بيئات ضوضاء مختلفة وقنوات اتصال هاتفية حقيقية (ITU-T G.712 Telephony Bandpass & Additive Noise).
2. **قياس دقة فصل وتحديد هوية السمسار (Broker) عن العميل (Client)** بدقة متناهية، واستخراج مصفوفة الالتباس (Confusion Matrix)، ومعدل خطأ تفريق المتحدثين (DER).
3. دعم كامل لمجموعة بيانات **MagicHub Egyptian Arabic Conversational Speech Corpus (ASR-EgArbCSC)**.
4. بنية برمجية معيارية (Modular & Production-Ready) خالية من الأخطاء والمسارات الصلبة (Hardcoded paths).

---

## 📁 هيكل المشروع (Project Structure)

```text
smart_semsar_speech/
├── README.md                          # الدليل الشامل باللغتين العربية والإنجليزية
├── requirements.txt                   # ملف الاعتماديات والمكتبات المختبرة
├── config.py                          # إعدادات المسارات، النماذج، ومصطلحات اللهجة المصرية
├── evaluate_asr.py                    # سكريبت تقييم WER و CER على الصوت المشوش
├── evaluate_diarization.py            # سكريبت تقييم فصل السمسار عن العميل ومصفوفة الالتباس
├── run_benchmark.py                   # مشغّل البنشمارك الموحّد وإنشاء التقارير الشاملة
├── smart_semsar_pipeline.ipynb        # دفتر Jupyter تفاعلي خالي من الأخطاء
├── src/                               # الوحدات البرمجية الأساسية
│   ├── __init__.py
│   ├── text_normalization.py          # معالجة وتطبيع العامية المصرية (أ، إ، آ، ة، ي، التشكيل، التطويل)
│   ├── asr_evaluator.py               # محرك حساب WER و CER بمحاذاة NIST SCLITE Levenshtein
│   ├── audio_processing.py            # توحيد الصوت، فلاتر الهاتف G.712، ومحاكاة الضوضاء
│   ├── vad_engine.py                  # كاشف النشاط الصوتي (Silero VAD مع بديل محلي سريع)
│   ├── diarization_engine.py          # تفريق المتحدثين (PyAnnote + Acoustic Clustering)
│   ├── role_separator.py              # محرك فصل وتحديد السمسار والعميل بناءً على مصطلحات العقارات
│   ├── diarization_evaluator.py       # حساب مقاييس DER، F1، الدقة، ومصفوفة الالتباس
│   ├── magichub_loader.py             # قارئ ومحلل بيانات كوربس ماجيك هب المصري
│   └── pipeline.py                    # خط المعالجة الذكي المتكامل للمكالمات
├── data/
│   ├── real_estate_calls/             # بيانات مكالمات تجريبية واقعية مع التفريغ الصوتي والأدوار
│   │   ├── audio/                     # ملفات صوت WAV نقية ومشوشة (16kHz PCM)
│   │   └── annotations/               # تفريغات معتمدة وتقسيم السمسار والعميل
│   ├── magichub_sample/               # عينات مهيأة بنفس نسق MagicHub ASR-EgArbCSC
│   └── outputs/
│       └── reports/                   # التقارير الناتجة (JSON, CSV, Markdown)
└── tests/                             # حزمة الاختبارات الآلية (Unit Tests)
    ├── test_normalization.py
    ├── test_metrics.py
    └── test_role_separation.py
```

---

## 🚀 التثبيت السريع (Quickstart & Installation)

### 1. إعداد البيئة وتثبيت المكتبات
```bash
# إنشاء بيئة افتراضية (موصى به)
python -m venv venv
source venv/bin/activate  # في لينكس / ماك
# أو في ويندوز: venv\Scripts\activate

# تثبيت الاعتماديات
pip install -r requirements.txt
```

> **ملاحظة لكروت الشاشة (CUDA GPU):**
> لتشغيل نماذج Whisper أو PyAnnote بأقصى سرعة على كروت Nvidia، ثبّت PyTorch المفعّل بـ CUDA:
> ```bash
> pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
> ```

---

## 📊 كيفية تشغيل التقييمات واستخراج التقارير

### 1. تقييم دقة التعرّف على الصوت (WER & CER Evaluation)
يقوم هذا السكريبت باختبار نموذج التعرّف على الصوت على المكالمات المسجلة تحت درجات ضوضاء مختلفة وقنوات هاتفية متعددة:
```bash
python evaluate_asr.py
```
**المخرجات:**
- جدول تفصيلي يوضح (WER, CER, Accuracy, Substitutions, Deletions, Insertions, Hits).
- محاذاة بصرية دقيقة كلمة بكلمة (Token-by-Token Levenshtein Alignment).
- حفظ التقرير آلياً في `data/outputs/reports/asr_noise_evaluation_report.json` و `.csv`.

### 2. تقييم فصل السمسار عن العميل (Broker vs. Client Separation)
يقوم هذا السكريبت باختبار دقة عزل كلام السمسار عن كلام العميل ومعدل خطأ تفريق المتحدثين (DER):
```bash
python evaluate_diarization.py
```
**المخرجات:**
- دقة تصنيف الأدوار على مستوى المكالمة (Call-Level Role Accuracy).
- دقة الفصل الموزونة زمنياً (Time-Weighted Separation Accuracy).
- مصفوفة الالتباس الكاملة (Confusion Matrix) بالثواني والنسب المئوية.
- درجات F1-Score المنفصلة للسمسار والعميل.
- حفظ التقرير في `data/outputs/reports/diarization_role_separation_report.json` و `.csv`.

### 3. تشغيل البنشمارك الموحّد الكامل (Master Benchmark Runner)
لتشغيل كافة الاختبارات دفعة واحدة وإصدار تقرير تنفيذي لشركة فيكسيد مصر:
```bash
python run_benchmark.py
```

### 4. تشغيل الاختبارات الآلية للتأكد من خلو الكود من المشاكل
```bash
python -m unittest discover tests
```

---

## 🗂️ الربط مع داتا MagicHub المصرية (ASR-EgArbCSC)

رابط مجموعة البيانات المعتمدة:
[MagicHub Egyptian Arabic Conversational Speech Corpus](https://magichub.com/datasets/egyptian-arabic-conversational-speech-corpus/)

### مواصفات الداتا:
- **الحجم:** 5.5 ساعة محادثات بالعامية المصرية (حوالي 250 ميجابايت).
- **التسجيل:** مكالمات ومحادثات عبر الهاتف المحمول في بيئات داخلية واقعية.
- **النسق:** ملفات `WAV` (16 kHz, 16-bit, Mono) وتفريغات نصية `TXT/TSV`.

### كيفية تشغيل البنشمارك على الداتا المحملة:
1. قم بفك ضغط ملف الكوربس المحمّل من موقع MagicHub.
2. شغّل البنشمارك مع تحديد مسار المجلد:
```bash
python evaluate_asr.py --annotations_dir /path/to/extracted_magichub
```
يقوم كلاس `MagicHubDatasetLoader` في مسار `src/magichub_loader.py` بقراءة وتنسيق الملفات تلقائياً.

---

## 🧠 المنهجية الرياضية والتقنية (Technical Methodology)

### 1. تطبيع النص العامي المصري (Egyptian Text Normalization):
- معالجة الألف بكافة أشكالها (`أ`, `إ`, `آ`, `ٱ` $\rightarrow$ `ا`).
- توحيد التاء المربوطة والهاء (`ة` $\rightarrow$ `ه`).
- توحيد الياء والألف المقصورة (`ى` $\rightarrow$ `ي`).
- إزالة علامات التشكيل، التنوين، الشدة، والسكون، والتطويل (`ـ`).
- توحيد الأرقام المشرقية (`٠-٩` $\rightarrow$ `0-9`).

### 2. حساب WER و CER:
$$\text{WER} = \frac{S + D + I}{N} = \frac{\text{Substitutions} + \text{Deletions} + \text{Insertions}}{\text{Reference Word Count}}$$
$$\text{CER} = \frac{S_c + D_c + I_c}{N_c} = \frac{\text{Char Substitutions} + \text{Char Deletions} + \text{Char Insertions}}{\text{Reference Character Count}}$$
نستخدم أوزان **NIST SCLITE** المعتمدة عالمياً ($S=4, D=3, I=3$) لمنع التداخل أثناء المحاذاة.

### 3. خوارزمية تمييز السمسار عن العميل (Role Separation Engine):
تعتمد على تحليل لغوي وسياقي لسوق العقارات المصري:
- **مؤشرات السمسار (Broker Indicators):** التعريف بالشركة، طرح أسئلة استكشاف الاحتياجات (الميزانية، طريقة السداد، الموقع، نوع الوحدة)، تقديم المشاريع، ومتابعة المعاينات والواتساب.
- **مؤشرات العميل (Client Indicators):** التعبير عن الرغبة في الشراء، تحديد سقف الميزانية، عدد الغرف، القرب من المدارس، والسؤال عن الأسعار والبدائل.

---

**فريق التطوير — مشروع سمسار العقارات الذكي (Smart Semsar)**
**جاهز للتسليم والتشغيل الفوري بنسبة 100%.**
