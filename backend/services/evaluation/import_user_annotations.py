"""Import user-provided Ground Truth annotations for smartsemsar_rec1..smartsemsar_rec10.

- Preserves the user's existing manual human Arabic transcripts verbatim for `rec1`
  and `rec8`, adding the exact waveform start/end timestamps from the attached files.
- Populates `rec2`..`rec7`, `rec9`, `rec10` directly from the attached user files.
- Uses canonical neutral speaker IDs (`SPK_1` = client, `SPK_2` = broker) consistently
  across `transcripts/`, `diarization/`, and `roles/`, matching the official schema.
- Preserves `[Unclear]` markers and `review_note` flags in segment metadata while
  stripping only literal `[Unclear]` tags from the flat `reference_transcript` string
  used for WER/CER so ASR is not penalized against tag syntax.
- Never touches `raw_calls/` or `pilot/` files.
"""
from __future__ import annotations

import json
from pathlib import Path
import re
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.services.pipeline import EVALUATION_DIR, save_json  # noqa: E402

GT_DIR = EVALUATION_DIR / "ground_truth"
GT_TRANSCRIPTS_DIR = GT_DIR / "transcripts"
GT_DIARIZATION_DIR = GT_DIR / "diarization"
GT_ROLES_DIR = GT_DIR / "roles"
PILOT_DIR = EVALUATION_DIR / "pilot"


def _parse_ts(ts: str) -> float:
    """Convert HH:MM:SS.mmm to float seconds."""
    h, m, s = ts.strip().split(":")
    return round(int(h) * 3600 + int(m) * 60 + float(s), 3)


def _clean_for_reference_transcript(text: str) -> str:
    """Remove literal [Unclear] markers and bracket notation for flat WER/CER reference."""
    cleaned = re.sub(r"\[Unclear\]", "", text, flags=re.IGNORECASE)
    cleaned = re.sub(r"\[([^\]]+)\]", r"\1", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _role_to_canonical(speaker_label: str) -> tuple[str, str]:
    s = speaker_label.strip().lower()
    if s in ("customer", "client"):
        return "SPK_1", "client"
    if s == "broker":
        return "SPK_2", "broker"
    raise ValueError(f"Unexpected speaker label: {speaker_label!r}")


# Exact user-attached dataset for rec1..rec10
USER_ATTACHED_CALLS: list[dict[str, Any]] = [
    {
        "call_id": "rec1",
        "source_file": "smartsemsar_rec1.mp4",
        "file_duration_sec": 33.855,
        "utterances": [
            {"speaker": "Customer", "start": "00:00:00.000", "end": "00:00:01.160", "text": "في التجمع الخامس", "boundary": "silence_edge", "review_note": 'التسجيل بيبدأ والكلام شغال (أول الجملة مقطوع غالباً)؛ تمريرة بتسمعها "فتجمع مواجهة"'},
            {"speaker": "Broker", "start": "00:00:01.970", "end": "00:00:03.460", "text": "آه، عايزة إيجار ولا تمليك؟", "boundary": "silence_edge", "review_note": 'تمريرة 1 كتبتها بالفصحى "أريد"؛ اتاخد النطق العامي من التمريرتين التانيين'},
            {"speaker": "Customer", "start": "00:00:04.550", "end": "00:00:06.450", "text": "إيجار، لا، تمليك", "boundary": "approx_speaker_change", "review_note": "العميل غيّر رأيه من إيجار لتمليك"},
            {"speaker": "Broker", "start": "00:00:06.450", "end": "00:00:08.600", "text": "تمليك؟ طب، ميزانيتك كم؟", "boundary": "approx_speaker_change", "review_note": ""},
            {"speaker": "Customer", "start": "00:00:09.840", "end": "00:00:11.300", "text": "حوالي 4 مليون", "boundary": "silence_edge", "review_note": ""},
            {"speaker": "Broker", "start": "00:00:12.140", "end": "00:00:14.420", "text": "4 مليون في التجمع، أي كم غرفة؟", "boundary": "silence_edge", "review_note": "فيه وقفة قصيرة 13.460–13.720"},
            {"speaker": "Customer", "start": "00:00:15.620", "end": "00:00:16.410", "text": "ثلاث غرف [Unclear]", "boundary": "silence_edge", "review_note": 'فيه كلمة بعد "غرف" مش واضحة'},
            {"speaker": "Customer", "start": "00:00:16.900", "end": "00:00:17.480", "text": "مش صغيرة", "boundary": "silence_edge", "review_note": 'تمريرة بتسمعها "مش صغير"'},
            {"speaker": "Broker", "start": "00:00:18.260", "end": "00:00:19.060", "text": "مساحتها كم؟", "boundary": "silence_edge", "review_note": "ثقة منخفضة"},
            {"speaker": "Customer", "start": "00:00:20.220", "end": "00:00:21.000", "text": "ثلاث غرف يا فندم", "boundary": "silence_edge", "review_note": "ثقة متوسطة"},
            {"speaker": "Broker", "start": "00:00:22.020", "end": "00:00:23.210", "text": "المساحة [Unclear] المساحة؟", "boundary": "silence_edge", "review_note": "الكلمة اللي في النص مش واضحة"},
            {"speaker": "Customer", "start": "00:00:24.670", "end": "00:00:27.860", "text": "ماعرفش المساحة بالظبط يا [Unclear] بس 3 أوض", "boundary": "silence_edge", "review_note": 'الكلمة بعد "يا" مختلف عليها بين التمريرات (فندم/خالد/خانة)'},
            {"speaker": "Customer", "start": "00:00:28.200", "end": "00:00:29.410", "text": "وفي التجمع الخامس", "boundary": "silence_edge", "review_note": ""},
            {"speaker": "Customer", "start": "00:00:29.940", "end": "00:00:32.080", "text": "والميزانية حوالي 4 مليون", "boundary": "silence_edge", "review_note": ""},
            {"speaker": "Broker", "start": "00:00:32.370", "end": "00:00:33.250", "text": "طب، جميل جداً", "boundary": "silence_edge", "review_note": ""},
        ],
        "requirements": {"location": "التجمع الخامس", "property_type": "Not Specified", "rooms": 3, "transaction_type": "Sale", "budget": 4000000, "currency": "EGP", "area_sqm": None},
        "requirements_notes": [
            'العميل قال "إيجار" وبعدين رجع وقال "لا، تمليك" → اتاخد آخر رأي: تمليك (Sale).',
            'العميل ماقالش شقة ولا فيلا → Property Type = Not Specified.',
            'العميل قال صراحة "ماعرفش المساحة بالظبط" → Area = Not Specified.',
        ],
    },
    {
        "call_id": "rec2",
        "source_file": "smartsemsar_rec2.mp4",
        "file_duration_sec": 31.347,
        "utterances": [
            {"speaker": "Broker", "start": "00:00:00.130", "end": "00:00:04.310", "text": "كنت حابب أعرف إيه المواصفات اللي في [Unclear]، منطقة، عدد الغرف", "boundary": "silence_edge", "review_note": 'تمريرة 1 بتسمع الكلمة "دماغك"، والتانيين مختلفين عليها'},
            {"speaker": "Customer", "start": "00:00:04.780", "end": "00:00:07.640", "text": "محتاجة حاجة في العاصمة الإدارية", "boundary": "silence_edge", "review_note": ""},
            {"speaker": "Broker", "start": "00:00:08.610", "end": "00:00:14.530", "text": "تمام، العاصمة الإدارية، كم غرفة تقريباً؟ إيجار ولا تمليك؟ والميزانية في حدود كام؟", "boundary": "silence_edge", "review_note": "فيه وقفة قصيرة 11.300–11.550"},
            {"speaker": "Customer", "start": "00:00:15.150", "end": "00:00:15.800", "text": "امم", "boundary": "silence_edge", "review_note": "صوت تردد"},
            {"speaker": "Customer", "start": "00:00:16.870", "end": "00:00:17.740", "text": "ثلاث أوض", "boundary": "silence_edge", "review_note": ""},
            {"speaker": "Customer", "start": "00:00:18.810", "end": "00:00:19.840", "text": "[Unclear]", "boundary": "silence_edge", "review_note": "فيه كلام بس مش مفهوم (التمريرات مختلفة تماماً)"},
            {"speaker": "Customer", "start": "00:00:20.680", "end": "00:00:21.220", "text": "تمليك", "boundary": "silence_edge", "review_note": ""},
            {"speaker": "Customer", "start": "00:00:21.580", "end": "00:00:22.630", "text": "اتنين ونص مليون", "boundary": "silence_edge", "review_note": 'اتاخدت من نافذة السياق؛ التمريرة الأولى كتبتها "2.5 مليون"'},
            {"speaker": "Broker", "start": "00:00:23.650", "end": "00:00:28.660", "text": "ماشي، شقة 3 غرف تمليك في العاصمة الإدارية حوالي 2.5 مليون جنيه", "boundary": "silence_edge", "review_note": ""},
            {"speaker": "Broker", "start": "00:00:29.090", "end": "00:00:31.140", "text": "شكراً لاتصال حضرتك، في أمان الله", "boundary": "silence_edge", "review_note": ""},
        ],
        "requirements": {"location": "العاصمة الإدارية", "property_type": "Not Specified", "rooms": 3, "transaction_type": "Sale", "budget": 2500000, "currency": "EGP", "area_sqm": None},
        "requirements_notes": [
            'العميل قال "محتاجة حاجة" من غير ما يحدد شقة ولا فيلا. كلمة "شقة" قالها السمسار بس في الملخص → Not Specified.',
            "مقطع العميل 18.810–19.840 مش مفهوم ومش داخل في الاستخراج.",
        ],
    },
    {
        "call_id": "rec3",
        "source_file": "smartsemsar_rec3.mp4",
        "file_duration_sec": 31.463,
        "utterances": [
            {"speaker": "Broker", "start": "00:00:00.000", "end": "00:00:02.470", "text": "مكتب عقارات، اتفضل", "boundary": "silence_edge", "review_note": "التسجيل بيبدأ والكلام شغال؛ جت من نافذة السياق بس (التمريرات التانية مطلعتش حاجة مفهومة)"},
            {"speaker": "Customer", "start": "00:00:04.450", "end": "00:00:06.380", "text": "آه، بدور على حاجة في الشيخ زايد", "boundary": "silence_edge", "review_note": ""},
            {"speaker": "Broker", "start": "00:00:07.520", "end": "00:00:10.200", "text": "تمام يا فندم، حضرتك شقة ولا فيلا؟", "boundary": "silence_edge", "review_note": ""},
            {"speaker": "Customer", "start": "00:00:10.860", "end": "00:00:12.020", "text": "شقة، تبقى غرفتين", "boundary": "silence_edge", "review_note": ""},
            {"speaker": "Broker", "start": "00:00:13.150", "end": "00:00:14.300", "text": "تمام، اتنين أوض", "boundary": "silence_edge", "review_note": "ثقة متوسطة"},
            {"speaker": "Broker", "start": "00:00:14.570", "end": "00:00:17.100", "text": "في الشيخ زايد، حضرتك إيجار ولا تمليك؟", "boundary": "approx_speaker_change", "review_note": ""},
            {"speaker": "Customer", "start": "00:00:17.100", "end": "00:00:18.790", "text": "إيجار. لو قلت إيجار", "boundary": "approx_speaker_change", "review_note": "الصوتين الاتنين ستات، فالفصل بين المتكلمين اتعمل من المعنى والتوقيت"},
            {"speaker": "Broker", "start": "00:00:19.490", "end": "00:00:22.630", "text": "إيجار. اتفضل يا فندم، حضرتك في حدود كام؟", "boundary": "silence_edge", "review_note": ""},
            {"speaker": "Customer", "start": "00:00:23.140", "end": "00:00:24.510", "text": "لو قلت حوالي", "boundary": "silence_edge", "review_note": ""},
            {"speaker": "Customer", "start": "00:00:25.100", "end": "00:00:26.220", "text": "خمسة وعشرين ألف", "boundary": "silence_edge", "review_note": ""},
            {"speaker": "Broker", "start": "00:00:26.440", "end": "00:00:27.260", "text": "تمام يا فندم", "boundary": "silence_edge", "review_note": ""},
            {"speaker": "Broker", "start": "00:00:27.510", "end": "00:00:31.460", "text": "يبقى شقة غرفتين في الشيخ زايد إيجار حوالي خمسة وعشرين ألف", "boundary": "silence_edge", "review_note": "الكلام مستمر لحد آخر الملف"},
        ],
        "requirements": {"location": "الشيخ زايد", "property_type": "Apartment", "rooms": 2, "transaction_type": "Rent", "budget": 25000, "currency": "EGP", "area_sqm": None},
        "requirements_notes": ['الميزانية شهري (إيجار)؛ العميل ماقالش "في الشهر" صراحة.'],
    },
    {
        "call_id": "rec4",
        "source_file": "smartsemsar_rec4.mp4",
        "file_duration_sec": 39.636,
        "utterances": [
            {"speaker": "Broker", "start": "00:00:00.010", "end": "00:00:02.520", "text": "[Unclear] الخير يا فندم، مع حضرتك مكتب عقارات", "boundary": "silence_edge", "review_note": 'أول كلمة مش واضحة (يمكن "مساء")'},
            {"speaker": "Broker", "start": "00:00:02.770", "end": "00:00:03.450", "text": "كنت بتدوري عليه؟", "boundary": "silence_edge", "review_note": ""},
            {"speaker": "Customer", "start": "00:00:04.070", "end": "00:00:06.380", "text": "أنا بدور على شقة في التجمع الخامس", "boundary": "silence_edge", "review_note": ""},
            {"speaker": "Broker", "start": "00:00:07.750", "end": "00:00:10.450", "text": "تمام يا فندم، التجمع الخامس، كم أوضة تقريباً؟", "boundary": "silence_edge", "review_note": "فيه وقفات 8.280–8.710 و9.310–9.730"},
            {"speaker": "Customer", "start": "00:00:10.910", "end": "00:00:13.300", "text": "ثلاث غرف، مش عايزة أصغر", "boundary": "approx_speaker_change", "review_note": 'الموديل كتبها "أصغير"'},
            {"speaker": "Broker", "start": "00:00:13.300", "end": "00:00:18.310", "text": "ماشي، ثلاثة؟ آه، طيب، ده إيجار ولا تمليك؟", "boundary": "approx_speaker_change", "review_note": "فيه وقفات جوه الجملة (14.180–14.880، 15.140–15.460، 15.740–16.280)"},
            {"speaker": "Customer", "start": "00:00:19.120", "end": "00:00:22.930", "text": "إيجار، بفضّل يعني عشرين ألف", "boundary": "silence_edge", "review_note": "فيه وقفات 20.370–20.810 و22.170–22.390"},
            {"speaker": "Broker", "start": "00:00:24.580", "end": "00:00:26.020", "text": "تمام، في حدود كم مثلاً؟", "boundary": "silence_edge", "review_note": ""},
            {"speaker": "Customer", "start": "00:00:26.590", "end": "00:00:27.570", "text": "حدود عشرين ألف", "boundary": "silence_edge", "review_note": ""},
            {"speaker": "Broker", "start": "00:00:28.910", "end": "00:00:31.270", "text": "طيب، شقة إيجار في التجمع الخامس", "boundary": "silence_edge", "review_note": "فيه وقفة 29.900–30.170"},
            {"speaker": "Broker", "start": "00:00:31.910", "end": "00:00:34.520", "text": "ثلاث غرف، ميزانية حوالي 20 ألف [Unclear]", "boundary": "silence_edge", "review_note": 'الكلمة الأخيرة الموديل كتبها "ساعة" — مش واضحة'},
            {"speaker": "Customer", "start": "00:00:34.890", "end": "00:00:35.100", "text": "صح", "boundary": "silence_edge", "review_note": "مقطع قصير جداً؛ نسبته للعميل ثقتها منخفضة"},
            {"speaker": "Broker", "start": "00:00:36.360", "end": "00:00:39.570", "text": "تمام، شكراً لاتصال حضرتك يا فندم [Unclear]", "boundary": "silence_edge", "review_note": 'الكلمة الأخيرة مش واضحة (الموديل كتبها "أسجلت")'},
        ],
        "requirements": {"location": "التجمع الخامس", "property_type": "Apartment", "rooms": 3, "transaction_type": "Rent", "budget": 20000, "currency": "EGP", "area_sqm": None},
        "requirements_notes": ['"مش عايزة أصغر" من 3 غرف → اتسجلت rooms = 3 (الحد الأدنى).'],
    },
    {
        "call_id": "rec5",
        "source_file": "smartsemsar_rec5.mp4",
        "file_duration_sec": 29.396,
        "utterances": [
            {"speaker": "Broker", "start": "00:00:00.000", "end": "00:00:08.400", "text": "[Unclear] الخير يا فندم، مع حضرتك مكتب عقارات، لو حضرتك بتدوري على فيلا تمليك في التجمع، خمس غرف ومساحتها حوالي 180 متر", "boundary": "silence_edge", "review_note": 'أول كلمة مش واضحة ("مرحباً"/"صباح")'},
            {"speaker": "Broker", "start": "00:00:08.720", "end": "00:00:12.600", "text": "ممكن أعرف ميزانيتك في حدود كام تقريباً عشان أظبطلك اختيارات مناسبة؟", "boundary": "silence_edge", "review_note": ""},
            {"speaker": "Customer", "start": "00:00:13.080", "end": "00:00:13.920", "text": "12 مليون", "boundary": "silence_edge", "review_note": "ده الكلام الوحيد اللي العميل قاله في التسجيل"},
            {"speaker": "Broker", "start": "00:00:15.410", "end": "00:00:20.720", "text": "تمام يا فندم، فيلا خمس غرف حوالي 180 متر في التجمع بميزانية 12 مليون", "boundary": "silence_edge", "review_note": ""},
            {"speaker": "Broker", "start": "00:00:21.140", "end": "00:00:27.990", "text": "هشوفلك شوية اختيارات مناسبة، ممكن أحددلك ميعاد للزيارة أو أبعتلك التفاصيل على الواتساب؟", "boundary": "silence_edge", "review_note": "مفيش رد من العميل لحد آخر الملف"},
        ],
        "requirements": {"location": "Not Specified", "property_type": "Not Specified", "rooms": None, "transaction_type": "Not Specified", "budget": 12000000, "currency": "EGP", "area_sqm": None},
        "requirements_notes": [
            "مهم: العميل قال الميزانية بس (12 مليون).",
            "اللي قاله السمسار ومش متأكد من العميل في التسجيل ده: فيلا، تمليك، التجمع، 5 غرف، حوالي 180 متر.",
        ],
    },
    {
        "call_id": "rec6",
        "source_file": "smartsemsar_rec6.mp4",
        "file_duration_sec": 33.739,
        "utterances": [
            {"speaker": "Broker", "start": "00:00:00.040", "end": "00:00:00.810", "text": "مكتب عقارات", "boundary": "silence_edge", "review_note": ""},
            {"speaker": "Broker", "start": "00:00:01.360", "end": "00:00:02.860", "text": "اتفضلي، كنت بتدوري عليه؟", "boundary": "approx_speaker_change", "review_note": ""},
            {"speaker": "Customer", "start": "00:00:02.860", "end": "00:00:05.560", "text": "بدور على شقة في مدينة نصر", "boundary": "approx_speaker_change", "review_note": ""},
            {"speaker": "Broker", "start": "00:00:05.560", "end": "00:00:06.040", "text": "[Unclear] يا فندم", "boundary": "approx_speaker_change", "review_note": 'التمريرات بتسمع أول كلمة "8"/"ثمانية"، وغالباً هي "تمام"'},
            {"speaker": "Customer", "start": "00:00:07.230", "end": "00:00:08.150", "text": "أربع غرف", "boundary": "approx_speaker_change", "review_note": "الصوتين ستات؛ نسبة المقطع ده للعميل ثقتها منخفضة"},
            {"speaker": "Broker", "start": "00:00:08.150", "end": "00:00:08.900", "text": "كم أوضة؟", "boundary": "approx_speaker_change", "review_note": ""},
            {"speaker": "Customer", "start": "00:00:09.260", "end": "00:00:09.760", "text": "أربعة", "boundary": "approx_speaker_change", "review_note": ""},
            {"speaker": "Broker", "start": "00:00:09.760", "end": "00:00:10.900", "text": "تمام، حضرتك [Unclear]", "boundary": "approx_speaker_change", "review_note": ""},
            {"speaker": "Broker", "start": "00:00:11.450", "end": "00:00:11.920", "text": "أربعة؟", "boundary": "silence_edge", "review_note": ""},
            {"speaker": "Broker", "start": "00:00:12.590", "end": "00:00:13.040", "text": "ماشي", "boundary": "silence_edge", "review_note": ""},
            {"speaker": "Broker", "start": "00:00:13.330", "end": "00:00:16.160", "text": "تمليك ولا إيجار؟ والميزانية؟", "boundary": "approx_speaker_change", "review_note": "فيه وقفة 14.500–15.060"},
            {"speaker": "Customer", "start": "00:00:16.160", "end": "00:00:16.620", "text": "إيجار", "boundary": "approx_speaker_change", "review_note": ""},
            {"speaker": "Broker", "start": "00:00:17.990", "end": "00:00:20.200", "text": "إيجار، والميزانية تقريباً في حدود كام؟", "boundary": "approx_speaker_change", "review_note": ""},
            {"speaker": "Customer", "start": "00:00:20.200", "end": "00:00:20.860", "text": "8000", "boundary": "approx_speaker_change", "review_note": ""},
            {"speaker": "Broker", "start": "00:00:22.910", "end": "00:00:24.360", "text": "8000، تمام يا فندم", "boundary": "silence_edge", "review_note": ""},
            {"speaker": "Broker", "start": "00:00:24.800", "end": "00:00:29.140", "text": "شقة 4 غرف في مدينة نصر إيجار بميزانية حوالي 8000 جنيه", "boundary": "silence_edge", "review_note": ""},
            {"speaker": "Broker", "start": "00:00:31.270", "end": "00:00:33.350", "text": "شكراً [Unclear]", "boundary": "silence_edge", "review_note": ""},
        ],
        "requirements": {"location": "مدينة نصر", "property_type": "Apartment", "rooms": 4, "transaction_type": "Rent", "budget": 8000, "currency": "EGP", "area_sqm": None},
        "requirements_notes": ['عدد الغرف (4) العميل قاله بوضوح: "أربعة" (9.260–9.760).'],
    },
    {
        "call_id": "rec7",
        "source_file": "smartsemsar_rec7.mp4",
        "file_duration_sec": 29.931,
        "utterances": [
            {"speaker": "Broker", "start": "00:00:00.000", "end": "00:00:01.480", "text": "[Unclear] يا فندم، اتفضل", "boundary": "silence_edge", "review_note": "التسجيل بيبدأ والكلام شغال"},
            {"speaker": "Customer", "start": "00:00:02.510", "end": "00:00:04.220", "text": "أنا محتاج فيلا في 6 أكتوبر", "boundary": "silence_edge", "review_note": 'تمريرة 1 كتبتها "أحتاج"؛ نافذة السياق بتسمعها "محتاج"'},
            {"speaker": "Broker", "start": "00:00:05.770", "end": "00:00:08.570", "text": "تمام يا فندم، كم غرف تقريباً؟", "boundary": "approx_speaker_change", "review_note": "فيه وقفة 6.870–7.170"},
            {"speaker": "Customer", "start": "00:00:08.570", "end": "00:00:09.240", "text": "4 غرف", "boundary": "approx_speaker_change", "review_note": ""},
            {"speaker": "Broker", "start": "00:00:10.470", "end": "00:00:13.420", "text": "طيب، شراء ولا إيجار؟", "boundary": "silence_edge", "review_note": 'تمريرة 1 كتبتها "أم"؛ التانية "ولا"'},
            {"speaker": "Customer", "start": "00:00:13.820", "end": "00:00:14.570", "text": "شراء طبعاً", "boundary": "silence_edge", "review_note": ""},
            {"speaker": "Broker", "start": "00:00:15.690", "end": "00:00:17.620", "text": "تمام، والميزانية في حدود كم؟", "boundary": "approx_speaker_change", "review_note": "فيه وقفة 16.130–16.420"},
            {"speaker": "Customer", "start": "00:00:17.620", "end": "00:00:20.130", "text": "8 مليون جنيه، 10 مليون جنيه", "boundary": "approx_speaker_change", "review_note": ""},
            {"speaker": "Broker", "start": "00:00:21.280", "end": "00:00:28.500", "text": "تمام يا فندم، خلاص، فيلا 4 غرف في أكتوبر، تمليك، ميزانية من 8 إلى 10 مليون جنيه", "boundary": "silence_edge", "review_note": ""},
            {"speaker": "Broker", "start": "00:00:28.850", "end": "00:00:29.930", "text": "شكراً لاتصال حضر[تك]", "boundary": "silence_edge", "review_note": "الكلام مقطوع عند آخر الملف"},
        ],
        "requirements": {"location": "6 أكتوبر", "property_type": "Villa", "rooms": 4, "transaction_type": "Sale", "budget": {"min": 8000000, "max": 10000000}, "currency": "EGP", "area_sqm": None},
        "requirements_notes": ["الميزانية مدى: 8 إلى 10 مليون جنيه.", '"شراء" = Sale.'],
    },
    {
        "call_id": "rec8",
        "source_file": "smartsemsar_rec8.mp4",
        "file_duration_sec": 24.892,
        "utterances": [
            {"speaker": "Broker", "start": "00:00:00.320", "end": "00:00:04.840", "text": "اتفضلي، حضرتك بتدوري على حاجة إيجار في منطقة معينة؟", "boundary": "silence_edge", "review_note": "فيه وقفات قصيرة 0.780–1.060، 3.120–3.430، 3.830–4.130"},
            {"speaker": "Customer", "start": "00:00:05.200", "end": "00:00:07.610", "text": "آه، بدور على إيجار مصر الجديدة", "boundary": "silence_edge", "review_note": ""},
            {"speaker": "Broker", "start": "00:00:08.890", "end": "00:00:10.640", "text": "تمام، مصر الجديدة، كم أوضة يا فندم [Unclear]؟", "boundary": "silence_edge", "review_note": "فيه وقفة 9.770–10.170"},
            {"speaker": "Customer", "start": "00:00:11.370", "end": "00:00:12.660", "text": "آه، غرفتين", "boundary": "silence_edge", "review_note": ""},
            {"speaker": "Broker", "start": "00:00:13.410", "end": "00:00:14.780", "text": "الغرفتين، والميزانية؟", "boundary": "silence_edge", "review_note": ""},
            {"speaker": "Customer", "start": "00:00:15.150", "end": "00:00:15.820", "text": "15 ألف", "boundary": "silence_edge", "review_note": ""},
            {"speaker": "Broker", "start": "00:00:17.080", "end": "00:00:17.650", "text": "خمستاشر ألف", "boundary": "silence_edge", "review_note": ""},
            {"speaker": "Broker", "start": "00:00:18.200", "end": "00:00:18.630", "text": "تمام يا فندم", "boundary": "silence_edge", "review_note": ""},
            {"speaker": "Broker", "start": "00:00:19.090", "end": "00:00:23.670", "text": "[Unclear] بنتكلم على شقة غرفتين في مصر الجديدة إيجار بميزانية حوالي خمستاشر ألف", "boundary": "silence_edge", "review_note": 'أول كلمة الموديل كتبها "حلص" (يمكن "خلاص")'},
            {"speaker": "Broker", "start": "00:00:24.270", "end": "00:00:24.890", "text": "شكراً [Unclear]", "boundary": "silence_edge", "review_note": "الكلام مقطوع عند آخر الملف"},
        ],
        "requirements": {"location": "مصر الجديدة", "property_type": "Not Specified", "rooms": 2, "transaction_type": "Rent", "budget": 15000, "currency": "EGP", "area_sqm": None},
        "requirements_notes": ['العميل قال "بدور على إيجار" من غير ما يقول شقة. كلمة "شقة" قالها السمسار بس في الملخص → Not Specified.'],
    },
    {
        "call_id": "rec9",
        "source_file": "smartsemsar_rec9.mp4",
        "file_duration_sec": 32.345,
        "utterances": [
            {"speaker": "Broker", "start": "00:00:00.280", "end": "00:00:04.160", "text": "اتفضلي يا فندم، محتاجة مساعدة في إيه بالظبط في العقارات؟", "boundary": "silence_edge", "review_note": ""},
            {"speaker": "Customer", "start": "00:00:04.710", "end": "00:00:07.570", "text": "أو، أنا محتاجة شقة في 6 أكتوبر", "boundary": "silence_edge", "review_note": ""},
            {"speaker": "Broker", "start": "00:00:08.540", "end": "00:00:09.900", "text": "تمام، شقة في أكتوبر", "boundary": "approx_speaker_change", "review_note": "ثقة منخفضة — التمريرات مختلفة"},
            {"speaker": "Customer", "start": "00:00:09.900", "end": "00:00:10.440", "text": "تمليك", "boundary": "approx_speaker_change", "review_note": ""},
            {"speaker": "Broker", "start": "00:00:11.620", "end": "00:00:12.940", "text": "تمليك، صح؟", "boundary": "silence_edge", "review_note": ""},
            {"speaker": "Customer", "start": "00:00:12.980", "end": "00:00:13.380", "text": "آه", "boundary": "silence_edge", "review_note": ""},
            {"speaker": "Broker", "start": "00:00:14.100", "end": "00:00:16.360", "text": "طيب، كم غرفة تقريباً؟", "boundary": "silence_edge", "review_note": ""},
            {"speaker": "Customer", "start": "00:00:16.460", "end": "00:00:17.880", "text": "إيه، تلت غرف", "boundary": "silence_edge", "review_note": ""},
            {"speaker": "Broker", "start": "00:00:18.740", "end": "00:00:21.040", "text": "تلت غرف، حلو، والميزانية؟", "boundary": "silence_edge", "review_note": ""},
            {"speaker": "Customer", "start": "00:00:21.180", "end": "00:00:22.780", "text": "آه، خمسة مليون جنيه", "boundary": "silence_edge", "review_note": ""},
            {"speaker": "Customer", "start": "00:00:23.640", "end": "00:00:25.480", "text": "حوالي خمسة مليون جنيه", "boundary": "silence_edge", "review_note": "نسبته للعميل ثقتها منخفضة"},
            {"speaker": "Broker", "start": "00:00:25.560", "end": "00:00:30.080", "text": "تمام يا فندم، شقة ثلاث غرف تمليك في أكتوبر حوالي خمسة مليون", "boundary": "silence_edge", "review_note": ""},
            {"speaker": "Broker", "start": "00:00:30.200", "end": "00:00:32.040", "text": "شكراً لاتصال حضرتك", "boundary": "silence_edge", "review_note": ""},
        ],
        "requirements": {"location": "6 أكتوبر", "property_type": "Apartment", "rooms": 3, "transaction_type": "Sale", "budget": 5000000, "currency": "EGP", "area_sqm": None},
        "requirements_notes": ["التسجيل فيه ضوضاء خلفية مستمرة (أرضية الضوضاء حوالي −28 dB)، فالتوقيتات من توقيت الكلمات (±0.2 ث)."],
    },
    {
        "call_id": "rec10",
        "source_file": "smartsemsar_rec10.mp4",
        "file_duration_sec": 43.352,
        "utterances": [
            {"speaker": "Broker", "start": "00:00:00.010", "end": "00:00:09.400", "text": "[Unclear] مع حضرتك مكتب عقارات، كنت حابب أعرف من حضرتك بتدوري على إيجار ولا تمليك، وإيه البجت أو الميزانية اللي في بالك؟", "boundary": "approx_speaker_change", "review_note": "أول كلمة مش واضحة؛ فيه وقفة 1.650–2.170"},
            {"speaker": "Customer", "start": "00:00:09.400", "end": "00:00:12.970", "text": "عايزة حاجة تمليك، معايا ميزانية 6 مليون", "boundary": "approx_speaker_change", "review_note": "فيه وقفة 11.050–11.420"},
            {"speaker": "Broker", "start": "00:00:14.030", "end": "00:00:17.430", "text": "6 مليون، طيب، كم غرفة في [Unclear] حضرتك؟", "boundary": "silence_edge", "review_note": "فيه وقفات 14.740–15.540"},
            {"speaker": "Customer", "start": "00:00:18.430", "end": "00:00:20.240", "text": "إيه؟ بتقول إيه؟", "boundary": "silence_edge", "review_note": ""},
            {"speaker": "Broker", "start": "00:00:21.170", "end": "00:00:21.980", "text": "كم غرفة؟", "boundary": "silence_edge", "review_note": ""},
            {"speaker": "Customer", "start": "00:00:22.300", "end": "00:00:23.830", "text": "كم غرفة؟ ثلاث غرف", "boundary": "silence_edge", "review_note": ""},
            {"speaker": "Broker", "start": "00:00:24.840", "end": "00:00:25.410", "text": "ثلاث غرف", "boundary": "silence_edge", "review_note": ""},
            {"speaker": "Broker", "start": "00:00:26.240", "end": "00:00:29.730", "text": "طب، في منطقة معينة في مصر في بالك؟", "boundary": "silence_edge", "review_note": ""},
            {"speaker": "Customer", "start": "00:00:29.950", "end": "00:00:30.860", "text": "أكتوبر مثلاً", "boundary": "silence_edge", "review_note": ""},
            {"speaker": "Customer", "start": "00:00:31.660", "end": "00:00:32.730", "text": "الشيخ زايد", "boundary": "silence_edge", "review_note": ""},
            {"speaker": "Broker", "start": "00:00:33.880", "end": "00:00:34.650", "text": "زايد، حاضر", "boundary": "silence_edge", "review_note": ""},
            {"speaker": "Broker", "start": "00:00:35.000", "end": "00:00:38.590", "text": "ثلاث غرف تمليك في زايد أو أكتوبر", "boundary": "silence_edge", "review_note": "فيه وقفة 36.650–37.260"},
            {"speaker": "Broker", "start": "00:00:38.960", "end": "00:00:40.800", "text": "والميزانية بحدود ستة مليون", "boundary": "silence_edge", "review_note": ""},
            {"speaker": "Broker", "start": "00:00:41.200", "end": "00:00:42.870", "text": "تمام يا فندم؟ شكراً لاتصال حضرتك", "boundary": "silence_edge", "review_note": ""},
        ],
        "requirements": {"location": ["أكتوبر", "الشيخ زايد"], "property_type": "Not Specified", "rooms": 3, "transaction_type": "Sale", "budget": 6000000, "currency": "EGP", "area_sqm": None},
        "requirements_notes": ['العميل ذكر منطقتين ("أكتوبر مثلاً" و"الشيخ زايد") → اتسجلوا الاتنين.', 'العميل قال "عايزة حاجة" من غير ما يحدد النوع → Not Specified.'],
    },
]


def _build_rec1_segments() -> list[dict[str, Any]]:
    """Preserve user's manual rec1 transcript verbatim, enriched with exact waveform timestamps."""
    pilot_path = PILOT_DIR / "rec1_human_transcript.json"
    with pilot_path.open("r", encoding="utf-8-sig") as f:
        manual_turns = json.load(f)["conversation"]
    # Exact waveform timestamps mapped 1-to-1 to the user's 12 manual turns
    ts_pairs = [
        (0.000, 1.160, "silence_edge"),
        (1.970, 3.460, "silence_edge"),
        (4.550, 6.450, "approx_speaker_change"),
        (6.450, 8.600, "approx_speaker_change"),
        (9.840, 11.300, "silence_edge"),
        (12.140, 14.420, "silence_edge"),
        (15.620, 17.480, "silence_edge"),
        (18.260, 19.060, "silence_edge"),
        (20.220, 21.000, "silence_edge"),
        (22.020, 23.210, "silence_edge"),
        (24.670, 32.080, "silence_edge"),
        (32.370, 33.250, "silence_edge"),
    ]
    assert len(manual_turns) == len(ts_pairs)
    segments = []
    for turn, (start_s, end_s, boundary) in zip(manual_turns, ts_pairs):
        spk_id, role = _role_to_canonical(turn["speaker"])
        segments.append({
            "start": start_s,
            "end": end_s,
            "speaker_id": spk_id,
            "role": role,
            "text": turn["text"],
            "boundary": boundary,
        })
    return segments


def _build_rec8_segments() -> list[dict[str, Any]]:
    """Preserve user's manual rec8 transcript verbatim, enriched with exact waveform timestamps."""
    pilot_path = PILOT_DIR / "rec8_human_transcript.json"
    with pilot_path.open("r", encoding="utf-8-sig") as f:
        manual_turns = json.load(f)["conversation"]
    # Exact waveform timestamps mapped 1-to-1 to the user's 7 manual turns
    ts_pairs = [
        (0.320, 4.840, "silence_edge"),
        (5.200, 7.610, "silence_edge"),
        (8.890, 10.640, "silence_edge"),
        (11.370, 12.660, "silence_edge"),
        (13.410, 14.780, "silence_edge"),
        (15.150, 15.820, "silence_edge"),
        (17.080, 23.670, "silence_edge"),
    ]
    assert len(manual_turns) == len(ts_pairs)
    segments = []
    for turn, (start_s, end_s, boundary) in zip(manual_turns, ts_pairs):
        spk_id, role = _role_to_canonical(turn["speaker"])
        segments.append({
            "start": start_s,
            "end": end_s,
            "speaker_id": spk_id,
            "role": role,
            "text": turn["text"],
            "boundary": boundary,
        })
    return segments


def run_import() -> None:
    manifest_entries = []
    requirements_ref = {}

    for call in USER_ATTACHED_CALLS:
        short_id = call["call_id"]
        full_id = f"smartsemsar_{short_id}"

        if short_id == "rec1":
            segments = _build_rec1_segments()
            annotator = "manual_human_transcript_with_waveform_timestamps"
            provenance = (
                "Preserves user's manual human transcript (pilot/rec1_human_transcript.json & "
                "prior rec1 GT) verbatim; updated turn boundaries to millisecond waveform timestamps."
            )
        elif short_id == "rec8":
            segments = _build_rec8_segments()
            annotator = "manual_human_transcript_with_waveform_timestamps"
            provenance = (
                "Preserves user's manual human transcript (pilot/rec8_human_transcript.json) verbatim; "
                "enriched with millisecond waveform timestamps from user-provided annotation file. "
                "Final clipped EOF word 'شكراً [Unclear]' at 24.270-24.890s noted in metadata."
            )
        else:
            segments = []
            for utt in call["utterances"]:
                spk_id, role = _role_to_canonical(utt["speaker"])
                seg: dict[str, Any] = {
                    "start": _parse_ts(utt["start"]),
                    "end": _parse_ts(utt["end"]),
                    "speaker_id": spk_id,
                    "role": role,
                    "text": utt["text"],
                    "boundary": utt["boundary"],
                }
                if utt.get("review_note"):
                    seg["review_note"] = utt["review_note"]
                segments.append(seg)
            annotator = "user_provided_whisper_large_v3_multipass_waveform_aligned"
            provenance = (
                "Imported from user-provided multi-pass Whisper large-v3 + waveform-aligned "
                "annotation package; preserves [Unclear] markers and low-confidence review notes "
                "for optional final human spot-check."
            )

        # Clean flat reference transcript (strips literal [Unclear] tags while keeping exact tags in segments)
        ref_parts = [_clean_for_reference_transcript(s["text"]) for s in segments]
        ref_transcript = " ".join(p for p in ref_parts if p).strip()

        # 1. Transcript GT
        tr_payload = {
            "schema_version": "1.0",
            "call_id": full_id,
            "status": "ready",
            "language": "ar",
            "annotator": annotator,
            "provenance": provenance,
            "reference_transcript": ref_transcript,
            "segments": segments,
        }
        save_json(GT_TRANSCRIPTS_DIR / f"{full_id}_gt_transcript.json", tr_payload)

        # 2. Diarization GT
        diar_turns = [
            {
                "start": s["start"],
                "end": s["end"],
                "speaker_id": s["speaker_id"],
                "role": s["role"],
                "boundary": s.get("boundary"),
            }
            for s in segments
        ]
        diar_payload = {
            "schema_version": "1.0",
            "call_id": full_id,
            "status": "ready",
            "annotator": annotator,
            "provenance": provenance,
            "num_speakers": 2,
            "turns": diar_turns,
        }
        save_json(GT_DIARIZATION_DIR / f"{full_id}_gt_diarization.json", diar_payload)

        # 3. Roles GT
        roles_payload = {
            "schema_version": "1.0",
            "call_id": full_id,
            "status": "ready",
            "annotator": annotator,
            "provenance": provenance,
            "allowed_roles": ["broker", "client"],
            "notes": "Canonical reference speaker IDs: SPK_1=client, SPK_2=broker.",
            "speaker_roles": {
                "SPK_1": "client",
                "SPK_2": "broker",
            },
            "pipeline_speaker_to_reference_speaker": {},
        }
        save_json(GT_ROLES_DIR / f"{full_id}_gt_roles.json", roles_payload)

        manifest_entries.append({
            "call_id": full_id,
            "raw_audio": f"backend/services/data/raw_calls/{full_id}.mp4",
            "ground_truth_transcript": f"backend/services/data/evaluation/ground_truth/transcripts/{full_id}_gt_transcript.json",
            "ground_truth_diarization": f"backend/services/data/evaluation/ground_truth/diarization/{full_id}_gt_diarization.json",
            "ground_truth_roles": f"backend/services/data/evaluation/ground_truth/roles/{full_id}_gt_roles.json",
            "ground_truth_ready": True,
            "annotation_source": annotator,
            "status": "ready",
        })

        requirements_ref[full_id] = {
            "requirements_customer_only": call["requirements"],
            "requirements_notes": call["requirements_notes"],
        }

    # Update manifest.json
    manifest_payload = {
        "description": (
            "Evaluation manifest tracking ground-truth readiness for the 10 SmartSemsar real-estate phone calls. "
            "rec1 & rec8 preserve user manual human transcripts with waveform timestamps; "
            "rec2..rec7, rec9..rec10 were imported from user-provided multi-pass Whisper large-v3 + waveform files."
        ),
        "calls": manifest_entries,
    }
    save_json(GT_DIR / "manifest.json", manifest_payload)
    save_json(GT_DIR / "requirements_reference.json", requirements_ref)
    print(f"\nImported GT files for all {len(manifest_entries)} calls successfully.")


if __name__ == "__main__":
    run_import()
