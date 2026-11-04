"""
Configuration Module for Smart Semsar (سمسار العقارات الذكي)
Fixed Solutions Project - Egyptian Arabic Real-Estate Call Intelligence
"""

import os
from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_CALLS_DIR = DATA_DIR / "real_estate_calls" / "audio"
ANNOTATIONS_DIR = DATA_DIR / "real_estate_calls" / "annotations"
MAGICHUB_SAMPLE_DIR = DATA_DIR / "magichub_sample"
OUTPUT_DIR = DATA_DIR / "outputs"
REPORTS_DIR = OUTPUT_DIR / "reports"

# Ensure all directories exist
for path in [
    DATA_DIR,
    RAW_CALLS_DIR,
    ANNOTATIONS_DIR,
    MAGICHUB_SAMPLE_DIR,
    OUTPUT_DIR,
    REPORTS_DIR,
]:
    path.mkdir(parents=True, exist_ok=True)

# Audio Standards
TARGET_SAMPLE_RATE = 16000  # 16 kHz
TARGET_CHANNELS = 1         # Mono
BIT_DEPTH = 16              # 16-bit PCM

# Telephony Channel Simulation (ITU-T G.712 standard for telephone calls)
TELEPHONY_BANDPASS_LOW_HZ = 300.0
TELEPHONY_BANDPASS_HIGH_HZ = 3400.0
SIMULATED_SNR_LEVELS = [None, 20.0, 15.0, 10.0, 5.0]  # None = Clean audio

# Whisper ASR Settings
WHISPER_MODEL_NAME = "large-v3-turbo"
WHISPER_FALLBACK_MODEL = "small"
WHISPER_LANGUAGE = "ar"
WHISPER_BEAM_SIZE = 5

# Diarization Settings
DIARIZATION_MODEL = "pyannote/speaker-diarization-community-1"
DIARIZATION_COLLAR_SECONDS = 0.25  # Collar for DER calculation (tolerance margin)
EXPECTED_SPEAKERS = 2              # Broker & Client

# Egyptian Real Estate Dialect Lexicons (for Role Separation: Broker vs. Client)
BROKER_INDICATORS = [
    # Introductions & Company representation
    "معاك", "مع حضرتك", "شركة", "المستشار العقاري", "مستشارك العقاري",
    "سمسار", "الوكيل العقاري", "فريق المبيعات", "مسؤول المبيعات", "بروكار",
    "أهلاً بحضرتك", "أقدر أساعدك", "تحت أمرك", "اتشرفت بيك",
    
    # Qualification & Customer Discovery
    "بتدور على إيه", "بتدور علي ايه", "الميزانية في حدود كام", "الميزانية كام",
    "عايزها في أي منطقة", "لوكيشن", "في التجمع ولا زايد", "العاصمة الإدارية",
    "الشروق", "المستقبل سيتي", "أكتوبر", "الساحل الشمالي",
    "سكني ولا استثماري", "غرض السكن", "غرض الاستثمار",
    "كاش ولا تقسيط", "جاهز للمعاينة", "يناسب حضرتك ميعاد",
    
    # Presentation of Inventory & Offers
    "عندنا مشروع", "في كمبوند", "المطور نزل طرح جديد", "طرح جديد",
    "مساحات تبدأ من", "سعر المتر", "سعر الوحدة", "نظام السداد", "أنظمة السداد",
    "المقدم", "دفعة التعاقد", "أقساط متساوية", "على 8 سنين", "على 7 سنين",
    "استلام فوري", "نصف تشطيب", "تشطيب كامل", "وديعة الصيانة", "النادي",
    "ممكن أبعت لحضرتك التفاصيل", "هبعتلك البروشور على الواتساب", "هتواصل مع حضرتك"
]

CLIENT_INDICATORS = [
    # Search Intent & Needs
    "أنا بدور على", "كنت بسأل عن", "عايز شقة", "عايزة شقة", "عايز فيلا", "عايز تاون هاوس",
    "محتاج شقة", "عاوز", "بدور على دوبلكس", "3 غرف", "غرفتين", "أوضتين", "تلات أوض",
    "مساحة في حدود", "تكون قريبة من المدارس", "قريبة من الجامعة", "فيو بحري",
    
    # Financial Constraints & Budget
    "ميزانيتي", "البادجت بتاعي", "البجتاتي", "معايا كاش", "أقصى حاجة معايا",
    "مليون", "خمسة مليون", "3 مليون", "4 مليون", "6 مليون", "10 مليون",
    "القسط الشهري كام", "المقدم كام", "مش هقدر أدفع أكتر من",
    
    # Objections, Inquiries & Preferences
    "المتر بكام هناك", "الاستلام إمتى", "الكمبوند ده ساكن؟", "مين المطور بتاعه؟",
    "مش عايز دور أرضي", "عايز دور أخير بروف", "في حديقة؟",
    "لا دي غالية شوية", "معندكش حاجة تانية؟", "وريني اختيارات تانية", "عايز أشوف بدائل",
    "هفكر وأرد عليك", "هستنى الواتساب", "رقمي هو"
]
