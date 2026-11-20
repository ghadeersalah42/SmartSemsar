"""clarify.py: "اسأل متخمنش" (Ask, don't guess).

لو حقل في طلب العميل ناقص أو مش واضح، البوت بيسأل سؤال واحد قصير بدل ما يختار تفسير لوحده.

    find_unclear(text, req) -> {"budget_max": "ambiguous:70 6"}
    clarification_question(unclear) -> "مش متأكد من الميزانية «70 6». تقصد كام بالظبط؟"

بنمسك حالتين:
  1) رقمين جنب بعض من غير وحدة ("ميزانيتي 70 6"): ممكن 6.7 مليون أو 706 ألف أو 70 ألف و6...
     أي ميزانية الموديل طلعها من الكلام ده تخمين، فبتتشال لحد ما العميل يوضح.
  2) قيمة برّه الحدود المنطقية (CustomerRequirements.unclear_fields من الـ schema).
"""
from __future__ import annotations

import re
from typing import Optional

from backend.schema.state import CustomerRequirements

_EASTERN = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")

# وحدات لو جت بعد الرقمين يبقى الكلام عن مساحة أو أوض، مش ميزانية ("2 3 اوض" = 2 أو 3 أوض)
_NON_MONEY_UNITS = (r"متر|م2|م²|مترمربع|sqm|m2|meter|غرف|غرفة|غرفه|اوض|أوض|اوضة|أوضة|اوضه|أوضه"
                    r"|room|rooms|bed|beds|bedroom|bedrooms|حمام|حمامات|دور|floor|سنة|سنين|شهر")
_MONEY_WORDS = r"مليون|ملايين|الف|ألف|آلاف|الاف|k|m|million|thousand|جنيه|ج"

PAIR_RE = re.compile(
    r"(?<![\d.,/])(?P<a>\d{1,3})\s+(?P<b>\d{1,3})(?![\d.,/])\s*(?P<unit>[^\s\d]+)?")


def normalize_digits(text: str) -> str:
    return (text or "").translate(_EASTERN)


def ambiguous_number_pairs(text: str) -> list[str]:
    """الرقمين اللي جنب بعض من غير وحدة توضحهم، زي "70 6" """
    found = []
    for m in PAIR_RE.finditer(normalize_digits(text)):
        unit = (m.group("unit") or "").lower()
        if unit and re.fullmatch(rf"(?:ب|و|ال)?(?:{_NON_MONEY_UNITS})", unit):
            continue
        a, b = int(m.group("a")), int(m.group("b"))
        # "2 3 مليون" = من 2 لـ 3 مليون (مدى واضح)، مش رقم ملخبط
        if unit and re.fullmatch(rf"(?:{_MONEY_WORDS})", unit) and a < b:
            continue
        found.append(f"{m.group('a')} {m.group('b')}")
    return found


def find_unclear(text: str, req: Optional[CustomerRequirements]) -> dict[str, str]:
    """الحقول اللي محتاجة سؤال: من الـ schema (برّه الحدود) ومن الكلام نفسه (رقمين ملخبطين)"""
    unclear = dict(getattr(req, "unclear_fields", None) or {})
    for pair in ambiguous_number_pairs(text):
        unclear.setdefault("budget_max", f"ambiguous:{pair}")
        break
    return unclear


def drop_guesses(req: CustomerRequirements, unclear: dict[str, str]) -> CustomerRequirements:
    """أي قيمة الموديل طلعها لحقل مش واضح بتتشال: منبنيش بحث على تخمين"""
    for field, reason in unclear.items():
        if field in ("budget_min", "budget_max") and reason.startswith("ambiguous:"):
            req.budget_min = req.budget_max = None
        elif hasattr(req, field):
            setattr(req, field, None)
        if field not in req.missing_fields:
            req.missing_fields.append(field)
    return req


FIELD_ORDER = ["budget_max", "budget_min", "bedrooms", "area_sqm"]


def clarification_question(unclear: dict[str, str]) -> Optional[str]:
    """سؤال واحد قصير عن أهم حقل مش واضح (الميزانية الأول)، أو None لو كله واضح"""
    if not unclear:
        return None
    field = next((f for f in FIELD_ORDER if f in unclear), next(iter(unclear)))
    kind, _, raw = unclear[field].partition(":")
    if field.startswith("budget"):
        if kind == "ambiguous":
            return f"مش متأكد من الميزانية «{raw}». تقصد كام بالظبط؟"
        return f"الميزانية «{raw}» مش واضحة. تقصد كام بالظبط بالجنيه؟"
    if field == "bedrooms":
        return f"عدد الأوض «{raw}» مش واضح. تقصد كام أوضة؟"
    if field == "area_sqm":
        return f"المساحة «{raw}» مش واضحة. تقصد كام متر؟"
    return "ممكن توضحلي طلبك أكتر؟"


def with_clarification(original: str, answer: str) -> str:
    """الكلام الأصلي ومعاه رد العميل، عشان الاستخراج يشوف الاتنين (الرد بيكسب)"""
    return f"{original}\nتوضيح العميل: {answer}"
