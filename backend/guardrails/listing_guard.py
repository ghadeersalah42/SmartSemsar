"""Poisoned listings: إعلان فيه أوامر للموديل ("ignore previous instructions") ميوصلش للعميل.

نص الإعلان (العنوان والعنوان الجغرافي) داتا، مش تعليمات. أي إعلان نصه فيه أوامر بيتشال من نتايج البحث
ويتسجل في الـ log، بنفس قواعد اختبار الهجوم بتاعة رسايل العميل (injection / jailbreak / طلب بيانات عملاء).
"""
from typing import Any, Iterable, List, Optional, Tuple

from backend.guardrails.input_guard import EXFILTRATION, INJECTION, JAILBREAK, _any, normalize
from backend.guardrails.pii import get_logger

log = get_logger("smartsemsar.listings")

TEXT_FIELDS = ("title", "location", "description")
# أوامر موجهة للموديل أو للصفحة جوه نص إعلان (مش مكانها في إعلان شقة أصلاً)
LISTING_ONLY = [
    r"<\s*/?\s*(?:script|iframe|img|svg|style)\b|javascript:|on\w+\s*=",
    r"\b(?:assistant|ai|bot|model|chatbot|llm)\s*[:,]|(?:يا\s+)?(?:بوت|موديل|مساعد)\s*[:،,]",
    r"(?:tell|say to|inform)\s+(?:the\s+)?(?:client|customer|user)|(?:قول|قولوا|بلغ|اقنع)\s+(?:لل)?(?:عميل|زبون)",
]


def listing_text(record: Any) -> str:
    parts = []
    for name in TEXT_FIELDS:
        v = record.get(name) if isinstance(record, dict) else getattr(record, name, None)
        if v not in (None, "") and v == v:
            parts.append(str(v))
    return " | ".join(parts)


def check_listing(record: Any) -> Optional[Tuple[str, str]]:
    """(النوع, القاعدة) لو الإعلان فيه أوامر، None لو نضيف"""
    text = normalize(listing_text(record))
    for category, rules in (("injection", INJECTION), ("jailbreak", JAILBREAK),
                            ("data_exfiltration", EXFILTRATION), ("instruction_in_listing", LISTING_ONLY)):
        rule = _any(rules, text)
        if rule:
            return category, rule
    return None


def quarantine(records: Iterable[Any]) -> Tuple[List[Any], List[Tuple[Any, str]]]:
    """(الإعلانات النضيفة, [(إعلان مشبوه, النوع)]). المشبوه مبيتعرضش ولا بيتبعت في تقرير."""
    clean, flagged = [], []
    for r in records:
        hit = check_listing(r)
        if hit:
            pid = r.get("property_id") if isinstance(r, dict) else getattr(r, "property_id", None)
            log.warning("listing %s removed from results (%s)", pid, hit[0])
            flagged.append((r, hit[0]))
        else:
            clean.append(r)
    return clean, flagged
