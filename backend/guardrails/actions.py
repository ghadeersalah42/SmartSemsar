"""actions.py: No false claims of actions.

البوت ميقولش "اتحجز" أو "اتبعت على تليجرام" غير لو الأداة نفسها رجعت success.

  - action_reply(action, result): الرسالة اللي بتتعرض للعميل بعد أي أداة، ومبنية على نتيجتها بس.
  - claims_action(text): الرد فيه ادعاء إن حاجة اتعملت ("اتحجز"، "تم الإرسال"، "booked"...)؟
  - strip_unconfirmed_claims(text, confirmed): لو رد الموديل بيدعي حاجة الأداة مأكدتهاش، بيتشال.
"""
from __future__ import annotations

import re
from typing import Any, Iterable, Optional

# كل أداة: (رسالة النجاح، رسالة الفشل)
ACTION_MESSAGES = {
    "booking": ("✅ تم حجز الموعد بنجاح وإضافته إلى Google Calendar الخاص بك!",
                "❌ الموعد ما اتحجزش: {error}"),
    "telegram_report": ("✅ تم إرسال التقرير لتليجرام بنجاح!",
                        "❌ التقرير ما اتبعتش على تليجرام: {error}"),
    "telegram_link": ("✅ تم ربط حسابك بـ Smart Semsar، تقرير الحجز هيوصلك هنا.",
                      "❌ الحساب ما اتربطش: {error}"),
}


def succeeded(result: Any) -> bool:
    """النجاح بييجي من الأداة بس: dict فيه success == True بالظبط"""
    return isinstance(result, dict) and result.get("success") is True


def action_reply(action: str, result: Optional[dict]) -> str:
    ok_msg, fail_msg = ACTION_MESSAGES[action]
    if succeeded(result):
        return ok_msg
    error = (result or {}).get("error") if isinstance(result, dict) else None
    return fail_msg.format(error=error or "حصلت مشكلة، جرّب تاني")


# جمل بتدعي إن حاجة اتعملت فعلاً (حجز / إرسال)
_CLAIM_PATTERNS = {
    "booking": [r"(?:تم|اتم)\s*(?:ال)?حجز", r"اتحجز(?:ت|لك)?", r"حجزت(?:لك|لك)?\s", r"حجزنا",
                r"(?:موعد|ميعاد)\s*(?:ال)?معاينة\s*(?:اتأكد|تأكد|اتحدد)", r"\bbooked\b",
                r"booking\s+(?:is\s+)?confirmed", r"\b(?:et|it)?7agaz\b"],
    "telegram_report": [r"(?:تم|اتم)\s*(?:ال)?(?:إرسال|ارسال)", r"اتبعت(?:لك)?", r"بعتلك", r"بعتنالك",
                        r"\bsent\b", r"\b(?:et|it)?ba3at\b"],
}


def claims_action(text: str) -> list[str]:
    """أنواع الأفعال اللي الرد بيدعي إنها اتعملت"""
    text = text or ""
    return [action for action, pats in _CLAIM_PATTERNS.items()
            if any(re.search(p, text, re.IGNORECASE) for p in pats)]


NEUTRAL_REPLY = "تمام 👌 قولي محتاج إيه وأنا أساعدك."


def strip_unconfirmed_claims(text: str, confirmed: Iterable[str] = ()) -> str:
    """رد الموديل لو بيدعي حجز أو إرسال الأداة مأكدتهوش، بيتبدل برد محايد"""
    confirmed = set(confirmed)
    if any(a not in confirmed for a in claims_action(text)):
        return NEUTRAL_REPLY
    return text
