"""Fix known Whisper mishearings of Egyptian real-estate words before requirement extraction.

Whisper often mishears a few words that decide the extraction ("فيلا" as "بيلا"/"بلا", "التجمع" as
"التجامعة", "تمليك" as "تمليئك"/"دمليك"). These rules fix only those forms. They are general
Egyptian real-estate confusions (most are already listed in pipeline.ROLE_ASSIGNMENT_SYSTEM_PROMPT and
the role cue patterns), not words picked from the 10 test calls.

A word that is also a real Arabic word is only changed in real-estate context:
  "بلا" (= "without") becomes "فيلا" only next to a deal/size word or after "على/عايز/محتاج/فندم...".
  "اتجار" (= "trading") becomes "إيجار" only in "اتجار ولا/أو/لا ..." or after "عايز/ولا".

Only the machine transcript is corrected; the speaker roles, timestamps and WER inputs stay as they are.
"""
from __future__ import annotations

import re

_W = r"(?<![\w])"      # start of an Arabic word
_E = r"(?![\w])"       # end of an Arabic word

_NUMBER_WORD = r"(?:\d+|واحد|اتنين|اثنين|تلات|ثلاث|تلاتة|ثلاثة|اربع|أربع|اربعة|أربعة|خمس|خمسة|ست|ستة|سبع|سبعة)"
_DEAL_OR_SIZE = rf"(?:تمليك|ايجار|إيجار|غرف|غرفة|اوض|أوض|متر|دور|{_NUMBER_WORD})"
_LEAD_IN = r"(?:على|علي|عايز|عايزة|محتاج|محتاجة|فندم|خلاص|ماشي|طيب|يبقى|يبقي|نفسي)"

# (pattern, replacement, label) — order matters only where patterns overlap
RULES: list[tuple[re.Pattern, str, str]] = [
    (re.compile(_W + r"بيلا" + _E), "فيلا", "بيلا→فيلا"),
    (re.compile(rf"({_W}{_LEAD_IN}\s+)بلا{_E}"), r"\g<1>فيلا", "بلا→فيلا (after lead-in)"),
    (re.compile(rf"{_W}بلا(?=[\s،,]+{_DEAL_OR_SIZE}{_E})"), "فيلا", "بلا→فيلا (before deal/size)"),
    (re.compile(_W + r"(?:تمليئك|تمليئ|تمليل|دمليك|تمليق)" + _E), "تمليك", "تمليئك/دمليك→تمليك"),
    (re.compile(_W + r"(?:بيجار|ايجاد)" + _E), "إيجار", "بيجار→إيجار"),
    (re.compile(rf"{_W}اتجار(?=\s*[؟?]?\s*(?:ولا|او|أو|لا){_E})"), "إيجار", "اتجار→إيجار (before ولا/لا)"),
    (re.compile(rf"({_W}(?:عايز|عايزة|ولا)\s+)اتجار{_E}"), r"\g<1>إيجار", "اتجار→إيجار (after عايز/ولا)"),
    (re.compile(_W + r"(?:التجامعة|التجامع|التجامة)" + _E), "التجمع", "التجامعة→التجمع"),
    (re.compile(_W + r"(?:عقراط|عكرات|عقاعد|عطارات|عقرات)" + _E), "عقارات", "عقراط→عقارات"),
]


def correct_transcript(text: str) -> tuple[str, list[str]]:
    """Return the corrected text and the labels of the rules that changed something."""
    applied: list[str] = []
    for pattern, repl, label in RULES:
        text, n = pattern.subn(repl, text)
        if n:
            applied.extend([label] * n)
    return text, applied
