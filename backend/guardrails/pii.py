"""pii.py: إخفاء أرقام التليفونات والأسامي قبل ما تتكتب في الـ logs أو تتبعت لـ LangSmith.

    mask_pii("اسمي أحمد علي ورقمي 01012345678")  ->  "اسمي [NAME] ورقمي [PHONE]"

ثلاث طرق بنمسك بيها البيانات:
  1) أرقام التليفون المصرية (موبايل وأرضي) بالأرقام العادية أو العربي (٠١٠...)، ومعاها +20 / 0020
  2) الأسامي بعد "اسمي / معاك / my name is / esmy ..."
  3) أسامي العملاء اللي العميل كتبها في فورم الحجز (register_name)، فبتتخفي في أي مكان تظهر فيه
  4) أي قيمة في trace مفتاحها اسم أو تليفون (customer_name, customer_phone, phone, ...)

install_langsmith_masking() بتخلي كل الـ traces اللي بتطلع من LangChain/LangGraph تعدي على mask_pii
قبل ما تتبعت. لازم تتنادى قبل أول trace (semsar_graph.py بتناديها وقت الـ import).
"""
from __future__ import annotations

import logging
import os
import re
import threading
from typing import Any, Iterable

PHONE_TOKEN = "[PHONE]"
NAME_TOKEN = "[NAME]"
EMAIL_TOKEN = "[EMAIL]"

# ---------- أرقام التليفون ----------
_D = "[0-9٠-٩۰-۹]"                    # رقم: إنجليزي أو عربي أو فارسي
_ZERO, _ONE, _TWO = "[0٠۰]", "[1١۱]", "[2٢۲]"
_OPERATOR = "[0125٠١٢٥۰۱۲۵]"           # 010 / 011 / 012 / 015
_SEP = r"[\s\-.]?"
_PREFIX = rf"(?:(?:\+|00|٠٠){_TWO}{_ZERO}{_SEP})"    # +20 / 0020

MOBILE_RE = re.compile(
    rf"(?<![0-9٠-٩۰-۹])(?:{_PREFIX}?{_ZERO}?{_ONE}{_OPERATOR}(?:{_SEP}{_D}){{8}})(?![0-9٠-٩۰-۹])")
LANDLINE_RE = re.compile(          # القاهرة/الجيزة: 02 وبعدها 8 أرقام
    rf"(?<![0-9٠-٩۰-۹])(?:{_PREFIX}?{_ZERO}{_TWO}(?:{_SEP}{_D}){{8}})(?![0-9٠-٩۰-۹])")
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")

# ---------- الأسامي ----------
# كلمات بتيجي بعد الاسم ومش جزء منه ("اسمي أحمد وعايز شقة")
_STOP = {
    "و", "وعايز", "وعايزة", "عايز", "عايزة", "محتاج", "محتاجة", "بدور", "بادور", "من", "في", "فى",
    "ورقمي", "رقمي", "رقم", "وده", "ودا", "دا", "ده", "انا", "أنا", "يا", "عاوز", "عاوزة", "بس",
    "and", "i", "im", "i'm", "want", "from", "my", "number", "w", "3ayez", "3ayza", "bdawar",
    "mn", "fi", "ana", "w3ayez", "raqmy", "rakamy",
}
_NAME_WORD = r"[^\s\d,.;:!?،؛؟()\[\]{}\"'0-9٠-٩]{2,}"
NAME_INTRO_RE = re.compile(
    r"(?P<intro>(?:انا\s+|أنا\s+)?(?:اسمي|إسمي|أسمي|اسمى|إسمى)\s*(?:هو\s+|:\s*)?"
    r"|معاك\s+|معاكي\s+|معاكم\s+"
    r"|my\s+name\s+is\s+|my\s+name's\s+"
    r"|(?:ana\s+)?(?:esmy|esmi|ismy|ismi|2smy|2esmy)\s+|ma3ak\s+)",
    re.IGNORECASE)

_known_names: set[str] = set()
_lock = threading.Lock()


def register_name(name: str | None) -> None:
    """اسم عميل اتكتب في فورم: من دلوقتي بيتخفي في أي log أو trace"""
    name = (name or "").strip()
    if len(name) >= 2:
        with _lock:
            _known_names.add(name)
            _known_names.update(p for p in name.split() if len(p) >= 3)


_WORD_RE = re.compile(rf"\s*({_NAME_WORD})")


def _name_end(text: str, start: int) -> int | None:
    """آخر الاسم اللي بيبدأ عند start: لحد 3 كلمات ورا بعض ("محمد أحمد علي")،
    وبيقف عند كلمة زي "وعايز" أو "من" أو أي علامة ترقيم"""
    end, pos = None, start
    for i in range(3):
        m = _WORD_RE.match(text, pos)
        if not m:
            break
        word = m.group(1)
        if word.lower() in _STOP or (i > 0 and word.startswith("و")):
            break
        end = pos = m.end()
    return end


def _mask_intro_names(text: str) -> str:
    out, pos = [], 0
    for m in NAME_INTRO_RE.finditer(text):
        if m.start() < pos:
            continue
        end = _name_end(text, m.end())
        if end is None:
            continue
        out.append(text[pos:m.end()])
        out.append(NAME_TOKEN)
        pos = end
    out.append(text[pos:])
    return "".join(out)


def mask_pii(text: Any, extra_names: Iterable[str] = ()) -> Any:
    """النص بعد إخفاء التليفونات والإيميلات والأسامي. أي حاجة مش نص بترجع زي ما هي."""
    if not isinstance(text, str) or not text:
        return text
    text = MOBILE_RE.sub(PHONE_TOKEN, text)
    text = LANDLINE_RE.sub(PHONE_TOKEN, text)
    text = EMAIL_RE.sub(EMAIL_TOKEN, text)
    text = _mask_intro_names(text)
    with _lock:
        names = sorted(set(_known_names) | {n for n in extra_names if n and len(n) >= 2}, key=len, reverse=True)
    for name in names:
        text = re.sub(rf"(?<!\w){re.escape(name)}(?!\w)", NAME_TOKEN, text)
    return text


# ---------- القيم اللي مفتاحها نفسه بيقول إنها بيانات شخصية ----------
PII_KEYS = {"customer_name", "cust_name", "client_name", "name", "full_name", "username",
            "telegram_username", "customer_phone", "cust_phone", "phone", "phone_number", "mobile",
            "telegram_chat_id", "chat_id", "email"}


def _mask_with_path(value: str, path: list) -> str:
    last = str(path[-1]).lower() if path else ""
    if last in PII_KEYS and value.strip():
        return PHONE_TOKEN if any(k in last for k in ("phone", "mobile", "chat_id")) else NAME_TOKEN
    return mask_pii(value)


def mask_data(data: Any) -> Any:
    """نفس mask_pii بس على dict / list متداخلة (زي inputs/outputs بتوع الـ trace)"""
    if isinstance(data, str):
        return mask_pii(data)
    if isinstance(data, dict):
        out = {}
        for k, v in data.items():
            if isinstance(v, str) and str(k).lower() in PII_KEYS and v.strip():
                out[k] = _mask_with_path(v, [k])
            else:
                out[k] = mask_data(v)
        return out
    if isinstance(data, (list, tuple)):
        return type(data)(mask_data(v) for v in data)
    return data


# ---------- LangSmith ----------
def tracing_enabled() -> bool:
    flags = ("LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2", "LANGCHAIN_TRACING")
    return any(os.getenv(f, "").strip().lower() in ("1", "true", "yes") for f in flags)


_installed = False


def install_langsmith_masking(force: bool = False) -> bool:
    """كل trace بيطلع من LangChain/LangGraph يعدي على mask_pii قبل ما يتبعت لـ LangSmith.
    LangChain بياخد الـ client من langsmith.run_trees._CLIENT، فبنحط هناك client معاه anonymizer.
    بترجع True لو اتركّب."""
    global _installed
    if _installed and not force:
        return True
    if not (tracing_enabled() or force):
        return False
    try:
        from langsmith import Client
        from langsmith import run_trees
        from langsmith.anonymizer import create_anonymizer

        client = Client(anonymizer=create_anonymizer(_mask_with_path))
        old = run_trees._CLIENT
        if old is not None:
            try:
                old.flush()
            except Exception:
                pass
        run_trees._CLIENT = client
        _installed = True
        return True
    except Exception as e:          # LangSmith مش متسطب أو مفيش مفتاح: الأبلكيشن يكمل عادي
        logging.getLogger(__name__).warning("LangSmith masking not installed: %s", e)
        return False


# ---------- الـ logs ----------
class PIIMaskingFilter(logging.Filter):
    """فلتر logging: أي رسالة بتتكتب في الـ log بتتخفي الأول"""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
        except Exception:
            return True
        record.msg, record.args = mask_pii(msg), ()
        return True


def get_logger(name: str = "smartsemsar") -> logging.Logger:
    """logger جاهز: بيطبع في الكونسول، وكل رسالة بتتخفي قبل ما تتكتب"""
    logger = logging.getLogger(name)
    if not getattr(logger, "_pii_ready", False):
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
            logger.addHandler(handler)
        logger.addFilter(PIIMaskingFilter())
        logger.setLevel(logging.INFO)
        logger.propagate = False
        logger._pii_ready = True
    return logger


def safe_print(*parts: Any) -> None:
    """بديل print للسطور اللي فيها كلام العميل"""
    print(*(mask_pii(str(p)) for p in parts))
