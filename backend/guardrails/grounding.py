"""Grounded replies: السعر والمساحة والأوض والكمبوند في أي رد جايين من سجل الشقة في الداتابيز بس.

  display_fields(record)     القيم اللي بتتعرض في الكروت (لو قيمة ناقصة: "غير متاح"، مش رقم من عندنا)
  report_info(record, ...)   نفس القيم للتقرير اللي بيتبعت على تليجرام
  check_reply(reply, record) بيطلّع كل رقم في الرد ويقارنه بالسجل؛ أي رقم ملوش أصل = mismatch
  strip_ungrounded_numbers   رد الموديل (supervisor) ميكتبش سعر أو مساحة أو أوض من عنده
"""
import html
import re
from typing import Any, Dict, Iterable, List, Optional

from backend.guardrails.clarify import normalize_digits

NOT_AVAILABLE = "غير متاح"
NEUTRAL_REPLY = "تمام 👌 قولي محتاج إيه وأنا أساعدك."
DEPOSIT_MONTHS = 2     # تأمين الإيجار: إيجار شهرين مقدم، بيرجع آخر مدة العقد لو مفيش تلفيات


def field(record: Any, *names: str) -> Any:
    """أول قيمة موجودة من السجل، سواء dict أو PropertyItem (NaN و"" = مش موجودة)"""
    if record is None:
        return None
    for name in names:
        value = record.get(name) if isinstance(record, dict) else getattr(record, name, None)
        if value not in (None, "") and value == value:
            return value
    return None


def compound_of(location: Optional[str]) -> Optional[str]:
    """"Hyde Park, Hyde Park, New Cairo City, Cairo" -> "Hyde Park" (أول جزء في العنوان)"""
    if not location:
        return None
    return str(location).split(",")[0].strip() or None


def facts(record: Any) -> Dict[str, Any]:
    """الحقايق اللي مسموح تتقال عن الشقة، كلها من السجل"""
    def num(*names):
        v = field(record, *names)
        try:
            return float(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    rooms = num("pf_bedrooms", "bedrooms")
    baths = num("pf_bathrooms", "bathrooms")
    location = field(record, "location")
    return {
        "property_id": field(record, "property_id", "id"),
        "title": field(record, "title"),
        "location": str(location) if location is not None else None,
        "compound": compound_of(location),
        "price": num("price"),
        "area_sqm": num("pf_area_sqm", "area_sqm"),
        "bedrooms": int(rooms) if rooms is not None else None,
        "bathrooms": int(baths) if baths is not None else None,
    }


def _area_text(area: Optional[float]) -> str:
    return NOT_AVAILABLE if area is None else f"{area:g}"


def display_fields(record: Any) -> Dict[str, str]:
    """قيم الكارت كنص جاهز للعرض. كله escaped عشان نص الإعلان ميبقاش HTML بيتنفذ في الصفحة."""
    f = facts(record)

    def text(v):
        return html.escape(str(v)) if v is not None else NOT_AVAILABLE

    return {
        "id": text(f["property_id"]),
        "title": text(f["title"]),
        "location": text(f["location"]),
        "bedrooms": text(f["bedrooms"]),
        "area": _area_text(f["area_sqm"]),
        "price": NOT_AVAILABLE if f["price"] is None else f"EGP {f['price']:,.0f}",
    }


def card_text(record: Any, is_rent: bool = True) -> str:
    """نفس سطور كارت الشقة اللي في الشات (من غير الـ HTML)"""
    d = display_fields(record)
    label = "الإيجار الشهري" if is_rent else "السعر الإجمالي"
    return (f"{d['title']} (#{d['id']})\n"
            f"الموقع: {d['location']} | المساحة: {d['area']} متر مربع | عدد الغرف: {d['bedrooms']}\n"
            f"{label}: {d['price']}")


def rent_details_text(price: float, deposit_months: int = DEPOSIT_MONTHS) -> str:
    """تفاصيل الإيجار اللي بتظهر في الشات: كلها حسابات على سعر السجل"""
    months = "شهرين" if deposit_months == 2 else f"{deposit_months} شهور"
    return (f"- **الإيجار الشهري:** EGP {price:,.0f}\n"
            f"- **التأمين (إيجار {months} مقدم):** EGP {price * deposit_months:,.0f}، "
            f"ويُسترد هذا المبلغ في نهاية مدة العقد في حال عدم وجود أي تلفيات.\n"
            f"- **أول دفعة عند الاستلام (التأمين فقط):** EGP {price * deposit_months:,.0f}\n"
            f"- **إجمالي الإيجار في السنة:** EGP {price * 12:,.0f}\n")


def report_info(record: Any, is_rent: bool, deposit_months: int) -> Dict[str, Any]:
    """بيانات تقرير تليجرام، من السجل بس (من غير قيم افتراضية مخترعة)"""
    f = facts(record)
    return {"id": str(f["property_id"]), "title": f["title"] or NOT_AVAILABLE,
            "location": f["location"] or NOT_AVAILABLE, "price": f["price"],
            "area_sqm": _area_text(f["area_sqm"]),
            "bedrooms": f["bedrooms"] if f["bedrooms"] is not None else NOT_AVAILABLE,
            "deal_type": "rent" if is_rent else "buy", "deposit_months": deposit_months}


# ---------- استخراج الأرقام من الرد ----------
_ID_RE = re.compile(r"#?\bPROP_\d+\b", re.I)
_NUM_RE = re.compile(r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?(?:\s*(مليون|million|ألف|الف|thousand|k\b))?",
                     re.I)
_MULT = {"مليون": 1e6, "million": 1e6, "ألف": 1e3, "الف": 1e3, "thousand": 1e3, "k": 1e3}


def numbers_in(text: str) -> List[float]:
    """كل الأرقام في النص كقيم (٥٠٬٠٠٠ و 50,000 و "50 ألف" كلهم 50000). رقم الشقة PROP_1001 مش بيتحسب."""
    t = _ID_RE.sub(" ", normalize_digits(text or "").replace("٬", ",").replace("٫", "."))
    out = []
    for m in _NUM_RE.finditer(t):
        value = float(m.group(1).replace(",", "") + ("." + m.group(2) if m.group(2) else ""))
        if m.group(3):
            value *= _MULT[m.group(3).lower()]
        out.append(value)
    return out


def allowed_numbers(record: Any, deposit_months: Optional[int] = None,
                    plan: Optional[Dict[str, float]] = None) -> List[float]:
    """الأرقام اللي ليها أصل: قيم السجل، والحسابات اللي الأبلكيشن بيعملها على السعر (التأمين، الإيجار السنوي، التقسيط)"""
    f = facts(record)
    nums = [v for v in (f["price"], f["area_sqm"], f["bedrooms"], f["bathrooms"]) if v is not None]
    nums += numbers_in(f"{f['title'] or ''} {f['location'] or ''}")   # "The 5th Settlement" نص من السجل
    if f["price"] is not None:
        nums.append(f["price"] * 12)                       # الإيجار في السنة
        if deposit_months:
            nums += [f["price"] * deposit_months, deposit_months]
    if plan:
        nums += [float(v) for v in plan.values() if isinstance(v, (int, float))]
    return nums


def _matches(value: float, allowed: Iterable[float]) -> bool:
    # الرد بيقرّب لأقرب جنيه (EGP {price:,.0f})، فنص جنيه فرق مسموح
    return any(abs(value - a) <= 0.5 or abs(value - a) <= 1e-6 * abs(a) for a in allowed)


def check_reply(reply: str, record: Any, all_compounds: Iterable[str] = (),
                deposit_months: Optional[int] = None, plan: Optional[Dict[str, float]] = None) -> List[str]:
    """بترجع قايمة بالمشاكل (فاضية = الرد مطابق للسجل).

    - أي رقم في الرد مش من السجل ولا من حساب عليه = mismatch
    - كمبوند شقة تانية مذكور في الرد = mismatch
    """
    f = facts(record)
    allowed = allowed_numbers(record, deposit_months, plan)
    problems = [f"number {v:g} is not in the record" for v in numbers_in(reply) if not _matches(v, allowed)]
    location = (f["location"] or "").lower()
    low = (reply or "").lower()
    for c in set(all_compounds):
        if c and c.lower() not in location and re.search(rf"(?<!\w){re.escape(c.lower())}(?!\w)", low):
            problems.append(f"compound '{c}' belongs to another listing")
    return problems


# ---------- رد الموديل ----------
_FACT_NUMBER = re.compile(
    r"(\d[\d,.]*)\s*(?:جنيه|ج\.?م|egp|le|ألف|الف|مليون|k\b|million|متر|م2|م²|sqm|m2|m²|"
    r"اوض|أوض|اوضه|أوضة|اوضة|غرف|غرفة|غرفه|bedrooms?|rooms?)", re.I)


def strip_ungrounded_numbers(reply: str, user_text: str = "") -> str:
    """الموديل اللي بيكتب الرد مبيشوفش الداتابيز، فأي سعر أو مساحة أو عدد أوض في رده لازم يكون العميل قاله.
    غير كده الرد بيتبدل برد عادي، والأرقام بتطلع في الكروت من السجل."""
    said = set(numbers_in(user_text))
    for m in _FACT_NUMBER.finditer(normalize_digits(reply or "")):
        if not set(numbers_in(m.group(0))) <= said:
            return NEUTRAL_REPLY
    return reply
