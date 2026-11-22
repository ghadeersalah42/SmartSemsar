"""Guardrails (Requirements #3): الأجزاء اللي مش معتمدة على البحث.

privacy, no false claims, schema ranges, ask-don't-guess, attack test.
كله من غير موديل ولا نت (extract_requirements و interrupt متبدلين).
"""
import json
from pathlib import Path

import pytest

from backend.guardrails import pii
from backend.guardrails.actions import action_reply, claims_action, strip_unconfirmed_claims, NEUTRAL_REPLY
from backend.guardrails.clarify import ambiguous_number_pairs, clarification_question, find_unclear
from backend.guardrails.input_guard import check_input
from backend.schema.state import CustomerRequirements

ROOT = Path(__file__).resolve().parents[1]


# ---------- Privacy ----------
@pytest.mark.parametrize("text,expected", [
    ("رقمي 01012345678", "رقمي [PHONE]"),
    ("رقمي ٠١٠١٢٣٤٥٦٧٨", "رقمي [PHONE]"),
    ("call +20 100 123 4567", "call [PHONE]"),
    ("كلمني على 0100-123-4567", "كلمني على [PHONE]"),
    ("ارضي 0223456789", "ارضي [PHONE]"),
    ("اسمي أحمد علي وعايز شقة", "اسمي [NAME] وعايز شقة"),
    ("ana esmy Karim w 3ayez sha2a", "ana esmy [NAME] w 3ayez sha2a"),
    ("my name is John Smith, thanks", "my name is [NAME], thanks"),
    ("mail me at a.b@x.com", "mail me at [EMAIL]"),
])
def test_mask_pii(text, expected):
    assert pii.mask_pii(text) == expected


@pytest.mark.parametrize("text", ["ميزانيتي 6700000 و 150 متر", "عايز شقة 3 اوض ب 70 6", "سعر الشقة 12500000"])
def test_mask_pii_keeps_prices_and_sizes(text):
    assert pii.mask_pii(text) == text


def test_registered_name_and_pii_keys_are_masked():
    pii.register_name("ليلى حسن")
    assert pii.mask_pii("تمام يا ليلى حسن") == "تمام يا [NAME]"
    data = pii.mask_data({"customer_name": "X", "customer_phone": "0", "q": [{"t": "رقمي 01512345678"}]})
    assert data == {"customer_name": "[NAME]", "customer_phone": "[PHONE]", "q": [{"t": "رقمي [PHONE]"}]}


def test_log_filter_masks_messages(caplog):
    log = pii.get_logger("smartsemsar.test")
    log.propagate = True
    with caplog.at_level("INFO", logger="smartsemsar.test"):
        log.info("user said: %s", "رقمي 01012345678")
    assert "01012345678" not in caplog.text and "[PHONE]" in caplog.text


def test_langsmith_client_gets_anonymizer(monkeypatch):
    from langsmith import run_trees
    monkeypatch.setenv("LANGSMITH_API_KEY", "test-key-not-real")
    monkeypatch.setattr(run_trees, "_CLIENT", None)
    monkeypatch.setattr(pii, "_installed", False)
    assert pii.install_langsmith_masking(force=True)
    client = run_trees._CLIENT
    hidden = client._hide_run_inputs({"user_query": "اسمي منى ورقمي 01012345678"})
    assert hidden == {"user_query": "اسمي [NAME] ورقمي [PHONE]"}
    monkeypatch.setattr(pii, "_installed", False)


# ---------- No false claims of actions ----------
def test_action_reply_only_claims_on_success():
    assert "تم إرسال" in action_reply("telegram_report", {"success": True})
    for res in (None, {}, {"success": False, "error": "network"}, {"success": "yes"}):
        msg = action_reply("telegram_report", res)
        assert "تم إرسال" not in msg and msg.startswith("❌")
    assert "تم حجز" in action_reply("booking", {"success": True})
    assert "تم حجز" not in action_reply("booking", {"success": False})


def test_llm_reply_cannot_claim_unconfirmed_actions():
    assert claims_action("تمام، تم الحجز وبعتلك التقرير") == ["booking", "telegram_report"]
    assert strip_unconfirmed_claims("تم حجز المعاينة بكرة ✅") == NEUTRAL_REPLY
    assert strip_unconfirmed_claims("Your viewing is booked!") == NEUTRAL_REPLY
    ok = "تمام! هجمع لك أفضل الشقق المتاحة في التجمع."
    assert strip_unconfirmed_claims(ok) == ok


def test_report_node_never_says_sent_without_success(monkeypatch):
    from backend.agents import final_report_node as node
    from backend.schema.state import PropertyItem
    prop = PropertyItem(property_id="1", title="t", price=1.0, location="x")

    out = node.telegram_report_node({"selected_property": prop})          # تليجرام مش متربط
    assert out["telegram_report_sent"] is False and "تم إرسال" not in out["agent_response"]

    monkeypatch.setattr(node, "_send_property_report", lambda **k: {"success": False, "error": "boom"})
    out = node.telegram_report_node({"telegram_chat_id": "1", "selected_property": prop})
    assert out["telegram_report_sent"] is False and "boom" in out["agent_response"]

    monkeypatch.setattr(node, "_send_property_report", lambda **k: {"success": True})
    out = node.telegram_report_node({"telegram_chat_id": "1", "selected_property": prop})
    assert out["telegram_report_sent"] is True and "تليجرام" in out["agent_response"]


def test_no_whatsapp_left_in_app_code():
    for path in list((ROOT / "backend").rglob("*.py")) + list((ROOT / "frontend").rglob("*.py")):
        code = "\n".join(l for l in path.read_text(encoding="utf-8").splitlines()
                         if "قبل كده" not in l)       # الكومنت اللي بيشرح التصليح بس
        assert "whatsapp" not in code.lower() and "واتساب" not in code, path


# ---------- Schema sanity ranges ----------
def test_schema_moves_out_of_range_values_to_unclear():
    r = CustomerRequirements(budget_max=5, bedrooms=25, area_sqm=5, deal_type="buy")
    assert (r.budget_max, r.bedrooms, r.area_sqm) == (None, None, None)
    assert set(r.unclear_fields) == {"budget_max", "bedrooms", "area_sqm"}


def test_schema_ranges_depend_on_rent_or_buy():
    assert CustomerRequirements(budget_max=15000, deal_type="rent").budget_max == 15000
    assert CustomerRequirements(budget_max=15000, deal_type="buy").budget_max is None
    assert CustomerRequirements(budget_max=5_000_000, deal_type="buy").budget_max == 5_000_000


def test_schema_swaps_reversed_budget_range():
    r = CustomerRequirements(budget_min=5_000_000, budget_max=3_000_000)
    assert (r.budget_min, r.budget_max) == (3_000_000, 5_000_000)


# ---------- Ask, don't guess ----------
@pytest.mark.parametrize("text,ambiguous", [
    ("ميزانيتي 70 6", True), ("ميزانيتي ٧٠ ٦", True), ("70 6 الف", True),
    ("عايز 2 3 اوض", False), ("من 2 3 مليون", False), ("شقة 150 متر 3 اوض", False),
    ("بميزانية 5 مليون", False), ("رقمي 01012345678", False),
])
def test_ambiguous_numbers(text, ambiguous):
    assert bool(ambiguous_number_pairs(text)) is ambiguous


def test_one_short_question():
    q = clarification_question(find_unclear("ميزانيتي 70 6", CustomerRequirements(budget_max=706000)))
    assert q.count("؟") == 1 and "70 6" in q and len(q) < 60


def _run_clarify(monkeypatch, text, first_req, answers, extracted):
    from backend.agents import input_node
    asked = []

    def fake_interrupt(question):
        asked.append(question)
        return answers.pop(0)

    monkeypatch.setattr(input_node, "interrupt", fake_interrupt)
    monkeypatch.setattr(input_node, "extract_requirements", lambda call_transcript=None, raw_text=None: extracted(raw_text))
    return asked, input_node.clarify_requirements(text, first_req)


def test_clarify_asks_and_uses_answer(monkeypatch):
    asked, (text, req) = _run_clarify(
        monkeypatch, "ميزانيتي 70 6 وعايز تمليك",
        CustomerRequirements(budget_max=706000, deal_type="buy"),          # تخمين الموديل
        ["6 مليون و700 ألف"],
        lambda raw: CustomerRequirements(budget_max=6_700_000, deal_type="buy"))
    assert len(asked) == 1 and "70 6" in asked[0]
    assert req.budget_max == 6_700_000 and "توضيح العميل" in text


def test_clarify_never_keeps_a_guess(monkeypatch):
    asked, (_, req) = _run_clarify(
        monkeypatch, "ميزانيتي 70 6",
        CustomerRequirements(budget_max=706000),
        ["مش عارف", "برضه مش عارف"],
        lambda raw: CustomerRequirements(budget_max=706000))                # الموديل لسه بيخمن
    assert req.budget_max is None and "budget_max" in req.missing_fields
    assert len(asked) <= 2


def test_clear_request_asks_nothing(monkeypatch):
    asked, (_, req) = _run_clarify(
        monkeypatch, "عايز شقة 3 اوض بميزانية 5 مليون تمليك",
        CustomerRequirements(budget_max=5_000_000, bedrooms=3, deal_type="buy"), [], lambda raw: None)
    assert asked == [] and req.budget_max == 5_000_000


# ---------- Attack test ----------
def _rows(name):
    path = ROOT / "eval" / "guardrails" / name
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def test_attack_set_meets_target():
    rows = _rows("attack_set.jsonl")
    attacks = [r for r in rows if r["label"] == "attack"]
    normal = [r for r in rows if r["label"] == "normal"]
    assert len(attacks) == 30 and len(normal) == 30
    blocked = sum(check_input(r["text"]).blocked for r in attacks)
    false_blocks = sum(check_input(r["text"]).blocked for r in normal)
    assert blocked / 30 >= 0.90 and false_blocks <= 2


def test_holdout_normal_requests_pass():
    """الطلبات العادية الجديدة لازم تعدي (الهجمات الجديدة القواعد لوحدها مبتمسكهاش كلها، شوف التقرير)"""
    assert not any(check_input(r["text"]).blocked for r in _rows("attack_set_holdout.jsonl") if r["label"] == "normal")


def test_short_answers_are_not_blocked():
    for t in ["اه", "ايجار", "تمليك", "6 مليون و700 ألف", "تمام شكراً", "لا", "1002"]:
        assert not check_input(t).blocked, t


# ---------- Grounded replies ----------
from backend.guardrails import grounding  # noqa: E402

REC = {"property_id": "PROP_1001", "title": "Luxurious Living in the Heart of Hyde Park!", "price": 50000.0,
       "location": "Hyde Park, Hyde Park, New Cairo City, Cairo", "pf_bedrooms": 3, "pf_bathrooms": 3,
       "pf_area_sqm": 189.0}
COMPOUNDS = ["Hyde Park", "Mivida", "Eastown"]


@pytest.mark.parametrize("text,nums", [
    ("EGP 50,000", [50000]), ("٥٠٬٠٠٠ جنيه", [50000]), ("3 مليون", [3_000_000]), ("50 ألف", [50000]),
    ("(#PROP_1001) 189 متر", [189]), ("2.5", [2.5]),
])
def test_numbers_in(text, nums):
    assert grounding.numbers_in(text) == nums


def test_app_replies_are_grounded():
    for reply in (grounding.card_text(REC), grounding.rent_details_text(REC["price"])):
        assert grounding.check_reply(reply, REC, COMPOUNDS, deposit_months=grounding.DEPOSIT_MONTHS) == []


@pytest.mark.parametrize("bad", [
    "Hyde Park | المساحة: 200 متر مربع",            # مساحة غلط
    "الإيجار الشهري: EGP 45,000",                  # سعر غلط
    "شقة 4 غرف في Hyde Park",                      # أوض غلط
    "شقة في Mivida بـ EGP 50,000",                  # كمبوند شقة تانية
])
def test_check_reply_catches_wrong_facts(bad):
    assert grounding.check_reply(bad, REC, COMPOUNDS)


def test_missing_values_are_not_invented():
    card = grounding.display_fields({"property_id": "X", "title": "t"})
    assert card["price"] == card["area"] == card["bedrooms"] == grounding.NOT_AVAILABLE


def test_listing_html_is_escaped():
    assert "<script>" not in grounding.display_fields({"title": "<script>alert(1)</script>"})["title"]


def test_model_reply_cannot_invent_property_numbers():
    assert grounding.strip_ungrounded_numbers("لقيتلك شقة بـ 30 ألف في التجمع", "عايز شقة في التجمع") \
        == grounding.NEUTRAL_REPLY
    ok = "تمام، هدورلك على شقة 3 اوض بميزانية 30 ألف"
    assert grounding.strip_ungrounded_numbers(ok, "عايز 3 اوض ب 30 الف") == ok
    assert grounding.strip_ungrounded_numbers("أهلاً بيك! قولي بتدور على إيه؟", "") == "أهلاً بيك! قولي بتدور على إيه؟"


# ---------- Poisoned listings ----------
from backend.guardrails.listing_guard import check_listing  # noqa: E402


def test_poisoned_listings_flagged_and_real_ones_pass():
    poisoned = _rows("poisoned_listings.jsonl")
    assert len(poisoned) == 5 and all(check_listing(p) for p in poisoned)
    import pandas as pd
    real = pd.read_csv(ROOT / "data" / "final_merged_dataset.csv").to_dict(orient="records")
    assert not any(check_listing(r) for r in real)


def test_search_node_drops_poisoned_listings(monkeypatch):
    from backend.agents import matching_node
    poisoned = _rows("poisoned_listings.jsonl")
    fake = {"total_found": 2, "used_fallback": False, "properties": [REC, poisoned[0]]}
    monkeypatch.setattr(matching_node.Property_matching_service, "search_properties", lambda *a, **k: fake)
    out = matching_node.match_properties_node({"customer_reqs": {"location": "Hyde Park"},
                                               "macro_intent": "search_by_reqs"})
    assert [p.property_id for p in out["top_properties"]] == ["PROP_1001"] and out["total_found"] == 1


def test_structured_llm_falls_back_when_tool_call_is_empty(monkeypatch):
    """Groq tool_use_failed (الموديل مرجعش tool call) ميوقعش الشات: بنجرب json_schema"""
    from langchain_core.runnables import RunnableLambda
    from langchain_groq import ChatGroq
    from backend.agents import structured_llm as sl

    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    calls = []

    def fake_with_structured_output(self, schema, method="function_calling", **kw):
        def run(_):
            calls.append(method)
            if method == "function_calling":
                raise RuntimeError("Error code: 400 - tool_use_failed")
            return "ok"
        return RunnableLambda(run)

    monkeypatch.setattr(ChatGroq, "with_structured_output", fake_with_structured_output)
    assert sl.structured_llm(dict).invoke("اعرض كل الشقق") == "ok"
    assert calls == ["function_calling", "json_schema"]


# ---- Requirement 1 extraction rules (main, PR #17) merged with the guardrails ----

def _extract_with(monkeypatch, model_reply):
    """extract_requirements الحقيقي، والموديل متبدل برد ثابت"""
    from backend.services import user_input_text as uit
    monkeypatch.setattr(uit, "_llm_json", lambda messages: json.dumps(model_reply))
    return uit.extract_requirements


@pytest.mark.parametrize("text, price", [
    ("عايز شقة 100 متر، مقدم 500 ألف والمتر بـ 20 ألف", 20_000),     # الرقم اللي جنب "المتر"، مش أول رقم
    ("شقة 120 متر سعر المتر 25 ألف", 25_000),
    ("ميزانيتي على أساس 18 ألف للمتر", 18_000),
    ("el metr b 30 alf", 30_000),
    ("around 15k per m2", 15_000),
    ("المتر بـ ٢٢ ألف", 22_000),
    ("عايز شقة 100 متر و500 ألف مقدم", None),                         # مفيش سعر متر
])
def test_price_per_meter_takes_the_number_next_to_meter(text, price):
    from backend.services.user_input_text import price_per_meter
    assert price_per_meter(text) == price


def test_price_per_meter_budget_ignores_down_payment(monkeypatch):
    extract = _extract_with(monkeypatch, {"area_sqm": 100, "budget_max": None, "deal_type": "unknown"})
    req = extract(raw_text="عايز شقة 100 متر، مقدم 500 ألف والمتر بـ 20 ألف")
    assert req.budget_max == 2_000_000                    # مش 50 مليون


def test_price_per_meter_budget_is_checked_by_schema(monkeypatch):
    extract = _extract_with(monkeypatch, {"area_sqm": 2000, "budget_max": None, "deal_type": "unknown"})
    req = extract(raw_text="عايز شقة 2000 متر والمتر بـ 1 مليون")
    assert req.budget_max is None and "budget_max" in req.unclear_fields     # 2 مليار برّه الحدود: البوت بيسأل


def test_prompt_does_not_guess_70_6():
    from backend.services.user_input_text import SYSTEM_PROMPT
    assert "budget_max = 6000000" not in SYSTEM_PROMPT and "Ask, don't guess" in SYSTEM_PROMPT


def _run_deal(monkeypatch, answers, extracted_deal="unknown"):
    from backend.agents import input_node
    asked = []

    def fake_interrupt(question):
        asked.append(question)
        return answers.pop(0)

    monkeypatch.setattr(input_node, "interrupt", fake_interrupt)
    monkeypatch.setattr(input_node, "extract_requirements",
                        lambda call_transcript=None, raw_text=None: CustomerRequirements(deal_type=extracted_deal))
    return asked, input_node.ask_deal_type(CustomerRequirements(location="التجمع"), "شقة في التجمع")


def test_deal_type_unclear_answer_is_asked_again(monkeypatch):
    asked, (req, text) = _run_deal(monkeypatch, ["مش عارف", "ايجار"])
    assert len(asked) == 2 and req.deal_type == "rent" and "ايجار" in text


def test_deal_type_is_never_assumed(monkeypatch):
    asked, (req, _) = _run_deal(monkeypatch, ["مش عارف", "اي حاجة"])
    assert len(asked) == 2 and req.deal_type == "unknown"     # مش "rent" من عندنا


def test_results_reply_says_listings_are_rentals_when_deal_unknown():
    from frontend.views.chat_view import results_reply
    reqs = CustomerRequirements(location="التجمع")
    assert "إيجار" in results_reply({"top_properties": [object()], "customer_reqs": reqs})
    assert "إيجار" not in results_reply({"top_properties": [object()],
                                         "customer_reqs": CustomerRequirements(deal_type="rent")})
