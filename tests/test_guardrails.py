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
        if path.name == "graph.py":                 # الجراف القديم، كله commented out ومش بيتستورد
            continue
        code = "\n".join(l for l in path.read_text(encoding="utf-8").splitlines()
                         if not l.strip().startswith(("#", '"""')) and "قبل كده" not in l)
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
