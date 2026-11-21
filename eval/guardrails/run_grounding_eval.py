"""Grounded replies: كل رقم في ردود الشقق لازم يطابق سجل الشقة في الداتابيز.

    python eval/guardrails/run_grounding_eval.py

لكل شقة من الـ 100 اللي في data/final_merged_dataset.csv بيبني نفس الردود اللي الأبلكيشن بيطلّعها،
وبنفس الدوال اللي chat_view.py بيستخدمها:
  1. كارت الشقة في الشات                 (grounding.card_text / display_fields)
  2. تفاصيل الإيجار والتأمين في الشات     (grounding.rent_details_text)
  3. التقرير اللي بيتبعت على تليجرام      (telegram_service.format_property_report)
وبعدين grounding.check_reply بيقارن كل رقم وكل كمبوند في الرد بالسجل.

Negative control: نفس الردود بعد ما نغيّر فيها السعر أو المساحة أو الأوض أو الكمبوند،
عشان نتأكد إن الـ check بيمسك الغلط فعلاً (يعني الـ 0 mismatches مش لأن الـ check مبيشوفش حاجة).

بيكتب grounding_report.md و grounding_results.json جنبه.
"""
import json
import random
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "smart_semsar_ui"))          # services.telegram_service زي ما chat_view بيستورده
from backend.guardrails.grounding import (DEPOSIT_MONTHS, card_text, check_reply, compound_of,  # noqa: E402
                                          facts, rent_details_text, report_info)
from services.telegram_service import format_property_report  # noqa: E402

HERE = Path(__file__).resolve().parent


def replies_for(record):
    price = facts(record)["price"]
    return {
        "chat_card": card_text(record, is_rent=True),
        "rent_details": rent_details_text(price, DEPOSIT_MONTHS),
        "telegram_report": format_property_report(report_info(record, True, DEPOSIT_MONTHS),
                                                  customer_name="Client", format_type="text"),
    }


def tamper(reply, record, other, rng):
    """نفس الرد بس بقيمة واحدة غلط (زي ما موديل كان ممكن يكتبها من عنده)"""
    f = facts(record)
    options = []
    if f["price"]:
        options.append(("price", f"{f['price']:,.0f}", f"{f['price'] * 1.1:,.0f}"))
    if f["area_sqm"]:
        options.append(("area", f"{f['area_sqm']:g} ", f"{f['area_sqm'] + 15:g} "))
    if f["compound"] and facts(other)["compound"] and facts(other)["compound"] != f["compound"]:
        options.append(("compound", f["compound"], facts(other)["compound"]))
    options = [o for o in options if o[1] in reply]
    if not options:
        return None, None
    kind, old, new = rng.choice(options)
    return kind, reply.replace(old, new)


def main() -> int:
    records = pd.read_csv(ROOT / "data" / "final_merged_dataset.csv").to_dict(orient="records")
    compounds = [compound_of(r["location"]) for r in records]
    rng = random.Random(7)

    results, mismatches, control = [], [], {"tampered": 0, "caught": 0, "missed": []}
    for i, record in enumerate(records):
        for kind, reply in replies_for(record).items():
            problems = check_reply(reply, record, compounds, deposit_months=DEPOSIT_MONTHS)
            results.append({"property_id": record["property_id"], "reply_type": kind,
                            "problems": problems, "reply": reply})
            if problems:
                mismatches.append(results[-1])

            other = records[(i + rng.randint(1, len(records) - 1)) % len(records)]
            what, bad = tamper(reply, record, other, rng)
            if bad:
                control["tampered"] += 1
                if check_reply(bad, record, compounds, deposit_months=DEPOSIT_MONTHS):
                    control["caught"] += 1
                else:
                    control["missed"].append(f"{record['property_id']} {kind} ({what})")

    by_type = {}
    for r in results:
        by_type.setdefault(r["reply_type"], [0, 0])
        by_type[r["reply_type"]][0] += 1
        by_type[r["reply_type"]][1] += bool(r["problems"])

    passed = not mismatches and control["caught"] == control["tampered"]
    lines = ["# Grounded replies report", "",
             f"Checked **{len(results)} replies** built for the {len(records)} listings in "
             "`data/final_merged_dataset.csv`, with the same functions the app uses.",
             "Every number and compound name in each reply was compared with the listing's database record.", "",
             "| Reply type | Replies | Mismatches |", "|---|---|---|"]
    lines += [f"| {k} | {n} | {m} |" for k, (n, m) in by_type.items()]
    lines += ["", f"**Total mismatches: {len(mismatches)} / {len(results)}**", ""]
    for r in mismatches[:20]:
        lines.append(f"- {r['property_id']} {r['reply_type']}: {'; '.join(r['problems'])}")
    lines += ["## Negative control", "",
              "The same replies with one value changed (price +10%, area +15 m², or another listing's compound).",
              f"The check caught **{control['caught']} / {control['tampered']}** tampered replies.",
              "Missed: " + ("; ".join(control["missed"]) or "none"), "",
              f"**Result: {'PASS' if passed else 'FAIL'}**", ""]
    (HERE / "grounding_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (HERE / "grounding_results.json").write_text(
        json.dumps({"results": results, "control": control}, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n".join(lines))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
