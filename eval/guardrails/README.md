# Guardrails (Requirements #3)

الأجزاء اللي **مش** معتمدة على طلب 2 (البحث). الأرقام من الداتابيز (Grounded replies) والإعلانات الملغّمة (Poisoned listings) مستنيين طلب 2 يخلص.

| المطلوب | فين | الحالة |
|---|---|---|
| No false claims + تصليح رسالة WhatsApp | `backend/guardrails/actions.py`, `backend/agents/final_report_node.py`, `frontend/views/chat_view.py`, `backend/agents/routing_node.py` | خلص |
| Privacy (logs + LangSmith) | `backend/guardrails/pii.py`, `backend/agents/semsar_graph.py` | خلص، والـ trace قبل وبعد في `privacy_before_after.md` |
| Schema + sanity ranges | `backend/schema/state.py` (`CustomerRequirements`) | خلص |
| Ask, don't guess | `backend/guardrails/clarify.py`, `backend/agents/input_node.py`, البرومبت في `backend/services/user_input_text.py` | خلص |
| Attack test (30 + 30) | `backend/guardrails/input_guard.py`, `attack_set.jsonl`, `run_attack_eval.py` | خلص، التقرير في `attack_report.md` |

## تشغيل

```bash
python -m pytest tests/test_guardrails.py -q
python eval/guardrails/run_attack_eval.py          # القواعد بس
python eval/guardrails/run_attack_eval.py --llm    # القواعد + الموديل (محتاج GROQ_API_KEY شغال)
python eval/guardrails/privacy_trace_demo.py       # لو LANGSMITH_API_KEY موجود بيبعت trace قبل و trace بعد
```

## نتيجة اختبار الهجوم (القواعد بس)

- المجموعة الأساسية (30 + 30): **30/30 هجمة اتمنعت، 0/30 طلب عادي اتمنع غلط**. المجموعة دي اتكتبت مع القواعد، فالرقم ده أعلى من الحقيقة.
- مجموعة holdout (جمل جديدة محدش عدّل القواعد عليها): **2/11 هجمة بس اتمنعت**، و0/11 طلب عادي اتمنع غلط.
  يعني القواعد لوحدها مش كفاية قدام صيغ جديدة. عشان كده فيه طبقة تانية بموديل (`--llm`)،
  بس **لسه متجربتش**: مفتاح Groq اللي كان موجود رجّع 401 (Invalid API Key). لما يبقى فيه مفتاح شغال، شغّلوا `--llm`
  واكتبوا الرقمين في التقرير.
- الأحسن قبل التقرير النهائي: حد من الفريق يكتب 30 هجمة جديدة مشافش القواعد، وتحطوا الرقم ده.
