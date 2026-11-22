# Privacy: one trace before and after masking

## Before (what was sent to LangSmith and printed in logs)
```json
{
  "inputs": {
    "user_query": "أنا اسمي منى السيد ورقمي ٠١٠٢٣٤٥٦٧٨٩، عايزة شقة 3 أوض في التجمع بميزانية 4 مليون",
    "booking": {
      "customer_name": "منى السيد",
      "customer_phone": "01023456789",
      "notes": "كلميني على 0100 234 5678 بعد الساعة 5"
    }
  },
  "outputs": {
    "customer_reqs": {
      "location": "التجمع",
      "bedrooms": 3,
      "budget_max": 4000000,
      "deal_type": "unknown"
    },
    "agent_response": "تمام يا منى، لقيتلك 4 شقق في التجمع 👇"
  }
}
```

## After (`backend/guardrails/pii.py`)
```json
{
  "inputs": {
    "user_query": "أنا اسمي [NAME] ورقمي [PHONE]، عايزة شقة 3 أوض في التجمع بميزانية 4 مليون",
    "booking": {
      "customer_name": "[NAME]",
      "customer_phone": "[PHONE]",
      "notes": "كلميني على [PHONE] بعد الساعة 5"
    }
  },
  "outputs": {
    "customer_reqs": {
      "location": "التجمع",
      "bedrooms": 3,
      "budget_max": 4000000,
      "deal_type": "unknown"
    },
    "agent_response": "تمام يا [NAME]، لقيتلك 4 شقق في التجمع 👇"
  }
}
```

Log line after masking:
```
guard blocked message (off_topic): أنا اسمي [NAME] ورقمي [PHONE]، عايزة شقة 3 أوض في التجمع بميزانية 4 مليون
```
