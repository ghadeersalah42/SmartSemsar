# SmartSemsar Controlled Extraction-Agreement Evaluation Report (All 10 Recordings)

**Evaluation Date (UTC):** `2026-11-23T23:47:37Z`
**Evaluation Scope:** All 10 recordings (`smartsemsar_rec1`–`smartsemsar_rec10`). Every reference transcript is human-checked: `rec1` and `rec8` were transcribed by hand; `rec2`–`rec7` and `rec9`–`rec10` started as multi-pass Whisper large-v3 drafts that a person listened to and corrected (2026-11-24).
**Machine-Readable Report:** `backend/services/data/evaluation/reports/extraction_agreement_report.json`
**Archived Previous Reports:**
- Legacy Groq Whisper 10-call report (2026-11-21): `reports/archive/extraction_agreement_report_legacy_groq_20261121.json`
- Manual-human 2-call report (`rec1` & `rec8`, 2026-11-22): `reports/archive/extraction_agreement_report_manual_2calls_20261122.json`
- Old-prompt 10-call report (1,888-char prompt, unreviewed references, 84.31%): `reports/archive/extraction_agreement_report_old_prompt_1888chars_20261122.json`

---

## 1. Controlled Experiment Configuration

Both transcript versions go through the **same** extraction function, model, prompt, schema, temperature and `Client:`/`Broker:` turn formatting, so the **only variable** is the transcript source:

| Parameter | Value |
|---|---|
| **Command Executed** | `python backend/services/evaluation/extraction_agreement.py --all-manifest-calls --pause 25` |
| **Extractor Function** | `backend.services.user_input_text.extract_requirements` (`raw_text=None`) |
| **Output Schema** | `backend.schema.state.CustomerRequirements` (9 fields: `location`, `budget_min`, `budget_max`, `area_sqm`, `bedrooms`, `property_type`, `purpose`, `deal_type`, `notes`) |
| **LLM Provider & Model** | Groq — `openai/gpt-oss-120b` (`SMARTSEMSAR_GROQ_MODEL`) |
| **Decoding Settings** | `temperature = 0`, `response_format = {"type": "json_object"}` |
| **System Prompt** | `backend.services.user_input_text.SYSTEM_PROMPT` (`2,802` chars, current `main`: guardrails "ask, don't guess" rules + price-per-metre rules) |
| **Rate-limit control** | `--pause 25` s before every LLM call so all 20 extractions stay on Groq (an earlier run hit the 8,000 tokens/min limit and fell back to Ollama on `rec10`; that run was discarded) |
| **Machine Transcript Source** | Local `faster-whisper large-v3-turbo` (`int8`, CPU) + `pyannote/speaker-diarization-community-1` + context-based role assignment (`conversations/smartsemsar_rec{1..10}_conversation.json`) |
| **Reference Transcript Source** | `ground_truth/transcripts/smartsemsar_rec{1..10}_gt_transcript.json` |
| **Input Formatting Parity** | Both formatted as `Client: ...` / `Broker: ...` lines with consecutive same-role turns merged. |

---

## 2. Metric Definitions & Denominator

A field is **present** if its value is not `null`, `""` or `"unknown"`. Fields absent in **both** extractions (`both_absent`) are excluded from the denominator so agreement is never inflated.

$H$ = reference requirements present (`exact + semantic + missing + conflicting`); $M$ = machine requirements present (`exact + semantic + extra + conflicting`).

| Metric | Formula |
|---|---|
| **Requirement Match Rate** | $(\text{Exact} + \text{Semantic}) / H \times 100$ |
| **Exact Match Rate** | $\text{Exact} / H \times 100$ |
| **Semantic Match Rate** | $\text{Semantic} / H \times 100$ *(equal after Arabic/numeric normalization)* |
| **Missing Requirement Rate** | $\text{Missing} / H \times 100$ |
| **Extra Requirement Rate** | $\text{Extra} / H \times 100$ *(and $\text{Extra} / M \times 100$)* |
| **Conflicting Value Rate** | $\text{Conflicting} / H \times 100$ |

---

## 3. Overall & Provenance Subgroup Results

| Scope | Calls | $H$ | $M$ | Excluded `both_absent` | Matched ($E+S$) | Exact ($E$) | Semantic ($S$) | Missing | Extra | Conflicting | **Req. Match Rate** | **Exact Rate** | **Semantic Rate** | **Missing Rate** | **Extra Rate (vs $H$ / vs $M$)** | **Conflict Rate** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **All 10 Recordings (`rec1`–`rec10`)** | **10** | **52** | **45** | **38** | **44** | **41** | **3** | **7** | **0** | **1** | **84.62%** | 78.85% | 5.77% | 13.46% | 0.00% / 0.00% | 1.92% |
| *Subgroup A: Manual human (`rec1`, `rec8`)* | 2 | 10 | 9 | 8 | 9 | 9 | 0 | 1 | 0 | 0 | **90.00%** | 90.00% | 0.00% | 10.00% | 0.00% / 0.00% | 0.00% |
| *Subgroup B: Human-reviewed Whisper draft (`rec2`–`rec7`, `rec9`–`rec10`)* | 8 | 42 | 36 | 30 | 35 | 32 | 3 | 6 | 0 | 1 | **83.33%** | 76.19% | 7.14% | 14.29% | 0.00% / 0.00% | 2.38% |

**Change vs. the previous run (84.31%, 43/51):** the rate is now **84.62% (44/52)**. Two things changed at once: the extraction prompt (1,888 → 2,802 chars, current `main`) and the reference transcripts (8 calls corrected by a human listener).

---

## 4. Per-Recording Results (All 10 Calls)

| Call ID | Reference Provenance | $H$ | $M$ | `both_absent` | **Req. Match Rate** | Exact | Semantic | Missing | Extra | Conflicting | Non-Exact Details |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `smartsemsar_rec1` | Manual human | 5 | 4 | 4 | **80.00%** (`4/5`) | 4 | 0 | 1 | 0 | 0 | `property_type` missing (`apartment` vs `unknown`) |
| `smartsemsar_rec2` | Human-reviewed Whisper draft | 5 | 5 | 4 | **100.00%** (`5/5`) | 4 | 1 | 0 | 0 | 0 | `location` semantic (`العاصمة الإدارية↔العاصمة الادارية`) |
| `smartsemsar_rec3` | Human-reviewed Whisper draft | 5 | 5 | 4 | **100.00%** (`5/5`) | 5 | 0 | 0 | 0 | 0 | All exact matches |
| `smartsemsar_rec4` | Human-reviewed Whisper draft | 5 | 4 | 4 | **60.00%** (`3/5`) | 3 | 0 | 1 | 0 | 1 | `location` conflict (`التجمع الخامس` vs `التجامعة الخامس`); `budget_max` missing (`20000.0` vs `None`) |
| `smartsemsar_rec5` | Human-reviewed Whisper draft | 6 | 4 | 3 | **66.67%** (`4/6`) | 4 | 0 | 2 | 0 | 0 | `property_type` missing (`villa` vs `unknown`); `deal_type` missing (`buy` vs `unknown`) |
| `smartsemsar_rec6` | Human-reviewed Whisper draft | 5 | 5 | 4 | **100.00%** (`5/5`) | 5 | 0 | 0 | 0 | 0 | All exact matches |
| `smartsemsar_rec7` | Human-reviewed Whisper draft | 6 | 5 | 3 | **83.33%** (`5/6`) | 4 | 1 | 1 | 0 | 0 | `location` semantic (`6 أكتوبر↔6 اكتوبر`); `property_type` missing (`villa` vs `unknown`) |
| `smartsemsar_rec8` | Manual human | 5 | 5 | 4 | **100.00%** (`5/5`) | 5 | 0 | 0 | 0 | 0 | All exact matches |
| `smartsemsar_rec9` | Human-reviewed Whisper draft *(4.8 dB noisy)* | 5 | 4 | 4 | **80.00%** (`4/5`) | 4 | 0 | 1 | 0 | 0 | `deal_type` missing (`buy` vs `unknown`) |
| `smartsemsar_rec10` | Human-reviewed Whisper draft | 5 | 4 | 4 | **80.00%** (`4/5`) | 3 | 1 | 1 | 0 | 0 | `location` semantic (`أكتوبر مثلاً الشيخ زايد↔أكتوبر مثلا الشيخ زايد`); `purpose` missing (`living` vs `unknown`) |

---

## 5. Per-Field Results (All 10 Calls)

| Field | $H$ | $M$ | `both_absent` | Exact | Semantic | Missing | Extra | Conflicting | **Req. Match Rate** | **Exact Rate** | **Semantic Rate** | **Missing Rate** | **Extra Rate (vs $H$)** | **Conflict Rate** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `budget_min` | 1 | 1 | 9 | 1 | 0 | 0 | 0 | 0 | **100.00%** | 100.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| `area_sqm` | 1 | 1 | 9 | 1 | 0 | 0 | 0 | 0 | **100.00%** | 100.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| `bedrooms` | 10 | 10 | 0 | 10 | 0 | 0 | 0 | 0 | **100.00%** | 100.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| `location` | 10 | 10 | 0 | 6 | 3 | 0 | 0 | 1 | **90.00%** | 60.00% | 30.00% | 0.00% | 0.00% | 10.00% |
| `budget_max` | 10 | 9 | 0 | 9 | 0 | 1 | 0 | 0 | **90.00%** | 90.00% | 0.00% | 10.00% | 0.00% | 0.00% |
| `deal_type` | 10 | 8 | 0 | 8 | 0 | 2 | 0 | 0 | **80.00%** | 80.00% | 0.00% | 20.00% | 0.00% | 0.00% |
| `property_type` | 9 | 6 | 1 | 6 | 0 | 3 | 0 | 0 | **66.67%** | 66.67% | 0.00% | 33.33% | 0.00% | 0.00% |
| `purpose` | 1 | 0 | 9 | 0 | 0 | 1 | 0 | 0 | **0.00%** | 0.00% | 0.00% | 100.00% | 0.00% | 0.00% |
| `notes` | 0 | 0 | 10 | 0 | 0 | 0 | 0 | 0 | **N/A** | N/A | N/A | N/A | N/A | N/A |

---

## 6. Root-Cause Evidence for Every Mismatch (8 Total Across All 10 Calls)

| # | Call | Field | Category | Reference vs. Machine Value | Transcript Evidence | Root Cause |
|---|---|---|---|---|---|---|
| 1 | `rec1` | `property_type` | `missing_requirement` | `"apartment"` vs `"unknown"` | Ref: `Client: عايز شقة في التجمع الخامس` → Mach: `Client: في التجمع الخامس` | **ASR error** (dropped `"عايز شقة"`) |
| 2 | `rec4` | `location` | `conflicting_value` | `"التجمع الخامس"` vs `"التجامعة الخامس"` | Ref: `في التجمع الخامس` → Mach: `في التجامعة الخامس` (3×) | **ASR error** (spelling corruption kept by the extractor) |
| 3 | `rec4` | `budget_max` | `missing_requirement` | `20000` vs `null` | Ref: `عشرين ألف` … `حدود عشرين ألف` → Mach: `عشرين الف` … `حدود عشر الف` | **ASR error + guardrail by design**: the machine text has two different budgets (20k / 10k), so the "ask, don't guess" rule leaves the budget empty and the bot asks the client instead of picking one |
| 4 | `rec5` | `property_type` | `missing_requirement` | `"villa"` vs `"unknown"` | Ref: `فيلا تمليك` → Mach: `بلا تمليك` (2×) | **ASR error** (`فيلا` → `بلا`) |
| 5 | `rec5` | `deal_type` | `missing_requirement` | `"buy"` vs `"unknown"` | Ref: `Broker: … فيلا تمليك …` → Mach: the same broker sentence is labeled `Client:` (`بلا تمليك`) | **Diarization + extraction**: `rec5` is the call with swapped roles (51.30% DER); with the broker's offer attributed to the client next to the misheard `بلا`, the extractor did not commit to `buy` |
| 6 | `rec7` | `property_type` | `missing_requirement` | `"villa"` vs `"unknown"` | Ref: `أنا محتاج فيلا في 6 أكتوبر` → Mach: `انا محتاج بيلا في 6 اكتوبر` | **ASR error** (`فيلا` → `بيلا`) |
| 7 | `rec9` | `deal_type` | `missing_requirement` | `"buy"` vs `"unknown"` | Ref: `Client: تمليك` → Mach: `تمليئك` / `تمليئ` (4.83 dB heavy noise) | **ASR error** (`تمليك` → `تمليئك`) |
| 8 | `rec10` | `purpose` | `missing_requirement` | `"living"` vs `"unknown"` | Neither transcript says the flat is for living or investment | **Extraction error on the reference side**: the LLM inferred `living` from the human text; the machine side correctly left it `unknown` |

**Attribution Summary:** Of the 8 mismatches, **5** are pure ASR errors (`"فيلا" → "بيلا"/"بلا"`, `"التجمع" → "التجامعة"`, `"تمليك" → "تمليئك"`, dropped `"عايز شقة"`), **1** is an ASR error that the guardrail turned into a clarification question (`rec4` budget), **1** comes from the swapped speaker roles on `rec5`, and **1** is an LLM over-inference on the reference transcript (`rec10` purpose).

---

## 7. Representative Matches

- **Numbers:** `bedrooms` matched **10/10**, `budget_max` **9/10**, `area_sqm` **1/1** (`180 sqm` on `rec5`) and `budget_min` **1/1**, even when the human text used Arabic words (`"خمستاشر ألف"`) and the machine text used digits (`"15000"`).
- **Semantic Arabic normalization (3 semantic matches):** `"العاصمة الإدارية"` ↔ `"العاصمة الادارية"` (`rec2`), `"6 أكتوبر"` ↔ `"6 اكتوبر"` (`rec7`), `"أكتوبر مثلاً الشيخ زايد"` ↔ `"أكتوبر مثلا الشيخ زايد"` (`rec10`).
- **Fixed since the previous run:** `rec1` `deal_type` (no longer a `buy`/`rent` conflict), `rec2` and `rec8` `property_type` (the broker-summary `"شقة"` is now extracted on both sides), `rec4` extra `budget_min` (gone).

---

## 8. Limitations

1. **Reference provenance:** only `rec1` and `rec8` were transcribed from scratch by a person (90.00%, `9/10`). The other 8 are Whisper drafts corrected by a human listener (83.33%, `35/42`); they are reported separately.
2. **Egyptian real-estate ASR vocabulary:** `property_type` (66.67%) is still the weakest field because `faster-whisper large-v3-turbo` hears `"فيلا"` as `"بيلا"`/`"بلا"`.
3. **Small sample:** 10 calls / 52 reference requirements, so one field flipping moves the rate by about 2 points.
4. **Unexercised fields:** `notes` was absent in all 10 calls; `purpose` appears only once (on the reference side of `rec10`).
