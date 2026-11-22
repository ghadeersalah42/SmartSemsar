# SmartSemsar Controlled Extraction-Agreement Evaluation Report (All 10 Recordings)

**Evaluation Date (UTC):** `2026-11-22T01:18:45Z`
**Evaluation Scope:** All 10 recordings (`smartsemsar_rec1`â€“`smartsemsar_rec10`), with transparent provenance breakdown separating manually transcribed calls (`rec1`, `rec8`) from user-provided multi-pass reference calls (`rec2`â€“`rec7`, `rec9`â€“`rec10`)
**Machine-Readable Report:** `backend/services/data/evaluation/reports/extraction_agreement_report.json`
**Archived Previous Reports:**
- Legacy Groq Whisper 10-call report (2026-11-21): `backend/services/data/evaluation/reports/archive/extraction_agreement_report_legacy_groq_20261121.json`
- Manual-human 2-call report (`rec1` & `rec8`, 2026-11-22): `backend/services/data/evaluation/reports/archive/extraction_agreement_report_manual_2calls_20261122.json`

---

## 1. Controlled Experiment Configuration

Both transcript versions were processed through the **exact same** extraction function, model, prompt, schema, temperature, and `Client:`/`Broker:` turn-formatting pipeline so that the **only variable** is the transcript source:

| Parameter | Value |
|---|---|
| **Command Executed** | `.venv\Scripts\python.exe backend/services/evaluation/extraction_agreement.py --all-manifest-calls` |
| **Extractor Function** | `backend.services.user_input_text.extract_requirements` (`raw_text=None`) |
| **Output Schema** | `backend.schema.state.CustomerRequirements` (9 fields: `location`, `budget_min`, `budget_max`, `area_sqm`, `bedrooms`, `property_type`, `purpose`, `deal_type`, `notes`) |
| **LLM Provider & Model** | Groq â€” `openai/gpt-oss-120b` (`SMARTSEMSAR_GROQ_MODEL`) |
| **Decoding Settings** | `temperature = 0`, `response_format = {"type": "json_object"}` |
| **System Prompt** | `backend.services.user_input_text.SYSTEM_PROMPT` (`1,888` chars) |
| **Machine Transcript Source** | Latest local `faster-whisper large-v3-turbo` (`int8`, CPU) + `pyannote/speaker-diarization-community-1` + context-based speaker role assignment (`conversations/smartsemsar_rec{1..10}_conversation.json`, batch run `20261122_000500`) |
| **Reference Transcript Source** | `ground_truth/transcripts/smartsemsar_rec{1..10}_gt_transcript.json` (`rec1`, `rec8`: manual human transcripts; `rec2`â€“`rec7`, `rec9`â€“`rec10`: user-provided multi-pass Whisper large-v3 + waveform-aligned references) |
| **Input Formatting Parity** | Both reference and machine conversations formatted identically as `Client: ...` / `Broker: ...` dialogue lines with consecutive same-role turns merged. |

---

## 2. Metric Definitions & Denominator

A field is **present** if its value is not `null`, `""`, or `"unknown"`. Fields absent in **both** extractions (`both_absent`) are **strictly excluded** from the Requirement Match Rate denominator so agreement is never inflated.

Let $H$ = total reference requirements present (`exact_match + semantic_match + missing_requirement + conflicting_value`) and $M$ = total machine-extracted requirements present (`exact_match + semantic_match + extra_requirement + conflicting_value`).

| Metric | Formula |
|---|---|
| **Requirement Match Rate** | $(\text{Exact} + \text{Semantic}) / H \times 100$ |
| **Exact Match Rate** | $\text{Exact} / H \times 100$ |
| **Semantic Match Rate** | $\text{Semantic} / H \times 100$ *(equivalent after Arabic/numeric/synonym normalization, not exact raw match)* |
| **Missing Requirement Rate** | $\text{Missing} / H \times 100$ |
| **Extra Requirement Rate** | $\text{Extra} / H \times 100$ *(and $\text{Extra} / M \times 100$ among machine extractions)* |
| **Conflicting Value Rate** | $\text{Conflicting} / H \times 100$ |

---

## 3. Overall & Provenance Subgroup Results

| Scope | Calls | $H$ | $M$ | Excluded `both_absent` | Matched ($E+S$) | Exact ($E$) | Semantic ($S$) | Missing | Extra | Conflicting | **Req. Match Rate** | **Exact Rate** | **Semantic Rate** | **Missing Rate** | **Extra Rate (vs $H$ / vs $M$)** | **Conflict Rate** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **All 10 Recordings (`rec1`â€“`rec10`)** | **10** | **51** | **46** | **38** | **43** | **39** | **4** | **6** | **1** | **2** | **84.31%** | **76.47%** | **7.84%** | **11.76%** | **1.96% / 2.17%** | **3.92%** |
| *Subgroup A: Manual Human (`rec1`, `rec8`)* | 2 | 10 | 8 | 8 | 7 | 7 | 0 | 2 | 0 | 1 | **70.00%** | 70.00% | 0.00% | 20.00% | 0.00% / 0.00% | 10.00% |
| *Subgroup B: User-Provided Multi-Pass Ref (`rec2`â€“`rec7`, `rec9`â€“`rec10`)* | 8 | 41 | 38 | 30 | 36 | 32 | 4 | 4 | 1 | 1 | **87.80%** | 78.05% | 9.76% | 9.76% | 2.44% / 2.63% | 2.44% |

---

## 4. Per-Recording Results (All 10 Calls)

| Call ID | Reference Provenance | $H$ | $M$ | `both_absent` | **Req. Match Rate** | Exact | Semantic | Missing | Extra | Conflicting | Non-Exact Details |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `smartsemsar_rec1` | Manual human | 5 | 4 | 4 | **60.00%** (`3/5`) | 3 | 0 | 1 | 0 | 1 | `property_type` missing; `deal_type` conflict (`buy` vs `rent`) |
| `smartsemsar_rec2` | Multi-pass ref | 5 | 4 | 4 | **80.00%** (`4/5`) | 3 | 1 | 1 | 0 | 0 | `location` semantic (`Ø§Ù„Ø¥Ø¯Ø§Ø±ÙŠØ©`â†”`Ø§Ù„Ø§Ø¯Ø§Ø±ÙŠØ©`); `property_type` missing |
| `smartsemsar_rec3` | Multi-pass ref | 5 | 5 | 4 | **100.00%** (`5/5`) | 5 | 0 | 0 | 0 | 0 | All exact matches |
| `smartsemsar_rec4` | Multi-pass ref | 5 | 6 | 3 | **80.00%** (`4/5`) | 4 | 0 | 0 | 1 | 1 | `location` conflict (`Ø§Ù„ØªØ¬Ù…Ø¹ Ø§Ù„Ø®Ø§Ù…Ø³` vs `Ø§Ù„ØªØ¬Ø§Ù…Ø¹Ø© Ø§Ù„Ø®Ø§Ù…Ø³`); `budget_min` extra (`10000`) |
| `smartsemsar_rec5` | Multi-pass ref | 6 | 5 | 3 | **83.33%** (`5/6`) | 5 | 0 | 1 | 0 | 0 | `property_type` missing (`villa` vs `unknown`) |
| `smartsemsar_rec6` | Multi-pass ref | 5 | 5 | 4 | **100.00%** (`5/5`) | 5 | 0 | 0 | 0 | 0 | All exact matches |
| `smartsemsar_rec7` | Multi-pass ref | 6 | 5 | 3 | **83.33%** (`5/6`) | 4 | 1 | 1 | 0 | 0 | `location` semantic (`6 Ø£ÙƒØªÙˆØ¨Ø±`â†”`6 Ø§ÙƒØªÙˆØ¨Ø±`); `property_type` missing (`villa` vs `unknown`) |
| `smartsemsar_rec8` | Manual human | 5 | 4 | 4 | **80.00%** (`4/5`) | 4 | 0 | 1 | 0 | 0 | `property_type` missing (`apartment` vs `unknown`) |
| `smartsemsar_rec9` | Multi-pass ref *(4.8 dB noisy)* | 5 | 4 | 4 | **80.00%** (`4/5`) | 3 | 1 | 1 | 0 | 0 | `location` semantic (`6 Ø£ÙƒØªÙˆØ¨Ø±`â†”`6 Ø§ÙƒØªÙˆØ¨Ø±`); `deal_type` missing (`buy` vs `unknown`) |
| `smartsemsar_rec10` | Multi-pass ref | 4 | 4 | 5 | **100.00%** (`4/4`) | 3 | 1 | 0 | 0 | 0 | `location` semantic (`Ù…Ø«Ù„Ø§Ù‹`â†”`Ù…Ø«Ù„Ø§`) |

---

## 5. Per-Field Results (All 10 Calls)

| Field | $H$ | $M$ | `both_absent` | Exact | Semantic | Missing | Extra | Conflicting | **Req. Match Rate** | **Exact Rate** | **Semantic Rate** | **Missing Rate** | **Extra Rate (vs $H$)** | **Conflict Rate** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `budget_max` | 10 | 10 | 0 | 10 | 0 | 0 | 0 | 0 | **100.00%** | 100.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| `bedrooms` | 10 | 10 | 0 | 10 | 0 | 0 | 0 | 0 | **100.00%** | 100.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| `budget_min` | 1 | 2 | 8 | 1 | 0 | 0 | 1 | 0 | **100.00%** | 100.00% | 0.00% | 0.00% | 100.00% (`1/1`) | 0.00% |
| `area_sqm` | 1 | 1 | 9 | 1 | 0 | 0 | 0 | 0 | **100.00%** | 100.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| `location` | 10 | 10 | 0 | 5 | 4 | 0 | 0 | 1 | **90.00%** | 50.00% | 40.00% | 0.00% | 0.00% | 10.00% |
| `deal_type` | 10 | 9 | 0 | 8 | 0 | 1 | 0 | 1 | **80.00%** | 80.00% | 0.00% | 10.00% | 0.00% | 10.00% |
| `property_type` | 9 | 4 | 1 | 4 | 0 | 5 | 0 | 0 | **44.44%** | 44.44% | 0.00% | 55.56% | 0.00% | 0.00% |
| `purpose` | 0 | 0 | 10 | 0 | 0 | 0 | 0 | 0 | N/A | N/A | N/A | N/A | N/A | N/A |
| `notes` | 0 | 0 | 10 | 0 | 0 | 0 | 0 | 0 | N/A | N/A | N/A | N/A | N/A | N/A |

---

## 6. Root-Cause Evidence for Every Mismatch (9 Total Across All 10 Calls)

| # | Call | Field | Category | Reference vs. Machine Value | Exact Transcript Evidence | Root Cause |
|---|---|---|---|---|---|---|
| 1 | `rec1` | `property_type` | `missing_requirement` | `"apartment"` vs `"unknown"` | Ref: `Client: Ø¹Ø§ÙŠØ² Ø´Ù‚Ø© ÙÙŠ Ø§Ù„ØªØ¬Ù…Ø¹ Ø§Ù„Ø®Ø§Ù…Ø³` â†’ Mach: `Client: ÙÙŠ Ø§Ù„ØªØ¬Ù…Ø¹ Ø§Ù„Ø®Ø§Ù…Ø³` | **ASR error** (dropped `"Ø¹Ø§ÙŠØ² Ø´Ù‚Ø©"`) |
| 2 | `rec1` | `deal_type` | `conflicting_value` | `"buy"` vs `"rent"` | Ref: `Client: Ø¥ÙŠØ¬Ø§Ø± Ù„Ø§ ØªÙ…Ù„ÙŠÙƒ Ù„Ùˆ Ø³Ù…Ø­Øª \| Broker: ØªÙ…Ù„ÙŠÙƒØŒ Ø·Ø¨...` â†’ Mach: `Client: Ø§ØªØ¬Ø§Ø± Ù„Ø§ ØªÙ…Ù„ÙŠÙƒ \| Broker: ØªÙ…Ù„ÙŠÙƒØŸ Ø·Ø¨...` | **ASR + Extraction error** (`"Ø§ØªØ¬Ø§Ø± Ù„Ø§ ØªÙ…Ù„ÙŠÙƒ"` misread as `"rent"`) |
| 3 | `rec2` | `property_type` | `missing_requirement` | `"apartment"` vs `"unknown"` | Both transcripts contain `Broker: Ù…Ø§Ø´ÙŠ Ø´Ù‚Ø© 3 ØºØ±Ù ØªÙ…Ù„ÙŠÙƒ...` verbatim (client never says `"Ø´Ù‚Ø©"`). | **Extraction error** (LLM sensitivity on broker-only summary) |
| 4 | `rec4` | `location` | `conflicting_value` | `"Ø§Ù„ØªØ¬Ù…Ø¹ Ø§Ù„Ø®Ø§Ù…Ø³"` vs `"Ø§Ù„ØªØ¬Ø§Ù…Ø¹Ø© Ø§Ù„Ø®Ø§Ù…Ø³"` | Ref: `ÙÙŠ Ø§Ù„ØªØ¬Ù…Ø¹ Ø§Ù„Ø®Ø§Ù…Ø³` â†’ Mach: `ÙÙŠ Ø§Ù„ØªØ¬Ø§Ù…Ø¹Ø© Ø§Ù„Ø®Ø§Ù…Ø³` (3Ã—) | **ASR error** (spelling corruption preserved verbatim by extractor) |
| 5 | `rec4` | `budget_min` | `extra_requirement` | `null` vs `10000.0` | Ref: `Client: Ø­Ø¯ÙˆØ¯ Ø¹Ø´Ø±ÙŠÙ† Ø£Ù„Ù` â†’ Mach: `Client: Ø­Ø¯ÙˆØ¯ Ø¹Ø´Ø± Ø§Ù„Ù` (while earlier turn had `Ø¹Ø´Ø±ÙŠÙ† Ø§Ù„Ù`, creating a `10kâ€“20k` range) | **ASR error** (`Ø¹Ø´Ø±ÙŠÙ†` â†’ `Ø¹Ø´Ø±`) |
| 6 | `rec5` | `property_type` | `missing_requirement` | `"villa"` vs `"unknown"` | Ref: `ÙÙŠÙ„Ø§ ØªÙ…Ù„ÙŠÙƒ` â†’ Mach: `Ø¨Ù„Ø§ ØªÙ…Ù„ÙŠÙƒ` (2Ã—) | **ASR error** (`ÙÙŠÙ„Ø§` â†’ `Ø¨Ù„Ø§`) |
| 7 | `rec7` | `property_type` | `missing_requirement` | `"villa"` vs `"unknown"` | Ref: `Ø£Ù†Ø§ Ù…Ø­ØªØ§Ø¬ ÙÙŠÙ„Ø§ ÙÙŠ 6 Ø£ÙƒØªÙˆØ¨Ø±` â†’ Mach: `Ø§Ù†Ø§ Ù…Ø­ØªØ§Ø¬ Ø¨ÙŠÙ„Ø§ ÙÙŠ 6 Ø§ÙƒØªÙˆØ¨Ø±` (2Ã—) | **ASR error** (`ÙÙŠÙ„Ø§` â†’ `Ø¨ÙŠÙ„Ø§`) |
| 8 | `rec8` | `property_type` | `missing_requirement` | `"apartment"` vs `"unknown"` | Ref: `Broker: Ø®Ù„Ø§Øµ Ø¨Ù†ØªÙƒÙ„Ù… Ø¹Ù„Ù‰ Ø´Ù‚Ø© ØºØ±ÙØªÙŠÙ†...` â†’ Mach: `Broker: Ø­Ù„Ùˆ Ø³ÙŠÙ†Ø§ØªÙƒÙ„Ù… Ø¹Ù„Ù‰ Ø´Ù‚Ø© ØºØ±ÙØªÙŠÙ†...` (`"Ø´Ù‚Ø©"` present verbatim in both) | **Extraction error** (LLM sensitivity on broker-only summary) |
| 9 | `rec9` | `deal_type` | `missing_requirement` | `"buy"` vs `"unknown"` | Ref: `Client: ØªÙ…Ù„ÙŠÙƒ \| Broker: ØªÙ…Ù„ÙŠÙƒØŒ ØµØ­ØŸ` â†’ Mach: `Broker: ØªÙ…Ù„ÙŠØ¦Ùƒ ØµØ­ØŸ ... Ø´Ù‚Ø© ØªÙ„Ø§Øª ØºØ±Ù ØªÙ…Ù„ÙŠØ¦` (4.83 dB heavy noise) | **ASR error** (`ØªÙ…Ù„ÙŠÙƒ` â†’ `ØªÙ…Ù„ÙŠØ¦Ùƒ`/`ØªÙ…Ù„ÙŠØ¦`) |

**Attribution Summary:** Of the 9 mismatches across 10 calls, **6 (66.7%)** are pure ASR transcription errors (notably Egyptian `"ÙÙŠÙ„Ø§" â†’ "Ø¨ÙŠÙ„Ø§"/"Ø¨Ù„Ø§"` and `"Ø§Ù„ØªØ¬Ù…Ø¹" â†’ "Ø§Ù„ØªØ¬Ø§Ù…Ø¹Ø©"`), **1 (11.1%)** is combined ASR + extraction (`rec1` self-correction), and **2 (22.2%)** are pure LLM extraction boundary sensitivity on broker-only closing summaries (`rec2`, `rec8`).

---

## 7. Representative Matches

- **Exact numeric normalization (`rec6`, `rec8`, `rec5`, `rec7`):** `budget_max` matched **10/10 (100%)** (`4M`, `2.5M`, `25k`, `20k`, `12M`, `8k`, `10M`, `15k`, `5M`, `6M`), `bedrooms` matched **10/10 (100%)**, and `area_sqm` matched **1/1 (100%)** (`180 sqm` on `rec5`), even when human text used Arabic words (`"Ø®Ù…Ø³ØªØ§Ø´Ø± Ø£Ù„Ù"`) and machine text used digits (`"15000"`).
- **Semantic Arabic normalization (`4` semantic matches):** `location` matched semantically across hamza/tanween spelling variants (`"Ø§Ù„Ø¹Ø§ØµÙ…Ø© Ø§Ù„Ø¥Ø¯Ø§Ø±ÙŠØ©"` â†” `"Ø§Ù„Ø¹Ø§ØµÙ…Ø© Ø§Ù„Ø§Ø¯Ø§Ø±ÙŠØ©"` in `rec2`, `"6 Ø£ÙƒØªÙˆØ¨Ø±"` â†” `"6 Ø§ÙƒØªÙˆØ¨Ø±"` in `rec7` & `rec9`, `"Ø£ÙƒØªÙˆØ¨Ø± Ù…Ø«Ù„Ø§Ù‹ Ø§Ù„Ø´ÙŠØ® Ø²Ø§ÙŠØ¯"` â†” `"Ø£ÙƒØªÙˆØ¨Ø± Ù…Ø«Ù„Ø§ Ø§Ù„Ø´ÙŠØ® Ø²Ø§ÙŠØ¯"` in `rec10`).

---

## 8. Limitations

1. **Reference Provenance Distinction:** Only `rec1` and `rec8` are independently manual human transcripts (`70.00%` match rate, `7/10`). The other 8 calls (`rec2`â€“`rec7`, `rec9`â€“`rec10`, `87.80%` match rate, `36/41`) use user-provided multi-pass Whisper large-v3 + waveform-aligned references, which are reported separately for full transparency.
2. **Egyptian Real-Estate ASR Vocabulary Errors:** `property_type` (`44.44%`) is the main bottleneck because local `faster-whisper large-v3-turbo` (`int8`) frequently transcribes `"ÙÙŠÙ„Ø§"` as `"Ø¨ÙŠÙ„Ø§"`/`"Ø¨Ù„Ø§"` (`rec5`, `rec7`) and dropped `"Ø¹Ø§ÙŠØ² Ø´Ù‚Ø©"` (`rec1`).
3. **Unexercised Fields:** `purpose` and `notes` were absent across all 10 calls (`both_absent = 10`).
