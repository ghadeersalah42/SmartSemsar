# SmartSemsar Speech Pipeline Evaluation Report (Days 3–5)

**Generated:** 2026-11-22 · **Updated:** 2026-11-24 (references for `rec2`–`rec7`, `rec9`–`rec10` corrected by a human listener; WER/CER and extraction re-run on the current `main` prompt)
**Project:** SmartSemsar
**Evaluation Artifacts Directory:** `backend/services/data/evaluation/`

---

## 1. Executive Summary & Task Status

| Component | Status | Key Measured Results |
|---|---|---|
| **1. Audio Dataset Standardization, Silero VAD & Quality Audit (10 calls)** | **COMPLETED (10/10)** | **329.956 s (5.499 min)** total standardized 16 kHz mono WAV audio; **249.787 s (4.163 min)** Silero VAD active speech; 0 clipping; 4 genuinely noisy recordings `< 20 dB` rough SNR (`rec9` at `4.83 dB`, `rec5`, `rec10`, `rec8`). |
| **2. Ground Truth & Timestamp Alignment (10 calls)** | **COMPLETED (10/10, all human-checked)** | `smartsemsar_rec1` & `smartsemsar_rec8` are manual human transcripts with millisecond waveform timestamps; `rec2`–`rec7` & `rec9`–`rec10` started as multi-pass Whisper large-v3 drafts and were **listened to and corrected by a human reviewer** (every `[Unclear]` resolved; `rec3` and `rec9` confirmed correct as-is). |
| **3. Speech Recognition (ASR) Evaluation (10 calls)** | **COMPLETED (10/10)** | **Local `faster-whisper large-v3-turbo` (`pipeline.py`):** Macro WER **37.90%**, Micro WER **38.18%**, Macro CER **16.39%**, Micro CER **16.31%** (`605` ref words / `3,176` ref chars). **`faster-whisper-small` baseline:** Macro WER **65.69%**, Micro WER **64.96%**, Macro CER **32.04%**, Micro CER **30.76%**. |
| **4. Audio Quality, 5 dB Noise & Phone Filter Experiments** | **COMPLETED (2 calls × 16 conditions × 2 models)** | Controlled multi-seed reproduction on **clean `rec1` (32.42 dB)** and **noisy `rec8` (19.88 dB)** (all augmented files explicitly labeled `SIMULATED`). Empirically explains non-monotonic 5 dB / 10 dB reversals and Phone Filter effects. |
| **5. Requirements Extraction Agreement (10 calls)** | **COMPLETED (10/10)** | **`large-v3-turbo` machine transcript vs. human reference:** **84.62% (44/52) of the client's requirements match** (78.85% exact + 5.77% semantic; 13.46% missing, 1.92% conflicting, 0 extra). Per field: `bedrooms` 100%, `area_sqm` 100%, `budget_min` 100%, `location` 90%, `budget_max` 90%, `deal_type` 80%, `property_type` 66.67%. Full detail: `EXTRACTION_AGREEMENT_REPORT.md`. |
| **6. Speaker Diarization (DER) & Broker/Client Role Accuracy (10 calls)** | **COMPLETED (10/10)** | **`pyannote/speaker-diarization-community-1` (`collar = 0.25 s`):** Macro DER **16.37%**, Micro DER **16.11%** (`5.14%` missed speech, `0.83%` false alarm, `10.14%` speaker confusion across `197.03 s` ref speech). **Role Accuracy:** **90.00% macro / micro** (**18/20 speakers**, **9/10 calls at 100%**). Batch QC: **10/10 passed QC** (`1 PASS, 9 WARNING, 0 FAIL, 0 ERROR`). |

> **Provenance & Integrity Statement:**
> - Original recordings in `backend/services/data/raw_calls/` and pilot files in `backend/services/data/evaluation/pilot/` were never modified.
> - `smartsemsar_rec1` and `smartsemsar_rec8` Ground Truth files contain **human-verified manual transcripts** enriched with waveform timestamps.
> - `smartsemsar_rec2`–`rec7` and `rec9`–`rec10` Ground Truth files started as the multi-pass Whisper large-v3 + waveform-aligned package and were then **listened to and corrected by a human reviewer on 2026-11-24** (`annotation_source: human_reviewed_whisper_multipass_waveform_aligned` in `manifest.json`). Timestamps and speaker roles were kept; only wording was fixed.
> - No pyannote diarization outputs, DER scores, or role accuracy metrics were fabricated.

---

## 2. Audio Dataset & Objective Quality Audit

All 10 recordings in `backend/services/data/raw_calls/` (`smartsemsar_rec1.mp4` .. `smartsemsar_rec10.mp4`) are provided dataset recordings (`is_simulated: false`). Every file standardized cleanly from 44.1 kHz mono MP4 to 16 kHz mono 16-bit PCM WAV (`0` clipped samples, `0.000 s` duration drift).

- **Total Audio Duration Tested:** **329.956 seconds (5.499 minutes)**
- **Total Active Speech Duration (Silero VAD):** **249.787 seconds (4.163 minutes)** (75.7% mean speech ratio)

*Note: `rough_snr_estimate_db` is a rough VAD speech-vs-nonspeech RMS energy ratio, not a calibrated SNR measurement.*

| Call ID | Origin | Duration (s) | VAD Speech (s / %) | RMS (dBFS) | Noise Floor RMS | Rough VAD SNR (dB) | Energy <300 Hz | Energy 300–3400 Hz | Relative Noise Tier |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `smartsemsar_rec9` | Provided recording | 32.345 | 24.336 (75.2%) | -25.74 | `0.032445` | **4.83** | 28.0% | 71.6% | **Noisy (Top-4 noisiest)** |
| `smartsemsar_rec5` | Provided recording | 29.396 | 25.320 (86.1%) | -34.95 | `0.002312` | **18.41** | 21.5% | 71.6% | **Noisy (Top-4 noisiest)** |
| `smartsemsar_rec10` | Provided recording | 43.352 | 31.764 (73.3%) | -30.91 | `0.003538` | **19.44** | 27.6% | 71.8% | **Noisy (Top-4 noisiest)** |
| `smartsemsar_rec8` | Provided recording | 24.892 | 20.494 (82.3%) | -31.61 | `0.002932` | **19.88** | 18.4% | 79.0% | **Noisy (Top-4 noisiest)** |
| `smartsemsar_rec6` | Provided recording | 33.739 | 21.424 (63.5%) | -25.69 | `0.004876` | 22.51 | 3.4% | 95.7% | Moderate |
| `smartsemsar_rec3` | Provided recording | 31.463 | 25.061 (79.7%) | -23.80 | `0.004648` | 23.84 | 3.6% | 95.0% | Moderate |
| `smartsemsar_rec4` | Provided recording | 39.636 | 30.866 (77.9%) | -28.10 | `0.002297` | 25.76 | 11.0% | 88.9% | Moderate |
| `smartsemsar_rec7` | Provided recording | 29.931 | 23.453 (78.4%) | -24.85 | `0.001679` | 31.71 | 3.8% | 95.8% | Cleaner |
| `smartsemsar_rec1` | Provided recording | 33.855 | 23.268 (68.7%) | -27.11 | `0.001272` | 32.42 | 13.9% | 85.8% | Cleaner |
| `smartsemsar_rec2` | Provided recording | 31.347 | 23.801 (75.9%) | -30.25 | `0.000370` | 39.57 | 20.2% | 78.3% | Cleanest |
| **Total** | **10 provided calls** | **329.956 s (5.499 min)** | **249.787 s (4.163 min)** | — | — | **4.83–39.57 dB** | — | — | **4 calls < 20 dB** |

**Simulated / Augmented Audio Identification:** All 30 modified WAVs created in Section 5 (`15` from `rec1`, `15` from `rec8`) are stored exclusively in `backend/services/data/evaluation/experiments/augmented_audio/*_SIMULATED_*.wav` with `"is_simulated": true` metadata and are never mixed into the 10-call real dataset scores.

---

## 3. Speech Recognition (ASR) Evaluation — All 10 Calls

Evaluated using `normalize_arabic_text` from `backend/services/pipeline.py` (official primary report: `reports/wer_cer_report.json`; offline baseline report: `reports/wer_cer_faster_whisper_small_report.json`).

### Per-Call and Corpus-Wide WER & CER (`605` Reference Words, `3,176` Reference Characters)

| Call ID | GT Provenance | Ref Words | `pipeline.py` (`large-v3-turbo`) WER | `pipeline.py` (`large-v3-turbo`) CER | `faster-whisper-small` WER | `faster-whisper-small` CER | WER Gain (`turbo` vs `small`) |
|---|---|---:|---:|---:|---:|---:|---:|
| `smartsemsar_rec1` | Manual human + waveform TS | 61 | 42.62% | 15.79% | 63.93% | 35.20% | -21.31 pp |
| `smartsemsar_rec2` | Human-reviewed Whisper draft + TS | 54 | **25.93%** | 9.93% | 68.52% | 26.82% | -42.59 pp |
| `smartsemsar_rec3` | Human-reviewed Whisper draft + TS | 62 | 40.32% | 16.77% | 45.16% | 14.87% | -4.84 pp |
| `smartsemsar_rec4` | Human-reviewed Whisper draft + TS | 72 | 41.67% | 14.48% | 58.33% | 22.79% | -16.67 pp |
| `smartsemsar_rec5` | Human-reviewed Whisper draft + TS | 62 | 29.03% | **8.07%** | 48.39% | 16.43% | -19.35 pp |
| `smartsemsar_rec6` | Human-reviewed Whisper draft + TS | 52 | 42.31% | 19.71% | 67.31% | 31.90% | -25.00 pp |
| `smartsemsar_rec7` | Human-reviewed Whisper draft + TS | 56 | 33.93% | 10.99% | **91.07%*** | **79.49%*** | **-57.14 pp** |
| `smartsemsar_rec8` | Manual human + waveform TS | 49 | 34.69% | 23.72% | 75.51% | 38.32% | -40.82 pp |
| `smartsemsar_rec9` | Human-reviewed Whisper draft + TS *(noisy 4.8 dB)* | 59 | 47.46% | 24.76% | 74.58% | 31.92% | -27.12 pp |
| `smartsemsar_rec10` | Human-reviewed Whisper draft + TS | 78 | 41.03% | 19.70% | 64.10% | 22.69% | -23.08 pp |
| **Macro Average (10 calls)** | — | — | **37.90%** | **16.39%** | **65.69%** | **32.04%** | **-27.79 pp** |
| **Micro / Corpus (Weighted)** | — | **605** | **38.18%** (231/605) | **16.31%** (518/3176) | **64.96%** (393/605) | **30.76%** (977/3176) | **-26.78 pp** |

*\*On `smartsemsar_rec7`, `faster-whisper-small` collapsed and dropped most of the call (`91.07%` WER), whereas `large-v3-turbo` transcribed it cleanly (`33.93%` WER / `10.99%` CER).*

*Effect of the human review (2026-11-24): fixing the 8 references changed corpus WER from `38.29%` to `38.18%` and CER from `16.42%` to `16.31%` (598 → 605 reference words), so the earlier unreviewed references were close but not exact.*

### Key ASR Takeaways
1. **Local `faster-whisper large-v3-turbo` (`pipeline.py`) decisively outperforms `faster-whisper-small`** across all 10 calls: **-26.78 pp lower corpus WER** (`64.96% → 38.18%`) and **-14.45 pp lower corpus CER** (`30.76% → 16.31%`).
2. **WER vs. CER Gap (`38.18%` WER vs. `16.31%` CER):** Because `normalize_arabic_text` preserves punctuation (`؟`, `،`, `.`) and Arabic numeral-vs-word spelling (`4` vs `اربعة`, `3` vs `ثلاث`, `يا فن` vs `يا فندم`), word-level WER is roughly 2× higher than character-level CER even when semantic meaning is preserved.
3. **Human-Verified Subset (`rec1` + `rec8`, 110 words):** `large-v3-turbo` averaged `38.66%` WER / `19.76%` CER vs. `faster-whisper-small` at `69.72%` WER / `36.76%` CER.

---

## 4. Speaker Diarization (DER) & Broker/Client Role Accuracy

### Ground Truth Readiness: **10 / 10 Ready & Schema-Validated**
All 10 calls now have complete, cross-file-consistent diarization turn timestamps (`ground_truth/diarization/smartsemsar_rec{1..10}_gt_diarization.json`) and canonical speaker role mappings (`SPK_1 = client`, `SPK_2 = broker` in `ground_truth/roles/smartsemsar_rec{1..10}_gt_roles.json`), verified with `0` schema errors in `run_evaluation.py`.

### Evaluation Metric & Collar
- **Diarization Error Rate (DER):** $(\text{Missed Speech} + \text{False Alarm} + \text{Speaker Confusion}) / \text{Total Scored Reference Speech}$, using a **0.25 s forgiveness collar** (`--collar 0.25`) around reference turn boundaries and exact optimal 1-to-1 permutation mapping between neutral pyannote IDs (`SPEAKER_00`, `SPEAKER_01`) and reference IDs (`SPK_1`, `SPK_2`).

### Evaluation Results — All 10 Calls (`pyannote/speaker-diarization-community-1`, `collar = 0.25 s`)

| Call ID | Ref Speech (s) | DER | Missed Speech | False Alarm | Speaker Confusion | Predicted Role Mapping | Role Accuracy | Pipeline QC |
|---|---:|---:|---:|---:|---:|---|---:|---|
| `smartsemsar_rec1` | 17.36 | **4.45%** | 4.18% | 0.00% | 0.27% | `SPEAKER_00: broker`, `SPEAKER_01: client` | **100% (2/2)** | WARNING |
| `smartsemsar_rec2` | 19.16 | 16.76% | 8.38% | 0.00% | 8.38% | `SPEAKER_00: broker`, `SPEAKER_01: client` | **100% (2/2)** | WARNING |
| `smartsemsar_rec3` | 18.01 | 13.88% | 1.03% | 0.00% | 12.85% | `SPEAKER_00: client`, `SPEAKER_01: broker` | **100% (2/2)** | WARNING |
| `smartsemsar_rec4` | 24.01 | **2.11%** | 2.11% | 0.00% | 0.00% | `SPEAKER_00: broker`, `SPEAKER_01: client` | **100% (2/2)** | WARNING |
| `smartsemsar_rec5` | 22.78 | **51.30%** | 4.44% | 0.00% | 46.85% | `SPEAKER_00: broker`, `SPEAKER_01: client` *(inverted due to split broker)* | **0% (0/2)** | PASS |
| `smartsemsar_rec6` | 15.35 | **33.66%** | 14.58% | 0.99% | 18.09% | `SPEAKER_00: broker`, `SPEAKER_01: client` | **100% (2/2)** | WARNING |
| `smartsemsar_rec7` | 18.10 | **7.82%** | 4.57% | 0.00% | 3.25% | `SPEAKER_00: broker`, `SPEAKER_01: client` | **100% (2/2)** | WARNING |
| `smartsemsar_rec8` | 15.10 | 12.17% | 4.25% | 6.51% | 1.40% | `SPEAKER_00: client`, `SPEAKER_01: broker` | **100% (2/2)** | WARNING |
| `smartsemsar_rec9` | 19.74 | 16.36% | 4.91% | 2.54% | 8.91% | `SPEAKER_00: broker`, `SPEAKER_01: client` | **100% (2/2)** | WARNING |
| `smartsemsar_rec10` | 27.42 | **5.17%** | 5.17% | 0.00% | 0.00% | `SPEAKER_00: client`, `SPEAKER_01: broker` | **100% (2/2)** | WARNING |
| **Macro Average (10 calls)** | — | **16.37%** | — | — | — | — | **90.00%** | — |
| **Micro / Corpus (Weighted)** | **197.03** | **16.11%** | **5.14%** | **0.83%** | **10.14%** | **18 / 20 speakers correct** | **90.00%** | **10/10 Passed** |

### Diarization & Role Assignment Findings
- **Strong performance on 8 of 10 calls:** DER ranges from **2.11% (`rec4`)** and **4.45% (`rec1`)** to **16.76% (`rec2`)**, with **0.00% speaker confusion** on `rec4` and `rec10` and **100% broker/client role accuracy across 9/10 calls** (including calls where neutral labels flip between `SPEAKER_00` and `SPEAKER_01`).
- **`smartsemsar_rec5` (51.30% DER, 0/2 Role Accuracy):** Extreme talk-time imbalance—the broker speaks **24.44 s (96.7%)** while the client speaks only a single **0.84 s turn (`"12 مليون"`, 3.3%)**. Forced to cluster into 2 speakers (`min_speakers=2, max_speakers=2`), pyannote splits the broker chronologically (`SPEAKER_01` first half including `"12 مليون"`, `SPEAKER_00` second half closing remarks), causing 46.85% speaker confusion and inverting downstream role assignment.
- **`smartsemsar_rec6` (33.66% DER, 100% Role Accuracy):** The closing turn (`31.27–33.35 s`, 2.08 s) is **-45.8 dBFS** (20.7 dB quieter than the `-25.1 dBFS` call body), causing both Silero VAD and pyannote to miss it (`14.58%` missed speech), while two acoustically similar speakers with rapid `<1 s` turns cause `18.09%` speaker confusion. Filtering sub-100 ms diarization glitches and Whisper EOF silence-overflow hallucinations (`"اشتركوا في القناة"`, `29.20–59.18 s`) upgraded `rec6` QC from `FAIL` to `WARNING`.

---

## 5. Audio Quality, 5 dB Noise & Phone Filter Experiments

No prior 5 dB noise or Phone Filter (`1.10%`) scripts or logs existed in the repository (`original_experiment_artifacts_found_in_repo: false`). Controlled multi-seed reproductions (`seeds = 42, 123, 2026`, active-VAD-speech RMS SNR calibration) were executed on **two human-verified calls**:
1. **Clean call `smartsemsar_rec1`** (`32.42 dB` rough SNR, 61 words) — `experiments/noise_and_phone_filter_results.json`
2. **Noisy call `smartsemsar_rec8`** (`19.88 dB` rough SNR, 49 words) — `experiments/noise_and_phone_filter_results_smartsemsar_rec8.json`

### Controlled Comparison Table (Real Baseline vs. `SIMULATED` Conditions)

| Condition | Simulated? | Seeds | `rec1` (Clean) `turbo` WER / CER | `rec1` (Clean) `small` WER / CER | `rec8` (Noisy) `turbo` WER / CER | `rec8` (Noisy) `small` WER / CER |
|---|---|---:|---:|---:|---:|---:|
| **`baseline_original_real`** | **No (Real)** | 1 | **47.54% / 18.42%** | **63.93% / 35.20%** | **42.86% / 28.10%** | **75.51% / 38.32%** |
| `phone_bandpass_300_3400hz` | Yes | 1 | **45.90%** / 19.41% | 73.77% / 43.75% | 46.94% / **25.91%** | **71.43% / 25.91%** |
| `phone_8khz_narrowband` | Yes | 1 | 49.18% / 20.07% | **62.30%** / 39.80% | 44.90% / **25.91%** | **69.39%** / 41.24% |
| `white_noise_20db` | Yes | 1 | 47.54% / 22.04% | **93.44%** / 70.39%* | 51.02% / 29.20% | 75.51% / 41.24% |
| `white_noise_10db` | Yes | 3 | 65.03% ± 1.55% / 33.55% | **80.33% ± 1.34%** / 51.32%* | 50.34% ± 5.36% / **25.06%** | **89.12% ± 5.09%** / 58.88%* |
| **`white_noise_5db`** | Yes | 3 | **81.42% ± 2.04%** / 49.23% | 93.44% ± 5.83% / 72.70% | 53.74% ± 2.55% / 30.54% | **82.99% ± 5.36%** / 54.14%* |
| `white_noise_0db` | Yes | 1 | 95.08% / 62.17% | 100.00% / 70.72% | 77.55% / 41.97% | 91.84% / 64.60% |
| `pink_noise_10db` | Yes | 1 | 47.54% / 21.05% | 68.85% / 46.05% | **40.82% / 25.55%**† | 79.59% / 41.24% |
| **`pink_noise_5db`** | Yes | 3 | 67.76% ± 4.70% / 38.05% | 86.89% ± 13.97% / 67.65% | 55.78% ± 5.85% / **26.28%** | 80.27% ± 1.92% / 49.27% |
| `phone_8khz_plus_pink_10db` | Yes | 1 | 63.93% / 35.20% | 81.97% / 56.58% | 57.14% / 27.01% | 79.59% / 43.43% |
| `phone_8khz_plus_pink_5db` | Yes | 1 | 78.69% / 49.01% | 96.72% / 67.11% | 69.39% / 45.99% | 85.71% / 48.91% |

### Evidence-Based Explanations (Grounded Strictly in Measurements)
1. **Why a prior report showed "5 dB noise better than clean audio":**
   - Across both calls and 3 random seeds, **5 dB noise consistently degraded WER vs. clean baseline** (`+10.9` to `+33.9 pp` on `turbo`, `+4.8` to `+29.5 pp` on `small`).
   - However, our controlled measurements **directly reproduced three real non-monotonic reversals on short clips**:
     - (`*`) On `rec1` (`small`), **10 dB white noise (`80.33%`) beat 20 dB white noise (`93.44%`) by 13.11 pp** because 20 dB triggered a hallucination/word-drop failure (`36` deleted words).
     - (`*`) On `rec8` (`small`), **5 dB white noise (`82.99% ± 5.36%`) beat 10 dB white noise (`89.12% ± 5.09%`) by 6.13 pp**!
     - (`†`) On `rec8` (`turbo`), **10 dB pink noise (`40.82%` WER / `25.55%` CER) beat the unmodified baseline (`42.86%` WER / `28.10%` CER)**, and **5 dB pink noise CER (`26.28%`) remained below baseline CER (`28.10%`)**!
     - Seed variance on single 50–60 word calls is huge (up to **31.15 pp range / 13.97% std** on `rec1` `small` at 5 dB pink). Any single-clip, single-seed experiment can flip direction due to Whisper decoding instability rather than genuine noise immunity.
2. **Why a prior report showed "Phone Filter = 1.10%":**
   - Real recordings already concentrate **71.6%–95.8%** of their energy in `300–3400 Hz`, with `13.9%–28.0%` low-frequency rumble (`<300 Hz`) on noisy calls.
   - Stripping `<300 Hz` rumble with `phone_bandpass_300_3400hz` **lowered noise floor RMS by 36.9% on `rec1`** (`32.42 → 35.58 dB` rough SNR) and **improved `rec8` CER on BOTH models** (`turbo`: `28.10% → 25.91%`; `small`: `38.32% → 25.91%`, fixing `"مصرق جديدة"` → `"مصر الجديدة"`!), while shifting `rec1` `turbo` by only **`-1.64 pp` WER / `+0.99 pp` CER** (`+1.64 pp` for 8 kHz narrowband).
   - **Conclusion:** `1.10%` is **not** an absolute Arabic WER on real-estate phone calls (real phone-filtered WER is `44.9%–49.2%` on `turbo` and `62.3%–73.8%` on `small`). It is only consistent with a **~1 percentage-point delta** (e.g., the `+0.99 pp` to `±1.64 pp` shifts measured here) or an unrelated clean synthetic sample absent from this repository.

---

## 6. Requirements Extraction Agreement — All 10 Calls

`extract_requirements()` (`backend/services/user_input_text.py`, Groq `openai/gpt-oss-120b`, `temperature=0`, current `main` prompt of 2,802 chars) was run on **both** the machine transcript (`large-v3-turbo` + pyannote roles) and the human reference transcript of every call, with identical `Client:`/`Broker:` formatting. A requirement counts when the reference extraction has it (not `null`/`unknown`); fields empty on both sides are excluded. Command: `python backend/services/evaluation/extraction_agreement.py --all-manifest-calls --pause 25` (the pause keeps every call on Groq under its 8,000 tokens/min limit).

| Scope | Requirements in reference | Matched | **Match rate** |
|---|---:|---:|---:|
| **All 10 calls** | **52** | **44** (41 exact + 3 semantic) | **84.62%** |
| Manual human references (`rec1`, `rec8`) | 10 | 9 | 90.00% |
| Human-reviewed references (other 8) | 42 | 35 | 83.33% |

| Field | Match rate |
|---|---:|
| `bedrooms` | 10/10 (100%) |
| `area_sqm` | 1/1 (100%) |
| `budget_min` | 1/1 (100%) |
| `location` | 9/10 (90%) |
| `budget_max` | 9/10 (90%) |
| `deal_type` | 8/10 (80%) |
| `property_type` | 6/9 (66.67%) |
| `purpose` | 0/1 (0%) |

**The 8 mismatches:** 5 are ASR errors (`فيلا` → `بيلا`/`بلا` on `rec5` and `rec7`, `التجمع` → `التجامعة` on `rec4`, `تمليك` → `تمليئك` on noisy `rec9`, dropped `عايز شقة` on `rec1`); 1 is an ASR error the guardrail turned into a question (`rec4`: the machine text has both 20k and 10k, so the budget is left empty and the bot asks instead of guessing); 1 comes from the swapped speaker roles on `rec5`; 1 is the LLM over-inferring `purpose = living` on the `rec10` reference. Per-call tables and evidence: `EXTRACTION_AGREEMENT_REPORT.md`.

---

## 6b. Whisper Vocabulary-Prompt Experiment (`initial_prompt`)

Most extraction mismatches came from Whisper mishearing a few real-estate words, so we tested giving `large-v3-turbo` a list of expected words (`initial_prompt`), the same idea the app already uses in `user_input_audio.py`. Script: `backend/services/evaluation/vocab_prompt_experiment.py`; outputs: `data/evaluation/experiments/vocab_prompt/`. Only the prompt changes (same model, int8 CPU, `beam_size=5`, same 16 kHz WAVs); speaker roles are copied from the official conversations by time overlap, so diarization stays fixed.

| Variant | Micro WER | WER without punctuation | `تمليك` heard right (of 17) | `التجمع` (of 8) | `فيلا` (of 5) | Notes |
|---|---:|---:|---:|---:|---:|---|
| No prompt (control, re-run on this machine) | 38.35% | 26.73% | 13 | 5 | 1 | reproduces the official 38.18% within run-to-run noise |
| App prompt (`user_input_audio.py`) | 43.64% | 31.02% | 12 | 8 | 3 | **hallucinated foreign text on `rec2`** ("Connect Yoga! …") |
| **Extended prompt (now the pipeline default)** | 40.83% | **27.23%** | **17** | **8** | 1 | no hallucination |

- **Why WER went up:** the prompt contains punctuation, so Whisper starts writing `،` `.` `؟`, and `normalize_arabic_text` keeps punctuation, so each mark counts as an error. Without punctuation the extended prompt is on par with no prompt (27.23% vs 26.73%). About 11 points of the official 38% WER are punctuation, not misheard words.
- **Extraction (what the prompt is for):** with the human-side extraction held fixed, the extended-prompt transcripts match **46/52 (88.46%)** vs **44/52 (84.62%)** for the official transcripts; against a second human-side run, 44/52 vs 42/52. Gained: `rec4` location (`التجامعة` → `التجمع`) and budget (`عشرين`), `rec5` and `rec9` deal type (`تمليك`). Lost: `rec8` property type. `فيلا` is still misheard.
- **Caveats:** some areas in the prompt (التجمع، 6 أكتوبر، الشيخ زايد، العاصمة الإدارية) occur in these 10 calls, so the gain may be optimistic; Groq's extraction itself varies by about ±4 points between identical runs (the human-side extraction of the same text changed between runs), which is the same size as the gain. A larger call set is needed to confirm it.
- **Decision:** the extended list is now `pipeline.WHISPER_INITIAL_PROMPT` and is passed to `transcribe()`; set `SMARTSEMSAR_WHISPER_PROMPT=""` to turn it off. The official numbers in sections 3 and 6 were produced **before** this change and were not regenerated.

---

## 7. Conclusions & Remaining Limitations

1. **End-to-End Evaluation Complete (10/10 Calls):** All 10 recordings passed pipeline QC (`1 PASS, 9 WARNING, 0 FAIL, 0 ERROR`) and evaluated cleanly across WER/CER (`38.18%` micro WER / `16.31%` micro CER), DER (`16.11%` micro DER at `collar = 0.25 s`), Role Accuracy (`90.00%`, 18/20 speakers), and Audio Quality with `0` validation errors.
2. **Main Engineering Improvement Opportunities Identified:**
   - **High-pass / Phone-band preprocessing (`300–3400 Hz`)** is beneficial on rumble-heavy noisy recordings (`rec8` CER improved by `-2.19 pp` on `turbo` and `-12.41 pp` on `small`).
   - **Extreme talk-time imbalance & very quiet tail turns (`rec5`, `rec6`):** Standard 2-speaker acoustic clustering struggles when one speaker talks `<1 s` (3.3% of the call in `rec5`) or when a turn drops to `-45.8 dBFS` (`rec6`).
   - **Self-correction & broker-vs-client rules in extraction prompt:** Leveraging the newly predicted `"broker"` / `"client"` turn roles in downstream extraction can further reduce confusion on broker-only suggestions (`rec2`, `rec5`) and client self-corrections (`rec1`).
3. **Reference transcripts:** all 10 are human-checked (2 transcribed by hand, 8 Whisper drafts corrected by a human listener). DER and role accuracy use timestamps and speaker labels only, so the wording review does not change them.
