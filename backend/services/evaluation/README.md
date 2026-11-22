# SmartSemsar Speech Evaluation Guide

This directory contains the speech evaluation toolkit for the 10 recorded real-estate phone calls (`smartsemsar_rec1` .. `smartsemsar_rec10`) in `backend/services/data/raw_calls/`.

---

## 1. Runtime Requirements

Before running the Speech Pipeline on real recordings:

- **Python dependencies:** `torch`, `torchaudio`, `faster-whisper`, `pyannote.audio`, `librosa`, `soundfile`, `imageio-ffmpeg`, `python-dotenv`, `pydantic`, `numpy`, `pytest`.
- **FFmpeg:** Either install `imageio-ffmpeg` (recommended) or install `ffmpeg` and add it to `PATH`.
- **`HF_TOKEN` (required for diarization):** Set `HF_TOKEN` in your environment, `SmartSemsar/.env`, or `backend/services/.env`, and accept the Hugging Face model terms for `pyannote/speaker-diarization-community-1`.
- **`GEMINI_API_KEY` (optional):** Required only when running customer profile extraction (i.e., without `--skip-profile` or with `--with-profile`). Never commit `.env` or hardcode tokens.

---

## 2. Running Processing & Evaluation

### Process One Recording
```bash
python backend/services/pipeline.py --call-id smartsemsar_rec1 --ext mp4 --skip-profile
```

### Process All 10 Recordings (Batch, Reusing Loaded Models)
```bash
python backend/services/batch_pipeline.py
```
Or directly via `pipeline.py`:
```bash
python backend/services/pipeline.py --batch --ext mp4 --skip-profile
```

**Pipeline Exit Code Policy (Single-Call and Batch):**
- `0`: Processing completed with QC `PASS` or `WARNING` (0 processing errors, 0 QC `FAIL`).
- `1`: Processing/runtime error (missing audio file, missing env vars, exception).
- `2`: Processing completed, but one or more calls failed QC (`qc_status == "FAIL"`).

### Run Evaluation
```bash
python backend/services/evaluation/run_evaluation.py
python backend/services/evaluation/run_evaluation.py --calls smartsemsar_rec1 smartsemsar_rec2 --collar 0.25
```

**Evaluation Exit Code Policy:**
- `0`: Completed with 0 evaluation errors (including when calls are `PENDING_GROUND_TRUTH`, `PENDING_ROLE_PREDICTIONS`, `EVALUATED`, or `PARTIALLY_EVALUATED`).
- `1`: One or more calls or `manifest.json` had an `ERROR` (malformed JSON, schema violation, invalid timestamps, placeholder roles, or speaker ID mismatch).
- `2`: `--strict` was passed and zero calls were evaluated across WER/CER, DER, and role accuracy.

---

## 3. Ground-Truth Annotation Workflow

All 10 calls are tracked in `backend/services/data/evaluation/ground_truth/manifest.json`. `run_evaluation.py` reads `manifest.json` for call discovery and verifies that file naming follows `<call_id>_gt_{transcript,diarization,roles}.json`.

### Step-by-Step Human Annotation Process
1. Listen to each raw recording in `backend/services/data/raw_calls/smartsemsar_rec{1..10}.mp4` (never overwrite raw audio).
2. Assign canonical neutral reference speaker IDs per call (e.g., `SPK_1`, `SPK_2`) consistently across all three ground-truth files for that call.
3. Copy the corresponding template file into the same directory, rename it using the exact call ID, **remove `"_is_template": true`**, set `"status": "ready"`, and fill in `"annotator"`:
   - `ground_truth/transcripts/<call_id>_gt_transcript.json`
   - `ground_truth/diarization/<call_id>_gt_diarization.json`
   - `ground_truth/roles/<call_id>_gt_roles.json`
4. Update `manifest.json` (`"ground_truth_ready": true`, `"status": "ready"`) once verified.

> **Integrity Rules:**
> - Never fabricate ground-truth annotations or metric values.
> - Only `"broker"` and `"client"` are accepted as final role values. Template files (`_is_template: true`), pending files (`status: "pending"`), `null` roles, and placeholder strings (such as `broker_or_client` or `client_or_broker`) are automatically rejected.
> - **Never infer speaker roles from pyannote's neutral IDs (`SPEAKER_00`, `SPEAKER_01`).** Human annotators must determine who is the broker and who is the client by listening to the call.

---

## 4. Ground-Truth Schemas & Speaker ID Mapping

### Transcript GT (`<call_id>_gt_transcript.json`)
```json
{
  "schema_version": "1.0",
  "call_id": "smartsemsar_rec1",
  "status": "ready",
  "language": "ar",
  "annotator": "annotator_name",
  "reference_transcript": "النص المرجعي الكامل للمكالمة",
  "segments": [
    {
      "start": 0.5,
      "end": 3.2,
      "speaker_id": "SPK_1",
      "role": "client",
      "text": "السلام عليكم"
    }
  ]
}
```

### Diarization GT (`<call_id>_gt_diarization.json`)
```json
{
  "schema_version": "1.0",
  "call_id": "smartsemsar_rec1",
  "status": "ready",
  "annotator": "annotator_name",
  "num_speakers": 2,
  "turns": [
    {
      "start": 0.5,
      "end": 3.2,
      "speaker_id": "SPK_1",
      "role": "client"
    }
  ]
}
```

### Roles GT (`<call_id>_gt_roles.json`)
```json
{
  "schema_version": "1.0",
  "call_id": "smartsemsar_rec1",
  "status": "ready",
  "annotator": "annotator_name",
  "allowed_roles": ["broker", "client"],
  "speaker_roles": {
    "SPK_1": "client",
    "SPK_2": "broker"
  },
  "pipeline_speaker_to_reference_speaker": {}
}
```

### How Speaker IDs and Roles Are Mapped
- **Across Ground-Truth files:** All three files use the same canonical reference speaker IDs (`SPK_1`, `SPK_2`). `run_evaluation.py` validates that every `speaker_id` and `role` in `gt_diarization` and `gt_transcript` matches `gt_roles.speaker_roles`.
- **To Pipeline Outputs (`SPEAKER_00`, `SPEAKER_01`):**
  - For **DER**, `compute_der()` computes the optimal 1-to-1 permutation mapping between neutral pipeline cluster IDs and reference `speaker_id`s based purely on timestamp overlap.
  - For **Broker/Client Role Accuracy**, if `pipeline_speaker_to_reference_speaker` is provided in `gt_roles`, that explicit mapping is used; otherwise `run_evaluation.py` aligns pipeline IDs via DER's objective `optimal_speaker_mapping`. If pipeline conversations still carry `"unassigned"` roles, role accuracy is explicitly reported as `PENDING_ROLE_PREDICTIONS` rather than fabricating a score.

---

## 5. Metrics: Macro vs. Micro (Corpus-Level) Aggregation

Every evaluation report distinguishes **macro** and **micro (corpus-level)** metrics:

| Metric | Macro Average (Unweighted) | Micro / Corpus-Level (Weighted) |
|---|---|---|
| **WER** | Mean of per-call `wer` values (`sum(wer_i) / N`) | `total_word_edits / total_reference_words` across all evaluated calls |
| **CER** | Mean of per-call `cer` values (`sum(cer_i) / N`) | `total_char_edits / total_reference_characters` across all evaluated calls |
| **DER** | Mean of per-call `der` values (`sum(der_i) / N`) | `(total_missed + total_false_alarm + total_confusion) / total_reference_speech_sec` |
| **Role Accuracy** | Mean of per-call `role_accuracy` values | `total_correct_roles / total_role_items` across all evaluated calls |

- **Macro** treats every call equally regardless of length (useful to spot outlier calls).
- **Micro (corpus-level)** weights calls proportionally by word count, character count, speech duration, or item count (standard for overall ASR/diarization benchmark reporting).

### Audio Quality & Rough SNR Note
`analyze_audio_quality()` computes signal-level metrics from standardized 16 kHz mono WAV audio and Silero VAD timestamps (`rms_dbfs`, `peak_dbfs`, `crest_factor_db`, `clipping_ratio`, `speech_ratio`, `rough_snr_estimate_db`). `rough_snr_estimate_db` is explicitly labeled as a **rough estimate** based on VAD speech vs. non-speech RMS energy ratio and is **not** a calibrated or reliable measured SNR.

---

## 6. Metrics Currently Pending Human Annotations

Until human annotations are completed for `smartsemsar_rec1` .. `smartsemsar_rec10` (and real pipeline runs are executed):
- **WER & CER:** `PENDING_GROUND_TRUTH` (`null`)
- **DER:** `PENDING_GROUND_TRUTH` (`null`)
- **Broker/Client Role Accuracy:** `PENDING_GROUND_TRUTH` / `PENDING_ROLE_PREDICTIONS` (`null`)
- **Audio Quality Analysis:** `PENDING_PIPELINE_OUTPUTS` (runs automatically once standardized WAVs and VAD JSONs are generated by the pipeline)
