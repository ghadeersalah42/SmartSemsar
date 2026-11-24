# Per-word speakers (word_timestamps) — full-pipeline experiment, 2026-11-24

Goal: bring back the Whisper real-estate word list (`REAL_ESTATE_WORD_LIST`) without breaking speaker roles.
With the list, Whisper wrote calls as 1–2 long segments and roles are given per segment, so roles fell to 70 %.
Change tested: Whisper returns word timestamps, each word takes the pyannote speaker talking at that moment,
and a segment that holds two speakers is split where the speaker changes (`split_segments_by_speaker`).

Full `batch_pipeline.py` runs on the 10 real calls (pyannote, role inference by Groq, QC), scored against the
human-checked references:

| Run | Word list | Per-word speakers | Micro WER | Micro CER | DER | Roles | QC |
|---|---|---|---:|---:|---:|---:|---|
| Official (Sara's pipeline) | off | off | 38.18 % | 16.31 % | 16.11 % | 90 % | 10/10 |
| List only (`vocab_prompt/full_pipeline_run`) | on | off | 41.49 % | 18.70 % | 16.11 % | 70 % | 8/10 |
| **A** | on | on | 40.50 % | 18.04 % | 16.11 % | 95 % | 10/10 |
| **B** | off | on | 38.35 % | 16.66 % | 16.11 % | 80 % | 10/10 |

Per-call WER, A vs official: better on rec3 (40.3 → 27.4), rec4 (41.7 → 26.4), rec7 (33.9 → 26.8);
worse on rec1 (42.6 → 55.7), rec6 (42.3 → 65.4), rec8 (34.7 → 53.1). Most of the WER gap is punctuation the prompt adds.

## Roles are not stable on the new conversations

Roles are inferred by Groq from the conversation text. Re-running only that step 3 times on the same files:

| Call | Official conversation | A conversation | B conversation |
|---|---|---|---|
| rec5 | same answer 3/3 | 1 of 3 correct-looking, 2 of 3 "broker + broker" | flips 2 vs 1 |
| rec6 | same answer 3/3 (correct) | correct 1 of 3 | wrong 3/3 |

So A's 95 % is partly luck (rec6 came out right on that run, but only 1 time in 3 on reruns), and B's 80 % shows
the per-word split on its own makes rec6 worse. Likely reason: per-word speakers inherit pyannote's speaker confusion
(rec6 has 18 % confusion, rec5 47 %), while one speaker per segment took a majority over the segment and smoothed it.

## Decision

Not adopted. `WORD_LEVEL_SPEAKERS` stays in the code, off by default (`SMARTSEMSAR_WORD_LEVEL_SPEAKERS=1` turns it on).
What would make it work: smooth the per-word speakers (e.g. ignore speaker runs shorter than ~1 s or 3 words) and make
role inference deterministic (e.g. majority over several runs, or the heuristic as a tie-breaker), then re-test on
more than 10 calls. Extraction was not re-run on these transcripts.
