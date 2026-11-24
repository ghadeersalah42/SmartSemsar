# Correction dictionary after Whisper — experiment, 2026-11-24

`backend/services/asr_corrections.py` fixes known Whisper mishearings of Egyptian real-estate words in the
**machine** transcript before requirement extraction: بيلا/بلا → فيلا (بلا only in real-estate context),
التجامعة → التجمع, تمليئك/تمليئ/دمليك/تمليل → تمليك, بيجار/اتجار → إيجار (اتجار only before ولا/لا or after
عايز/ولا), عقراط/عكرات/عقاعد → عقارات. Whisper, pyannote and the speaker roles are untouched.

Safety checks (`tests/test_asr_corrections.py`): the rules change **none** of the 10 human reference transcripts
(every sentence), and real words are left alone ("بلا سكر", "الاتجار بالعملة").

## Results on the official machine transcripts (same 10 calls)

| | Without dictionary (official) | With dictionary |
|---|---:|---:|
| Micro WER | 38.18 % | **35.70 %** |
| Micro CER | 16.31 % | **15.52 %** |
| Extraction match, against the official human-side extraction | 84.62 % (44/52) | **96.15 % (50/52)** |
| Extraction match, against this run's human-side extraction | 85.71 % (42/49) | **95.92 % (47/49)** |

Command: `extraction_agreement.py --all-manifest-calls --pause 25 --asr-corrections` (all 20 extractions on Groq,
no Ollama fallback). Both comparisons hold the human side fixed, so the gain is not Groq run-to-run noise.

Fixed: rec4 location (التجامعة) and budget, rec5 property type (villa) and deal type (buy), rec7 property type
(villa), rec9 deal type (buy). Still wrong: rec1 property type (Whisper dropped "عايز شقة"), rec10 purpose
(the extractor guesses "living" on the reference side).

## Caveats

- Most forms are general Egyptian ASR confusions already listed in the pipeline's role prompt, but some
  (التجامعة, تمليئك, اتجار) were added after seeing these 10 calls, so the gain on new calls will likely be smaller.
  It must be re-measured on calls that were not used to write the rules.
- The brief asks for extraction on the machine transcript as it is; the official number stays the raw one
  (84.62 %), and this is reported as a separate improvement.
