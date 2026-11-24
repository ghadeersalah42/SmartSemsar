import json
from pathlib import Path

import pytest

from backend.services.asr_corrections import correct_transcript

GT = Path(__file__).resolve().parents[1] / "backend/services/data/evaluation/ground_truth/transcripts"


@pytest.mark.parametrize("wrong, right", [
    ("انا محتاج بيلا في 6 اكتوبر", "انا محتاج فيلا في 6 اكتوبر"),
    ("لو حضرتك بتدوري على بلا تمليك", "لو حضرتك بتدوري على فيلا تمليك"),
    ("تمام يا فندم بلا خمس غرف", "تمام يا فندم فيلا خمس غرف"),
    ("شقة في التجامعة الخامس", "شقة في التجمع الخامس"),
    ("تلات غرف تمليئ في اكتوبر", "تلات غرف تمليك في اكتوبر"),
    ("مش دمليك ولا ايجار", "مش تمليك ولا ايجار"),
    ("عايز اتجار ولا تمليك؟", "عايز إيجار ولا تمليك؟"),
    ("بتدوري على بيجار أو تمليك", "بتدوري على إيجار أو تمليك"),
    ("مع حضرتك مكتب عقراط", "مع حضرتك مكتب عقارات"),
])
def test_known_mishearings_are_fixed(wrong, right):
    assert correct_transcript(wrong)[0] == right


@pytest.mark.parametrize("text", [
    "سيبها بلا سكر",                 # "without", not a villa
    "جه بلا ميعاد",
    "شغال في الاتجار بالعملة",        # "trading"
    "التجمع الخامس تمليك فيلا",       # already right
])
def test_real_words_are_not_touched(text):
    assert correct_transcript(text) == (text, [])


def test_human_reference_transcripts_are_not_changed():
    # the rules must only fix Whisper errors: every hand-checked sentence stays exactly the same
    files = sorted(GT.glob("smartsemsar_rec*_gt_transcript.json"))
    assert len(files) == 10
    for f in files:
        d = json.loads(f.read_text(encoding="utf-8"))
        for text in [d["reference_transcript"]] + [s["text"] for s in d["segments"]]:
            assert correct_transcript(text)[0] == text, f.name
