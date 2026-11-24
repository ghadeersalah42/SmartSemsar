"""Unit tests for SmartSemsar Speech Pipeline, batch orchestration, QC, and evaluation.

Note: Model-dependent tests use deterministic mocks for Whisper, Silero VAD, and
pyannote; they verify pipeline logic, QC, and evaluation behavior without claiming
real Whisper or pyannote inference execution.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import struct
import types
import wave

import numpy as np
import pytest

from backend.services import batch_pipeline, pipeline
from backend.services.evaluation.metrics import (
    analyze_audio_quality,
    compute_cer,
    compute_corpus_cer,
    compute_corpus_der,
    compute_corpus_role_accuracy,
    compute_corpus_wer,
    compute_der,
    compute_role_accuracy,
    compute_wer,
)
from backend.services.evaluation.run_evaluation import (
    EVAL_EXIT_ERROR,
    EVAL_EXIT_NO_EVALUATED,
    EVAL_EXIT_OK,
    evaluate_calls,
    get_evaluation_exit_code,
)


def _write_test_wav(path: Path, sr: int = 16000, channels: int = 1, duration: float = 2.0, freq: float = 440.0):
    """Write a synthetic 16-bit PCM WAV file for deterministic tests."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n_frames = int(sr * duration)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        samples = []
        for i in range(n_frames):
            amp = 0.25 if i < n_frames // 2 else 0.01
            val = int(max(-32767, min(32767, amp * 32767 * math.sin(2 * math.pi * freq * i / sr))))
            for _ in range(channels):
                samples.append(val)
        wf.writeframes(struct.pack(f"<{len(samples)}h", *samples))


def _make_out_dirs(tmp_path: Path, raw_dir: Path) -> dict[str, Path]:
    return {
        "data": tmp_path,
        "raw_audio": raw_dir,
        "processed_audio": tmp_path / "processed_audio",
        "speech_segments": tmp_path / "processed_audio" / "speech_segments",
        "transcripts": tmp_path / "transcripts",
        "vad": tmp_path / "transcripts" / "vad",
        "diarization": tmp_path / "transcripts" / "diarization",
        "conversations": tmp_path / "transcripts" / "conversations",
        "qc": tmp_path / "transcripts" / "qc",
        "profiles": tmp_path / "transcripts" / "profiles",
        "batch_logs": tmp_path / "batch_logs",
        "evaluation": tmp_path / "evaluation",
    }


# ------------------------------------------------------------------ models & Arabic normalization
def test_preserved_model_constants():
    assert pipeline.WHISPER_MODEL == "large-v3-turbo"
    assert pipeline.DIARIZATION_MODEL == "pyannote/speaker-diarization-community-1"
    assert pipeline.EXPECTED_SPEAKERS == 2
    assert pipeline.TARGET_SAMPLE_RATE == 16000
    assert pipeline.TARGET_CHANNELS == 1


def test_normalize_arabic_text_utf8_preserved():
    # Tests alef hamza/madda (أ، إ، آ -> ا), alef maksura (ى -> ي), tatweel (ـ),
    # Arabic diacritics (fathatan, shadda, kasra, damma), and whitespace collapsing.
    raw = "أهلاً إزيك آه شقــة في التجمّع   الأولى بِسعرٍ مُناسب"
    expected = "اهلا ازيك اه شقة في التجمع الاولي بسعر مناسب"
    normalized = pipeline.normalize_arabic_text(raw)
    assert normalized == expected
    assert normalized.encode("utf-8").decode("utf-8") == expected


# ------------------------------------------------------------------ audio path & extension handling
def test_normalize_and_resolve_audio_path_extensions(tmp_path):
    (tmp_path / "smartsemsar_rec1.mp4").write_bytes(b"fake-mp4")
    (tmp_path / "call_00.ogg").write_bytes(b"fake-ogg")

    # Standard stem + ext
    cid1, path1 = pipeline.resolve_audio_path("smartsemsar_rec1", ext="mp4", raw_audio_dir=tmp_path)
    assert cid1 == "smartsemsar_rec1"
    assert path1 == tmp_path / "smartsemsar_rec1.mp4"

    # Explicit filename without --ext preserves filename extension (.mp4, not default .ogg)
    cid1_auto, path1_auto = pipeline.resolve_audio_path(
        "smartsemsar_rec1.mp4", ext=None, raw_audio_dir=tmp_path
    )
    assert cid1_auto == "smartsemsar_rec1"
    assert path1_auto == tmp_path / "smartsemsar_rec1.mp4"

    # Explicit filename with matching --ext succeeds
    cid1_match, path1_match = pipeline.resolve_audio_path(
        "smartsemsar_rec1.mp4", ext=".mp4", raw_audio_dir=tmp_path
    )
    assert cid1_match == "smartsemsar_rec1"
    assert path1_match == tmp_path / "smartsemsar_rec1.mp4"

    # Explicit full path resolves directly
    cid_full, path_full = pipeline.resolve_audio_path(
        str(tmp_path / "smartsemsar_rec1.mp4"), ext=None, raw_audio_dir=tmp_path
    )
    assert cid_full == "smartsemsar_rec1"
    assert path_full == tmp_path / "smartsemsar_rec1.mp4"

    # OGG default preserved when call_id has no extension and ext=None
    cid2, path2 = pipeline.resolve_audio_path("call_00", ext=None, raw_audio_dir=tmp_path)
    assert cid2 == "call_00"
    assert path2 == tmp_path / "call_00.ogg"


def test_normalize_audio_path_rejects_mismatched_extension(tmp_path):
    (tmp_path / "smartsemsar_rec1.mp4").write_bytes(b"fake-mp4")
    with pytest.raises(ValueError, match="extension mismatch"):
        pipeline.normalize_call_id_and_ext("smartsemsar_rec1.mp4", ext="ogg")
    with pytest.raises(ValueError, match="extension mismatch"):
        pipeline.resolve_audio_path("smartsemsar_rec1.mp4", ext="ogg", raw_audio_dir=tmp_path)


def test_resolve_audio_path_missing_file_clear_error(tmp_path):
    (tmp_path / "smartsemsar_rec1.mp4").write_bytes(b"fake")
    with pytest.raises(FileNotFoundError) as exc_info:
        pipeline.resolve_audio_path("smartsemsar_rec99", ext="mp4", raw_audio_dir=tmp_path)
    msg = str(exc_info.value)
    assert "smartsemsar_rec99.mp4" in msg
    assert "smartsemsar_rec1.mp4" in msg


def test_validate_environment_missing_vars(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(pipeline, "HF_TOKEN", None)
    monkeypatch.setattr(pipeline, "GEMINI_API_KEY", None)

    with pytest.raises(RuntimeError, match="HF_TOKEN"):
        pipeline.validate_environment(require_hf=True, require_gemini=False)

    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        pipeline.validate_environment(require_hf=False, require_gemini=True)

    monkeypatch.setenv("HF_TOKEN", "dummy-test-token")
    pipeline.validate_environment(require_hf=True, require_gemini=False)


def test_discover_calls_natural_order(tmp_path):
    for i in range(1, 11):
        (tmp_path / f"smartsemsar_rec{i}.mp4").write_bytes(b"x")
    discovered = pipeline.discover_calls(raw_audio_dir=tmp_path, ext="mp4")
    assert discovered == [f"smartsemsar_rec{i}" for i in range(1, 11)]


# ------------------------------------------------------------------ diarization & alignment
def test_run_diarization_enforces_two_speakers_and_supports_custom_bounds(monkeypatch, tmp_path):
    wav_path = tmp_path / "sample.wav"
    _write_test_wav(wav_path, sr=16000, channels=1, duration=1.0)

    captured_kwargs = {}

    class _FakeSeg:
        def __init__(self, start, end):
            self.start = start
            self.end = end

    class _FakeAnnotation:
        def itertracks(self, yield_label=True):
            yield _FakeSeg(0.0, 0.5), None, "SPEAKER_00"
            yield _FakeSeg(0.5, 1.0), None, "SPEAKER_01"

    class _FakeOutput:
        exclusive_speaker_diarization = _FakeAnnotation()

    class _FakeDiarPipeline:
        def __call__(self, audio_input, **kwargs):
            captured_kwargs.update(kwargs)
            assert audio_input["sample_rate"] == 16000
            return _FakeOutput()

    class _FakeTensor:
        def __init__(self, arr):
            self.arr = arr

        def unsqueeze(self, axis):
            return np.expand_dims(self.arr, axis=axis)

    class _FakeTorch:
        @staticmethod
        def from_numpy(arr):
            return _FakeTensor(arr)

    class _FakeSF:
        @staticmethod
        def read(_path, dtype="float32"):
            return np.zeros(16000, dtype=np.float32), 16000

    monkeypatch.setattr(
        pipeline,
        "_require_module",
        lambda name, pip_name=None: (
            _FakeSF if name == "soundfile" else (_FakeTorch if name == "torch" else __import__(name))
        ),
    )

    turns, speakers = pipeline.run_diarization(
        wav_path, "cpu", diarization_pipeline=_FakeDiarPipeline()
    )
    assert captured_kwargs["min_speakers"] == 2
    assert captured_kwargs["max_speakers"] == 2
    assert speakers == ["SPEAKER_00", "SPEAKER_01"]
    assert turns[0]["start"] == 0.0 and turns[0]["end"] == 0.5

    # Configurable speaker bounds work too
    pipeline.run_diarization(
        wav_path, "cpu", min_speakers=1, max_speakers=3, diarization_pipeline=_FakeDiarPipeline()
    )
    assert captured_kwargs["min_speakers"] == 1
    assert captured_kwargs["max_speakers"] == 3


def test_assign_speaker_overlap_threshold_and_ambiguity_flagging():
    turns = [
        {"start": 0.0, "end": 1.2, "speaker": "SPEAKER_00"},
        {"start": 1.2, "end": 2.7, "speaker": "SPEAKER_01"},
        {"start": 2.7, "end": 4.2, "speaker": "SPEAKER_00"},
        {"start": 6.0, "end": 6.03, "speaker": "SPEAKER_01"},
    ]
    # Segment [0.0, 4.2]: SPEAKER_00 total overlap = 2.7s vs SPEAKER_01 = 1.5s
    # Dominance = 2.7 / 4.2 = 0.6429 < 0.65 -> flagged ambiguous multi-speaker!
    spk, ov, details = pipeline.assign_speaker(0.0, 4.2, turns, return_details=True)
    assert spk == "SPEAKER_00"
    assert round(ov, 3) == 2.7
    assert details["is_ambiguous"] is True
    assert details["assignment_status"] == "assigned_ambiguous_multi_speaker"
    assert details["overlapping_speakers"] == ["SPEAKER_00", "SPEAKER_01"]

    # Segment [6.0, 8.0] overlaps SPEAKER_01 by only 0.03s (< min_overlap_sec 0.05s) -> rejected!
    spk_low, ov_low, details_low = pipeline.assign_speaker(6.0, 8.0, turns, return_details=True)
    assert spk_low is None
    assert round(ov_low, 3) == 0.03
    assert details_low["assignment_status"] == "unassigned_below_threshold"

    # Segment with NO overlap must NOT guess a speaker
    spk_none, ov_none, details_none = pipeline.assign_speaker(10.0, 12.0, turns, return_details=True)
    assert spk_none is None
    assert ov_none == 0.0
    assert details_none["assignment_status"] == "unassigned_no_overlap"


def test_build_conversation_marks_unassigned_and_keeps_neutral_labels():
    segments = [
        {"start": 0.0, "end": 2.0, "text": "ألو السلام عليكم", "normalized_text": "الو السلام عليكم"},
        {"start": 2.5, "end": 4.0, "text": "وعليكم السلام", "normalized_text": "وعليكم السلام"},
        {"start": 8.0, "end": 9.5, "text": "مفيش صوت", "normalized_text": "مفيش صوت"},
    ]
    turns = [
        {"start": 0.0, "end": 2.1, "speaker": "SPEAKER_01"},
        {"start": 2.4, "end": 4.2, "speaker": "SPEAKER_00"},
    ]
    conv = pipeline.build_conversation(segments, turns)
    assert len(conv) == 3
    assert conv[0]["speaker"] == "SPEAKER_01"
    assert conv[0]["speaker_assigned"] is True
    assert conv[1]["speaker"] == "SPEAKER_00"
    assert conv[1]["speaker_assigned"] is True
    assert conv[2]["speaker"] == pipeline.UNASSIGNED_SPEAKER
    assert conv[2]["speaker_assigned"] is False
    assert conv[2]["assignment_status"] == "unassigned_no_overlap"


# ------------------------------------------------------------------ QC improvements
def test_run_qc_timestamp_severity_vad_vs_transcription_and_configurable_speakers(tmp_path):
    f_wav = tmp_path / "rec1_standardized.wav"
    f_wav.write_bytes(b"nonempty")

    # 1) Minor EOF overshoot (10.3s vs 10.0s duration) -> WARNING, not FAIL
    conv_minor = [
        {"turn_id": 1, "start": 0.0, "end": 5.0, "speaker": "SPEAKER_00", "speaker_assigned": True, "text": "اهلا"},
        {"turn_id": 2, "start": 5.0, "end": 10.3, "speaker": "SPEAKER_01", "speaker_assigned": True, "text": "تمام"},
    ]
    qc_minor = pipeline.run_qc(
        "rec1",
        raw_duration=10.0,
        processed_duration=10.0,
        speakers=["SPEAKER_00", "SPEAKER_01", "SPEAKER_02"],
        conversation=conv_minor,
        processed_sample_rate=16000,
        processed_channels=1,
        expected_speakers=3,
        vad_result={"total_speech_duration": 8.2, "speech_ratio": 0.82},
        expected_output_files={"standardized_wav": f_wav},
    )
    assert qc_minor["checks"]["speakers"] == "PASS"  # Uses configurable expected_speakers=3!
    assert qc_minor["expected_speakers"] == 3
    assert qc_minor["checks"]["timestamps"] == "WARNING"
    assert qc_minor["minor_timestamp_overshoots"] == 1
    assert qc_minor["severe_invalid_timestamps"] == 0
    assert qc_minor["vad_speech_duration"] == 8.2
    assert qc_minor["transcription_segment_duration"] == 10.3
    assert qc_minor["vad_speech_duration"] != qc_minor["transcription_segment_duration"]
    assert qc_minor["status"] == "WARNING"

    # 2) Severe inverted timestamp (start >= end) -> FAIL
    conv_severe = [
        {"turn_id": 1, "start": 5.0, "end": 4.0, "speaker": "SPEAKER_00", "speaker_assigned": True, "text": "اهلا"},
    ]
    qc_severe = pipeline.run_qc(
        "rec1",
        raw_duration=10.0,
        processed_duration=10.0,
        speakers=["SPEAKER_00", "SPEAKER_01"],
        conversation=conv_severe,
        processed_sample_rate=16000,
        processed_channels=1,
        expected_output_files={"standardized_wav": f_wav},
    )
    assert qc_severe["checks"]["timestamps"] == "FAIL"
    assert qc_severe["severe_invalid_timestamps"] == 1
    assert qc_severe["status"] == "FAIL"


def test_run_qc_multi_speaker_overlap_policy(tmp_path):
    f_wav = tmp_path / "rec1_standardized.wav"
    f_wav.write_bytes(b"nonempty")

    # 1 of 3 turns has multi-speaker overlap (non-ambiguous) -> WARNING
    conv_warn = [
        {"turn_id": 1, "start": 0.0, "end": 2.0, "speaker": "SPEAKER_00", "speaker_assigned": True,
         "overlapping_speakers": ["SPEAKER_00"], "is_ambiguous": False, "text": "الو"},
        {"turn_id": 2, "start": 2.0, "end": 4.0, "speaker": "SPEAKER_01", "speaker_assigned": True,
         "overlapping_speakers": ["SPEAKER_00", "SPEAKER_01"], "is_ambiguous": False, "text": "ايوه"},
        {"turn_id": 3, "start": 4.0, "end": 6.0, "speaker": "SPEAKER_00", "speaker_assigned": True,
         "overlapping_speakers": ["SPEAKER_00"], "is_ambiguous": False, "text": "تمام"},
    ]
    qc_warn = pipeline.run_qc(
        "rec1", 6.0, 6.0, ["SPEAKER_00", "SPEAKER_01"], conv_warn,
        processed_sample_rate=16000, processed_channels=1,
        expected_output_files={"standardized_wav": f_wav},
    )
    assert qc_warn["checks"]["multi_speaker_overlap"] == "WARNING"
    assert qc_warn["multi_speaker_overlap_segments"] == 1
    assert qc_warn["ambiguous_speaker_segments"] == 0
    assert qc_warn["status"] == "WARNING"

    # >= 50% ambiguous multi-speaker turns -> FAIL
    conv_fail = [
        {"turn_id": 1, "start": 0.0, "end": 2.0, "speaker": "SPEAKER_00", "speaker_assigned": True,
         "overlapping_speakers": ["SPEAKER_00", "SPEAKER_01"], "is_ambiguous": True, "text": "الو"},
        {"turn_id": 2, "start": 2.0, "end": 4.0, "speaker": "SPEAKER_01", "speaker_assigned": True,
         "overlapping_speakers": ["SPEAKER_00", "SPEAKER_01"], "is_ambiguous": True, "text": "ايوه"},
    ]
    qc_fail = pipeline.run_qc(
        "rec1", 4.0, 4.0, ["SPEAKER_00", "SPEAKER_01"], conv_fail,
        processed_sample_rate=16000, processed_channels=1,
        expected_output_files={"standardized_wav": f_wav},
    )
    assert qc_fail["checks"]["multi_speaker_overlap"] == "FAIL"
    assert qc_fail["ambiguous_speaker_segments"] == 2
    assert qc_fail["status"] == "FAIL"


# ------------------------------------------------------------------ QC propagation & output validation
def test_process_call_qc_fail_propagation_and_profile_output_validation(monkeypatch, tmp_path):
    raw_dir = tmp_path / "raw_calls"
    raw_dir.mkdir()
    (raw_dir / "rec1.mp4").write_bytes(b"raw")
    out_dirs = _make_out_dirs(tmp_path, raw_dir)

    fake_models = pipeline.PipelineModels(
        device="cpu", vad_bundle=("v", [None]), whisper_model="w", diarization_pipeline="d"
    )

    monkeypatch.setattr(
        pipeline, "inspect_audio",
        lambda *_a, **_kw: (4.0, {"format": ".mp4", "sample_rate": 44100, "channels": 2, "duration": 4.0}),
    )

    def _fake_standardize(_ff, _src, dst, return_details=False):
        _write_test_wav(Path(dst), sr=16000, channels=1, duration=4.0)
        info = {"sample_rate": 16000, "channels": 1, "duration": 4.0}
        return (4.0, info) if return_details else 4.0

    monkeypatch.setattr(pipeline, "standardize_audio", _fake_standardize)
    monkeypatch.setattr(
        pipeline, "run_vad",
        lambda *_a, **_kw: {"call_id": "rec1", "total_speech_duration": 3.0, "speech_ratio": 0.75, "speech_segments": []},
    )
    monkeypatch.setattr(
        pipeline, "run_diarization",
        lambda *_a, **_kw: (
            [{"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"}, {"start": 2.0, "end": 4.0, "speaker": "SPEAKER_01"}],
            ["SPEAKER_00", "SPEAKER_01"],
        ),
    )

    # Case A: Empty transcript causes QC FAIL -> process_call must report QC_FAILED & exit code 2
    monkeypatch.setattr(
        pipeline, "transcribe",
        lambda *_a, **_kw: (types.SimpleNamespace(language="ar", language_probability=0.99), [], ""),
    )
    res_fail = pipeline.process_call(
        "rec1", ext="mp4", skip_profile=True, ffmpeg_path="ffmpeg",
        models=fake_models, raw_audio_dir=raw_dir, output_dirs=out_dirs,
    )
    assert res_fail["qc_status"] == "FAIL"
    assert res_fail["status"] == "QC_FAILED"
    assert pipeline.get_call_exit_code(res_fail) == pipeline.EXIT_QC_FAILED
    assert "qc_json" in res_fail["qc"]["checked_output_files"]
    assert "customer_profile_json" not in res_fail["qc"]["checked_output_files"]

    # Case B: Valid transcript with skip_profile=False validates customer_profile_json too
    monkeypatch.setenv("GEMINI_API_KEY", "dummy-key")
    monkeypatch.setattr(
        pipeline, "transcribe",
        lambda *_a, **_kw: (
            types.SimpleNamespace(language="ar", language_probability=0.99),
            [
                {"segment_id": 1, "start": 0.0, "end": 2.0, "text": "أهلاً", "normalized_text": "اهلا"},
                {"segment_id": 2, "start": 2.0, "end": 4.0, "text": "تمام", "normalized_text": "تمام"},
            ],
            "اهلا تمام",
        ),
    )
    monkeypatch.setattr(
        pipeline, "extract_customer_profile",
        lambda _conv: pipeline.CustomerProfile(location="التجمع الخامس", rooms=3),
    )
    res_pass = pipeline.process_call(
        "rec1", ext="mp4", skip_profile=False, ffmpeg_path="ffmpeg",
        models=fake_models, raw_audio_dir=raw_dir, output_dirs=out_dirs,
    )
    assert res_pass["qc_status"] == "PASS"
    assert res_pass["status"] == "SUCCESS"
    assert pipeline.get_call_exit_code(res_pass) == pipeline.EXIT_OK
    assert "customer_profile_json" in res_pass["qc"]["checked_output_files"]
    assert "qc_json" in res_pass["qc"]["checked_output_files"]
    assert Path(res_pass["outputs"]["customer_profile_json"]).exists()


def test_process_batch_distinguishes_qc_fail_from_processing_errors_and_exit_codes(monkeypatch, tmp_path):
    raw_dir = tmp_path / "raw_calls"
    raw_dir.mkdir()
    for cid in ("smartsemsar_rec1", "smartsemsar_rec2", "smartsemsar_rec3", "smartsemsar_rec4"):
        (raw_dir / f"{cid}.mp4").write_bytes(b"original-raw-bytes")
    out_dirs = _make_out_dirs(tmp_path, raw_dir)

    load_count = {"models": 0}
    seen_model_ids = []
    fake_models = pipeline.PipelineModels(
        device="cpu", vad_bundle=("vad", [None]), whisper_model="whisper", diarization_pipeline="diar"
    )

    monkeypatch.setattr(pipeline, "load_pipeline_models", lambda _d: (load_count.__setitem__("models", load_count["models"] + 1) or fake_models))
    monkeypatch.setattr(pipeline, "get_device", lambda: "cpu")
    monkeypatch.setattr(pipeline, "setup_ffmpeg", lambda: "ffmpeg")
    monkeypatch.setenv("HF_TOKEN", "dummy")

    def _fake_process_call(cid, **kwargs):
        seen_model_ids.append(id(kwargs["models"]))
        if cid == "smartsemsar_rec1":
            return {"call_id": cid, "status": "SUCCESS", "qc_status": "PASS", "qc": {"checks": {}}, "outputs": {}}
        if cid == "smartsemsar_rec2":
            return {"call_id": cid, "status": "SUCCESS_WITH_WARNINGS", "qc_status": "WARNING", "qc": {"checks": {}}, "outputs": {}}
        if cid == "smartsemsar_rec3":
            return {"call_id": cid, "status": "QC_FAILED", "qc_status": "FAIL", "qc": {"checks": {"transcript": "FAIL"}}, "outputs": {}}
        raise RuntimeError("Simulated corrupt MP4 container")

    monkeypatch.setattr(pipeline, "process_call", _fake_process_call)

    summary = pipeline.process_batch(
        ext="mp4", skip_profile=True, raw_audio_dir=raw_dir, output_dirs=out_dirs
    )
    assert load_count["models"] == 1
    assert len(set(seen_model_ids)) == 1
    assert summary["qc_pass"] == 1
    assert summary["qc_warning"] == 1
    assert summary["qc_fail"] == 1
    assert summary["processing_errors"] == 1
    assert summary["qc_summary"] == {"PASS": 1, "WARNING": 1, "FAIL": 1, "ERROR": 1}
    assert summary["succeeded"] == 2
    assert summary["failed"] == 2
    assert pipeline.get_batch_exit_code(summary) == pipeline.EXIT_PROCESSING_ERROR

    # Batch with ONLY QC FAIL (0 processing errors) must return EXIT_QC_FAILED (2, nonzero)
    summary_qc_only = pipeline.process_batch(
        call_ids=["smartsemsar_rec1", "smartsemsar_rec3"],
        ext="mp4", skip_profile=True, raw_audio_dir=raw_dir, output_dirs=out_dirs,
    )
    assert summary_qc_only["processing_errors"] == 0
    assert summary_qc_only["qc_fail"] == 1
    assert pipeline.get_batch_exit_code(summary_qc_only) == pipeline.EXIT_QC_FAILED

    # Batch with PASS + WARNING only returns EXIT_OK (0)
    summary_ok = pipeline.process_batch(
        call_ids=["smartsemsar_rec1", "smartsemsar_rec2"],
        ext="mp4", skip_profile=True, raw_audio_dir=raw_dir, output_dirs=out_dirs,
    )
    assert pipeline.get_batch_exit_code(summary_ok) == pipeline.EXIT_OK

    # Raw recordings never touched
    for cid in ("smartsemsar_rec1", "smartsemsar_rec2", "smartsemsar_rec3", "smartsemsar_rec4"):
        assert (raw_dir / f"{cid}.mp4").read_bytes() == b"original-raw-bytes"


def test_single_call_main_returns_nonzero_on_qc_fail_and_preserves_explicit_filename(monkeypatch):
    captured = {}

    def _fake_process(call_id, **kwargs):
        captured["call_id"] = call_id
        captured["ext"] = kwargs.get("ext")
        return {"call_id": "smartsemsar_rec1", "status": "QC_FAILED", "qc_status": "FAIL"}

    monkeypatch.setattr(pipeline, "process_call", _fake_process)
    exit_code = pipeline.main(["--call-id", "smartsemsar_rec1.mp4", "--skip-profile"])
    assert exit_code == pipeline.EXIT_QC_FAILED
    assert captured["call_id"] == "smartsemsar_rec1.mp4"
    assert captured["ext"] is None
    # Verify resolution from those CLI args preserves .mp4 instead of defaulting to .ogg
    assert pipeline.normalize_call_id_and_ext(captured["call_id"], ext=captured["ext"]) == (
        "smartsemsar_rec1",
        "mp4",
    )


# ------------------------------------------------------------------ evaluation metrics: micro vs macro & validation
def test_wer_cer_micro_vs_macro_aggregation():
    # Call 1: 2 words, 1 error -> WER = 0.50
    w1 = compute_wer("شقة جديدة", "شقة قديمة")
    # Call 2: 8 words, 0 errors -> WER = 0.00
    w2 = compute_wer("عايز شقة ثلاث غرف في التجمع الخامس القاهرة", "عايز شقة ثلاث غرف في التجمع الخامس القاهرة")
    agg_w = compute_corpus_wer([w1, w2])
    # Macro = (0.5 + 0.0) / 2 = 0.25; Micro = 1 / 10 = 0.10
    assert agg_w["macro_wer"] == pytest.approx(0.25)
    assert agg_w["micro_wer"] == pytest.approx(0.10)
    assert agg_w["total_reference_words"] == 10
    assert agg_w["total_edit_distance"] == 1

    c1 = compute_cer("أريد شقة", "اريد فيلا")
    c2 = compute_cer("التجمع الخامس القاهرة الجديدة", "التجمع الخامس القاهرة الجديدة")
    agg_c = compute_corpus_cer([c1, c2])
    assert agg_c["macro_cer"] != agg_c["micro_cer"]
    assert agg_c["micro_cer"] < agg_c["macro_cer"]


def test_der_validation_and_weighted_micro_vs_macro():
    # Validation checks: negative collar, inverted timestamps, empty speaker
    with pytest.raises(ValueError, match="collar"):
        compute_der([{"start": 0.0, "end": 1.0, "speaker": "SPK_1"}], [], collar=-0.1)
    with pytest.raises(ValueError, match="strictly greater"):
        compute_der([{"start": 2.0, "end": 1.0, "speaker": "SPK_1"}], [], collar=0.0)
    with pytest.raises(ValueError, match="speaker label"):
        compute_der([{"start": 0.0, "end": 1.0, "speaker": "   "}], [], collar=0.0)

    # Call 1: 2s ref speech, 1s missed -> DER = 0.50
    d1 = compute_der(
        [{"start": 0.0, "end": 2.0, "speaker_id": "SPK_1"}],
        [{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"}],
        collar=0.0,
    )
    # Call 2: 8s ref speech, 0s error -> DER = 0.00 (swapped neutral labels still 0.0)
    d2 = compute_der(
        [{"start": 0.0, "end": 4.0, "speaker_id": "SPK_1"}, {"start": 4.0, "end": 8.0, "speaker_id": "SPK_2"}],
        [{"start": 0.0, "end": 4.0, "speaker": "SPEAKER_01"}, {"start": 4.0, "end": 8.0, "speaker": "SPEAKER_00"}],
        collar=0.0,
    )
    agg_d = compute_corpus_der([d1, d2])
    # Macro = (0.5 + 0.0) / 2 = 0.25; Micro = 1.0 / 10.0 = 0.10
    assert agg_d["macro_der"] == pytest.approx(0.25)
    assert agg_d["micro_der"] == pytest.approx(0.10)
    assert agg_d["total_reference_speech_sec"] == pytest.approx(10.0)


def test_role_accuracy_placeholder_rejection_and_audio_rough_snr_label(tmp_path):
    with pytest.raises(ValueError, match="placeholders are rejected"):
        compute_role_accuracy({"SPK_1": "broker_or_client"}, {"SPK_1": "broker"})
    with pytest.raises(ValueError, match="placeholders are rejected"):
        compute_role_accuracy({"SPK_1": "client_or_broker"}, {"SPK_1": "client"})

    pending_res = compute_role_accuracy(
        {"SPEAKER_00": "broker", "SPEAKER_01": "client"},
        {"SPEAKER_00": "unassigned", "SPEAKER_01": "unassigned"},
    )
    assert pending_res["status"] == "PENDING_ROLE_PREDICTIONS"
    assert pending_res["role_accuracy"] is None

    wav = tmp_path / "quality.wav"
    _write_test_wav(wav, sr=16000, channels=1, duration=2.0)
    aq = analyze_audio_quality(wav, vad_segments=[{"start": 0.0, "end": 1.0}])
    assert aq["snr_is_rough_estimate"] is True
    assert "rough" in aq["snr_disclaimer"].lower()
    assert aq["rough_snr_estimate_db"] is not None and aq["rough_snr_estimate_db"] > 15.0


# ------------------------------------------------------------------ end-to-end evaluation robustness
def test_evaluate_calls_completed_gt_consistent_speaker_mapping_and_pending_roles(tmp_path):
    gt_dir = tmp_path / "evaluation" / "ground_truth"
    for sub in ("transcripts", "diarization", "roles"):
        (gt_dir / sub).mkdir(parents=True)
    for d in ("processed_audio", "vad", "transcripts", "diarization", "conversations", "qc"):
        (tmp_path / d).mkdir(parents=True)

    # Manifest listing 2 calls
    (gt_dir / "manifest.json").write_text(json.dumps({
        "calls": [
            {
                "call_id": "rec1",
                "ground_truth_transcript": "gt/transcripts/rec1_gt_transcript.json",
                "ground_truth_diarization": "gt/diarization/rec1_gt_diarization.json",
                "ground_truth_roles": "gt/roles/rec1_gt_roles.json",
            },
            {
                "call_id": "rec2",
                "ground_truth_transcript": "gt/transcripts/rec2_gt_transcript.json",
                "ground_truth_diarization": "gt/diarization/rec2_gt_diarization.json",
                "ground_truth_roles": "gt/roles/rec2_gt_roles.json",
            },
        ]
    }), encoding="utf-8")

    # rec1: full GT + pipeline outputs WITH predicted roles (aligned via DER overlap: SPEAKER_01->SPK_1, SPEAKER_00->SPK_2)
    (gt_dir / "transcripts" / "rec1_gt_transcript.json").write_text(json.dumps({
        "call_id": "rec1", "status": "ready", "reference_transcript": "ألو عايز شقة في التجمع",
        "segments": [{"start": 0.0, "end": 2.0, "speaker_id": "SPK_1", "role": "client", "text": "ألو عايز شقة في التجمع"}],
    }), encoding="utf-8")
    (gt_dir / "diarization" / "rec1_gt_diarization.json").write_text(json.dumps({
        "call_id": "rec1", "status": "ready",
        "turns": [
            {"start": 0.0, "end": 2.0, "speaker_id": "SPK_1", "role": "client"},
            {"start": 2.0, "end": 4.0, "speaker_id": "SPK_2", "role": "broker"},
        ],
    }), encoding="utf-8")
    (gt_dir / "roles" / "rec1_gt_roles.json").write_text(json.dumps({
        "call_id": "rec1", "status": "ready",
        "speaker_roles": {"SPK_1": "client", "SPK_2": "broker"},
    }), encoding="utf-8")

    (tmp_path / "transcripts" / "rec1_transcript.json").write_text(
        json.dumps({"call_id": "rec1", "full_transcript": "الو عايز شقة في التجمع"}), encoding="utf-8"
    )
    (tmp_path / "diarization" / "rec1_diarization.json").write_text(json.dumps({
        "call_id": "rec1",
        "turns": [
            {"start": 0.0, "end": 2.0, "speaker": "SPEAKER_01"},
            {"start": 2.0, "end": 4.0, "speaker": "SPEAKER_00"},
        ],
    }), encoding="utf-8")
    (tmp_path / "conversations" / "rec1_conversation.json").write_text(json.dumps({
        "call_id": "rec1",
        "speaker_role_mapping": {"SPEAKER_01": "client", "SPEAKER_00": "broker"},
    }), encoding="utf-8")

    # rec2: completed GT roles, but pipeline conversation still has neutral/unassigned roles -> PENDING_ROLE_PREDICTIONS
    (gt_dir / "roles" / "rec2_gt_roles.json").write_text(json.dumps({
        "call_id": "rec2", "status": "ready",
        "speaker_roles": {"SPK_1": "broker", "SPK_2": "client"},
    }), encoding="utf-8")
    (tmp_path / "conversations" / "rec2_conversation.json").write_text(json.dumps({
        "call_id": "rec2",
        "speaker_role_mapping": {"SPEAKER_00": "unassigned", "SPEAKER_01": "unassigned"},
    }), encoding="utf-8")

    summary = evaluate_calls(
        processed_audio_dir=tmp_path / "processed_audio",
        vad_dir=tmp_path / "vad",
        transcript_dir=tmp_path / "transcripts",
        diarization_dir=tmp_path / "diarization",
        conversation_dir=tmp_path / "conversations",
        qc_dir=tmp_path / "qc",
        gt_dir=gt_dir,
        reports_dir=tmp_path / "evaluation" / "reports",
        collar=0.0,
    )
    assert summary["manifest_loaded"] is True
    assert summary["call_ids"] == ["rec1", "rec2"]
    assert summary["total_errors"] == 0
    assert summary["status"] == "PARTIALLY_EVALUATED"
    assert summary["wer_cer"]["status"] == "PARTIALLY_EVALUATED"
    assert summary["wer_cer"]["micro_wer"] == 0.0
    assert summary["der"]["micro_der"] == 0.0
    assert summary["role_accuracy"]["status"] == "PARTIALLY_EVALUATED"
    assert summary["role_accuracy"]["evaluated_calls"] == 1
    assert summary["role_accuracy"]["pending_role_prediction_calls"] == 1
    assert summary["role_accuracy"]["micro_role_accuracy"] == 1.0
    assert get_evaluation_exit_code(summary) == EVAL_EXIT_OK


def test_evaluate_calls_error_isolation_malformed_json_and_speaker_mismatch(tmp_path):
    gt_dir = tmp_path / "evaluation" / "ground_truth"
    for sub in ("transcripts", "diarization", "roles"):
        (gt_dir / sub).mkdir(parents=True)
    for d in ("processed_audio", "vad", "transcripts", "diarization", "conversations", "qc"):
        (tmp_path / d).mkdir(parents=True)

    # rec1: valid GT transcript & pipeline output
    (gt_dir / "transcripts" / "rec1_gt_transcript.json").write_text(
        json.dumps({"call_id": "rec1", "status": "ready", "reference_transcript": "اهلا بك"}), encoding="utf-8"
    )
    (tmp_path / "transcripts" / "rec1_transcript.json").write_text(
        json.dumps({"call_id": "rec1", "full_transcript": "اهلا بك"}), encoding="utf-8"
    )

    # rec2: malformed JSON in gt_transcript + speaker ID mismatch between gt_diarization and gt_roles
    (gt_dir / "transcripts" / "rec2_gt_transcript.json").write_text("{broken json", encoding="utf-8")
    (gt_dir / "diarization" / "rec2_gt_diarization.json").write_text(json.dumps({
        "call_id": "rec2", "status": "ready",
        "turns": [{"start": 0.0, "end": 2.0, "speaker_id": "SPK_UNKNOWN"}],
    }), encoding="utf-8")
    (tmp_path / "diarization" / "rec2_diarization.json").write_text(json.dumps({
        "call_id": "rec2", "turns": [{"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"}],
    }), encoding="utf-8")
    (gt_dir / "roles" / "rec2_gt_roles.json").write_text(json.dumps({
        "call_id": "rec2", "status": "ready",
        "speaker_roles": {"SPK_1": "broker", "SPK_2": "client"},
    }), encoding="utf-8")
    (tmp_path / "conversations" / "rec2_conversation.json").write_text(
        json.dumps({"call_id": "rec2", "speaker_role_mapping": {"SPEAKER_00": "broker"}}), encoding="utf-8"
    )

    summary = evaluate_calls(
        call_ids=["rec1", "rec2"],
        processed_audio_dir=tmp_path / "processed_audio",
        vad_dir=tmp_path / "vad",
        transcript_dir=tmp_path / "transcripts",
        diarization_dir=tmp_path / "diarization",
        conversation_dir=tmp_path / "conversations",
        qc_dir=tmp_path / "qc",
        gt_dir=gt_dir,
        reports_dir=tmp_path / "evaluation" / "reports",
    )
    # rec1 succeeded despite rec2 malformed JSON & speaker mismatch
    assert summary["wer_cer"]["evaluated_calls"] == 1
    assert summary["wer_cer"]["error_calls"] == 1
    assert summary["wer_cer"]["status"] == "PARTIALLY_EVALUATED"
    assert summary["role_accuracy"]["error_calls"] == 1
    assert summary["role_accuracy"]["status"] == "ERROR"
    assert summary["total_errors"] == 2
    assert get_evaluation_exit_code(summary) == EVAL_EXIT_ERROR


def test_evaluate_calls_does_not_fabricate_without_ground_truth(tmp_path):
    gt_dir = tmp_path / "evaluation" / "ground_truth"
    for sub in ("transcripts", "diarization", "roles"):
        (gt_dir / sub).mkdir(parents=True)
    # Template and pending files must be ignored (PENDING_GROUND_TRUTH, null metrics)
    (gt_dir / "transcripts" / "smartsemsar_rec1_gt_transcript.json").write_text(
        json.dumps({"_is_template": True, "reference_transcript": "ignored"}), encoding="utf-8"
    )
    (gt_dir / "roles" / "smartsemsar_rec1_gt_roles.json").write_text(
        json.dumps({"status": "pending", "speaker_roles": {"SPK_1": None}}), encoding="utf-8"
    )

    summary = evaluate_calls(
        call_ids=["smartsemsar_rec1"],
        processed_audio_dir=tmp_path / "processed_audio",
        vad_dir=tmp_path / "vad",
        transcript_dir=tmp_path / "transcripts",
        diarization_dir=tmp_path / "diarization",
        conversation_dir=tmp_path / "conversations",
        qc_dir=tmp_path / "qc",
        gt_dir=gt_dir,
        reports_dir=tmp_path / "evaluation" / "reports",
    )
    assert summary["status"] == "PENDING_GROUND_TRUTH"
    assert summary["wer_cer"]["status"] == "PENDING_GROUND_TRUTH"
    assert summary["wer_cer"]["macro_wer"] is None
    assert summary["wer_cer"]["micro_wer"] is None
    assert summary["der"]["status"] == "PENDING_GROUND_TRUTH"
    assert summary["der"]["macro_der"] is None
    assert summary["der"]["micro_der"] is None
    assert summary["role_accuracy"]["status"] == "PENDING_GROUND_TRUTH"
    assert summary["role_accuracy"]["macro_role_accuracy"] is None
    assert get_evaluation_exit_code(summary, strict=False) == EVAL_EXIT_OK
    assert get_evaluation_exit_code(summary, strict=True) == EVAL_EXIT_NO_EVALUATED


# ------------------------------------------------------------------ speaker role assignment, diarization cleanup, UTF-8
def test_infer_speaker_roles_heuristic_is_speaker_id_agnostic():
    conv_a = [
        {"start": 0.0, "end": 3.0, "speaker": "SPEAKER_00", "text": "مساء الخير يا فندم مع حضرتك مكتب عقارات كنت بتدوري علي ايه"},
        {"start": 3.0, "end": 6.0, "speaker": "SPEAKER_01", "text": "انا بدور علي شقة في التجمع الخامس"},
        {"start": 6.0, "end": 9.0, "speaker": "SPEAKER_00", "text": "كام غرفة تقريبا ايجار ولا تمليك والميزانية في حدود كام"},
        {"start": 9.0, "end": 11.0, "speaker": "SPEAKER_01", "text": "محتاجة ثلاث غرف ايجار عشرين الف"},
        {"start": 11.0, "end": 14.0, "speaker": "SPEAKER_00", "text": "تمام يا فندم شكرا لاتصال حضرتك"},
    ]
    roles_a, details_a = pipeline.infer_speaker_roles(conv_a, use_llm=False)
    assert roles_a == {"SPEAKER_00": "broker", "SPEAKER_01": "client"}
    assert details_a["method"] == "heuristic"

    # Swap neutral speaker IDs -> predicted roles MUST swap cleanly (proves zero hardcoding)
    conv_swapped = [
        {**t, "speaker": "SPEAKER_01" if t["speaker"] == "SPEAKER_00" else "SPEAKER_00"}
        for t in conv_a
    ]
    roles_b, _ = pipeline.infer_speaker_roles(conv_swapped, use_llm=False)
    assert roles_b == {"SPEAKER_00": "client", "SPEAKER_01": "broker"}

    # Uninformative/empty conversation must NOT guess
    roles_empty, _ = pipeline.infer_speaker_roles(
        [{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00", "text": "الو"},
         {"start": 1.0, "end": 2.0, "speaker": "SPEAKER_01", "text": "الو"}],
        use_llm=False,
    )
    assert roles_empty == {"SPEAKER_00": "unassigned", "SPEAKER_01": "unassigned"}


def test_infer_speaker_roles_llm_and_fallback(monkeypatch):
    conv = [
        {"start": 0.0, "end": 2.0, "speaker": "SPEAKER_01", "text": "مع حضرتك مكتب عقارات ايجار ولا تمليك"},
        {"start": 2.0, "end": 4.0, "speaker": "SPEAKER_00", "text": "انا عايزة شقة تمليك في زايد"},
    ]

    # Case 1: LLM succeeds
    monkeypatch.setattr(
        pipeline,
        "_call_role_assignment_llm",
        lambda _prompt: (
            {
                "speaker_role_mapping": {"SPEAKER_01": "broker", "SPEAKER_00": "client"},
                "confidence": "high",
                "reasoning": "SPEAKER_01 greets as real-estate office and asks rent vs buy.",
            },
            "groq:mock-model",
            None,
        ),
    )
    roles, details = pipeline.infer_speaker_roles(conv, use_llm=True)
    assert roles == {"SPEAKER_00": "client", "SPEAKER_01": "broker"}
    assert details["method"] == "groq:mock-model"

    # Case 2: LLM fails -> deterministic heuristic fallback succeeds cleanly
    monkeypatch.setattr(
        pipeline,
        "_call_role_assignment_llm",
        lambda _prompt: (None, None, "Simulated API timeout"),
    )
    roles_fb, details_fb = pipeline.infer_speaker_roles(conv, use_llm=True)
    assert roles_fb == {"SPEAKER_00": "client", "SPEAKER_01": "broker"}
    assert details_fb["method"] == "heuristic"
    assert "Simulated API timeout" in details_fb["llm_fallback_reason"]


def test_clean_diarization_turns_removes_micro_glitches_and_merges():
    raw_turns = [
        {"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"},
        {"start": 2.0, "end": 2.017, "speaker": "SPEAKER_01"},  # 17ms glitch (SPEAKER_01 has longer turn below)
        {"start": 2.017, "end": 4.0, "speaker": "SPEAKER_00"},
        {"start": 4.5, "end": 6.5, "speaker": "SPEAKER_01"},
    ]
    cleaned = pipeline.clean_diarization_turns(raw_turns, min_turn_duration_sec=0.10, merge_gap_sec=0.05, audio_duration=6.2)
    assert len(cleaned) == 2
    assert cleaned[0]["speaker"] == "SPEAKER_00"
    assert cleaned[0]["start"] == 0.0 and cleaned[0]["end"] == 4.0
    assert cleaned[1]["speaker"] == "SPEAKER_01"
    assert cleaned[1]["start"] == 4.5 and cleaned[1]["end"] == 6.2  # Clamped to audio_duration


def test_transcribe_filters_eof_hallucination_overflow():
    class _FakeSeg:
        def __init__(self, start, end, text):
            self.start = start
            self.end = end
            self.text = text

    class _FakeWhisper:
        def transcribe(self, *_a, **_kw):
            segs = [
                _FakeSeg(0.0, 29.2, "شقة في مدينة نصر"),
                _FakeSeg(29.2, 59.18, "اشتركوا في القناة"),  # Whisper 30s window EOF overflow hallucination
            ]
            info = types.SimpleNamespace(language="ar", language_probability=1.0, duration=33.74)
            return segs, info

    _info, result, full_text = pipeline.transcribe("dummy.wav", "cpu", whisper_model=_FakeWhisper())
    assert len(result) == 1
    assert result[0]["text"] == "شقة في مدينة نصر"
    assert "اشتركوا في القناة" not in full_text


def _capture_transcribe_kwargs():
    seen = {}

    class _FakeWhisper:
        def transcribe(self, *_a, **kw):
            seen.update(kw)
            return [], types.SimpleNamespace(language="ar", language_probability=1.0, duration=10.0)

    pipeline.transcribe("dummy.wav", "cpu", whisper_model=_FakeWhisper())
    return seen


def test_word_list_is_off_by_default():
    # it broke role assignment in the full pipeline (90% -> 70%), so it stays off
    assert _capture_transcribe_kwargs().get("initial_prompt") is None


def test_word_list_can_be_turned_on(monkeypatch):
    monkeypatch.setattr(pipeline, "WHISPER_INITIAL_PROMPT", pipeline.REAL_ESTATE_WORD_LIST)
    prompt = _capture_transcribe_kwargs().get("initial_prompt") or ""
    for word in ("فيلا", "تمليك", "إيجار", "التجمع الخامس"):
        assert word in prompt


def test_pipeline_word_list_matches_the_tested_one():
    from backend.services.evaluation.vocab_prompt_experiment import EXTENDED_PROMPT
    assert pipeline.REAL_ESTATE_WORD_LIST == EXTENDED_PROMPT


def _w(start, end, word):
    return {"start": start, "end": end, "word": word}


_TWO_SPEAKER_TURNS = [
    {"start": 0.0, "end": 3.0, "speaker": "SPEAKER_00"},
    {"start": 3.2, "end": 6.0, "speaker": "SPEAKER_01"},
]


def test_segment_with_two_speakers_is_split_at_the_change():
    seg = {"segment_id": 1, "start": 0.0, "end": 6.0, "text": "كام غرفة؟ تلات غرف",
           "normalized_text": "كام غرفة؟ تلات غرف",
           "words": [_w(0.1, 1.0, " كام"), _w(1.0, 2.5, " غرفة؟"), _w(3.3, 4.5, " تلات"), _w(4.5, 5.8, " غرف")]}
    out = pipeline.split_segments_by_speaker([seg], _TWO_SPEAKER_TURNS)
    assert [s["text"] for s in out] == ["كام غرفة؟", "تلات غرف"]
    assert (out[0]["start"], out[0]["end"]) == (0.1, 2.5)
    assert (out[1]["start"], out[1]["end"]) == (3.3, 5.8)
    conv = pipeline.build_conversation(out, _TWO_SPEAKER_TURNS)
    assert [t["speaker"] for t in conv] == ["SPEAKER_00", "SPEAKER_01"]


def test_one_speaker_segment_and_no_word_timestamps_stay_as_they_are():
    one = {"segment_id": 1, "start": 0.0, "end": 2.0, "text": "اهلا يا فندم", "normalized_text": "اهلا يا فندم",
           "words": [_w(0.1, 0.8, " اهلا"), _w(0.8, 1.9, " يا فندم")]}
    plain = {"segment_id": 2, "start": 3.3, "end": 5.0, "text": "شقة", "normalized_text": "شقة"}
    out = pipeline.split_segments_by_speaker([one, plain], _TWO_SPEAKER_TURNS)
    assert [s["text"] for s in out] == ["اهلا يا فندم", "شقة"]
    assert "split_from_segment" not in out[0]


def test_short_word_glitch_does_not_split():
    turns = [{"start": 0.0, "end": 2.0, "speaker": "A"}, {"start": 2.0, "end": 2.2, "speaker": "B"},
             {"start": 2.2, "end": 5.0, "speaker": "A"}]
    seg = {"segment_id": 1, "start": 0.0, "end": 5.0, "text": "x", "normalized_text": "x",
           "words": [_w(0.0, 1.9, " طيب"), _w(2.0, 2.2, " يعني"), _w(2.3, 4.0, " تمام")]}
    out = pipeline.split_segments_by_speaker([seg], turns)
    assert len(out) == 1


def test_word_level_speakers_ask_whisper_for_word_timestamps():
    assert _capture_transcribe_kwargs().get("word_timestamps") is pipeline.WORD_LEVEL_SPEAKERS


def test_utf8_arabic_json_roundtrip(tmp_path):
    p = tmp_path / "arabic_conv.json"
    payload = {"text": "في التجمع الخامس", "speaker_role_mapping": {"SPEAKER_00": "broker"}}
    pipeline.save_json(p, payload)
    raw_bytes = p.read_bytes()
    assert "في التجمع الخامس".encode("utf-8") in raw_bytes
    loaded = pipeline.load_json(p)
    assert loaded == payload


def test_extraction_agreement_normalization_and_metrics():
    from backend.services.evaluation.extraction_agreement import compare_extractions

    human = {
        "location": "التجمع الخامس",
        "budget_min": None,
        "budget_max": 12000000.0,
        "area_sqm": 180.0,
        "bedrooms": 3,
        "property_type": "apartment",
        "purpose": "unknown",
        "deal_type": "buy",
        "notes": "مش صغيرة",
    }
    machine = {
        "location": "Fifth Settlement",       # semantic_match
        "budget_min": None,                    # both_absent (must NOT inflate denominator!)
        "budget_max": "12 million",            # semantic_match
        "area_sqm": "180 sqm",                 # semantic_match
        "bedrooms": 3,                         # exact_match
        "property_type": "unknown",            # missing_requirement
        "purpose": "living",                   # extra_requirement
        "deal_type": "rent",                   # conflicting_value
        "notes": "not small",                  # semantic_match
    }
    comp = compare_extractions(machine, human)
    m = comp["metrics"]
    # H = 7 present human requirements (budget_min & purpose absent in human)
    assert m["total_human_requirements"] == 7
    assert m["both_absent_fields_excluded"] == 1
    assert m["exact_matches"] == 1
    assert m["semantic_matches"] == 4
    assert m["matched_requirements"] == 5
    assert m["missing_requirements"] == 1
    assert m["extra_requirements"] == 1
    assert m["conflicting_values"] == 1
    assert m["requirement_match_rate_pct"] == pytest.approx(71.43, abs=0.01)
    assert m["exact_match_rate_pct"] == pytest.approx(14.29, abs=0.01)
    assert m["semantic_match_rate_pct"] == pytest.approx(57.14, abs=0.01)
    assert m["missing_requirement_rate_pct"] == pytest.approx(14.29, abs=0.01)
    assert m["extra_requirement_rate_vs_human_pct"] == pytest.approx(14.29, abs=0.01)
    assert m["conflicting_value_rate_pct"] == pytest.approx(14.29, abs=0.01)
