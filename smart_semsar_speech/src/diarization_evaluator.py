"""
Diarization & Role Separation Evaluator
Evaluates:
1. Diarization Error Rate (DER) - False Alarm, Missed Speech, Speaker Confusion.
2. Broker vs. Client Separation Accuracy - Precision, Recall, F1, and Confusion Matrix.
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np


def intervals_to_grid(
    turns: List[Dict[str, Any]],
    total_duration: float,
    frame_step: float = 0.05,
    role_key: str = "role"
) -> np.ndarray:
    """
    Converts segment intervals into a uniform frame-by-frame label grid.
    
    Returns 1D array of string labels for each frame.
    """
    num_frames = int(np.ceil(total_duration / frame_step))
    grid = np.array(["silence"] * num_frames, dtype=object)
    
    for turn in turns:
        start_frame = max(0, int(round(turn["start"] / frame_step)))
        end_frame = min(num_frames, int(round(turn["end"] / frame_step)))
        label = turn.get(role_key) or turn.get("speaker") or "speech"
        grid[start_frame:end_frame] = label
        
    return grid


def evaluate_der(
    reference_turns: List[Dict[str, Any]],
    hypothesis_turns: List[Dict[str, Any]],
    total_duration: float,
    collar: float = 0.25,
    frame_step: float = 0.05
) -> Dict[str, Any]:
    """
    Computes Diarization Error Rate (DER):
    DER = (Missed Speech + False Alarm + Speaker Confusion) / Total Reference Speech Time.
    
    Includes collar tolerance around speaker transitions.
    """
    ref_grid = intervals_to_grid(reference_turns, total_duration, frame_step=frame_step, role_key="speaker")
    hyp_grid = intervals_to_grid(hypothesis_turns, total_duration, frame_step=frame_step, role_key="speaker")
    
    # Identify unique speakers
    ref_speakers = sorted(list(set(ref_grid) - {"silence"}))
    hyp_speakers = sorted(list(set(hyp_grid) - {"silence"}))
    
    # Map hypothesis speaker labels to reference speaker labels (optimal Hungarian assignment)
    best_mapping = {}
    if len(hyp_speakers) == 2 and len(ref_speakers) == 2:
        # Check both permutations
        match_1 = np.sum((hyp_grid == hyp_speakers[0]) & (ref_grid == ref_speakers[0])) + \
                  np.sum((hyp_grid == hyp_speakers[1]) & (ref_grid == ref_speakers[1]))
        match_2 = np.sum((hyp_grid == hyp_speakers[0]) & (ref_grid == ref_speakers[1])) + \
                  np.sum((hyp_grid == hyp_speakers[1]) & (ref_grid == ref_speakers[0]))
        if match_1 >= match_2:
            best_mapping = {hyp_speakers[0]: ref_speakers[0], hyp_speakers[1]: ref_speakers[1]}
        else:
            best_mapping = {hyp_speakers[0]: ref_speakers[1], hyp_speakers[1]: ref_speakers[0]}
    else:
        for hs in hyp_speakers:
            best_mapping[hs] = hs
            
    mapped_hyp_grid = np.array([best_mapping.get(label, label) for label in hyp_grid], dtype=object)
    
    # Collar masking (ignore frames within `collar` of reference transitions)
    collar_frames = int(round(collar / frame_step))
    collar_mask = np.ones(len(ref_grid), dtype=bool)
    
    for turn in reference_turns:
        for boundary_time in [turn["start"], turn["end"]]:
            b_frame = int(round(boundary_time / frame_step))
            start_c = max(0, b_frame - collar_frames)
            end_c = min(len(ref_grid), b_frame + collar_frames + 1)
            collar_mask[start_c:end_c] = False
            
    # Compute counts on unmasked frames
    eval_ref = ref_grid[collar_mask]
    eval_hyp = mapped_hyp_grid[collar_mask]
    
    total_ref_speech_frames = int(np.sum(eval_ref != "silence"))
    total_ref_speech_sec = total_ref_speech_frames * frame_step
    
    # Missed Speech: Ref is speech, Hyp is silence
    missed_frames = int(np.sum((eval_ref != "silence") & (eval_hyp == "silence")))
    # False Alarm: Ref is silence, Hyp is speech
    false_alarm_frames = int(np.sum((eval_ref == "silence") & (eval_hyp != "silence")))
    # Speaker Confusion: Both are speech, but labels differ
    confusion_frames = int(np.sum((eval_ref != "silence") & (eval_hyp != "silence") & (eval_ref != eval_hyp)))
    
    total_error_frames = missed_frames + false_alarm_frames + confusion_frames
    
    der = total_error_frames / total_ref_speech_frames if total_ref_speech_frames > 0 else 0.0
    
    return {
        "der": round(der, 4),
        "der_pct": f"{der * 100:.2f}%",
        "diarization_accuracy_pct": f"{max(0.0, 1.0 - der) * 100:.2f}%",
        "missed_speech_sec": round(missed_frames * frame_step, 2),
        "false_alarm_sec": round(false_alarm_frames * frame_step, 2),
        "speaker_confusion_sec": round(confusion_frames * frame_step, 2),
        "total_speech_sec": round(total_ref_speech_sec, 2),
        "collar_seconds": collar,
        "speaker_mapping": best_mapping
    }


def evaluate_broker_client_separation(
    reference_turns: List[Dict[str, Any]],
    predicted_turns: List[Dict[str, Any]],
    total_duration: float,
    frame_step: float = 0.05
) -> Dict[str, Any]:
    """
    Evaluates how accurately the system separates the Broker from the Client.
    
    Calculates:
    - Call-Level Role Assignment Accuracy
    - Time-Weighted Separation Accuracy
    - Precision, Recall, F1-Score for Broker and Client
    - Confusion Matrix (Duration in seconds & percentage)
    """
    ref_grid = intervals_to_grid(reference_turns, total_duration, frame_step=frame_step, role_key="role")
    pred_grid = intervals_to_grid(predicted_turns, total_duration, frame_step=frame_step, role_key="role")
    
    # Mask speech frames (exclude silence for role classification evaluation)
    speech_mask = (ref_grid != "silence") & (pred_grid != "silence")
    ref_speech = ref_grid[speech_mask]
    pred_speech = pred_grid[speech_mask]
    
    total_eval_frames = len(ref_speech)
    total_eval_seconds = total_eval_frames * frame_step
    
    if total_eval_frames == 0:
        return {"status": "No overlapping speech to evaluate"}
        
    # Confusion Matrix Counts (in frames)
    # Broker = 'broker', Client = 'client'
    bb_frames = int(np.sum((ref_speech == "broker") & (pred_speech == "broker")))
    bc_frames = int(np.sum((ref_speech == "broker") & (pred_speech == "client")))
    cc_frames = int(np.sum((ref_speech == "client") & (pred_speech == "client")))
    cb_frames = int(np.sum((ref_speech == "client") & (pred_speech == "broker")))
    
    correct_frames = bb_frames + cc_frames
    time_weighted_accuracy = correct_frames / total_eval_frames if total_eval_frames > 0 else 0.0
    
    # Precision, Recall, F1 for Broker
    broker_tp = bb_frames
    broker_fp = cb_frames
    broker_fn = bc_frames
    broker_prec = broker_tp / (broker_tp + broker_fp) if (broker_tp + broker_fp) > 0 else 0.0
    broker_rec = broker_tp / (broker_tp + broker_fn) if (broker_tp + broker_fn) > 0 else 0.0
    broker_f1 = (2 * broker_prec * broker_rec / (broker_prec + broker_rec)) if (broker_prec + broker_rec) > 0 else 0.0
    
    # Precision, Recall, F1 for Client
    client_tp = cc_frames
    client_fp = bc_frames
    client_fn = cb_frames
    client_prec = client_tp / (client_tp + client_fp) if (client_tp + client_fp) > 0 else 0.0
    client_rec = client_tp / (client_tp + client_fn) if (client_tp + client_fn) > 0 else 0.0
    client_f1 = (2 * client_prec * client_rec / (client_prec + client_rec)) if (client_prec + client_rec) > 0 else 0.0
    
    macro_f1 = (broker_f1 + client_f1) / 2.0
    
    # Call-level role assignment: considered 100% correct if dominant accuracy > 70%
    call_role_assigned_correctly = bool(time_weighted_accuracy >= 0.70)
    
    return {
        "call_level_role_correct": call_role_assigned_correctly,
        "time_weighted_separation_accuracy": round(time_weighted_accuracy, 4),
        "time_weighted_separation_accuracy_pct": f"{time_weighted_accuracy * 100:.2f}%",
        "macro_f1": round(macro_f1, 4),
        "macro_f1_pct": f"{macro_f1 * 100:.2f}%",
        "evaluated_speech_duration_sec": round(total_eval_seconds, 2),
        "broker_metrics": {
            "precision": round(broker_prec, 4),
            "precision_pct": f"{broker_prec * 100:.2f}%",
            "recall": round(broker_rec, 4),
            "recall_pct": f"{broker_rec * 100:.2f}%",
            "f1_score": round(broker_f1, 4),
            "f1_score_pct": f"{broker_f1 * 100:.2f}%",
        },
        "client_metrics": {
            "precision": round(client_prec, 4),
            "precision_pct": f"{client_prec * 100:.2f}%",
            "recall": round(client_rec, 4),
            "recall_pct": f"{client_rec * 100:.2f}%",
            "f1_score": round(client_f1, 4),
            "f1_score_pct": f"{client_f1 * 100:.2f}%",
        },
        "confusion_matrix": {
            "broker_as_broker_sec": round(bb_frames * frame_step, 2),
            "broker_as_client_sec": round(bc_frames * frame_step, 2),
            "client_as_client_sec": round(cc_frames * frame_step, 2),
            "client_as_broker_sec": round(cb_frames * frame_step, 2),
            "broker_accuracy_pct": f"{(bb_frames / max(1, bb_frames + bc_frames)) * 100:.2f}%",
            "client_accuracy_pct": f"{(cc_frames / max(1, cc_frames + cb_frames)) * 100:.2f}%",
        }
    }


def evaluate_corpus_diarization(
    corpus_samples: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Evaluates Diarization and Broker-Client separation across multiple calls.
    """
    der_scores = []
    separation_accuracies = []
    broker_f1s = []
    client_f1s = []
    macro_f1s = []
    call_level_hits = 0
    
    total_bb_sec = 0.0
    total_bc_sec = 0.0
    total_cc_sec = 0.0
    total_cb_sec = 0.0
    
    per_call_reports = []
    
    for sample in corpus_samples:
        call_id = sample.get("call_id", "unknown")
        duration = sample.get("duration", 0.0)
        ref_turns = sample.get("reference_turns", [])
        pred_turns = sample.get("predicted_turns", [])
        
        der_res = evaluate_der(ref_turns, pred_turns, duration)
        sep_res = evaluate_broker_client_separation(ref_turns, pred_turns, duration)
        
        der_scores.append(der_res["der"])
        separation_accuracies.append(sep_res["time_weighted_separation_accuracy"])
        broker_f1s.append(sep_res["broker_metrics"]["f1_score"])
        client_f1s.append(sep_res["client_metrics"]["f1_score"])
        macro_f1s.append(sep_res["macro_f1"])
        
        if sep_res["call_level_role_correct"]:
            call_level_hits += 1
            
        cm = sep_res["confusion_matrix"]
        total_bb_sec += cm["broker_as_broker_sec"]
        total_bc_sec += cm["broker_as_client_sec"]
        total_cc_sec += cm["client_as_client_sec"]
        total_cb_sec += cm["client_as_broker_sec"]
        
        per_call_reports.append({
            "call_id": call_id,
            "duration_sec": duration,
            "der_pct": der_res["der_pct"],
            "separation_accuracy_pct": sep_res["time_weighted_separation_accuracy_pct"],
            "role_classification_correct": sep_res["call_level_role_correct"],
            "broker_f1_pct": sep_res["broker_metrics"]["f1_score_pct"],
            "client_f1_pct": sep_res["client_metrics"]["f1_score_pct"],
            "macro_f1_pct": sep_res["macro_f1_pct"],
            "confusion_matrix": cm
        })
        
    avg_der = float(np.mean(der_scores)) if der_scores else 0.0
    avg_sep_acc = float(np.mean(separation_accuracies)) if separation_accuracies else 0.0
    call_role_acc = (call_level_hits / len(corpus_samples)) if corpus_samples else 0.0
    avg_broker_f1 = float(np.mean(broker_f1s)) if broker_f1s else 0.0
    avg_client_f1 = float(np.mean(client_f1s)) if client_f1s else 0.0
    avg_macro_f1 = float(np.mean(macro_f1s)) if macro_f1s else 0.0
    
    return {
        "summary": {
            "total_calls": len(corpus_samples),
            "call_level_role_accuracy_pct": f"{call_role_acc * 100:.2f}%",
            "time_weighted_separation_accuracy_pct": f"{avg_sep_acc * 100:.2f}%",
            "average_der_pct": f"{avg_der * 100:.2f}%",
            "macro_f1_pct": f"{avg_macro_f1 * 100:.2f}%",
            "broker_f1_pct": f"{avg_broker_f1 * 100:.2f}%",
            "client_f1_pct": f"{avg_client_f1 * 100:.2f}%",
            "aggregate_confusion_matrix": {
                "broker_as_broker_sec": round(total_bb_sec, 2),
                "broker_as_client_sec": round(total_bc_sec, 2),
                "client_as_client_sec": round(total_cc_sec, 2),
                "client_as_broker_sec": round(total_cb_sec, 2),
            }
        },
        "per_call_details": per_call_reports
    }
