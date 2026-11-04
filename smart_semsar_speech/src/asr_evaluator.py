"""
ASR Evaluation Engine for Egyptian Arabic
Computes Word Error Rate (WER), Character Error Rate (CER),
Alignment Matrix (Substitutions, Deletions, Insertions, Hits), and detailed error reports.
Uses standard NIST SCLITE alignment weighting (Sub=4, Del=3, Ins=3) for optimal alignment.
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np

try:
    from .text_normalization import (
        normalize_arabic_text,
        tokenize_arabic_words,
        tokenize_arabic_chars,
    )
except ImportError:
    from text_normalization import (
        normalize_arabic_text,
        tokenize_arabic_words,
        tokenize_arabic_chars,
    )


class ASRErrorMetrics:
    """Container for ASR error rate evaluation metrics."""
    
    def __init__(
        self,
        hits: int = 0,
        substitutions: int = 0,
        deletions: int = 0,
        insertions: int = 0,
        reference_length: int = 0,
    ):
        self.hits = hits
        self.substitutions = substitutions
        self.deletions = deletions
        self.insertions = insertions
        self.reference_length = reference_length
        
    @property
    def total_errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions
    
    @property
    def error_rate(self) -> float:
        if self.reference_length == 0:
            return 0.0 if self.insertions == 0 else 1.0
        return self.total_errors / self.reference_length

    @property
    def accuracy(self) -> float:
        return max(0.0, 1.0 - self.error_rate)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error_rate": round(self.error_rate, 4),
            "error_rate_pct": f"{self.error_rate * 100:.2f}%",
            "accuracy_pct": f"{self.accuracy * 100:.2f}%",
            "hits": self.hits,
            "substitutions": self.substitutions,
            "deletions": self.deletions,
            "insertions": self.insertions,
            "total_errors": self.total_errors,
            "reference_length": self.reference_length,
        }


def compute_levenshtein_alignment(
    ref_tokens: List[str],
    hyp_tokens: List[str],
    sub_cost_weight: int = 4,
    del_cost_weight: int = 3,
    ins_cost_weight: int = 3
) -> Tuple[ASRErrorMetrics, List[Tuple[str, Optional[str], Optional[str]]]]:
    """
    Computes exact NIST SCLITE Levenshtein alignment between reference and hypothesis tokens.
    
    Using Sub=4, Del=3, Ins=3 prevents misaligning identical words, matching industry standards.
    """
    n = len(ref_tokens)
    m = len(hyp_tokens)
    
    dp = np.zeros((n + 1, m + 1), dtype=int)
    
    for i in range(n + 1):
        dp[i][0] = i * del_cost_weight
    for j in range(m + 1):
        dp[0][j] = j * ins_cost_weight
        
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_tokens[i - 1] == hyp_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                sub_c = dp[i - 1][j - 1] + sub_cost_weight
                del_c = dp[i - 1][j] + del_cost_weight
                ins_c = dp[i][j - 1] + ins_cost_weight
                dp[i][j] = min(sub_c, del_c, ins_c)
                
    # Backtrack to reconstruct alignment operations
    i, j = n, m
    alignment: List[Tuple[str, Optional[str], Optional[str]]] = []
    hits = 0
    subs = 0
    dels = 0
    ins = 0
    
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref_tokens[i - 1] == hyp_tokens[j - 1]:
            alignment.append(("hit", ref_tokens[i - 1], hyp_tokens[j - 1]))
            hits += 1
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + del_cost_weight:
            alignment.append(("del", ref_tokens[i - 1], None))
            dels += 1
            i -= 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + ins_cost_weight:
            alignment.append(("ins", None, hyp_tokens[j - 1]))
            ins += 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + sub_cost_weight:
            alignment.append(("sub", ref_tokens[i - 1], hyp_tokens[j - 1]))
            subs += 1
            i -= 1
            j -= 1
        else:
            # Fallback
            if i > 0 and j > 0:
                alignment.append(("sub", ref_tokens[i - 1], hyp_tokens[j - 1]))
                subs += 1
                i -= 1
                j -= 1
            elif i > 0:
                alignment.append(("del", ref_tokens[i - 1], None))
                dels += 1
                i -= 1
            else:
                alignment.append(("ins", None, hyp_tokens[j - 1]))
                ins += 1
                j -= 1
            
    alignment.reverse()
    metrics = ASRErrorMetrics(
        hits=hits,
        substitutions=subs,
        deletions=dels,
        insertions=ins,
        reference_length=n
    )
    return metrics, alignment


def calculate_wer(
    reference: str,
    hypothesis: str,
    normalize: bool = True
) -> Dict[str, Any]:
    """Calculates Word Error Rate (WER) between reference and hypothesis Egyptian Arabic texts."""
    if normalize:
        ref_tokens = tokenize_arabic_words(reference)
        hyp_tokens = tokenize_arabic_words(hypothesis)
    else:
        ref_tokens = reference.split()
        hyp_tokens = hypothesis.split()
        
    metrics, alignment = compute_levenshtein_alignment(ref_tokens, hyp_tokens)
    
    result = metrics.to_dict()
    result["metric"] = "WER"
    result["reference_tokens"] = len(ref_tokens)
    result["hypothesis_tokens"] = len(hyp_tokens)
    result["alignment"] = alignment
    return result


def calculate_cer(
    reference: str,
    hypothesis: str,
    normalize: bool = True,
    include_spaces: bool = True
) -> Dict[str, Any]:
    """Calculates Character Error Rate (CER) between reference and hypothesis Egyptian Arabic texts."""
    if normalize:
        ref_chars = tokenize_arabic_chars(reference, include_spaces=include_spaces)
        hyp_chars = tokenize_arabic_chars(hypothesis, include_spaces=include_spaces)
    else:
        ref_chars = list(reference) if include_spaces else [c for c in reference if c != " "]
        hyp_chars = list(hypothesis) if include_spaces else [c for c in hypothesis if c != " "]
        
    metrics, alignment = compute_levenshtein_alignment(ref_chars, hyp_chars)
    
    result = metrics.to_dict()
    result["metric"] = "CER"
    result["reference_chars"] = len(ref_chars)
    result["hypothesis_chars"] = len(hyp_chars)
    result["alignment"] = alignment
    return result


def evaluate_corpus_asr(
    samples: List[Dict[str, str]],
    normalize: bool = True
) -> Dict[str, Any]:
    """Evaluates corpus-level WER and CER across multiple call utterances."""
    total_word_hits = 0
    total_word_subs = 0
    total_word_dels = 0
    total_word_ins = 0
    total_word_ref_len = 0
    
    total_char_hits = 0
    total_char_subs = 0
    total_char_dels = 0
    total_char_ins = 0
    total_char_ref_len = 0
    
    per_sample_results = []
    
    for idx, sample in enumerate(samples, start=1):
        sample_id = sample.get("id", f"call_{idx:02d}")
        ref = sample.get("reference", "")
        hyp = sample.get("hypothesis", "")
        
        wer_res = calculate_wer(ref, hyp, normalize=normalize)
        cer_res = calculate_cer(ref, hyp, normalize=normalize)
        
        total_word_hits += wer_res["hits"]
        total_word_subs += wer_res["substitutions"]
        total_word_dels += wer_res["deletions"]
        total_word_ins += wer_res["insertions"]
        total_word_ref_len += wer_res["reference_length"]
        
        total_char_hits += cer_res["hits"]
        total_char_subs += cer_res["substitutions"]
        total_char_dels += cer_res["deletions"]
        total_char_ins += cer_res["insertions"]
        total_char_ref_len += cer_res["reference_length"]
        
        per_sample_results.append({
            "id": sample_id,
            "reference": ref,
            "hypothesis": hyp,
            "normalized_reference": normalize_arabic_text(ref) if normalize else ref,
            "normalized_hypothesis": normalize_arabic_text(hyp) if normalize else hyp,
            "wer": wer_res["error_rate"],
            "wer_pct": wer_res["error_rate_pct"],
            "cer": cer_res["error_rate"],
            "cer_pct": cer_res["error_rate_pct"],
            "word_errors": wer_res["total_errors"],
            "word_ref_length": wer_res["reference_length"],
            "word_breakdown": {
                "S": wer_res["substitutions"],
                "D": wer_res["deletions"],
                "I": wer_res["insertions"],
                "H": wer_res["hits"],
            },
            "char_errors": cer_res["total_errors"],
            "char_ref_length": cer_res["reference_length"],
        })
        
    aggregate_wer = (
        (total_word_subs + total_word_dels + total_word_ins) / total_word_ref_len
        if total_word_ref_len > 0 else 0.0
    )
    aggregate_cer = (
        (total_char_subs + total_char_dels + total_char_ins) / total_char_ref_len
        if total_char_ref_len > 0 else 0.0
    )
    
    return {
        "summary": {
            "total_calls": len(samples),
            "aggregate_wer": round(aggregate_wer, 4),
            "aggregate_wer_pct": f"{aggregate_wer * 100:.2f}%",
            "aggregate_cer": round(aggregate_cer, 4),
            "aggregate_cer_pct": f"{aggregate_cer * 100:.2f}%",
            "word_accuracy_pct": f"{(1.0 - aggregate_wer) * 100:.2f}%",
            "character_accuracy_pct": f"{(1.0 - aggregate_cer) * 100:.2f}%",
            "total_reference_words": total_word_ref_len,
            "total_word_substitutions": total_word_subs,
            "total_word_deletions": total_word_dels,
            "total_word_insertions": total_word_ins,
            "total_word_hits": total_word_hits,
            "total_reference_chars": total_char_ref_len,
            "total_char_substitutions": total_char_subs,
            "total_char_deletions": total_char_dels,
            "total_char_insertions": total_char_ins,
        },
        "per_call_details": per_sample_results
    }


def format_alignment_display(alignment: List[Tuple[str, Optional[str], Optional[str]]]) -> str:
    """Formats an alignment list into a readable visual side-by-side string."""
    ref_line = []
    hyp_line = []
    op_line = []
    
    for op, ref, hyp in alignment:
        r_str = ref if ref is not None else "*"
        h_str = hyp if hyp is not None else "*"
        width = max(len(r_str), len(h_str), len(op.upper())) + 2
        
        ref_line.append(r_str.center(width))
        hyp_line.append(h_str.center(width))
        op_line.append(op.upper().center(width))
        
    return f"REF: {''.join(ref_line)}\nHYP: {''.join(hyp_line)}\nOP : {''.join(op_line)}"
