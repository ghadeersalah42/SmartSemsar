"""
Unit Tests for ASR Evaluation Metrics (WER, CER, Levenshtein Alignment)
"""

import unittest
from src.asr_evaluator import (
    calculate_wer,
    calculate_cer,
    compute_levenshtein_alignment,
    evaluate_corpus_asr
)


class TestASRMetrics(unittest.TestCase):

    def test_identical_strings(self):
        ref = "مساء الخير لو سمحت"
        hyp = "مساء الخير لو سمحت"
        wer_res = calculate_wer(ref, hyp)
        self.assertEqual(wer_res["error_rate"], 0.0)
        self.assertEqual(wer_res["hits"], 4)
        self.assertEqual(wer_res["substitutions"], 0)

    def test_single_substitution(self):
        ref = "عايز شقة في التجمع"
        hyp = "عايز فيلا في التجمع"
        wer_res = calculate_wer(ref, hyp)
        self.assertEqual(wer_res["substitutions"], 1)
        self.assertEqual(wer_res["hits"], 3)
        self.assertAlmostEqual(wer_res["error_rate"], 0.25, places=3)

    def test_deletion_and_insertion(self):
        ref = "واحد اتنين تلاتة"
        hyp = "واحد تلاتة اربعة"
        wer_res = calculate_wer(ref, hyp)
        # 'اتنين' deleted, 'اربعة' inserted, 'واحد' and 'تلاتة' hit
        self.assertEqual(wer_res["hits"], 2)
        self.assertEqual(wer_res["deletions"], 1)
        self.assertEqual(wer_res["insertions"], 1)
        self.assertAlmostEqual(wer_res["error_rate"], 2.0 / 3.0, places=3)

    def test_cer_calculation(self):
        ref = "شقة"
        hyp = "شقه"
        # Without normalization: 1 substitution out of 3 chars
        cer_no_norm = calculate_cer(ref, hyp, normalize=False)
        self.assertEqual(cer_no_norm["substitutions"], 1)
        # With normalization: ة -> ه so hits = 3, CER = 0
        cer_norm = calculate_cer(ref, hyp, normalize=True)
        self.assertEqual(cer_norm["error_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
