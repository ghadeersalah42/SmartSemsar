"""
Unit Tests for Broker vs Client Role Separation and Diarization Evaluation
"""

import unittest
from src.role_separator import RoleSeparator
from src.diarization_evaluator import (
    evaluate_der,
    evaluate_broker_client_separation
)


class TestRoleSeparation(unittest.TestCase):

    def setUp(self):
        self.separator = RoleSeparator()

    def test_broker_client_classification(self):
        conversation = [
            {
                "speaker": "SPEAKER_00",
                "text": "مساء الخير أنا بدور على شقة في التجمع وميزانيتي خمسة مليون"
            },
            {
                "speaker": "SPEAKER_01",
                "text": "أهلاً بحضرتك يا فندم معاك شركة هومز للوساطة العقارية عندنا كمبوند ممتاز بمقدم 10%"
            }
        ]
        result = self.separator.separate_roles(conversation)
        
        self.assertEqual(result["role_mapping"]["SPEAKER_00"], "client")
        self.assertEqual(result["role_mapping"]["SPEAKER_01"], "broker")
        self.assertGreaterEqual(result["confidence"], 0.70)

    def test_der_calculation(self):
        ref_turns = [
            {"start": 0.0, "end": 5.0, "speaker": "SPEAKER_00"},
            {"start": 5.0, "end": 10.0, "speaker": "SPEAKER_01"}
        ]
        hyp_turns = [
            {"start": 0.0, "end": 5.0, "speaker": "SPEAKER_00"},
            {"start": 5.0, "end": 10.0, "speaker": "SPEAKER_01"}
        ]
        res = evaluate_der(ref_turns, hyp_turns, total_duration=10.0)
        self.assertEqual(res["der"], 0.0)

    def test_role_separation_accuracy(self):
        ref_turns = [
            {"start": 0.0, "end": 5.0, "role": "client"},
            {"start": 5.0, "end": 10.0, "role": "broker"}
        ]
        pred_turns = [
            {"start": 0.0, "end": 5.0, "role": "client"},
            {"start": 5.0, "end": 10.0, "role": "broker"}
        ]
        res = evaluate_broker_client_separation(ref_turns, pred_turns, total_duration=10.0)
        self.assertEqual(res["time_weighted_separation_accuracy"], 1.0)
        self.assertTrue(res["call_level_role_correct"])
        self.assertEqual(res["broker_metrics"]["f1_score"], 1.0)


if __name__ == "__main__":
    unittest.main()
