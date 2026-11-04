"""
Unit Tests for Egyptian Arabic Text Normalization
"""

import unittest
from src.text_normalization import (
    normalize_arabic_text,
    tokenize_arabic_words,
    tokenize_arabic_chars,
    remove_diacritics
)


class TestTextNormalization(unittest.TestCase):

    def test_alef_normalization(self):
        self.assertEqual(normalize_arabic_text("أحمد"), "احمد")
        self.assertEqual(normalize_arabic_text("إبراهيم"), "ابراهيم")
        self.assertEqual(normalize_arabic_text("آية"), "ايه")

    def test_yeh_and_taa_normalization(self):
        self.assertEqual(normalize_arabic_text("على"), "علي")
        self.assertEqual(normalize_arabic_text("شقة"), "شقه")

    def test_tatweel_and_diacritics(self):
        self.assertEqual(normalize_arabic_text("شـقـةٌ"), "شقه")
        self.assertEqual(normalize_arabic_text("مَسَاءُ الخَيْرِ"), "مساء الخير")

    def test_punctuation_and_whitespace(self):
        self.assertEqual(
            normalize_arabic_text("  مساء الخير، لو سمحت!  "),
            "مساء الخير لو سمحت"
        )

    def test_digits_normalization(self):
        self.assertEqual(normalize_arabic_text("٥ مليون"), "5 مليون")


if __name__ == "__main__":
    unittest.main()
