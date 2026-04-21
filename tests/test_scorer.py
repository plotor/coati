import os
import tempfile

import pytest

from coati.scorer.grammar import GrammarScorer


class TestGrammarScorer:

    def test_load_model_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            GrammarScorer("/nonexistent/path/model.lm")

    def test_model_loading(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".lm", delete=False) as f:
            f.write("-1.2345\tthe\t-0.5\n")
            f.write("-2.3456\tthe cat\t-0.3\n")
            f.write("-3.4567\tthe cat sat\n")
            f.flush()
            scorer = GrammarScorer(f.name)

        assert "the" in scorer.ngram_model
        assert "the cat" in scorer.ngram_model
        assert "the cat sat" in scorer.ngram_model
        os.unlink(f.name)

    def test_cal_fluency(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".lm", delete=False) as f:
            f.write("-1.0\tthe\t-0.5\n")
            f.write("-2.0\tthe cat\t-0.3\n")
            f.write("-3.0\tthe cat sat\n")
            f.write("-1.5\tcat\t-0.2\n")
            f.write("-2.5\tcat sat\t-0.1\n")
            f.write("-3.5\tcat sat on\n")
            f.write("-1.2\tsat\t-0.4\n")
            f.write("-2.2\tsat on\t-0.15\n")
            f.write("-3.2\tsat on the\n")
            f.write("-1.1\ton\t-0.3\n")
            f.write("-2.1\ton the\t-0.2\n")
            f.write("-3.1\ton the mat\n")
            f.write("-1.3\tmat\t-0.1\n")
            f.flush()
            scorer = GrammarScorer(f.name)

        score = scorer.cal_fluency("the cat sat on the mat")
        assert isinstance(score, float)
        os.unlink(f.name)

    def test_empty_model(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".lm", delete=False) as f:
            f.write("")
            f.flush()
            scorer = GrammarScorer(f.name)

        assert len(scorer.ngram_model) == 0
        os.unlink(f.name)
