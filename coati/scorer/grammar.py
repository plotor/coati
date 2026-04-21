import logging
import re
from typing import Dict, Tuple


class GrammarScorer:
    """N-gram language model scorer for evaluating sentence fluency.

    Loads an ARPA-format n-gram model and computes fluency scores for
    input sentences using trigram probabilities with Katz back-off.
    """

    def __init__(self, modelpath: str):
        """Initialize the GrammarScorer.

        Args:
            modelpath: Path to the ARPA-format n-gram model file.
        """
        self.modelpath = modelpath
        self.ngram_model: Dict[str, Tuple[float, float]] = self._load_ngram_model()

    def cal_fluency(self, sentence: str) -> float:
        """Calculate the fluency score of a sentence.

        Uses trigram probabilities with back-off to compute the overall
        log-probability of the sentence, then converts it to a linear score.

        Args:
            sentence: A whitespace-tokenized sentence string.

        Returns:
            The fluency score as a float. Higher values indicate better fluency.
        """
        score = 0.0
        sent = "<s> " + sentence + " </s>"
        tokens = re.split(r"\s+", sent)

        for i in range(2, len(tokens)):
            w1 = tokens[i - 2]
            w2 = tokens[i - 1]
            w3 = tokens[i]

            if (w1 not in self.ngram_model) and w1 not in ("<s>", "</s>"):
                w1 = "<unk>"
            if w2 not in self.ngram_model:
                w2 = "<unk>"
            if (w3 not in self.ngram_model) and w3 not in ("<s>", "</s>"):
                w3 = "<unk>"

            score += float(10 ** self._extract_ngram_score(f"{w1} {w2} {w3}"))

        return score

    def _extract_ngram_score(self, wordstr: str) -> float:
        """Recursively extract the log-probability of an n-gram.

        Applies Katz back-off: if the full n-gram is not found, backs off
        to (n-1)-gram probability plus the back-off weight of the context.

        Args:
            wordstr: A whitespace-separated n-gram string (1-gram to 3-gram).

        Returns:
            The log-probability score as a float.
        """
        words = re.split(r"\s+", wordstr)

        if len(words) == 3:
            if wordstr in self.ngram_model:
                return self.ngram_model[wordstr][0]
            bigram = f"{words[0]} {words[1]}"
            if bigram in self.ngram_model:
                return self.ngram_model[bigram][1] + self._extract_ngram_score(bigram)
            return self._extract_ngram_score(f"{words[1]} {words[2]}")

        if len(words) == 2:
            bigram = f"{words[0]} {words[1]}"
            if bigram in self.ngram_model:
                return self.ngram_model[bigram][0]
            unigram_score = self.ngram_model.get(words[0], (0.0, 0.0))
            return unigram_score[1] + self._extract_ngram_score(words[1])

        return self.ngram_model.get(words[0], (0.0, 0.0))[0]

    def _load_ngram_model(self) -> Dict[str, Tuple[float, float]]:
        """Load an ARPA-format n-gram model from disk.

        Each line in the model file is expected to have the format:
            log_prob\\tngram\\tback_off_weight
        The back-off weight is optional and defaults to 0.0.

        Args:
            None (uses self.modelpath).

        Returns:
            A dictionary mapping n-gram strings to (log_prob, back_off_weight) tuples.
        """
        ngram_model: Dict[str, Tuple[float, float]] = {}
        logging.info("loading ngram model...")

        with open(self.modelpath, mode="r") as modelfile:
            for line in modelfile:
                strs = re.split(r"\t", line.strip())
                if len(strs) < 2:
                    logging.warning("invalid line format, ignoring: %s", line.strip())
                    continue

                prob_score = float(strs[0])
                tri_gram = strs[1]
                back_off_score = float(strs[2]) if len(strs) == 3 else 0.0
                ngram_model[tri_gram] = (prob_score, back_off_score)

        logging.info("load ngram model finished!")
        return ngram_model
