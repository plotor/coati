import bisect
import math
import re
from typing import Dict, List, Set, Tuple

import networkx as nx


class KeyphraseReranker:
    """Reranks N-best compression candidates using keyphrase extraction.

    Extracts keyphrases from the input sentences using a modified TextRank
    algorithm, then reranks compression candidates by normalizing their
    scores with the sum of keyphrase scores they contain. Keyphrases that
    are substrings of longer keyphrases are clustered and deduplicated.

    Reference:
        Boudin, F. and Morin, E. (2013). Keyphrase Extraction for N-best
        Reranking in Multi-Sentence Compression. NAACL-HLT 2013.
    """

    def __init__(
        self,
        sentence_list: List[str],
        nbest_compressions: List[Tuple[float, list]],
        lang: str = "en",
        patterns: List[str] = None,
        stopwords: Set[str] = None,
        pos_separator: str = "/",
    ):
        """Initialize the KeyphraseReranker.

        Args:
            sentence_list: List of POS-tagged sentence strings.
            nbest_compressions: N-best compression candidates as
                (score, path) tuples.
            lang: Language code, "en" or "fr" (default: "en").
            patterns: Additional POS patterns for keyphrase filtering.
            stopwords: Set of stopwords to exclude from keyphrase extraction.
            pos_separator: Character separating word and POS tag (default: "/").
        """
        self.sentences: List[str] = list(sentence_list)
        self.nbest_compressions: List[Tuple[float, list]] = nbest_compressions
        self.graph: nx.Graph = nx.Graph()
        self.lang: str = lang
        self.stopwords: Set[str] = set(stopwords or [])
        self.pos_separator: str = pos_separator

        self.syntactic_filter: List[str] = ["JJ", "NNP", "NNS", "NN", "NNPS"]
        self.keyphrase_candidates: Dict[str, list] = {}
        self.word_scores: Dict[tuple, float] = {}
        self.keyphrase_scores: Dict[str, float] = {}
        self.syntactic_patterns: List[str] = [r"^(JJ)*(NNP|NNS|NN)+$"]

        if self.lang == "fr":
            self.syntactic_filter = ["NPP", "NC", "ADJ"]
            self.syntactic_patterns = [r"^(ADJ)*(NC|NPP)+(ADJ)*$"]

        if patterns:
            self.syntactic_patterns.extend(patterns)

        self._build_graph()
        self._generate_candidates()
        self._undirected_text_rank()
        self._score_keyphrase_candidates()
        self._cluster_keyphrase_candidates()

    def _build_graph(self, window: int = 0) -> None:
        """Build an undirected co-occurrence graph from the input sentences.

        Nodes are (word, POS) tuples belonging to the syntactic filter
        categories. Edges represent co-occurrence within a window (default:
        the whole sentence), weighted by co-occurrence count.

        Args:
            window: Co-occurrence window size. 0 or negative means the
                entire sentence (default: 0).
        """
        for i in range(len(self.sentences)):
            self.sentences[i] = re.sub(" +", " ", self.sentences[i])
            sentence = self.sentences[i].split(" ")

            for j in range(len(sentence)):
                word, pos = self._wordpos_to_tuple(sentence[j])
                sentence[j] = (word.lower(), pos)

                if sentence[j][0] in self.stopwords:
                    sentence[j] = (sentence[j][0], "STOPWORD")

                if sentence[j][1] in self.syntactic_filter:
                    if not self.graph.has_node(sentence[j]):
                        self.graph.add_node(sentence[j])

            for j in range(len(sentence)):
                first_node = sentence[j]
                max_window = window if window >= 1 else len(sentence)

                for k in range(j + 1, min(len(sentence), j + max_window)):
                    second_node = sentence[k]

                    if self.graph.has_node(first_node) and self.graph.has_node(
                        second_node
                    ):
                        if not self.graph.has_edge(first_node, second_node):
                            self.graph.add_edge(
                                first_node, second_node, weight=1
                            )
                        else:
                            self.graph[first_node][second_node]["weight"] += 1

            self.sentences[i] = sentence

    def _generate_candidates(self) -> None:
        """Extract keyphrase candidates from the sentences.

        A keyphrase candidate is the longest contiguous sequence of words
        whose POS tags all belong to the syntactic filter, and whose
        combined POS pattern matches one of the syntactic patterns.
        """
        for i in range(len(self.sentences)):
            sentence = self.sentences[i]
            candidate: list = []

            for j in range(len(sentence)):
                word, pos = sentence[j]

                if pos in self.syntactic_filter:
                    candidate.append(sentence[j])
                elif len(candidate) > 0 and self._is_a_candidate(candidate):
                    keyphrase = " ".join(u[0] for u in candidate)
                    self.keyphrase_candidates[keyphrase] = candidate
                    candidate = []
                else:
                    candidate = []

            if len(candidate) > 0 and self._is_a_candidate(candidate):
                keyphrase = " ".join(u[0] for u in candidate)
                self.keyphrase_candidates[keyphrase] = candidate

    def _is_a_candidate(self, keyphrase_candidate: list) -> bool:
        """Check if a keyphrase candidate matches the syntactic patterns.

        Args:
            keyphrase_candidate: A list of (word, pos) tuples.

        Returns:
            True if the combined POS pattern matches all syntactic patterns.
        """
        candidate_pattern = "".join(u[1] for u in keyphrase_candidate)
        for pattern in self.syntactic_patterns:
            if not re.search(pattern, candidate_pattern):
                return False
        return True

    def _undirected_text_rank(self, d: float = 0.85, f_conv: float = 0.0001, max_iter: int = 100) -> None:
        """Compute word scores using the TextRank algorithm.

        Iteratively updates node scores until convergence or the maximum
        number of iterations is reached.

        Args:
            d: Damping factor (default: 0.85).
            f_conv: Convergence threshold (default: 0.0001).
            max_iter: Maximum number of iterations (default: 100).
        """
        if self.graph.number_of_nodes() == 0:
            return

        self.word_scores = {node: 1.0 for node in self.graph.nodes()}

        for _ in range(max_iter):
            current_node_scores = self.word_scores.copy()
            max_node_difference = 0.0

            for node_i in self.graph.nodes():
                sum_vj = 0.0

                for node_j in self.graph.neighbors(node_i):
                    wji = self.graph[node_j][node_i]["weight"]
                    ws_vj = current_node_scores[node_j]
                    sum_wjk = sum(
                        self.graph[node_j][node_k]["weight"]
                        for node_k in self.graph.neighbors(node_j)
                    )
                    if sum_wjk == 0:
                        continue
                    sum_vj += (wji * ws_vj) / sum_wjk

                self.word_scores[node_i] = (1 - d) + (d * sum_vj)

                score_difference = math.fabs(
                    self.word_scores[node_i] - current_node_scores[node_i]
                )
                max_node_difference = max(max_node_difference, score_difference)

            if max_node_difference < f_conv:
                break

    def _score_keyphrase_candidates(self) -> None:
        """Compute the score of each keyphrase candidate.

        The score is the sum of its word TextRank scores normalized by
        (keyphrase length + 1).
        """
        for keyphrase in self.keyphrase_candidates:
            keyphrase_score = sum(
                self.word_scores[word_pos_tuple]
                for word_pos_tuple in self.keyphrase_candidates[keyphrase]
            )
            keyphrase_score /= len(self.keyphrase_candidates[keyphrase]) + 1.0
            self.keyphrase_scores[keyphrase] = keyphrase_score

    def _cluster_keyphrase_candidates(self) -> None:
        """Cluster keyphrase candidates to remove redundancy.

        Keyphrases whose words are all contained within a longer keyphrase
        are grouped into the same cluster. Only the highest-scoring
        keyphrase from each cluster is retained. Substring redundancy
        between cluster representatives is also removed.
        """
        descending = sorted(
            self.keyphrase_candidates,
            key=lambda x: len(self.keyphrase_candidates[x]),
            reverse=True,
        )

        clusters: Dict[str, list] = {}

        for keyphrase in descending:
            found_cluster = False
            keyphrase_words = set(keyphrase.split(" "))

            for cluster in clusters:
                cluster_words = set(cluster.split(" "))
                if len(keyphrase_words.difference(cluster_words)) == 0:
                    clusters[cluster].append(keyphrase)
                    found_cluster = True

            if not found_cluster:
                clusters[keyphrase] = [keyphrase]

        best_candidate_keyphrases: list = []
        for cluster in clusters:
            sorted_cluster = sorted(
                clusters[cluster],
                key=lambda c: self.keyphrase_scores[c],
                reverse=True,
            )
            best_candidate_keyphrases.append(sorted_cluster[0])

        non_redundant_keyphrases: list = []
        sorted_keyphrases = sorted(
            best_candidate_keyphrases,
            key=lambda kp: self.keyphrase_scores[kp],
            reverse=True,
        )

        for keyphrase in sorted_keyphrases:
            is_redundant = any(
                keyphrase in prev_keyphrase for prev_keyphrase in non_redundant_keyphrases
            )
            if not is_redundant:
                non_redundant_keyphrases.append(keyphrase)

        for keyphrase in list(self.keyphrase_candidates.keys()):
            if keyphrase not in non_redundant_keyphrases:
                del self.keyphrase_candidates[keyphrase]
                del self.keyphrase_scores[keyphrase]

    def rerank_nbest_compressions(self) -> List[Tuple[float, list]]:
        """Rerank the N-best compressions according to keyphrase scores.

        Each compression's original score is normalized by
        (path_length * total_keyphrase_score), where total_keyphrase_score
        is the sum of scores of all keyphrases found in the compression.

        Returns:
            A sorted list of (reranked_score, path) tuples.
        """
        reranked_compressions: list = []

        for cummulative_score, path in self.nbest_compressions:
            compression = " ".join([u[0] for u in path])
            total_keyphrase_score = 1.0

            for keyphrase in self.keyphrase_candidates:
                if keyphrase in compression:
                    total_keyphrase_score += self.keyphrase_scores[keyphrase]

            score = cummulative_score / (len(path) * total_keyphrase_score)
            bisect.insort(reranked_compressions, (score, path))

        return reranked_compressions

    def _wordpos_to_tuple(self, word: str) -> Tuple[str, str]:
        """Convert a "word/POS" string to a (word, pos) tuple.

        Args:
            word: A word/POS string.

        Returns:
            A (lowercase_word, pos) tuple.
        """
        pos_separator_re = re.escape(self.pos_separator)
        m = re.match(r"^(.+)" + pos_separator_re + r"(.+)$", word)
        token, pos = m.group(1), m.group(2)
        return token.lower(), pos

    def _tuple_to_wordpos(self, wordpos_tuple: Tuple[str, str]) -> str:
        """Convert a (word, pos) tuple back to a "word/POS" string.

        Args:
            wordpos_tuple: A (word, pos) tuple.

        Returns:
            A "word/POS" string.
        """
        return wordpos_tuple[0] + self.pos_separator + wordpos_tuple[1]
