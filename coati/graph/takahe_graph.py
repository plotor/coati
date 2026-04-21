import re
from typing import Dict, List, Set, Tuple

from .base import BaseWordGraph


class TakaheWordGraph(BaseWordGraph):
    """Word graph implementation based on the original Takahe algorithm.

    Uses word frequency for edge weight computation. Input sentences are
    expected in "word/POS" format (e.g., "the/DT cat/NN sat/VBD").
    All words are assigned a uniform weight of 1.0.

    Reference:
        Filippova, K. (2010). Multi-sentence compression: Finding shortest
        paths in word graphs. COLING 2010.
    """

    def pre_process_sentences(self) -> None:
        """Parse each sentence from "word/POS" format into (token, pos, 1.0) tuples.

        Adds start and end markers. All word weights are set to 1.0 since
        the Takahe algorithm uses frequency-based rather than event-based weighting.
        """
        for i in range(self.length):
            self.sentence[i] = re.sub(" +", " ", self.sentence[i])
            self.sentence[i] = self.sentence[i].strip()

            words = self.sentence[i].split(" ")

            container: List[Tuple[str, str, float]] = [
                (self.start, self.start, 1.0)
            ]

            for w in words:
                pos_separator_re = re.escape(self.pos_separator)
                m = re.match(r"^(.+)" + pos_separator_re + r"(.+)$", w)
                token, pos = m.group(1), m.group(2)
                container.append((token.lower(), pos, 1.0))

            container.append((self.stop, self.stop, 1.0))
            self.sentence[i] = container

    def compute_statistics(self) -> None:
        """Compute term frequency for each word/POS node in the graph."""
        terms: Dict[str, List[int]] = {}

        for i in range(self.length):
            for token, pos, _weight in self.sentence[i]:
                node = token.lower() + self.sep + pos
                if node not in terms:
                    terms[node] = [i]
                else:
                    terms[node].append(i)

        for key in terms:
            self.term_freq[key] = len(terms[key])

    def _add_edges(self, sent_idx: int, mapping: list) -> None:
        """Add consecutive directed edges between adjacent mapped words.

        Args:
            sent_idx: Index of the current sentence.
            mapping: Mapping list from word positions to graph nodes.
        """
        for j in range(1, len(mapping)):
            self.graph.add_edge(mapping[j - 1], mapping[j])

    def _compute_edge_weights(self) -> None:
        """Compute edge weights using frequency-based formula.

        Edge weight = ((freq1 + freq2) / sum_diff) / (freq1 * freq2),
        where freq is the number of sentences containing the word.
        """
        for node1, node2 in self.graph.edges():
            edge_weight = self._get_edge_weight(node1, node2)
            self.graph.add_edge(node1, node2, weight=edge_weight)

    def _get_edge_weight(self, node1: tuple, node2: tuple) -> float:
        """Compute the weight of an edge using word frequency.

        Args:
            node1: The source node tuple.
            node2: The target node tuple.

        Returns:
            The computed edge weight as a float.
        """
        info1 = self.graph.nodes[node1]["info"]
        info2 = self.graph.nodes[node2]["info"]

        freq1 = len(info1)
        freq2 = len(info2)

        diff = self._compute_diff_for_edge(node1, node2)

        sum_diff = sum(diff)
        if sum_diff == 0:
            return 0.0

        return ((freq1 + freq2) / sum_diff) / (freq1 * freq2)

    def get_compression(self, nb_candidates: int = 50) -> List[Tuple[float, list]]:
        """Extract compression candidates using k-shortest paths.

        Args:
            nb_candidates: Maximum number of compression candidates to return.

        Returns:
            A sorted list of (score, path) tuples.
        """
        return self._get_compression_from_ksp(nb_candidates)
