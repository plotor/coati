import logging
import re
from queue import Queue
from typing import Dict, List, Optional, Tuple

import networkx as nx

from .base import BaseWordGraph
from ..scorer.grammar import GrammarScorer


class CoatiWordGraph(BaseWordGraph):
    """Word graph implementation with event-guided multi-sentence compression.

    Extends the base word graph with event-based edge weighting and a
    pruning BFS search strategy. Input sentences are expected in
    "word/POS/weight" format where weight reflects the event distance
    (e.g., "the/DT/1.54 cat/NN/1.82").

    The edge weight formula incorporates event weights instead of raw
    word frequency, and the compression search combines path score with
    language model fluency score.

    Reference:
        Wang, Z. (2015). Coati: Event-guided multi-sentence compression.
    """

    def __init__(
        self,
        sentence_list: List[str],
        grammar_scorer: Optional[GrammarScorer] = None,
        nb_words: int = 8,
        lang: str = "en",
        punct_tag: str = "PUNCT",
        pos_separator: str = "/",
    ):
        """Initialize the CoatiWordGraph.

        Args:
            sentence_list: List of "word/POS/weight" formatted sentence strings.
            grammar_scorer: Optional GrammarScorer for fluency evaluation.
                Required for event_guided_multi_compress.
            nb_words: Minimum number of words in a valid compression (default: 8).
            lang: Language code for stopword selection (default: "en").
            punct_tag: POS tag for punctuation marks (default: "PUNCT").
            pos_separator: Character separating word, POS, and weight (default: "/").
        """
        self.grammar_scorer: Optional[GrammarScorer] = grammar_scorer
        self.term_weight: Dict[str, float] = {}
        super().__init__(sentence_list, nb_words, lang, punct_tag, pos_separator)

    def pre_process_sentences(self) -> None:
        """Parse each sentence from "word/POS/weight" format into tuples.

        Extracts the event-based weight for each word and stores it as
        a float. Adds start and end markers with weight 1.0.
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
                m = re.match(
                    r"^(.+)"
                    + pos_separator_re
                    + r"(.+)"
                    + pos_separator_re
                    + r"(\d+(\.\d+)*)$",
                    w,
                )
                token, pos, weight = m.group(1), m.group(2), m.group(3)
                container.append((token.lower(), pos, float(weight)))

            container.append((self.stop, self.stop, 1.0))
            self.sentence[i] = container

    def compute_statistics(self) -> None:
        """Compute term frequency and cumulative event weight for each word/POS node."""
        terms: Dict[str, List[int]] = {}
        weights: Dict[str, List[float]] = {}

        for i in range(self.length):
            for token, pos, weight in self.sentence[i]:
                node = token.lower() + self.sep + pos

                if node not in terms:
                    terms[node] = [i]
                else:
                    terms[node].append(i)

                if weight == self.start or weight == self.stop:
                    continue

                if node not in weights:
                    weights[node] = [weight]
                else:
                    weights[node].append(weight)

        for key in terms:
            self.term_freq[key] = len(terms[key])

        for key in weights:
            self.term_weight[key] = sum(weights[key])

    def _add_edges(self, sent_idx: int, mapping: list) -> None:
        """Add edges between all valid predecessor-successor pairs.

        Unlike the Takahe method which only adds consecutive edges, this
        method adds edges between all pairs of mapped words where the
        predecessor precedes the successor, ensuring no cycles are created.

        Args:
            sent_idx: Index of the current sentence.
            mapping: Mapping list from word positions to graph nodes.
        """
        for pre in range(len(mapping) - 1):
            for pos in range(pre + 1, len(mapping)):
                if not nx.has_path(self.graph, mapping[pos], mapping[pre]):
                    self.graph.add_edge(mapping[pre], mapping[pos])

    def _compute_edge_weights(self) -> None:
        """Compute edge weights using event-based weight formula.

        Edge weight = ((w1 + w2) / sum_diff) / (w1 * w2),
        where w1 and w2 are the cumulative event weights of the connected nodes.
        """
        for node1, node2 in self.graph.edges():
            edge_weight = self._get_edge_weight(node1, node2)
            self.graph.add_edge(node1, node2, weight=edge_weight)

    def _get_edge_weight(self, node1: tuple, node2: tuple) -> float:
        """Compute the weight of an edge using event-based node weights.

        Args:
            node1: The source node tuple.
            node2: The target node tuple.

        Returns:
            The computed edge weight as a float, or 0.0 if either node
            has zero weight.
        """
        key1 = node1[0]
        weight1 = self.term_weight.get(key1, 0.0)
        if weight1 == 0:
            return 0.0

        key2 = node2[0]
        weight2 = self.term_weight.get(key2, 0.0)
        if weight2 == 0:
            return 0.0

        diff = self._compute_diff_for_edge(node1, node2)
        sum_diff = sum(diff)
        if sum_diff == 0:
            return 0.0

        return ((weight1 + weight2) / sum_diff) / (weight1 * weight2)

    def get_compression(self, nb_candidates: int = 50) -> List[Tuple[float, list]]:
        """Extract compression candidates using k-shortest paths.

        Args:
            nb_candidates: Maximum number of compression candidates to return.

        Returns:
            A sorted list of (score, path) tuples.
        """
        return self._get_compression_from_ksp(nb_candidates)

    def event_guided_multi_compress(
        self,
        lambd: float,
        max_neighbors: int,
        queue_size: int,
        sentence_count: int,
    ) -> List[Tuple[float, str]]:
        """Perform event-guided multi-sentence compression.

        Uses a pruning BFS to explore paths through the word graph,
        combining path score with language model fluency score to rank
        candidate compressions.

        Args:
            lambd: Weight balancing path score and fluency score.
            max_neighbors: Maximum number of successor neighbors to explore
                at each BFS step.
            queue_size: Maximum size of the BFS queue (controls memory usage).
            sentence_count: Number of top compressions to return.

        Returns:
            A list of (combined_score, sentence_string) tuples sorted by
            score in descending order.

        Raises:
            ValueError: If grammar_scorer was not provided at initialization.
        """
        if self.grammar_scorer is None:
            raise ValueError("grammar_scorer is required for event_guided_multi_compress")

        sentences = self._pruning_bfs(lambd, max_neighbors, queue_size)

        for i in range(len(sentences)):
            sentence = sentences[i]
            path_weight = 0.0
            str_sentence = ""

            for j in range(1, len(sentence) - 2):
                path_weight += self.graph.get_edge_data(
                    sentence[j], sentence[j + 1]
                )["weight"]
                str_sentence += sentence[j][0].split(self.sep)[0] + " "

            fluency_weight = self.grammar_scorer.cal_fluency(
                str_sentence
            ) / len(re.split(r"\s+", str_sentence))

            sentences[i] = (
                len(sentence) / path_weight + lambd * fluency_weight,
                str_sentence.strip(),
            )

        sentences.sort(key=lambda x: x[0], reverse=True)
        return sentences[:sentence_count]

    def _pruning_bfs(
        self, lambd: float, max_neighbors: int, queue_size: int
    ) -> List[list]:
        """Perform a pruning breadth-first search through the word graph.

        At each step, evaluates successor nodes using a combined score of
        inverse edge weight and language model fluency. Only the top
        max_neighbors candidates are enqueued, and the queue is bounded
        by queue_size.

        Args:
            lambd: Weight for the fluency component of the scoring function.
            max_neighbors: Maximum number of top-scoring successors to enqueue.
            queue_size: Maximum queue capacity.

        Returns:
            A list of valid paths from start to end with at least 8 words.
        """
        results: List[list] = []
        start = (self.start + self.sep + self.start, 0)
        stop = (self.stop + self.sep + self.stop, 0)

        queue: Queue = Queue(queue_size)
        queue.put([start])

        while not queue.empty():
            phrase = queue.get()
            node = phrase[-1]

            if node == stop:
                if len(phrase) >= 8:
                    results.append(phrase)
                continue

            str_phrase = ""
            for nodeflag, _num in phrase:
                str_phrase += nodeflag.split(self.sep)[0] + " "

            logging.info(
                "results size[%d] queue size[%d] phrase[%s]",
                len(results),
                queue.qsize(),
                str_phrase,
            )

            pos_neighbors = list(self.graph.neighbors(node))
            neighbor_weight: Dict[tuple, float] = {}

            for pos_neighbor in pos_neighbors:
                edge_weight = self.graph.get_edge_data(node, pos_neighbor)["weight"]
                if edge_weight == 0:
                    continue

                fluency_weight = self.grammar_scorer.cal_fluency(
                    str_phrase + pos_neighbor[0].split(self.sep)[0]
                )

                general_score = (
                    1 / edge_weight
                    + lambd * fluency_weight / (len(re.split(r"\s+", str_phrase)) + 1)
                )

                logging.info(
                    "lambd[%f] general score[%f] edge weight[%f] fluency weight[%f]",
                    lambd,
                    general_score,
                    edge_weight,
                    fluency_weight,
                )

                neighbor_weight[pos_neighbor] = general_score

            sorted_neighbors = sorted(
                neighbor_weight.items(),
                key=lambda item: item[1],
                reverse=True,
            )

            for i in range(min(max_neighbors, len(sorted_neighbors))):
                if queue.full():
                    break

                new_phrase = phrase + [sorted_neighbors[i][0]]
                queue.put(new_phrase)

        return results
