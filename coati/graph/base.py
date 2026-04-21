import bisect
import codecs
import os
import re
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx


class BaseWordGraph(ABC):
    """Abstract base class for word graph construction and multi-sentence compression.

    A word graph is a directed graph built from a set of POS-tagged sentences.
    Words that share the same form and POS tag are merged into shared nodes,
    while ambiguous mappings are resolved using context overlap. Subclasses
    must implement sentence preprocessing, statistics computation, edge
    addition, edge weight computation, and the compression extraction method.

    Attributes:
        sentence: List of preprocessed sentences (each a list of tuples).
        length: Number of input sentences.
        nb_words: Minimum number of words required in a compression.
        graph: The directed word graph (networkx.DiGraph).
        start: Special start token.
        stop: Special end token.
        sep: Separator used between word and POS in node identifiers.
        term_freq: Frequency of each term in the graph.
        verbs: Set of POS tags considered as verbs.
    """

    def __init__(
        self,
        sentence_list: List[str],
        nb_words: int = 8,
        lang: str = "en",
        punct_tag: str = "PUNCT",
        pos_separator: str = "/",
    ):
        """Initialize the word graph.

        Args:
            sentence_list: List of POS-tagged sentence strings.
            nb_words: Minimum number of words in a valid compression (default: 8).
            lang: Language code for stopword selection, "en" or "fr" (default: "en").
            punct_tag: POS tag used for punctuation marks (default: "PUNCT").
            pos_separator: Character separating word and POS tag (default: "/").
        """
        self.sentence: List[list] = list(sentence_list)
        self.length: int = len(sentence_list)
        self.nb_words: int = nb_words
        self.resources: str = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "resources"
        )
        self.stopword_path: str = os.path.join(
            self.resources, f"stopwords.{lang}.dat"
        )
        self.stopwords: Set[str] = self.load_stopwords(self.stopword_path)
        self.punct_tag: str = punct_tag
        self.pos_separator: str = pos_separator

        self.graph: nx.DiGraph = nx.DiGraph()
        self.start: str = "-start-"
        self.stop: str = "-end-"
        self.sep: str = "/-/"
        self.term_freq: Dict[str, int] = {}
        self.verbs: Set[str] = {
            "VB", "VBD", "VBP", "VBZ", "VH", "VHD", "VHP", "VBZ",
            "VV", "VVD", "VVP", "VVZ",
        }
        if lang == "fr":
            self.verbs = {"V", "VPP", "VINF"}

        self.pre_process_sentences()
        self.compute_statistics()
        self.build_graph()

    @abstractmethod
    def pre_process_sentences(self) -> None:
        """Pre-process the raw sentence strings into structured tuples.

        Subclasses must parse each sentence string into a list of
        (token, pos, weight) tuples, adding start and end markers.
        """

    @abstractmethod
    def compute_statistics(self) -> None:
        """Compute term frequency and (optionally) term weight statistics.

        Subclasses must populate self.term_freq and may also populate
        self.term_weight if event-based weighting is used.
        """

    @abstractmethod
    def get_compression(self, nb_candidates: int = 50) -> List[Tuple[float, list]]:
        """Extract compression candidates from the word graph.

        Args:
            nb_candidates: Maximum number of compression candidates to return.

        Returns:
            A list of (score, path) tuples sorted by score.
        """

    def build_graph(self) -> None:
        """Construct the directed word graph from all input sentences.

        Iterates over each sentence and performs four mapping phases:
        1. Non-stopwords with unambiguous or no candidate nodes.
        2. Non-stopwords with multiple candidate nodes (context-based resolution).
        3. Stopwords (context-based resolution using non-stopword neighbors).
        4. Punctuation marks (context-based resolution with stricter overlap).

        After all sentences are mapped, edges are added and weighted.
        """
        for i in range(self.length):
            sentence_len = len(self.sentence[i])
            mapping: list = [0] * sentence_len

            self._map_non_stopwords_unambiguous(i, sentence_len, mapping)
            self._map_non_stopwords_ambiguous(i, sentence_len, mapping)
            self._map_stopwords(i, sentence_len, mapping)
            self._map_punctuation(i, sentence_len, mapping)
            self._add_edges(i, mapping)

        self._compute_edge_weights()

    def _map_non_stopwords_unambiguous(
        self, sent_idx: int, sentence_len: int, mapping: list
    ) -> None:
        """Map non-stopwords that have zero or one candidate node in the graph.

        Words with no existing node create a new node. Words with exactly one
        existing node are mapped to it if they come from a different sentence,
        otherwise a new node is created for the redundant occurrence.

        Args:
            sent_idx: Index of the current sentence.
            sentence_len: Length of the current sentence.
            mapping: Mutable mapping list from word positions to graph nodes.
        """
        for j in range(sentence_len):
            token, pos, _weight = self.sentence[sent_idx][j]

            if token in self.stopwords or re.search(r"(?u)^\W$", token):
                continue

            node = token.lower() + self.sep + pos
            k = self.ambiguous_nodes(node)

            if k == 0:
                self.graph.add_node(
                    (node, 0), info=[(sent_idx, j)], label=token.lower()
                )
                mapping[j] = (node, 0)
            elif k == 1:
                ids = [sid for sid, _ in self.graph.nodes[(node, 0)]["info"]]
                if sent_idx not in ids:
                    self.graph.nodes[(node, 0)]["info"].append((sent_idx, j))
                    mapping[j] = (node, 0)
                else:
                    self.graph.add_node(
                        (node, 1), info=[(sent_idx, j)], label=token.lower()
                    )
                    mapping[j] = (node, 1)

    def _map_non_stopwords_ambiguous(
        self, sent_idx: int, sentence_len: int, mapping: list
    ) -> None:
        """Map non-stopwords that have multiple candidate nodes in the graph.

        Selects the candidate with the highest context overlap (or highest
        frequency as a tiebreaker) while avoiding cycles (same-sentence
        mappings). If no suitable candidate is found, a new node is created.

        Args:
            sent_idx: Index of the current sentence.
            sentence_len: Length of the current sentence.
            mapping: Mutable mapping list from word positions to graph nodes.
        """
        for j in range(sentence_len):
            token, pos, _weight = self.sentence[sent_idx][j]

            if token in self.stopwords or re.search(r"(?u)^\W$", token):
                continue

            if mapping[j] != 0:
                continue

            node = token.lower() + self.sep + pos

            prev_token, prev_pos, _ = self.sentence[sent_idx][j - 1]
            next_token, next_pos, _ = self.sentence[sent_idx][j + 1]
            prev_node = prev_token.lower() + self.sep + prev_pos
            next_node = next_token.lower() + self.sep + next_pos

            k = self.ambiguous_nodes(node)

            ambinode_overlap: List[int] = []
            ambinode_frequency: List[int] = []

            for l in range(k):
                l_context = self.get_directed_context(node, l, "left")
                r_context = self.get_directed_context(node, l, "right")

                val = l_context.count(prev_node) + r_context.count(next_node)
                ambinode_overlap.append(val)
                ambinode_frequency.append(len(self.graph.nodes[(node, l)]["info"]))

            found = False
            selected = 0
            while not found:
                selected = self.max_index(ambinode_overlap)
                if ambinode_overlap[selected] == 0:
                    selected = self.max_index(ambinode_frequency)

                ids = [sid for sid, _ in self.graph.nodes[(node, selected)]["info"]]

                if sent_idx not in ids:
                    found = True
                    break
                else:
                    del ambinode_overlap[selected]
                    del ambinode_frequency[selected]

                if len(ambinode_overlap) == 0:
                    break

            if found:
                self.graph.nodes[(node, selected)]["info"].append((sent_idx, j))
                mapping[j] = (node, selected)
            else:
                self.graph.add_node(
                    (node, k), info=[(sent_idx, j)], label=token.lower()
                )
                mapping[j] = (node, k)

    def _map_stopwords(
        self, sent_idx: int, sentence_len: int, mapping: list
    ) -> None:
        """Map stopwords to existing or new nodes in the graph.

        Stopwords are only mapped to existing nodes if there is at least one
        non-stopword neighbor overlap in the context. Otherwise a new node
        is created.

        Args:
            sent_idx: Index of the current sentence.
            sentence_len: Length of the current sentence.
            mapping: Mutable mapping list from word positions to graph nodes.
        """
        for j in range(sentence_len):
            token, pos, _weight = self.sentence[sent_idx][j]

            if token not in self.stopwords:
                continue

            node = token.lower() + self.sep + pos
            k = self.ambiguous_nodes(node)

            if k == 0:
                self.graph.add_node(
                    (node, 0), info=[(sent_idx, j)], label=token.lower()
                )
                mapping[j] = (node, 0)
            else:
                prev_token, prev_pos, _ = self.sentence[sent_idx][j - 1]
                next_token, next_pos, _ = self.sentence[sent_idx][j + 1]
                prev_node = prev_token.lower() + self.sep + prev_pos
                next_node = next_token.lower() + self.sep + next_pos

                ambinode_overlap: List[int] = []

                for l in range(k):
                    l_context = self.get_directed_context(node, l, "left", True)
                    r_context = self.get_directed_context(node, l, "right", True)

                    val = l_context.count(prev_node) + r_context.count(next_node)
                    ambinode_overlap.append(val)

                selected = self.max_index(ambinode_overlap)

                ids = [sid for sid, _ in self.graph.nodes[(node, selected)]["info"]]

                if sent_idx not in ids and ambinode_overlap[selected] > 0:
                    self.graph.nodes[(node, selected)]["info"].append((sent_idx, j))
                    mapping[j] = (node, selected)
                else:
                    self.graph.add_node(
                        (node, k), info=[(sent_idx, j)], label=token.lower()
                    )
                    mapping[j] = (node, k)

    def _map_punctuation(
        self, sent_idx: int, sentence_len: int, mapping: list
    ) -> None:
        """Map punctuation marks to existing or new nodes in the graph.

        Punctuation is only merged with an existing node if the context
        overlap is greater than 1 (stricter than stopwords). Otherwise
        a new node is created.

        Args:
            sent_idx: Index of the current sentence.
            sentence_len: Length of the current sentence.
            mapping: Mutable mapping list from word positions to graph nodes.
        """
        for j in range(sentence_len):
            token, pos, _weight = self.sentence[sent_idx][j]

            if not re.search(r"(?u)^\W$", token):
                continue

            node = token.lower() + self.sep + pos
            k = self.ambiguous_nodes(node)

            if k == 0:
                self.graph.add_node(
                    (node, 0), info=[(sent_idx, j)], label=token.lower()
                )
                mapping[j] = (node, 0)
            else:
                prev_token, prev_pos, _ = self.sentence[sent_idx][j - 1]
                next_token, next_pos, _ = self.sentence[sent_idx][j + 1]
                prev_node = prev_token.lower() + self.sep + prev_pos
                next_node = next_token.lower() + self.sep + next_pos

                ambinode_overlap: List[int] = []

                for l in range(k):
                    l_context = self.get_directed_context(node, l, "left")
                    r_context = self.get_directed_context(node, l, "right")

                    val = l_context.count(prev_node) + r_context.count(next_node)
                    ambinode_overlap.append(val)

                selected = self.max_index(ambinode_overlap)

                ids = [sid for sid, _ in self.graph.nodes[(node, selected)]["info"]]

                if sent_idx not in ids and ambinode_overlap[selected] > 1:
                    self.graph.nodes[(node, selected)]["info"].append((sent_idx, j))
                    mapping[j] = (node, selected)
                else:
                    self.graph.add_node(
                        (node, k), info=[(sent_idx, j)], label=token.lower()
                    )
                    mapping[j] = (node, k)

    @abstractmethod
    def _add_edges(self, sent_idx: int, mapping: list) -> None:
        """Add directed edges between mapped word nodes for a sentence.

        Args:
            sent_idx: Index of the current sentence.
            mapping: Mapping list from word positions to graph nodes.
        """

    @abstractmethod
    def _compute_edge_weights(self) -> None:
        """Compute and assign weights to all edges in the graph."""

    def ambiguous_nodes(self, node: str) -> int:
        """Return the number of candidate nodes for a given word/POS identifier.

        Args:
            node: A word/POS identifier string (e.g., "the/-/DET").

        Returns:
            The count of existing nodes with identifiers (node, 0), (node, 1), etc.
        """
        k = 0
        while self.graph.has_node((node, k)):
            k += 1
        return k

    def get_directed_context(
        self,
        node: str,
        k: int,
        direction: str = "all",
        non_stop: bool = False,
    ) -> List[str]:
        """Return the directed context (neighboring node identifiers) of a graph node.

        Args:
            node: A word/POS identifier string.
            k: The node disambiguation index.
            direction: "left" for preceding context, "right" for following
                context, or "all" for both (default: "all").
            non_stop: If True, exclude stopwords from the context (default: False).

        Returns:
            A list of word/POS identifier strings for the neighboring nodes.
        """
        l_context: List[str] = []
        r_context: List[str] = []

        for sid, off in self.graph.nodes[(node, k)]["info"]:
            prev = (
                self.sentence[sid][off - 1][0].lower()
                + self.sep
                + self.sentence[sid][off - 1][1]
            )
            next_w = (
                self.sentence[sid][off + 1][0].lower()
                + self.sep
                + self.sentence[sid][off + 1][1]
            )

            if non_stop:
                if self.sentence[sid][off - 1][0] not in self.stopwords:
                    l_context.append(prev)
                if self.sentence[sid][off + 1][0] not in self.stopwords:
                    r_context.append(next_w)
            else:
                l_context.append(prev)
                r_context.append(next_w)

        if direction == "left":
            return l_context
        elif direction == "right":
            return r_context
        else:
            l_context.extend(r_context)
            return l_context

    def load_stopwords(self, path: str) -> Set[str]:
        """Load a stopword list from a file.

        Lines starting with '#' and empty lines are ignored. All words
        are converted to lowercase.

        Args:
            path: Path to the stopword list file.

        Returns:
            A set of lowercase stopword strings.
        """
        stopwords: Set[str] = set()
        for line in codecs.open(path, "r", "utf-8"):
            if not re.search("^#", line) and len(line.strip()) > 0:
                stopwords.add(line.strip().lower())
        return stopwords

    @staticmethod
    def max_index(lst: list) -> Optional[int]:
        """Return the index of the maximum value in a list.

        Args:
            lst: A list of comparable values.

        Returns:
            The index of the maximum value, or None if the list is empty.
        """
        if len(lst) <= 0:
            return None
        elif len(lst) == 1:
            return 0
        max_val = lst[0]
        max_ind = 0
        for z in range(1, len(lst)):
            if lst[z] > max_val:
                max_val = lst[z]
                max_ind = z
        return max_ind

    def write_dot(self, dotfile: str) -> None:
        """Write the word graph in Graphviz DOT format.

        Args:
            dotfile: Path to the output DOT file.
        """
        nx.write_dot(self.graph, dotfile)

    def k_shortest_paths(
        self, start: tuple, end: tuple, k: int = 10
    ) -> List[Tuple[list, float]]:
        """Find the k shortest paths from start to end in the word graph.

        Uses a modified Dijkstra-like algorithm with path deduplication.
        Valid paths must contain at least one verb, meet the minimum word
        count, have balanced parentheses and quotation marks, and not be
        duplicates.

        Args:
            start: The starting node tuple.
            end: The ending node tuple.
            k: Maximum number of shortest paths to find (default: 10).

        Returns:
            A list of (path, cumulative_weight) tuples.
        """
        kshortestpaths: List[Tuple[list, float]] = []
        ordered_x: list = [(0, start, 0)]
        paths: Dict[tuple, list] = {(0, start, 0): [start]}
        visited: Dict[tuple, int] = {start: 0}
        sentence_container: Dict[str, int] = {}

        while len(kshortestpaths) < k and len(ordered_x) > 0:
            shortest = ordered_x.pop(0)
            shortest_path = paths[shortest]
            del paths[shortest]

            for node in self.graph.neighbors(shortest[1]):
                if node in shortest_path:
                    continue

                w = shortest[0] + self.graph[shortest[1]][node]["weight"]

                if node == end:
                    nb_verbs = 0
                    length = 0
                    paired_parentheses = 0
                    quotation_mark_number = 0
                    raw_sentence = ""

                    for i in range(len(shortest_path) - 1):
                        word, tag = shortest_path[i][0].split(self.sep)
                        if tag in self.verbs:
                            nb_verbs += 1
                        if not re.search(r"(?u)^\W$", word):
                            length += 1
                        else:
                            if word == "(":
                                paired_parentheses -= 1
                            elif word == ")":
                                paired_parentheses += 1
                            elif word == '"':
                                quotation_mark_number += 1
                        raw_sentence += word + " "

                    raw_sentence = raw_sentence.strip()

                    if (
                        nb_verbs > 0
                        and length >= self.nb_words
                        and paired_parentheses == 0
                        and (quotation_mark_number % 2) == 0
                        and raw_sentence not in sentence_container
                    ):
                        path = [node]
                        path.extend(shortest_path)
                        path.reverse()
                        weight = float(w)
                        kshortestpaths.append((path, weight))
                        sentence_container[raw_sentence] = 1
                else:
                    if node in visited:
                        visited[node] += 1
                    else:
                        visited[node] = 0
                    node_id = visited[node]

                    bisect.insort(ordered_x, (w, node, node_id))

                    paths[(w, node, node_id)] = [node]
                    paths[(w, node, node_id)].extend(shortest_path)

        return kshortestpaths

    def _get_compression_from_ksp(
        self, nb_candidates: int = 50
    ) -> List[Tuple[float, list]]:
        """Extract compression candidates using k-shortest paths.

        Searches for paths from the start node to the end node and
        converts them into (score, word_list) tuples.

        Args:
            nb_candidates: Maximum number of candidates to return.

        Returns:
            A sorted list of (score, sentence) tuples where sentence
            is a list of (word, pos) tuples.
        """
        self.paths = self.k_shortest_paths(
            (self.start + self.sep + self.start, 0),
            (self.stop + self.sep + self.stop, 0),
            nb_candidates,
        )

        fusions: List[Tuple[float, list]] = []

        if len(self.paths) > 0:
            for i in range(min(nb_candidates, len(self.paths))):
                nodes = self.paths[i][0]
                sentence: list = []

                for j in range(1, len(nodes) - 1):
                    word, tag = nodes[j][0].split(self.sep)
                    sentence.append((word, tag))

                bisect.insort(fusions, (self.paths[i][1], sentence))

        return fusions

    def _compute_diff_for_edge(self, node1: tuple, node2: tuple) -> List[float]:
        """Compute the diff function for edge weight calculation.

        For each sentence s, computes 1 / min_diff(s, i, j) where
        min_diff is the minimum positional distance between words i
        and j in sentence s (only when i precedes j). Returns 0.0
        for sentences where neither word appears or i never precedes j.

        Args:
            node1: The source node tuple.
            node2: The target node tuple.

        Returns:
            A list of diff values, one per input sentence.
        """
        info1 = self.graph.nodes[node1]["info"]
        info2 = self.graph.nodes[node2]["info"]

        diff: List[float] = []

        for s in range(self.length):
            pos_i_in_s: List[int] = []
            pos_j_in_s: List[int] = []

            for sentence_id, pos_in_sentence in info1:
                if sentence_id == s:
                    pos_i_in_s.append(pos_in_sentence)

            for sentence_id, pos_in_sentence in info2:
                if sentence_id == s:
                    pos_j_in_s.append(pos_in_sentence)

            all_diff_pos_i_j: List[float] = []

            for x in range(len(pos_i_in_s)):
                for y in range(len(pos_j_in_s)):
                    diff_i_j = pos_i_in_s[x] - pos_j_in_s[y]
                    if diff_i_j < 0:
                        all_diff_pos_i_j.append(-1.0 * diff_i_j)

            if len(all_diff_pos_i_j) > 0:
                diff.append(1.0 / min(all_diff_pos_i_j))
            else:
                diff.append(0.0)

        return diff
