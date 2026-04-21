import pytest

from coati.graph.takahe_graph import TakaheWordGraph
from coati.graph.coati_graph import CoatiWordGraph
from coati.graph.reranker import KeyphraseReranker


SENTENCES = [
    "Turkish/JJ warplanes/NNS have/VBP shot/VBN down/RP a/DT Russian/JJ military/JJ aircraft/NN on/IN the/DT border/NN with/IN Syria/NNP ./PUNCT",
    "Turkey/NNP says/VBZ it/PRP has/VBZ shot/VBN down/RP a/DT Russian/JJ made/VBN warplane/NN on/IN the/DT Syrian/JJ border/NN for/IN violating/VBG Turkish/JJ airspace/NN ./PUNCT",
    "A/DT Turkish/JJ Air/NNP Force/NNP F16/NN fighter/NN jet/NN shot/VBD down/RP a/DT Russian/JJ Sukhoi/NNP Su24M/NN bomber/NN aircraft/NN near/IN the/DT Syria/NNP border/NN on/IN 24/CD November/NNP 2015/CD ./PUNCT",
    "A/DT Russian/JJ warplane/NN has/VBZ crashed/VBN in/IN Syria/NNP near/IN the/DT Turkish/JJ border/NN on/IN 24/CD November/NNP 2015/CD according/VBG to/TO local/JJ reports/NNS ./PUNCT",
    "Turkey/NNP apparently/RB shot/VBD down/RP a/DT Russian/JJ bomber/NN which/WDT they/PRP say/VBP was/VBD in/IN their/PRP$ air/NN space/NN this/DT morning/NN ./PUNCT",
]

WEIGHTED_SENTENCES = [
    "Turkish/JJ/2.335849 warplanes/NNS/1.880854 have/VBP/1.533773 shot/VBN/1.55875 down/RP/1.582315 a/DT/1.541407 Russian/JJ/1.690761 military/JJ/1.732811 aircraft/NN/1.763314 on/IN/1.519171 the/DT/1.580312 border/NN/1.921725 with/IN/1.599827 Syria/NNP/2.172298 ./PUNCT/1.492299",
    "Turkey/NNP/2.473707 says/VBZ/1.54218 it/PRP/1.621644 has/VBZ/1.590734 shot/VBN/1.55875 down/RP/1.582315 a/DT/1.541407 Russian-made/JJ/1.70222 warplane/NN/1.675197 on/IN/1.519171 the/DT/1.580312 Syrian/JJ/2.037667 border/NN/1.921725 for/IN/1.544959 violating/VBG/1.523598 Turkish/JJ/2.335849 airspace/NN/1.824718 ./PUNCT/1.492299",
    "A/DT/1.541407 Turkish/JJ/2.335849 Air/NNP/1.761123 Force/NNP/1.651825 F-16/NNP/1.88025 fighter/NN/1.766968 jet/NN/1.719154 shot/VBD/1.55875 down/RP/1.582315 a/DT/1.541407 Russian/JJ/1.690761 Sukhoi/NNP/1.717303 Su-24M/NN/1.668692 bomber/NN/1.76194 aircraft/NN/1.763314 near/IN/1.548048 the/DT/1.580312 Syria/NNP/2.172298 border/NN/1.921725 on/IN/1.519171 24/CD/1.596672 November/NNP/1.541376 2015/CD/1.6215 ./PUNCT/1.492299",
    "A/DT/1.541407 Russian/JJ/1.690761 warplane/NN/1.675197 has/VBZ/1.590734 crashed/VBN/1.63605 in/IN/1.570426 Syria/NNP/2.172298 near/IN/1.548048 the/DT/1.580312 Turkish/JJ/2.335849 border/NN/1.921725 on/IN/1.519171 24/CD/1.596672 November/NNP/1.541376 ,/PUNCT/1.550442 according/VBG/1.507866 to/TO/1.593164 local/JJ/1.559173 reports/NNS/1.598159 ./PUNCT/1.492299",
    "Turkey/NNP/2.473707 apparently/RB/1.455763 shot/VBD/1.55875 down/RP/1.582315 a/DT/1.541407 Russian/JJ/1.690761 bomber/NN/1.76194 which/WDT/1.583918 they/PRP/1.549086 say/VBP/1.536582 was/VBD/1.508024 in/IN/1.570426 their/PRP$/1.48893 air/NN/1.761123 space/NN/1.53048 this/DT/1.557884 morning/NN/1.508587 ./PUNCT/1.492299",
]


class TestTakaheWordGraph:

    def test_construction(self):
        wg = TakaheWordGraph(SENTENCES)
        assert wg.length == 5
        assert wg.graph is not None
        assert len(wg.graph.nodes()) > 0
        assert len(wg.graph.edges()) > 0

    def test_compression(self):
        wg = TakaheWordGraph(SENTENCES)
        candidates = wg.get_compression(10)
        assert len(candidates) > 0
        for score, path in candidates:
            assert isinstance(score, float)
            assert len(path) > 0

    def test_pre_process_sentences(self):
        wg = TakaheWordGraph(SENTENCES)
        for sentence in wg.sentence:
            assert isinstance(sentence, list)
            for item in sentence:
                assert len(item) == 3

    def test_compute_statistics(self):
        wg = TakaheWordGraph(SENTENCES)
        assert len(wg.term_freq) > 0

    def test_stopwords_loaded(self):
        wg = TakaheWordGraph(SENTENCES)
        assert len(wg.stopwords) > 0

    def test_ambiguous_nodes(self):
        wg = TakaheWordGraph(SENTENCES)
        node = wg.start + wg.sep + wg.start
        k = wg.ambiguous_nodes(node)
        assert k >= 1


class TestCoatiWordGraph:

    def test_construction(self):
        wg = CoatiWordGraph(WEIGHTED_SENTENCES)
        assert wg.length == 5
        assert wg.graph is not None
        assert len(wg.term_weight) > 0

    def test_compression(self):
        wg = CoatiWordGraph(WEIGHTED_SENTENCES)
        candidates = wg.get_compression(10)
        assert len(candidates) > 0
        for score, path in candidates:
            assert isinstance(score, float)
            assert len(path) > 0

    def test_pre_process_sentences(self):
        wg = CoatiWordGraph(WEIGHTED_SENTENCES)
        for sentence in wg.sentence:
            assert isinstance(sentence, list)
            for item in sentence:
                assert len(item) == 3
                assert isinstance(item[2], float)

    def test_compute_statistics(self):
        wg = CoatiWordGraph(WEIGHTED_SENTENCES)
        assert len(wg.term_freq) > 0
        assert len(wg.term_weight) > 0

    def test_event_guided_requires_scorer(self):
        wg = CoatiWordGraph(WEIGHTED_SENTENCES)
        with pytest.raises(ValueError, match="grammar_scorer is required"):
            wg.event_guided_multi_compress(1.0, 6, 1024, 50)


class TestKeyphraseReranker:

    def test_reranking(self):
        wg = TakaheWordGraph(SENTENCES)
        candidates = wg.get_compression(10)
        assert len(candidates) > 0

        reranker = KeyphraseReranker(SENTENCES, candidates, lang="en")
        reranked = reranker.rerank_nbest_compressions()
        assert len(reranked) > 0
        for score, path in reranked:
            assert isinstance(score, float)

    def test_keyphrase_candidates_generated(self):
        wg = TakaheWordGraph(SENTENCES)
        candidates = wg.get_compression(10)
        reranker = KeyphraseReranker(SENTENCES, candidates, lang="en")
        assert len(reranker.keyphrase_candidates) >= 0
