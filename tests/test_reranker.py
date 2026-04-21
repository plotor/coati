from coati.graph.takahe_graph import TakaheWordGraph
from coati.graph.reranker import KeyphraseReranker


SENTENCES = [
    "Turkish/JJ warplanes/NNS have/VBP shot/VBN down/RP a/DT Russian/JJ military/JJ aircraft/NN on/IN the/DT border/NN with/IN Syria/NNP ./PUNCT",
    "Turkey/NNP says/VBZ it/PRP has/VBZ shot/VBN down/RP a/DT Russian/JJ made/VBN warplane/NN on/IN the/DT Syrian/JJ border/NN for/IN violating/VBG Turkish/JJ airspace/NN ./PUNCT",
    "A/DT Turkish/JJ Air/NNP Force/NNP F16/NN fighter/NN jet/NN shot/VBD down/RP a/DT Russian/JJ Sukhoi/NNP Su24M/NN bomber/NN aircraft/NN near/IN the/DT Syria/NNP border/NN on/IN 24/CD November/NNP 2015/CD ./PUNCT",
]


class TestKeyphraseReranker:

    def test_rerank_produces_results(self):
        wg = TakaheWordGraph(SENTENCES)
        candidates = wg.get_compression(10)
        reranker = KeyphraseReranker(SENTENCES, candidates, lang="en")
        reranked = reranker.rerank_nbest_compressions()
        assert len(reranked) > 0

    def test_word_scores_computed(self):
        wg = TakaheWordGraph(SENTENCES)
        candidates = wg.get_compression(10)
        reranker = KeyphraseReranker(SENTENCES, candidates, lang="en")
        assert len(reranker.word_scores) > 0

    def test_french_language(self):
        wg = TakaheWordGraph(SENTENCES)
        candidates = wg.get_compression(10)
        reranker = KeyphraseReranker(SENTENCES, candidates, lang="fr")
        assert "NPP" in reranker.syntactic_filter
