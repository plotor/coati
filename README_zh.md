## Coati

Coati 是一个多语句压缩算法的实现项目，基于词图对描述相同和相似主题的多个语句实施压缩，最终生成承载主题主干信息的语句。

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

### 参考文献

1. [Multi-sentence compression: finding shortest paths in word graphs](http://dl.acm.org/citation.cfm?id=1873818)
2. [Multiple alternative sentence compressions for automatic text summarization](http://www.umiacs.umd.edu/~dmzajic/papers/DUC2007.pdf)

### 算法

论文提出了一种基于词图和词频，并结合 K 最短路径算法为描述相同主题的多个语句生成摘要语句的高效方法。构造词图的过程如下：

1. 添加第一条语句，以每个词作为图的一个结点，附加一个 "start" 结点和一个 "end" 结点初始化词图。
2. 依次添加剩余的语句到词图：

> - 如果当前句子中的词在图中已有映射结点，即图中某个词与当前词具有相同的词形和词性，那么直接将该词映射到该结点；
> - 如果当前句子中的词在图中没有相应的映射结点，则创建一个新的结点。

![image](https://github.com/procyon-lotor/procyon-lotor.github.io/blob/master/images/2017/wordgraph.png?raw=false)

在计算词图边的权重方面，原方法仅考虑词频因素，仅考虑词频因素往往会引入较多的噪声，导致生成的句子丢失主干信息。本实现考虑事件（可以简单理解为 "主-谓-宾" 词组）因素来计算词图边的权重。首先利用事件聚类的结果，将当前事件与聚类中心的距离作为事件的权重，然后利用下面的公式计算词图中每个词的权重：

![image](https://github.com/procyon-lotor/procyon-lotor.github.io/blob/master/images/2017/msc_1.png?raw=false)

其中，dis(i, e) 表示词 i 与事件 e 的余弦距离，w(e) 表示事件的权重，size(E) 表示当前主题的大小。如果某个结点对应的词在一个或多个句子中出现多次，则将其权重进行累加。在边的权重 w(i, j) 计算上，考虑当前边所连接的词 i 和 j 的权重，按照如下公式进行计算：

![image](https://github.com/procyon-lotor/procyon-lotor.github.io/blob/master/images/2017/msc_2.png?raw=false)

其中 pos(s, i) 表示词 i 在句子 s 中的水平位移。

语句的路径得分由路径上所有边的权重之和除以路径长度计算得到。原压缩方法中将这 K 条语句作为压缩的最终输出，本实现引入了 Tri-gram 语言模型为每个压缩候选语句进行打分，以此反映语句的语言流畅度。

[takahe](https://github.com/boudinfl/takahe) 对原论文中的方法进行了实现，本项目在 takahe 的基础上进行了改进，对 takahe 作者表示感谢。

### 安装

```bash
pip install -e .
```

开发模式安装（包含 pytest）：

```bash
pip install -e ".[dev]"
```

### 使用

**Takahe 压缩**（基于词频的原始算法）：

```bash
coati takahe --input ./data --output ./result
coati takahe --input ./data --output ./result --keyphrase
```

**事件指导压缩**（基于事件权重 + 语言模型的改进算法）：

```bash
coati event --input ./data --output ./result --ngram-model ./model.lm --lambda 1.0
coati event --input ./data --output ./result --ngram-model ./model.lm --lambda 1.0 --keyphrase
```

**兼容旧配置文件模式**：

```bash
coati config setting.conf
```

### Python API

```python
from coati import TakaheWordGraph, CoatiWordGraph, KeyphraseReranker, GrammarScorer

# Takahe 压缩
wg = TakaheWordGraph(sentences, nb_words=8, lang="en", punct_tag="PUNCT")
candidates = wg.get_compression(50)

# 关键短语重排序
reranker = KeyphraseReranker(sentences, candidates, lang="en")
reranked = reranker.rerank_nbest_compressions()

# 事件指导压缩
scorer = GrammarScorer("path/to/ngram_model.lm")
wg = CoatiWordGraph(weighted_sentences, grammar_scorer=scorer)
results = wg.event_guided_multi_compress(lambd=1.0, max_neighbors=6, queue_size=1024, sentence_count=50)
```

### 项目结构

```
coati/
├── pyproject.toml
├── coati/
│   ├── __init__.py
│   ├── cli.py
│   ├── graph/
│   │   ├── base.py             # BaseWordGraph 基类
│   │   ├── takahe_graph.py     # TakaheWordGraph
│   │   ├── coati_graph.py      # CoatiWordGraph
│   │   └── reranker.py         # KeyphraseReranker
│   ├── scorer/
│   │   └── grammar.py          # GrammarScorer
│   ├── utils/
│   │   └── logger.py           # setup_logging
│   └── resources/
│       ├── stopwords.en.dat
│       └── stopwords.fr.dat
└── tests/
    ├── test_graph.py
    ├── test_scorer.py
    └── test_reranker.py
```

### 测试

```bash
python -m pytest tests/ -v
```

### 许可证

[MIT](LICENSE)
