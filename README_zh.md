## Coati

Coati 是一个多语句压缩算法的实现项目，基于词图对描述相同和相似主题的多个语句实施压缩，最终生成承载主题主干信息语句的过程。具体算法思想可以参考下面两篇论文：

1. [Multi-sentence compression: finding shortest paths in word graphs](http://dl.acm.org/citation.cfm?id=1873818)
2. [Multiple alternative sentence compressions for automatic text summarization](http://www.umiacs.umd.edu/~dmzajic/papers/DUC2007.pdf)

论文提出了一种基于词图和词频，并结合 K 最短路径算法为描述相同主题的多个语句生成摘要语句的高效方法。方法构造词图的过程如下：

1. 添加第一条语句，以每个词作为图的一个结点，附加一个 “start” 结点和一个 “end” 结点初始化词图。
2. 依次添加剩余的语句到词图：

> - 如果当前句子中的词在图中已有映射结点，即图中某个词与当前词具有相同的词形和词性，那么直接将该词映射到该结点；
> - 如果当前句子中的词在图中没有相应的映射结点， 则创建一个新的结点。

![image](https://github.com/procyon-lotor/procyon-lotor.github.io/blob/master/images/2017/wordgraph.png?raw=false)

在计算词图边的权重方面，该方法仅考虑词频因素，个人觉得这样的设计有些单薄，仅考虑词频因素往往会引入较多的噪声，导致生成的句子丢失主干信息，这里的实现考虑事件（可以简单理解为 “主-谓-宾” 词组）因素来计算词图边的权重。我们首先利用事件聚类的结果，将当前事件与聚类中心的距离作为事件的权重， 然后利用下面的公式计算词图中每个词的权重：

![image](https://github.com/procyon-lotor/procyon-lotor.github.io/blob/master/images/2017/msc_1.png?raw=false)

其中，dis(i, e) 表示词 i 与事件 e 的余弦距离， w(e) 表示事件的权重， size(E) 表示当前主题的大小。如果某个结点对应的词在一个或多个句子中出现多次，则将其权重进行累加。在边的权重 w(i, j) 计算上，我们考虑当前边所连接的词 i 和 j 的权重， 按照如下公式进行计算：

![image](https://github.com/procyon-lotor/procyon-lotor.github.io/blob/master/images/2017/msc_2.png?raw=false)

其中 pos(s, i) 表示词 i 在句子 s 中的水平位移。

语句的路径得分由路径上所有边的权重之和除以路径长度计算得到。原压缩方法中将这 K 条语句作为压缩的最终输出，这里的实现引入了 Tri-gram 语言模型为每个压缩候选语句进行打分，以此反应语句的语言流畅度。

[takahe](https://github.com/boudinfl/takahe) 对原论文中的方法进行了实现，而本项目是在 takahe 的基础上进行了改进，具体使用方式和依赖库可以参考 takahe，在这里对 takahe 作者表示感谢。
