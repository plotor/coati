## Coati

Coati is an implementation of multi-sentence compression algorithms. Based on the word graph, it compresses multiple sentences describing the same or similar topics, and ultimately generates sentences carrying the main topic information. For the specific algorithm, please refer to the following two papers:

[![AUR](https://img.shields.io/aur/license/yaourt.svg)](https://github.com/procyon-lotor/coati/blob/master/LICENSE)

1. [Multi-sentence compression: finding shortest paths in word graphs](http://dl.acm.org/citation.cfm?id=1873818)
2. [Multiple alternative sentence compressions for automatic text summarization](http://www.umiacs.umd.edu/~dmzajic/papers/DUC2007.pdf)

These papers propose an effective method based on word graph and word frequency, combined with the K-shortest paths algorithm to generate summary sentences for multiple sentences with similar topics. The process of constructing the word graph is as follows:

1. Add the first sentence, taking each word as a node in the graph. Attach a "start" node and an "end" node to initialize the word graph.
2. Add the remaining sentences to the word graph in turn:

> - If a word in the current sentence already has a mapped node in the graph, i.e., a word in the graph has the same word form and POS tag as the current word, then map the word to that node directly.
> - If a word in the current sentence has no corresponding mapped node in the graph, create a new node.


![image](https://github.com/procyon-lotor/procyon-lotor.github.io/blob/master/images/2017/wordgraph.png?raw=false)

When calculating the weight of word graph edges, the original method only considers the factor of word frequency. In my opinion, such a design is somewhat one-dimensional — considering only word frequency tends to introduce more noise and leads to the generated sentences losing core information. This implementation considers events (which can be simply understood as "subject-verb-object" phrases) as a factor to calculate the weight of word graph edges. We first use the results of event clustering and take the distance between the current event and the cluster center as the event's weight, then use the formula below to calculate the weight of each word in the word graph:

![image](https://github.com/procyon-lotor/procyon-lotor.github.io/blob/master/images/2017/msc_1.png?raw=false)

Where dis(i, e) denotes the cosine distance between word i and event e, w(e) denotes the weight of the event, and size(E) denotes the size of the current topic. If the word corresponding to a node appears multiple times in one or more sentences, its weight is accumulated. For the calculation of edge weight w(i, j), we consider the weights of words i and j connected by the current edge, and compute it using the following formula:

![image](https://github.com/procyon-lotor/procyon-lotor.github.io/blob/master/images/2017/msc_2.png?raw=false)

Where pos(s, i) denotes the horizontal displacement of word i in sentence s.

The path score of a sentence is calculated by dividing the sum of all edge weights on the path by the path length. The original compression method takes the K sentences as the final output, while this implementation introduces a Tri-gram language model to score each compression candidate sentence, thereby reflecting the fluency of the sentences.

[takahe](https://github.com/boudinfl/takahe) implements the method from the original paper, and this project improves upon it. For usage and dependencies, please refer to takahe. Thanks to the author of takahe.
