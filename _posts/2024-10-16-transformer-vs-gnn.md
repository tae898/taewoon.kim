---
layout: post
title: Can the Transformer be viewed as a special case of a Graph Neural Network (GNN)?
subtitle: A natural language text can be seen as a knowledge graph
cover-img: /assets/img/posts/2024-10-16/transformer-gnn.webp
thumbnail-img: /assets/img/posts/2024-10-16/transformer-gnn.webp
tags: [GNN, transformer, knowledge graph, NLP, natural language]
author: Taewoon Kim
mathjax: true
---

In recent years, **Transformers** have dominated the field of natural language
processing (NLP), while **Graph Neural Networks (GNNs)** have proven essential for tasks
involving graph-structured data. Interestingly, the Transformer can be seen as a special
case of GNNs, particularly as an **attention-based GNN**. This connection emerges when
we treat natural language as graph data, where tokens (words) are nodes, and their
sequential relationships form edges.

In this post, we’ll explore how the Transformer fits into the broader class of GNNs,
with an emphasis on the mathematical framework and how natural language sequences can be
viewed as a specific case of graph-structured data.

## What Are Graph Neural Networks?

Graph Neural Networks (GNNs) are designed to process graph-structured data. A graph $$ G
= (V, E) $$ consists of:
- **Nodes** $$ V $$ (vertices), which represent entities, and 
- **Edges** $$ E $$ (connections between nodes), which represent relationships between
  these entities.

The graph structure is typically encoded using an **adjacency matrix** $$ \boldsymbol{A}
$$, where $$ \boldsymbol{A}_{ij} $$ is non-zero if there is an edge between node $$ i $$
and node $$ j $$.

### GNNs in Mathematical Form

In a basic GNN, each node $$ v_i $$ has a feature vector $$ \boldsymbol{h}_i^{(0)} $$ at
the initial layer. The goal of the GNN is to update these node features by aggregating
information from neighboring nodes. For each node, the update rule can be generalized
as:

$$ \boldsymbol{h}_i^{(k+1)} = \sigma \left( \sum_{j \in \mathcal{N}(i)} \boldsymbol{W}
\boldsymbol{h}_j^{(k)} + \boldsymbol{b} \right) $$

Where:
- $$ \boldsymbol{h}_i^{(k)} $$ is the feature vector of node $$ i $$ at layer $$ k $$,
- $$ \mathcal{N}(i) $$ denotes the set of neighbors of node $$ i $$,
- $$ \boldsymbol{W} $$ is the weight matrix (learned parameters),
- $$ \boldsymbol{b} $$ is a bias term,
- $$ \sigma $$ is an activation function (e.g., ReLU).

The adjacency matrix $$ \boldsymbol{A} $$ plays a crucial role in determining which
nodes are connected, controlling the message passing process by encoding the graph
structure.

### Types of GNNs

Different variants of GNNs have been developed to handle specific types of graph data:

- **Graph Convolutional Networks (GCNs)**: These apply a spectral or spatial convolution
  to aggregate information from neighboring nodes.
- **Graph Attention Networks (GATs)**: These use an attention mechanism to assign
  different weights to neighboring nodes, learning which neighbors are most important.
- **Relational Graph Convolutional Networks (R-GCNs)**: These handle graphs with
  multiple types of edges (relations) by associating different weights with different
  edge types. This is particularly useful for **knowledge graphs**, where relationships
  between entities vary (e.g., "friendOf," "worksAt").

## What Are Transformers?

Transformers, introduced in ["Attention is All You
Need"](https://arxiv.org/abs/1706.03762), are designed to model sequential data like
text. The key feature of the Transformer is **self-attention**, which allows each token
to attend to every other token in the sequence. This can be mathematically framed using
the **scaled dot-product attention** mechanism.

### Transformer Self-Attention

In a Transformer, given an input sequence of tokens $$ \boldsymbol{x}_1,
\boldsymbol{x}_2, \dots, \boldsymbol{x}_N $$, where $$ N $$ is the length of the
sequence, we compute attention scores between all token pairs. For each token $$ i $$,
its representation $$ \boldsymbol{z}_i $$ at the next layer is computed as a weighted
sum of all token representations:

$$ \text{Attention}(\boldsymbol{Q}, \boldsymbol{K}, \boldsymbol{V}) = \text{softmax}
\left( \frac{\boldsymbol{Q} \boldsymbol{K}^T}{\sqrt{d_k}} \right) \boldsymbol{V} $$

Where:
- $$ \boldsymbol{Q} = \boldsymbol{W}_q \boldsymbol{X} $$ (queries),
- $$ \boldsymbol{K} = \boldsymbol{W}_k \boldsymbol{X} $$ (keys),
- $$ \boldsymbol{V} = \boldsymbol{W}_v \boldsymbol{X} $$ (values),
- $$ \boldsymbol{X} $$ is the input token matrix (where each row is a token embedding),
- $$ d_k $$ is the dimensionality of the queries/keys.

The attention mechanism computes a fully connected graph between all tokens, where
attention weights determine the "edges" (connections) between nodes (tokens). This can
be interpreted as a graph where every token can communicate with every other token.

### Positional Embeddings and Masking

In sequence modeling, the order of the tokens is crucial. Instead of explicitly encoding
this order using an adjacency matrix (as done in GNNs), the Transformer uses
**positional embeddings** $$ \boldsymbol{P} $$ to encode the position of each token:

$$ \boldsymbol{z}_i = \text{Attention}(\boldsymbol{Q}, \boldsymbol{K}, \boldsymbol{V}) +
\boldsymbol{P}_i $$

In **standard Transformers**, masking is not inherently required. However, in models
like **large language models (LLMs)**, we use **decoder-only Transformers** where tokens
attend to other tokens in an **autoregressive (causal) manner**. This ensures that each
token only attends to preceding tokens, which is critical for generating text
sequentially. 

Therefore, in these cases, **masking** is applied to enforce this autoregressive
behavior, ensuring that future tokens are masked out during training. The combination of
**masking** and **positional encodings** creates a structure where the Transformer
attends only to past tokens, mimicking the adjacency matrix that would be used in a GNN.
It’s important to note that masking and positional encodings are not part of the
Transformer architecture itself—they are techniques applied in specific contexts like
LLMs.

Finally, like many GNNs, the **Transformer itself** is **input permutation-invariant**.
This means that, without positional encodings, the model does not inherently preserve
the order of the inputs, treating all tokens symmetrically.

## Natural Language as a Special Case of Graph Data

One key observation is that **natural language** can be treated as a form of
**graph-structured data**. In a sentence, tokens form nodes, and their sequential
relationships form edges. For instance, a sequence of tokens $$ \text{Token}_1,
\text{Token}_2, \dots, \text{Token}_N $$ can be visualized as a **directed graph** where
each token is connected to every preceding token:

$$ \text{Token}_1 \rightarrow \text{Token}_2 \rightarrow \dots \rightarrow
\text{Token}_N $$

In reality, for token $$ N $$, there are directed edges from each of the previous $$ N-1
$$ tokens:

$$ \text{Token}_1 \rightarrow \text{Token}_N, \quad \text{Token}_2 \rightarrow
\text{Token}_N, \quad \dots, \quad \text{Token}_{N-1} \rightarrow \text{Token}_N $$

In traditional GNNs, such as **R-GCNs**, we might explicitly encode these relationships
with **multiple adjacency matrices** to represent different types of relationships. For
example, in a sequence of tokens, we would have separate adjacency matrices to define
the "1-next," "2-next," ..., "N-next" relationships.

For each relationship type $$ r $$ (e.g., 1-next, 2-next, etc.), we define a separate
adjacency matrix $$ \boldsymbol{A}^{(r)} $$ that represents the connections for that
specific relation:

$$ \boldsymbol{A}_{ij}^{(r)} = \begin{cases} 1 & \text{if token } j \text{ has a
relation } r \text{ with token } i, \\
0 & \text{otherwise}. \end{cases} $$

In the case of a sequence, we would have a matrix for each "$$ k $$-next" relation,
where $$ k $$ defines the step size between tokens in the sequence (1-next, 2-next, ...,
N-next).

However, in the Transformer, we do not need this explicit adjacency matrix because
**positional embeddings** serve a similar purpose. Instead of encoding relationships
directly as edge types, the positional embeddings implicitly encode the sequential
relationships between tokens. Thus, positional embeddings replace the need for an
adjacency matrix while maintaining the graph structure.

### Attention-Based GNNs and Transformers

In GNNs like **Graph Attention Networks (GATs)**, attention is used to compute weights
for each neighboring node, allowing the model to focus on the most relevant nodes during
the message-passing process. The Transformer takes this idea to the extreme by using
**global self-attention**, where every token can attend to every other token. This
global connectivity forms a fully connected graph in GNN terms.

While **R-GCNs** use relation-specific weights to handle different types of edges in
knowledge graphs, the Transformer simplifies this by using **positional embeddings** to
implicitly handle the sequential relationships between tokens.

---

## Visualization: Natural Language as Graph Data

To better illustrate how natural language sequences can be represented as graph data,
consider the following structure. Suppose we have a sentence composed of tokens:
$$\text{Token}_1$$, $$\text{Token}_2$$, $$\text{Token}_3$$, ..., $$\text{Token}_N$$. The
sequential nature of the sentence can be represented as a directed graph:

![](/assets/img/posts/2024-10-16/text-as-kg.svg)

For visualization purposes, only the edges from $$\text{Token}_1$$ are shown.

In a **relational GNN** like R-GCN, we would typically encode these $$\text{1-next}$$,
$$\text{2-next}$$, ... $$\text{(N-1)-next}$$ relations with an adjacency matrix and
relation-specific embeddings. However, in a **Transformer**, we replace this structure
with **positional embeddings** that capture the token order and **masking** that
enforces autoregressive behavior during training. If we were to create the embeddings
for all these relations, it also becomes infeasible as we would need more and more of
them as the input length grows.


### Conclusion

In summary, **Transformers** can be viewed as a special case of **Graph Neural
Networks**, particularly **attention-based GNNs**. Natural language, in this context, is
a specific type of graph data, where masking and positional embeddings replace the need
for explicit adjacency matrices. This allows Transformers to model sequential data
efficiently without requiring the edge-specific representations used in GNNs like
**R-GCN**.


### References:
- [Attention is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)
- [Semi-Supervised Classification with Graph Convolutional Networks (Kipf & Welling,
  2016)](https://arxiv.org/abs/1609.02907)
- [Graph Attention Networks (Veličković et al., 2017)](https://arxiv.org/abs/1710.10903)
- [Modeling Relational Data with Graph Convolutional Networks (Schlichtkrull et al.,
  2018)](https://arxiv.org/abs/1703.06103)
- [A Comprehensive Introduction to Graph Neural Networks
  (GNNs)](https://distill.pub/2021/gnn-intro/)
- [Graph Neural Networks: A Review of Methods and Applications (Zhou et al.,
  2020)](https://arxiv.org/abs/1812.08434)
