---
layout: post
title: Training a GCN-based edge classifier
subtitle: Exploring the challenges and strategies for effective edge classification with graph neural networks (GNNs)
cover-img: /assets/img/posts/2024-08-21/gnn-edge-classifier.png
thumbnail-img: /assets/img/posts/2024-08-21/gnn-edge-classifier.png
tags: [knowledge graph, machine learning, deep learning, gnn]
author: Taewoon Kim
mathjax: true
---

In this post, I'll walk you through the process of training a Graph Convolutional
Network (GCN) for edge classification, a common task in graph-based machine learning
applications. Edge classification involves predicting the type of relationship (or edge)
between two nodes in a graph, which is particularly useful in areas like knowledge graph
completion, social network analysis, and recommendation systems.

### The Basics of GCNs

Graph Convolutional Networks (GCNs) are a class of neural networks specifically designed
to operate on graph-structured data. Instead of assuming that data points are
independent and identically distributed (as in traditional neural networks), GCNs take
into account the connections (edges) between data points (nodes). The core idea of a GCN
is to iteratively update the feature representation of each node by aggregating
information from its neighbors.

Mathematically, the update rule for a single layer in a GCN can be expressed as:

$$ \boldsymbol{H}^{(l+1)} = \sigma\left( \hat{\boldsymbol{D}}^{-1/2}
\hat{\boldsymbol{A}} \hat{\boldsymbol{D}}^{-1/2} \boldsymbol{H}^{(l)}
\boldsymbol{W}^{(l)} \right) $$

Where:

- $$ \boldsymbol{H}^{(l)} $$ is the matrix of node features at layer $$ l $$.
- $$ \hat{\boldsymbol{A}} = \boldsymbol{A} + \boldsymbol{I} $$ is the adjacency matrix
  with added self-loops.
- $$ \hat{\boldsymbol{D}} $$ is the diagonal degree matrix of $$ \hat{\boldsymbol{A}} $$.
- $$ \boldsymbol{W}^{(l)} $$ is the layer-specific trainable weight matrix.
- $$ \sigma $$ is the activation function, such as ReLU.

### Spatial vs. Spectral Methods in PyTorch Geometric

It's important to note that while the original GCN paper by Kipf & Welling can be viewed
as deriving from a spectral perspective (using the graph Laplacian), **the PyTorch
Geometric implementation (`GCNConv`) adopts a spatial, message-passing approach.** In
this spatial formulation, each node's representation is updated based on the features of
its neighbors, without explicitly computing the Laplacian's eigenvalues or eigenvectors.

**Why is this done?**  
1. **Scalability**: Computing and storing eigenvectors (as in purely spectral methods)
   can be expensive for large graphs. Spatial methods rely on localized computations
   using direct adjacency information, making them more scalable.
2. **Flexibility**: Message passing allows for easy integration of additional node- and
   edge-level features.  
3. **Efficiency**: By avoiding eigen-decomposition and instead performing neighbor
   aggregation (via adjacency lookups), the implementation can handle large and dynamic
   graphs with far less overhead.

With this in mind, PyTorch Geometric's GCNs operate in a **spatial** manner: they work
locally, passing messages from neighbors and updating node embeddings, rather than using
global spectral transformations.

### Initial Approach: Random Node Features

In the first approach, we create random graphs by generating random node features and
edges. Each graph is independent, and the GCN is trained to predict the type of
relationship (edge) between nodes based purely on these random features. While this
method is simple and provides a good introduction to GCNs, it has limitations. Since the
node features and edges are random, there's no meaningful structure for the GCN to
learn, which limits the model's performance.

When batching multiple graphs together during training, we need to increment the edge
indices to ensure that each graph’s nodes are correctly referenced within the combined
set of node features. This process ensures that the graphs remain independent within the
batch, allowing the GCN to aggregate information correctly without mixing up nodes from
different graphs.

### Alternative Approach 1: Learnable Node Embeddings

To address the shortcomings of the initial approach, we introduce learnable node
embeddings. Instead of generating random node features for each graph, we use a fixed
set of embeddings that are shared across all graphs in the dataset. These embeddings are
updated during training, allowing the model to learn a more meaningful representation of
each node.

Similar to the initial approach, batching requires incrementing the edge indices. This
ensures that nodes from different graphs in a batch are correctly referenced,
maintaining the independence of the graphs while leveraging the shared node embeddings.

### Alternative Approach 2: Treating All Graphs as One Large Graph

One might consider treating all graphs in a batch as part of one large, interconnected
graph. This method involves using a single set of node embeddings and keeping the edge
indices as-is, without any adjustments. The idea here is to have a consistent node
representation across all graphs.

However, **this approach doesn't work!** The loss (cross entropy loss) doesn't go down at
all. If you decrease the batch size, the loss does start to go down, but very slowly.
What's happening here?

Essentially, by combining all graphs into one, the model can no longer distinguish which
edges belong to which original graph. The node embeddings across different graphs may
interfere with each other if they share node indices. This confusion leads to poor
training signals and stalls the improvement of the model.

### Code

If you're interested in exploring this further or experimenting with the code, you can
check out the full implementation on
[GitHub](https://github.com/tae898/gnn-edge-classifier). The repository contains the
Jupyter notebook used for this post, along with the data and scripts needed to reproduce
the experiments.
