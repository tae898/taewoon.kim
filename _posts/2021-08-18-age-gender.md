---
layout: post
title: "Generalizing MLPs with dropouts, batch normalization, and skip connections"
subtitle: A simple age-gender classification model using Arcface embeddings and MLPs
cover-img: /assets/img/posts/2021-08-18/age-gender.png
thumbnail-img: /assets/img/posts/2021-08-18/age-gender.png
tags: [mlp, age, gender]
author: Taewoon Kim
mathjax: true
---

I trained a simple age-gender classification model using ArcFace embeddings and MLPs.
Check out the paper [https://arxiv.org/abs/2108.08186](https://arxiv.org/abs/2108.08186)
and the
code [https://github.com/tae898/age-gender/](https://github.com/tae898/age-gender/)

Multilayer Perceptrons (MLPs) are foundational neural network architectures composed of
stacked fully connected layers with nonlinear activation functions. Despite numerous
attempts to enhance MLPs for faster convergence and improved performance, systematic
testing methods are often lacking. In our work, we propose a generalized MLP
architecture that integrates dropouts, batch normalization, and skip connections, and we
evaluate its effectiveness on age and gender classification tasks.

Our architecture introduces two key building blocks: the **residual block** and the
**downsample block**. Both blocks utilize an Independent Component (IC) layer—comprising
batch normalization and dropout—to whiten inputs before each fully connected layer. The
residual block includes skip connections to facilitate training deeper networks and is
defined as:

$$
\mathbf{y} = \mathrm{ReLU}\left( \mathbf{W}^{(2)} \, \mathrm{Dropout}\left(
\mathrm{BN}\left( \mathrm{ReLU}\left( \mathbf{W}^{(1)} \, \mathrm{Dropout}\left(
\mathrm{BN}\left( \mathbf{x} \right) \right) \right) \right) \right) + \mathbf{x}
\right)
$$

By incorporating dropouts, our MLP not only benefits from regularization but also
enables uncertainty estimation through **Monte Carlo (MC) dropout**. In a Bayesian
framework, the posterior predictive distribution is obtained by marginalizing over the
neural network weights $$\mathbf{w}$$:

$$
p(\mathbf{y} \mid \mathbf{x}, \mathbf{X}, \mathbf{Y}) = \int p(\mathbf{y} \mid
\mathbf{x}, \mathbf{w}) \, p(\mathbf{w} \mid \mathbf{X}, \mathbf{Y}) \, d\mathbf{w}
$$

Since this integral is intractable, MC dropout approximates it by performing multiple
stochastic forward passes with dropout enabled during inference, effectively sampling
from the posterior distribution of the weights.

We tested our architecture on the Adience and IMDB-WIKI datasets for age and gender
classification. Preprocessing steps included using RetinaFace for alignment and ArcFace
for feature extraction, resulting in 512-dimensional vectors as inputs. Our empirical
results show that whitening inputs before every linear layer and adding skip connections
lead to improved convergence speed and accuracy compared to variants without these
components. Moreover, the use of MC dropout allows our MLP to estimate prediction
uncertainties effectively, providing a measure of confidence in the model's predictions.
