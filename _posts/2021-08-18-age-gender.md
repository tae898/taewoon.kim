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
Check out [https://arxiv.org/abs/2108.08186](https://arxiv.org/abs/2108.08186)

**_Abstract_**: A multilayer perceptron (MLP) is typically made of multiple fully
connected layers with nonlinear activation functions. There have been several approaches
to make them better (e.g. faster convergence, better convergence limit, etc.). But the
researches lack structured ways to test them. We test different MLP architectures by
carrying out the experiments on the age and gender datasets. We empirically show that by
whitening inputs before every linear layer and adding skip connections, our proposed MLP
architecture can result in better performance. Since the whitening process includes
dropouts, it can also be used to approximate Bayesian inference. We have open-sourced
our code, and released models and docker images at
[https://github.com/tae898/age-gender/](https://github.com/tae898/age-gender/)
