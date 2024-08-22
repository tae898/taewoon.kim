---
layout: post
title: Multitask prompted training enables zero-shot task generalization
subtitle: Achieving superior zero-shot generalization through explicit multitask prompting
cover-img: /assets/img/posts/2021-10-15/Octopus.png
thumbnail-img: /assets/img/posts/2021-10-15/Octopus.png
tags: [llm, prompt, gpt]
author: Taewoon Kim
mathjax: true
---

This paper was a result of the Hugging Face BigScience research workshop that I
participated in 2021. The paper can be found at
[https://arxiv.org/abs/2110.08207](https://arxiv.org/abs/2110.08207).

**_Abstract_**: Large language models have recently been shown to attain reasonable
zero-shot generalization on a diverse set of tasks (Brown et al., 2020). It has been
hypothesized that this is a consequence of implicit multitask learning in language
models' pretraining (Radford et al., 2019). Can zero-shot generalization instead be
directly induced by explicit multitask learning? To test this question at scale, we
develop a system for easily mapping any natural language tasks into a human-readable
prompted form. We convert a large set of supervised datasets, each with multiple prompts
with diverse wording. These prompted datasets allow for benchmarking the ability of a
model to perform completely held-out tasks. We fine-tune a pretrained encoder-decoder
model (Raffel et al., 2020; Lester et al., 2021) on this multitask mixture covering a
wide variety of tasks. The model attains strong zero-shot performance on several
standard datasets, often outperforming models up to 16x its size. Further, our approach
attains strong performance on a subset of tasks from the BIG-bench benchmark,
outperforming models up to 6x its size. All trained models are available at
[https://github.com/bigscience-workshop/t-zero](https://github.com/bigscience-workshop/t-zero)
and all prompts are available at
[https://github.com/bigscience-workshop/promptsource](https://github.com/bigscience-workshop/promptsource).
