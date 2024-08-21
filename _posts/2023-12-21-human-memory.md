---
layout: post
title: A Machine With Human-Like Memory Systems
subtitle: Can machines think like humans?
cover-img: /assets/img/posts/2023-12-21/knowledge-graph-vs-neural-network.png
thumbnail-img: /assets/img/posts/2023-12-21/knowledge-graph-vs-neural-network.png
tags:
  [
    reinforcement learning,
    AI,
    human memory,
    knowledge graph,
    machine learning,
    deep learning,
    PhD,
    LLM,
    symbolic AI,
    GOFAI,
    DALL-E
  ]
author: Taewoon Kim
---

The project ["A Machine With Human-Like Memory Systems"](https://humem.ai) is
the core of my PhD work. It was heavily inspired by the cognitive science theories,
such as the ones from [Endel Tulving](https://scholar.google.com/citations?user=OxmLLMEAAAAJ&hl=en).
It's about developing agents equipped with human-like external memory systems, modeled
using [knowledge graphs](https://arxiv.org/abs/2003.02320). These agents are designed to
learn essential human skills such as memory management, reasoning, and exploration through reinforcement learning.

The biggest difference between my agent and the agents such as
[GPTs](https://en.wikipedia.org/wiki/Generative_pre-trained_transformer) is that what my
agent remembers is explicitly stored in its memory system as knowledge graphs, whlie the
[LLMs](../projects/llm) such as GPT remember things in their weights. These weights are
floating values, and although there have been many works to understand what these values
actually mean, it's still not interpretable. Plus, the knowledge graphs of my agent are
designed to mimic the human-memory systems, so it's extra explainable.

Knowledge graphs aim to capture knowledge in the form of graphs. The captured knowledge
is highly symbolic, and we have [Good old fashioned AI (GOFAI)](https://en.wikipedia.org/wiki/Symbolic_artificial_intelligence)
to process such data. But as we all know that GOFAI suffered from generalization, and that's why
we also have [machine learning](https://en.wikipedia.org/wiki/Machine_learning)!
[Reinforcement learning (RL)](https://arxiv.org/abs/1811.12560), a subfield of machine learning,
helps my agent to be more generalizable. Putting too many symbolic constraints to my agent
can harm its generalization capability. So I eased some of them, and let RL learn the rest.
Sometimes we also call something like this [Neuro-symbolic AI](https://en.wikipedia.org/wiki/Neuro-symbolic_AI).

At the moment the work is purely academic. The works are being published in academic
conferences. But I believe that in the next upcoming years, I can scale this up and make it
even production ready, so that everyone can use it. Maybe it'll be running on your smartphones!
All my works are open-sourced. You can find them in [my GitHub](https://github.com/tae898).
