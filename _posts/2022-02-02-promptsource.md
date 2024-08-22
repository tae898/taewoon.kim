---
layout: post
title: "Promptsource: An integrated development environment and repository for natural language prompts"
subtitle: Revolutionizing NLP with a collaborative hub for crafting and sharing prompts
cover-img: /assets/img/posts/2022-02-02/PromptSource.png
thumbnail-img: /assets/img/posts/2022-02-02/PromptSource.png
tags: [llm, prompt, gpt, promptsource]
author: Taewoon Kim
mathjax: true
---

This paper was a result of the Hugging Face BigScience research workshop that I
participated in 2021. The paper can be found at
[https://arxiv.org/abs/2202.01279](https://arxiv.org/abs/2202.01279).

**_Abstract_**: PromptSource is a system for creating, sharing, and using natural
language prompts. Prompts are functions that map an example from a dataset to a natural
language input and target output. Using prompts to train and query language models is an
emerging area in NLP that requires new tools that let users develop and refine these
prompts collaboratively. PromptSource addresses the emergent challenges in this new
setting with (1) a templating language for defining data-linked prompts, (2) an
interface that lets users quickly iterate on prompt development by observing outputs of
their prompts on many examples, and (3) a community-driven set of guidelines for
contributing new prompts to a common pool. Over 2,000 prompts for roughly 170 datasets
are already available in PromptSource. PromptSource is available at
[https://github.com/bigscience-workshop/promptsource](https://github.com/bigscience-workshop/promptsource).
