---
layout: page
title: A Machine With Human-Like Memory Systems
subtitle: A machine that can think and talk like us.
cover-img: /assets/img/projects/human-memory.png
thumbnail-img: /assets/img/projects/human-memory.png
tags: [AI, human memory, knowledge graph, machine learning, deep learning]
author: Taewoon Kim
comments: true
---

First written on 04 March 2024.

## Table of Contents

- [Table of Contents](#table-of-contents)
- [Motivation](#motivation)
- [Human memory system](#human-memory-system)
- [Incoprating the human memory system in AI](#incoprating-the-human-memory-system-in-ai)
- [Stage 1: Understanding how humans work and see if machines can do the same](#stage-1-understanding-how-humans-work-and-see-if-machines-can-do-the-same)
  - [How humans work](#how-humans-work)
  - [What's missing in the current literature](#whats-missing-in-the-current-literature)
  - [Fundamental components in my machine](#fundamental-components-in-my-machine)
  - [Scientific / engineering contributions](#scientific--engineering-contributions)
- [Stage 2: Scaling things up](#stage-2-scaling-things-up)
  - [Scientific / engineering contributions](#scientific--engineering-contributions-1)
- [Stage 3: Production ready in the digital world](#stage-3-production-ready-in-the-digital-world)
  - [Scientific / engineering contributions](#scientific--engineering-contributions-2)
- [Stage 4: Production ready in the real world](#stage-4-production-ready-in-the-real-world)
  - [Scientific / engineering contributions](#scientific--engineering-contributions-3)
- [Cite this project](#cite-this-project)
- [References](#references)

## Motivation

I've always been fascinated with intelligent machines. They have the power to augment
our lives. They become more powerful if we can talk to them in a natural language. This
became reality with [ChatGPT](https://openai.com/blog/chatgpt) (OpenAI 2022). ChatGPT is
by no means perfect. Everytime you start a new conversation, it starts from scratch,
meaning that it does not remember who you are. OpenAI is trying to tackle this problem
with its ["memory"](https://openai.com/blog/memory-and-new-controls-for-chatgpt) (OpenAI
2024), which seems to be another prompt engineering based feature. A more effective
strategy would be to prioritize the development of an AI with its memory capabilities as
the foundational element. That's why I started my project "A Machine With Human-Like
Memory Systems".

## Human memory system

![alt text](<memory hierarchy.png>)

Let's first try to understand how human memory systems work. Above is the human memory
hierarchy. At the heart of this system are two critical components: short-term (or
working) memory and long-term memory, each playing unique roles in the cognition
process.

**Short-term Memory (STM) or Working Memory**: This stage temporarily holds and
processes a limited amount of information, typically for about 20 to 30 seconds. It's
not just a passive storage space but an active workshop where information is manipulated
for various cognitive tasks, such as problem-solving, language comprehension, and
planning. Working memory is where conscious thought primarily occurs, integrating new
sensory inputs with information retrieved from long-term memory to make sense of the
world around us.

**Long-term Memory (LTM)**: Information that is deemed important or is repeatedly
rehearsed in short-term memory can be transferred to long-term memory, where it can
remain for days, years, or even a lifetime. Long-term memory is vast and can store a
huge quantity of information. It is divided into explicit (or declarative) memory, which
includes memories that can be consciously recalled, such as facts and events, and
implicit (or non-declarative) memory, for the skills and habits we've acquired, the
phenomena of priming, and our emotional responses. Priming is an aspect of implicit
memory that deals with the unconscious influence of an earlier presented stimulus on the
response to a later stimulus. Emotional conditioning is another facet of implicit
memory, involving the learning of emotional responses to certain stimuli. Through
experiences, certain neutral stimuli can become associated with emotional responses,
shaping our preferences, fears, and even our interpersonal relationships.

What's not included in the above hierarchy is **sensory memory (information)**. Sensory
memory acts as the initial stage in our memory system, capturing all the information
from our environment through our senses. It quickly filters through this vast amount of
data to decide what is important enough to pass on to our short-term memory. This
process is like a brief moment of consideration before some of this sensory information
is selected for further attention and use. Therefore, sensory memory is directly linked
to short-term memory as it serves as the gateway, ensuring that only the most relevant
information makes it to the next stage where we can consciously work with it. The reason
why it's not included is that I model this memory differently from short-term and
long-term memories for my AI .

## Incoprating the human memory system in AI


**Graphs to represent memory**: foo

**


As mentioned in the [Motivation](#motivation), I want to prioritize the development 
of an AI with its memory capabilities as the foundational element. 


## Stage 1: Understanding how humans work and see if machines can do the same

### How humans work

Cognitive science has studied the human brain and its memory systems. They have come up
with a human memory hierarchy. It looks like the image below

- The memory systems closely follow those of humans.
  - The memory hierarchy of my machine should resemble that of humans. This might be a
    too strict of of a restriction, but I believe in the end this will really pay off.

### What's missing in the current literature

- This stage aims to search existing literatures and see what is missing. It also
  studies some scientific contributions in its early phase.
- The existing literature is quite sparse.
  - There is this community called “cognitive architecture”
    https://en.m.wikipedia.org/wiki/Cognitive_architecture
  - Machine learning communities, especially reinforcement learning, use some memory
    systems to learn the policies of interest.
    - I like what they are doing. But I want to do more than just learning the
      functions. I want to see these functions actually being used for real world tasks,
      e.g., communication with humans.

### Fundamental components in my machine

- The format of the memory systems is a graph.
  - Graph structured data has been well studied in computer science and I can take
    advantage of it.
  - If the graph is too big to be loaded into RAM, we can take advantage of graph
    databases to store them in storage.
  - Graph neural networks (GNNs) in geometric deep learning have become a very powerful
    tool to learn from graphs. This includes node classification, link prediction, etc.
  - Graph structured data can easily be turned into a knowledge graph. The knowledge
    graph community has been doing this for a while and it supports many things such as
    logical inferences. They have also built large free public knowledge bases which my
    machine can use as a prior. One more nice thing is that updating the memory of the
    agent can theoretically be done by simply modifying an entity in the graph.
  - This graph is potentially a multi-modal graph, that doesn't just include string
    values.
- The policies (functions) are learned with data. I don't want to use heuristics for
  them. As we humans have learned them, the machine should be able to learn them too.
  - Encoding sensory information into short-term memory.
  - Memory management policy. This agent learns how to remember things.
  - Memory retrieval policy. The agent learns what to retrieve to perform a given task,
    e.g., question answering.
  - Exploration policy. My machine is curious. It explores the world by itself without
    being told to do so. Exploration comes with a cost.

### Scientific / engineering contributions

- Encoding sensory information into short-term memory.
  - The “sensory information” here is basically is raw multimodal data in deep learning,
    e.g., natural language, audio, image, video, time-series, tables, etc.
  - The short-term memory has to be a graph, as mentioned!
  - This might be tackled in my PhD. Since this is already a really big topic, a small
    fraction of it can be tackled in my PhD.
- Learning the memory management policy.
  - This can be broken down to several contributions, and that’s basically what I did
    for my PhD thesis.
    - What to do with a short-term memory. Move it to the episodic? semantic? or forget
      it?
    - What to do with episodic and semantic memories. Merge them? Move one to another?
      etc.
  - This was the first thing I tackled in my PhD and even that was hard.
- Learning the memory retrieval policy.
  - We humans retrieve information in many situations. It can be emotion based in
    episodic memory retireval. It can also be context based, and it can also be content
    based (semantic memory)
  - I might tackle this in my PhD but not so sure. I've just been using heuristics for
    this.
  - Emotion based memory retrieval is very interesting. I am sure that this is not
    studied a lot. I might do this after my PhD.
- Learning the exploration policy.
  - Adding this policy can complicate things dramatically, but I did tackle a small part
    of it.
- Learning multiple policies at once.
  - I will tackle this in my PhD. This is not an easy task. The biggest reason that I am
    doing this is that this can potentially lead to better generalization than learning
    policies one at a time. We humans probably learn them simultaneously. Our robots can
    do this too.
  - We can formulate this into an MARL problem where each agent is responsible for one
    policy.

## Stage 2: Scaling things up

- This stage is about scaling things up. This won't be easy, and definitely won't be
  addressed during my PhD.
- Graph databases will definitely help me here. I wanna learn them anyways so it'll be
  helpful.

### Scientific / engineering contributions

- I am not sure if scaling things up counts as a scientific contribution. But we'll see.
  I still want to write papers about it.

## Stage 3: Production ready in the digital world

- This stage is production ready phase. Humans will actually interact with the machine,
  and it'll use all the mentioned policies.
- This will involve a lot of software engineering.
  - Many things to be considered here, e.g., cloud, front-end (web based? android app?)
- Input / output modalities should be considered. The easiest is when both input and
  output are natural language. I'll probably start with that. But if it can extend to
  audio and vision, it'll be amazing.

### Scientific / engineering contributions

- The scientific contributions made here are mostly human computer interaction (HCI).

## Stage 4: Production ready in the real world

- This is the last stage. Now things are ready to be deployed to the real analog
  physical world.
- An embodied agent can have different forms, from a 3d-printed toy robot to a full
  humanoid robot. Of course simpler it is, the better it'll be.
- Navigation can include procedural (implicit) memories. This type of memory is very
  different from the explicit memories that I dealt with. It probably doesn’t make sense
  to model procedural memory with a graph anymore.

### Scientific / engineering contributions

- The contributions made here are mostly robotics.

<!-- # [A Machine With Human-Like Memory Systems](https://arxiv.org/abs/2204.01611)

Inspired by the cognitive science theory, we explicitly model an agent with both
semantic and episodic memory systems, and show that it is better than having just one of
the two memory systems. In order to show this, we have designed and released our own
challenging environment, "the Room", compatible with OpenAI Gym, where an agent has to
properly learn how to encode, store, and retrieve memories to maximize its rewards. The
Room environment allows for a hybrid intelligence setup where machines and humans can
collaborate. We show that two agents collaborating with each other results in better
performance than one agent acting alone. We have open-sourced our code and models at
[https://github.com/tae898/explicit-memory](https://github.com/tae898/explicit-memory).

# foo

# [A Machine with Short-Term, Episodic, and Semantic Memory Systems](https://arxiv.org/abs/2212.02098)

<!-- padding-bottom: 56.25% is for 16:9. For an aspect ratio of 1:1 change to this value to 100% */  -->
<!-- <div style="position: relative; padding-bottom: 56.25%">
  <iframe
    style="width: 100%; height: 100%; position: absolute; left: 0px; top: 0px"
    frameborder="0"
    width="100%"
    height="100%"
    allowfullscreen
    allow="autoplay"
    src="
  https://www.youtube.com/embed/MsoyjiYuHF0
 "
  >
  </iframe>
</div> -->

<!-- Inspired by the cognitive science theory of the explicit human memory systems, we have
modeled an agent with short-term, episodic, and semantic memory systems, each of which
is modeled with a knowledge graph. To evaluate this system and analyze the behavior of
this agent, we designed and released our own reinforcement learning agent environment,
"the Room", where an agent has to learn how to encode, store, and retrieve memories to
maximize its return by answering questions. We show that our deep Q-learning based agent
successfully learns whether a short-term memory should be forgotten, or rather be stored
in the episodic or semantic memory systems. Our experiments indicate that an agent with
human-like memory systems can outperform an agent without this memory structure in the
environment. The environment is open-sourced at
[https://github.com/tae898/room-env](https://github.com/tae898/room-env), and the agent
is open-sourced at
[https://github.com/tae898/explicit-memory](https://github.com/tae898/explicit-memory). -->

## Cite this project

```bibtex
@article{kim2024machine,
  title   = "A Machine With Human-Like Memory Systems",
  author  = "Kim, Taewoon",
  journal = "https://taewoon.kim",
  year    = "2024",
  month   = "Mar",
  url     = "https://taewoon.kim/projects/human-memory/"
}
```

## References

[1] OpenAI Blog. ["Introducing ChatGPT"](https://openai.com/blog/chatgpt) November 30,
2022

[2] OpenAI Blog. ["Memory and new controls for
ChatGPT"](https://openai.com/blog/memory-and-new-controls-for-chatgpt) February 13, 2024

<!-- [1] Wei et al. ["Chain of thought prompting elicits reasoning in large language
models."](https://arxiv.org/abs/2201.11903) NeurIPS 2022

[2] Yao et al. ["Tree of Thoughts: Deliberate Problem Solving with Large Language
Models."](https://arxiv.org/abs/2305.10601) arXiv preprint arXiv:2305.10601 (2023).

[3] Liu et al. ["Chain of Hindsight Aligns Language Models with
Feedback"](https://arxiv.org/abs/2302.02676) arXiv preprint arXiv:2302.02676 (2023).

[4] Liu et al. ["LLM+P: Empowering Large Language Models with Optimal Planning
Proficiency"](https://arxiv.org/abs/2304.11477) arXiv preprint arXiv:2304.11477 (2023).

[5] Yao et al. ["ReAct: Synergizing reasoning and acting in language
models."](https://arxiv.org/abs/2210.03629) ICLR 2023.

[6] Google Blog. ["Announcing ScaNN: Efficient Vector Similarity
Search"](https://ai.googleblog.com/2020/07/announcing-scann-efficient-vector.html) July
28, 2020.

[7]
[https://chat.openai.com/share/46ff149e-a4c7-4dd7-a800-fc4a642ea389](https://chat.openai.com/share/46ff149e-a4c7-4dd7-a800-fc4a642ea389)

[8] Shinn & Labash. ["Reflexion: an autonomous agent with dynamic memory and
self-reflection"](https://arxiv.org/abs/2303.11366) arXiv preprint arXiv:2303.11366
(2023).

[9] Laskin et al. ["In-context Reinforcement Learning with Algorithm
Distillation"](https://arxiv.org/abs/2210.14215) ICLR 2023.

[10] Karpas et al. ["MRKL Systems A modular, neuro-symbolic architecture that combines
large language models, external knowledge sources and discrete
reasoning."](https://arxiv.org/abs/2205.00445) arXiv preprint arXiv:2205.00445 (2022).

[11] Weaviate Blog. [Why is Vector Search so
fast?](https://weaviate.io/blog/why-is-vector-search-so-fast) Sep 13, 2022.

[12] Li et al. ["API-Bank: A Benchmark for Tool-Augmented
LLMs"](https://arxiv.org/abs/2304.08244) arXiv preprint arXiv:2304.08244 (2023).

[13] Shen et al. ["HuggingGPT: Solving AI Tasks with ChatGPT and its Friends in
HuggingFace"](https://arxiv.org/abs/2303.17580) arXiv preprint arXiv:2303.17580 (2023).

[14] Bran et al. ["ChemCrow: Augmenting large-language models with chemistry
tools."](https://arxiv.org/abs/2304.05376) arXiv preprint arXiv:2304.05376 (2023).

[15] Boiko et al. ["Emergent autonomous scientific research capabilities of large
language models."](https://arxiv.org/abs/2304.05332) arXiv preprint arXiv:2304.05332
(2023).

[16] Joon Sung Park, et al. ["Generative Agents: Interactive Simulacra of Human
Behavior."](https://arxiv.org/abs/2304.03442) arXiv preprint arXiv:2304.03442 (2023).

[17] AutoGPT.
[https://github.com/Significant-Gravitas/Auto-GPT](https://github.com/Significant-Gravitas/Auto-GPT)

[18] GPT-Engineer.
[https://github.com/AntonOsika/gpt-engineer](https://github.com/AntonOsika/gpt-engineer) -->
