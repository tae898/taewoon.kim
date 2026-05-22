---
title: My KG Memory Transfer paper was accepted at RLC 2026
subtitle: Why I care about learned transfer from short-term observation to long-term symbolic memory
cover-img: /assets/img/posts/2026-05-22/memory_state_099.png
thumbnail-img: /assets/img/posts/2026-05-22/memory_state_099.png
tags: [reinforcement learning, knowledge graph, neuro-symbolic AI]
author: Taewoon Kim
mathjax: true
---

I am happy to share that our paper, [Short-Term-to-Long-Term Memory Transfer for
Knowledge Graphs under Partial Observability](https://arxiv.org/abs/2605.22142), was
accepted at [RLC 2026](https://rl-conference.cc/).

This paper is part of a larger research direction I have been working on for a while:
explicit memory systems for reinforcement learning agents. In earlier work, I focused on
how an agent can store and use knowledge graphs as memory. In this paper, I focus on a
more specific question: once an agent has a short-term memory buffer and a long-term
symbolic memory, how should information move from one to the other?

That question sounds small at first, but I think it is actually quite central. If an
agent is acting under partial observability, then it cannot keep everything forever.
It has to be selective. Some observations are worth promoting into long-term memory, and
some are not. I wanted to study that transfer process directly, instead of treating it
as an invisible side effect inside a recurrent hidden state.

## What the paper studies

The setup uses the RoomKG benchmark, where the agent receives symbolic observations in
the form of RDF triples. At each step, the agent sees only a local part of the world,
stores some facts in short-term memory, and has limited long-term memory capacity.

The main decision is simple to state: for each short-term fact, should the agent
**keep** it and transfer it into long-term memory, or **drop** it?

That is the core of the paper. We turn short-term-to-long-term transfer into an explicit
reinforcement learning problem over symbolic memories. Instead of learning one opaque
latent state, the agent learns a keep/drop policy over individual knowledge-graph facts.

If the short-term memory contains $n_t$ items at time $t$, then the transfer action
space is

$$
\mathcal{A}^{\text{tr}}_t := \{0,1\}^{n_t},
$$

which means the agent makes one binary keep/drop decision for each short-term fact.
I like this formulation because it makes the variable-cardinality part of the problem
explicit. The number of transfer decisions changes over time as the short-term memory
changes.

I like this formulation because it keeps the decision process interpretable. After the
agent runs, I can inspect which triples were kept, which were dropped, and whether those
choices make sense for navigation and question answering.

At a high level, the internal symbolic memory state evolves as

$$
M_{t+1} = U\!\left(M_t, o_{t+1}, \mathbf{a}^{\text{tr}}_t\right),
$$

where $M_t$ is the current memory state, $o_{t+1}$ is the next local observation, and
$\mathbf{a}^{\text{tr}}_t$ is the vector of keep/drop decisions. I find this a useful
way to think about the paper because it makes transfer a first-class part of the memory
dynamics rather than a hidden implementation detail.

## Why I think this matters

One reason I keep coming back to symbolic memory is that I do not want everything to be
buried inside weights. Latent-state approaches are powerful, but when I care about
memory, I also care about visibility. I want to know what the agent retained, why it was
retained, and how that affects later behavior.

This paper shows that explicit transfer decisions are not only more inspectable, but can
also work well empirically. In the controlled RoomKG setting, learned symbolic transfer
outperforms symbolic heuristics and history-based neural baselines. That does not mean
the problem is solved, but it does suggest that explicit memory transfer is a useful
thing to learn directly.

More broadly, this is how I think about neuro-symbolic reinforcement learning. I am not
interested in adding symbolic structure only for philosophical reasons. I am interested
in it when it gives me better control, better interpretability, and a better way to ask
precise questions about agent behavior.

## A useful next step in the project line

For me personally, this paper feels like a natural continuation of the earlier memory
systems work. First I asked whether explicit memory systems help. Then I asked whether
memory management can be learned. Now I am asking whether transfer into long-term
knowledge-graph memory can itself be treated as the learning problem.

That progression is important to me. I do not see memory as a generic add-on module. I
see it as a collection of concrete subproblems: storage, retrieval, forgetting,
exploration, and transfer. Each of those can be studied separately if the agent’s memory
is explicit enough.

If you want a more structured overview of this particular project, I also wrote a
[project page on HumemAI](https://humem.ai/projects/kg-memory-transfer/).

## Links

- Paper: [arXiv](https://arxiv.org/abs/2605.22142)
- Project page: [KG Memory Transfer](https://humem.ai/projects/kg-memory-transfer/)
- GitHub: [kg-memory-transfer](https://github.com/humemai/kg-memory-transfer)
