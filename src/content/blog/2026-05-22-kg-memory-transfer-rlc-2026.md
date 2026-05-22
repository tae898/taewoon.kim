---
title: My KG Memory Transfer paper was accepted at RLC 2026
subtitle: Why I care about learning which symbolic facts should persist
cover-img: /assets/img/posts/2026-05-22/memory_state_099.png
thumbnail-img: /assets/img/posts/2026-05-22/memory_state_099.png
tags: [reinforcement learning, knowledge graph, neuro-symbolic AI]
author: Taewoon Kim
mathjax: true
---

Our paper, [Short-Term-to-Long-Term Memory Transfer for Knowledge Graphs under Partial
Observability](https://arxiv.org/abs/2605.22142), was accepted at
[RLC 2026](https://rl-conference.cc/).

This project sits inside a research direction I care about a lot: explicit memory for
reinforcement learning agents. In earlier work, I focused on how an agent can represent
and use knowledge graphs as memory. In this paper, I narrow the question to something
more specific and, I think, more fundamental: once an agent has both short-term
observation memory and long-term symbolic memory, how should information move from one
to the other?

That may sound like a small systems question, but under partial observability it is not
small at all. An agent cannot keep everything forever. It has to decide which facts are
worth preserving and which ones should be dropped. I wanted to study that transfer step
directly instead of hiding it inside a recurrent latent state.

## What the paper is about

We use the RoomKG setting, where the agent receives local symbolic observations in the
form of RDF triples. At each step, it sees only part of the world, stores those local
facts in short-term memory, and operates under a limited long-term memory budget.

The core decision is simple: for each short-term fact, should the agent **keep** it and
promote it into long-term memory, or **drop** it?

Formally, if the short-term memory contains $n_t$ items at step $t$, then the transfer
action space is

$$
\mathcal{A}^{\text{tr}}_t := \{0,1\}^{n_t},
$$

so the agent makes one binary keep/drop decision per short-term fact. I like this
formulation because it makes the variable-cardinality part of the problem explicit. The
number of transfer decisions changes over time as the short-term memory changes.

At a higher level, the internal symbolic memory evolves as

$$
M_{t+1} = U\!\left(M_t, o_{t+1}, \mathbf{a}^{\text{tr}}_t\right),
$$

where $M_t$ is the current memory state, $o_{t+1}$ is the next local observation, and
$\mathbf{a}^{\text{tr}}_t$ is the vector of keep/drop decisions. This viewpoint matters
to me because it makes transfer a first-class part of the memory dynamics rather than a
hidden implementation detail.

One technical wrinkle is that the short-term memories are a **set** of RDF triples with
annotations, not an ordered sequence. That means there is no canonical way to align
short-term items across consecutive steps for temporal-difference learning. In practice,
we use randomized, order-agnostic matching. I think that is the right choice here
because it is the simplest way to do matching without pretending the set has an order
that it does not actually have.

The DQN part is also slightly unusual because the network does not output one action
value for the entire memory state. Instead, it outputs one keep/drop value pair for each
current short-term item:

$$
Q_\theta(M_t) \to \{\mathbf{q}_{t,i}\}_{i=1}^{n_t},
\qquad
\mathbf{q}_{t,i} \in \mathbb{R}^2.
$$

So for each short-term fact $i$, the agent gets two values: one for **drop** and one
for **keep**. That is the part I find elegant. A single shared network can score a
variable-size short-term set without pretending the action space has fixed size.

With the notation simplified a bit, the per-item TD target looks like

$$
y_j = r + \gamma (1-d) \max_{a \in \{0,1\}} Q_{\bar{\theta}}(M', j, a),
$$

where $j$ is the index of a sampled matched pair under the randomized set-based matching
procedure, not the identity of a persistent fact across time. If the current estimate is
$q_j$, then the loss is just the average squared TD error over the matched items,

$$
\mathcal{L}(\theta) = \frac{1}{\ell} \sum_{j=1}^{\ell} (q_j - y_j)^2,
$$

with $\ell$ equal to the number of matched pairs available in that transition. I think
this is a neat example of how standard DQN ideas can be adapted to a variable-cardinality
symbolic memory problem without giving up interpretability.

## Why I find the result interesting

What I like about this formulation is not only that it is interpretable, but that it
works. In the controlled RoomKG setting at long-term memory capacity 128, the best
configuration is a DQN temporal-RDF agent with a GCN encoder and a Local-STM transfer
policy. It reaches **38.920** test QA accuracy. That is meaningfully higher than the
strongest symbolic baseline, **Novel-Only** at **31.960**, and far above the
history-based neural baselines, where the Transformer reaches **11.800** and the LSTM
**7.600**.

Those numbers are encouraging, but what matters even more to me is *why* the policy
works. Because the transfer decisions act on explicit triples, I can inspect what was
kept and what was dropped. The learned policy tends to preserve self-location and
question-relevant object-location facts while discarding many lower-value local map-link
observations. That is exactly the kind of behavior I hoped explicit memory would make
visible.

## Why this fits the broader project

This paper feels like a natural continuation of the memory-systems line of work I have
been building. First I asked whether explicit memory helps. Then I asked whether parts of
memory management can be learned. Now I am asking whether short-term-to-long-term
transfer itself can be isolated as the learning problem.

That progression is important to me. I do not think of memory as a single monolithic
module. I think of it as a collection of concrete subproblems: storage, retrieval,
forgetting, exploration, and transfer. If memory is explicit enough, each of those can
be studied directly rather than absorbed into one opaque hidden state.

More broadly, this is how I think about neuro-symbolic reinforcement learning. I am not
interested in symbolic structure only for philosophical reasons. I care about it when it
gives better control, better interpretability, and a cleaner way to ask precise questions
about agent behavior.

If you want a more structured overview of the project itself, I also wrote a
[project page on HumemAI](https://humem.ai/projects/kg-memory-transfer/).

## Links

- Paper: [arXiv](https://arxiv.org/abs/2605.22142)
- Project page: [KG Memory Transfer](https://humem.ai/projects/kg-memory-transfer/)
- GitHub: [kg-memory-transfer](https://github.com/humemai/kg-memory-transfer)
