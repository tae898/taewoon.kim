---
title: My Neuro-Symbolic Meta-Policies paper was accepted at ISWC 2026
subtitle: Why I want agents to choose their memory rules, not hide them
cover-img: /assets/img/posts/2026-07-28/q_values_qa.png
thumbnail-img: /assets/img/posts/2026-07-28/q_values_qa.png
tags: [neuro-symbolic AI, knowledge graph, reinforcement learning]
author: Taewoon Kim
mathjax: true
---

Our paper, *Neuro-Symbolic Meta-Policies for Temporal Knowledge-Graph Memory under
Partial Observability*, was accepted at [ISWC
2026](https://iswc2026.semanticweb.org/). I worked on this with Vincent Francois-Lavet
and Michael Cochez. I wrote a structured overview on the [project
page](https://humem.ai/projects/roomkg-meta-policies/), and the paper itself is on
[arXiv](https://arxiv.org/abs/2607.18368).

This paper closes a loop I have been circling for a while. In my earlier RoomKG work, I
showed that fixed symbolic heuristics over temporal knowledge-graph memory beat
end-to-end neural agents by a wide margin. Then I studied which observed facts should be
promoted into long-term memory. But one thing stayed fixed the whole time: the
heuristics themselves. The agent answered with one predetermined rule, explored with
another, and forgot with a third, at every single step, no matter what its memory looked
like.

That always bothered me, because no single rule is right everywhere. So the question in
this paper is: can an agent *learn which rule to apply* at each moment, without giving up
the property I care about most, which is that every memory operation stays symbolic and
inspectable?

## What the paper is about

The setting is again RoomKG: a partially observable world where the hidden state and the
observations are RDF graphs, and the agent maintains a bounded long-term memory whose
facts carry the temporal annotations `time_added`, `last_accessed`, and `num_recalled`.

The named heuristics rank over exactly those annotations. For question answering, the
agent can trust the most recently added matching fact (MRA), the most recently used one
(MRU), or the most frequently used one (MFU). For forgetting, it can evict FIFO, LRU, or
LFU. Exploration prioritizes frontiers using the same annotation properties.

A tiny example shows why the choice matters. Say memory contains `(sarah, at_location,
living)` with annotations $(4, 11, 5)$ and `(sarah, at_location, kitchen)` with
$(10, 10, 1)$. Asked where Sarah is, MRA answers `kitchen`, while MRU and MFU answer
`living`. If she has just moved, only MRA is right. If the older fact keeps getting
confirmed, the usage-based rules are right. A fixed choice is wrong somewhere.

So instead of fixing the rule, we learn a *meta-policy*: a controller that looks at the
current memory graph and selects one named heuristic per function. The selected
heuristics are then executed by deterministic symbolic procedures. The neural part only
ever chooses among named options. It never replaces symbolic control with an opaque
latent action.

## The part I find elegant

The controller has to read a variable-size annotated RDF graph, and the annotations are
the whole point: the heuristics it chooses among are rankings over those annotation
values. So the encoder should be able to see them. We compare three graph encoders that
form a natural ladder. A GCN sees only topology. An R-GCN adds relation types. And a
StarE-GNN additionally embeds the qualifiers into its messages:

$$
\mathbf{g}^{(k)}_v = \sigma\!\left(\sum_{(u,r)\in\mathcal{N}(v)}
\mathbf{W}^{(k)}_{\lambda(r)}\,\phi_r\!\big(\mathbf{g}^{(k-1)}_u,\,\psi(\mathbf{g}_r,\mathbf{q}_{u,r,v})\big)\right),
$$

where $\mathbf{q}_{u,r,v}$ embeds the temporal annotations of the fact connecting $u$
and $v$. Only this last encoder is qualifier-aware, meaning it can condition on the very
values the heuristics rank over.

On top of the encoder sit three value heads, one for question answering, one for
exploration, one for forgetting. Each head pools the fact embeddings with its own
attention,

$$
\mathbf{m}^{(p)}_t = \sum_j \alpha^{(p)}_{t,j} \mathbf{W}_V \mathbf{z}_{t,j},
\qquad p \in \{\mathrm{qa}, e, f\},
$$

so the three functions can attend to different parts of the same memory, and each head
scores its own named candidates with Q-values trained by standard temporal-difference
learning. At test time, selection is greedy.

## Why I find the result interesting

At long-term memory capacity 512, the qualifier-aware StarE-GNN configuration reaches
**47.28** test QA accuracy, the best held-out result among everything we compared. The
R-GCN follows at **47.00** and the plain GCN at **46.04**. I like this ordering because
it matches the design logic: the more of the annotation structure the encoder can see,
the better it selects among heuristics defined over that structure.

The reference points matter too. The strongest fixed symbolic controls reach **45.64 to
46.28**, so learned selection buys roughly one to two points over the best fixed rule,
and every learned configuration beats every fixed one. The end-to-end neural baselines
sit at **11.40** (Transformer) and **9.40** (LSTM) on the same splits. The symbolic
memory substrate does the heavy lifting; the meta-policy refines it.

But the number is not my favorite part. My favorite part is that the controller's
behavior can be read. Because the choices are named, we can plot the Q-values of the QA
head over an episode and watch the preference shift from MRU early on, when recently
accessed facts are still likely relevant, toward MRA later, when the newest observations
carry the decisive evidence. That switching is the adaptivity we wanted, and it is
visible precisely because the actions are named heuristics rather than latent vectors.

One ablation is worth mentioning. Replacing the three per-function heads with a single
learned 27-way head over all heuristic combinations performs clearly worse on train.
Factoring the decision by function is the right structure, which I find satisfying: the
architecture mirrors how the memory operations are actually organized.

## Why this fits the broader project

If you have followed this line of work, the progression is: first, does explicit
knowledge-graph memory help at all? Then, which facts deserve to enter long-term memory?
And now, which rules should govern the memory at each moment? Storage, transfer, and now
rule selection, each isolated and studied as its own concrete problem.

To me this paper is also the clearest statement so far of why I care about
neuro-symbolic designs. It is not symbolic structure for its own sake. The result shows
that adaptivity and inspectability do not need to be traded off: the learned part makes
the agent better, and the symbolic part keeps every decision attributable to a named
rule over explicit annotations. Getting to present this at ISWC, the Semantic Web
community's home venue, feels right, because the whole construction rests on RDF
representation and annotation-compatible graph semantics.

## Links

- Project page: [RoomKG Meta-Policies](https://humem.ai/projects/roomkg-meta-policies/)
- Paper: [arXiv](https://arxiv.org/abs/2607.18368)
- GitHub: [roomkg-meta-policies](https://github.com/humemai/roomkg-meta-policies)
