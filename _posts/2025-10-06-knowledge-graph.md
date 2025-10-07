---
layout: post
title: "What Is a Knowledge Graph? A Practical Guide Across RDF and Property Graphs"
subtitle: "Formal basics, real systems, and why industry favors property graphs while RDF remains important"
cover-img: /assets/img/posts/2025-10-07/knowledge_graphs.png
thumbnail-img: /assets/img/posts/2025-10-07/knowledge_graphs.png
tags:
  [
    knowledge-graphs,
    rdf,
    property-graphs,
    semantic-web,
    neo4j,
    jena,
    neptune,
    graphdb,
    data-engineering,
  ]
author: Taewoon Kim
mathjax: true
---

## Motivation

“Knowledge graph” gets used everywhere. This post pins down a minimal formal meaning,
relates it to a widely cited definition from Hogan et al. (2020), and compares **RDF
triple stores** with **property graph** databases. We also note platforms like **Amazon
Neptune** that support _both_ models.

---

## 1) Minimal Formal Definition + Hogan et al.

A knowledge graph can be specified as a labeled directed graph:

$$ G = (V, R, E),\quad E \subseteq V \times R \times V $$

- $$V$$: entities (nodes)
- $$R$$: relation types (edge labels)
- $$E$$: triples $$(h, r, t)$$ representing facts

This aligns with **Hogan et al. (2020)**: a (knowledge graph) KG is “a graph of data
intended to accumulate and convey knowledge of the real world,” with nodes as entities
and edges as relations, which is **model-agnostic**. They emphasize that a KG may be a
**directed edge-labelled graph, a heterogeneous graph, a property graph, and so on**.
RDF and property graphs are two most concrete ways to implement the same abstraction,
which will be discussed in depth below.

---

## 2) The Term “Knowledge Graph”

The term was popularized by industry, notably **Google’s 2012 “Knowledge Graph”**
announcement. It was not coined inside the Semantic Web community, though that community
had already standardized RDF, RDFS/OWL, and SPARQL.

**Why Google did this (product motivations):**

- **Disambiguation at web scale.** Move from keyword matching to **entity-centric**
  search (e.g., “Paris” the city vs. the person).
- **Richer answers.** Power **Knowledge Panels**, instant facts, and multi-hop result
  snippets without users clicking through.
- **Intent understanding.** Connect queries, entities, and relations to improve ranking,
  query rewriting, and “people also ask.”
- **Data unification.** Fuse signals from **Freebase** (acquired 2010), Wikipedia, CIA
  World Factbook, and publishers into a single internal graph.
- **Structured ingestion.** Leverage **schema.org** markup (launched 2011 with other
  engines) to harvest entity facts directly from websites.
- **Latency and scale.** Use internal, non-RDF infrastructure optimized for low-latency
  lookups and massive read volumes.
- **Revenue alignment.** Better entity understanding improves ad relevance and new
  surfaces (local, travel, shopping).

**Relationship to the Semantic Web:**

- Conceptually similar (entities + relations), but **not tied to RDF tooling or OWL
  reasoning**.
- A pragmatic, product-first graph with selective semantics, built for speed and
  coverage rather than formal inference.

Taken together, Google’s rollout illustrates the tension between rich semantics and
pragmatic delivery. Before choosing a data model, it’s worth grounding where teams
actually implement knowledge graphs.

---

## 3) Implementation Choices Before Picking a Model

Before comparing RDF and property graph tooling, it’s helpful to see where the
abstraction can live. The KG definition is model-agnostic; you can represent it in any
language or storage engine, and many teams prototype with lightweight structures before
committing to dedicated databases.

**In-memory Python dicts (toy):**

```python
graph = {
  "Paris": [("capital_of", "France")],
  "France": [("located_in", "Europe")],
}

for relation, target in graph.get("Paris", []):
  print("Paris", relation, target)
```

**Relational tables (SQL) with triples:**

```sql
CREATE TABLE triples (
  head TEXT,
  relation TEXT,
  tail TEXT
);

INSERT INTO triples VALUES
  ('Paris', 'capital_of', 'France'),
  ('France', 'located_in', 'Europe');

SELECT relation, tail
FROM triples
WHERE head = 'Paris';
```

This works, but **multi-hop graph queries** tend to require **multiple JOINs**, which
can be costly. For graph-centric workloads, native graph stores usually perform better
than general-purpose SQL systems.

In practice, large KGs live in **persistent databases** with **declarative graph
queries**—most often RDF triple stores or property graph systems. The next two sections
unpack why those models emerged instead of reinventing bespoke infrastructure for every
project.

---

## 4) RDF: Semantics-First Modeling

With those implementation choices in mind, the first production-grade family to consider
is **RDF**, which bakes semantics into the core model.

**Data model:** Triples $$(s,p,o)$$ with IRIs for global identity, literals, and blank
nodes.  
**Query:** SPARQL.  
**Schema and reasoning:** RDFS/OWL description logics for classes, properties, and
axioms.

RDF Schema (RDFS) and the Web Ontology Language (OWL) let you publish an **ontology**:
you declare class hierarchies, property domains/ranges, cardinality constraints, and
logical characteristics such as _transitive_, _symmetric_, or _inverse_ relations. A
description-logic reasoner consumes those axioms to check consistency and **derive new
facts**. This is the “logic layer” that classic AI researchers love—knowledge isn’t just
stored, it can be entailed.

**Ontology + inference toy example:**

```turtle
@prefix ex: <http://example.org/> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

ex:Capital a rdfs:Class ;
           rdfs:subClassOf ex:Settlement .

ex:capitalOf a rdf:Property ;
             rdfs:domain ex:Capital ;
             rdfs:range ex:Country .

ex:City rdfs:subClassOf ex:Settlement .

ex:Paris  a ex:City ;
          ex:capitalOf ex:France .

ex:France a ex:Country .
```

**Query (SPARQL):**

```sparql
PREFIX ex: <http://example.org/>
SELECT ?capital
WHERE {
  ?capital a ex:Capital ;
           ex:capitalOf ex:France .
}
```

**Query breakdown.** `SELECT ?capital` asks for every resource that satisfies the
pattern in the `WHERE` block. The first triple pattern `?capital a ex:Capital` requires
each result to be typed as a capital (the keyword `a` is shorthand for `rdf:type`). The
semicolon keeps the same subject and adds a second condition:
`?capital ex:capitalOf ex:France`. In short, we are searching for “anything that is
declared to be a capital _and_ is the capital of France.” Without reasoning the pattern
has no match; with reasoning the inferred type makes `ex:Paris` qualify.

**Inference outcome.** The data explicitly states that `ex:Paris ex:capitalOf ex:France`
and that `ex:Paris` is a `ex:City`, but it never asserts that Paris belongs to the class
`ex:Capital`. Because the ontology declares the **domain** of `ex:capitalOf` to be
`ex:Capital`, an RDFS/OWL reasoner infers the missing triple `ex:Paris a ex:Capital`
(and, via subclass axioms, `ex:Settlement`). The SPARQL query therefore returns
`ex:Paris` once reasoning is enabled. This ability to derive implicit knowledge—and to
validate that an ontology is logically consistent—is a core differentiator of RDF
systems.

Historically, this style of symbolic, logic-driven modeling powered much of “classic”
AI before the deep-learning boom: knowledge bases, rule engines, and deterministic
reasoners flourished because they offered transparency and provable guarantees. The
industry’s shift toward statistical learning has pulled attention away from heavy
ontologies, yet RDF’s semantics remain valuable wherever governance, explainability,
or cross-organization data sharing outweigh raw model velocity.

From today’s data-centric AI lens, many teams expect models to _learn_ reasoning
patterns rather than encode them manually. Large language models, graph neural networks,
and differentiable reasoners try to induce soft logical structure from examples,
because hand-authoring ontological rules hits diminishing returns once domains get messy
or rapidly evolving. RDF still gives you a way to declare the guard rails explicitly,
but modern ML groups often reserve that effort for high-impact constraints while letting
data-driven models discover the rest.

### Strengths (RDF)

- Global identifiers and formal semantics by design
- Interoperability across datasets and organizations
- Mature standards (RDF 1.1, SPARQL 1.1, JSON-LD)
- Automated reasoning over ontologies (classification, entailment, validation)

### Trade-offs (RDF)

- Ontology/reasoning add conceptual and operational overhead
- Join-heavy execution for multi-hop queries can be slower at scale
- Developer ergonomics can feel heavier for fast-moving product teams
- Most RDF stores still evaluate SPARQL by binding variables via joins—whether triples
  sit in relational tables (Virtuoso, Jena TDB), key–value or bitmap indexes
  (Blazegraph, RDF4J NativeStore, HDT), or hybrid adjacency layouts (gStore, AnzoGraph,
  parts of Neptune)—so traversals step through joins instead of pointer-chasing.

---

## 5) Property Graphs: Pragmatics-First Modeling

If RDF prioritizes semantics, **property graphs** lean into pragmatic modeling and
traversal speed.

**Data model:** Nodes and edges with **key–value properties**; labels for types.  
**Query:** Cypher (Neo4j, openCypher) or Gremlin (Apache TinkerPop), also GSQL
(TigerGraph).

**Example (Cypher from Neo4j):**

```cypher
CREATE (p:City {name: "Paris"})
CREATE (f:Country {name: "France"})
CREATE (p)-[:CAPITAL_OF]->(f);
```

**Query:**

```cypher
MATCH (c:City)-[:CAPITAL_OF]->(n:Country)
RETURN c.name, n.name;
```

### Property graphs in the AI era

Leading property-graph vendors now pitch themselves as AI infrastructure: Neo4j,
TigerGraph, and others bundle graph-native vector search, agent tooling, and GNN-ready
pipelines so teams can mix relational structure with learned embeddings inside one
stack.

Why that resonates with modern ML teams:

- **Adjacency-first access.** Traversals follow stored neighbor lists, so iterative
  sampling or message passing for GNNs is a pointer hop, not a JOIN.
- **Flexible attributes.** Nodes and edges accept arbitrary key–value metadata, making
  it trivial to attach embeddings, feature vectors, or model scores inline.
- **Familiar tooling.** Cypher/Gremlin APIs plug into Pythonic dataflows and feature
  engineering scripts without requiring semantic-web stacks.
- **Hybrid search.** Built-in vector indexes let symbolic graph context and neural
  similarity queries live side by side.
- **Fast iteration.** With no ontology reasoning to maintain, loading and retraining
  cycles stay tight for data-centric experimentation.

RDF stores still matter for knowledge-rich reasoning, but property graphs align closely
with the speed, flexibility, and embedding-centric workflows dominating 2025 AI
projects.

### Strengths (Property Graphs)

- Simple traversals and developer-friendly syntax  
- Properties directly on nodes/edges without ontology boilerplate  
- Strong fit for operational graph workloads and ML (embeddings, GNNs, hybrid RAG)
- Traversals follow explicit adjacency lists (pointer-based lookups with good cache
  locality), so multi-hop paths avoid relational joins and stay low-latency.

### Trade-offs (Property Graphs)

- No built-in formal semantics or global identity unless you add them  
- Interoperability depends on your ID conventions

### Adoption snapshot (Oct 2025, DB-Engines)

- **Neo4j** ~ #20 overall  
- **Apache Jena – TDB** (RDF) ~ #102 overall  

This reflects higher _practical adoption_ of property-graph systems, not that RDF is
obsolete.

---

## 6) Systems That Support Both Models

**Amazon Neptune** supports **RDF (SPARQL)** and **property graphs (Gremlin,
openCypher)** in one managed service. In practice you pick a model per dataset/endpoint;
the dual support lets you align model choice with workload needs.

---

## 7) When to Prefer Which

### Prefer RDF when

- You need **interoperability** with external datasets (IRIs, shared vocabularies)
- You need **reasoning** or schema validation via RDFS/OWL
- You publish or consume **Linked Data** / JSON-LD

### Prefer Property Graphs when

- You need **agile modeling** with product teams
- You run traversal-heavy operational queries
- You want straightforward **ML pipelines** (embeddings, GNNs, RAG)

---

## 8) Query Languages in One Glance

- **SPARQL** (RDF): declarative pattern matching with federation and reasoning hooks
- **Cypher** (PG): declarative pattern matching with concise path syntax
- **Gremlin** (PG): traversal DSL that composes well inside applications

---

## 9) Conclusion

- A **knowledge graph** is a simple formal object: a labeled directed graph with
  semantic meaning.
- **Hogan et al.**’s definition is intentionally **model-agnostic** and matches this
  view.
- You can implement a KG in many ways; at scale, teams adopt **graph databases** and
  **graph query languages**.
- **RDF** remains important where semantics and interoperability matter.
- **Property graphs** see broader industry adoption today due to developer ergonomics
  and traversal performance.
- Pick the model that fits your interoperability, performance, and governance
  requirements. You can bridge both when needed.

---

## 📚 References

- [Hogan, A. et al. (2020). **Knowledge Graphs.**](https://arxiv.org/abs/2003.02320)
- [W3C. **RDF 1.1 Concepts and Abstract Syntax.**](https://www.w3.org/TR/rdf11-concepts/)
- [W3C. **SPARQL 1.1 Query Language.**](https://www.w3.org/TR/sparql11-query/)
- [Neo4j. **Cypher Query Language Manual.**](https://neo4j.com/docs/cypher-manual/)
- [Apache Jena. **TDB Storage and Query.**](https://jena.apache.org/documentation/tdb/)
- [DB-Engines. **DB-Engines Ranking.**](https://db-engines.com/en/ranking)
- [Amazon Web Services. **Neptune Data Models: RDF and Property Graph.**](https://docs.aws.amazon.com/neptune/latest/userguide/feature-overview-data-model.html)
- [Google. **Introducing the Knowledge Graph (2012).**](https://blog.google/products/search/introducing-knowledge-graph-things-not/)
