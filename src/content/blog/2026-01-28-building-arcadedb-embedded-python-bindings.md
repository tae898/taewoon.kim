---
title: Building embedded Python bindings for ArcadeDB
subtitle: Why I wanted a lightweight, local-first way to use ArcadeDB from Python
cover-img: /assets/img/posts/2026-01-28/arcadedb-embedded-python-bindings.png
thumbnail-img: /assets/img/posts/2026-01-28/arcadedb-embedded-python-bindings.png
tags: [ArcadeDB, Python, open source, databases, graphs, embeddings, JVM, humemai]
author: Taewoon Kim
---

I've been spending a lot of time thinking about memory systems for AI agents. A big part
of that work is not just the model, but the storage layer underneath it. I want
something that can represent structured relationships, support fast retrieval, and still
be practical to use in day-to-day experimentation. That led me to
[ArcadeDB](https://github.com/ArcadeData/arcadedb).

The problem was that most of my experimentation happens in Python, while ArcadeDB lives
in the JVM world. I didn't want to put a server and a driver hop in the middle every
time I wanted to run a local experiment. I wanted to open a database directly from
Python, run transactions, create graph structures, build vector indexes, and close it
again. So I built embedded Python bindings for ArcadeDB.

## Why ArcadeDB caught my attention

What I like about ArcadeDB is that it doesn't force me to split my thinking across too
many systems. It gives me documents, graphs, key/value storage, full-text search, and
vector embeddings in one engine. For the kind of memory work I'm interested in, that is
very appealing.

When I think about an agent memory system, I don't see vectors and graphs as competing
representations. I see them as complementary. Vectors are great for approximate
retrieval. Graphs are great when I care about explicit relationships, provenance, and
explainability. Documents and properties are useful for everything around that.
ArcadeDB gives me all of those pieces in one place.

## Why I wanted the embedded route

For local R&D work, I usually prefer fewer moving parts.

If I can avoid spinning up another service, managing ports, and dealing with a driver
boundary, I usually will. Embedded access means Python can talk directly to the JVM
engine inside the same process. That makes local workflows feel simpler and often lower
latency.

This matters a lot when I'm iterating quickly. Sometimes I just want to create a small
database, try a schema change, load some data, test a retrieval idea, and throw it away.
An embedded setup feels much better for that kind of work than standing up a full remote
service every time.

## What I built

The project packages ArcadeDB in a way that is easy to use from Python. The wheel is
self-contained and includes:

- a lightweight JRE built with `jlink`
- the required ArcadeDB JARs
- Python bindings built on top of `JPype`

The goal was simple: make the Python experience feel natural while still exposing the
core parts of ArcadeDB that I actually use.

That includes things like:

- creating databases and schemas
- creating document, vertex, and edge types
- transactions and batch-style operations
- import/export utilities
- vector search through HNSW indexing

## A small example

Here is the kind of workflow I had in mind.

```python
import arcadedb_embedded as arcadedb

with arcadedb.create_database("./mydb") as db:
    db.schema.create_vertex_type("Person")
    db.schema.create_edge_type("KNOWS")

    with db.transaction():
        alice = db.new_vertex("Person").set("name", "Alice").save()
        bob = db.new_vertex("Person").set("name", "Bob").save()
        alice.new_edge("KNOWS", bob).save()

    result = db.query(
        "opencypher",
        "MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN a.name AS from, b.name AS to",
    )

    for row in result:
        print(row.get("from"), "->", row.get("to"))
```

That is the experience I wanted: open a database from Python, use it directly, and keep
the feedback loop tight.

## Why this matters to me

This is not just a packaging exercise for me. It is part of a broader direction I've
been pushing for a while: making memory-heavy AI systems more explicit, inspectable, and
practical.

At [HumemAI](https://humem.ai), a lot of what I care about revolves around memory,
knowledge representation, and retrieval. I don't think everything should disappear into
model weights or a black-box vector store. Sometimes you really do want explicit
structure. Sometimes you want to see the graph, inspect the edges, and understand why a
retrieval happened.

That is why a multi-model database is interesting to me, and that is why I wanted a
good embedded Python workflow around it.

## Links

- Repo: [arcadedb-embedded-python](https://github.com/humemai/arcadedb-embedded-python)
- Docs: [HumemAI ArcadeDB Docs](https://docs.humem.ai/arcadedb/)
- ArcadeDB: [ArcadeDB on GitHub](https://github.com/ArcadeData/arcadedb)

I'll probably write more about this later, especially around vectors, graph modeling,
and how I think about memory systems for agents in practice. For now, I mainly wanted to
share the project and the motivation behind it.