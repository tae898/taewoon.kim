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

The project packages ArcadeDB in a way that is easy to use from Python. But the current
direction is not to recreate the whole database surface as a Python object API. The
examples now lean much more on ArcadeDB's own DSLs from Python, especially SQL,
SQL MATCH, and OpenCypher.

The wheel is self-contained and includes:

- a lightweight JRE built with `jlink`
- the required ArcadeDB JARs
- Python bindings built on top of `JPype`
- prebuilt platform-specific wheels for Linux x86_64, Linux ARM64, macOS ARM64, and Windows x86_64

The packaging story has also gotten much better. The wheel is now about `74MB`
compressed, with some variation by platform and version, which is much better than the
older numbers I had in mind when I first drafted this post.

The goal was simple: make Python a good place to drive ArcadeDB without pretending that
Python needs its own parallel query language.

That includes things like:

- creating databases and schemas through SQL
- driving graph workflows through SQL and OpenCypher
- transactions and batch-style operations
- import/export utilities
- vector search through `LSM_VECTOR` indexes and `vectorNeighbors(...)`

## A couple of small examples

Here is the kind of workflow I had in mind.

### OpenCypher

```python
import arcadedb_embedded as arcadedb

with arcadedb.create_database("./mydb") as db:
    db.command("sql", "CREATE VERTEX TYPE Person")
    db.command("sql", "CREATE PROPERTY Person.name STRING")
    db.command("sql", "CREATE EDGE TYPE KNOWS")

    with db.transaction():
        db.command("sql", "INSERT INTO Person SET name = ?", "Alice")
        db.command("sql", "INSERT INTO Person SET name = ?", "Bob")
        db.command(
            "sql",
            "CREATE EDGE KNOWS FROM (SELECT FROM Person WHERE name = ?) TO (SELECT FROM Person WHERE name = ?)",
            "Alice",
            "Bob",
        )

    result = db.query(
        "opencypher",
        "MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN a.name AS from, b.name AS to",
    )

    for row in result:
        print(row.get("from"), "->", row.get("to"))
```

### Vector search

```python
import arcadedb_embedded as arcadedb

with arcadedb.create_database("./mydb") as db:
    db.command("sql", "CREATE VERTEX TYPE Doc")
    db.command("sql", "CREATE PROPERTY Doc.name STRING")
    db.command("sql", "CREATE PROPERTY Doc.embedding ARRAY_OF_FLOATS")
    db.command(
        "sql",
        """
        CREATE INDEX ON Doc (embedding)
        LSM_VECTOR
        METADATA {"dimensions": 4, "similarity": "COSINE"}
        """,
    )

    with db.transaction():
        db.command(
            "sql",
            "INSERT INTO Doc SET name = :name, embedding = :embedding",
            {"name": "Apple", "embedding": arcadedb.to_java_float_array([1.0, 0.0, 0.0, 0.0])},
        )
        db.command(
            "sql",
            "INSERT INTO Doc SET name = :name, embedding = :embedding",
            {"name": "Banana", "embedding": arcadedb.to_java_float_array([0.9, 0.1, 0.0, 0.0])},
        )

    result = db.query(
        "sql",
        "SELECT name, distance FROM (SELECT expand(vectorNeighbors(?, ?, ?))) ORDER BY distance",
        "Doc[embedding]",
        arcadedb.to_java_float_array([0.95, 0.05, 0.0, 0.0]),
        2,
    )

    for row in result:
        print(row.get("name"), row.get("distance"))
```

That is the experience I wanted: open a database from Python, drive it mostly with
SQL/OpenCypher, use vector search when I need it, and keep the feedback loop tight.

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

- GitHub: [arcadedb-embedded-python](https://github.com/humemai/arcadedb-embedded-python)
- Docs: [HumemAI ArcadeDB Docs](https://docs.humem.ai/arcadedb/)
- ArcadeDB: [ArcadeDB on GitHub](https://github.com/ArcadeData/arcadedb)
