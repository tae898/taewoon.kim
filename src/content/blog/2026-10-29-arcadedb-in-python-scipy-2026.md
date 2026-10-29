---
title: My ArcadeDB in Python paper is in the SciPy Proceedings
subtitle: One embedded database for documents, a graph and vectors
cover-img: /assets/img/posts/2026-10-29/arcadedb-python-cover.png
thumbnail-img: /assets/img/posts/2026-10-29/arcadedb-python-cover.png
tags: [ArcadeDB, Python, databases, open source, SciPy]
author: Taewoon Kim
---

My paper, *ArcadeDB in Python: An In-Process Multi-Model Database*, was accepted at SciPy
2026 and is now in the [SciPy
Proceedings](https://proceedings.scipy.org/articles/fwvt2525) (DOI
[10.25080/fwvt2525](https://doi.org/10.25080/fwvt2525)). I wrote it alone. The [project
page](https://humem.ai/research/arcadedb-python) on humem.ai has the method and the
benchmark tables.

## What it is

Some Python programs need transactions over documents, graph traversals, and vector
search over the same data. The usual answer is three systems, each with its own copy of the
data, and application code that keeps the copies in sync.

ArcadeDB is a multi-model database written in Java. The package runs it inside the Python
process, with a Java runtime bundled in the wheel, so one database does all three and you
do not need Java installed. You query documents with SQL, the graph with openCypher, and
vectors through an HNSW-style index.

The paper shows a workflow that chains a vector search, a SQL filter, and a graph traversal
over one dataset. It also benchmarks the package against SQLite, DuckDB, LadybugDB, and
Chroma on a Q&A corpus. Each specialist wins on its own model. ArcadeDB is competitive on
all three, and what it costs is memory and loading time.

The ArcadeDB project wrote the engine. I wrote the Python bindings and the packaging. The
paper calls the package `arcadedb-embedded`; from the 26.11.1 release it is named
`arcadedb`.

## Links

- Paper: [SciPy Proceedings](https://proceedings.scipy.org/articles/fwvt2525)
- DOI: [10.25080/fwvt2525](https://doi.org/10.25080/fwvt2525)
- Project page: [ArcadeDB in Python](https://humem.ai/research/arcadedb-python)
- Python docs: [docs.arcadedb.com](https://docs.arcadedb.com/arcadedb-python/)
- GitHub: [ArcadeData/arcadedb-python](https://github.com/ArcadeData/arcadedb-python)
- Benchmark suite and results:
  [humemai/arcadedb-embedded-python](https://github.com/humemai/arcadedb-embedded-python/tree/main/benchmarks/python-bindings)

## Cite

```bibtex
@article{kim2026,
  author = {Kim, Taewoon},
  title = {ArcadeDB in Python: An In-Process Multi-Model Database},
  journal = {Python in Science Conference},
  year = {2026},
  doi = {10.25080/fwvt2525},
  url = {https://doi.org/10.25080/fwvt2525}
}
```
