"""Hierarchical LDA (nested Chinese restaurant process; Blei et al., 2010), a
baseline of the TaxoGen paper."""
from __future__ import annotations

import numpy as np

from taxorec.data import Corpus
from taxorec.taxonomy.tree import ROOT, Taxonomy


def build_hlda(corpus: Corpus, max_depth: int = 3, alpha: float = 0.1, eta: float = 0.1,
               gamma: float = 0.1, iterations: int = 500, burn_in: int = 100, top_n: int = 10,
               min_docs: int = 20, max_docs: int | None = 100000, seed: int = 0) -> Taxonomy:
    """Hierarchical LDA (Blei et al., 2010) by tomotopy, on the documents'
    candidate terms (phrases included), so it chooses among the same terms as
    every other method. The root topic is the taxonomy root; each topic below
    it is a node with its top_n words, and a node's documents are those whose
    path passes through it. Topics with fewer than min_docs documents are
    dropped with their subtrees. One worker thread and a fixed seed make the
    sampler deterministic; at most max_docs documents (a seeded sample) are
    used."""
    try:
        import tomotopy as tp
    except Exception as e:  # missing package, or an installed one that fails to import
        raise ImportError(f"hLDA baseline unavailable ({type(e).__name__}: {e}); "
                          "install with `pip install tomotopy`") from e
    rows = np.arange(corpus.n_docs)
    if max_docs and corpus.n_docs > max_docs:
        rows = np.sort(np.random.default_rng(seed).choice(rows, max_docs, replace=False))
    model = tp.HLDAModel(depth=max_depth + 1, alpha=alpha, eta=eta, gamma=gamma, seed=seed)
    sub = corpus.doc_term[rows]
    used = []
    for r in range(sub.shape[0]):
        lo, hi = sub.indptr[r], sub.indptr[r + 1]
        words = [corpus.terms[j] for j, n in zip(sub.indices[lo:hi], sub.data[lo:hi]) for _ in range(int(n))]
        if words:
            model.add_doc(words)
            used.append(r)
    model.burn_in = burn_in
    model.train(iterations, workers=1)

    tax = Taxonomy("hlda", {"max_depth": max_depth, "alpha": alpha, "eta": eta, "gamma": gamma,
                            "iterations": iterations, "burn_in": burn_in, "top_n": top_n,
                            "min_docs": min_docs, "max_docs": max_docs, "seed": seed,
                            "n_docs_used": len(used)})
    tax.root.terms = list(corpus.terms)
    tax.root.n_docs = corpus.n_docs
    root_topic = next(k for k in range(model.k) if model.is_live_topic(k) and model.level(k) == 0)
    node_of = {root_topic: ROOT}
    live = [k for k in range(model.k) if model.is_live_topic(k) and model.level(k) > 0]
    for k in sorted(live, key=lambda t: (model.level(t), -model.num_docs_of_topic(t), t)):
        parent = model.parent_topic(k)
        if parent not in node_of or model.num_docs_of_topic(k) < min_docs:
            continue
        terms = [w for w, _ in model.get_topic_words(k, top_n=top_n)]
        node = tax.add_child(node_of[parent], terms, int(model.num_docs_of_topic(k)))
        node_of[k] = node.id
    return tax
