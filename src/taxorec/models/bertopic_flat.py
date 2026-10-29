"""Flat BERTopic topics as a depth-1 taxonomy (optional baseline)."""
from __future__ import annotations

import numpy as np

from taxorec.data import Corpus
from taxorec.taxonomy.components import Encoder
from taxorec.taxonomy.tree import ROOT, Taxonomy


def build_bertopic(corpus: Corpus, encoder: Encoder, top_n: int = 30, seed: int = 0,
                   max_docs: int | None = 100000, **kwargs) -> Taxonomy:
    """Flat BERTopic topics as a depth-1 taxonomy. Documents are embedded with
    the same encoder as the other methods, and topic words are counted over
    the shared candidate vocabulary (phrases included), so every method
    chooses among the same terms. At most max_docs documents (a seeded
    sample) are clustered, which keeps UMAP within a 16 GB machine."""
    try:
        from bertopic import BERTopic
        from sklearn.feature_extraction.text import CountVectorizer
        from umap import UMAP
    except Exception as e:  # missing package, or an installed one that fails to import
        raise ImportError(f"BERTopic baseline unavailable ({type(e).__name__}: {e}); "
                          "install with `pip install bertopic -c requirements.txt`") from e
    rows = np.arange(corpus.n_docs)
    if max_docs and corpus.n_docs > max_docs:
        rows = np.sort(np.random.default_rng(seed).choice(rows, max_docs, replace=False))
    docs = corpus.get_docs(rows)
    embeddings = encoder.encode([" ".join(t.replace("_", " ") for t in d) for d in docs])
    # only terms that occur in the sample: a zero document frequency makes
    # BERTopic's c-TF-IDF divide by zero
    present = np.asarray(corpus.doc_term[rows].sum(axis=0)).ravel() > 0
    vocabulary = [t for t, ok in zip(corpus.terms, present) if ok]
    vectorizer = CountVectorizer(vocabulary=vocabulary, tokenizer=str.split, lowercase=False,
                                 token_pattern=None)
    umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=seed)
    # language="english" makes BERTopic strip every non-alphanumeric character
    # (so 'long_sleeve' becomes 'longsleeve' and matches no vocabulary term);
    # the language only picks a default embedding model, unused here since
    # embeddings are passed in.
    model = BERTopic(language=kwargs.pop("language", "multilingual"), umap_model=umap_model,
                     vectorizer_model=vectorizer, top_n_words=top_n, **kwargs)
    model.fit([" ".join(d) for d in docs], embeddings=embeddings)
    tax = Taxonomy("bertopic", {"top_n": top_n, "seed": seed, "max_docs": max_docs,
                                "encoder": encoder.name, **kwargs})
    tax.root.terms = list(corpus.terms)
    tax.root.n_docs = corpus.n_docs
    placed = set()
    for topic_id in sorted(t for t in model.get_topics() if t != -1):
        words = [w.replace(" ", "_") for w, _ in model.get_topic(topic_id)]
        terms = [w for w in words if w in corpus.term_index and w not in placed]
        if terms:
            placed.update(terms)
            tax.add_child(ROOT, terms, int(len(corpus.docs_with_any(corpus.ids(terms)))))
    tax.root.general_terms = [t for t in corpus.terms if t not in placed]
    return tax
