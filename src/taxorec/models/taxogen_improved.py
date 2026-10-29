"""TaxoGen-Improved: the recursive (encoder x clusterer x relation scorer)
taxonomy builder.

The proposed method is the configuration `taxonomy.methods.improved`: the
node's documents clustered by SBERT (UMAP + HDBSCAN), anchor terms chosen by
CaseOLAP, centred context-SBERT term vectors and BM25-cosine relation scoring
(Eq. 1). Every other combination of the same parts is an ablation or a
baseline: `sbert_kmeans` (SBERT term vectors + spherical k-means) and `hclus`
(word2vec term vectors + Ward agglomerative clustering).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Callable

import numpy as np
import scipy.sparse as sp

from taxorec.data import Corpus
from taxorec.taxonomy.components import (
    Encoder, RelationScorer, adaptive_min_cluster_size, agglomerative_labels, caseolap_scores,
    cell_statistics, hdbscan_labels, kl_representativeness, rank_terms, recenter,
    spherical_kmeans, umap_reduce,
)
from taxorec.taxonomy.tree import ROOT, Taxonomy


def _from_dict(cls, d: dict):
    known = {f.name for f in fields(cls)}
    unknown = set(d) - known
    if unknown:
        raise ValueError(f"unknown {cls.__name__} keys: {sorted(unknown)}")
    return cls(**d)


def _mean_vector(emb: np.ndarray, ids) -> np.ndarray:
    v = emb[np.asarray(ids, dtype=np.int64)].mean(axis=0)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _invalid_issues(tax: Taxonomy) -> list[str]:
    return [msg for sev, msg in tax.validate() if sev == "error"]


def relation_summary(tax: Taxonomy) -> dict:
    """What the relation scorer did: candidate children it scored, how many it
    rejected (s_rel < tau), and whether the rejected ones were simply the
    small ones (median sizes, Spearman correlation of s_rel with size)."""
    from scipy.stats import spearmanr
    scored = [c for n in tax.nodes.values() for c in n.scores.get("candidates", []) if c.get("s_rel") is not None]
    rejected = [c for c in scored if c["s_rel"] < tax.params.get("tau", 0.0)]
    out = {"children_scored": len(scored), "children_rejected": len(rejected)}
    if len(scored) >= 3:
        sizes = [c["n_docs"] for c in scored]
        rho = spearmanr(sizes, [c["s_rel"] for c in scored]).correlation
        out["s_rel_size_spearman"] = float(rho) if np.isfinite(rho) else None
        out["median_docs_rejected"] = float(np.median([c["n_docs"] for c in rejected])) if rejected else None
        out["median_docs_kept"] = float(np.median([c["n_docs"] for c in scored if c not in rejected]))
    return out


# ---------------------------------------------------------------- generic builder

@dataclass
class BuilderConfig:
    clusterer: str = "hdbscan"          # hdbscan | kmeans | agglomerative
    split: str = "terms"                # terms: cluster term vectors | docs: cluster the node's documents
    split_max_docs: int = 20000         # docs: documents sampled per node for clustering
    split_min_kl: float = 0.25          # docs: a term joins a child only if its CaseOLAP scores are this far from uniform
    split_min_df: int = 0               # docs: a term joins a child only if it occurs in this many of its sampled documents
    split_min_df_frac: float = 0.0      # docs: ... and in at least this fraction of them (scales with the cluster)
    split_attach_min_sim: float | None = None  # docs: rare terms join the child with the closest anchor centroid (cosine >= this)
    center: bool = False                # subtract the mean term vector before any cosine (context vectors share one direction)
    reduce: str = "none"                # none | umap (before clustering, at every node)
    umap_dim: int = 5
    umap_neighbors: int = 15
    k: int = 5                          # kmeans and agglomerative
    hdb_divisor: float = 15.0           # min_cluster_size = pool / divisor, clamped
    hdb_floor: int = 15
    hdb_ceiling: int = 2000
    hdb_min_samples: int = 3
    hdb_selection: str = "leaf"
    relation: str = "bm25cos"           # none | cos | bm25 | bm25cos
    w: float = 0.5
    tau: float = 0.3
    profile_k: int = 10
    # the parent's profile in Eq. 1: its top profile_k terms ("top"), or its
    # general terms, those that no child takes ("general"; docs split: spread
    # evenly over the clusters, terms split: noise), falling back to "top"
    # when the node has none
    relation_profile: str = "top"
    score_root_children: bool = False   # the root has no topic to be related to
    local: str = "recenter"             # none | recenter (terms split: remove the node's mean vector)
    max_depth: int = 3
    min_terms_to_split: int = 30
    min_docs_to_split: int = 20
    min_child_terms: int = 5
    min_children: int = 2
    seed: int = 0

    @classmethod
    def from_dict(cls, d: dict) -> "BuilderConfig":
        return _from_dict(cls, d)


class TaxonomyBuilder:
    def __init__(self, cfg: BuilderConfig, corpus: Corpus, encoder: Encoder,
                 method: str = "improved", log: Callable[[str], None] = print):
        self.cfg, self.corpus, self.encoder, self.method, self.log = cfg, corpus, encoder, method, log
        self.scorer = RelationScorer(cfg.relation, cfg.w, cfg.tau, cfg.profile_k)

    def cluster(self, x: np.ndarray) -> np.ndarray:
        c = self.cfg
        if c.clusterer not in ("kmeans", "hdbscan", "agglomerative"):
            raise ValueError(f"unknown clusterer {c.clusterer!r}")
        if c.reduce not in ("none", "umap"):
            raise ValueError(f"unknown reduce {c.reduce!r}")
        unit = c.reduce == "none"
        if not unit:
            x = umap_reduce(x, c.umap_dim, c.umap_neighbors, seed=c.seed)
        if c.clusterer in ("kmeans", "agglomerative"):
            if len(x) <= c.k:
                return np.full(len(x), -1)
            return (spherical_kmeans(x, c.k, seed=c.seed, normalize=unit) if c.clusterer == "kmeans"
                    else agglomerative_labels(x, c.k))
        mcs = adaptive_min_cluster_size(len(x), c.hdb_divisor, c.hdb_floor, c.hdb_ceiling)
        return hdbscan_labels(x, mcs, c.hdb_min_samples, c.hdb_selection, normalize=unit)

    def build(self) -> Taxonomy:
        corpus = self.corpus
        emb = self.encoder.encode_terms(corpus.terms)
        pool = np.where(np.linalg.norm(emb, axis=1) > 0)[0]
        if self.cfg.center:
            # averaged context vectors all point one way (random term pairs have
            # cosine ~0.6 on reviews), which makes every cosine threshold pass
            emb = emb.copy()
            emb[pool] = recenter(emb[pool])
        self.log(f"[{self.method}] {len(pool)}/{corpus.n_terms} terms have a representation")
        tax = Taxonomy(self.method, {**asdict(self.cfg), "encoder": self.encoder.name})
        all_docs = np.arange(corpus.n_docs)
        ranked = rank_terms(corpus, pool, all_docs)
        tax.root.terms = [corpus.terms[i] for i in ranked]
        tax.root.n_docs = corpus.n_docs
        self._expand(tax, ROOT, ranked, emb, all_docs, level=0)
        problems = _invalid_issues(tax)
        if problems:
            raise RuntimeError(f"invalid taxonomy: {problems[:5]}")
        self.log(f"[{self.method}] done: {tax.stats()}")
        return tax

    def _expand(self, tax: Taxonomy, node_id: str, pool_ranked: np.ndarray, emb: np.ndarray,
                docs: np.ndarray, level: int) -> None:
        c, corpus = self.cfg, self.corpus
        node = tax.nodes[node_id]
        node.general_terms = list(node.terms)  # a leaf keeps all of its terms
        if level >= c.max_depth or len(pool_ranked) < c.min_terms_to_split or len(docs) < c.min_docs_to_split:
            return
        pool = np.sort(pool_ranked)
        if c.split == "docs":
            groups, child_docs, n_noise, general = self._split_docs(pool, emb, docs, level)
        elif c.split == "terms":
            groups, child_docs, n_noise, general = self._split_terms(pool, emb, docs, level)
        else:
            raise ValueError(f"unknown split {c.split!r}")
        if len(groups) < c.min_children:
            return  # one (or no) dense group: the node is a leaf, never a copy of itself
        ranked = [rank_terms(corpus, g, child_docs[i], [d for j, d in enumerate(child_docs) if j != i])
                  for i, g in enumerate(groups)]

        if level == 0 and not c.score_root_children:
            decisions = [{"s_rel": None, "accept": True} for _ in groups]
        else:
            profile = self._parent_profile(pool_ranked, general)
            parent_vec = _mean_vector(emb, profile[:c.profile_k])
            decisions = self.scorer.score(
                [corpus.terms[i] for i in profile], parent_vec,
                [[corpus.terms[i] for i in r] for r in ranked],
                [_mean_vector(emb, r[:c.profile_k]) for r in ranked])
        accepted = [i for i, d in enumerate(decisions) if d["accept"] and len(child_docs[i]) > 0]
        # every candidate child with its relation score, kept or not (relation_summary)
        node.scores["candidates"] = [{"n_terms": int(len(g)), "n_docs": int(len(child_docs[i])),
                                      "kept": i in accepted and len(accepted) >= c.min_children,
                                      **{k: v for k, v in decisions[i].items() if k != "accept"}}
                                     for i, g in enumerate(groups)]
        if len(accepted) < c.min_children:
            return
        placed = set(np.concatenate([groups[i] for i in accepted]).tolist())
        node.general_terms = [t for t in node.terms if corpus.term_index[t] not in placed]
        self.log(f"[{self.method}] {node_id} level={level} pool={len(pool)} clusters={len(groups)} "
                 f"noise={n_noise} accepted={len(accepted)}")

        order = sorted(accepted, key=lambda i: (-len(groups[i]), int(ranked[i][0])))
        for i in order:
            child = tax.add_child(node_id, [corpus.terms[j] for j in ranked[i]], int(len(child_docs[i])),
                                  {k: v for k, v in decisions[i].items() if k != "accept"})
            self._expand(tax, child.id, ranked[i], emb, child_docs[i], level + 1)

    def _parent_profile(self, pool_ranked: np.ndarray, general: np.ndarray) -> np.ndarray:
        """The parent's terms used in Eq. 1, in the node's ranking."""
        if self.cfg.relation_profile == "top":
            return pool_ranked
        if self.cfg.relation_profile != "general":
            raise ValueError(f"unknown relation_profile {self.cfg.relation_profile!r}")
        own = pool_ranked[np.isin(pool_ranked, general)]
        return own if len(own) else pool_ranked

    def _split_terms(self, pool: np.ndarray, emb: np.ndarray, docs: np.ndarray, level: int):
        """Children = dense groups of term vectors; a child's documents contain any of its terms."""
        c, corpus = self.cfg, self.corpus
        x = emb[pool]
        if c.local == "recenter" and level > 0:
            x = recenter(x)
        labels = self.cluster(x)
        groups = [pool[labels == lab] for lab in sorted(set(labels.tolist()) - {-1})]
        groups = [g for g in groups if len(g) >= c.min_child_terms]
        child_docs = [np.intersect1d(corpus.docs_with_any(g), docs, assume_unique=True) for g in groups]
        return groups, child_docs, int((labels == -1).sum()), pool[labels == -1]

    def _split_docs(self, pool: np.ndarray, emb: np.ndarray, docs: np.ndarray, level: int):
        """Children = dense groups of the node's documents (a sample of them,
        embedded by the encoder's base model). Embedding single terms groups
        synonyms and spelling variants (cheap/cheapy/cheep), which never
        co-occur; document groups are what the reviews (titles) talk about.
        A term joins the group where its CaseOLAP score is highest when its
        scores are far enough from uniform (KL), else it stays at the parent;
        every document then goes to the group whose terms carry the most idf
        mass in it, as in TaxoGen."""
        c, corpus = self.cfg, self.corpus
        if not hasattr(self.encoder, "encode_docs"):
            raise ValueError("split: docs needs a context encoder (it embeds the node's documents)")
        rng = np.random.default_rng((c.seed, level, len(docs)))
        sample = np.sort(rng.choice(docs, min(len(docs), c.split_max_docs), replace=False))
        labels = self.cluster(self.encoder.encode_docs(sample))
        labs = sorted(set(labels.tolist()) - {-1})
        n_noise = int((labels == -1).sum())
        if len(labs) < c.min_children:
            return [], [], n_noise, pool[:0]
        tf, df = cell_statistics(corpus, [sample[labels == lab] for lab in labs], pool)
        scores = caseolap_scores(tf, df)
        kl = kl_representativeness(scores)
        keep = (scores.max(axis=0) > 0) & (kl > c.split_min_kl)
        best = scores.argmax(axis=0)
        if c.split_min_df > 0 or c.split_min_df_frac > 0:
            # a term seen in a handful of documents cannot describe a subtopic;
            # the fraction keeps the bar proportional to the cluster's size
            sizes = np.array([(labels == lab).sum() for lab in labs])
            need = np.maximum(c.split_min_df, c.split_min_df_frac * sizes)
            keep &= df[best, np.arange(len(pool))] >= need[best]
        groups = [pool[(best == i) & keep] for i in range(len(labs))]
        groups = [g for g in groups if len(g) >= c.min_child_terms]
        if len(groups) < c.min_children:
            return [], [], n_noise, pool[:0]
        # seen often enough but spread evenly over the children (low KL): the
        # term describes the parent and stays there
        general = (kl <= c.split_min_kl) & (df.sum(axis=0) >= max(c.split_min_df, 1))
        anchors = groups                 # the anchors decide which documents go where
        terms = np.concatenate(anchors)
        if c.split_attach_min_sim is not None:
            # long-tail terms are too rare for document statistics; they join
            # the child whose anchors they are closest to in the encoder space;
            # general terms stay at the parent
            rest = np.setdiff1d(pool[~general], terms)
            rest = rest[np.linalg.norm(emb[rest], axis=1) > 0]
            if len(rest):
                cent = np.stack([_mean_vector(emb, g) for g in groups])
                sims = emb[rest] @ cent.T
                to, ok = sims.argmax(axis=1), sims.max(axis=1) >= c.split_attach_min_sim
                groups = [np.sort(np.concatenate([g, rest[ok & (to == i)]])) for i, g in enumerate(groups)]
        idf = np.log(1.0 + corpus.n_docs / np.maximum(corpus.df[terms], 1))
        weights = sp.csr_matrix((idf, (np.arange(len(terms)), np.repeat(np.arange(len(anchors)),
                                                                          [len(g) for g in anchors]))),
                                shape=(len(terms), len(anchors)))
        mass = (corpus.doc_term[docs][:, terms].astype(np.float64) @ weights).toarray()
        has = mass.max(axis=1) > 0
        best_doc = mass.argmax(axis=1)
        child_docs = [docs[has & (best_doc == i)] for i in range(len(groups))]
        return groups, child_docs, n_noise, pool[general]
