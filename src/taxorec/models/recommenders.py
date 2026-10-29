"""Recommenders, each scoring every catalog item for a batch of users:
interactions only (Popularity, ItemKNN, RP3beta, BPR-MF), content (ContentKNN:
SBERT on product metadata), taxonomy (TaxonomyKNN: shared taxonomy path), and
Hybrid, a weighted fusion of any of them (Sem+Tax, Sem+CF, Sem+Tax+CF, ...)."""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp


class Recommender:
    name = "base"

    def fit(self, history: sp.csr_matrix) -> "Recommender":
        self.history = history.tocsr()
        return self

    def score(self, users: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class Popularity(Recommender):
    name = "popularity"

    def fit(self, history):
        super().fit(history)
        self.counts = np.asarray(history.sum(axis=0), dtype=np.float32).ravel()
        return self

    def score(self, users):
        return np.tile(self.counts, (len(users), 1))


class ItemKNN(Recommender):
    """Item-based CF: s(u, i) = sum_{j in H_u} sim(i, j), cosine co-occurrence."""
    name = "itemknn"

    def __init__(self, shrink: float = 0.0, topk: int | None = None):
        self.shrink, self.topk = shrink, topk

    def fit(self, history):
        super().fit(history)
        x = (history > 0).astype(np.float32).tocsc()
        co = _without_diagonal(x.T @ x).tocoo()
        n = np.asarray(x.sum(axis=0)).ravel()
        vals = co.data / (np.sqrt(n[co.row] * n[co.col]) + self.shrink)
        sim = sp.csr_matrix((vals.astype(np.float32), (co.row, co.col)), shape=co.shape)
        if self.topk:
            sim = _keep_topk_rows(sim, self.topk)
        self.sim = sim
        return self

    def score(self, users):
        return np.asarray((self.history[users] @ self.sim).todense(), dtype=np.float32)


class RP3beta(Recommender):
    """Three-step random walk item -> user -> item with transition probabilities
    raised to alpha and the arrival item's popularity penalised by its degree
    to the power beta (Paudel et al., 2016; among the strongest simple CF
    baselines in Dacrema et al., RecSys 2019): s(u, i) = sum_{j in H_u} W_ji."""
    name = "rp3beta"

    def __init__(self, alpha: float = 1.0, beta: float = 0.5, topk: int | None = 200):
        self.alpha, self.beta, self.topk = alpha, beta, topk

    def fit(self, history):
        super().fit(history)
        x = (history > 0).astype(np.float32).tocsr()
        user_deg = np.asarray(x.sum(axis=1)).ravel()
        item_deg = np.asarray(x.sum(axis=0)).ravel()
        p_ui = sp.diags(1.0 / np.maximum(user_deg, 1)) @ x            # user -> item
        p_iu = sp.diags(1.0 / np.maximum(item_deg, 1)) @ x.T.tocsr()  # item -> user
        if self.alpha != 1.0:
            p_ui, p_iu = p_ui.power(self.alpha), p_iu.power(self.alpha)
        w = (p_iu @ p_ui).tocsr()
        w = _without_diagonal(w @ sp.diags(np.power(np.maximum(item_deg, 1), -self.beta))).astype(np.float32)
        self.sim = _keep_topk_rows(w, self.topk) if self.topk else w
        return self

    def score(self, users):
        return np.asarray((self.history[users] @ self.sim).todense(), dtype=np.float32)


def _without_diagonal(m: sp.spmatrix) -> sp.csr_matrix:
    """m with its diagonal (an item's similarity to itself) removed."""
    m = m.tocoo()
    off = m.row != m.col
    return sp.csr_matrix((m.data[off], (m.row[off], m.col[off])), shape=m.shape)


def _keep_topk_rows(m: sp.csr_matrix, k: int) -> sp.csr_matrix:
    m = m.tocsr()
    rows, cols, vals = [], [], []
    for r in range(m.shape[0]):
        s, e = m.indptr[r], m.indptr[r + 1]
        if e - s == 0:
            continue
        idx = m.indices[s:e]
        val = m.data[s:e]
        top = np.argsort(-val, kind="stable")[:k]
        rows.extend([r] * len(top))
        cols.extend(idx[top].tolist())
        vals.extend(val[top].tolist())
    return sp.csr_matrix((vals, (rows, cols)), shape=m.shape, dtype=np.float32)


class ContentKNN(Recommender):
    """Semantic retrieval: s(u, i) = sum_{j in H_u} cos(e_i, e_j)."""
    name = "content"

    def __init__(self, item_emb: np.ndarray):
        norms = np.linalg.norm(item_emb, axis=1, keepdims=True)
        norms[norms == 0] = 1
        self.emb = (item_emb / norms).astype(np.float32)

    def score(self, users):
        profile = np.asarray(self.history[users] @ self.emb)
        return profile @ self.emb.T


class TaxonomyKNN(Recommender):
    """s(u, i) = sum_{j in H_u} |common path prefix(i, j)| / L. Implemented
    as an inner product over one-hot path prefixes."""
    name = "taxonomy"

    def __init__(self, item_paths: list[tuple], depth: int | None = None):
        prefix_index: dict[tuple, int] = {}
        rows, cols = [], []
        for i, path in enumerate(item_paths):
            for d in range(1, len(path) + 1):
                rows.append(i)
                cols.append(prefix_index.setdefault(tuple(path[:d]), len(prefix_index)))
        self.depth = depth or max((len(p) for p in item_paths), default=1) or 1
        self.prefixes = sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)),
                                      shape=(len(item_paths), max(len(prefix_index), 1)))

    def score(self, users):
        profile = np.asarray((self.history[users] @ self.prefixes).todense())
        return np.asarray(self.prefixes @ profile.T).T / self.depth


class BPRMF(Recommender):
    """Matrix factorization with the BPR loss (Rendle et al., 2009), numpy SGD."""
    name = "bpr"

    def __init__(self, dim: int = 64, lr: float = 0.05, reg: float = 1e-4, epochs: int = 30,
                 batch_size: int = 4096, seed: int = 0):
        self.dim, self.lr, self.reg, self.epochs = dim, lr, reg, epochs
        self.batch_size, self.seed = batch_size, seed

    def fit(self, history):
        super().fit(history)
        rng = np.random.default_rng(self.seed)
        n_users, n_items = history.shape
        coo = history.tocoo()
        u_all, i_all = coo.row.astype(np.int64), coo.col.astype(np.int64)
        positives = np.sort(u_all * n_items + i_all)
        self.U = (rng.standard_normal((n_users, self.dim)) * 0.01).astype(np.float32)
        self.V = (rng.standard_normal((n_items, self.dim)) * 0.01).astype(np.float32)
        self.losses = []
        for _ in range(self.epochs):
            order = rng.permutation(len(u_all))
            total = 0.0
            for s in range(0, len(order), self.batch_size):
                b = order[s:s + self.batch_size]
                u, i = u_all[b], i_all[b]
                j = rng.integers(0, n_items, len(b))
                key = u * n_items + j
                pos = np.searchsorted(positives, key)
                clash = (pos < len(positives)) & (positives[np.minimum(pos, len(positives) - 1)] == key)
                u, i, j = u[~clash], i[~clash], j[~clash]
                pu, qi, qj = self.U[u], self.V[i], self.V[j]
                x = np.sum(pu * (qi - qj), axis=1)
                g = (1.0 / (1.0 + np.exp(x)))[:, None]
                total += float(np.sum(np.logaddexp(0, -x)))
                np.add.at(self.U, u, self.lr * (g * (qi - qj) - self.reg * pu))
                np.add.at(self.V, i, self.lr * (g * pu - self.reg * qi))
                np.add.at(self.V, j, self.lr * (-g * pu - self.reg * qj))
            self.losses.append(total / max(len(u_all), 1))
        return self

    def score(self, users):
        return self.U[users] @ self.V.T


class Hybrid(Recommender):
    """Weighted sum of per-user min-max normalized component scores. One class
    covers Sem, Tax, CF, Sem+Tax, Sem+Tax+CF ... by its weights."""
    name = "hybrid"

    def __init__(self, components: dict[str, Recommender], weights: dict[str, float]):
        unknown = set(weights) - set(components)
        if unknown:
            raise ValueError(f"weights for unknown components: {sorted(unknown)}")
        self.components, self.weights = components, weights

    def fit(self, history):
        super().fit(history)
        for name, w in self.weights.items():
            if w:
                self.components[name].fit(history)
        return self

    def score(self, users):
        total = np.zeros((len(users), self.history.shape[1]), dtype=np.float32)
        for name, w in self.weights.items():
            if not w:
                continue
            s = self.components[name].score(users)
            lo = s.min(axis=1, keepdims=True)
            span = s.max(axis=1, keepdims=True) - lo
            span[span <= 0] = 1.0
            total += w * (s - lo) / span
        return total
