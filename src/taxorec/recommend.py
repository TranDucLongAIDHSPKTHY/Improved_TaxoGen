"""Recommendation experiments: interactions, leave-last-out split with a
validation item, full-catalog ranking metrics and tuning (the models are in
taxorec.models.recommenders).

Protocol: per user, the last interaction (by time) is the test item, the
second-to-last the validation item, the rest train. Hyper-parameters and
fusion weights are chosen on validation only; the final model is refit on
train+validation and scored once on test. Every model ranks the whole
catalog (no candidate pool), history items are excluded, and ties are broken
pessimistically (the target ranks after every item with an equal score), so
a model gets no credit for the arbitrary order of items it cannot tell apart,
e.g. the thousands of zero-score items of a sparse CF model.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
import scipy.sparse as sp

from taxorec.models.recommenders import Recommender

INF_RANK = np.iinfo(np.int64).max


# ---------------------------------------------------------------- data & split

def kcore(df: pd.DataFrame, k: int, k_item: int | None = None) -> pd.DataFrame:
    """Iteratively drop users with fewer than k and items with fewer than
    k_item (default k) interactions."""
    k_item = k if k_item is None else k_item
    if k <= 1 and k_item <= 1:
        return df
    while True:
        u = df["user_id"].map(df["user_id"].value_counts())
        i = df["item_id"].map(df["item_id"].value_counts())
        keep = (u >= k) & (i >= k_item)
        if keep.all():
            return df
        df = df[keep]


def prepare_interactions(reviews: pd.DataFrame, valid_items=None, min_rating: float | None = None,
                         k: int = 5, k_item: int | None = None) -> pd.DataFrame:
    df = reviews[["user_id", "item_id", "rating", "timestamp"]].copy()
    if min_rating is not None:
        df = df[df["rating"] >= min_rating]
    if valid_items is not None:
        df = df[df["item_id"].isin(set(valid_items))]
    df = df.sort_values(["timestamp", "item_id"]).drop_duplicates(["user_id", "item_id"], keep="first")
    return kcore(df, k, k_item).reset_index(drop=True)


@dataclass
class Split:
    users: np.ndarray            # user index -> user id
    items: np.ndarray            # item index -> item id
    train: sp.csr_matrix         # users x items, binary
    valid_item: np.ndarray       # per user, -1 when absent
    test_item: np.ndarray

    @property
    def n_users(self) -> int:
        return len(self.users)

    @property
    def n_items(self) -> int:
        return len(self.items)

    def history(self, include_valid: bool) -> sp.csr_matrix:
        if not include_valid:
            return self.train
        has = self.valid_item >= 0
        extra = sp.csr_matrix((np.ones(has.sum(), dtype=np.float32),
                               (np.where(has)[0], self.valid_item[has])), shape=self.train.shape)
        return ((self.train + extra) > 0).astype(np.float32).tocsr()

    def eval_users(self, role: str) -> tuple[np.ndarray, np.ndarray]:
        target = self.valid_item if role == "valid" else self.test_item
        users = np.where(target >= 0)[0]
        return users, target[users]

    def heldout_pairs(self) -> pd.DataFrame:
        rows = []
        for role, arr in (("valid", self.valid_item), ("test", self.test_item)):
            for u in np.where(arr >= 0)[0]:
                rows.append((self.users[u], self.items[arr[u]], role))
        return pd.DataFrame(rows, columns=["user_id", "item_id", "role"])

    def save(self, path) -> None:
        np.savez_compressed(path, users=self.users.astype(str), items=self.items.astype(str),
                            train_data=self.train.data, train_indices=self.train.indices,
                            train_indptr=self.train.indptr, valid_item=self.valid_item,
                            test_item=self.test_item)

    @classmethod
    def load(cls, path) -> "Split":
        z = np.load(path, allow_pickle=False)
        train = sp.csr_matrix((z["train_data"], z["train_indices"], z["train_indptr"]),
                              shape=(len(z["users"]), len(z["items"])))
        return cls(z["users"], z["items"], train, z["valid_item"], z["test_item"])


def leave_last_out(df: pd.DataFrame, min_len: int = 3) -> Split:
    users = np.array(sorted(df["user_id"].unique()))
    items = np.array(sorted(df["item_id"].unique()))
    uidx = {u: i for i, u in enumerate(users)}
    iidx = {it: i for i, it in enumerate(items)}
    df = df.assign(u=df["user_id"].map(uidx), i=df["item_id"].map(iidx)).sort_values(["u", "timestamp", "i"])
    valid = np.full(len(users), -1, dtype=np.int64)
    test = np.full(len(users), -1, dtype=np.int64)
    rows, cols = [], []
    for u, grp in df.groupby("u", sort=True):
        seq = grp["i"].to_numpy()
        if len(seq) >= min_len:
            test[u], valid[u] = seq[-1], seq[-2]
            seq = seq[:-2]
        rows.extend([u] * len(seq))
        cols.extend(seq.tolist())
    train = sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(len(users), len(items)))
    return Split(users, items, train, valid, test)


# ---------------------------------------------------------------- metrics

def target_ranks(scores: np.ndarray, targets: np.ndarray, history: sp.csr_matrix) -> np.ndarray:
    """0-based rank of each user's target among unseen items, placing it after
    every other item with an equal score; INF_RANK if the target is masked."""
    scores = scores.astype(np.float64, copy=True)
    hist = history.tocoo()
    scores[hist.row, hist.col] = -np.inf
    t = scores[np.arange(len(targets)), targets]
    rank = (scores > t[:, None]).sum(axis=1) + (scores == t[:, None]).sum(axis=1) - 1
    return np.where(np.isfinite(t), rank, INF_RANK)


def coverage_mask(scores: np.ndarray, history: sp.csr_matrix, k: int) -> np.ndarray:
    """A user is covered when at least k unseen items score above the user's
    most common score (the 'no evidence' level: 0 for CF/taxonomy, the
    popularity floor, ...). Sign-agnostic, so it also works for cosine or
    factorization scores."""
    scores = scores.astype(np.float64, copy=True)
    hist = history.tocoo()
    scores[hist.row, hist.col] = np.nan
    out = np.empty(len(scores), dtype=bool)
    for r, row in enumerate(scores):
        row = row[~np.isnan(row)]
        vals, counts = np.unique(row, return_counts=True)
        out[r] = (row > vals[np.argmax(counts)]).sum() >= k
    return out


def summarize_ranks(ranks: np.ndarray, ks=(5, 10, 20)) -> dict:
    out = {"n_users": int(len(ranks))}
    finite = ranks != INF_RANK
    for k in ks:
        hit = ranks < k
        out[f"recall@{k}"] = float(hit.mean())
        out[f"ndcg@{k}"] = float(np.where(hit, 1.0 / np.log2(np.where(finite, ranks, 0) + 2), 0.0).mean())
    out["mrr"] = float(np.where(finite, 1.0 / (np.where(finite, ranks, 0) + 1), 0.0).mean())
    return out


def per_user_metric(ranks: np.ndarray, metric: str) -> np.ndarray:
    name, k = metric.split("@")
    k = int(k)
    hit = ranks < k
    if name == "recall":
        return hit.astype(float)
    return np.where(hit, 1.0 / np.log2(np.where(hit, ranks, 0) + 2), 0.0)


def evaluate(model: Recommender, users: np.ndarray, targets: np.ndarray, ks=(5, 10, 20),
             batch_size: int = 512, coverage_k: int | None = 10) -> dict:
    ranks = np.empty(len(users), dtype=np.int64)
    covered = np.zeros(len(users), dtype=bool)
    for s in range(0, len(users), batch_size):
        u = users[s:s + batch_size]
        scores = model.score(u)
        ranks[s:s + len(u)] = target_ranks(scores, targets[s:s + len(u)], model.history[u])
        if coverage_k:
            covered[s:s + len(u)] = coverage_mask(scores, model.history[u], coverage_k)
    summary = summarize_ranks(ranks, ks)
    if coverage_k:
        summary["coverage"] = float(covered.mean())
    return {"summary": summary, "ranks": ranks, "covered": covered}


# ---------------------------------------------------------------- tuning

def param_grid(grid: dict[str, list]) -> list[dict]:
    if not grid:
        return [{}]
    keys = sorted(grid)
    return [dict(zip(keys, vals)) for vals in itertools.product(*(grid[k] for k in keys))]


def simplex_grid(names: list[str], step: float = 0.1) -> list[dict]:
    """All weight vectors on the simplex with the given step and every weight > 0."""
    n = round(1 / step)
    out = []
    for combo in itertools.product(range(1, n + 1), repeat=len(names)):
        if sum(combo) == n:
            out.append({name: c / n for name, c in zip(names, combo)})
    return out or [{names[0]: 1.0}]


def tune_and_test(factory: Callable[[dict], Recommender], grid: list[dict], split: Split,
                  metric: str = "ndcg@10", ks=(5, 10, 20), max_valid_users: int | None = None,
                  seed: int = 0, log: Callable[[str], None] = print) -> dict:
    """Pick params on validation (optionally on a fixed random subset of
    validation users, to bound tuning cost), refit on train+valid, test once."""
    v_users, v_targets = split.eval_users("valid")
    if max_valid_users and len(v_users) > max_valid_users:
        pick = np.sort(np.random.default_rng(seed).choice(len(v_users), max_valid_users, replace=False))
        v_users, v_targets = v_users[pick], v_targets[pick]
    t_users, t_targets = split.eval_users("test")
    trials = []
    for params in grid:
        model = factory(params).fit(split.history(include_valid=False))
        res = evaluate(model, v_users, v_targets, ks, coverage_k=None)
        trials.append({"params": params, "valid": res["summary"]})
        log(f"  valid {params} -> {metric}={res['summary'][metric]:.4f}")
    best = max(trials, key=lambda t: (t["valid"][metric], -grid.index(t["params"])))
    model = factory(best["params"]).fit(split.history(include_valid=True))
    test = evaluate(model, t_users, t_targets, ks)
    return {"best_params": best["params"], "valid": best["valid"], "test": test["summary"],
            "trials": trials, "test_ranks": test["ranks"], "test_users": t_users}
