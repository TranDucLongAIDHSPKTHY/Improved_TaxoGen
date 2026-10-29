"""TaxoGen run by its original code (github.com/franticnerd/taxogen, commit
929a7534; Zhang et al., KDD 2018), the main baseline.

The original code needs Python 3.7 (tools/taxogen_original/setup_original.py),
so it runs in a subprocess through tools/taxogen_original/driver.py. Its input
is written from the same corpus every other method sees: one line of
phrase-merged tokens per document of the given split, and the corpus's
candidate terms as keywords. The global embeddings are trained by the original
word2vec on those documents only.

The original code fixes no random seed; the driver seeds its k-means
initialisations with `seed`, and word2vec (global and local embeddings) runs
with `threads` threads, deterministic only with 1. With threads = 1 the same
seed gives the same taxonomy. The global embeddings take no seed and are shared
by every run on the same documents; each seed gets its own run, and a finished
run is reused, so the main taxonomy (config seed) and the run of that seed are
the same tree.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import threading
from pathlib import Path

from taxorec.common import ensure_dir, write_json
from taxorec.taxonomy.tree import ROOT, Taxonomy

DRIVER = Path(__file__).resolve().parents[3] / "tools" / "taxogen_original" / "driver.py"
# runs of several configurations or seeds on the same documents may be started
# in parallel threads; their shared global embeddings are prepared once
_PREPARE_LOCK = threading.Lock()


def convert(root: Path, compressed: Path, method: str = "taxogen") -> Taxonomy:
    """compress.py writes '*/a/b<TAB>term,term,...' per node; the node's documents
    are the lines of its folder's doc_ids.txt."""
    tax = Taxonomy(method, {"source": str(compressed), "root": str(root)})
    by_path = {"*": ROOT}
    rows = [line.rstrip("\n").split("\t") for line in open(compressed, encoding="utf-8") if "\t" in line]
    for path, terms in sorted(rows, key=lambda r: (r[0].count("/"), r[0])):
        ids = Path(root) / Path(*path.split("/")[1:]) / "doc_ids.txt"
        n_docs = sum(1 for line in open(ids, encoding="utf-8") if line.strip()) if ids.exists() else 0
        node = tax.add_child(by_path[path.rsplit("/", 1)[0]], [t for t in terms.split(",") if t], n_docs)
        by_path[path] = node.id
    return tax


def corpus_key(corpus) -> str:
    """Identifies the documents and vocabulary of a corpus or split."""
    h = hashlib.sha256((corpus.docstore_path or "").encode("utf-8"))
    h.update("\n".join(corpus.terms).encode("utf-8"))
    if corpus.doc_map is not None:
        h.update(corpus.doc_map.astype("int64").tobytes())
    else:
        h.update(str(corpus.n_docs).encode())
    return h.hexdigest()[:16]


def write_inputs(corpus, raw_dir: Path) -> None:
    raw_dir = ensure_dir(raw_dir)
    with open(raw_dir / "papers.txt", "w", encoding="utf-8", newline="\n") as f:
        for tokens in corpus.documents():
            f.write(" ".join(tokens) + "\n")
    with open(raw_dir / "keywords.txt", "w", encoding="utf-8", newline="\n") as f:
        f.writelines(t + "\n" for t in corpus.terms)


def _call(env_cfg: dict, args: list[str], log_path: Path, log) -> None:
    cmd = [str(Path(env_cfg["python"]).resolve()), str(DRIVER), *args,
           "--code", str(Path(env_cfg["code"]).resolve())]
    log(f"[taxogen-original] {' '.join(cmd)}")
    # the local embeddings write each node's documents in set order, which
    # depends on the hash seed
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1", "PYTHONHASHSEED": "0"}
    with open(log_path, "a", encoding="utf-8") as f:
        done = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
    if done.returncode:
        raise RuntimeError(f"original TaxoGen failed (exit {done.returncode}); see {log_path}")


def build_original(env_cfg: dict, params: dict, corpus, work_dir: Path, seed: int, log=print,
                   method: str = "taxogen") -> Taxonomy:
    """params: n_cluster, filter_thre, n_expand, n_cluster_iter (MAX_LEVEL is 3 in main.py)."""
    threads = int(env_cfg.get("threads", 1))
    data = Path(work_dir) / f"data-{corpus_key(corpus)}-t{threads}"
    with _PREPARE_LOCK:
        if not (data / "prepared").exists():
            write_inputs(corpus, data / "raw")
            _call(env_cfg, ["prepare", "--data", str(data), "--threads", str(threads)],
                  data / "prepare.log", log)
            (data / "prepared").write_text("ok")
    p = {k: params[k] for k in ("n_cluster", "filter_thre", "n_expand", "n_cluster_iter")}
    out = data / "run-{n_cluster}-{filter_thre}-{n_expand}-{n_cluster_iter}-seed{seed}".format(**p, seed=seed)
    if not (out / "taxonomy.txt").exists():
        if out.exists():                # an interrupted run: start it again from scratch
            import shutil
            shutil.rmtree(out)
        ensure_dir(out)
        write_json(out / "params.json", {**p, "seed": seed, "threads": threads})
        _call(env_cfg, ["run", "--data", str(data), "--out", str(out),
                        "--n-cluster", str(p["n_cluster"]), "--filter-thre", str(p["filter_thre"]),
                        "--n-expand", str(p["n_expand"]), "--n-cluster-iter", str(p["n_cluster_iter"]),
                        "--seed", str(seed), "--threads", str(threads)],
              out / "run.log", log)
    else:
        log(f"[taxogen-original] reusing finished run {out}")
    tax = convert(out / "tree", out / "taxonomy.txt", method)
    tax.params.update({"params": p, "seed": seed, "threads": threads, "run_dir": str(out)})
    return tax
