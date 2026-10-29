"""Runs the original TaxoGen code; executed by its Python 3.7 environment
(`taxorec.models.taxogen` calls it, nothing else imports it).

    driver.py prepare --code C --data D [--threads 20]
        D/raw/papers.txt, D/raw/keywords.txt (written by taxorec) ->
        global embeddings with the original word2vec (its default settings, as in
        run.sh's expected input), then the original preprocessing
        (cluster-preprocess.py, preprocess.py) and run.sh's init/ files.
    driver.py run --code C --data D --out O --n-cluster K --filter-thre F --n-expand E --n-cluster-iter I
                  [--seed S] [--threads T]
        main.py's recursion on D with these parameters into O/tree, then
        compress.py into O/taxonomy.txt.

main() of the original hard-codes the parameters and the output folder, so the
driver repeats its three lines (copy init/ to the root, recur) with them as
arguments; every function it calls is the original one.

Seeding: the original fixes no seed. Its only random choices are the k-means
initialisations of spherecluster (random_state=None, i.e. numpy's global
generator), which `run` seeds with S, and word2vec, which is deterministic
with one thread (multi-threaded training updates the vectors in a racing
order). The local embeddings call word2vec with `-threads 20` hard-coded;
`run` passes T instead, every other argument unchanged. With T = 1 and a fixed
PYTHONHASHSEED (set by taxorec), the same S gives the same taxonomy.

One addition: when CaseOLAP has filtered a
child's own label out of the node's keywords, the original local-embedding step
fails with a KeyError looking up the label's vector. The driver then takes that
vector from the node's full embeddings.txt; the candidate neighbours stay the
remaining keywords, and nothing changes when no label is missing.
"""
import argparse
import importlib.util
import os
import random
import shutil
import subprocess
import sys
from distutils.dir_util import copy_tree


def _word2vec(code):
    return os.path.join(code, "word2vec.exe" if os.name == "nt" else "word2vec")


def _load(code, filename, name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(code, filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def prepare(code, data, threads):
    raw, inp, init = (os.path.join(data, d) + "/" for d in ("raw", "input", "init"))
    for d in (inp, init):
        os.makedirs(d, exist_ok=True)
    subprocess.check_call([_word2vec(code), "-threads", str(threads), "-train", raw + "papers.txt",
                           "-output", inp + "embeddings.txt"])
    sys.path.insert(0, code)
    _load(code, "cluster-preprocess.py", "cluster_preprocess").main(raw, inp, init)
    _load(code, "preprocess.py", "tg_preprocess").get_reidx_file(inp + "papers.txt", inp + "keywords.txt",
                                                                 inp + "index.txt")
    shutil.copyfile(inp + "embeddings.txt", init + "embeddings.txt")
    shutil.copyfile(inp + "keywords.txt", init + "seed_keywords.txt")


class _LabelVectors(dict):
    """The remaining keywords' vectors, iterated as in the original; a lookup of
    one of `labels` that is not among them falls back to `full`."""

    def __init__(self, embs, full, labels):
        super().__init__(embs)
        self.full, self.labels = full, labels

    def __missing__(self, key):
        if key not in self.labels:
            raise KeyError(key)
        return self.full[key]


def read_files_keeping_labels(lem, folder, parent):
    """lem.read_files, with child labels missing from the remaining keywords
    looked up in the node's full embeddings.txt."""
    embs, keywords, cates = lem.read_files(folder, parent)
    missing = sorted(c for c in cates if c not in embs)
    if missing:
        print("[driver] child labels of %s not among its remaining keywords, vectors from %s/embeddings.txt: %s"
              % (parent, folder.rstrip("/"), " ".join(missing)))
        embs = _LabelVectors(embs, lem.utils.load_embeddings("%s/embeddings.txt" % folder), set(missing))
    return embs, keywords, cates


def _main_local_embedding(lem):
    def main_local_embedding(folder, doc_file, reidx, parent, N):   # the original, but for read_files
        embs, keywords, cates = read_files_keeping_labels(lem, folder, parent)
        cates = lem.relevant_phs(embs, cates, int(N))
        pd_map, docs = lem.revevant_docs(doc_file, reidx, cates)
        lem.run_word2vec(pd_map, docs, cates, folder)
    return main_local_embedding


class _Word2vecThreads:
    """Stands in for `subprocess` in local_embedding_training, whose only use of
    it is the word2vec call with `-threads 20`: runs that call with `threads`."""

    def __init__(self, threads):
        self.threads = str(threads)

    def call(self, args, **kwargs):
        args = list(args)
        if "-threads" in args:
            args[args.index("-threads") + 1] = self.threads
        return subprocess.call(args, **kwargs)


def run(code, data, out, n_cluster, filter_thre, n_expand, n_cluster_iter, seed=None, threads=None):
    data, out = os.path.abspath(data), os.path.abspath(out)
    os.chdir(code)                      # local embeddings call ./word2vec
    sys.path.insert(0, code)
    import numpy as np
    import main as taxogen_main
    import local_embedding_training
    taxogen_main.main_local_embedding = _main_local_embedding(local_embedding_training)
    if threads is not None:
        local_embedding_training.subprocess = _Word2vecThreads(threads)
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    root = os.path.join(out, "tree") + "/"
    copy_tree(os.path.join(data, "init") + "/", root)
    taxogen_main.recur(os.path.join(data, "input") + "/", root, n_cluster, "*", n_cluster_iter, filter_thre,
                       n_expand, 0, True, True)
    subprocess.check_call([sys.executable, "compress.py", "-root", root.rstrip("/"),
                           "-output", os.path.join(out, "taxonomy.txt")])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["prepare", "run"])
    ap.add_argument("--code", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out")
    ap.add_argument("--threads", type=int, default=20)
    ap.add_argument("--n-cluster", type=int)
    ap.add_argument("--filter-thre", type=float)
    ap.add_argument("--n-expand", type=int)
    ap.add_argument("--n-cluster-iter", type=int)
    ap.add_argument("--seed", type=int)
    a = ap.parse_args()
    code = os.path.abspath(a.code)
    if a.step == "prepare":
        prepare(code, os.path.abspath(a.data), a.threads)
    else:
        run(code, a.data, a.out, a.n_cluster, a.filter_thre, a.n_expand, a.n_cluster_iter, a.seed, a.threads)


if __name__ == "__main__":
    main()
