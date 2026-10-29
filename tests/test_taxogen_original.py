"""The original TaxoGen code through taxorec.models.taxogen. The end-to-end
test needs tools/taxogen_original/setup_original.py to have been run."""
from pathlib import Path

import pytest

from taxorec.common import load_config
from taxorec.data import Corpus, DocStore
from taxorec.models.taxogen import build_original, convert, corpus_key
from taxorec.taxonomy.tree import ROOT

ENV = load_config("configs/dblp.yaml")["taxogen_original"]
INSTALLED = Path(ENV["python"]).exists() and Path(ENV["code"]).exists()


def test_convert_reads_compress_output_and_node_documents(tmp_path):
    root = tmp_path / "tree"
    for folder, n in [("ml", 3), ("ml/deep_learning", 2), ("db", 1)]:
        (root / folder).mkdir(parents=True, exist_ok=True)
        (root / folder / "doc_ids.txt").write_text("".join(f"{i}\n" for i in range(n)))
    compressed = tmp_path / "taxonomy.txt"
    compressed.write_text("*/ml\tml,learning\n*/db\tdb,query\n*/ml/deep_learning\tdeep_learning,cnn\n",
                          encoding="utf-8")
    tax = convert(root, compressed)
    assert tax.size == 3 and tax.depth == 2
    ml = next(n for n in tax.children(ROOT) if n.terms[0] == "ml")
    assert ml.n_docs == 3 and [c.terms for c in tax.children(ml.id)] == [["deep_learning", "cnn"]]


def test_corpus_key_separates_splits(synthetic_corpus):
    a, b = synthetic_corpus.subset(range(0, 400)), synthetic_corpus.subset(range(400, 800))
    assert corpus_key(a) != corpus_key(b) and corpus_key(a) == corpus_key(synthetic_corpus.subset(range(0, 400)))


@pytest.mark.skipif(not INSTALLED, reason="original TaxoGen not set up (tools/taxogen_original/setup_original.py)")
def test_original_code_builds_and_reuses_a_finished_run(hierarchy, tmp_path):
    # on a corpus this small, CaseOLAP's filtering can drop a child's own label from
    # the node's keywords, and the original local-embedding step then fails with a
    # KeyError: filter_thre 0 keeps every keyword
    store = DocStore.write(tmp_path / "docs.txt", hierarchy[1] * 6)
    corpus = Corpus.build(lambda: iter(store), min_df=2, max_df_ratio=0.5, docstore_path=tmp_path / "docs.txt")
    params = {"n_cluster": 3, "filter_thre": 0.0, "n_expand": 20, "n_cluster_iter": 1}
    tax = build_original(ENV, params, corpus, tmp_path, seed=42, log=lambda m: None)
    assert len(tax.children(ROOT)) == 3 and all(n.terms and n.n_docs > 0 for n in tax.non_root())
    again = build_original(ENV, params, corpus, tmp_path, seed=42, log=lambda m: None)
    assert [n.terms for n in again.non_root()] == [n.terms for n in tax.non_root()]


@pytest.mark.skipif(not INSTALLED, reason="original TaxoGen not set up (tools/taxogen_original/setup_original.py)")
def test_original_code_is_deterministic_for_a_seed_with_one_thread(hierarchy, tmp_path):
    store = DocStore.write(tmp_path / "docs.txt", hierarchy[1] * 6)
    corpus = Corpus.build(lambda: iter(store), min_df=2, max_df_ratio=0.5, docstore_path=tmp_path / "docs.txt")
    params = {"n_cluster": 3, "filter_thre": 0.0, "n_expand": 20, "n_cluster_iter": 1}
    env = {**ENV, "threads": 1}

    def shape(tax):
        return [(n.id, n.terms, n.n_docs) for n in tax.non_root()]

    runs = [build_original(env, params, corpus, tmp_path / f"w{i}", seed=7, log=lambda m: None)
            for i in range(2)]                     # separate work dirs: two fresh runs, nothing reused
    assert shape(runs[0]) == shape(runs[1])

    # two seeds started in parallel on one work dir (shared global embeddings,
    # prepared once) give what they give when run one after the other
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=2) as pool:
        parallel = list(pool.map(lambda s: build_original(env, params, corpus, tmp_path / "par", seed=s,
                                                          log=lambda m: None), [7, 8]))
    assert shape(parallel[0]) == shape(runs[0])
    assert shape(parallel[1]) == shape(build_original(env, params, corpus, tmp_path / "seq8", seed=8,
                                                      log=lambda m: None))


def _load_module(path, name):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.skipif(not INSTALLED, reason="original TaxoGen not set up (tools/taxogen_original/setup_original.py)")
def test_driver_takes_a_filtered_out_label_vector_from_the_full_embeddings(tmp_path, monkeypatch):
    # as at node monitor/clear of Baby Products: CaseOLAP kept 'camera' as a child
    # but filtered it out of the node's keywords
    monkeypatch.syspath_prepend(ENV["code"])
    lem = _load_module(Path(ENV["code"]) / "local_embedding_training.py", "local_embedding_training")
    driver = _load_module(Path("tools/taxogen_original/driver.py"), "tg_driver")
    vectors = {"camera": "1 0 0", "lens": "0.9 0.1 0", "zoom": "0.8 0.2 0", "crib": "0 1 0.1", "sleep": "0 1 0"}
    (tmp_path / "embeddings.txt").write_text("5 3\n" + "".join(f"{w} {v}\n" for w, v in vectors.items()))
    (tmp_path / "keywords.txt").write_text("lens\nzoom\ncrib\nsleep\n")
    (tmp_path / "hierarchy.txt").write_text("camera clear\nsleep clear\nother elsewhere\n")
    folder = str(tmp_path) + "/"

    embs, _, cates = lem.read_files(folder, "clear")
    with pytest.raises(KeyError, match="camera"):
        lem.relevant_phs(embs, cates, 2)

    embs, _, cates = driver.read_files_keeping_labels(lem, folder, "clear")
    assert sorted(embs) == ["crib", "lens", "sleep", "zoom"]       # candidates: remaining keywords only
    with pytest.raises(KeyError):
        embs["other"]                                               # only the missing labels fall back
    assert lem.relevant_phs(embs, cates, 2) == {"camera": {"lens", "zoom"}, "sleep": {"sleep", "crib"}}

    (tmp_path / "keywords.txt").write_text("camera\nlens\nzoom\ncrib\nsleep\n")
    embs, _, _ = driver.read_files_keeping_labels(lem, folder, "clear")
    assert type(embs) is dict                                       # nothing missing: the original's dict
