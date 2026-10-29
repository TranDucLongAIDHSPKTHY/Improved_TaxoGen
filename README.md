# TaxoGen-Improved: taxonomy construction and taxonomy-guided recommendation

Reproducible experiments comparing TaxoGen (Zhang et al., KDD 2018), run by its
original code, with TaxoGen-Improved (SBERT document clusters found by UMAP +
HDBSCAN, BM25-cosine parent-child scoring) and further baselines. Taxonomies
are evaluated directly (automatic metrics, human judgments, Amazon category
paths where available) and downstream (product recommendation on Amazon
Reviews 2023).

Every number in the paper is produced by one command below and written to
`outputs/<dataset>/`, together with the exact config, package versions, git
commit and sha256 of every input file (`outputs/<dataset>/runs/`).

## Quick start (a few minutes, no downloads)

```bash
python -m venv .venv && .venv\Scripts\activate        # Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel     # old pip cannot install pyproject-only packages
pip install -r requirements-lock.txt && pip install -e . # exact verified environment (incl. spaCy model, BERTopic)
# or, direct dependencies only:
#   pip install -r requirements.txt && pip install -e . && python -m spacy download en_core_web_sm
#   pip install bertopic -c requirements.txt             # optional baseline; -c keeps numpy/torch pinned
python tools/taxogen_original/setup_original.py --python37 <path to python 3.7>   # see below
python -m pytest                                         # tests that need the original TaxoGen skip without it

python -m taxorec make-toy-data --config configs/toy.yaml
python -m taxorec all --config configs/toy.yaml          # whole pipeline on synthetic data
# -> outputs/toy/REPORT.md
```

The toy dataset only shows that the pipeline runs; its numbers mean nothing.

## Datasets

| dataset | role | config | data |
|---|---|---|---|
| Amazon Fashion | development | `configs/amazon_fashion.yaml` | `python -m taxorec download --config configs/amazon_fashion.yaml` |
| DBLP (TaxoGen release) | development | `configs/dblp.yaml` | `python -m taxorec download --config configs/dblp.yaml` (the [TaxoGen](https://github.com/franticnerd/taxogen) release from Google Drive, ~380 MB, sha256-verified) |
| Amazon Baby Products | confirmation | `configs/baby_products.yaml` | `python -m taxorec download --config configs/baby_products.yaml` |

The design of TaxoGen-Improved was developed on the training and validation
documents of the two development datasets. The confirmation dataset was not
used before the method, its tuning grid and the selection rule were fixed
(`docs/NHAT_KY.md`). Baby Products also has Amazon category paths for almost
every product, which serve as a silver-standard hierarchy (`evaluation.gold`).

## Methods

Taxonomy construction (`taxonomy.methods`; every method sees the same corpus
and candidate vocabulary):

| name | method |
|---|---|
| `improved` | TaxoGen-Improved: the node's documents clustered by SBERT (UMAP + HDBSCAN); anchor terms by CaseOLAP; centred context-SBERT term vectors; children kept when the BM25-cosine relation score (Eq. 1) reaches tau |
| `taxogen` | TaxoGen by its original code (below), the main baseline |
| `hclus` | HClus: recursive Ward agglomerative clustering of word2vec term vectors (a baseline of the TaxoGen paper) |
| `hlda` | hierarchical LDA (nested Chinese restaurant process; tomotopy), a baseline of the TaxoGen paper |
| `sbert_kmeans` | SBERT term vectors + recursive spherical k-means, no relation scoring |
| `bertopic` | flat BERTopic topics (depth 1; reported at full size only) |

The ablation (`taxonomy.ablation`) crosses split (documents / terms) x
clusterer (HDBSCAN / k-means) x relation scorer (none / cosine / BM25 /
BM25-cosine) and adds single-factor variants (no UMAP, no anchors, no
long-tail attachment, no centring, term-only SBERT, word2vec, and the parent
described by its general terms instead of its top terms in Eq. 1).

Recommendation (`rec.models`): Popularity, ItemKNN, RP3beta and BPR-MF
(interactions only); Sem (SBERT on product metadata); Tax(X) (shared taxonomy
path); the fusions Sem+Tax(X), Sem+CF and Sem+Tax(Improved)+CF. Significance is
tested against Sem with a paired bootstrap over test users and Holm correction.

## The original TaxoGen

The `taxogen` baseline is the original code (commit 929a7534), run in its own
Python 3.7 environment. Set it up once (needs git, a Python 3.7 interpreter,
and gcc or `pip install ziglang`):

```bash
python tools/taxogen_original/setup_original.py --python37 <path to python 3.7>
# -> external/taxogen_original/ (code + original_code.patch: Python 3 syntax only; venv37)
```

`tools/taxogen_original/driver.py` calls the original functions with the
parameters as arguments. The original fixes no seed; the driver seeds its
k-means initialisations, and word2vec runs single-threaded
(`taxogen_original.threads: 1`), so the same seed gives the same taxonomy.
Tuning configurations and seeds run as `taxogen_original.workers` parallel
processes; the results do not depend on that number.

## Running the experiments

```bash
C=configs/amazon_fashion.yaml
python -m taxorec prepare-rec     --config $C   # split (train / valid / test), item embeddings
python -m taxorec prepare-corpus  --config $C   # review corpus WITHOUT held-out interactions
python -m taxorec tune-taxonomy   --config $C   # parameters built on train, chosen on val
python -m taxorec sensitivity     --config $C   # one parameter at a time around the tuned one (val)
python -m taxorec build-taxonomy  --config $C   # main taxonomies (seed 42)
python -m taxorec build-taxonomy  --config $C --all-seeds   # seeds 42, 0, 1, 2, 3, for mean +- std
python -m taxorec ablation        --config $C   # ablation grid x seeds, on test
python -m taxorec eval-taxonomy   --config $C   # automatic metrics on test, at matched size and full size (+ gold)
python -m taxorec make-human-eval --config $C   # blinded sheets + Google Forms input
python -m taxorec freeze          --config $C   # snapshot the batch BEFORE sending it out
python -m taxorec import-forms    --config $C --file export.csv   # keep raw export, convert
python -m taxorec analyze-human-eval --config $C
python -m taxorec human-eval-holm --config configs/dblp.yaml --config configs/amazon_fashion.yaml --config configs/baby_products.yaml
python -m taxorec run-rec         --config $C   # tuned on validation, tested once
python -m taxorec report          --config $C   # outputs/<dataset>/REPORT.md
python -m taxorec archive         --config $C   # zip + sha256 manifest for Zenodo
```

`bash tools/run_chain.sh <config> [--human-eval]` runs every step of one
dataset that needs no human input, in order (DBLP has no recommendation steps).
Any config value can be overridden from the command line, e.g.
`--set taxonomy.methods.improved.builder.tau=0.4`; the resolved config is
recorded in the run directory.

### Human evaluation

`make-human-eval` samples relation and term-intrusion items from the
taxonomies of `human_eval.methods`, pruned to the same size, with the same
number of items per depth for every method, and writes blinded sheets plus
`forms_<dataset>.csv`. `tools/google_forms/create_form.gs` builds the Google
Form from that file. The batch is frozen (`freeze`) before any link is sent;
`import-forms` keeps the raw export unchanged and converts it;
`analyze-human-eval` refuses to run on a batch that differs from the frozen
one or on answers that fail the integrity checks.

## Reproducibility

* Every command runs with `PYTHONHASHSEED=0` (the CLI restarts itself with it
  when it is not set).
* Seeds: `seed` (main run), `seeds` (mean +- std), `taxonomy.split.seed`
  (document split, independent of the build seeds).
* The SBERT model is pinned to a commit; word2vec (gensim and the original
  word2vec.c) runs single-threaded; UMAP, HDBSCAN, k-means and hLDA are seeded.
* `requirements-lock.txt` is the exact environment the results were produced
  with (Python 3.10, Windows, CPU).

## Layout

```
configs/                 one YAML per dataset (+ shared _base_*.yaml)
src/taxorec/
  cli.py                 the `python -m taxorec <command>` entry point
  pipelines.py           one function per command; fixed output locations
  data.py                download, tokenization, phrases, Corpus, Amazon/DBLP loaders
  models/
    taxogen_improved.py  TaxoGen-Improved (the recursive builder; also sbert_kmeans, hclus, ablations)
    taxogen.py           TaxoGen by its original code (tools/taxogen_original/driver.py)
    hlda.py              hierarchical LDA
    bertopic_flat.py     flat BERTopic topics
    recommenders.py      Popularity, ItemKNN, RP3beta, BPR-MF, Sem, Tax, Hybrid fusions
  taxonomy/
    tree.py              Taxonomy data structure (+ validation, pruning, I/O)
    components.py        encoders, clustering, CaseOLAP ranking, BM25-cosine scoring
    evaluate.py          automatic metrics and item-to-path assignment
  human_eval.py          displayed terms, sampling, blinded sheets, integrity checks, statistics
  recommend.py           split, full-ranking metrics, tuning
  stats.py               bootstrap, Holm, Wilson, Fleiss, Krippendorff
  toy.py                 synthetic smoke-test data
tests/                   pytest suite (runs on synthetic data)
tools/run_chain.sh       every step of one dataset, in order
tools/google_forms/      Apps Script that builds the annotation form
tools/taxogen_original/  set-up and driver of the original TaxoGen code
docs/NHAT_KY.md          dated log of decisions and rule changes (Vietnamese)
```

## Protocol choices that matter

* Same corpus and candidate vocabulary for every taxonomy method on a dataset.
* Documents are split once into train (20%), val (30%) and test (50%):
  construction parameters are built on train and chosen on val with a
  pre-registered rule (mean rank over NPMI, subsumption and sibling overlap,
  with minimum size and term coverage), before any human judgment; final
  taxonomies are built on train + val; the test documents are read once, by
  eval-taxonomy and the ablation (a second reading is refused and every
  reading is logged). Every word2vec model is trained on the documents its
  method is built from.
* Methods are compared at matched size (every taxonomy pruned to the smallest
  one's size, shallower and larger nodes first), for the automatic metrics and
  for the human-evaluation sample, with full-size numbers reported too, as
  mean +- std over 5 seeds.
* Automatic metrics and annotation items use each node's own top 10 terms;
  the terms shown to annotators are those 10 re-ranked by one shared,
  embedding-agnostic ranker (CaseOLAP), sampled with equal per-depth quotas.
* The builders reject a child identical to its parent; the output of the
  original TaxoGen and of hLDA is kept as it is, and such edges are counted
  (`n_degenerate_edges`).
* Annotator answers pass an integrity check before any statistics are computed;
  method differences use a two-way (item x annotator) bootstrap, Holm-adjusted
  over every dataset and task together.
* Item representations use product metadata only (no review text), and reviews
  of validation/test interactions are excluded from the taxonomy corpus.
* Interactions: users need >= 3 (train/valid/test), items have no minimum;
  5-core leaves nothing in Amazon_Fashion (2023 release).
* Recommendation hyper-parameters and fusion weights are tuned on the
  validation item only; the test item is scored once, over the full catalog,
  with pessimistic tie-breaking.
