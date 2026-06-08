# Network Analysis — Movies, Link Prediction, and Enron Managers (with a Local LLM)

This repository is my submission for **"The Art of Analyzing Big Data — Network Analysis,
Visualization, Graph Embeddings, and Link Prediction."** It is organised as one runnable
Jupyter notebook per part, plus a single written report that ties everything together.

> **Single-file submission:** [`notebooks/combined_submission.ipynb`](notebooks/combined_submission.ipynb)
> is one self-contained notebook with **every** part (all code, embedded outputs, explanations, and the
> Gephi/Cytoscape screenshots embedded inline). The per-part notebooks below are kept as the sources.

A **network** (also called a **graph**) is just a set of **nodes** (dots — e.g. a movie
character, a subreddit, an employee) connected by **edges** (lines — e.g. "appeared together",
"linked to", "emailed"). Every task below studies some real network with these ideas.

## What is covered

| Part | Topic | Notebook | Write-up |
|------|-------|----------|----------|
| **A1.1** | Degree distribution of a movie network | `notebooks/partA_core.ipynb` | `report/parts/partA_core.md` |
| **A1.2** | Graph-level embeddings of all 15,538 movie networks + 2-D map | `notebooks/partA_core.ipynb` | `report/parts/partA_core.md` |
| **A2** | Top-12 characters by centrality (circular layout) | `notebooks/partA_core.ipynb` | `report/parts/partA_core.md` |
| **A3** | PageRank, triangles, average shortest path per character | `notebooks/partA_core.ipynb` | `report/parts/partA_core.md` |
| **A5** | Ego-network function (in + out neighbours) | `notebooks/partA_core.ipynb` | `report/parts/partA_core.md` |
| **A4** | Cytoscape + Gephi visualisation (export files + emulated renders) | `notebooks/partA_viz.ipynb` | `report/parts/partA_viz.md` |
| **A7** | Lord of the Rings couples network (colour = gender, shape = race) | `notebooks/partA_viz.ipynb` | `report/parts/partA_viz.md` |
| **A6** | Large FICS chess network (429 M edges) — top players + viz | `notebooks/partA6_chess.ipynb` | `report/parts/partA6_chess.md` |
| **B** | Directed link prediction (Reddit hyperlink network) | `notebooks/partB_linkpred.ipynb` | `report/parts/partB_linkpred.md` |
| **Bonus** | Future (temporal) link prediction | `notebooks/partB_linkpred.ipynb` | `report/parts/partB_linkpred.md` |
| **C** | Enron manager detection (3 centralities + a local LLM) | `notebooks/partC_enron_llm.ipynb` | `report/parts/partC_enron_llm.md` |

**The single combined report is `report/REPORT.md`.** Each part's write-up contains a
plain-English "How I solved this task" section, as the assignment requires.

## Repository structure

```
.
├── README.md                  # this file
├── report/
│   ├── REPORT.md              # combined report (start here)
│   └── parts/                 # one detailed write-up per part
├── notebooks/                 # one executed .ipynb per part (outputs embedded)
├── figures/                   # all generated plots (PNG)
├── exports/                   # Gephi (.gexf) / Cytoscape (.cyjs/.graphml) network files
├── src/na_utils.py            # shared helpers (graph loader, Ollama client, plotting)
├── scripts/                   # SLURM batch files + the chess streaming-aggregation script
└── data/                      # raw datasets — NOT committed (see .gitignore); fetch instructions below
```

## Datasets (downloaded by the code, not stored in git)

- **Movie Dynamics Networks** (Parts A1–A5, A7-style movie work): Kaggle
  `michaelfire/movie-dynamics-over-15000-movie-social-networks` — 15,538 per-movie character
  co-appearance graphs.
- **Lord of the Rings** (A7): public character race reference (`juandes/lotr-names-classification`)
  plus a hand-built, documented couples edge list with gender + race.
- **FICS chess** (A6): `http://dynamics.cs.washington.edu/nobackup/chess/fcis.tar.gz` — 6.85 GB,
  ~429.7 M interaction edges between ~519k players.
- **Reddit hyperlink network** (Part B + Bonus): SNAP `soc-redditHyperlinks-body` — directed,
  timestamped subreddit-to-subreddit links.
- **Enron emails** (Part C): the CMU corpus `enron_mail_20150507.tar.gz` (real addresses + content),
  plus public role/title labels for the ~150 core employees.

## Tools and libraries

Python 3.9 with: `networkx`, `python-igraph`, `node2vec` + `gensim`, `scikit-learn`,
`umap-learn`, `python-louvain`, `polars` (streaming the 15 GB chess file), `pandas`, `numpy`,
`scipy`, `matplotlib`, `seaborn`. Notebooks are executed with `nbconvert`.

**Local LLM (Part C):** an **Ollama** server (a tool that runs large language models locally)
running the **`qwen2.5:14b`** model, launched as a GPU job on the BGU SLURM cluster and queried
over HTTP. No email text ever leaves the local machine/cluster.

## Notes / honest limitations

- **Headless cluster:** this ran on a display-less Linux server, so Cytoscape and Gephi (which
  are desktop GUI apps) cannot produce real screenshots here. For tasks A4/A7 I therefore provide
  (a) ready-to-import network files in `exports/`, (b) faithful matplotlib renders that emulate the
  intended GUI look, and (c) step-by-step import instructions so the official screenshots can be
  captured by opening those files in Cytoscape/Gephi locally.
- **Big networks:** the chess network (429 M edges) is never loaded whole into NetworkX; it is
  stream-aggregated with polars into a compact weighted edge list, and the structural analysis runs
  on a justified subgraph. The Reddit network is sampled to its most active subreddits for speed.

## Reproducing

1. Install the libraries listed above (`pip install --user networkx python-igraph node2vec gensim
   scikit-learn umap-learn python-louvain polars pandas numpy scipy matplotlib seaborn nbconvert`).
2. Download the datasets into `data/` (see links above; the notebooks/scripts document the exact files).
3. For Part C, start a local Ollama server and point `logs/ollama_endpoint.txt` at it.
4. Run any notebook top-to-bottom, e.g.
   `jupyter nbconvert --to notebook --execute --inplace notebooks/partA_core.ipynb`.

*Author: Mickael Zeitoun (mickaelz@post.bgu.ac.il).*
