# CLAUDE.md — project brief for future sessions

This repo is a completed university assignment: **"The Art of Analyzing Big Data — Network Analysis,
Visualization, Graph Embeddings, and Link Prediction."** Author: Mickael Zeitoun. It is **done and
submitted**; published (public) at https://github.com/mikigit97/bgu-network-analysis-assignment.

## What the assignment is
Five parts, all complete:
- **Part A (movies)** — A1.1 degree distribution, A1.2 graph-level embeddings of all 15,538 movie
  networks (PCA/t-SNE/UMAP + NetLSD), A2 top-12 by weighted betweenness (circular layout), A3
  PageRank/triangles/shortest-paths, A5 ego-network function. Selected network = **The Dark Knight
  (2008)** character co-appearance graph (`na_utils.SELECTED_MOVIE`).
- **Part A viz** — A4 (Dark Knight in **both** Gephi and Cytoscape) + A7 (Lord of the Rings couples;
  colour = gender, shape = race).
- **Part A6** — FICS chess network (~429.7M edges); handled by streaming aggregation, not NetworkX.
- **Part B + Bonus** — directed link prediction on the SNAP Reddit hyperlink network; topology
  baseline vs node2vec-augmented; temporal future-link split for the bonus.
- **Part C** — Enron manager detection: 3 centralities + precision@10, plus a **local LLM**
  (Ollama `qwen2.5:14b`) classifying managers from email content. LLM answers are cached.

## Layout
- `notebooks/combined_submission.ipynb` — **the deliverable**: one self-contained notebook with every
  part (code, embedded outputs, the 3 real Gephi/Cytoscape screenshots embedded as base64). Built by
  `scripts/build_combined_notebook.py` from the five per-part notebooks (do not hand-edit the combined
  file — edit the builder or the per-part notebooks and rebuild).
- `notebooks/partA_core.ipynb`, `partA_viz.ipynb`, `partA6_chess.ipynb`, `partB_linkpred.ipynb`,
  `partC_enron_llm.ipynb` — per-part sources (each executed top-to-bottom, 0 errors).
- `report/REPORT.md` — combined Markdown report; `report/parts/*.md` — per-part write-ups, each with a
  plain-English "How I solved this task" section.
- `figures/` — all plots + the official Gephi/Cytoscape PNGs. `exports/` — Gephi `.gexf` / Cytoscape
  `.cyjs`/`.graphml` files (used to make the GUI screenshots locally).
- `src/na_utils.py` — shared helpers: `load_movie_graph`, `list_movie_slugs`, `SELECTED_MOVIE`,
  `ollama_generate`, `save_fig`, `set_style` (forces matplotlib Agg — this is a headless cluster).
- `scripts/` — `build_combined_notebook.py`, `chess_aggregate.py` (+ `.sbatch`), `ollama*.sh/.sbatch`.
- `data/` — **git-ignored** (large). Only small derived files are kept locally (chess `.parquet`,
  `movies/graph_features.csv`, `enron/llm_cache.json`, `enron/enron_email_positions.csv`, the Dark
  Knight movie JSON). The big raw datasets were deleted; re-fetch them with the **Setup** cells at the
  top of the combined notebook (download flags default to `False`).

## How to rebuild / re-run
- Rebuild the combined notebook after changing any per-part notebook or a screenshot:
  `cd "/home/mickaelz/Network analysis" && PATH="$HOME/.local/bin:$PATH" python3 scripts/build_combined_notebook.py`
  (Embeds `figures/dark_knight_network_gephi.png`, `figures/dark_knight_cytoscape.png`,
  `figures/lotr_couples_cytoscape.png`; guards the optional "emulated" matplotlib renders behind
  `SHOW_EMULATED`.)
- Heavy data (chess 6.85GB, etc.): the chess parquet is rebuilt by `scripts/chess_aggregate.sbatch`
  (a SLURM CPU job, ~22GB RAM peak) — don't run the streaming aggregation in a small interactive shell.

## Environment notes (BGU SLURM cluster, headless)
- No display: Cytoscape/Gephi are GUI-only and were run **on the user's laptop**; the cluster side only
  produces import-ready files + emulated previews. The real screenshots are embedded in the notebook.
- **Local LLM** = Ollama as a SLURM GPU job exposing an HTTP API; endpoint written to
  `logs/ollama_endpoint.txt`; `na_utils.ollama_generate` reads it. Model `qwen2.5:14b` (cached). For
  cluster job management see `~/.claude/cluster-bgu-hpc.md` and the `bgu-cluster` skill.
- Python 3.9; packages installed in `--user` site: networkx, python-igraph, node2vec, gensim,
  scikit-learn, umap-learn, python-louvain, polars, pandas, numpy, matplotlib, seaborn, nbconvert.
- Reproducibility: seeds = 42 everywhere.

## Conventions / gotchas
- Never commit secrets or large data (`.gitignore` covers `data/`, `**/kaggle.json`, course PDFs/docx).
- The project path contains a space (`/home/mickaelz/Network analysis`) — quote it; and `#SBATCH
  --output` paths must NOT contain spaces.
- `gh` CLI is not installed; pushing to GitHub used a Personal Access Token over HTTPS (since rotated).
- Write explanations in detailed, plain English and define terms inline (the user's standing
  preference; see the global `~/.claude/CLAUDE.md`).
