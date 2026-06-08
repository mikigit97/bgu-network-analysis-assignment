"""
na_utils.py — shared helpers for the Network Analysis assignment.

Usage from any notebook/script:
    import sys; sys.path.insert(0, "/home/mickaelz/Network analysis/src")
    import na_utils as na
    G = na.load_movie_graph("2008_The_Dark_Knight")
    txt = na.ollama_generate("Say hi")
"""
import os
import json
import warnings
import networkx as nx

BASE = "/home/mickaelz/Network analysis"
MOVIES_DIR = os.path.join(BASE, "data/movies/moviedynamics")
FIG_DIR = os.path.join(BASE, "figures")
EXPORT_DIR = os.path.join(BASE, "exports")
ENDPOINT_FILE = os.path.join(BASE, "logs/ollama_endpoint.txt")

# The single movie network selected for Part A tasks A1.1, A2, A3, A5.
SELECTED_MOVIE = "2008_The_Dark_Knight"


# ----------------------------------------------------------------------------
# Movie networks (Kaggle: michaelfire/movie-dynamics-over-15000-...)
# Each movie has <slug>.json (nodes keyed by CHARACTER, attr 'role'=actor) and
# <slug>.actors.json (nodes keyed by ACTOR, attr 'role'=character). Graphs are
# undirected and weighted ('weight' = number of co-appearances). Node/edge attrs
# 'first'/'last' = first/last frame index the character/pair appears.
# ----------------------------------------------------------------------------
def list_movie_slugs():
    """Return sorted list of movie slugs (without extension)."""
    files = os.listdir(MOVIES_DIR)
    return sorted(f[:-5] for f in files
                  if f.endswith(".json") and not f.endswith(".actors.json"))


def load_movie_graph(slug, by="character"):
    """Load one movie network as a NetworkX graph.

    by='character' -> <slug>.json  (node id = character name)
    by='actor'     -> <slug>.actors.json (node id = actor name)
    """
    fname = slug + (".json" if by == "character" else ".actors.json")
    path = os.path.join(MOVIES_DIR, fname)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with open(path) as fh:
            data = json.load(fh)
        G = nx.node_link_graph(data)
    G.graph.setdefault("slug", slug)
    return G


# ----------------------------------------------------------------------------
# Local LLM via Ollama (runs as a SLURM GPU job; endpoint in logs/ollama_endpoint.txt)
# ----------------------------------------------------------------------------
def ollama_endpoint():
    with open(ENDPOINT_FILE) as fh:
        return fh.read().strip()


def ollama_generate(prompt, model="qwen2.5:14b", system=None, temperature=0.2,
                    endpoint=None, timeout=600, num_predict=None):
    """Single-shot generation against the local Ollama server. Returns text."""
    import requests
    ep = endpoint or ollama_endpoint()
    opts = {"temperature": temperature}
    if num_predict is not None:
        opts["num_predict"] = num_predict
    payload = {"model": model, "prompt": prompt, "stream": False, "options": opts}
    if system:
        payload["system"] = system
    r = requests.post(f"http://{ep}/api/generate", json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json().get("response", "")


# ----------------------------------------------------------------------------
# Plotting helpers (headless-safe: forces the Agg backend).
# ----------------------------------------------------------------------------
def set_style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 150, "font.size": 11,
        "axes.grid": True, "grid.alpha": 0.3, "figure.autolayout": False,
    })


def save_fig(fig, name, dpi=150):
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path
