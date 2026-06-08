"""
Build ONE self-contained submission notebook from the five per-part notebooks.

- Concatenates all cells (markdown + code + embedded outputs) under section headers.
- Embeds the official Gephi/Cytoscape screenshots as base64 ATTACHMENTS so the single
  .ipynb file shows every image even when uploaded on its own (no external figure files).
- Leaves clearly-marked placeholders for screenshots not provided yet.
- Does NOT modify or delete the per-part notebooks.
"""
import os, base64
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell

BASE = "/home/mickaelz/Network analysis"
os.chdir(BASE)
OUT = "notebooks/combined_submission.ipynb"

PARTS = [
    ("Part A — Movie Network Core (A1.1, A1.2, A2, A3, A5)", "notebooks/partA_core.ipynb"),
    ("Part A — Visualization (A4: Cytoscape + Gephi · A7: Lord of the Rings)", "notebooks/partA_viz.ipynb"),
    ("Part A6 — Large Chess Network (~429.7M edges)", "notebooks/partA6_chess.ipynb"),
    ("Part B — Directed Link Prediction (+ Bonus: Future Links)", "notebooks/partB_linkpred.ipynb"),
    ("Part C — Enron Manager Detection with a Local LLM", "notebooks/partC_enron_llm.ipynb"),
]

# Official GUI screenshots to embed (title, image path, caption). Missing files -> placeholder.
SCREENSHOTS = {
    "a4_gephi": ("A4 — Official **Gephi** screenshot (Dark Knight)",
                 "figures/partA4_gephi_screenshot.png",
                 "Rendered in Gephi 0.11.2 from `exports/dark_knight.gexf`. Node **size mapped to degree** "
                 "(Appearance ▸ Nodes ▸ Ranking ▸ degree); **ForceAtlas 2** layout; colour by community. "
                 "Context panel confirms 25 nodes / 106 edges, undirected — the Dark Knight character network."),
    "a4_cyto": ("A4 — Official **Cytoscape** screenshot (Dark Knight)",
                "figures/partA4_cytoscape_screenshot.png",
                "Rendered in Cytoscape from `exports/dark_knight.cyjs`/`.graphml`, with node size mapped to the "
                "`degree` attribute (Style ▸ Size ▸ Continuous Mapping)."),
    "a7_cyto": ("A7 — Official **Cytoscape** screenshot (Lord of the Rings couples)",
                "figures/partA7_cytoscape_screenshot.png",
                "Rendered in Cytoscape from `exports/lotr_couples.cyjs`, with node **fill colour = gender** "
                "(discrete mapping on `gender`) and node **shape = race** (discrete mapping on `race`)."),
}

def img_cell(key):
    title, path, caption = SCREENSHOTS[key]
    if os.path.exists(path):
        b64 = base64.b64encode(open(path, "rb").read()).decode()
        fn = os.path.basename(path)
        c = new_markdown_cell(f"#### {title}\n\n![{title}](attachment:{fn})\n\n*{caption}*")
        c.attachments = {fn: {"image/png": b64}}
        return c
    # placeholder if not provided yet
    return new_markdown_cell(
        f"#### {title}\n\n> **[screenshot to be inserted]** — open the matching file in `exports/` in the GUI "
        f"as described in the section above, capture the view, and it will appear here.\n\n*{caption}*")

TITLE = """# Network Analysis — Combined Submission Notebook

**The Art of Analyzing Big Data — Network Analysis, Visualization, Graph Embeddings, and Link Prediction**

**Author:** Mickael Zeitoun (mickaelz@post.bgu.ac.il)
**Repository:** https://github.com/mikigit97/bgu-network-analysis-assignment

This single notebook collects **every part** of the assignment. Each part below is assembled from its
own per-part notebook (which was executed top-to-bottom **without errors**), so all code, outputs,
figures, and the required **"How I solved this task"** explanations are included here with their results
already embedded. The official Gephi/Cytoscape screenshots are embedded directly in this file.

### A one-minute vocabulary primer (terms used throughout)

- **Network / graph:** dots called **nodes** (a movie character, a chess player, a subreddit, an
  employee) joined by lines called **edges** (a relationship: "appeared together", "played a game",
  "linked to", "emailed").
- **Directed vs undirected edge:** a directed edge has a direction (A→B, e.g. "A emailed B"); an
  undirected edge has none.
- **Degree:** how many edges a node has. **Weighted degree / strength:** the same but adding up edge
  weights (total games, total co-appearances…).
- **Centrality:** any score for "how important / well-connected is this node?"
- **Embedding:** turning a graph (or node) into a list of numbers (a **vector**) so ordinary maths and
  machine learning can compare them.
- **Link prediction:** training a model to guess which not-yet-present edges are most likely real / next.

### What each part answers

| Task(s) | Points | Section |
|---|:---:|---|
| A1.1 degree distribution · A1.2 graph embeddings of all 15,538 movies · A2 top-12 (circular) · A3 PageRank/triangles/paths · A5 ego network | 25 | Part A — Movie Network Core |
| A4 Cytoscape + Gephi · A7 Lord of the Rings couples | 10 | Part A — Visualization |
| A6 large chess network (429M edges) | 5 | Part A6 |
| B1.1–B1.7 directed link prediction · Bonus future links | 25 + 5 | Part B |
| C1.1–C1.6 Enron managers + local LLM | 35 | Part C |

### Notes

- **How to read it:** every part has its own self-contained section with explanations in plain English.
  Outputs are already embedded; you do not need to re-run anything to read the results.
- **Reproducibility:** all randomness is seeded (42); the per-part notebooks remain in `notebooks/` and
  the combined report in `report/REPORT.md`.
- **Visualization tasks (A4/A7):** these used Cytoscape and Gephi (desktop GUI apps); the official
  screenshots are embedded in the Visualization section, alongside matplotlib "emulated" previews and
  the exact import steps; ready-to-import network files are in `exports/`.

---
"""

out = new_notebook()
out.cells.append(new_markdown_cell(TITLE))
for name, path in PARTS:
    nb = nbformat.read(path, as_version=4)
    out.cells.append(new_markdown_cell(
        f"---\n# {name}\n\n*Assembled from `{path}` — executed top-to-bottom without errors; outputs preserved.*"))
    out.cells.extend(nb.cells)

def find_idx(substrings, start=0):
    """First markdown cell index whose source contains ALL given substrings (case-insensitive)."""
    for i in range(start, len(out.cells)):
        c = out.cells[i]
        if c.cell_type == "markdown":
            s = c.source.lower()
            if all(sub.lower() in s for sub in substrings):
                return i
    return None

# Group the three official GUI screenshots in one clearly-labeled subsection at the END of the
# Visualization part (right before the Part A6 divider). The emulated previews + import steps stay
# in their A4/A7 spots above; these are the real Gephi/Cytoscape captures.
a6_div_idx = find_idx(["notebooks/parta6_chess.ipynb"])    # start of Part A6 == end of Visualization
heading = new_markdown_cell(
    "## Official Gephi / Cytoscape screenshots (A4 & A7)\n\n"
    "These are the **real captures from the desktop GUI apps**, fulfilling the A4 requirement of "
    "screenshots from *both* Gephi and Cytoscape and the A7 Cytoscape requirement. Each was produced by "
    "opening the matching ready-to-import file from `exports/` and applying the styling described above.")
block = [heading, img_cell("a4_gephi"), img_cell("a4_cyto"), img_cell("a7_cyto")]
if a6_div_idx is not None:
    out.cells[a6_div_idx:a6_div_idx] = block
else:
    out.cells.extend(block)

out.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
out.metadata["language_info"] = {"name": "python", "version": "3.9"}

nbformat.write(out, OUT)
n_code = sum(1 for c in out.cells if c.cell_type == "code")
n_md = sum(1 for c in out.cells if c.cell_type == "markdown")
embedded = [k for k in SCREENSHOTS if os.path.exists(SCREENSHOTS[k][1])]
print(f"wrote {OUT}: {len(out.cells)} cells ({n_code} code, {n_md} markdown)")
print(f"screenshots embedded: {embedded}")
print(f"a6 divider index (screenshot block inserted before it): {a6_div_idx}")
print(f"size: {os.path.getsize(OUT)/1e6:.1f} MB")
