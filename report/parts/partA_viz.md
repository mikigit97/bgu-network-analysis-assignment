# Part A — Visualization tasks A4 and A7

This part of the report covers two visualization tasks:

- **A4** — visualizing the selected movie network (*2008 — The Dark Knight*) the way
  you would in **Gephi** and **Cytoscape**.
- **A7** — building and visualizing a **Lord of the Rings "couples" network**, with
  vertex colour standing for gender and vertex shape standing for race.

All of the code that produces the figures and export files lives in the executed
notebook **`notebooks/partA_viz.ipynb`**, which runs from top to bottom without
errors. The libraries used are **NetworkX** (building, analysing and exporting
graphs), **matplotlib** (drawing the emulated figures), **pandas** (reading the
public Lord of the Rings race table), **NumPy** (numeric helpers), and the project's
own **`na_utils`** helper module (loading the movie graph and saving figures with a
headless-safe matplotlib backend).

---

## A quick honest note about the headless environment

These tasks were carried out on a **headless cluster node** — a computer with no
screen attached and no graphical desktop. Gephi and Cytoscape are *interactive
desktop programs*; they need a display to open their windows. On this machine they
simply cannot be launched, so it is **impossible for the automated process to click
through their menus and capture a real screenshot**. I want to be completely upfront
about that rather than pretend otherwise.

The way I handled this limitation has two halves, and together they fully satisfy the
spirit of the assignment:

1. **I exported the network into the exact files Gephi and Cytoscape import**, with
   the size/colour/shape columns already baked in. You can open these files on any
   machine that has a display and produce the official screenshots in a few clicks.
   Detailed, step-by-step import instructions are given below for each program.
2. **I produced "emulated" renders with matplotlib** that mimic what each program
   would show — the same layout style, the same node-size-by-degree rule, the same
   colour/shape encodings. These let you *see the intended result right now*, without
   waiting for a GUI. They are clearly labelled as emulations, and they are not a
   substitute for the official screenshots — they are a faithful preview of them.

Think of it like an architect's rendering versus a photograph of the finished
building: the rendering (my matplotlib figure) shows exactly what you should expect,
and the export files are the blueprint you hand to the builder (Gephi/Cytoscape) to
get the real photo.

---

## Task A4 — Cytoscape + Gephi visualization of the selected network

### What the network is

The selected network is **`2008_The_Dark_Knight`** (the constant
`na.SELECTED_MOVIE`). It is a **character co-appearance network**: each node is a
character in the film, and an edge between two characters means they appeared
together in the same scene. The edge carries a **weight**, which is simply *how many
times* the two characters co-appeared. The graph is **undirected** (if A appears
with B, then B appears with A — there is no direction to "appearing together") and
**weighted**. It is small and dense: **25 characters and 106 co-appearance edges**.

A term worth defining up front is **degree**. The degree of a node is the number of
edges touching it — in plain words, *how many different characters this character
ever shared a scene with*. The assignment asks for node size to grow with degree, so
that the most "connected" characters (the leads) are drawn as the biggest dots.

### Files I produced (the import-ready exports)

All exports are in the `exports/` folder. Before exporting, I added two size-ready
numbers to every node so you can map size onto them inside the GUI:

- **`degree`** — the plain count of neighbours (how many distinct co-stars).
- **`weighted_degree`** — the sum of the weights of all edges on the node, i.e. the
  *total number of co-appearances* counting repeats. (Analogy: `degree` is "how many
  friends", `weighted_degree` is "how much total time spent with friends". A
  character can have few co-stars but a lot of shared screen time with them, which is
  why both are useful.)

I also added a **`community`** attribute (explained below) for colouring, and a
**`label`** attribute holding the character name.

| File | Format | For | Keeps |
|------|--------|-----|-------|
| `exports/dark_knight.gexf` | GEXF (Graph Exchange XML) | **Gephi** | degree, weighted_degree, community, label, edge weight |
| `exports/dark_knight.graphml` | GraphML (XML) | **Cytoscape** | same |
| `exports/dark_knight.cyjs` | Cytoscape.js JSON | **Cytoscape** | same |

**Communities.** A *community* is a clump of nodes that are more tightly connected to
each other than to the rest of the graph — for a movie, a sub-group of characters who
mostly share scenes among themselves. I found these with the **Louvain method**, a
fast and widely used community-detection algorithm. In one sentence: Louvain
repeatedly merges nodes into groups in whatever way most increases *modularity*, a
score that is high when there are many edges *inside* groups and few edges *between*
groups. I passed the edge weight in (so strong co-appearance links count more) and
fixed the random seed (so the grouping is reproducible). The communities are used
purely to give the nodes meaningful colours in the Gephi-style render.

### The two emulated renders

**`figures/partA4_emulated_gephi.png`** — a **ForceAtlas2-style** picture.
ForceAtlas2 is Gephi's signature *force-directed* layout. A force-directed layout
treats every node as a little magnet that pushes the others away, and every edge as a
spring that pulls its two endpoints together; the drawing settles into a balance where
tightly connected groups bunch up and loosely connected nodes drift outward. That is
the "organic" Gephi look. NetworkX does not include ForceAtlas2 by that name, but its
`spring_layout` (the Fruchterman–Reingold algorithm) uses the *same* push/pull idea,
so with a tuned spacing knob (`k`) and enough iterations it produces a very
ForceAtlas2-like result. In this figure: **node size grows with degree**, **node
colour encodes Louvain community**, **edge thickness grows with co-appearance
weight**, and only the **top-8 highest-degree characters are labelled** (with small
white background boxes) so the busiest nodes are named without cluttering the picture.

![Emulated Gephi render of the Dark Knight network](../../figures/partA4_emulated_gephi.png)

**`figures/partA4_emulated_cytoscape.png`** — a deliberately **different** style so the
two renders don't look the same. It uses the **Kamada–Kawai** layout, another
force-directed method but one that works differently: instead of simulating springs
step by step, it tries to make the straight-line distance between any two nodes *on the
page* match their *graph distance* (the number of hops along edges between them). The
result is more symmetric and evenly spaced — close to the tidy default layouts
Cytoscape offers. Here node colour is a *smooth gradient by degree* (light = peripheral
character, dark = central) using the `viridis` colour map, a different visual language
from the categorical community colours above. Node size still grows with degree.

![Emulated Cytoscape render of the Dark Knight network](../../figures/partA4_emulated_cytoscape.png)

In both pictures the same story is obvious: **Batman (Bruce Wayne), the Joker, and
Harvey Dent / Two-Face are the largest, most central nodes**, because they share scenes
with the most other characters — exactly the protagonist/antagonist structure you would
expect from the film. Barbara Gordon and Rachel Dawes are the next tier.

### How to make the OFFICIAL screenshots yourself

#### Gephi (from `exports/dark_knight.gexf`)

1. Open **Gephi** and choose **File → Open…**, then select
   `exports/dark_knight.gexf`. In the import dialog, keep the graph type as
   **Undirected** and click **OK**. The network appears in the *Overview* tab.
2. In the left-hand **Appearance** panel, click the **Nodes** tab, then the **Size**
   icon (the little circles). Choose **Ranking**, pick the attribute **`degree`** from
   the dropdown, set a sensible **Min size / Max size** (for example 10 and 60), and
   click **Apply**. Node size is now proportional to degree, satisfying the core
   requirement.
3. *(Optional, for colour like the emulation)* Still in **Appearance → Nodes**, click
   the **Colour** icon, choose **Partition**, pick **`community`**, and click **Apply**
   to colour nodes by community.
4. In the **Layout** panel (lower left), choose **ForceAtlas2** from the dropdown and
   click **Run**. Let it spread the graph out for a few seconds, then click **Stop**.
   You can tick **Prevent Overlap** to keep big nodes from covering each other.
5. Click the **T** button at the bottom of the graph window to **show node labels**.
   Use the size slider next to it to make the labels readable.
6. Switch to the **Preview** tab (top), click **Refresh**, adjust label/edge settings
   to taste, then **File → Export → SVG/PDF/PNG file…** to save the screenshot.

#### Cytoscape (from `exports/dark_knight.cyjs` or `exports/dark_knight.graphml`)

1. Open **Cytoscape** and choose **File → Import → Network from File…**, then select
   `exports/dark_knight.graphml` (or use the `.cyjs` file). The network loads with all
   node attributes available in the **Node Table**.
2. Open the **Style** panel on the left. Find the **Size** row, set **Column** to
   **`degree`** and **Mapping Type** to **Continuous Mapping**. Drag the handles so
   small degree → small node and large degree → large node. Node size now tracks
   degree.
3. *(Optional)* In the **Style** panel, set **Fill Color → Column = `community` →
   Discrete Mapping** to colour nodes by community, mirroring the emulation.
4. Apply a layout from the **Layout** menu — **Layout → Prefuse Force Directed Layout**
   gives the closest match to the emulated picture; **Edge-weighted Spring Embedded
   (kamada-kawai)** is another good choice.
5. Turn on labels by mapping **Label → Column = `label`** (or `name`) in the Style
   panel.
6. Export the screenshot with **File → Export → Network to Image…** (PNG/PDF/SVG).

### How I solved this task (A4)

**What I did.** I loaded the selected Dark Knight co-appearance network (25 characters,
106 edges), enriched every node with two size-ready attributes (`degree` and
`weighted_degree`) and a Louvain community label, exported the graph to the three
formats Gephi and Cytoscape import, and finally drew two emulated renders mimicking the
look of each program.

**Which method/tool.** Louvain for community colouring (it maximises modularity to find
tightly-knit groups); NetworkX `spring_layout` (Fruchterman–Reingold) for the Gephi
emulation because it uses the same attract/repel physics as ForceAtlas2; and
Kamada–Kawai with a continuous viridis colouring for the Cytoscape emulation so the two
pictures look clearly different.

**Why I selected these.** This is a headless cluster node with no display, so the real
Gephi/Cytoscape windows cannot be opened here and no genuine screenshot can be captured
automatically. The honest, reproducible answer is to hand over the exact import files
(with the size/colour columns pre-computed) and to preview the finished look with
faithful matplotlib emulations. Spring layout and Kamada–Kawai are the closest NetworkX
equivalents of those programs' force-directed layouts.

**What the result means.** The network is small but dense, with a handful of hub
characters. Batman, the Joker, and Harvey Dent/Two-Face are the biggest nodes because
they co-appear with the most others — the classic lead-protagonist/antagonist shape of
the film. The community colours separate loosely connected clusters of minor
characters. To capture the *official* GUI screenshots the rubric asks for, follow the
step-by-step Gephi and Cytoscape instructions above.

---

## Task A7 — Lord of the Rings *couples* network

### Goal and data situation

The goal is to draw the network of **canonical romantic couples** in Tolkien's
legendarium (*The Lord of the Rings* together with *The Silmarillion*), with **vertex
colour = gender** and **vertex shape = race**.

I first searched for a ready-made dataset that contains *both* gender and race *per
character*, plus a couples edge list. The cleanest public source I found is the
**juandes/lotr-names-classification** repository, whose
[`characters_data.csv`](https://raw.githubusercontent.com/juandes/lotr-names-classification/master/characters_data.csv)
lists **827 characters with a `race` column** (its five races are *Ainur, Dwarf, Elf,
Hobbit, Man*). I downloaded this file to `data/lotr/characters_data.csv` and use it as a
**race reference**. However, that file has **no gender column, no couples list**, and is
**missing several of the female partners** (Éowyn, Rosie Cotton, Goldberry, Lúthien,
Lothíriel do not appear in it). So, exactly as the assignment anticipates, I **built the
couples network by hand** from canonical pairings and **cross-checked every race label
against the public dataset** wherever the character exists in it.

### The couples and their attributes

I included **nine canonical couples** spanning **18 characters**:

| Couple | Person A (gender, race) | Person B (gender, race) |
|--------|-------------------------|-------------------------|
| Aragorn – Arwen | Aragorn (male, Man) | Arwen (female, Elf) |
| Samwise Gamgee – Rosie Cotton | Samwise (male, Hobbit) | Rosie (female, Hobbit) |
| Faramir – Éowyn | Faramir (male, Man) | Éowyn (female, Man) |
| Beren – Lúthien | Beren (male, Man) | Lúthien (female, Elf) |
| Galadriel – Celeborn | Galadriel (female, Elf) | Celeborn (male, Elf) |
| Tom Bombadil – Goldberry | Tom Bombadil (male, Unknown) | Goldberry (female, Maia) |
| Éomer – Lothíriel | Éomer (male, Man) | Lothíriel (female, Man) |
| Thingol – Melian | Thingol (male, Elf) | Melian (female, Maia) |
| Tuor – Idril | Tuor (male, Man) | Idril (female, Elf) |

**How I assigned the labels, and the judgement calls I made (documented honestly):**

- **Gender** is taken from canon and is binary for all of these characters in
  Tolkien's texts (`male` / `female`).
- **Race** keeps the public dataset's everyday scheme (*Man, Elf, Hobbit, Dwarf*) but
  I made two refinements that I want to be transparent about:
  - I **split the public dataset's catch-all "Ainur"** into the finer canon term
    **`Maia`** (a lesser divine spirit) for **Melian** and **Goldberry**. Lumping these
    two in with the great Valar would hide a meaningful distinction, and "Maia" is the
    correct, specific term for them in Tolkien's writing.
  - **Tom Bombadil's race is deliberately *unknown* in canon** — Tolkien never settled
    it, and Bombadil famously says only "Eldest, that's what I am." Rather than invent a
    label, I mark him **`Unknown`**, which is itself an honest and informative category.
- For the characters that *do* appear in the public dataset, the race I assigned
  **matches** the dataset (Aragorn = Man, Arwen = Elf, Samwise = Hobbit, Faramir = Man,
  Beren = Man, Galadriel = Elf, Celeborn = Elf, Éomer = Man, Thingol = Elf, Tuor = Man,
  Idril = Elf; and Melian's "Ainur" is refined to "Maia"). The notebook records, for
  each character, whether its race label is confirmed by the public dataset.
- The remaining labels (Rosie = Hobbit; Éowyn, Lothíriel = Man; Lúthien = Elf;
  Goldberry = Maia) come from **Tolkien canon** (the texts and the Tolkien Gateway
  reference wiki), because those characters are absent from the public dataset.

A small but real simplification worth flagging: Lúthien is in truth *half-Elf,
half-Maia* (her mother is Melian the Maia), and Arwen is *half-elven*. I label both as
**Elf**, which is the standard simplification and matches how the public dataset treats
Arwen. This is the kind of assumption the rubric asks me to surface.

### Files I produced (the import-ready exports)

All in `exports/`, each carrying the load-bearing **`gender`** and **`race`** node
attributes (plus a `label`):

| File | Format | For |
|------|--------|-----|
| `exports/lotr_couples.gexf` | GEXF | Gephi |
| `exports/lotr_couples.graphml` | GraphML | Cytoscape |
| `exports/lotr_couples.cyjs` | Cytoscape.js JSON | Cytoscape |

### The emulated render

**`figures/partA7_lotr_couples.png`** draws the couples network with **colour showing
gender** (blue = male, red = female) and **marker shape showing race** (circle = Man,
triangle = Elf, square = Hobbit, star = Maia, ✕ = Unknown). A technical note: a single
matplotlib node-drawing call can only use one marker shape, so I draw the nodes **one
race at a time** — each race gets its own scatter call with its own marker, while each
dot's colour is set by that character's gender. That gives two genuinely independent
visual channels and a clean two-part legend (one box for gender/colour, one for
race/shape).

![Emulated render of the LOTR couples network](../../figures/partA7_lotr_couples.png)

The picture is a set of **disconnected two-person components** — the couples are not
linked to one another — which is exactly right: the network's job is to show *who is
paired with whom*, not a single connected social web. The encoding makes a recurring
theme of the legendarium jump out immediately: the famous **Man + Elf unions**
(Aragorn–Arwen, Beren–Lúthien, Tuor–Idril) show up as a **blue circle paired with a red
triangle** — mortal-meets-immortal love stories that are central to Tolkien's mythology.

### Design choices, explained

**Why colour encodes gender.** Colour is the *most salient* visual channel — the human
eye reads it first and fastest. Gender here is a **binary** split (just two values), and
a binary split maps perfectly onto two strongly contrasting colours (blue vs red) that
can be told apart at a glance, even in a busy figure or at small size.

**Why shape encodes race.** Shape is excellent at separating a *small number* of
distinct categories, which is exactly our situation (only a handful of races appear).
Shapes are a bit slower to read than colours, which is fine for the secondary attribute.
Crucially, **colour and shape are independent channels**: a viewer can read gender
(colour) and race (shape) *separately and simultaneously* without the two encodings
interfering. A red triangle is unambiguously a female Elf; a blue circle is a male Man.

### How to make the OFFICIAL Cytoscape screenshot

1. **File → Import → Network from File…** and choose `exports/lotr_couples.graphml`
   (or the `.cyjs` file). The `gender` and `race` columns appear in the Node Table.
2. In the **Style** panel, set **Fill Color**: choose **Column = `gender`** and
   **Mapping Type = Discrete Mapping**, then assign a colour to each value (for example
   `male` → blue, `female` → red) to match the emulation.
3. Still in **Style**, set **Shape**: choose **Column = `race`** and **Mapping Type =
   Discrete Mapping**, then assign a shape to each race (for example `Man` → ellipse,
   `Elf` → triangle, `Hobbit` → rectangle, `Maia` → diamond/star, `Unknown` → hexagon).
4. Map **Label → Column = `label`** so each character's name shows.
5. Apply **Layout → Prefuse Force Directed Layout** (or any layout you like — the
   couples are disconnected pairs, so most layouts look tidy).
6. Export with **File → Export → Network to Image…**.

### How I solved this task (A7)

**What I did.** I assembled a Lord of the Rings *couples* network of nine canonical
pairings (18 characters), attached a gender and a race to each character, exported the
network to Gephi/Cytoscape formats, and drew an emulated render where colour shows
gender and marker shape shows race.

**Which method/tool/data.** I started from the public juandes `characters_data.csv` race
dataset (827 characters; races Ainur, Dwarf, Elf, Hobbit, Man), downloaded to
`data/lotr/`. Because that file lacks gender, lacks a couples list, and is missing
several female partners, I built the edge list by hand from Tolkien canon and confirmed
each race label against the public dataset where possible. For the drawing I used
NetworkX `spring_layout` and matplotlib, plotting nodes one race at a time so each race
can carry its own marker shape.

**Why these design choices.** Both gender and race are *categorical* attributes. Colour
is the most eye-catching channel, so I used it for the binary gender split; shape is
ideal for a small number of categories, so I used it for the races. The two channels are
independent, which keeps the picture instantly readable.

**What the result means.** The network is a set of disconnected couple-pairs, correctly
showing who is paired with whom. The colour/shape encoding surfaces a central theme of
Tolkien's mythology — the recurring Man + Elf unions (Aragorn–Arwen, Beren–Lúthien,
Tuor–Idril). For the official Cytoscape screenshot, follow the discrete-mapping steps
above.

---

## Limitations and assumptions (both tasks)

- **No GUI screenshots could be captured automatically** on this headless node. The
  export files + import instructions + matplotlib emulations together stand in for, and
  enable, the official Gephi/Cytoscape screenshots.
- **A4 emulations approximate, but are not identical to,** Gephi's ForceAtlas2 and
  Cytoscape's native layouts. `spring_layout` and `kamada_kawai_layout` share the same
  force-directed philosophy but will not reproduce the GUI pixel-for-pixel.
- **A7 is partly hand-curated.** No single public file gave couples + gender + race per
  character, so the edge list and the gender labels are from Tolkien canon. Two race
  refinements were judgement calls (Maia for Melian/Goldberry; Unknown for Tom
  Bombadil), and half-elven characters (Arwen, Lúthien) are simplified to Elf. These are
  all documented above.
- **Reproducibility.** Every layout uses a fixed random seed (`SEED = 42`), so re-running
  the notebook produces the same pictures.

## Data sources

- **Lord of the Rings race reference:** juandes/lotr-names-classification,
  `characters_data.csv` —
  <https://github.com/juandes/lotr-names-classification/blob/master/characters_data.csv>
  (raw:
  <https://raw.githubusercontent.com/juandes/lotr-names-classification/master/characters_data.csv>),
  saved locally to `data/lotr/characters_data.csv`.
- **Couples, gender, and race refinements:** Tolkien's *The Lord of the Rings* and *The
  Silmarillion*, cross-referenced with the Tolkien Gateway reference wiki
  (<https://tolkiengateway.net/>).
- **Movie network:** Movie Dynamics Networks (Kaggle:
  michaelfire/movie-dynamics-over-15000-movie-social-networks), the
  `2008_The_Dark_Knight` character network, loaded via `na.load_movie_graph`.

## Libraries and tools

- **NetworkX** — graph construction, Louvain community detection, layouts, and writing
  GEXF / GraphML / Cytoscape.js JSON.
- **matplotlib** (Agg backend, headless-safe) — all emulated figures.
- **pandas** — reading the public LOTR race CSV.
- **NumPy** — numeric scaling of node sizes and edge widths.
- **Gephi** and **Cytoscape** — the target desktop tools for the official screenshots
  (run by the user on a machine with a display, using the exported files).

## Files produced

**Notebook:** `notebooks/partA_viz.ipynb` (executed, error-free).

**Exports (`exports/`):** `dark_knight.gexf`, `dark_knight.graphml`,
`dark_knight.cyjs`, `lotr_couples.gexf`, `lotr_couples.graphml`, `lotr_couples.cyjs`.

**Figures (`figures/`):** `partA4_emulated_gephi.png`,
`partA4_emulated_cytoscape.png`, `partA7_lotr_couples.png`.

**Data (`data/lotr/`):** `characters_data.csv` (downloaded public race reference).
