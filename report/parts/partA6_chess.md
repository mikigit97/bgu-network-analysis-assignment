# Part A6 — The Large Chess Network (Free Internet Chess Server)

This is the written report for **Task A6**. The goal is to find the **top-10 most
central players** in the Free Internet Chess Server (FICS) game network and to
**draw a meaningful slice** of it. The whole challenge of this task is *size*: the
network has about **429,747,477 edges**, far too many to load into an ordinary
in-memory graph library. So most of the work — and most of this writeup — is about
*how I handled that size*. The runnable, fully-executed code with all outputs lives
in the companion notebook `notebooks/partA6_chess.ipynb`.

I have written this report the way I like explanations: **long but plain**. Every
technical term is defined in everyday words the first time it appears, and I build
each idea up from the underlying intuition rather than just stating a conclusion.

**Data source (University of Washington):**
<http://dynamics.cs.washington.edu/nobackup/chess/fcis.tar.gz> — a 6.85 GB
gzip-compressed tar archive. ("gzip" is a common file-compression format; a "tar
archive" is one file that bundles many files together, like a zip.)

**Libraries used:** `polars` (fast streaming dataframes), `python-igraph`
(imported as `igraph`; a graph library written in C, so it is fast on big graphs),
`networkx` (used only to lay out and draw the small picture), `numpy` / `pandas`
(number crunching and small tables), `scipy` (one statistics function — Spearman
rank correlation), and `matplotlib` (plots, forced to the headless **Agg** backend
because the cluster has no screen).

---

## 1. How I handled the network size (the core deliverable)

### The problem, in concrete numbers

The raw download expands into `data/chess/FCIS/`, containing two plain-text **CSV**
files. (CSV = "comma-separated values", a text table where each line is a row and
commas separate the columns.)

- `fcis_chess.interactions.csv` — **15.4 gigabytes**, with **429,747,477 rows**.
  Each row is `datetime, src_id, dst_id`: a timestamp, the *source* player of that
  interaction, and the *destination* player. This is an **edge list** — literally a
  long list, one line per edge, naming the two endpoints. Crucially, **each game is
  stored twice**: once as `(a, b)` and once as `(b, a)`. So 429.7 million rows are
  about **214.9 million actual games**.
- `fcis_chess.vertices.csv` — one row per player: `mindate, v_id, maxdate` (first
  activity date, username, last activity date). There are **519,584 players**.
  Players are identified by their **username** (a short text handle like `mscp` or
  `GriffyJr`), not by a number.

Why can this not just be loaded? NetworkX (a pure-Python graph library) stores each
edge as Python objects inside nested dictionaries, costing on the order of a few
hundred bytes per edge. Multiply by 429 million and you need **hundreds of
gigabytes** of RAM — far more than the machine has. So `nx.read_edgelist(...)` is
simply off the table.

### The idea: *streaming aggregation* instead of loading the graph

Here is the key trick. To answer "who plays the most games?" and "what does the
busy core look like?", we do **not** need all 429 million individual edges in
memory at once. We only need *summaries* — and a summary can be computed by reading
the file **once, in small chunks**, keeping a tiny running tally, and throwing each
chunk away after counting it.

That technique is **streaming aggregation**. "Streaming" means reading the file
like water through a pipe — a little at a time — rather than pouring the whole lake
into a bucket. "Aggregation" means *combining many rows into a few summary numbers*
(counts, sums) as they flow past.

> **Tiny analogy.** To count how many cars of each colour pass your window in a
> day, you do not photograph all 100,000 cars and store the photos. You keep a
> small notepad with one tally mark per colour and update it as each car passes. At
> the end you have the counts, and you never needed a warehouse of photos.
> Streaming aggregation is exactly this: the file is the stream of cars; the
> running counts are the notepad.

`polars` does this with `scan_csv(...).collect(engine="streaming")`. `scan_csv`
does **not** read the file immediately — it builds a *plan* (a recipe) of what we
want. `collect(engine="streaming")` then runs that plan chunk-by-chunk with bounded
memory.

### This heavy step was already run as a SLURM batch job (and is not re-run here)

Reading 15.4 GB still takes a couple of minutes, so the aggregation was run **once**
ahead of time as a **SLURM batch job**. (SLURM is the cluster's scheduling system:
you hand it a script, it finds a free machine, runs the job unattended, and saves
the output.) The job asked for 8 CPU cores and 48 GB of RAM, finished in **1 minute
51 seconds**, and peaked at only about **22 GB** of memory — comfortably under the
limit, precisely *because* it streams instead of loading everything. The exact code
is `scripts/chess_aggregate.py`; the job description is
`scripts/chess_aggregate.sbatch`.

That job produced **two small files** the notebook loads instantly. They are stored
as **Parquet** (a compact, column-oriented binary table format; "column-oriented"
means it stores all values of one column together, which reads and compresses very
efficiently):

1. `data/chess/player_stats.parquet` — columns `(player, games)`. The **exact**
   total number of games each of the **519,583** players ever played, over the
   **full** 429-million-edge graph. Nothing sampled, nothing approximated.
2. `data/chess/edges_top.parquet` — columns `(src_id, dst_id, w)`. The **induced
   subgraph** on the **top-5000 most active players**: every game played *between
   two of those 5000 players*, with `w` = how many games that ordered pair played.
   About **7.99 million** directed edges.

### Why this two-file design is the right call (justification)

- **The per-player game counts are EXACT, not sampled.** Because every game appears
  as both `(a, b)` and `(b, a)`, grouping the 429M rows by the *source* column and
  counting rows per source gives each player's exact total games. Grouping by a
  single column produces only ~519k groups — a tiny tally that fits easily in
  memory. So our *primary* centrality is computed on the **whole** graph with no
  approximation at all.
- **The top-5000 induced subgraph captures the players who matter for structure.**
  "Induced subgraph" means: pick a set of players, then keep *only* the edges whose
  **both** endpoints are inside that set (you "induce" the sub-network from the
  chosen nodes). We pick the 5000 busiest players. Why is that sound rather than an
  arbitrary sample? Because in interaction networks the most structurally central
  players are *overwhelmingly* among the most active — you cannot be a hub that
  everything flows through if you barely play. So restricting structural analysis to
  the busy core keeps the players who would top any centrality ranking, while
  shrinking 429M edges to a graph small enough for real algorithms in under a
  second. We *measure* this claim below: the PageRank top-10 shares **9 of 10**
  names with the exact games top-10.
- **The 15 GB CSV is never re-streamed inside the notebook.** The streaming code is
  shown for the record but guarded by `RUN_HEAVY = False`, so it does not execute.
  The notebook runs entirely from the two small Parquet files: fast and
  reproducible.

---

## 2. Top-10 most central players (two notions, compared)

"**Centrality**" is a family of scores that try to capture *how important a node is*
inside a network. There is no single correct definition — different notions of
"important" give different rankings — so the task asks for **at least two** and a
comparison. I use three, of two fundamentally different kinds.

### 2a. PRIMARY measure — weighted degree = total games (EXACT, full graph)

The **degree** of a node is normally "how many edges touch it". When edges carry
weights, the **weighted degree** (also called **strength**) is the *sum* of those
weights. Here an edge's weight is a game count, so a player's weighted degree is
their **total number of games** — exactly the `games` column computed over the
entire 429-million-edge graph.

> **Tiny example.** If Alice played Bob 3 times and Carol 5 times, Alice's *plain*
> degree is 2 (two distinct opponents) but her *weighted* degree (strength) is
> 3 + 5 = 8 (total games). We report the weighted version.

This is the most trustworthy ranking: exact, using every game ever played.

| Rank | Player | Total games (weighted degree) |
|---:|:---|---:|
| 1 | `mscp` | 1,622,052 |
| 2 | `inemuri` | 1,410,447 |
| 3 | `IFDThor` | 856,922 |
| 4 | `GriffyJr` | 798,206 |
| 5 | `GriffySr` | 738,643 |
| 6 | `callipygian` | 505,557 |
| 7 | `BabyLurking` | 390,377 |
| 8 | `parrot` | 314,696 |
| 9 | `AndreD` | 283,455 |
| 10 | `Uirapuru` | 272,717 |

### 2b. STRUCTURAL measures — PageRank, degree, strength on the top-5000 subgraph

The game-count ranking only knows *how much* someone played, not *whom against*.
**Structural** centrality looks at the *shape of the connections*. We build the
top-5000 graph in `igraph` and compute three things.

**Choice: the graph is treated as UNDIRECTED.** The raw data is directed (it stores
`(a,b)` and `(b,a)` separately), but a chess game has no real "direction" — A vs B
is the same encounter as B vs A. So we **collapse** each pair of opposite-direction
edges into one undirected edge whose weight is the sum (total games that pair
played). This halves the edge count (~8.0M directed -> ~4.0M undirected) and matches
the real-world meaning. We state this as a deliberate choice.

The three structural scores, in plain English:

- **PageRank** — originally Google's web-page ranking idea. Picture a random player
  who keeps hopping from opponent to opponent, more likely along *heavier*
  (more-games) edges. PageRank is the long-run fraction of time spent at each
  player. You score high if **many** players (especially other high-scoring
  players) play you a lot. It rewards being well-connected to other hubs, not just
  being busy.
  > *Tiny example:* a player with only one opponent — but that opponent is the
  > single busiest hub — can out-rank a player with several minor opponents,
  > because PageRank "flows" importance in from important neighbours.
- **Degree (here = number of distinct opponents)** — how many *different* people
  this player faced *within the top-5000 set*. Rewards *variety* of opponents.
- **Strength (weighted degree within the subgraph)** — total games played *against
  other top-5000 players*. Like the primary measure but restricted to the busy
  core, so it can differ from the full-graph total.

**Performance note (important).** The subgraph has millions of edges. PageRank,
degree, and strength are cheap on `igraph` (well under a second). But **path-based
measures like betweenness and closeness are O(V·E)** — their cost grows with nodes
*times* edges — so on ~4 million edges they would run for an impractically long
time. We therefore **do not** compute betweenness/closeness on the full subgraph
(the task explicitly warns against it). If a path measure were needed, the correct
move is to first shrink the graph (keep only heavy edges, or the top ~300–500
players) and say so. PageRank already gives us a strong path-flavoured ranking
cheaply.

**Top-10 by PageRank (undirected top-5000 subgraph):**

| Rank | Player | PageRank | Distinct opp. | Strength (subgraph) | Total games (full) |
|---:|:---|---:|---:|---:|---:|
| 1 | `mscp` | 0.00503 | 2,683 | 1,161,538 | 1,622,052 |
| 2 | `inemuri` | 0.00396 | 2,154 | 723,526 | 1,410,447 |
| 3 | `IFDThor` | 0.00325 | 1,863 | 727,022 | 856,922 |
| 4 | `GriffyJr` | 0.00275 | 2,048 | 649,868 | 798,206 |
| 5 | `GriffySr` | 0.00201 | 1,825 | 462,440 | 738,643 |
| 6 | `AndreD` | 0.00158 | 1,624 | 411,984 | 283,455 |
| 7 | `Uirapuru` | 0.00141 | 3,210 | 334,558 | 272,717 |
| 8 | `mrlighting` | 0.00138 | 2,158 | 346,282 | 252,289 |
| 9 | `BabyLurking` | 0.00129 | 1,472 | 240,164 | 390,377 |
| 10 | `callipygian` | 0.00122 | 1,686 | 242,458 | 505,557 |

**Top-10 by DEGREE (most distinct opponents within the top-5000):**

| Rank | Player | Distinct opponents | Strength | PageRank | Total games (full) |
|---:|:---|---:|---:|---:|---:|
| 1 | `Heidrun` | 3,690 | 116,688 | 0.00053 | 108,494 |
| 2 | `blore` | 3,526 | 74,918 | 0.00041 | 79,688 |
| 3 | `andreasw` | 3,517 | 107,504 | 0.00054 | 106,793 |
| 4 | `felipedj` | 3,459 | 163,782 | 0.00072 | 136,267 |
| 5 | `Carl` | 3,393 | 99,704 | 0.00049 | 93,098 |
| 6 | `korrin` | 3,383 | 33,092 | 0.00020 | 38,970 |
| 7 | `monacan` | 3,375 | 122,380 | 0.00066 | 128,641 |
| 8 | `sphinx` | 3,349 | 35,526 | 0.00021 | 41,704 |
| 9 | `Pushkin` | 3,345 | 78,670 | 0.00048 | 103,966 |
| 10 | `naomi` | 3,342 | 111,008 | 0.00062 | 117,256 |

### 2c. Comparing the rankings — do the notions agree?

The interesting question: **do different definitions of "central" pick the same
people?** We measure the *overlap* of the top-10 lists and the **Spearman rank
correlation** between the full scores. (Spearman correlation = replace each score by
its *rank* — 1st, 2nd, 3rd … — and see how well the two rank orders line up. It runs
from +1 "identical ordering" through 0 "no relation" to −1 "exactly reversed". We use
ranks because these scores live on wildly different scales.)

| Comparison | Result |
|:---|---:|
| Top-10 overlap: weighted-degree (games) vs **PageRank** | **9 / 10** |
| Top-10 overlap: weighted-degree (games) vs strength | 7 / 10 |
| Top-10 overlap: weighted-degree (games) vs degree | **0 / 10** |
| Top-10 overlap: PageRank vs degree | 0 / 10 |
| Spearman: PageRank vs total games | **+0.897** |
| Spearman: strength vs total games | +0.808 |
| Spearman: PageRank vs strength | +0.958 |
| Spearman: degree vs total games | +0.378 |
| Spearman: degree vs PageRank | +0.499 |

**What the comparison shows.**

- **PageRank and total-games agree almost perfectly at the top.** Their top-10 lists
  share **9 of 10** names, and across all 5000 players their rank correlation is
  about **+0.90**. This is the empirical proof of the Section 1 justification: the
  structurally central players really are the most active ones, so restricting
  structural analysis to the busy core loses essentially nobody who would have
  topped the ranking. Strength agrees even more tightly with PageRank (rho ≈
  **+0.96**), as both reward heavy, well-connected play.
- **Plain degree (distinct opponents) tells a *different* story.** Its top-10 has
  **zero** overlap with the games top-10, and it correlates only weakly with the
  others (rho ≈ +0.4–0.5). The players with the *most distinct opponents* (`Heidrun`,
  `blore`, `andreasw`, each facing ~3,300–3,700 different people) are not the players
  with the most *games*. A natural reading: these are accounts that play a *huge
  variety* of opponents a *few* times each — the fingerprint of automated pairing
  engines or popular "open challenge" accounts — whereas the games leaders (`mscp`,
  `inemuri`) reach enormous totals by playing a narrower set of opponents *very*
  many times.
  > **Misconception to avoid:** "more games must mean more opponents." Not here.
  > Volume (strength) and variety (degree) are genuinely different axes of being
  > central, and this dataset separates them cleanly.

So the two notions we were asked to compare — exact **weighted degree** and
structural **PageRank** — broadly *agree* on who the kingpins are (good: the answer
is robust), while the extra **degree** measure usefully reveals a *second* kind of
central player (the high-variety accounts) that the volume measures hide.

---

## 3. Visualising a meaningful part of the network

We obviously cannot draw 5,000 nodes and ~4,000,000 edges — it would be an
unreadable black blob. The skill is choosing a **small, legible slice** that still
tells a true story. We take the **top-40 players by total games** and draw only their
**heaviest head-to-head rivalries**: we keep only player-pairs whose *combined* game
count is in the **top 20%** among those 40 players (an "edge-weight threshold" — a
cutoff that throws away the thin, faint edges so the strong relationships stand out).
Any player left with no surviving edge is dropped, so the picture shows the connected
heart of the elite (it ends up as 30 nodes and 65 edges).

**Visual encoding (how data maps to ink):** node size and colour are proportional to
each player's **total games** over the full graph (bigger and brighter = more games);
edge width is proportional to the **number of games in that specific rivalry**
(thicker = more head-to-head games). The **layout is force-directed ("spring")**: a
physics simulation where edges act like springs pulling connected nodes together and
all nodes repel each other, so tightly-linked players settle near each other. The
seed is fixed for reproducibility.

![Top players and their heaviest rivalries](../../figures/partA6_chess_top_players.png)

**Reading this picture.** The biggest, brightest nodes — `mscp`, `inemuri`,
`IFDThor`, `GriffyJr`, `GriffySr` — are the game-count leaders, sitting at the hubs of
the busiest rivalries. The thickest single edge typically links two superstars who
have faced each other an enormous number of times (for example the
`GriffyJr`–`GriffySr` pairing, very plausibly two closely-linked accounts that play
each other constantly). Several large nodes connect out to smaller satellites: top
players whose heaviest opponent is *not* itself a top-40 player. The thresholding is
what makes the structure legible — without it, the 40 elite players would be joined by
hundreds of faint edges into an unreadable mesh.

### Second plot — the games (weighted-degree) distribution on log-log axes

Real social/interaction networks are **heavy-tailed**: a few nodes have enormous
degree, most have tiny degree. On log-log axes (both axes on a logarithmic scale,
where each step multiplies rather than adds) that shows up as a near-straight, slowly
decaying line. We show it two ways: a **log-binned histogram** (counts per
logarithmic bucket) and a **complementary CDF** (CCDF = the fraction of players who
have *at least* x games, plotted against x — a robust way to view a tail).

![Games distribution (log-log)](../../figures/partA6_chess_degree_distribution.png)

**Reading this picture.** Both panels say the same thing two ways. The distribution
is **extremely heavy-tailed**: the *median* player has only about **13** games, the
*mean* is ~**827**, yet the busiest single player has **over 1.6 million**, and the
busiest **1% of players account for roughly 43% of all games ever played**. The
near-straight downward line on log-log axes is the classic look of a
**scale-free-like** network — one with no "typical" scale, where a tiny elite of
super-active hubs coexists with a vast crowd of casual players. This is exactly why a
*top-K* induced subgraph is the right tool: the network's structure is dominated by
that small hub elite, so capturing the busiest few thousand players captures the part
that drives the centrality results.

---

## 4. Limitations, assumptions, and sampling/filtering justification

- **Structural measures use the top-5000 induced subgraph, not the full graph.** We
  justified this and *confirmed* it empirically (the PageRank top-10 shares 9 of 10
  names with the exact games top-10). The blind spot: a player whose importance came
  entirely from connecting *low-activity* players would be invisible to it. The
  primary game-count ranking has no such limitation — it is exact and global.
- **We collapsed direction.** Any genuinely directional effect in the raw data is
  intentionally discarded as not meaningful for chess encounters (A vs B = B vs A).
- **No game outcomes or ratings exist in the data.** So "central" here means "central
  by activity/structure", not "strongest player". The handful of accounts that
  dominate the counts are very likely **bots or engine/pairing accounts** rather than
  human grandmasters.
- **We did not compute betweenness/closeness on the full subgraph** for the
  performance reasons above; doing so would require first reducing to a few hundred
  heavy-edge nodes.
- **Sampling/filtering method, stated plainly.** Two distinct reductions are used:
  (1) the *primary* result is **not sampled** at all — it is an exact full-graph
  aggregation; (2) the *structural* result is computed on a **deterministic
  top-K=5000 filter by activity** (an induced subgraph), justified because central
  players are active players; (3) the *drawing* further filters to the top-40 players
  and an 80th-percentile edge-weight threshold purely for legibility.

---

## 5. How I solved this task

**What I did.** I found the top-10 most central FICS chess players under three
centrality notions and drew a readable slice of the busy core.

**The methods and why.**

1. **Streaming aggregation (polars).** The 429-million-edge / 15.4 GB graph cannot be
   loaded whole, so it was pre-summarised by reading the CSV once in memory-bounded
   chunks (`scan_csv(...).collect(engine="streaming")`) as a SLURM batch job.
   Grouping the doubled edge list by the source column gives **exact** per-player game
   counts (weighted degree) over the *entire* graph; a second streaming pass extracts
   the **induced subgraph on the top-5000 busiest players** for structural work. I
   chose streaming because it is the only way to get *exact* full-graph statistics on
   a machine that cannot hold the graph.
2. **Centrality (igraph + the exact counts).** The **primary** ranking is weighted
   degree = total games, exact on the full graph. The **structural** rankings —
   PageRank, degree (distinct opponents), and strength — run on the undirected
   top-5000 subgraph in igraph (C-fast, sub-second). I treated the graph as undirected
   because a chess game has no direction, and I deliberately avoided
   betweenness/closeness because their O(V·E) cost would hang on millions of edges.
3. **Visualisation (networkx + matplotlib).** I drew the top-40 players keeping only
   their heaviest 20% of rivalries (edge-weight threshold), with node size/colour =
   total games and edge width = rivalry games, on a reproducible force-directed
   layout. A second log-log plot shows the heavy-tailed games distribution of the full
   graph.

**What the main result means.** The kingpins of FICS are `mscp` (~1.62M games) and
`inemuri` (~1.41M games), followed by `IFDThor`, `GriffyJr`, and `GriffySr`. The exact
volume ranking and the structural PageRank ranking **agree on 9 of their top 10**
(rank correlation ≈ +0.90), which both validates the answer and proves that analysing
only the busy core was sound. Plain degree reveals a *different* elite — high-variety
accounts that face thousands of distinct opponents a few times each — showing that
"central" has more than one meaning here.

**Data source:** <http://dynamics.cs.washington.edu/nobackup/chess/fcis.tar.gz>
**Libraries:** polars, python-igraph, networkx, numpy, pandas, scipy, matplotlib.
