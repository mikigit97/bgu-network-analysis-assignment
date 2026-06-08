# Part B — Directed Link Prediction (25 pts) + Bonus: Future Link Prediction (+5 pts)

This part builds a classifier that predicts **directed** links in a real social
network: given two communities `u` and `v`, will community `u` post a hyperlink
that points to community `v`? Everything below is implemented and executed in the
notebook `notebooks/partB_linkpred.ipynb`, which runs end-to-end without errors.
All numbers in this report are the exact values produced by that executed
notebook.

---

## What "link prediction" means, from first principles

A **graph** (also called a network) is a collection of dots called **nodes**
joined by lines called **edges**. When the lines have a direction — an arrow that
goes *from* one node *to* another — we call them **directed edges**, and the
graph is a **directed graph** (NetworkX calls this a `DiGraph`). In a directed
graph the arrow `u -> v` is a *different* thing from the arrow `v -> u`, exactly
like "Alice phoned Bob" is different from "Bob phoned Alice".

**Link prediction** asks: looking at the arrows we can already see, can we guess
which arrows are missing, or which arrows will appear next? We turn that question
into a yes/no **classification** problem. A *classifier* is just a function that
takes a row of numbers describing a candidate pair `(u, v)` and outputs a
probability between 0 and 1 that the arrow `u -> v` should exist. To train such a
function we need examples labelled with the right answer — that is the whole
point of the positive/negative examples and the train/test split below.

A natural misconception worth clearing up immediately: because the graph is
directed, a good predictor cannot just measure "are `u` and `v` close to each
other". It has to respect direction. Two communities can be tightly related yet
only link one way (a small community constantly links to a giant one, but the
giant one never links back). Every feature we build is therefore
*direction-aware*.

---

## B1.1 — Network description (2 pts)

### Data source

**SNAP Reddit Hyperlink Network (body version).**
- Landing page: https://snap.stanford.edu/data/soc-RedditHyperlinks.html
- File downloaded: https://snap.stanford.edu/data/soc-redditHyperlinks-body.tsv

(Note: the old `.tsv.gz` URL now returns 404; SNAP currently serves the file
uncompressed as plain `.tsv`, which is what the download script fetches.)

This dataset records, over about two and a half years, every time one
**subreddit** (a topic-based community on the website Reddit) posts a
**hyperlink** that points into another subreddit. Each row of the raw file is one
such hyperlink event, and it carries a timestamp.

### What the nodes and edges mean

- A **node** is a subreddit, for example `askreddit` or `nfl`.
- A **directed edge** `u -> v` means community `u` posted a hyperlink that points
  to community `v`.

Because the same pair `u -> v` can be posted many times, I **aggregated** all the
repeated events for a pair into a single directed edge and stored three pieces of
information on it:
- `weight` = how many times `u` linked to `v` (a strength/frequency measure),
- `first` = the timestamp of the *first* time `u -> v` ever happened,
- `last`  = the timestamp of the *last* time it happened.

The `first`/`last` timestamps are what make the bonus possible: I can literally
train on the past and test on the future.

### Sampling (and the justification)

The full graph has **35,776 subreddits** and the raw file has **286,561
hyperlink events** spanning **2013-12-31 to 2017-04-30**. Running node2vec (the
embedding step in B1.5) on the whole graph on a *shared 4-CPU cluster node* would
be slow and memory-heavy. I therefore kept the **subgraph induced by the top-2000
most active subreddits**, where "active" means the total number of times a
subreddit appears as either a source or a target.

This is a principled sample, not an arbitrary cut:
- The most active communities are precisely the ones with enough links to learn
  from; rarely-seen subreddits contribute almost no signal but a lot of noise.
- Restricting to a dense core keeps the graph well-connected instead of a haze of
  one-off links, which makes negative sampling and path features meaningful.
- It still retains **155,088 of the 286,561 events (54.1%)**, so it is a large,
  representative slice rather than a toy.

### Size and basic statistics (from the executed notebook)

| Property | Value |
|---|---|
| Nodes (subreddits) | **1,995** |
| Directed edges (aggregated pairs) | **48,998** |
| Directed? | Yes |
| Edge time range | 2013-12-31 → 2017-04-30 |
| Average in-degree | 24.56 |
| Average out-degree | 24.56 |
| Max in-degree (most linked-TO) | 850 |
| Max out-degree (most linking-OUT) | 773 |
| **Reciprocity** | **0.2585** |
| # strongly connected components (SCC) | **107** |
| # weakly connected components (WCC) | **2** |

Reading these numbers in plain English:

- **In-degree of a node `v`** = how many different communities link *to* `v`.
  **Out-degree of `u`** = how many different communities `u` links *to*. Tiny
  example: if only `a -> c` and `b -> c` exist, then `c` has in-degree 2 and
  out-degree 0. The averages are equal (24.56) for a simple reason: every edge
  adds exactly one to some node's out-degree and one to some node's in-degree, so
  the totals — and hence the averages over the same node set — must match.
- **Reciprocity = 0.2585** means about a quarter of arrows are "answered back":
  if `u -> v` exists, roughly 26% of the time `v -> u` also exists. (Misconception
  to avoid: reciprocity is *not* a measure of overall connectedness; it is
  specifically the share of *mutual* arrows.)
- A **strongly connected component (SCC)** is a group of nodes where you can get
  from any one to any other *while obeying the arrow directions*. A **weakly
  connected component (WCC)** is the same but you may also walk *against* arrows.
  Having **many SCCs (107) but essentially one WCC (2, one of which is tiny)** is
  the classic shape of a directed social graph: ignore direction and almost
  everything is one big blob, but respect direction and the graph splinters into
  many one-way pockets.

The figure `../../figures/partB_degree_distributions.png` plots the in-degree and
out-degree distributions on log-log axes (both axes spaced by powers of ten). The
roughly straight, downward scatter is the signature of a **heavy-tailed** network:
most communities have a small degree, but a few "hub" communities (like
`askreddit`) have an enormous one.

**How I solved this task (B1.1).** I downloaded the SNAP Reddit hyperlink body
file, aggregated repeated `u -> v` posts into single directed edges (keeping a
weight and first/last timestamps), and — because the full graph is large for
embedding on a shared CPU node — kept the subgraph induced by the 2,000 most
active subreddits, which still retains over half of all link events while staying
dense and almost entirely connected. I then reported the standard directed
descriptors (in/out-degree averages and maxima, reciprocity, SCC/WCC counts) and
plotted the degree distributions. The finding is that this is a typical
heavy-tailed directed social graph: a few hubs dominate, about a quarter of links
are reciprocated, and there is one giant weakly connected blob but many small
strongly connected pockets.

---

## B1.2 — Positive and negative examples (3 pts)

A classifier learns from **labelled examples** — inputs tagged with the correct
answer. For link prediction the two labels are:

- **Positive (label 1)** = a *real* directed edge `u -> v` that exists in the
  graph (a pair that genuinely is linked).
- **Negative (label 0)** = a *non-edge* `u -> v`, an ordered pair with **no**
  arrow from `u` to `v` (a pair that is not linked).

### Why negatives must be sampled, and how

With ~2,000 nodes there are about `2000 x 1999 ≈ 4,000,000` possible ordered
pairs, but only ~49,000 are real edges. So **non-edges vastly outnumber edges**.
If I used all of them, the data would be ~99% negative and a lazy model could
score 99% "accuracy" by always answering "no link" — completely useless. I
therefore **balance** the classes: I sample exactly as many negatives as there
are positives, giving a 50/50 mix. The executed notebook confirms:

```
positive examples (real directed edges): 48998
negative examples (sampled non-edges)  : 48998
class balance: 50% / 50%
```

Two rules make the negatives a *fair* challenge rather than a giveaway:

1. **Only active nodes.** Both `u` and `v` must be inside the top-2000 sample, so
   I never invent a pair involving a community I know nothing about.
2. **Same weakly connected component.** I require `u` and `v` to be in the same
   WCC (reachable if you ignore direction). This removes
   *trivially-disconnected* pairs — two communities on totally separate islands,
   which any method could reject without learning anything. Forcing the pair into
   the same blob makes the negative "look plausible", so the classifier must
   actually use structure to tell a real link from a fake one.

**How I solved this task (B1.2).** I labelled all real directed edges as
positives and randomly drew an equal number of directed non-edges as negatives,
keeping only pairs that are not self-loops, not real edges, and lie in the same
weakly connected component. The "same component" filter is the key design choice:
it strips out trivially unlinkable pairs so the model is forced to learn from
genuine structural signal.

---

## B1.3 — Train/test split and leakage avoidance (3 pts)

To judge a model honestly I hide some examples during training and reveal them
only at test time. I use a **random 80/20 edge split**: 80% of the real edges
(and 80% of the negatives) form the **training set**, the remaining 20% form the
**test set**. From the executed notebook:

```
train positives: 39199 | test positives: 9799
train negatives: 39199 | test negatives: 9799
G_train edges (features come ONLY from here): 39199
```

### What "leakage" is and how I prevent it

**Leakage** means the model accidentally sees information about the test answers
while training — like a student who saw the exam in advance. The subtle danger in
link prediction lives in *feature computation*. I describe each candidate pair
`(u, v)` with numbers computed from the graph (degrees, shared neighbours, paths,
and so on). If I computed those numbers from the **full** graph, then for a test
edge `u -> v` the graph would still *contain* that edge, and a feature like "do
`u` and `v` share neighbours?" would secretly leak the answer.

**My fix:** I build a separate **train graph** `G_train` that contains *only the
training positive edges*. Every feature — for both train and test pairs — is
computed from `G_train` alone. A test edge `u -> v` is therefore *absent* from
the graph I measure, exactly as a truly unknown link would be. This is the
standard leakage-free setup for link prediction.

(The bonus uses a different, *temporal* split — train on edges before a cutoff
date, test on edges after it — which is even more realistic.)

**How I solved this task (B1.3).** I randomly assigned 20% of the real edges to a
held-out test set and 80% to training, splitting the negatives the same way.
Crucially I rebuilt a *train-only* directed graph and computed every feature from
it, so a test edge is invisible while it is being described. This guarantees the
evaluation measures real prediction rather than memorised answers.

---

## B1.4 — Baseline classifier from directed topology features (7 pts)

"**Topology**" means "the shape of the graph" — who points to whom. A "**feature**"
is one number measured for a candidate pair `(u, v)`. Below, let `succ(x)` be the
set of nodes `x` **points to** (its out-neighbours) and `pred(x)` be the set of
nodes that **point to** `x` (its in-neighbours). Each feature is defined in plain
words with a tiny example.

1. **`src_out`, `src_in`, `tgt_out`, `tgt_in`** — out- and in-degrees of source
   `u` and target `v`. Raw "how busy is each endpoint" signals.
2. **`common_succ`** = `|succ(u) ∩ succ(v)|`, how many nodes *both* point to. If
   `u` and `v` link to the same communities they behave alike.
3. **`common_pred`** = `|pred(u) ∩ pred(v)|`, how many nodes point to *both*.
4. **`dir_jaccard_succ`** = **Jaccard similarity** of the successor sets,
   `|succ(u) ∩ succ(v)| / |succ(u) ∪ succ(v)|`. Jaccard is "share in common out
   of everything either has", from 0 (nothing shared) to 1 (identical). Example:
   `succ(u)={a,b}`, `succ(v)={b,c}` → 1 shared / 3 total = 0.33.
5. **`dir_jaccard_pred`** — the same Jaccard idea on the predecessor sets.
6. **`dir_adamic_adar`** — **directed Adamic-Adar**. I look at "stepping-stone"
   nodes `w` on a path `u -> w -> v` (so `w ∈ succ(u)` and `w ∈ pred(v)`) and add
   up `1 / log(degree(w))` over them. The idea: a shared connector that is itself
   *rare* (low degree) is strong evidence, while a giant hub everyone touches is
   weak evidence — so rare connectors are upweighted. Analogy: two people who
   both know your reclusive aunt are probably connected; two people who both
   follow a megastar are not.
7. **`pref_attach`** — **preferential attachment**, `out_deg(u) * in_deg(v)`. The
   "rich get richer" intuition: a node that already links out a lot tends to form
   *new* outgoing links, and a node already linked-to a lot is a likely target.
8. **`reciprocity_ind`** — a 0/1 flag: does the reverse edge `v -> u` exist in the
   train graph? Mutual links are common, so this is a strong hint.
9. **`path_len2`** — a 0/1 flag: is there a directed two-step path `u -> w -> v`
   (equivalently `succ(u) ∩ pred(v)` non-empty)?
10. **`path_len3_flag`** — a 0/1 flag: is there a directed path `u → ... → v` of
    length at most 3? (Capped to stay cheap; longer paths add little.)

I fed these 13 features to two standard classifiers:
- **Logistic Regression (LR)** — fits a weighted sum of the features and squashes
  it into a probability; simple, fast, interpretable.
- **Random Forest (RF)** — an ensemble of many decision trees that vote; captures
  non-linear interactions automatically.

### Baseline results (held-out test set)

| Model | AUC | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Baseline LR (topology) | 0.9414 | 0.8726 | 0.8881 | 0.8527 | 0.8701 |
| Baseline RF (topology) | 0.9417 | 0.8712 | 0.8738 | 0.8677 | 0.8708 |

Both models already predict links well, with **AUC around 0.94**. The
Random-Forest feature-importance plot
(`../../figures/partB_feature_importance.png`) ranks the features; the top ones
from the executed notebook are:

```
dir_adamic_adar     0.2315
pref_attach         0.1395
path_len2           0.1078
dir_jaccard_pred    0.0986
dir_jaccard_succ    0.0704
```

In words: most of the predictive power comes from "do `u` and `v` share a *rare*
stepping-stone connector (Adamic-Adar), are they both popular (preferential
attachment), and can you already reach `v` from `u` in two hops (length-2 path)".

**How I solved this task (B1.4).** I engineered 13 direction-aware features for
each pair — in/out-degrees of both endpoints, counts and Jaccard overlaps of
common successors and predecessors, a directed Adamic-Adar score that rewards
rare shared connectors, preferential attachment, a reverse-edge flag, and short
directed-path flags — all computed from the train-only graph. I trained both
Logistic Regression and Random Forest. The baseline already reaches AUC ~0.94,
and the importance plot shows directed Adamic-Adar, preferential attachment, and
the length-2 path flag carry most of the signal.

---

## B1.5 — Improved classifier with node2vec embeddings (5 pts)

### What is a node embedding? (plain words + analogy)

An **embedding** turns each node into a short list of numbers — a point in, say,
64-dimensional space — so that nodes playing *similar roles* in the graph land
*near each other*. Picture placing every subreddit on a giant map where
"communities that get linked in the same contexts" sit close together. A
classifier can then ask geometric questions ("are these two points arranged like
a real edge?") instead of relying only on hand-made counts.

### What node2vec does, step by step

**node2vec** learns those points with a "you are the company you keep" trick
borrowed from language modelling:
1. **Random walks.** Start at a node and take a stroll, repeatedly hopping to a
   neighbour at random, recording the sequence visited — like a sentence whose
   "words" are subreddits. We do this many times from every node. On a *directed*
   graph the walker follows arrow directions, so the walks capture direction.
2. **word2vec on the walks.** We feed these node-sequences to **word2vec**, which
   learns a vector for each "word" so that words appearing in similar surroundings
   get similar vectors. Result: nodes appearing in similar walks get similar
   embeddings.

Tiny picture: if walks often read `... gaming -> leagueoflegends -> esports ...`
and also `... gaming -> dota2 -> esports ...`, then `leagueoflegends` and `dota2`
get placed near each other because they sit in the same neighbourhood.

I used modest settings so it runs fast on the shared node:
`dimensions=64, walk_length=20, num_walks=10, workers=2`. In the executed
notebook node2vec finished in about **25 seconds** and produced 1,995 vectors of
length 64.

### Turning two node vectors into one *pair* feature

A classifier needs **one** feature row per pair, but I have **two** vectors
(`emb(u)` and `emb(v)`). Two standard combiners:
- **Hadamard product** — multiply the vectors element-by-element. A coordinate
  stays large only if it is large in *both* nodes, so this highlights dimensions
  where the two nodes *agree*. It is the most common choice for link prediction.
- **Concatenation** — stick `emb(u)` then `emb(v)` into one long vector
  `[src | tgt]`, keeping source and target separate so the model can learn
  *direction*.

### Improved results (held-out test set)

| Model | AUC | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Baseline LR (topology) | 0.9414 | 0.8726 | 0.8881 | 0.8527 | 0.8701 |
| Baseline RF (topology) | 0.9417 | 0.8712 | 0.8738 | 0.8677 | 0.8708 |
| Emb-only Hadamard (LR) | 0.8023 | 0.7234 | 0.7497 | 0.6707 | 0.7080 |
| Emb-only Concat (LR) | 0.7444 | 0.6804 | 0.6806 | 0.6800 | 0.6803 |
| Emb-only Hadamard (RF) | 0.8778 | 0.7958 | 0.8219 | 0.7553 | 0.7872 |
| **Topo+Emb Hadamard (RF)** | **0.9493** | **0.8823** | **0.8932** | **0.8685** | **0.8806** |
| Topo+Emb Hadamard (LR) | 0.9439 | 0.8717 | 0.8994 | 0.8369 | 0.8671 |

Reading the table: embeddings alone (Hadamard + Random Forest) reach AUC 0.8778
— good, and notable because they use *no* hand-made graph counts, only learned
geometry — but they sit below the topology baseline because random walks blur some
of the exact local overlap that the topology counts measure precisely. The
**combination of topology + embeddings (Random Forest) is the best model, AUC
0.9493**, edging out the topology-only baseline.

**How I solved this task (B1.5).** I ran node2vec on the directed train graph to
learn a 64-number vector for each subreddit, combined each pair of vectors into a
single pair-feature using both the Hadamard product and concatenation, and
trained classifiers on embeddings alone and on embeddings combined with the
topology features. Embeddings alone are good but a notch below the topology
baseline; the combination is the best overall, because topology counts and
embedding geometry capture complementary information.

---

## B1.6 — Evaluation metrics and ROC curves (3 pts)

I score every model with five numbers. In plain English:

- **Accuracy** — fraction of test pairs labelled correctly. Meaningful here
  because our data is balanced 50/50.
- **Precision** — of the pairs the model *claims* are links, what fraction really
  are? High precision = few false alarms.
- **Recall** — of the real links, what fraction did the model *catch*? High
  recall = few misses.
- **F1** — the balanced (harmonic) mean of precision and recall; high only when
  *both* are high.
- **AUC (Area Under the ROC Curve)** — the headline metric. Pick one real link
  and one non-link at random; AUC is the probability the model gives the real link
  the *higher* score. **AUC = 1.0** is perfect, **AUC = 0.5** is coin-flipping.
  It is threshold-free, so it measures ranking quality regardless of where we put
  the 0.5 cutoff.

An **ROC curve** plots, as the decision threshold sweeps from strict to lenient,
the **true-positive rate** (recall) on the y-axis against the **false-positive
rate** (share of non-links wrongly flagged) on the x-axis. A curve hugging the
top-left corner is excellent; the diagonal is random guessing; the AUC is
literally the area under the curve. The figure
`../../figures/partB_roc_curves.png` shows the curves for the four headline
models — all bow well above the diagonal, with the topology+embeddings Random
Forest highest.

**How I solved this task (B1.6).** I evaluated every model on the held-out 20%
test set with AUC, accuracy, precision, recall and F1, and drew ROC curves for the
headline models. AUC is the metric I trust most because it judges how well each
model *ranks* real links above non-links regardless of cutoff; the curves confirm
the topology+embeddings Random Forest is the strongest ranker.

---

## B1.7 — Comparison: baseline vs improved (2 pts)

From the executed notebook:

```
Best baseline AUC (topology only): 0.9417
Best improved AUC (topology+emb) : 0.9493
Absolute AUC gain from embeddings: 0.0076
```

- The **baseline** (directed topology only) is already strong at AUC **0.9417**,
  showing most of the signal lives in simple, interpretable structure — shared
  rare connectors, joint popularity, and short directed paths.
- **Embeddings alone** reach AUC **0.8778**, respectable given they use only
  learned geometry, but slightly below the baseline.
- The **improved combined model** (topology + node2vec, Random Forest) is best at
  AUC **0.9493**. The gain (+0.0076 AUC) is small in absolute terms but consistent
  and exactly the expected outcome: embeddings add *complementary* "soft
  similarity" information on top of the exact counts, so together they beat either
  alone.

**Takeaway.** For this network, hand-made directed topology features are a very
strong, cheap baseline; node2vec embeddings give a modest, reliable extra lift
when *combined* with them, rather than replacing them.

**How I solved this task (B1.7).** I assembled all models into one metrics table
and computed the AUC gap between the best topology-only baseline and the best
topology+embedding model. The comparison shows embeddings provide a small but
consistent improvement on top of an already-strong baseline.

---

## Bonus — Future Link Prediction with a temporal split (+5 pts)

The random split above mixes past and future edges freely. A tougher, more
realistic test is **temporal**: stand at a moment in time, learn only from what
happened *before* it, then predict links that appear *after* it.

### The setup, in plain words

1. **Pick a cutoff `T`** = the 80th percentile of edges' *first* appearance, so
   80% of links are "past" and 20% are "future". The executed notebook gives
   `T = 2016-07-01`.
2. **Past graph** = a directed graph from only the edges whose first appearance is
   before `T`. All bonus features come from this past-only graph
   (1,948 nodes, 39,198 edges).
3. **Positive future links** = pairs `u -> v` whose *first* appearance is at or
   after `T`, where both `u` and `v` already existed before `T` (we only judge
   links between *known* communities) and the edge did *not* exist before `T` (it
   is genuinely new). The notebook found **8,915** such links.
4. **Negative future links** = pairs of existing nodes that *never* become an edge
   (same WCC filter as before), balanced 1:1 with the positives (8,915).
5. **Two predictors compared:**
   - **(a) Random** — assign each test pair a random score; AUC ≈ 0.5 by design,
     the "do-nothing" reference.
   - **(b) Informed** — the classifier (topology, and topology+node2vec) trained
     on the *past* graph, scoring the future pairs.

### Bonus results (from the executed notebook)

```
=== BONUS: future link prediction AUCs ===
  (a) Random predictor          : 0.4962
  (b) Informed topology (RF)    : 0.9038
  (b) Informed topo+emb (RF)    : 0.9142
```

| Predictor | AUC on future links |
|---|---|
| (a) Random | **0.4962** |
| (b) Informed — topology (RF) | **0.9038** |
| (b) Informed — topology + embeddings (RF) | **0.9142** |

The ROC curves for these three are in
`../../figures/partB_bonus_future_roc.png`.

### Discussion — does the classifier generalize to the future?

**Yes, clearly.** The random predictor scores AUC ≈ 0.50 (a coin flip, exactly as
expected), while the informed classifier scores **AUC ≈ 0.91** on genuinely new
future links it has never seen. So the structural patterns learned from the past
— shared rare connectors, joint popularity, short directed paths, and embedding
geometry — really do carry forward and let us anticipate which communities will
start linking next. Adding embeddings on top of topology again gives a small lift
(0.9038 → 0.9142), the same complementary effect seen in the random split.

**Why future prediction is harder than the random split (and why the AUC is a bit
lower).** On the random split we reached 0.9493; here we land around 0.91. The
reason is built into the setup: future positives are *brand-new* edges, so at
cutoff time their two endpoints were not yet joined through that link, and the
past graph offers a thinner signal than the rich same-period neighbourhood
available in the random split. The network also *drifts* over time — new fads, new
communities, changing interests — so a model fit on the past is always chasing a
slightly moving target. The gap between ~0.95 (random split) and ~0.91 (temporal)
is exactly this "predicting the future is harder than filling in the present"
effect, and it is reassuringly small, which means the learned structure is fairly
stable over time.

**How I solved this task (Bonus).** I chose a cutoff at the 80th percentile of
first-appearance times, built a past-only directed graph, defined positives as new
links first appearing after the cutoff between communities that already existed
before it, and balanced them with non-edge negatives. I computed all features
(directed topology + node2vec) from the past graph only, trained the classifier to
predict past edges, and tested on the future links. Against a random-score
baseline, the informed model's AUC (~0.91) towers over random (~0.50),
demonstrating real generalization to the future while sitting modestly below the
random-split AUC because forecasting brand-new links is inherently harder.

---

## Summary table

| Setting | Model | AUC |
|---|---|---|
| Random split | Baseline (directed topology, RF) | 0.9417 |
| Random split | Embeddings only (node2vec Hadamard, RF) | 0.8778 |
| Random split | **Topology + embeddings (RF, best)** | **0.9493** |
| Temporal (bonus) | Random predictor | 0.4962 |
| Temporal (bonus) | Informed (topology, RF) | 0.9038 |
| Temporal (bonus) | Informed (topology + embeddings, RF) | 0.9142 |

---

## Limitations, assumptions, and sampling justification

- **Sampling.** I kept only the top-2,000 most active subreddits (54.1% of all
  link events). This favours the dense, well-studied core of the network; results
  may differ for the long tail of rarely-linked communities. The choice was made
  to keep node2vec fast on a shared 4-CPU node and to keep negative sampling and
  path features meaningful, and it is stated explicitly so the reader can judge
  it.
- **Balanced negatives.** Real link prediction is extremely imbalanced (non-edges
  hugely outnumber edges). I trained and evaluated on a *balanced* 50/50 set,
  which is standard and makes AUC/accuracy interpretable, but absolute precision
  in a real deployment (scanning all 4M pairs) would be lower simply because there
  are far more chances for false positives.
- **Same-WCC negative filter.** Requiring negatives to share a weakly connected
  component makes them harder (more realistic) but also means the reported numbers
  reflect the "hard" regime; trivially-disconnected pairs would be even easier to
  reject.
- **Modest node2vec settings.** dimensions=64, walk_length=20, num_walks=10. Larger
  settings might lift the embedding-only results, but the combined model is already
  the best and the gain would likely be marginal.
- **Temporal assumption (bonus).** I only judge future links between communities
  that already existed before the cutoff; predicting links involving brand-new
  communities (cold start) is a separate, harder problem not addressed here.

## Reproducibility, data, and libraries

- **Reproducibility.** All random seeds fixed to 42. The notebook runs
  end-to-end without errors via
  `jupyter nbconvert --to notebook --execute --inplace notebooks/partB_linkpred.ipynb`.
- **Data source.** SNAP Reddit Hyperlink Network (body):
  https://snap.stanford.edu/data/soc-RedditHyperlinks.html
  (file: https://snap.stanford.edu/data/soc-redditHyperlinks-body.tsv)
- **Libraries.** `networkx` (graph construction + topology features), `node2vec`
  + `gensim` (embeddings), `scikit-learn` (Logistic Regression, Random Forest,
  metrics), `pandas`/`numpy` (data handling), `matplotlib` (figures, Agg backend).

## Figures (all titled, saved under `figures/`)

- `../../figures/partB_degree_distributions.png` — in/out-degree distributions (log-log).
- `../../figures/partB_feature_importance.png` — which directed topology features drive the baseline.
- `../../figures/partB_roc_curves.png` — ROC curves for the random-split models.
- `../../figures/partB_bonus_future_roc.png` — ROC curves for the temporal/bonus models (random vs informed).
