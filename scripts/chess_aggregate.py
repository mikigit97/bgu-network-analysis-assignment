"""
Memory-safe aggregation of the 15 GB FICS chess interactions CSV on an 8 GB job.

Strategy (avoids building a 100M-key hash table that would OOM):
  1. EXACT full-graph activity per player: group_by a SINGLE column (src_id) ->
     only ~519k groups, tiny memory. 'games' = weighted degree (total games).
     (Every game appears as both (a,b) and (b,a), so grouping by src_id counts
      each of a player's games exactly once.)
  2. INDUCED SUBGRAPH on the top-K most active players: stream the file again,
     keep only rows whose BOTH endpoints are in the top-K set, then aggregate
     (src,dst)->count. The output is small, so this fits comfortably in memory.

Inputs : data/chess/FCIS/fcis_chess.interactions.csv  (datetime, src_id, dst_id)
Outputs: data/chess/player_stats.parquet              (player, games)  [FULL graph]
         data/chess/edges_top.parquet                 (src_id,dst_id,w) [top-K subgraph]
"""
import time
import polars as pl

BASE = "/home/mickaelz/Network analysis"
SRC = f"{BASE}/data/chess/FCIS/fcis_chess.interactions.csv"
STATS_OUT = f"{BASE}/data/chess/player_stats.parquet"
EDGES_OUT = f"{BASE}/data/chess/edges_top.parquet"
TOP_K = 5000  # build the structural subgraph on the 5000 most active players

t0 = time.time()
schema = {"datetime": pl.Utf8, "src_id": pl.Utf8, "dst_id": pl.Utf8}

# ---- Step 1: exact games per player (single-column group_by) ----
stats = (pl.scan_csv(SRC, schema_overrides=schema)
           .group_by("src_id").agg(pl.len().alias("games"))
           .rename({"src_id": "player"})
           .sort("games", descending=True)
           .collect(engine="streaming"))
stats.write_parquet(STATS_OUT)
print(f"[{time.time()-t0:.0f}s] players: {stats.height:,}")
print("Top 15 by total games:")
print(stats.head(15))

# ---- Step 2: induced subgraph on the top-K most active players ----
top_players = stats.head(TOP_K)["player"].to_list()
top_set = pl.Series(top_players)
edges = (pl.scan_csv(SRC, schema_overrides=schema)
           .filter(pl.col("src_id").is_in(top_set) & pl.col("dst_id").is_in(top_set))
           .group_by(["src_id", "dst_id"]).agg(pl.len().alias("w"))
           .collect(engine="streaming"))
edges.write_parquet(EDGES_OUT)
print(f"[{time.time()-t0:.0f}s] top-{TOP_K} induced subgraph edges: {edges.height:,}")
print("CHESS_AGGREGATE_DONE")
