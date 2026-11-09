"""
VECTOR DB 2 of 4 - EXACT SEARCH, FILTERS, PERSISTENCE (7 min)

Read flat_index.py first (60 lines). Then this script:
  1. inserts vectors with metadata
  2. searches top-k
  3. filters by metadata
  4. saves to disk and reloads
  5. times a scan as N grows, to show why brute force stops scaling

Run:  python 5_vector_db/15_flat_search.py
Next: python 5_vector_db/16_ivf_search.py
"""
import os
import tempfile
import time
import numpy as np
from flat_index import FlatIndex

rng = np.random.default_rng(0)
db = FlatIndex(dim=8)

# Three made-up "topics", each a direction in 8-d space, plus noise.
topics = {"fruit": rng.normal(size=8), "car": rng.normal(size=8), "code": rng.normal(size=8)}
for tag, center in topics.items():
    for i in range(5):
        db.add(f"{tag}-{i}", center + 0.2 * rng.normal(size=8), {"tag": tag, "n": i})
print(f"stored {len(db)} vectors\n")

q = topics["car"] + 0.2 * rng.normal(size=8)
print("Top 3 for a car-ish query:")
for id_, score, meta in db.search(q, k=3):
    print(f"  {id_:<8} cos={score:.3f}  {meta}")

print("\nSame query, but only where tag == 'code':")
for id_, score, meta in db.search(q, k=3, where={"tag": "code"}):
    print(f"  {id_:<8} cos={score:.3f}  {meta}")

path = os.path.join(tempfile.mkdtemp(), "demo")
db.save(path)
db2 = FlatIndex.load(path)
assert [r[0] for r in db.search(q, 3)] == [r[0] for r in db2.search(q, 3)]
print("\nSaved and reloaded: identical results.")

print("\nBrute-force latency vs size (dim=128, one query):")
print(f"{'N':>10} {'ms/query':>10}")
for n in (1_000, 10_000, 100_000, 500_000):
    big = FlatIndex(128)
    big.add_batch(range(n), rng.normal(size=(n, 128)))
    t = time.perf_counter()
    for _ in range(5):
        big.search(rng.normal(size=128), k=10)
    print(f"{n:>10,} {(time.perf_counter() - t) / 5 * 1000:>10.2f}")
print("\nLatency grows linearly with N. Next: skip most of the vectors.")
