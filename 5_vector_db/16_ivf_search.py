"""
VECTOR DB 3 of 4 - APPROXIMATE SEARCH AND RECALL (10 min)

Build an IVF index over 100k clustered vectors and compare it to the exact flat scan:
  speed:   how much faster is it?
  recall@10: of the true 10 nearest neighbors, how many did IVF return?

Turn `nprobe` up and recall rises while speed falls. That trade-off is the central
fact of vector databases. Every ANN index (IVF, HNSW, PQ, ...) lives on this curve.

Run:  python 5_vector_db/16_ivf_search.py
Next: python 5_vector_db/17_text_search.py
"""
import time
import numpy as np
from flat_index import FlatIndex
from ivf_index import IVFIndex

N, DIM, N_LISTS, N_QUERIES, K = 100_000, 64, 256, 200, 10
rng = np.random.default_rng(0)

# Real embeddings are clumpy (topics). Pure random vectors have no clusters, so IVF would look bad.
centers = rng.normal(size=(300, DIM))
data = centers[rng.integers(0, 300, N)] + 0.5 * rng.normal(size=(N, DIM))
queries = centers[rng.integers(0, 300, N_QUERIES)] + 0.5 * rng.normal(size=(N_QUERIES, DIM))

flat = FlatIndex(DIM)
flat.add_batch(range(N), data)

t = time.perf_counter()
ivf = IVFIndex(DIM, N_LISTS)
ivf.train(data[rng.choice(N, 20_000, replace=False)])   # k-means on a sample is enough
ivf.add_batch(data)
print(f"built IVF: {N:,} vectors into {N_LISTS} buckets in {time.perf_counter() - t:.1f}s\n")

t = time.perf_counter()
truth = [{r[0] for r in flat.search(q, K)} for q in queries]
flat_ms = (time.perf_counter() - t) / N_QUERIES * 1000
print(f"exact flat scan: {flat_ms:.2f} ms/query, recall 100%\n")

print(f"{'nprobe':>7} {'ms/query':>9} {'speedup':>8} {'recall@10':>10} {'scanned':>9}")
for nprobe in (1, 2, 4, 8, 16, 32, 64):
    t = time.perf_counter()
    res = [ivf.search(q, K, nprobe) for q in queries]
    ms = (time.perf_counter() - t) / N_QUERIES * 1000
    recall = np.mean([len(truth[i] & {r[0] for r in res[i]}) / K for i in range(N_QUERIES)])
    scanned = np.mean([sum(len(ivf.buckets[j]) for j in np.argsort(-(ivf.centroids @ q))[:nprobe])
                       for q in queries[:50]]) / N
    print(f"{nprobe:>7} {ms:>9.2f} {flat_ms / ms:>7.1f}x {recall:>10.1%} {scanned:>9.1%}")

print("\nPick the smallest nprobe whose recall you can live with. 100% recall = no savings.")
