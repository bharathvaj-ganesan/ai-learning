"""
IVF ("inverted file") approximate index. Imported by 16_ivf_search.py.

Idea: cluster the vectors with k-means. Each cluster has a centroid and a bucket
of member vectors. At query time, find the few centroids closest to the query and
scan ONLY their buckets. Fewer vectors compared -> faster, but a true neighbor
sitting in a bucket you skipped is missed. `nprobe` (buckets scanned) is the knob.
This is the same idea as FAISS IndexIVFFlat.
"""
import numpy as np
from flat_index import normalize


def kmeans(X, k, iters=15, seed=0):
    """Spherical k-means on unit vectors: assign by dot product, re-average, renormalize."""
    rng = np.random.default_rng(seed)
    C = X[rng.choice(len(X), k, replace=False)].copy()
    for _ in range(iters):
        assign = np.argmax(X @ C.T, axis=1)
        for j in range(k):
            members = X[assign == j]
            if len(members):
                C[j] = members.mean(axis=0)
        C = normalize(C)
    return C


class IVFIndex:
    def __init__(self, dim, n_lists):
        self.dim, self.n_lists = dim, n_lists
        self.centroids = None
        self.vectors = np.zeros((0, dim), dtype=np.float32)
        self.buckets = [[] for _ in range(n_lists)]   # bucket j -> row numbers in self.vectors

    def train(self, sample):
        self.centroids = kmeans(normalize(sample), self.n_lists)

    def add_batch(self, vectors):
        vectors = normalize(vectors)
        start = len(self.vectors)
        self.vectors = np.vstack([self.vectors, vectors])
        for off, j in enumerate(np.argmax(vectors @ self.centroids.T, axis=1)):
            self.buckets[j].append(start + off)

    def search(self, query, k=10, nprobe=4):
        q = normalize(query)
        nearest = np.argsort(-(self.centroids @ q))[:nprobe]          # which buckets to open
        cand = np.concatenate([self.buckets[j] for j in nearest]).astype(int)
        if len(cand) == 0:
            return []
        scores = self.vectors[cand] @ q                               # scan only the candidates
        k = min(k, len(cand))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(int(cand[i]), float(scores[i])) for i in top]
