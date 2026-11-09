"""
Flat (brute-force) vector index. Imported by the step scripts; not run directly.

Stores every vector in one NumPy matrix and compares the query against ALL of them.
Exact answers, O(N * dim) per query. This is the baseline every approximate index
is measured against, and it is what many small production systems actually use.
"""
import json
import numpy as np


def normalize(x):
    x = np.asarray(x, dtype=np.float32)
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


class FlatIndex:
    def __init__(self, dim):
        self.dim = dim
        self.vectors = np.zeros((0, dim), dtype=np.float32)  # unit-length rows
        self.ids = []
        self.meta = []

    def __len__(self):
        return len(self.ids)

    def add_batch(self, ids, vectors, metas=None):
        vectors = normalize(np.asarray(vectors).reshape(-1, self.dim))
        self.vectors = np.vstack([self.vectors, vectors])
        self.ids += list(ids)
        self.meta += list(metas) if metas is not None else [{} for _ in ids]

    def add(self, id, vector, metadata=None):
        self.add_batch([id], [vector], [metadata or {}])

    def search(self, query, k=5, where=None):
        """Top-k by cosine. `where={"tag": "x"}` keeps only rows whose metadata matches."""
        scores = self.vectors @ normalize(query)           # one score per stored vector
        if where:                                          # pre-filter: masked rows can never win
            keep = np.array([all(m.get(f) == v for f, v in where.items()) for m in self.meta])
            scores = np.where(keep, scores, -np.inf)
        k = min(k, len(self))
        top = np.argpartition(-scores, k - 1)[:k]          # O(N) pick of the k best, unordered
        top = top[np.argsort(-scores[top])]                # sort just those k
        return [(self.ids[i], float(scores[i]), self.meta[i]) for i in top if scores[i] > -np.inf]

    def save(self, path):
        np.save(path + ".npy", self.vectors)
        with open(path + ".json", "w") as f:
            json.dump({"dim": self.dim, "ids": self.ids, "meta": self.meta}, f)

    @classmethod
    def load(cls, path):
        with open(path + ".json") as f:
            d = json.load(f)
        idx = cls(d["dim"])
        idx.vectors, idx.ids, idx.meta = np.load(path + ".npy"), d["ids"], d["meta"]
        return idx
