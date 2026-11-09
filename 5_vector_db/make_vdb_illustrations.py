"""Draws one picture per vector-DB step into illustrations/.  Run: python 5_vector_db/make_vdb_illustrations.py"""
import os
os.environ["TQDM_DISABLE"] = "1"
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from flat_index import FlatIndex, normalize
from ivf_index import IVFIndex

OUT = "illustrations"
os.makedirs(OUT, exist_ok=True)
C = {"blue": "#2b6cb0", "orange": "#dd6b20", "green": "#2f855a", "red": "#c53030", "gray": "#a0aec0", "purple": "#6b46c1"}
rng = np.random.default_rng(3)


def save(fig, name):
    fig.savefig(f"{OUT}/{name}", dpi=130, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- 1. distances
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
a, b, c = np.array([1.0, 1.5]), np.array([2.0, 3.0]), np.array([2.5, -0.5])
for v, name, col in [(a, "a", C["blue"]), (b, "b  (same direction, 2x longer)", C["orange"]), (c, "c  (different direction)", C["green"])]:
    ax1.annotate("", xy=v, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=col, lw=2.5))
    ax1.text(v[0] + 0.08, v[1] + 0.08, name, color=col, fontsize=10)
ax1.plot([a[0], b[0]], [a[1], b[1]], ":", color=C["gray"])
ax1.text(1.55, 2.05, f"L2(a,b)={np.linalg.norm(a-b):.2f}\ncosine(a,b)=1.00", fontsize=9, color="#555")
cosac = a @ c / (np.linalg.norm(a) * np.linalg.norm(c))
ax1.text(1.4, 0.3, f"cosine(a,c)={cosac:.2f}", fontsize=9, color="#555")
ax1.set_xlim(-0.3, 4); ax1.set_ylim(-1.2, 3.8); ax1.set_aspect("equal"); ax1.grid(alpha=.25)
ax1.set_title("Raw vectors: cosine ignores length,\nL2 does not")

ax2.add_patch(Circle((0, 0), 1, fill=False, color=C["gray"], lw=1.5))
pts = normalize(rng.normal(size=(14, 2)) + np.array([1.2, 0.6]))
q = normalize(np.array([1.0, 0.45]))
order = np.argsort(-(pts @ q))
ax2.scatter(*pts.T, c=C["gray"], s=40, zorder=3)
ax2.scatter(*pts[order[:3]].T, c=C["blue"], s=70, zorder=4, label="top-3 by dot product")
ax2.scatter(*q, marker="*", s=300, c=C["red"], zorder=5, label="query")
ang = np.arctan2(pts[order[2], 1] - q[1], pts[order[2], 0] - q[0])
r = np.linalg.norm(pts[order[2]] - q)
ax2.add_patch(Circle(q, r, fill=False, ls="--", color=C["blue"], lw=1.2))
ax2.text(-1.25, -1.45, "Same 3 points are inside the L2 circle: after normalizing,\ndot, cosine and L2 all give the same ranking.", fontsize=8.5)
ax2.set_xlim(-1.3, 1.5); ax2.set_ylim(-1.5, 1.5); ax2.set_aspect("equal"); ax2.legend(loc="upper left", fontsize=8)
ax2.set_title("Normalized (length 1): everything lives on a circle")
fig.suptitle("Step 1 - Distance metrics", fontweight="bold", y=1.04)
save(fig, "vdb_1_distances.png")

# ---------------------------------------------------------------- 2. flat search
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
centers = np.array([[0, 0], [4, 1], [1.5, 4.5], [5, 5]])
X = np.vstack([cen + 0.7 * rng.normal(size=(40, 2)) for cen in centers])
q = np.array([4.2, 1.8])
d = np.linalg.norm(X - q, axis=1)
top = np.argsort(d)[:5]
for p in X:
    ax1.plot([q[0], p[0]], [q[1], p[1]], color=C["gray"], lw=0.35, alpha=.6)
ax1.scatter(*X.T, c=C["gray"], s=22, zorder=3)
ax1.scatter(*X[top].T, c=C["blue"], s=70, zorder=4, label="top-5 (exact)")
ax1.scatter(*q, marker="*", s=300, c=C["red"], zorder=5, label="query")
ax1.set_title("Flat index: query is compared with EVERY vector\n(160 comparisons for 160 vectors)")
ax1.legend(fontsize=8); ax1.set_aspect("equal"); ax1.axis("off")

ns = [1_000, 10_000, 100_000, 300_000, 1_000_000]
ms = []
for n in ns:
    big = FlatIndex(128)
    big.add_batch(range(n), rng.normal(size=(n, 128)))
    t = time.perf_counter()
    for _ in range(3):
        big.search(rng.normal(size=128), k=10)
    ms.append((time.perf_counter() - t) / 3 * 1000)
ax2.plot(ns, ms, "o-", color=C["orange"], lw=2)
for n, m in zip(ns, ms):
    ax2.text(n, m * 1.15, f"{m:.2g} ms", ha="center", fontsize=8)
ax2.set_xscale("log"); ax2.set_yscale("log"); ax2.grid(alpha=.3, which="both")
ax2.set_xlabel("vectors stored (N)"); ax2.set_ylabel("ms per query (dim=128)")
ax2.set_title("Cost grows in a straight line with N\n10x more data = ~10x slower")
fig.suptitle("Step 2 - Exact (brute-force) search", fontweight="bold", y=1.04)
save(fig, "vdb_2_flat_search.png")

# ---------------------------------------------------------------- 3. IVF
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
K = 8
cent = X[rng.choice(len(X), K, replace=False)].copy()
for _ in range(15):
    asg = np.argmin(((X[:, None] - cent[None]) ** 2).sum(-1), axis=1)
    for j in range(K):
        if (asg == j).any():
            cent[j] = X[asg == j].mean(0)
gx, gy = np.meshgrid(np.linspace(-2.5, 7.5, 300), np.linspace(-2.5, 7.5, 300))
cell = np.argmin(((np.c_[gx.ravel(), gy.ravel()][:, None] - cent[None]) ** 2).sum(-1), axis=1).reshape(gx.shape)
cmap = plt.get_cmap("tab10")
def misses(qq, nprobe):
    t5 = set(np.argsort(np.linalg.norm(X - qq, axis=1))[:5])
    o = np.argsort(np.linalg.norm(cent - qq, axis=1))[:nprobe]
    return len([i for i in t5 if asg[i] not in o])


cands = [np.array([x, y]) for x in np.linspace(-1, 6.5, 60) for y in np.linspace(-1, 6.5, 60)]
good = [qq for qq in cands if misses(qq, 1) >= 2 and misses(qq, 3) == 0]
q = good[len(good) // 2] if good else np.array([3.0, 1.7])
true5 = set(np.argsort(np.linalg.norm(X - q, axis=1))[:5])
order = np.argsort(np.linalg.norm(cent - q, axis=1))
for ax, nprobe in zip(axes[:2], (1, 3)):
    ax.imshow(cell, extent=(-2.5, 7.5, -2.5, 7.5), origin="lower", cmap="tab10", alpha=.12, vmin=0, vmax=9)
    opened = order[:nprobe]
    scan = np.isin(asg, opened)
    ax.scatter(*X[~scan].T, c=C["gray"], s=18, alpha=.6, zorder=3)
    ax.scatter(*X[scan].T, c=C["blue"], s=26, zorder=4)
    found = [i for i in true5 if scan[i]]
    missed = [i for i in true5 if not scan[i]]
    ax.scatter(*X[found].T, facecolors="none", edgecolors=C["green"], s=140, lw=2, zorder=5) if found else None
    ax.scatter(*X[missed].T, facecolors="none", edgecolors=C["red"], s=140, lw=2, zorder=5) if missed else None
    ax.scatter(*cent.T, marker="X", s=90, c="black", zorder=6)
    ax.scatter(*q, marker="*", s=300, c=C["red"], zorder=7)
    ax.set_title(f"nprobe={nprobe}: scan {scan.sum()}/{len(X)} vectors\n{len(found)}/5 true neighbors found" +
                 ("  (red ring = missed)" if missed else ""))
    ax.set_xlim(-2.5, 7.5); ax.set_ylim(-2.5, 7.5); ax.set_aspect("equal"); ax.axis("off")

N, DIM = 50_000, 64
cs = rng.normal(size=(200, DIM))
data = cs[rng.integers(0, 200, N)] + 1.0 * rng.normal(size=(N, DIM))
qs = cs[rng.integers(0, 200, 100)] + 1.0 * rng.normal(size=(100, DIM))
flat = FlatIndex(DIM); flat.add_batch(range(N), data)
ivf = IVFIndex(DIM, 128); ivf.train(data[:15_000]); ivf.add_batch(data)
truth = [{r[0] for r in flat.search(qq, 10)} for qq in qs]
t = time.perf_counter(); [flat.search(qq, 10) for qq in qs]; flat_ms = (time.perf_counter() - t) / 100
probes = [1, 2, 4, 8, 16, 32, 64]
rec, spd = [], []
for p in probes:
    t = time.perf_counter(); res = [ivf.search(qq, 10, p) for qq in qs]; ms_ = (time.perf_counter() - t) / 100
    rec.append(np.mean([len(truth[i] & {r[0] for r in res[i]}) / 10 for i in range(100)]) * 100)
    spd.append(flat_ms / ms_)
ax = axes[2]
ax.plot(probes, rec, "o-", color=C["green"], lw=2, label="recall@10 (%)")
ax.set_xscale("log", base=2); ax.set_ylim(0, 105); ax.set_xlabel("nprobe (buckets opened)"); ax.set_ylabel("recall@10 (%)", color=C["green"])
ax2 = ax.twinx()
ax2.plot(probes, spd, "s--", color=C["orange"], lw=2)
ax2.set_ylabel("speedup vs flat (x)", color=C["orange"]); ax2.axhline(1, color=C["gray"], lw=.8, ls=":")
ax.set_title("Real data (50k x 64-d): recall saturates fast,\nspeedup shrinks as nprobe grows")
ax.grid(alpha=.3)
fig.suptitle("Step 3 - IVF approximate search: open fewer buckets, risk missing neighbors", fontweight="bold", y=1.04)
save(fig, "vdb_3_ivf.png")

# ---------------------------------------------------------------- 4. text search
import logging, torch
from transformers import AutoModel, AutoTokenizer
logging.getLogger("transformers").setLevel(logging.ERROR)
dev = "mps" if torch.backends.mps.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained("bert-base-uncased")
bert = AutoModel.from_pretrained("bert-base-uncased").to(dev).eval()


def embed(text):
    enc = tok(text, return_tensors="pt", truncation=True, max_length=128).to(dev)
    with torch.no_grad():
        h = bert(**enc).last_hidden_state[0]
    return (h * enc.attention_mask[0].unsqueeze(-1)).sum(0).cpu().numpy() / enc.attention_mask.sum().item()


docs = [("cat on windowsill", "animals"), ("puppy chased ball", "animals"), ("birds migrate south", "animals"),
        ("bank raised rates", "finance"), ("stocks fell sharply", "finance"), ("inflation cooled", "finance"),
        ("compiler error line 42", "tech"), ("deployed model to prod", "tech"), ("training network took hours", "tech")]
sent = ["The cat sat on the warm windowsill.", "A puppy chased the ball across the yard.", "Birds migrate south when winter arrives.",
        "The central bank raised interest rates again.", "Stocks fell sharply after the earnings report.", "Inflation cooled slightly last quarter.",
        "The compiler threw an error on line 42.", "She deployed the new model to production.", "Training the neural network took three hours."]
queries = ["my dog loves playing fetch", "the economy and borrowing costs", "debugging my machine learning code"]
E = normalize(np.array([embed(s) for s in sent])); Q = normalize(np.array([embed(s) for s in queries]))
mu = E.mean(0)
_, _, Vt = np.linalg.svd(E - mu, full_matrices=False)
P, PQ = (E - mu) @ Vt[:2].T, (Q - mu) @ Vt[:2].T
fig, ax = plt.subplots(figsize=(9, 6.5))
colors = {"animals": C["green"], "finance": C["blue"], "tech": C["purple"]}
for (label, topic), p in zip(docs, P):
    ax.scatter(*p, c=colors[topic], s=90, zorder=3)
    ax.text(p[0] + .004, p[1] + .004, label, fontsize=8, color=colors[topic])
for qq, pq, name in zip(Q, PQ, queries):
    hit = int(np.argmax(E @ qq))
    ax.scatter(*pq, marker="*", s=320, c=C["red"], zorder=5)
    ax.annotate("", xy=P[hit], xytext=pq, arrowprops=dict(arrowstyle="->", color=C["red"], lw=1.5))
    ax.text(pq[0] + .004, pq[1] - .012, f'"{name}"', fontsize=8, color=C["red"], style="italic")
for t_, col in colors.items():
    ax.scatter([], [], c=col, s=60, label=t_)
ax.scatter([], [], marker="*", c=C["red"], s=150, label="query -> nearest stored sentence")
ax.legend(fontsize=8); ax.grid(alpha=.2)
ax.set_title("Step 4 - BERT sentence vectors projected to 2-D (PCA)\nsame-topic sentences sit together; a query lands near its topic and the index returns the closest", fontsize=10)
ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
save(fig, "vdb_4_text_search.png")
print("Saved 4 illustrations in illustrations/")
