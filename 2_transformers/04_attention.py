"""
STEP 4 - ATTENTION (25 min) - the core trick of transformers

Problem: bigram only sees 1 previous token. We want each token to look at ALL
previous tokens and pull in the useful information.

Mechanism (database-lookup analogy):
   every token emits a QUERY ("what am I looking for?"),
                     a KEY   ("what do I contain?"),
                     a VALUE ("what I'll hand over if you attend to me").
   score = query . key  -> softmax -> weights -> weighted average of values.
   Causal mask: a token may NOT look at the future (otherwise it would cheat at next-token prediction).

Run:  python 2_transformers/04_attention.py   -> also saves illustrations/attention_weights.png
"""
import os, torch, torch.nn.functional as F
torch.manual_seed(0)

# --- A. Averaging past tokens, 3 ways (same result, each one closer to attention) ----
T = 5
x = torch.randn(T, 4)                       # 5 tokens, each a 4-number vector

# 1. for-loop
avg1 = torch.stack([x[:t + 1].mean(0) for t in range(T)])

# 2. matrix multiply with a lower-triangular matrix
w = torch.tril(torch.ones(T, T)); w = w / w.sum(1, keepdim=True)
avg2 = w @ x

# 3. softmax with -inf mask (this is what attention does; weights become LEARNED instead of uniform)
scores = torch.zeros(T, T).masked_fill(torch.tril(torch.ones(T, T)) == 0, float("-inf"))
avg3 = F.softmax(scores, dim=1) @ x

print("all three equal:", torch.allclose(avg1, avg2), torch.allclose(avg2, avg3))
print("\nmask-then-softmax weights (row = token, col = who it attends to):\n", F.softmax(scores, 1))

# --- B. Real self-attention head ------------------------------------------------------
C, head = 8, 4                              # embedding size, head size
x = torch.randn(T, C)
Wq, Wk, Wv = (torch.randn(C, head) for _ in range(3))   # learnable in real models

q, k, v = x @ Wq, x @ Wk, x @ Wv
scores = (q @ k.T) / head ** 0.5            # dot product = "how relevant is key j to query i"; /sqrt avoids huge numbers
mask = torch.tril(torch.ones(T, T)) == 0
weights = F.softmax(scores.masked_fill(mask, float("-inf")), dim=1)
out = weights @ v                           # each token = weighted blend of earlier tokens' values
print("\nattention weights:\n", weights)
print("output shape:", out.shape, "(same number of tokens; each now 'knows' about the past)")

# --- C. Visualize --------------------------------------------------------------------
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
os.makedirs("illustrations", exist_ok=True)
toks = ["The", "cat", "sat", "on", "it"]
plt.figure(figsize=(4, 4))
plt.imshow(weights.detach(), cmap="Blues")
plt.xticks(range(T), toks); plt.yticks(range(T), toks)
plt.xlabel("attends TO (key)"); plt.ylabel("token (query)")
plt.title("Causal attention weights")
plt.tight_layout(); plt.savefig("illustrations/attention_weights.png", dpi=120)
print("saved illustrations/attention_weights.png  [look]")

"""
TRY THIS
1. Remove the mask line. The upper triangle lights up: tokens see the future. Why is that bad for training?
2. Remove `/ head ** 0.5` and use larger head sizes (64). Weights become one-hot-ish. Why? (softmax of big numbers)
3. Multi-head attention = run several heads in parallel with different Wq/Wk/Wv, concat outputs.
   Each head can learn a different relationship (e.g. "previous word", "matching bracket").

TAKEAWAY:
  Attention = every token computes a learned, content-dependent weighted average over earlier tokens.
  That's how "it" can look back at "cat".
"""
