"""
STEP 3 - SAME MODEL, BUT LEARNED WITH GRADIENT DESCENT (20 min)

We replace the counting table with a matrix of *learnable numbers* (weights) and let
the computer find good values by nudging them to reduce the loss. This is ALL of
"training" - everything later is just a bigger function in the middle.

Pipeline for one step:
   token id -> embedding row (vector of numbers) -> logits (raw scores) ->
   softmax (probabilities) -> cross-entropy loss -> backward() -> nudge weights

Run:  python 1_foundations/03_neural_bigram.py   (needs input.txt from step 2)
"""
import torch
import torch.nn.functional as F

text = open("input.txt").read()
chars = sorted(set(text)); V = len(chars)
stoi = {c: i for i, c in enumerate(chars)}; itos = {i: c for c, i in stoi.items()}
data = torch.tensor([stoi[c] for c in text])

xs, ys = data[:-1], data[1:]            # input char -> target next char

# The "model": a V x V table of learnable numbers (logits for each prev char)
W = torch.randn(V, V, requires_grad=True)

print("step  loss")
for step in range(1001):
    logits = W[xs]                         # lookup: shape (N, V)  <- this is an "embedding"
    loss = F.cross_entropy(logits, ys)     # softmax + -log(prob of correct char), averaged
    if step % 200 == 0:
        print(f"{step:4d}  {loss.item():.3f}")
    W.grad = None
    loss.backward()                        # compute d(loss)/d(W) for every weight - automatic calculus
    W.data -= 20.0 * W.grad                # gradient descent: move weights downhill. 20 = learning rate

# Compare to step 2: the loss creeps toward ~2.45, the SAME number the counting table gave. Same model, learned instead of counted.

def generate(n=300):
    cur, out = stoi["\n"], []
    for _ in range(n):
        p = F.softmax(W[cur], dim=0)
        cur = torch.multinomial(p, 1).item()
        out.append(itos[cur])
    return "".join(out)

print(generate())

"""
TRY THIS
1. Learning rate 20 -> 0.01. Why does it learn slowly? 20 -> 500. What happens? (divergence)
2. Print W[stoi['q']].softmax(0) and compare to step 2's probs[stoi['q']]. Same!
3. Replace `W[xs]` with a 2-layer net: emb = E[xs] (V x 16), logits = emb @ U (16 x V).
   It works with fewer numbers (V*16*2 instead of V*V). That's a learned *embedding*:
   each token becomes a small vector of numbers capturing its "meaning".

TAKEAWAY:
  Training loop = forward -> loss -> backward -> update. Repeat millions of times.
  GPT training is literally this loop with a fancier function than `W[xs]`.
  The fancier function must let tokens look at *earlier tokens*. That's attention (step 4).
"""
