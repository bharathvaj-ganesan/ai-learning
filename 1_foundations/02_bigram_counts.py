"""
STEP 2 - THE DUMBEST LANGUAGE MODEL (15 min)

A language model = a function: (previous tokens) -> probability of every possible next token.
Here: look at ONLY the previous character, and count what followed it in the training text.
No neural net, no ML. Just a table.

Run:  python 1_foundations/02_bigram_counts.py
"""
import urllib.request, os, random
import torch

# Download tiny Shakespeare (1MB) once
if not os.path.exists("input.txt"):
    url = "https://cdn.jsdelivr.net/gh/karpathy/char-rnn@master/data/tinyshakespeare/input.txt"
    urllib.request.urlretrieve(url, "input.txt")
text = open("input.txt").read()
print("chars in dataset:", len(text))
print(text[:200])

chars = sorted(set(text))
V = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
print("vocab size:", V)

# --- Count table: counts[a][b] = how often char b follows char a --------------
counts = torch.zeros(V, V)
for a, b in zip(text, text[1:]):
    counts[stoi[a], stoi[b]] += 1

probs = counts / counts.sum(1, keepdim=True)    # normalize rows -> probabilities

print("\nAfter 'q', most likely next chars:")
top = probs[stoi["q"]].topk(3)
for p, i in zip(top.values, top.indices):
    print(f"  {itos[i.item()]!r}: {p:.2f}")

# --- Generate text by sampling -------------------------------------------------
def generate(n=300, start="\n"):
    out, cur = [], stoi[start]
    for _ in range(n):
        cur = torch.multinomial(probs[cur], 1).item()   # roll a weighted die
        out.append(itos[cur])
    return "".join(out)

print("\n--- generated (looks like gibberish with Shakespeare-ish letter patterns) ---")
print(generate())

# --- Measure quality: average negative log-likelihood (the LOSS) --------------
nll = 0
for a, b in zip(text, text[1:]):
    nll -= torch.log(probs[stoi[a], stoi[b]])
loss = (nll / (len(text) - 1)).item()
print(f"\nLoss (avg -log prob of the true next char): {loss:.3f}")
print(f"Random guessing would be: {torch.log(torch.tensor(float(V))):.3f}")

"""
TRY THIS
1. Run generate() a few times. Same table, different output - why? (sampling = randomness)
2. Extend to look at the previous TWO characters (table size V*V*V). Does the text improve?
   Now imagine looking at 1000 previous tokens: the table is impossible. That's why we need
   a neural net that *generalizes* instead of memorizing a table.

TAKEAWAY:
  - LM = next-token probability distribution.
  - LOSS = how surprised the model is by the real next token. Lower = better.
  - Generation = repeatedly sample, append, repeat.
"""
