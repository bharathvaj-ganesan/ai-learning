"""
STEP 5 - MINI-GPT (25 min) - the whole thing, ~100 lines

Architecture (same as GPT-2/3/4, just smaller):
   tokens -> token embedding + position embedding
          -> N x [ LayerNorm -> multi-head causal attention -> residual add
                   LayerNorm -> MLP (feed-forward)            -> residual add ]
          -> LayerNorm -> linear -> logits over vocab

   - position embedding: attention has no sense of order, so we add "I'm token #i" info.
   - residual (x + f(x)): lets gradients flow, each block only has to learn a small *edit*.
   - MLP: per-token thinking; attention = communication between tokens, MLP = computation.

Run:  python 2_transformers/05_mini_gpt.py          (~1 min on Apple-silicon GPU, longer on CPU; prints loss + samples)
Then set max_iters = 10000 and run again (~4 min): same code, visibly better Shakespeare.
Watch the samples go: gibberish -> real-looking words -> Shakespeare-ish lines.
"""
import torch, torch.nn as nn, torch.nn.functional as F, time

# ---- hyperparameters (TRY THIS: change these) -------------------------------------
block_size = 128      # context window: how many previous chars the model can see
n_embd     = 128      # size of each token's vector
n_head     = 4        # attention heads
n_layer    = 4        # transformer blocks
batch_size = 32
max_iters  = 3000
lr         = 3e-4
device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(1337)

# ---- data -------------------------------------------------------------------------
text = open("input.txt").read()
chars = sorted(set(text)); V = len(chars)
stoi = {c: i for i, c in enumerate(chars)}; itos = {i: c for c, i in stoi.items()}
data = torch.tensor([stoi[c] for c in text])
n = int(0.9 * len(data)); train_data, val_data = data[:n], data[n:]

def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - block_size - 1, (batch_size,))
    x = torch.stack([d[i:i + block_size] for i in ix])
    y = torch.stack([d[i + 1:i + block_size + 1] for i in ix])   # y is x shifted by one: predict next char AT EVERY position
    return x.to(device), y.to(device)

# ---- model ------------------------------------------------------------------------
class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)     # q, k, v for all heads in one matmul
        self.proj = nn.Linear(n_embd, n_embd)
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)))
    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q, k, v = (t.view(B, T, n_head, C // n_head).transpose(1, 2) for t in (q, k, v))  # (B, heads, T, hs)
        att = (q @ k.transpose(-2, -1)) / (C // n_head) ** 0.5
        att = att.masked_fill(self.mask[:T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        out = (att @ v).transpose(1, 2).reshape(B, T, C)           # concat heads
        return self.proj(out)

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(n_embd), nn.LayerNorm(n_embd)
        self.attn = Attention()
        self.mlp = nn.Sequential(nn.Linear(n_embd, 4 * n_embd), nn.GELU(), nn.Linear(4 * n_embd, n_embd))
    def forward(self, x):
        x = x + self.attn(self.ln1(x))     # tokens talk to each other
        x = x + self.mlp(self.ln2(x))      # each token thinks on its own
        return x

class MiniGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(V, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block() for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, V)
    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device))
        x = self.ln_f(self.blocks(x))
        logits = self.head(x)                                       # (B, T, V)
        loss = None if targets is None else F.cross_entropy(logits.view(-1, V), targets.view(-1))
        return logits, loss
    @torch.no_grad()
    def generate(self, idx, n_new, temperature=1.0, top_k=None):
        for _ in range(n_new):
            logits, _ = self(idx[:, -block_size:])                  # crop to context window
            logits = logits[:, -1, :] / temperature                 # only the LAST position predicts the next token
            if top_k:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float("-inf")
            idx = torch.cat([idx, torch.multinomial(F.softmax(logits, -1), 1)], dim=1)
        return idx

model = MiniGPT().to(device)
print(f"device={device}  params={sum(p.numel() for p in model.parameters())/1e6:.2f}M  vocab={V}")
opt = torch.optim.AdamW(model.parameters(), lr=lr)

def sample(n=300, **kw):
    ctx = torch.zeros((1, 1), dtype=torch.long, device=device)
    return "".join(itos[i] for i in model.generate(ctx, n, **kw)[0].tolist())

@torch.no_grad()
def eval_loss():
    model.eval(); out = {}
    for s in ("train", "val"):
        out[s] = sum(model(*get_batch(s))[1].item() for _ in range(20)) / 20
    model.train(); return out

# ---- train ------------------------------------------------------------------------
t0 = time.time(); losses = []
for it in range(max_iters + 1):
    if it % 500 == 0:
        l = eval_loss()
        print(f"\n=== iter {it} | train {l['train']:.3f} | val {l['val']:.3f} | {time.time()-t0:.0f}s ===")
        print(sample(200))
    xb, yb = get_batch("train")
    _, loss = model(xb, yb)
    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    losses.append(loss.item())

torch.save(model.state_dict(), "mini_gpt.pt")

# loss curve illustration
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, os
os.makedirs("illustrations", exist_ok=True)
sm = [sum(losses[max(0, i - 50):i + 1]) / len(losses[max(0, i - 50):i + 1]) for i in range(len(losses))]
plt.figure(figsize=(6, 3.5)); plt.plot(sm); plt.axhline(2.45, ls="--", c="gray")
plt.text(max_iters * .5, 2.5, "bigram baseline (step 2/3)", color="gray")
plt.xlabel("training step"); plt.ylabel("loss"); plt.title("Mini-GPT learning"); plt.tight_layout()
plt.savefig("illustrations/mini_gpt_loss.png", dpi=120)

print("\n\n##### FINAL SAMPLES: temperature knob #####")
for temp in (0.5, 1.0, 1.5):
    print(f"\n--- temperature={temp} ---\n{sample(250, temperature=temp)}")

"""
TRY THIS
1. Set n_layer=1, n_head=1: weaker model, higher loss. Set block_size=8: it can't use long context.
2. Compare train vs val loss. If train << val, the model is memorizing (overfitting).
3. Replace `ctx` with a seed prompt: encode "ROMEO:" and generate from it (prompt = initial context!).
4. Comment out the pos_emb line. What breaks? (The model becomes order-blind - a bag of words.)

TAKEAWAY: GPT = embeddings + (attention + MLP) x N + linear head, trained to predict the next token.
That's it. ChatGPT-class models = this + 1000x scale + lots of data + post-training (RLHF) to behave like an assistant.
"""
