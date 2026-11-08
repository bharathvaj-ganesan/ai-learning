"""
AUTOENCODER (10 min)

An autoencoder has a narrow middle, called the bottleneck.
The encoder squashes the input into a few numbers.
The decoder rebuilds the input from only those numbers.
The loss is "how far is the rebuild from the original?" No labels.

BERT's fill-in-the-blank (08) is this with one change: some characters
are erased before the encoder sees them, and the loss is still against
the original. That version is called a denoising autoencoder.

Five short sentences, so this can actually learn in about a second.
The bottleneck has 4 numbers. The sentence is longer than that, so the
model has to keep what matters and drop the rest.

Run:  python 12_autoencoder.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

phrases = [
    "to be or not to be",
    "the bank by the river",
    "the bank holds money",
    "a man plays guitar",
    "a dog sleeps outside",
]
L = max(len(p) for p in phrases)
phrases = [p.ljust(L) for p in phrases]
chars = sorted(set("".join(phrases)))
MASK = len(chars)                       # extra id, used only on the way in
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
itos[MASK] = "_"

def ids(s):
    return torch.tensor([stoi[c] for c in s])

clean = torch.stack([ids(p) for p in phrases])   # (5, L)

LATENT = 4


class AE(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(len(chars) + 1, 32)
        self.to_z = nn.Linear(L * 32, LATENT)          # encoder: wide -> 4 numbers
        self.from_z = nn.Linear(LATENT, L * len(chars))  # decoder: 4 numbers -> a char per position

    def forward(self, x):
        z = self.to_z(self.emb(x).flatten(1))              # 4 numbers, no other constraint
        logits = self.from_z(z).view(x.shape[0], L, len(chars))
        return z, logits


def corrupt(x, p=0.3):
    seen = x.clone()
    seen[torch.rand(x.shape) < p] = MASK
    return seen


def text(t):
    return "".join(itos[i] for i in t.tolist())


torch.manual_seed(0)
ae = AE()
opt = torch.optim.Adam(ae.parameters(), lr=1e-2)

print("step  loss     (same loop as step 3: forward, loss, backward, update)")
for step in range(801):
    x = clean[torch.randint(0, len(phrases), (64,))]
    _, logits = ae(corrupt(x))
    loss = F.cross_entropy(logits.reshape(-1, len(chars)), x.reshape(-1))
    opt.zero_grad()
    loss.backward()
    opt.step()
    if step % 200 == 0:
        print(f"{step:4d}  {loss.item():.3f}")

ae.eval()
with torch.no_grad():
    # --- A. The encoder saw a damaged sentence. The decoder still aims at the original.
    print("\nA. damaged input  ->  rebuild        original")
    for row in clean:
        seen = corrupt(row.unsqueeze(0), p=0.35)
        _, logits = ae(seen)
        pred = logits[0].argmax(-1)
        print(f"   {text(seen[0]):22}  {text(pred):22}  {text(row)}")

    # --- B. The whole memory of a sentence is those 4 numbers.
    print("\nB. the bottleneck, one row per sentence")
    zs = []
    for row, phrase in zip(clean, phrases):
        z, logits = ae(row.unsqueeze(0))
        zs.append(z[0])
        pred = text(logits[0].argmax(-1))
        nums = " ".join(f"{v:+.2f}" for v in z[0])
        print(f"   [{nums}]  {pred}")

    def dist(i, j):
        return (zs[i] - zs[j]).norm().item()

    pairs = sorted(
        ((dist(i, j), phrases[i].strip(), phrases[j].strip())
         for i in range(len(phrases)) for j in range(i + 1, len(phrases))),
    )
    print(f"\n   closest  {pairs[0][0]:.2f}   {pairs[0][1]!r}  ~  {pairs[0][2]!r}")
    print(f"   farthest {pairs[-1][0]:.2f}   {pairs[-1][1]!r}  ~  {pairs[-1][2]!r}")
    print("   Distance is not meaning. The 4 numbers only have to be distinct")
    print("   enough for the decoder to tell these five sentences apart.")

    # --- C. A point halfway between two codes is still a sentence-shaped vector.
    i, j = 1, 2  # the two "bank" sentences
    mid = (zs[i] + zs[j]) / 2
    blend = ae.from_z(mid).view(L, len(chars)).argmax(-1)
    print(f"\nC. halfway between the two bank sentences:\n   {text(blend)!r}")

"""
TRY THIS
1. Set LATENT = 2, then LATENT = 16, and rerun. Watch section A.
   Too narrow and the rebuild starts dropping letters. Wider is an easier copy.
2. In corrupt(), change 0.3 to 0.0. The model only ever sees clean input.
   Section A should get worse: it never practiced filling holes.
3. In section C, average a bank sentence with 'a dog sleeps outside'.
   The blend is usually not a real sentence. The space between codes
   was never trained, only the codes themselves.

TAKEAWAY: encoder squeezes, decoder rebuilds, loss compares to the input.
Hide part of the input first and you have BERT's training game.
The squeezed vector is the same kind of object you measured in 09 and 10.
"""
