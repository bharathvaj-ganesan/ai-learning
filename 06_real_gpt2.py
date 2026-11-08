"""
STEP 6 - A REAL MODEL (10 min)

GPT-2 small (124M params) has the SAME architecture as your mini-GPT. Here we:
  A. look at the actual next-token probability distribution
  B. write the generation loop ourselves (this is what every chat API does)
  C. play with the sampling knobs (temperature, top-k)
  D. count parameters per component

Run:  python 06_real_gpt2.py    (downloads ~500MB once)
"""
import torch, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

tok = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2").eval()

# --- A. Next-token distribution ---------------------------------------------------
prompt = "The capital of France is"
ids = tok(prompt, return_tensors="pt").input_ids
with torch.no_grad():
    logits = model(ids).logits[0, -1]           # scores for the next token, over 50,257 tokens
probs = F.softmax(logits, dim=-1)
print(f'Prompt: "{prompt}"\nTop next-token candidates:')
for p, i in zip(*probs.topk(8)):
    print(f"  {tok.decode(i)!r:12} {p:.3f}")
print("-> The model doesn't 'know' facts like a database; it assigns probabilities. Higher for ' the', ' Paris', etc.")

# --- B. Our own generation loop (autoregression) ------------------------------------
def generate(prompt, n=30, temperature=1.0, top_k=None, seed=0):
    torch.manual_seed(seed)
    ids = tok(prompt, return_tensors="pt").input_ids
    for _ in range(n):
        with torch.no_grad():
            logits = model(ids).logits[0, -1] / max(temperature, 1e-5)
        if top_k:
            kth = torch.topk(logits, top_k).values[-1]
            logits[logits < kth] = float("-inf")
        nxt = torch.multinomial(F.softmax(logits, -1), 1)
        ids = torch.cat([ids, nxt[None]], dim=1)      # append and feed everything back in
    return tok.decode(ids[0])

# --- C. Sampling knobs --------------------------------------------------------------
p = "Once upon a time, a software engineer"
print("\n--- temperature ~0 (greedy, always top choice; deterministic) ---\n", generate(p, temperature=0))
print("\n--- temperature 0.7 ---\n", generate(p, temperature=0.7))
print("\n--- temperature 1.5 (chaotic) ---\n", generate(p, temperature=1.5))
print("\n--- top_k=5 ---\n", generate(p, top_k=5))

# --- D. Anatomy ----------------------------------------------------------------------
print("\nParameters:")
tot = sum(p.numel() for p in model.parameters())
emb = model.transformer.wte.weight.numel()
blk = sum(p.numel() for p in model.transformer.h.parameters())
print(f"  total {tot/1e6:.0f}M | token embeddings {emb/1e6:.0f}M | {len(model.transformer.h)} transformer blocks {blk/1e6:.0f}M")
print("  config:", model.config.n_layer, "layers,", model.config.n_head, "heads,", model.config.n_embd, "dim, context", model.config.n_positions)

"""
TRY THIS
1. Prompt: "Q: What is 17 * 23?\\nA:" - GPT-2 fails. Why? (no reasoning training, too small, tokenization of digits)
2. Few-shot prompting: give 3 examples of "English: ... French: ..." then a new English line. It "learns in context" with no weight updates!
3. Same seed + temperature 0 twice -> identical. Real APIs are 'random' only because of temperature > 0.
4. Why does generation get slower for long outputs? (each step re-reads the whole context. Real systems cache K/V = "KV cache".)

TAKEAWAY: LLM = loop { probs = model(context); next = sample(probs); context += next }.
"""
