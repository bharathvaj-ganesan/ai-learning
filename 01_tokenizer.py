"""
STEP 1 - TOKENIZATION (15 min)

Idea: a neural network only does math on numbers. So text -> list of integers ("tokens").
Real LLMs use BPE (byte-pair encoding): start with characters, repeatedly merge the most
frequent adjacent pair into a new token. We build that in ~20 lines.

Run:  python 01_tokenizer.py
"""
from collections import Counter

text = "low lower lowest newer newest wider widest low low lower"

# --- A. Simplest tokenizer: one token per character -------------------------
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
encode = lambda s: [stoi[c] for c in s]
decode = lambda ids: "".join(itos[i] for i in ids)
print("vocab:", chars)
print("encode('low'):", encode("low"), "-> decode:", decode(encode("low")))

# --- B. BPE: merge frequent pairs -------------------------------------------
ids = list(text.encode("utf-8"))          # start from raw bytes (0..255)
print("\nstart: %d tokens" % len(ids))

def count_pairs(ids):
    return Counter(zip(ids, ids[1:]))

def merge(ids, pair, new_id):
    out, i = [], 0
    while i < len(ids):
        if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
            out.append(new_id); i += 2
        else:
            out.append(ids[i]); i += 1
    return out

merges = {}
for step in range(8):                      # TRY THIS: change 8 to 20
    pair, n = count_pairs(ids).most_common(1)[0]
    new_id = 256 + step
    ids = merge(ids, pair, new_id)
    merges[pair] = new_id
    a, b = (bytes([p]) if p < 256 else f"<{p}>" for p in pair)
    print(f"merge {step}: {a} + {b} -> token {new_id} (seen {n}x) | now {len(ids)} tokens")

print("\nFinal token ids:", ids)
print("Compression: %.1fx fewer tokens than bytes" % (len(text) / len(ids)))

# --- C. See a real tokenizer (GPT-2) -----------------------------------------
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("gpt2")
for s in ["Hello world", "unbelievably", "ChargeBee billing", "    indentation", "1234567890"]:
    t = tok.encode(s)
    print(f"{s!r:22} -> {t} -> {[tok.decode([x]) for x in t]}")

"""
TRY THIS
1. Increase merges to 20. Which words become single tokens?
2. In section C, type an emoji or a non-English sentence. Why does it use MANY more tokens?
   (Hint: tokenizer was trained mostly on English text.)
3. Why do LLMs struggle to count letters in 'strawberry'? Look at how it's tokenized.

TAKEAWAY: LLM input = list of ints. Output = a probability over the next int.
"""
