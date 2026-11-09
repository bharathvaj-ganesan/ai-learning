"""
BERT 2 of 5 - FILL IN THE BLANK (10 min)

BERT's first training game: hide a word, guess it from BOTH sides.
GPT-2 only gets the words on the left. The right-hand words are why
'bank' by a river and 'bank' for money come out as different guesses.

Run:  python 3_bert/08_bert_fill_mask.py
Next: python 3_bert/09_bert_word_meaning.py
"""
import os
os.environ["TQDM_DISABLE"] = "1"  # the weight-loading bar is not part of the lesson

import sys
import logging
import torch
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM, AutoTokenizer

logging.getLogger("torch").setLevel(logging.ERROR)          # library noise, not part of the demo
logging.getLogger("transformers").setLevel(logging.ERROR)

MODEL = "bert-base-uncased"
device = "mps" if torch.backends.mps.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(MODEL)
mlm = AutoModelForMaskedLM.from_pretrained(MODEL).to(device).eval()


def predict_mask(text, top_k=5):
    """Top guesses for the first [MASK]. Returns [(word, probability), ...]."""
    if tok.mask_token not in text:
        raise ValueError(f"put {tok.mask_token} where the missing word goes")
    enc = tok(text, return_tensors="pt").to(device)
    pos = (enc.input_ids == tok.mask_token_id).nonzero()[0, 1]  # first blank only
    with torch.no_grad():
        logits = mlm(**enc).logits[0, pos]                      # one score per vocab word
    probs = F.softmax(logits, dim=-1)
    return [(tok.decode([i]).strip(), float(p)) for p, i in zip(*probs.topk(top_k))]


text = "The capital of France is [MASK]."
tokens = tok.convert_ids_to_tokens(tok(text).input_ids)
print(text)
print("tokens:", tokens)
print(f"the blank is position {tokens.index('[MASK]')} of {len(tokens)}\n")

for s in [
    text,
    "I deposited my salary in the [MASK].",
    "I sat on the [MASK] of the river and watched the water flow.",
    "The doctor examined the [MASK] and prescribed some medicine.",
    "She spread butter on the warm [MASK] and took a bite.",
]:
    guesses = "  ".join(f"{w} ({p:.2f})" for w, p in predict_mask(s))
    print(f"  {s}\n    -> {guesses}\n")

print("Rows 2 and 3 both answer 'bank'. The surrounding words pick which bank.")

if sys.stdin.isatty():
    print("\nYour turn. Use [MASK], or a single _ . Empty line moves on.")
    while True:
        try:
            line = input("mask> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            break
        if "[MASK]" not in line:
            line = line.replace("_", "[MASK]")
        try:
            for w, p in predict_mask(line):
                print(f"    {w:15} {'#' * int(p * 40):40} {p:.3f}")
        except Exception as e:
            print(f"    oops: {e}")

"""
TRY THIS
1. Two blanks: "The [MASK] of France is [MASK]."
   Only the first blank is scored. Fill that guess back into the sentence
   yourself, then run the second blank. The second guess depends on the first.
2. "He spread the warm toast with [MASK]." Then delete 'warm toast' and run
   it again. The right-hand words were doing a lot of the work.
3. "The capital of France is [MASK] [MASK] [MASK]."
   BERT fills one slot. It was not trained to keep writing. That is step 6.

TAKEAWAY: mask a word, read the whole sentence, score every word in the vocab.
"""
