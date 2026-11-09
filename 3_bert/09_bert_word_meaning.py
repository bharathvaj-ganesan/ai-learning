"""
BERT 3 of 5 - ONE WORD, MANY VECTORS (10 min)

Fill-in-the-blank showed that context changes the guess.
Here you look at the thing BERT actually builds: a 768-number vector
for every token, computed from the whole sentence.

Same spelling, different sentence -> different vector.
Cosine similarity compares two vectors. 1.0 = same direction, 0 = unrelated.

Run:  python 3_bert/09_bert_word_meaning.py
Next: python 3_bert/10_bert_similarity.py
"""
import os
os.environ["TQDM_DISABLE"] = "1"

import sys
import logging
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

logging.getLogger("torch").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

MODEL = "bert-base-uncased"
device = "mps" if torch.backends.mps.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(MODEL)
bert = AutoModel.from_pretrained(MODEL).to(device).eval()


def token_vector(sentence, word):
    """The 768-dim vector for the first occurrence of `word` in `sentence`."""
    enc = tok(sentence, return_tensors="pt").to(device)
    tokens = tok.convert_ids_to_tokens(enc.input_ids[0])
    if word not in tokens:
        raise ValueError(f"{word!r} is not its own token in {tokens}. Try a simpler word.")
    with torch.no_grad():
        hidden = bert(**enc).last_hidden_state[0]  # (seq_len, 768)
    return hidden[tokens.index(word)]


def cos(a, b):
    return F.cosine_similarity(a, b, dim=0).item()


river = token_vector("I sat on the bank of the river and skipped stones.", "bank")
money = token_vector("I went to the bank to deposit my paycheck.", "bank")
loan = token_vector("The bank approved my home loan yesterday.", "bank")
river_word = token_vector("I sat on the bank of the river and skipped stones.", "river")

print(f"bank (river)  vs bank (money)   {cos(river, money):+.3f}   same spelling, different sentence")
print(f"bank (money)  vs bank (loan)    {cos(money, loan):+.3f}   two money sentences")
print(f"bank (river)  vs river          {cos(river, river_word):+.3f}   different words, same sentence")
print("\nBERT vectors sit in a cone, so even a loose pair scores well above 0.")
print("Compare the three numbers with each other, not with a fixed cutoff.")

if sys.stdin.isatty():
    print("\nYour turn:  word | sentence     Empty line moves on.")
    print("Example:    bank | I went to the bank to deposit money.")
    while True:
        try:
            line = input("word> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            break
        word, _, sent = line.partition("|")
        word, sent = word.strip().lower(), sent.strip()
        if not sent:
            print("    usage: bank | I went to the bank to deposit money.")
            continue
        try:
            v = token_vector(sent, word)
            print(f"    768-dim vector, length {v.norm():.2f}, "
                  f"first 5 dims {[round(float(x), 3) for x in v[:5]]}")
        except Exception as e:
            print(f"    oops: {e}")

"""
TRY THIS
1. Ask for 'bank' in the identical sentence twice. Cosine of a vector with
   itself is 1. That checks the plumbing.
2. 'apple' the fruit vs 'apple' the company. The model is uncased, so the
   capital letter is gone before BERT ever sees it. Does the sentence still
   pull the two apart?
3. Try 'unbelievably'. It is five tokens (see 07), so it has no single vector.

TAKEAWAY: a word's vector is rebuilt for every sentence. Context is the input, not a footnote.
"""
