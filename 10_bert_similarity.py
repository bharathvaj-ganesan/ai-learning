"""
BERT 4 of 5 - SENTENCE SIMILARITY (10 min)

Average every token vector in a sentence and you get one 768-dim vector
for the whole sentence. Cosine similarity then ranks other sentences
against a query.

This is the idea behind semantic search. Raw BERT is only mediocre at it:
it was trained to fill blanks, not to compare sentences. Sentence-BERT is
the same idea after training on sentence pairs, and that is what retrieval uses.

Run:  python 10_bert_similarity.py
Next: python 11_bert_next_sentence.py
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


def sentence_vector(text):
    """Mean of the token vectors, skipping padding. Includes [CLS] and [SEP]."""
    enc = tok(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        hidden = bert(**enc).last_hidden_state[0]
    mask = enc.attention_mask[0].unsqueeze(-1)
    return (hidden * mask).sum(0) / mask.sum()


def cos(a, b):
    return F.cosine_similarity(a, b, dim=0).item()


query = "A man is playing a guitar on stage."
candidates = [
    "A musician is performing with his instrument.",
    "A man is playing soccer in the park.",
    "There is a guitar leaning against the wall.",
    "A dog is sleeping on the porch.",
    "The concert was loud and the crowd loved it.",
]
qv = sentence_vector(query)
print(f'Query: "{query}"\n')
for c in sorted(candidates, key=lambda s: cos(qv, sentence_vector(s)), reverse=True):
    print(f"  {cos(qv, sentence_vector(c)):+.3f}  {c}")

print("\nEvery score is high. 'playing soccer' can beat 'musician performing'.")
print("Read the order, then the gap between first and last. The gap is small.")

if sys.stdin.isatty():
    print("\nYour turn:  sentence A | sentence B     Empty line moves on.")
    while True:
        try:
            line = input("sim> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            break
        a, _, b = line.partition("|")
        if not b.strip():
            print("    usage: I love dogs | I adore puppies")
            continue
        print(f"    cosine = {cos(sentence_vector(a.strip()), sentence_vector(b.strip())):+.3f}")

"""
TRY THIS
1. Add 'On stage a guitar is playing a man.' as a candidate.
   Same words, different meaning. The score stays high: these vectors
   mostly carry topic, not who-did-what.
2. Compare a sentence with itself. You should get +1.000.
3. Compare two unrelated sentences. Still well above 0. That is the cone
   from the previous file, now at sentence scale.

TAKEAWAY: one vector per sentence, compared by angle. Useful, and blunt, until the model is trained for it.
"""
