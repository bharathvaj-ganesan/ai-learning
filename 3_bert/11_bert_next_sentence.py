"""
BERT 5 of 5 - NEXT SENTENCE (5 min)

BERT's second training game: given sentence A and sentence B, did B
really come next in the document, or was it a random sentence?

The input is [CLS] A [SEP] B [SEP]. A small head on top of [CLS]
outputs two scores. Index 0 is "B follows A".

Run:  python 3_bert/11_bert_next_sentence.py
Next: python 4_applications/12_autoencoder.py
"""
import os
os.environ["TQDM_DISABLE"] = "1"

import sys
import logging
import torch
import torch.nn.functional as F
from transformers import AutoModelForNextSentencePrediction, AutoTokenizer

logging.getLogger("torch").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

MODEL = "bert-base-uncased"
device = "mps" if torch.backends.mps.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(MODEL)
nsp = AutoModelForNextSentencePrediction.from_pretrained(MODEL).to(device).eval()


def next_sentence_prob(a, b):
    enc = tok(a, b, return_tensors="pt").to(device)            # [CLS] A [SEP] B [SEP]
    with torch.no_grad():
        probs = F.softmax(nsp(**enc).logits, dim=-1)[0]        # [follows, does not]
    return float(probs[0])


pairs = [
    ("I am going to the store.", "I need to buy some milk."),
    ("I am going to the store.", "The quantum field theory exam was hard."),
    ("She opened the book.", "She began reading the first chapter."),
    ("She opened the book.", "Bananas are a good source of potassium."),
]
for a, b in pairs:
    print(f"  {next_sentence_prob(a, b):.3f}   {a}  ->  {b}")

print("\nRoBERTa, the follow-up to BERT, dropped this game. It barely helped.")

if sys.stdin.isatty():
    print("\nYour turn:  sentence A | sentence B     Empty line stops.")
    while True:
        try:
            line = input("next> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            break
        a, _, b = line.partition("|")
        if not b.strip():
            print("    usage: She opened the book. | She began to read.")
            continue
        print(f"    P(follows) = {next_sentence_prob(a.strip(), b.strip()):.3f}")

"""
TRY THIS
1. Take the milk sentence and replace 'milk' with something you can still
   buy. Watch the probability. Then replace it with something you cannot
   buy at a store.
2. Feed the same sentence as both A and B.

TAKEAWAY: BERT was trained to fill a blank and to judge whether two sentences belong together.
The blank-filling is the part that lasted.
"""
