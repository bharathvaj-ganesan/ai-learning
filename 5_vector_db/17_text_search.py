"""
VECTOR DB 4 of 4 - SEMANTIC SEARCH OVER TEXT (5 min)

Put it together: BERT turns each sentence into a 768-d vector (as in
3_bert/10_bert_similarity.py), FlatIndex stores them, and a text query finds
the nearest sentences by meaning rather than by shared words.

Raw BERT is only a mediocre embedder (see the note in script 10). Production RAG
uses Sentence-BERT style models, but the vector DB side is identical.

Run:  python 5_vector_db/17_text_search.py
"""
import os
os.environ["TQDM_DISABLE"] = "1"
import logging
import torch
from transformers import AutoModel, AutoTokenizer
from flat_index import FlatIndex

logging.getLogger("transformers").setLevel(logging.ERROR)

MODEL = "bert-base-uncased"
device = "mps" if torch.backends.mps.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(MODEL)
bert = AutoModel.from_pretrained(MODEL).to(device).eval()


def embed(text):
    enc = tok(text, return_tensors="pt", truncation=True, max_length=128).to(device)
    with torch.no_grad():
        h = bert(**enc).last_hidden_state[0]
    return (h * enc.attention_mask[0].unsqueeze(-1)).sum(0).cpu().numpy() / enc.attention_mask.sum().item()


docs = [
    ("The cat sat on the warm windowsill.", "animals"),
    ("A puppy chased the ball across the yard.", "animals"),
    ("Birds migrate south when winter arrives.", "animals"),
    ("The central bank raised interest rates again.", "finance"),
    ("Stocks fell sharply after the earnings report.", "finance"),
    ("Inflation cooled slightly last quarter.", "finance"),
    ("The compiler threw an error on line 42.", "tech"),
    ("She deployed the new model to production.", "tech"),
    ("Training the neural network took three hours.", "tech"),
]

db = FlatIndex(768)
db.add_batch(range(len(docs)), [embed(t) for t, _ in docs], [{"text": t, "topic": c} for t, c in docs])

for query in ["my dog loves playing fetch", "the economy and borrowing costs", "debugging my machine learning code"]:
    print(f"\nQuery: {query!r}")
    for _, score, meta in db.search(embed(query), k=3):
        print(f"  {score:.3f}  [{meta['topic']:<7}] {meta['text']}")

print("\nSame query restricted to topic == 'tech':")
for _, score, meta in db.search(embed("the economy and borrowing costs"), k=2, where={"topic": "tech"}):
    print(f"  {score:.3f}  [{meta['topic']:<7}] {meta['text']}")
print("\nFilters change the answer set, not the ranking math.")
