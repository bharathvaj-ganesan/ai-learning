"""
STEP 7 (BONUS) - BERT PLAYGROUND (as long as you want)

GPT-2 (step 6) is a DECODER: it reads left-to-right and predicts the NEXT token.
BERT is an ENCODER: it reads the WHOLE sentence at once (both directions) and
builds a rich vector for every token. It can't write stories. It's built to
UNDERSTAND text. It was trained with two games:

  1. Masked Language Modeling - hide 15% of words, guess them from BOTH sides.
  2. Next Sentence Prediction - is sentence B really the sentence after A?

That one difference (bidirectional vs left-to-right) is why BERT-style models
still win at classification / search / embeddings, while GPT-style models win
at generation.

  A. How BERT tokenizes (special tokens [CLS] [SEP] [MASK], subwords)
  B. Fill-in-the-blanks  <- the classic BERT demo (masked language modeling)
  C. Words change meaning <- "bank" by a river != "bank" for money (contextual embeddings)
  D. Sentence similarity <- turn a sentence into a vector, compare with cosine
  E. Next Sentence Prediction (BERT's second training game)

Run:  python 07_bert_playground.py          (downloads ~430MB once, then demos + playground)
      python 07_bert_playground.py --mask "The capital of France is [MASK]."
      python 07_bert_playground.py --similar "I love dogs" "I adore puppies"
"""
import sys

import torch
import torch.nn.functional as F
from transformers import (
    AutoModel,
    AutoModelForMaskedLM,
    AutoModelForNextSentencePrediction,
    AutoTokenizer,
)

MODEL = "bert-base-uncased"  # 110M params, 12 layers, 768-dim vectors, vocab 30,522
device = "mps" if torch.backends.mps.is_available() else "cpu"

print(f"Loading {MODEL} on {device} ... (first run downloads ~430MB)")
tok = AutoTokenizer.from_pretrained(MODEL)
mlm = AutoModelForMaskedLM.from_pretrained(MODEL).to(device).eval()  # for B
bert = AutoModel.from_pretrained(MODEL).to(device).eval()            # for C + D (raw vectors)
print(f"  {sum(p.numel() for p in bert.parameters()) / 1e6:.0f}M params | "
      f"{bert.config.num_hidden_layers} layers | {bert.config.num_attention_heads} heads/layer | "
      f"{bert.config.hidden_size}-dim vectors | vocab {tok.vocab_size:,}\n")


# --- A. Tokenizer: BERT never sees raw text either --------------------------------
print("=" * 72)
print("A. TOKENIZER - [CLS] starts every input, [SEP] separates sentences")
print("=" * 72)
for s in ["Hello world", "unbelievably", "Playing with BERT is fun"]:
    ids = tok(s, return_tensors="pt").input_ids[0].tolist()
    print(f"  {s!r:30} -> {tok.convert_ids_to_tokens(ids)}")
print(f"\n  special tokens: mask={tok.mask_token!r} (id {tok.mask_token_id}), "
      f"cls={tok.cls_token!r}, sep={tok.sep_token!r}")
print("  Note 'unbelievably' -> subwords. BERT splits rare words, never says 'unknown'.")
print("  TRY: it lowercases everything - this checkpoint is 'uncased'.\n")


# --- B. Fill-in-the-blanks (Masked Language Modeling) ------------------------------
def predict_mask(text, top_k=5):
    """Replace [MASK] with BERT's top guesses. Returns [(token, prob), ...]."""
    if tok.mask_token not in text:
        raise ValueError(f"put {tok.mask_token} where the missing word goes: {text!r}")
    enc = tok(text, return_tensors="pt").to(device)
    pos = (enc.input_ids == tok.mask_token_id).nonzero()[0, 1]  # where is the mask?
    with torch.no_grad():
        logits = mlm(**enc).logits[0, pos]                      # scores over all 30,522 words
    probs = F.softmax(logits, dim=-1)
    return [(tok.decode([i]).strip(), float(p)) for p, i in zip(*probs.topk(top_k))]


print("=" * 72)
print("B. FILL-IN-THE-BLANKS - BERT guesses the hidden word from BOTH sides")
print("=" * 72)
print("  (GPT-2 could only use the words on the left. BERT uses left AND right.)\n")
for s in [
    "The capital of France is [MASK].",
    "I deposited my salary in the [MASK].",
    "I sat on the [MASK] of the river and watched the water flow.",
    "The doctor examined the [MASK] and prescribed some medicine.",
    "She spread butter on the warm [MASK] and took a bite.",
]:
    guesses = "  ".join(f"{w} ({p:.2f})" for w, p in predict_mask(s))
    print(f"  {s}\n    -> {guesses}\n")
print("  Watch #2 vs #3: same missing word 'bank', different answer. Context decides.\n")


# --- C. Contextual embeddings: one word, many vectors -------------------------------
def token_vector(sentence, word):
    """The 768-dim vector BERT built for ONE occurrence of `word` in `sentence`."""
    enc = tok(sentence, return_tensors="pt").to(device)
    tokens = tok.convert_ids_to_tokens(enc.input_ids[0])
    if word not in tokens:
        raise ValueError(f"{word!r} is not a single token in {tokens} - try a simpler word")
    with torch.no_grad():
        hidden = bert(**enc).last_hidden_state[0]  # one vector per token, (seq_len, 768)
    return hidden[tokens.index(word)]


print("=" * 72)
print("C. 'BANK' IS NOT ONE VECTOR - meaning depends on the sentence around it")
print("=" * 72)
print("  (Old word2vec gave every word ONE fixed vector. BERT builds it fresh")
print("   for every sentence - that's the whole point of attention.)\n")
bank_river = token_vector("I sat on the bank of the river and skipped stones.", "bank")
bank_money = token_vector("I went to the bank to deposit my paycheck.", "bank")
bank_money2 = token_vector("The bank approved my home loan yesterday.", "bank")
river_vec = token_vector("I sat on the bank of the river and skipped stones.", "river")
cos = lambda a, b: F.cosine_similarity(a, b, dim=0).item()
print(f"  bank (river)  vs bank (money)   cosine = {cos(bank_river, bank_money):+.3f}   <- same spelling, different meaning")
print(f"  bank (money)  vs bank (loan)    cosine = {cos(bank_money, bank_money2):+.3f}   <- same meaning, should be higher")
print(f"  bank (river)  vs river          cosine = {cos(bank_river, river_vec):+.3f}   <- different words, same sentence")
print("\n  Cosine: 1.0 = same direction/meaning, 0 = unrelated. BERT vectors are")
print("  'anisotropic' (all huddle in a cone), so even unrelated pairs score > 0.\n")


# --- D. Sentence similarity: a sentence becomes one vector ---------------------------
def sentence_vector(text):
    """Average all token vectors (ignoring padding) -> one 768-dim vector."""
    enc = tok(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        hidden = bert(**enc).last_hidden_state[0]
    mask = enc.attention_mask[0].unsqueeze(-1)  # (seq_len, 1)
    return (hidden * mask).sum(0) / mask.sum()


print("=" * 72)
print("D. SENTENCE SIMILARITY - rank sentences by meaning, not keywords")
print("=" * 72)
query = "A man is playing a guitar on stage."
candidates = [
    "A musician is performing with his instrument.",
    "A man is playing soccer in the park.",
    "There is a guitar leaning against the wall.",
    "A dog is sleeping on the porch.",
    "The concert was loud and the crowd loved it.",
]
qv = sentence_vector(query)
ranked = sorted(candidates, key=lambda c: cos(qv, sentence_vector(c)), reverse=True)
print(f'  Query: "{query}"\n')
for c in ranked:
    print(f"    {cos(qv, sentence_vector(c)):+.3f}  {c}")
print("\n  Honest result: the ranking is only roughly right, and every score is")
print("  high (0.65-0.88). Raw BERT was never trained to compare sentences, so")
print("  its sentence vectors are mediocre at this - 'playing soccer' can edge")
print("  out 'musician performing'. Fine-tuning on sentence pairs fixes it:")
print("  that model is called Sentence-BERT, and it's what semantic search /")
print("  RAG retrieval actually uses. Same idea, trained for the job.\n")


# --- E. Next Sentence Prediction (BERT's second training game) ----------------------
print("=" * 72)
print("E. NEXT SENTENCE PREDICTION - does B actually follow A?")
print("=" * 72)
nsp = AutoModelForNextSentencePrediction.from_pretrained(MODEL).to(device).eval()


def next_sentence_prob(a, b):
    enc = tok(a, b, return_tensors="pt").to(device)  # [CLS] A [SEP] B [SEP]
    with torch.no_grad():
        probs = F.softmax(nsp(**enc).logits, dim=-1)[0]  # [is_next, not_next]
    return float(probs[0])


for a, b in [
    ("I am going to the store.", "I need to buy some milk."),
    ("I am going to the store.", "The quantum field theory exam was hard."),
    ("She opened the book.", "She began reading the first chapter."),
    ("She opened the book.", "Bananas are a good source of potassium."),
]:
    print(f"  P(next) = {next_sentence_prob(a, b):.3f}   {a!r} -> {b!r}")
print("\n  Fun fact: later research found this game barely helps, so BERT's")
print("  successor RoBERTa just dropped it. Science working as intended.\n")


# --- Interactive playground -----------------------------------------------------------
def playground():
    print("=" * 72)
    print("PLAYGROUND - type a command, 'q' to quit")
    print("=" * 72)
    print(f"  mask  The Eiffel Tower is in [MASK].     fill in the blank (top 5)")
    print(f"  sim   <sentence A> | <sentence B>        how similar are they? (-1..1)")
    print(f"  word  bank | I sat by the river bank.    vector for a word in context")
    print()
    while True:
        try:
            line = input("bert> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not line or line.lower() in ("q", "quit", "exit"):
            return
        cmd, _, rest = line.partition(" ")
        try:
            if cmd == "mask":
                text = rest.strip() or "The capital of France is [MASK]."
                if "[MASK]" not in text:
                    text = text.replace("_", "[MASK]")  # allow _ as a shortcut
                for w, p in predict_mask(text):
                    print(f"    {w:15} {'#' * int(p * 40):40} {p:.3f}")
            elif cmd == "sim":
                a, _, b = rest.partition("|")
                if not b.strip():
                    print("    usage: sim <sentence A> | <sentence B>")
                    continue
                print(f"    cosine = {cos(sentence_vector(a.strip()), sentence_vector(b.strip())):+.3f}")
            elif cmd == "word":
                word, _, sent = rest.partition("|")
                word, sent = word.strip().lower(), sent.strip()
                if not sent:
                    print("    usage: word bank | I sat by the river bank.")
                    continue
                v = token_vector(sent, word)
                print(f"    {word!r} in context -> 768-dim vector, |v| = {v.norm():.2f}, "
                      f"first 5 dims = {[round(float(x), 3) for x in v[:5]]}")
                print("    now run it again inside a different sentence and compare with 'sim'-style cosine by hand!")
            else:
                print("    commands: mask ... | sim A | B | word w | sentence | q")
        except Exception as e:  # keep the playground alive on bad input
            print(f"    oops: {e}")


if __name__ == "__main__":
    args = sys.argv[1:]
    if args[:1] == ["--mask"] and len(args) > 1:
        for w, p in predict_mask(" ".join(args[1:])):
            print(f"  {w:15} {p:.3f}")
    elif args[:1] == ["--similar"] and len(args) > 2:
        print(f"cosine = {cos(sentence_vector(args[1]), sentence_vector(args[2])):+.3f}")
    elif sys.stdin.isatty():
        playground()
    else:
        print("Tip: run this in a terminal to enter the interactive playground.")

"""
TRY THIS
1. Mask TWO words: "The [MASK] of France is [MASK]." It fails - this code only
   handles one mask. Fix it: predict one, fill it in, then predict the other.
   Notice the second guess now depends on your first (possibly wrong) guess.
2. In section B, mask a word where the RIGHT side matters most, e.g.
   "He spread the warm toast with [MASK]." Then delete the right context
   and watch the guesses get worse. That's bidirectionality, felt directly.
3. In section C, compare "bank" with ITSELF in the identical sentence
   (cosine = 1.0, sanity check), then try "apple" the fruit vs Apple the company.
4. In section D, add a candidate with the SAME words shuffled, e.g.
   "On stage a guitar is playing a man." High or low? Embeddings mostly
   capture topic, not word order / who-did-what. A known BERT weakness.
5. Break it on purpose: "The capital of France is [MASK] [MASK] [MASK]."
   BERT was never trained to generate - it fills ONE slot, it doesn't write.
   Generation is why the world moved to GPT-style decoders (step 6).

TAKEAWAY: BERT = encoder that turns text into context-aware vectors.
Fill-mask shows what it learned; those vectors power search, classification
and clustering. It understands (in the vector sense) but it cannot generate.
"""
