"""
BERT 1 of 5 - TOKENS (5 min)

BERT never sees text. It sees a list of tokens, same as GPT-2 in step 6.
Three tokens are special:

  [CLS]  glued to the start of every input
  [SEP]  marks the end, and the gap between two sentences
  [MASK] the blank you will fill in the next file

Rare words are split into pieces. A piece that continues a word starts with ##.

Run:  python 07_bert_tokens.py
Next: python 08_bert_fill_mask.py
"""
import sys
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("bert-base-uncased")

for s in ["Hello world", "unbelievably", "Playing with BERT is fun"]:
    ids = tok(s).input_ids
    print(f"{s!r:30} -> {tok.convert_ids_to_tokens(ids)}")

print(f"\nmask={tok.mask_token!r} (id {tok.mask_token_id})  "
      f"cls={tok.cls_token!r}  sep={tok.sep_token!r}  vocab {tok.vocab_size:,}")
print("'Hello' and 'hello' become the same token. This checkpoint is uncased.")

if sys.stdin.isatty():
    print("\nYour turn. Type a sentence. Empty line moves on.")
    while True:
        try:
            line = input("tokens> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            break
        print("   ", tok.convert_ids_to_tokens(tok(line).input_ids))

"""
TRY THIS
1. Type an emoji, then a Tamil or Japanese sentence. Count the tokens.
   Emojis and Tamil characters often become [UNK]. Japanese splits into single-character pieces.
2. Type 'blueberry'. It splits into 'blue' and '##berry'. The model sees two pieces,
   not the letters b-l-u-e-b-e-r-r-y, which is why letter-counting questions are hard.

TAKEAWAY: input to BERT = [CLS] + your tokens + [SEP].
"""
