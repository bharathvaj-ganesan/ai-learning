# Learn LLMs in 2 Hours (hands-on, for SWE folks)

Goal: by the end you can explain **how an LLM works** to someone else, because you
built a tiny one yourself and poked at a real one.

Every step = run a script, change a number, watch what happens. Theory is limited to
one-liners inside the scripts.

## Setup (5 min)

```bash
cd /Users/bharathvajganesan/personal/ai/learn-llms
poetry install                   # creates .venv/ in this folder from poetry.lock
eval $(poetry env activate)      # activate it; every `python <folder>/XX.py` below now just works (run from the repo root)
python 1_foundations/00_check.py
```

Without activating, prefix any command with `poetry run`, e.g. `poetry run python 2_transformers/06_real_gpt2.py`.
Add a package with `poetry add <name>` (updates `pyproject.toml` and `poetry.lock`).

## The 2-hour schedule

| Time | Step | File | You will learn |
|------|------|------|----------------|
| 0:00-0:05 | Setup | `1_foundations/00_check.py` | Env works |
| 0:05-0:20 | 1. Tokenization | `1_foundations/01_tokenizer.py` | Text -> integers. LLMs never see text. |
| 0:20-0:35 | 2. Dumbest LM | `1_foundations/02_bigram_counts.py` | "Language model" = predict next token. Sampling. |
| 0:35-0:55 | 3. Learn it with a neural net | `1_foundations/03_neural_bigram.py` | Embeddings, logits, softmax, loss, gradient descent |
| 0:55-1:20 | 4. Attention | `2_transformers/04_attention.py` | The one idea that makes transformers work |
| 1:20-1:45 | 5. Mini-GPT | `2_transformers/05_mini_gpt.py` | Full transformer, trained on Shakespeare, on your laptop |
| 1:45-1:55 | 6. Real model | `2_transformers/06_real_gpt2.py` | Same thing, 124M params. Sampling knobs. |
| 1:55-2:00 | 7. Teach-back | `TEACH_BACK.md` | Explain it in 5 minutes |
| bonus | BERT 1 | `3_bert/07_bert_tokens.py` | `[CLS]` `[SEP]` `[MASK]`, and how words get split |
| bonus | BERT 2 | `3_bert/08_bert_fill_mask.py` | Guess a hidden word from both sides |
| bonus | BERT 3 | `3_bert/09_bert_word_meaning.py` | Same word, different sentence, different vector |
| bonus | BERT 4 | `3_bert/10_bert_similarity.py` | One vector per sentence, compared by angle |
| bonus | BERT 5 | `3_bert/11_bert_next_sentence.py` | Did sentence B really follow sentence A? |
| bonus | Autoencoder | `4_applications/12_autoencoder.py` | Train on ordinary card swipes, flag a swipe that will not rebuild |

Illustrations: run `python make_illustrations.py` -> PNGs land in `illustrations/`.
Look at them when a step says **[look]**.

## How to work each step
1. Read the header comment of the script (30 seconds).
2. Run it.
3. Do the **TRY THIS** exercises at the bottom of the script. This is where learning happens.
4. Move on even if it's not 100% clear. Step 5 ties everything together.

## Rule of thumb
If you're stuck >5 min on one step, skip it. The final teach-back only needs the big picture.
