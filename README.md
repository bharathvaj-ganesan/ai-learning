# Learn LLMs in 2 Hours (hands-on, for SWE folks)

Goal: by the end you can explain **how an LLM works** to someone else, because you
built a tiny one yourself and poked at a real one.

Every step = run a script, change a number, watch what happens. Theory is limited to
one-liners inside the scripts.

## Setup (5 min)

```bash
cd /Users/bharathvajganesan/personal/ai/learn-llms
python3 -m venv .venv            # already created for you
source .venv/bin/activate
pip install torch numpy matplotlib transformers
python 00_check.py
```

## The 2-hour schedule

| Time | Step | File | You will learn |
|------|------|------|----------------|
| 0:00-0:05 | Setup | `00_check.py` | Env works |
| 0:05-0:20 | 1. Tokenization | `01_tokenizer.py` | Text -> integers. LLMs never see text. |
| 0:20-0:35 | 2. Dumbest LM | `02_bigram_counts.py` | "Language model" = predict next token. Sampling. |
| 0:35-0:55 | 3. Learn it with a neural net | `03_neural_bigram.py` | Embeddings, logits, softmax, loss, gradient descent |
| 0:55-1:20 | 4. Attention | `04_attention.py` | The one idea that makes transformers work |
| 1:20-1:45 | 5. Mini-GPT | `05_mini_gpt.py` | Full transformer, trained on Shakespeare, on your laptop |
| 1:45-1:55 | 6. Real model | `06_real_gpt2.py` | Same thing, 124M params. Sampling knobs. |
| 1:55-2:00 | 7. Teach-back | `TEACH_BACK.md` | Explain it in 5 minutes |

Illustrations: run `python make_illustrations.py` -> PNGs land in `illustrations/`.
Look at them when a step says **[look]**.

## How to work each step
1. Read the header comment of the script (30 seconds).
2. Run it.
3. Do the **TRY THIS** exercises at the bottom of the script. This is where learning happens.
4. Move on even if it's not 100% clear. Step 5 ties everything together.

## Rule of thumb
If you're stuck >5 min on one step, skip it. The final teach-back only needs the big picture.
