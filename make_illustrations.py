"""Generates concept illustrations into illustrations/. Run: python make_illustrations.py"""
import os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
os.makedirs("illustrations", exist_ok=True)

def box(ax, x, y, w, h, text, color="#dbeafe", fs=10):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", fc=color, ec="#1e3a8a", lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)

def arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14, lw=1.4, color="#334155"))

# 1. Pipeline: text -> answer -----------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 3.2)); ax.axis("off"); ax.set_xlim(0, 12); ax.set_ylim(0, 3)
steps = [("Text\n\"The cat sat\"", "#fef3c7"), ("Tokenizer\n[464, 3797, 3332]", "#dbeafe"),
         ("Embeddings\nvector per token", "#dbeafe"), ("Transformer\nblocks x N", "#dcfce7"),
         ("Logits ->\nsoftmax", "#fce7f3"), ("Sample next\ntoken: \" on\"", "#fef3c7")]
for i, (t, c) in enumerate(steps):
    box(ax, 0.1 + i * 1.95, 1, 1.6, 1.1, t, c, 9)
    if i < len(steps) - 1: arrow(ax, 1.72 + i * 1.95, 1.55, 2.03 + i * 1.95, 1.55)
ax.annotate("", xy=(0.9, 0.7), xytext=(11.0, 0.7), arrowprops=dict(arrowstyle="-|>", color="#b91c1c", lw=1.4, connectionstyle="arc3,rad=-0.0"))
ax.text(6, 0.35, "append token to the text and repeat (autoregression)", ha="center", color="#b91c1c", fontsize=10)
ax.set_title("How an LLM generates text", fontsize=13)
plt.savefig("illustrations/1_pipeline.png", dpi=130, bbox_inches="tight"); plt.close()

# 2. Transformer block ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5, 7)); ax.axis("off"); ax.set_xlim(0, 5); ax.set_ylim(0, 8)
box(ax, 1, 0.2, 3, 0.7, "Token + position embeddings", "#fef3c7", 9)
box(ax, 0.9, 1.5, 3.2, 1.3, "", "#f1f5f9"); ax.text(2.5, 2.65, "Transformer block (x N)", fontsize=9, ha="center", weight="bold")
box(ax, 1.2, 1.6, 1.2, 0.7, "Multi-head\nattention", "#dbeafe", 8); box(ax, 2.6, 1.6, 1.2, 0.7, "MLP", "#dcfce7", 8)
box(ax, 1, 3.4, 3, 0.7, "LayerNorm", "#e2e8f0", 9); box(ax, 1, 4.5, 3, 0.7, "Linear -> logits (vocab size)", "#fce7f3", 9)
box(ax, 1, 5.6, 3, 0.7, "Softmax -> probabilities", "#fce7f3", 9); box(ax, 1, 6.7, 3, 0.7, "Next-token prediction", "#fef3c7", 9)
for y1, y2 in [(0.9, 1.5), (2.8, 3.4), (4.1, 4.5), (5.2, 5.6), (6.3, 6.7)]: arrow(ax, 2.5, y1, 2.5, y2)
ax.text(0.3, 2.15, "x + ...\n(residual)", fontsize=7, color="#475569")
ax.set_title("GPT in one picture", fontsize=12)
plt.savefig("illustrations/2_transformer_block.png", dpi=130, bbox_inches="tight"); plt.close()

# 3. Temperature ------------------------------------------------------------------------
import numpy as np
words = ["Paris", "the", "a", "London", "banana"]; logits = np.array([3.0, 2.2, 1.0, 0.8, -2])
fig, axs = plt.subplots(1, 3, figsize=(11, 3), sharey=True)
for ax, T in zip(axs, (0.3, 1.0, 2.5)):
    p = np.exp(logits / T); p /= p.sum()
    ax.bar(words, p, color="#3b82f6"); ax.set_title(f"temperature = {T}"); ax.tick_params(axis="x", rotation=30)
axs[0].set_ylabel("probability")
fig.suptitle("Temperature: low = confident/repetitive, high = creative/random", y=1.03)
plt.tight_layout(); plt.savefig("illustrations/3_temperature.png", dpi=130, bbox_inches="tight"); plt.close()

# 4. Scale -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 3.2))
names = ["Your mini-GPT", "GPT-2 small", "GPT-3", "Frontier models\n(est. 100B-1T+)"]
params = [0.8, 124, 175000, 1000000]
ax.barh(names, params, color="#10b981"); ax.set_xscale("log"); ax.set_xlabel("parameters (millions, log scale)")
ax.set_title("Same recipe, different scale")
plt.tight_layout(); plt.savefig("illustrations/4_scale.png", dpi=130, bbox_inches="tight"); plt.close()

# 5. Training stages --------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 2.8)); ax.axis("off"); ax.set_xlim(0, 11); ax.set_ylim(0, 2.6)
st = [("1. Pretraining", "predict next token on\ntrillions of tokens of text\n-> knows language & facts\n(you did this in step 5)", "#dbeafe"),
      ("2. Supervised fine-tuning", "train on (question, ideal\nanswer) pairs\n-> follows instructions", "#dcfce7"),
      ("3. RLHF / preference tuning", "humans rank answers; model\nlearns to prefer good ones\n-> helpful, safe, chatty", "#fce7f3")]
for i, (t, d, c) in enumerate(st):
    box(ax, 0.2 + i * 3.7, 0.2, 3.3, 1.9, "", c); ax.text(1.85 + i * 3.7, 1.8, t, ha="center", weight="bold", fontsize=10)
    ax.text(1.85 + i * 3.7, 0.95, d, ha="center", va="center", fontsize=8.5)
    if i < 2: arrow(ax, 3.55 + i * 3.7, 1.15, 3.9 + i * 3.7, 1.15)
plt.savefig("illustrations/5_training_stages.png", dpi=130, bbox_inches="tight"); plt.close()
print("Saved 5 illustrations in illustrations/")
