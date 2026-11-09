import torch, numpy, matplotlib, transformers
dev = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
print("torch", torch.__version__, "| device:", dev)
print("All good. Start with 1_foundations/01_tokenizer.py")
