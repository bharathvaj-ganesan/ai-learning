# Teach-back sheet (final 5 min)

Say each answer out loud in your own words. If you can, you're ready to explain LLMs.
Illustrations: `illustrations/1_pipeline.png`, `2_transformer_block.png`, `5_training_stages.png`.

## The 60-second explanation
> An LLM is a function that takes a list of tokens (chunks of text as integers) and outputs a
> probability for every possible next token. To generate text, you sample one, append it, and repeat.
> It's trained by showing it huge amounts of text and nudging its billions of numbers (weights) via
> gradient descent so it assigns higher probability to the real next token. Inside, a transformer lets
> every token look at earlier tokens via attention, then "think" via an MLP, stacked many layers deep.
> After pretraining, it's fine-tuned on instructions and human preferences so it behaves like an assistant.

## Self-quiz (answers are in the scripts you ran)
1. Why tokens instead of characters or words? (01)
2. What is the loss, and what does 'lower loss' mean? (02, 03)
3. What does `loss.backward()` do? What is the learning rate? (03)
4. What are Q, K, V? Why the causal mask? (04)
5. Attention vs MLP: what does each do? (05)
6. Why do we need position embeddings? (05)
7. What does temperature do? Why is output different each time? (06)
8. Why does the model "hallucinate"? (It samples plausible continuations; no built-in fact lookup/truth check.)
9. Why do bigger models do better? (More params + more data -> lower loss; scaling laws.)
10. What's the difference between a base model and a chat model? (post-training: SFT + RLHF)

## Common questions you'll get, with short answers
- **"Does it understand?"** It builds internal representations good enough to predict text well; whether that's understanding is debated. Practically: powerful pattern completion that generalizes.
- **"How does it learn facts?"** Facts that repeat in training data get baked into weights because they lower next-token loss.
- **"What's a context window?"** Max tokens (`block_size`) attended to at once. Beyond it, the model sees nothing.
- **"What's RAG?"** Paste retrieved documents into the prompt so facts come from context, not weights.
- **"What's fine-tuning vs prompting?"** Fine-tuning changes weights; prompting changes only the input.
- **"Why is inference costly?"** One forward pass per generated token; attention cost grows with context length (mitigated by KV cache).

## Bonus (if you have extra time)
- Watch Karpathy's "Let's build GPT" (2h) - it's exactly steps 2-5 with narration.
- Read `model.py` in karpathy/nanoGPT (300 lines, you'll understand all of it now).
