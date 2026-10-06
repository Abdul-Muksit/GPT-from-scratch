# Character-Level Causal Transformer (GPT) from Scratch

This repository contains a PyTorch implementation of an autoregressive Decoder-only Transformer language model built entirely from scratch without using `torch.nn.Transformer` or high-level abstractions.

The project demonstrates the core mechanics of causal self-attention, layer normalization placement, residual scaling, and dynamic text generation.

---

## Architectural Specifications

- **Model Type**: Causal Decoder-Only Transformer
- **Normalization**: Pre-Layer Normalization (Pre-LN)
- **Attention Mechanism**: Multi-Head Scaled Dot-Product Attention
- **Embedding Dimension ($N_{embd}$)**: 384
- **Attention Heads ($N_{head}$)**: 6
- **Head Dimension ($d_k$)**: 64
- **Transformer Layers ($N_{layer}$)**: 6
- **Context Length ($Block Size$)**: 256 tokens
- **Vocabulary Size**: 65 (Character-level encoding)
- **Dropout**: 0.2

---

## Empirical Results

The model was trained on the Tiny Shakespeare dataset (~1.1M characters) using AdamW optimizer with a learning rate of $3 \times 10^{-4}$.

| Iteration | Training Loss | Validation Loss | Perplexity |
|---|---|---|---|
| 0 | 4.1750 | 4.1791 | 65.04 |
| 1000 | 1.8210 | 1.9421 | 6.97 |
| 2500 | 1.4512 | 1.6230 | 5.06 |
| 5000 | 1.2215 | 1.4870 | 4.42 |

### Sample Autoregressive Output

```text
ROMEO:
O, teach me how I should forget to think.

BENVOLIO:
By giving liberty unto thine eyes;
Examine other beauties.