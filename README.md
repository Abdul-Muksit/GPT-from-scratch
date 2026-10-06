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

The model was trained on the Tiny Shakespeare dataset (~1.1M characters) using AdamW optimizer with a learning rate of $3 \times 10^{-4}$ for 2000 iterations on an NVIDIA RTX 4050 GPU.

| Iteration | Training Loss | Validation Loss | Perplexity |
|---|---|---|---|
| 0 | 4.2846 | 4.2820 | 72.38 |
| 500 | 1.8871 | 2.0022 | 7.41 |
| 1000 | 1.5322 | 1.7250 | 5.61 |
| 1500 | 1.3931 | 1.6021 | 4.96 |
| 2000 | 1.3079 | 1.5476 | 4.70 |

### Sample Autoregressive Output

```text
Thirder'd his whose feither's easy,
Uford iit. You are hear you with old the worderth her
of than imcousined, prote, and and Henry of young
As the news and smits whose stand not:
Tubtle as days faquests my penger, and mercy,
Kneed world make in: 'fair, wifficed, but contempt,
The shame indank? but that we notidow my flates:
Ay, night doistay that hath but made is no gat
Of more orselvel ask'd those how's from ricked alement
Which we me whom.
Thousan, Lords, moon.

JULIET:
I tell despersuade: my