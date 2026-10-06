# Character-Level Causal Transformer (GPT) from Scratch

This repository contains a PyTorch implementation of an autoregressive **Decoder-only Transformer language model** built entirely from scratch without using `torch.nn.Transformer` or other high-level Transformer abstractions.

The project demonstrates the core mechanics of **causal self-attention, Pre-Layer Normalization, residual connections, and autoregressive text generation**.

---

## Architectural Specifications

- **Model Type:** Causal Decoder-Only Transformer
- **Normalization:** Pre-Layer Normalization (Pre-LN)
- **Attention Mechanism:** Multi-Head Scaled Dot-Product Attention
- **Embedding Dimension (`N_embd`):** 384
- **Attention Heads (`N_head`):** 6
- **Head Dimension (`d_k`):** 64
- **Transformer Layers (`N_layer`):** 6
- **Context Length (`Block Size`):** 256 tokens
- **Vocabulary Size:** 65 (Character-level encoding)
- **Dropout:** 0.2

---

## Code Structure

```text
.
├── model.py
├── train.py
├── test.py
├── requirements.txt
└── README.md
```
###model.py

Contains the custom implementation of:

Head
MultiHeadAttention
FeedForward
Block
GPTLanguageModel
train.py

Handles:

Dataset loading
Character-level encoding
Batch extraction
Training loop
Validation/evaluation
Model optimization
test.py

Contains sanity and unit tests for:

Causal mask correctness
Tensor shape validation
Single-batch overfitting
Empirical Results

The model was trained on the Tiny Shakespeare dataset (~1.1M characters) using the AdamW optimizer with a learning rate of 3 × 10⁻⁴ for 2000 iterations on an NVIDIA RTX 4050 GPU.

Iteration	Training Loss	Validation Loss	Perplexity
0	4.2846	4.2820	72.38
500	1.8871	2.0022	7.41
1000	1.5322	1.7250	5.61
1500	1.3931	1.6021	4.96
2000	1.3079	1.5476	4.70
Sample Autoregressive Output

The model generates text autoregressively, predicting one character at a time based on the previous context.

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

Note: Since this is a character-level model trained from scratch on a relatively small dataset, imperfect spelling, grammar, and malformed words are expected.

Verification & Sanity Checks

To ensure numerical correctness and prevent implementation errors, the project includes several sanity checks.

Causal Mask Test

Ensures that the upper-triangular portion of the attention matrix is masked so that a token cannot attend to future positions.

Single-Batch Overfit

Tests whether the model has sufficient capacity and whether the training pipeline is functioning correctly by attempting to drive the training loss close to zero on a single mini-batch.

Shape Assertions

Validates tensor dimensions throughout the Transformer architecture, including:

Multi-head attention projections
Head reshaping
Attention outputs
Residual connections
LayerNorm blocks
Getting Started
Installation

Clone the repository and install the required dependencies:

pip install -r requirements.txt
Run Sanity Tests
python test.py
Run Training Pipeline
python train.py
Key Observations
Pre-LN Stability

Implementing Pre-Layer Normalization (Pre-LN) provided substantially better gradient flow and optimization stability compared to Post-LN during early iterations.

Efficient Causal Masking

Autoregressive attention masking using a lower-triangular matrix (torch.tril) enforces the causal constraint by preventing tokens from attending to future positions.

Character-Level Modeling

Character-level tokenization results in a very small vocabulary (65 tokens), allowing the complete language-modeling pipeline to be implemented and studied from first principles.

Autoregressive Generation

During generation, the model repeatedly predicts the next character using the previous Block Size tokens as context, demonstrating the fundamental autoregressive mechanism used by larger language models.

Requirements
Python 3.x
PyTorch
NVIDIA GPU with CUDA support (recommended)
CUDA-compatible PyTorch installation
Project Goal

The primary goal of this project is to understand and implement the fundamental components of a GPT-style Transformer language model from scratch, rather than relying on high-level Transformer APIs.

The implementation focuses on understanding:

Token and positional embeddings
Causal self-attention
Multi-head attention
Feed-forward networks
Residual connections
Layer normalization
Autoregressive language modeling
Training and text generation

This project serves as a foundation for further exploration of Large Language Models (LLMs), Transformer architectures, and generative AI systems.