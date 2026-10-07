import math

import pytest
import torch

from model import CausalSelfAttention, GPTLanguageModel

VOCAB = 65


def make_model(dropout=0.0, block_size=8):
    return GPTLanguageModel(
        vocab_size=VOCAB,
        n_embd=32,
        block_size=block_size,
        n_head=2,
        n_layer=2,
        dropout=dropout,
    )


def test_attention_output_shape():
    attn = CausalSelfAttention(n_embd=32, n_head=2, block_size=8, dropout=0.0)
    x = torch.randn(2, 8, 32)
    assert attn(x).shape == (2, 8, 32)


def test_attention_is_causal():
    """Changing future tokens must not change the attention output at earlier positions."""
    torch.manual_seed(0)
    attn = CausalSelfAttention(n_embd=32, n_head=2, block_size=8, dropout=0.0).eval()
    x = torch.randn(1, 8, 32)
    x2 = x.clone()
    x2[:, 5:] = torch.randn(1, 3, 32)  # modify positions 5, 6, 7 only
    out1, out2 = attn(x), attn(x2)
    assert torch.allclose(out1[:, :5], out2[:, :5], atol=1e-6)
    assert not torch.allclose(out1[:, 5:], out2[:, 5:])


def test_model_is_causal():
    """Full-model version: logits at positions < 5 are unaffected by tokens at positions >= 5."""
    torch.manual_seed(0)
    model = make_model().eval()
    x = torch.randint(0, VOCAB, (1, 8))
    x2 = x.clone()
    x2[0, 5:] = (x[0, 5:] + 1) % VOCAB
    logits1, _ = model(x)
    logits2, _ = model(x2)
    assert torch.allclose(logits1[:, :5], logits2[:, :5], atol=1e-6)
    assert not torch.allclose(logits1[:, 5:], logits2[:, 5:])


def test_forward_shapes():
    model = make_model()
    idx = torch.randint(0, VOCAB, (2, 8))
    targets = torch.randint(0, VOCAB, (2, 8))
    logits, loss = model(idx)
    assert logits.shape == (2, 8, VOCAB)
    assert loss is None
    logits, loss = model(idx, targets)
    assert logits.shape == (2, 8, VOCAB)  # logits keep (B, T, C) shape
    assert loss.ndim == 0


def test_initial_loss_is_near_uniform():
    """With small init, the first loss should be close to ln(vocab_size)."""
    torch.manual_seed(0)
    model = make_model().eval()
    idx = torch.randint(0, VOCAB, (16, 8))
    targets = torch.randint(0, VOCAB, (16, 8))
    _, loss = model(idx, targets)
    assert abs(loss.item() - math.log(VOCAB)) < 0.5


def test_rejects_sequence_longer_than_block_size():
    model = make_model(block_size=8)
    with pytest.raises(ValueError):
        model(torch.randint(0, VOCAB, (1, 9)))


def test_rejects_bad_head_count():
    with pytest.raises(ValueError):
        GPTLanguageModel(vocab_size=VOCAB, n_embd=30, block_size=8, n_head=4, n_layer=1, dropout=0.0)


def test_overfit_single_batch():
    torch.manual_seed(0)
    model = make_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    x = torch.randint(0, VOCAB, (1, 8))
    y = torch.randint(0, VOCAB, (1, 8))

    _, loss = model(x, y)
    initial_loss = loss.item()
    for _ in range(100):
        _, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    assert loss.item() < 0.5 * initial_loss


def test_generate_beyond_block_size():
    model = make_model(block_size=8)
    out = model.generate(torch.zeros((1, 1), dtype=torch.long), max_new_tokens=20)
    assert out.shape == (1, 21)
    assert out.min() >= 0 and out.max() < VOCAB


def test_generate_restores_training_mode_and_disables_grad():
    model = make_model(dropout=0.2)
    model.train()
    out = model.generate(torch.zeros((1, 1), dtype=torch.long), max_new_tokens=5)
    assert model.training  # mode restored
    assert not out.requires_grad

    model.eval()
    model.generate(torch.zeros((1, 1), dtype=torch.long), max_new_tokens=5)
    assert not model.training


def test_generate_top_k_one_is_deterministic():
    model = make_model().eval()
    start = torch.zeros((1, 1), dtype=torch.long)
    a = model.generate(start, max_new_tokens=10, top_k=1)
    b = model.generate(start, max_new_tokens=10, top_k=1)
    assert torch.equal(a, b)


def test_checkpoint_roundtrip(tmp_path):
    torch.manual_seed(0)
    config = dict(vocab_size=VOCAB, n_embd=32, block_size=8, n_head=2, n_layer=2, dropout=0.0)
    model = GPTLanguageModel(**config).eval()
    path = tmp_path / "ckpt.pt"
    torch.save({"model_state_dict": model.state_dict(), "config": config}, path)

    ckpt = torch.load(path, weights_only=True)
    restored = GPTLanguageModel(**ckpt["config"]).eval()
    restored.load_state_dict(ckpt["model_state_dict"])

    x = torch.randint(0, VOCAB, (2, 8))
    assert torch.allclose(model(x)[0], restored(x)[0])


def test_generate_rejects_non_positive_temperature():
    model = make_model().eval()
    with pytest.raises(ValueError):
        model.generate(torch.zeros((1, 1), dtype=torch.long), max_new_tokens=1, temperature=0.0)
