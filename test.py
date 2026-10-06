import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from model import GPTLanguageModel, Head

def test_causal_mask():
    head = Head(head_size=16, n_embd=32, block_size=8, dropout=0.0)
    x = torch.randn(2, 8, 32)
    out = head(x)
    assert out.shape == (2, 8, 16)

def test_model_forward():
    model = GPTLanguageModel(
        vocab_size=65,
        n_embd=32,
        block_size=8,
        n_head=2,
        n_layer=2,
        dropout=0.0,
        device='cpu'
    )
    idx = torch.randint(0, 65, (2, 8))
    targets = torch.randint(0, 65, (2, 8))
    logits, loss = model(idx, targets)
    assert logits.shape == (16, 65)
    assert loss is not None

def test_overfit_single_batch():
    model = GPTLanguageModel(
        vocab_size=65,
        n_embd=32,
        block_size=8,
        n_head=2,
        n_layer=2,
        dropout=0.0,
        device='cpu'
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    x = torch.randint(0, 65, (1, 8))
    y = torch.randint(0, 65, (1, 8))
    
    initial_loss = None
    for i in range(50):
        logits, loss = model(x, y)
        if i == 0:
            initial_loss = loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    final_loss = loss.item()
    assert final_loss < initial_loss

if __name__ == '__main__':
    test_causal_mask()
    test_model_forward()
    test_overfit_single_batch()
    print("All unit tests and sanity checks passed.")