"""Train a character-level GPT on the Tiny Shakespeare dataset.

Usage:
    python train.py
    python train.py --max_iters 5000 --batch_size 32
"""

import argparse
import os
import urllib.error
import urllib.request

import torch

from model import GPTLanguageModel

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def parse_args():
    p = argparse.ArgumentParser(description="Train a character-level GPT")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--block_size", type=int, default=256)
    p.add_argument("--max_iters", type=int, default=2000)
    p.add_argument("--eval_interval", type=int, default=500)
    p.add_argument("--eval_iters", type=int, default=200)
    p.add_argument("--learning_rate", type=float, default=3e-4)
    p.add_argument("--n_embd", type=int, default=384)
    p.add_argument("--n_head", type=int, default=6)
    p.add_argument("--n_layer", type=int, default=6)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--grad_clip", type=float, default=1.0, help="max grad norm (0 disables)")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--data_path", type=str, default="input.txt")
    p.add_argument("--out_path", type=str, default="checkpoints/gpt_model.pt")
    p.add_argument("--sample_tokens", type=int, default=500)
    return p.parse_args()


def load_text(path):
    if not os.path.exists(path):
        print(f"{path} not found, downloading Tiny Shakespeare ...")
        try:
            urllib.request.urlretrieve(DATA_URL, path)
        except (urllib.error.URLError, OSError) as e:
            raise SystemExit(
                f"Could not download the dataset ({e}).\n"
                f"Download it manually from {DATA_URL} and save it as '{path}'."
            )
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    # ---- data ----
    text = load_text(args.data_path)
    chars = sorted(set(text))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    decode = lambda ids: "".join(itos[i] for i in ids)

    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]

    def get_batch(split):
        d = train_data if split == "train" else val_data
        ix = torch.randint(len(d) - args.block_size, (args.batch_size,))
        x = torch.stack([d[i : i + args.block_size] for i in ix])
        y = torch.stack([d[i + 1 : i + args.block_size + 1] for i in ix])
        return x.to(device), y.to(device)

    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split in ("train", "val"):
            losses = torch.zeros(args.eval_iters)
            for k in range(args.eval_iters):
                X, Y = get_batch(split)
                _, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean().item()
        model.train()
        return out

    # ---- model ----
    config = dict(
        vocab_size=vocab_size,
        n_embd=args.n_embd,
        block_size=args.block_size,
        n_head=args.n_head,
        n_layer=args.n_layer,
        dropout=args.dropout,
    )
    model = GPTLanguageModel(**config).to(device)
    print(f"device: {device} | parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    # ---- training loop ----
    model.train()
    for step in range(args.max_iters):
        if step % args.eval_interval == 0 or step == args.max_iters - 1:
            losses = estimate_loss()
            print(f"step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        xb, yb = get_batch("train")
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

    # ---- save everything needed to rebuild the model and decode its output ----
    out_dir = os.path.dirname(args.out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
            "stoi": stoi,
            "itos": itos,
        },
        args.out_path,
    )
    print(f"saved checkpoint to {args.out_path}")

    # ---- sample (generate() disables dropout and restores the training mode itself) ----
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decode(model.generate(context, max_new_tokens=args.sample_tokens)[0].tolist()))


if __name__ == "__main__":
    main()
