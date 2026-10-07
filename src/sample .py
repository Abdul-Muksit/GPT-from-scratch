"""Generate text from a trained checkpoint.

Usage:
    python sample.py
    python sample.py --prompt "ROMEO:" --max_new_tokens 300 --temperature 0.8 --top_k 40
"""

import argparse

import torch

from model import GPTLanguageModel


def parse_args():
    p = argparse.ArgumentParser(description="Sample text from a trained GPT checkpoint")
    p.add_argument("--checkpoint", type=str, default="checkpoints/gpt_model.pt")
    p.add_argument("--prompt", type=str, default="", help="starting text (must use characters seen in training)")
    p.add_argument("--max_new_tokens", type=int, default=500)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_k", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.seed is not None:
        torch.manual_seed(args.seed)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model = GPTLanguageModel(**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    stoi, itos = ckpt["stoi"], ckpt["itos"]
    unknown = sorted({c for c in args.prompt if c not in stoi})
    if unknown:
        raise SystemExit(f"Prompt contains characters not in the training vocabulary: {unknown}")

    ids = [stoi[c] for c in args.prompt] or [0]  # empty prompt -> start from token id 0
    context = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(context, args.max_new_tokens, temperature=args.temperature, top_k=args.top_k)
    print("".join(itos[i] for i in out[0].tolist()))


if __name__ == "__main__":
    main()
