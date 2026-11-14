"""Generate 16x16 sprites from text prompts using UbaNova.

Usage examples:
  python generate.py "gold double-ended spear"
  python generate.py "wooden golem" "emerald staff" --n 4 --out sprites.png
"""
import argparse
import sys
import time

from infer import SpriteGenerator


def parse_args():
    parser = argparse.ArgumentParser(description="Generate 16x16 sprites from text prompts.")
    parser.add_argument("prompts", nargs="+", help="One or more text prompts")
    parser.add_argument("--n", type=int, default=4, help="Number of sprites per prompt (default: 4)")
    parser.add_argument("--cfg", type=float, default=3.0, help="Classifier-free guidance scale (default: 3.0)")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature (default: 1.0)")
    parser.add_argument("--top-p", type=float, default=0.9, help="Nucleus sampling cutoff; 1.0 disables it (default: 0.9)")
    parser.add_argument("--model", default=None, help="Path to model.pt (default: weights/model.pt, else downloaded from Hugging Face)")
    parser.add_argument("--palette", default=None, help="Path to palette.npy (default: weights/palette.npy, else downloaded from Hugging Face)")
    parser.add_argument("--out", default="sprites.png", help="Output PNG filename (default: sprites.png)")
    parser.add_argument("--scale", type=int, default=8, help="Upscale factor for output PNG (default: 8)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--device", default=None, help="Device to use (cpu or cuda)")
    return parser.parse_args()


def main():
    args = parse_args()
    generator = SpriteGenerator(
        model_path=args.model,
        palette_path=args.palette,
        device=args.device,
    )

    all_images = []
    total_start = time.time()
    for prompt in args.prompts:
        start_t = time.time()
        print(f"Generating {args.n} sprite(s) for prompt: '{prompt}'...")
        imgs = generator.generate(
            prompt=prompt,
            n=args.n,
            cfg=args.cfg,
            temperature=args.temperature,
            top_p=args.top_p,
            seed=args.seed,
        )
        elapsed = time.time() - start_t
        print(f"  done in {elapsed:.2f}s ({elapsed / args.n:.2f}s per sprite)")
        all_images.extend(imgs)

    cols = args.n if len(args.prompts) == 1 else None
    grid = generator.create_grid(all_images, cols=cols, scale=args.scale)
    grid.save(args.out)

    total_elapsed = time.time() - total_start
    print(f"Saved {len(all_images)} sprite(s) to {args.out} (total time: {total_elapsed:.2f}s)")


if __name__ == "__main__":
    main()
