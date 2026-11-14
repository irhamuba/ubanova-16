"""Train a small text-to-sprite transformer on Pixel Art Atlas (16x16 palette tokens).

Runs as a Kaggle script kernel on a TPU VM (PyTorch/XLA, one process per core).
Training is EPOCHS long with one cosine schedule over all steps. A full checkpoint
(ckpt.pt: weights, optimizer, step) is written at the end of every epoch. A new session
resumes from ckpt.pt found in the output dir or in any attached input dataset, so a run
cut off by the session limit continues where it stopped. When all steps are done it
writes model.pt, palette.npy, samples.png. Saved weights are plain CPU tensors, so
generate.py runs them on CPU or GPU.
"""
import glob
import json
import math
import os
import time
import zipfile

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SMOKE = False  # True = tiny run that checks the pipeline end to end
OUT = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
DATA = "/tmp/pixel-atlas"
CLIP = "openai/clip-vit-base-patch32"
VOCAB = 4097  # palette entries; index 0 is transparent
TEXT_LEN = 32
D, LAYERS, HEADS = 512, 12, 8
BATCH = 128  # per core
LR, WARMUP = 6e-4, 1000
CFG_DROP = 0.1  # share of captions replaced by "" so the model learns the unconditional case
EPOCHS = 1 if SMOKE else 10
BUDGET_S = (8 if SMOKE else 510) * 60  # 8.5 h leaves margin under Kaggle's 9 h TPU limit
STOP_CHECK = 50  # steps between wall-clock checks (cores must agree on stopping)
N_SAMPLES = 32


class Block(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.heads = heads
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, T, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(y.transpose(1, 2).reshape(B, T, C))
        return x + self.mlp(self.ln2(x))


class UbaNova(nn.Module):
    """Sequence = [CLIP text states | BOS | 256 pixels]; predicts each pixel's palette index."""

    def __init__(self):
        super().__init__()
        self.text_proj = nn.Linear(512, D)
        self.tok = nn.Embedding(VOCAB + 1, D)  # last id is BOS
        self.pos = nn.Parameter(torch.randn(1, TEXT_LEN + 257, D) * 0.02)
        self.blocks = nn.ModuleList(Block(D, HEADS) for _ in range(LAYERS))
        self.ln_f = nn.LayerNorm(D)
        self.head = nn.Linear(D, VOCAB)

    def forward(self, text_h, prev):
        """prev: (B, t) pixels so far -> logits (B, t + 1, VOCAB)."""
        x = torch.cat([self.text_proj(text_h), self.tok(F.pad(prev, (1, 0), value=VOCAB))], 1)
        x = x + self.pos[:, : x.shape[1]]
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln_f(x[:, TEXT_LEN:]))


def load_clip(dev):
    from transformers import CLIPTextModel, CLIPTokenizerFast

    dtype = torch.float16 if dev == "cuda" else torch.float32
    clip = CLIPTextModel.from_pretrained(CLIP, torch_dtype=dtype).to(dev).eval().requires_grad_(False)
    return CLIPTokenizerFast.from_pretrained(CLIP), clip


def encode_text(tokenizer, captions):
    ids = tokenizer(captions, padding="max_length", max_length=TEXT_LEN, truncation=True, return_tensors="np")
    return torch.from_numpy(ids["input_ids"].astype(np.int32))


@torch.no_grad()
def sample(model, text_h, empty_h, cfg=3.0, temperature=1.0):
    """Autoregressive sampling with classifier-free guidance. Returns (B, 256) palette indices."""
    B = text_h.shape[0]
    th = torch.cat([text_h, empty_h.expand(B, -1, -1)])
    img = torch.zeros(B, 0, dtype=torch.long, device=text_h.device)
    for _ in range(256):
        cond, unc = model(th, torch.cat([img, img]))[:, -1].float().chunk(2)
        logits = unc + cfg * (cond - unc)
        img = torch.cat([img, torch.multinomial(F.softmax(logits / temperature, -1), 1)], 1)
    return img


def save_grid(img, palette, path, cols=8, scale=8):
    from PIL import Image

    idx = img.cpu().numpy().reshape(-1, 16, 16)
    rgba = np.concatenate([palette[idx], ((idx != 0) * 255).astype(np.uint8)[..., None]], -1)
    rows = math.ceil(len(rgba) / cols)
    grid = np.zeros((rows * 16, cols * 16, 4), np.uint8)
    for i, im in enumerate(rgba):
        r, c = divmod(i, cols)
        grid[r * 16 : (r + 1) * 16, c * 16 : (c + 1) * 16] = im
    Image.fromarray(grid).resize((cols * 16 * scale, rows * 16 * scale), Image.NEAREST).save(path)


def prepare():
    """Download, extract and tokenize once, before the TPU workers start."""
    from huggingface_hub import hf_hub_download

    if not os.path.exists(f"{DATA}/tokens.npy"):
        z = hf_hub_download("Angais/pixel-art-atlas-1.4m", "pixel-art-16x16-1.4m.zip", repo_type="dataset")
        zipfile.ZipFile(z).extractall(DATA)
    if not os.path.exists(f"{DATA}/text_ids.npy"):
        tokenizer, _ = load_clip("cpu")  # also puts the CLIP weights in the HF cache for the workers
        with open(f"{DATA}/captions.json", encoding="utf-8") as f:
            captions = json.load(f)
        np.save(f"{DATA}/text_ids.npy", encode_text(tokenizer, captions + [""]).numpy())
    print("data ready", flush=True)


def worker(index):
    import torch_xla.core.xla_model as xm
    import torch_xla.runtime as xr

    dev = xm.xla_device()
    rank, world = xr.global_ordinal(), xr.world_size()
    log = (lambda *a: print(*a, flush=True)) if rank == 0 else (lambda *a: None)

    tokens = np.load(f"{DATA}/tokens.npy", mmap_mode="r")
    text_ids = np.load(f"{DATA}/text_ids.npy", mmap_mode="r")  # last row is the empty caption
    empty = np.array(text_ids[-1])
    split = np.load(f"{DATA}/split.npy")
    train_idx, val_idx = np.flatnonzero(split == 0), np.flatnonzero(split == 1)
    if SMOKE:
        train_idx, val_idx = train_idx[:20000], val_idx[:512]
    log(f"train {len(train_idx):,}  val {len(val_idx):,}  cores {world}  global batch {BATCH * world}")

    torch.manual_seed(0)  # identical init on every core
    _, clip = load_clip(dev)
    model = UbaNova().to(dev)
    log(f"params {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=0.1)

    def to_dev(rows, drop=None):
        x = torch.from_numpy(tokens[rows].astype(np.int64)).to(dev)
        t = np.array(text_ids[rows])
        if drop is not None:
            t[drop] = empty
        return x, torch.from_numpy(t.astype(np.int64)).to(dev)

    gb = BATCH * world
    per_epoch = len(train_idx) // gb
    total = EPOCHS * per_epoch

    def batches(first):
        epoch, skip = divmod(first, per_epoch)
        while True:
            perm = np.random.default_rng(1234 + epoch).permutation(train_idx)  # same on every core
            for j in range(skip, per_epoch):
                yield np.sort(perm[j * gb + rank * BATCH : j * gb + (rank + 1) * BATCH])
            skip, epoch = 0, epoch + 1

    @torch.no_grad()
    def val_loss():
        tot, n = 0.0, 0
        mine = val_idx[rank::world]
        for i in range(0, len(mine), BATCH):
            x, t = to_dev(mine[i : i + BATCH])
            logits = model(clip(input_ids=t).last_hidden_state, x[:, :-1])
            tot += F.cross_entropy(logits.flatten(0, 1), x.flatten(), reduction="sum").item()
            n += x.numel()
        tot, n = xm.mesh_reduce("val", (tot, n), lambda v: (sum(a for a, _ in v), sum(b for _, b in v)))
        return tot / n

    def save_ckpt(step):
        xm.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step}, f"{OUT}/ckpt.pt")

    step = 0
    found = [p for p in [f"{OUT}/ckpt.pt"] + glob.glob("/kaggle/input/**/ckpt.pt", recursive=True) if os.path.exists(p)]
    if found:
        ck = torch.load(found[0], map_location="cpu")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        step = ck["step"]
        log(f"resumed from {found[0]} at step {step}")
    log(f"epochs {EPOCHS}  steps/epoch {per_epoch}  total {total:,}  starting at {step:,}")

    drop_rng = np.random.default_rng([rank, step])
    start = time.time()
    for rows in batches(step):
        if step >= total:
            break
        x, t = to_dev(rows, drop_rng.random(BATCH) < CFG_DROP)
        decay = 0.5 * (1 + math.cos(math.pi * step / total))
        for g in opt.param_groups:
            g["lr"] = LR * min(1.0, (step + 1) / WARMUP) * decay
        logits = model(clip(input_ids=t).last_hidden_state, x[:, :-1])
        loss = F.cross_entropy(logits.flatten(0, 1), x.flatten())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        xm.reduce_gradients(opt)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        xm.mark_step()
        step += 1

        if step % 200 == 0 or step == 20:
            log(f"step {step}/{total}  epoch {step / per_epoch:.2f}  loss {loss.item():.4f}  {(time.time() - start) / 60:.0f} min")
        if step % per_epoch == 0 or step == total:
            log(f"epoch end, step {step}  val_loss {val_loss():.4f}")
            save_ckpt(step)
        if step % STOP_CHECK == 0:
            out_of_time = xm.mesh_reduce("stop", time.time() - start > BUDGET_S, max)
            if out_of_time:
                log(f"time budget reached at step {step}; saving checkpoint")
                save_ckpt(step)
                break

    if step >= total:
        xm.save({"model": {k: v.half() for k, v in model.state_dict().items()}, "step": step}, f"{OUT}/model.pt")
        xm.rendezvous("saved")
        if rank == 0:
            write_samples(split)
    else:
        xm.rendezvous("saved")
        log(f"UNFINISHED at step {step}/{total}: attach this output as an input dataset and run again")


def write_samples(split):
    """Sample on CPU: growing sequence lengths would force a recompile per step on TPU."""
    sd = torch.load(f"{OUT}/model.pt", map_location="cpu")["model"]
    model = UbaNova()
    model.load_state_dict(sd)
    model.eval()
    tokenizer, clip = load_clip("cpu")
    with open(f"{DATA}/captions.json", encoding="utf-8") as f:
        captions = json.load(f)
    prompts = [captions[i] for i in np.flatnonzero(split == 1)[:N_SAMPLES]]
    with torch.no_grad():
        th = clip(input_ids=encode_text(tokenizer, prompts).long()).last_hidden_state
        img = sample(model, th, clip(input_ids=encode_text(tokenizer, [""]).long()).last_hidden_state)
    palette = np.load(f"{DATA}/palette.npy")
    np.save(f"{OUT}/palette.npy", palette)
    save_grid(img, palette, f"{OUT}/samples.png")
    with open(f"{OUT}/samples.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(prompts))
    print("samples written", flush=True)


if __name__ == "__main__":
    import torch_xla.distributed.xla_multiprocessing as xmp

    prepare()
    xmp.spawn(worker)
