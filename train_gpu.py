"""Resumable single-GPU training of UbaNova-16 (used on a T4 in Colab and Kaggle). Needs train.py.

Usage: PIXEL_DIR=out EPOCHS=3 python train_gpu.py
Checkpoints (weights, optimizer, scaler, step) go to PIXEL_DIR every CKPT_S seconds and at
the end of every epoch. Running the script again resumes from the last checkpoint. When all
steps are done it writes model.pt, palette.npy and samples.png to PIXEL_DIR.
"""
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import train as T

PIXEL_DIR = os.environ.get("PIXEL_DIR", "out")
EPOCHS = int(os.environ.get("EPOCHS", 3))
BATCH = 128
LR, WARMUP = 3e-4, 1000
CKPT_S = 900  # Colab can disconnect at any time, so save often


def atomic_save(obj, path):
    torch.save(obj, path + ".tmp")
    os.replace(path + ".tmp", path)  # a disconnect mid-write must not corrupt the last good checkpoint


def main():
    dev = "cuda"
    os.makedirs(PIXEL_DIR, exist_ok=True)
    T.OUT = PIXEL_DIR
    T.prepare()  # download + tokenize captions into T.DATA (local disk, redone each session)
    tokens = np.load(f"{T.DATA}/tokens.npy")
    text_ids = np.load(f"{T.DATA}/text_ids.npy")  # last row is the empty caption
    split = np.load(f"{T.DATA}/split.npy")
    train_idx, val_idx = np.flatnonzero(split == 0), np.flatnonzero(split == 1)
    img_tr = torch.from_numpy(tokens[train_idx].astype(np.int16)).to(dev)
    txt_tr = torch.from_numpy(text_ids[train_idx]).to(dev)
    img_va = torch.from_numpy(tokens[val_idx].astype(np.int16)).to(dev)
    txt_va = torch.from_numpy(text_ids[val_idx]).to(dev)
    empty = torch.from_numpy(text_ids[-1:]).long().to(dev)
    del tokens

    _, clip = T.load_clip(dev)
    model = T.UbaNova().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=0.1)
    scaler = torch.amp.GradScaler()
    per_epoch = len(img_tr) // BATCH
    total = EPOCHS * per_epoch

    step = 0
    ckpt_path = f"{PIXEL_DIR}/ckpt.pt"
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=dev)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        scaler.load_state_dict(ck["scaler"])
        step = ck["step"]
        print(f"resumed at step {step}", flush=True)
    print(f"train {len(img_tr):,}  epochs {EPOCHS}  steps/epoch {per_epoch}  total {total:,}  start {step:,}", flush=True)

    def save_ckpt():
        atomic_save({"model": model.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(), "step": step}, ckpt_path)

    @torch.no_grad()
    def val_loss():
        tot = 0.0
        for i in range(0, len(img_va), BATCH):
            x, t = img_va[i : i + BATCH].long(), txt_va[i : i + BATCH].long()
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(clip(input_ids=t).last_hidden_state, x[:, :-1])
            tot += F.cross_entropy(logits.flatten(0, 1).float(), x.flatten(), reduction="sum").item()
        return tot / img_va.numel()

    drop_gen = torch.Generator(device=dev).manual_seed(step)
    last_ckpt = start = time.time()
    while step < total:
        epoch, j = divmod(step, per_epoch)
        perm = torch.randperm(len(img_tr), generator=torch.Generator().manual_seed(1234 + epoch)).to(dev)
        for j in range(j, per_epoch):
            b = perm[j * BATCH : (j + 1) * BATCH]
            x, t = img_tr[b].long(), txt_tr[b].long()
            t = torch.where((torch.rand(BATCH, device=dev, generator=drop_gen) < T.CFG_DROP)[:, None], empty, t)
            for g in opt.param_groups:
                g["lr"] = LR * min(1.0, (step + 1) / WARMUP) * 0.5 * (1 + math.cos(math.pi * step / total))
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(clip(input_ids=t).last_hidden_state, x[:, :-1])
            loss = F.cross_entropy(logits.flatten(0, 1).float(), x.flatten())
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            step += 1
            if step % 200 == 0:
                print(f"step {step}/{total}  epoch {step / per_epoch:.2f}  loss {loss.item():.4f}  {(time.time() - start) / 60:.0f} min", flush=True)
            if time.time() - last_ckpt > CKPT_S:
                save_ckpt()
                last_ckpt = time.time()
        print(f"epoch {epoch + 1} done  val_loss {val_loss():.4f}", flush=True)
        save_ckpt()
        last_ckpt = time.time()

    sd = {k: v.half() for k, v in model.state_dict().items()}
    atomic_save({"model": sd, "step": step}, f"{PIXEL_DIR}/model.pt")
    np.save(f"{PIXEL_DIR}/palette.npy", np.load(f"{T.DATA}/palette.npy"))
    model.eval()
    tokenizer, _ = T.load_clip("cpu")
    import json

    with open(f"{T.DATA}/captions.json", encoding="utf-8") as f:
        captions = json.load(f)
    prompts = [captions[i] for i in val_idx[:T.N_SAMPLES]]
    with torch.autocast("cuda", dtype=torch.float16):
        th = clip(input_ids=T.encode_text(tokenizer, prompts).long().to(dev)).last_hidden_state
        img = T.sample(model, th, clip(input_ids=empty).last_hidden_state)
    T.save_grid(img, np.load(f"{T.DATA}/palette.npy"), f"{PIXEL_DIR}/samples.png")
    print("training finished; model.pt written", flush=True)


if __name__ == "__main__":
    main()
