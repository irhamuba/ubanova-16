"""Check the KV-cached sampler in infer.py and pick default sampling settings.

1. The cached path must give the same logits as a full forward pass.
2. Sweep temperature / top_p at cfg 3 on 256 val captions; score with CLIP and pixel stats.
Writes sampling_report.json and sampling_compare.png to /kaggle/working.
"""
import glob
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

MODEL_DIR = os.path.dirname(glob.glob("/kaggle/input/**/model.pt", recursive=True)[0])
sys.path.insert(0, MODEL_DIR)
import infer  # noqa: E402
import train as T  # noqa: E402

OUT, dev = "/kaggle/working", "cuda"
R = {}
torch.manual_seed(0)
np.random.seed(0)

T.prepare()
tokens = np.load(f"{T.DATA}/tokens.npy")
text_ids = np.load(f"{T.DATA}/text_ids.npy")
val_idx = np.flatnonzero(np.load(f"{T.DATA}/split.npy") == 1)
palette = np.load(f"{T.DATA}/palette.npy")
with open(f"{T.DATA}/captions.json", encoding="utf-8") as f:
    captions = json.load(f)

tokenizer, clip = T.load_clip(dev)
model = T.UbaNova()
model.load_state_dict({k: v.float() for k, v in torch.load(f"{MODEL_DIR}/model.pt", map_location="cpu")["model"].items()})
model.to(dev).eval()


@torch.no_grad()
def text_h(ids):
    with torch.autocast("cuda", dtype=torch.float16):
        return clip(input_ids=torch.as_tensor(ids).long().to(dev)).last_hidden_state.float()


empty_h = text_h(text_ids[-1:])

# 1. cached logits == full forward logits (fp32, teacher forcing on real sprites)
with torch.no_grad():
    b = val_idx[:8]
    x = torch.from_numpy(tokens[b].astype(np.int64)).to(dev)
    th = text_h(text_ids[b])
    full = model(th, x[:, :-1]).float()
    first, kv = infer._prefill(model, th)
    cached = [first] + [infer._step(model, x[:, i - 1 : i], T.TEXT_LEN + i, kv) for i in range(1, 256)]
    diff = (torch.stack(cached, 1) - full).abs().max().item()
R["kv_cache_max_abs_logit_diff"] = diff
print("kv diff", diff, flush=True)
assert diff < 1e-2, diff

# 2. sweep
from transformers import CLIPModel  # noqa: E402

cm = CLIPModel.from_pretrained(T.CLIP, torch_dtype=torch.float16).to(dev).eval()
pal_t = torch.from_numpy(palette.astype(np.float32)).to(dev)
MEAN = torch.tensor([0.4815, 0.4578, 0.4082], device=dev)[:, None, None]
STD = torch.tensor([0.2686, 0.2613, 0.2758], device=dev)[:, None, None]


def _feat(o):
    return o if torch.is_tensor(o) else o.pooler_output


@torch.no_grad()
def clip_metrics(img, caps):
    idx = torch.as_tensor(img).long().to(dev).view(-1, 16, 16)
    a = (idx != 0).float()[..., None]
    x = (pal_t[idx] / 255.0 * a + 0.5 * (1 - a)).permute(0, 3, 1, 2)
    x = F.interpolate(x, size=224, mode="nearest")
    ie = F.normalize(_feat(cm.get_image_features(pixel_values=((x - MEAN) / STD).half())).float(), dim=-1)
    ids = tokenizer(caps, padding=True, truncation=True, max_length=77, return_tensors="pt").to(dev)
    te = F.normalize(_feat(cm.get_text_features(**ids)).float(), dim=-1)
    sim = ie @ te.T
    rank = (sim > sim.diag()[:, None]).sum(1)
    return {"clip_score": float(sim.diag().mean() * 100), "r_at_1": float((rank < 1).float().mean()),
            "r_at_5": float((rank < 5).float().mean())}


def pixel_stats(img):
    x = np.asarray(img).reshape(-1, 16, 16)
    op = x != 0
    p = np.pad(op, ((0, 0), (1, 1), (1, 1)))
    neigh = p[:, :-2, 1:-1] | p[:, 2:, 1:-1] | p[:, 1:-1, :-2] | p[:, 1:-1, 2:]
    isolated = (op & ~neigh).sum((1, 2)) / np.maximum(op.sum((1, 2)), 1)
    return {"colors_per_sprite": float(np.mean([len(np.unique(r[r != 0])) for r in x])),
            "isolated_px_frac": float(isolated.mean()), "opaque_frac": float(op.mean()),
            "border_opaque_frac": float((np.concatenate([x[:, 0], x[:, -1], x[:, :, 0], x[:, :, -1]], 1) != 0).mean())}


ev = val_idx[np.random.choice(len(val_idx), 256, replace=False)]
caps = [captions[i] for i in ev]
th = text_h(text_ids[ev])
R["real"] = {**clip_metrics(tokens[ev].astype(np.int64), caps), **pixel_stats(tokens[ev])}
grid = [tokens[ev][:16].astype(np.int64)]
settings = [(1.0, 1.0), (0.8, 1.0), (1.0, 0.9), (0.9, 0.9), (0.85, 0.9), (0.8, 0.9), (0.9, 0.8), (0.7, 1.0)]
for temp, top_p in settings:
    torch.manual_seed(1)
    t0 = time.time()
    with torch.autocast("cuda", dtype=torch.float16):
        g = infer.sample(model, th, empty_h, 3.0, temp, top_p).cpu().numpy()
    torch.cuda.synchronize()
    key = f"t{temp:g}_p{top_p:g}"
    R[key] = {**clip_metrics(g, caps), **pixel_stats(g), "sec_per_sprite_batch256": (time.time() - t0) / len(g)}
    print(key, R[key], flush=True)
    grid.append(g[:16])

# single-prompt latency (what the web UI does: n=4)
t0 = time.time()
with torch.autocast("cuda", dtype=torch.float16):
    infer.sample(model, th[:4], empty_h, 3.0, 0.85, 0.9)
torch.cuda.synchronize()
R["sec_for_4_sprites"] = time.time() - t0

T.save_grid(torch.from_numpy(np.concatenate(grid)), palette, f"{OUT}/sampling_compare.png", cols=16, scale=6)
with open(f"{OUT}/sampling_report.json", "w") as f:
    json.dump({"rows": ["real"] + [f"t{t:g}_p{p:g}" for t, p in settings], "captions": caps[:16], **R}, f, indent=1)
print("done", flush=True)
