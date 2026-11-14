"""Evaluate the trained UbaNova (model.pt) on Pixel Art Atlas.

Runs as a Kaggle GPU script kernel with the model dataset attached. Writes
report.json plus image grids to /kaggle/working. Sections:
  data   - dataset statistics (captions, CLIP truncation, palette use, duplicates)
  nll    - val/train loss, per-position loss, text vs no-text, color error of mistakes
  gen    - samples at several CFG/temperature settings scored with CLIP and pixel stats
  color  - does "<color> <object>" produce that color
  memo   - nearest training sprite of each generated sprite (copying check)
"""
import colorsys
import glob
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

MODEL_DIR = os.path.dirname(glob.glob("/kaggle/input/**/model.pt", recursive=True)[0])
sys.path.insert(0, MODEL_DIR)
import train as T  # noqa: E402

OUT = "/kaggle/working"
dev = "cuda"
R = {}
torch.manual_seed(0)
np.random.seed(0)


def log(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------- data
T.prepare()
tokens = np.load(f"{T.DATA}/tokens.npy")
text_ids = np.load(f"{T.DATA}/text_ids.npy")
split = np.load(f"{T.DATA}/split.npy")
palette = np.load(f"{T.DATA}/palette.npy")
with open(f"{T.DATA}/captions.json", encoding="utf-8") as f:
    captions = json.load(f)
objects = None
if os.path.exists(f"{T.DATA}/objects.json"):
    with open(f"{T.DATA}/objects.json", encoding="utf-8") as f:
        objects = json.load(f)
train_idx, val_idx = np.flatnonzero(split == 0), np.flatnonzero(split == 1)

from transformers import CLIPTokenizerFast  # noqa: E402

_tok = CLIPTokenizerFast.from_pretrained(T.CLIP)
_sub = np.random.choice(len(captions), 20000, replace=False)
full_len = np.array([len(x) for x in _tok([captions[i] for i in _sub])["input_ids"]])
cap_len = np.array([len(c) for c in captions])
opaque = tokens != 0
n_colors = np.array([len(np.unique(r[r != 0])) for r in tokens[np.random.choice(len(tokens), 20000, replace=False)]])
freq = np.bincount(tokens.ravel(), minlength=T.VOCAB)
uniq_rows = len(np.unique(tokens.view(np.dtype((np.void, tokens.dtype.itemsize * 256)))))
R["data"] = {
    "n": len(tokens), "train": len(train_idx), "val": len(val_idx),
    "caption_chars": {"mean": float(cap_len.mean()), "p50": float(np.median(cap_len)), "max": int(cap_len.max())},
    "clip_tokens_untruncated": {"mean": float(full_len.mean()), "p95": float(np.percentile(full_len, 95)),
                                "max": int(full_len.max()), "truncated_frac_at_32": float((full_len > T.TEXT_LEN).mean())},
    "unique_captions": len(set(captions)), "unique_images": int(uniq_rows),
    "opaque_frac": float(opaque.mean()),
    "colors_per_sprite": {"mean": float(n_colors.mean()), "p5": float(np.percentile(n_colors, 5)), "p95": float(np.percentile(n_colors, 95))},
    "palette_used": int((freq[1:] > 0).sum()),
    "palette_top100_share": float(np.sort(freq[1:])[::-1][:100].sum() / freq[1:].sum()),
    "palette_rare_lt100": int(((freq[1:] > 0) & (freq[1:] < 100)).sum()),
    "caption_examples": [captions[i] for i in val_idx[:10]],
    "objects_type": type(objects).__name__,
    "objects_examples": (objects[:5] if isinstance(objects, list) else list(objects.items())[:5]) if objects else None,
}
log(json.dumps(R["data"], indent=1, default=str))

# ---------------------------------------------------------------- model
tokenizer, clip = T.load_clip(dev)
model = T.UbaNova()
model.load_state_dict({k: v.float() for k, v in torch.load(f"{MODEL_DIR}/model.pt", map_location="cpu")["model"].items()})
model.to(dev).eval()
R["model"] = {"params_M": sum(p.numel() for p in model.parameters()) / 1e6}
empty = torch.from_numpy(text_ids[-1:]).long().to(dev)
with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
    empty_h = clip(input_ids=empty).last_hidden_state


@torch.no_grad()
def text_h(ids):
    with torch.autocast("cuda", dtype=torch.float16):
        return clip(input_ids=torch.as_tensor(ids).long().to(dev)).last_hidden_state


# ---------------------------------------------------------------- nll
pal_t = torch.from_numpy(palette.astype(np.float32)).to(dev)


@torch.no_grad()
def nll_stats(idx, use_text=True):
    pos_loss = torch.zeros(256, device=dev)
    tot = acc = op_l = op_n = tr_l = tr_n = 0.0
    rgb_err = []
    for i in range(0, len(idx), 256):
        b = idx[i : i + 256]
        x = torch.from_numpy(tokens[b].astype(np.int64)).to(dev)
        th = text_h(text_ids[b]) if use_text else empty_h.expand(len(b), -1, -1)
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(th, x[:, :-1]).float()
        l = F.cross_entropy(logits.transpose(1, 2), x, reduction="none")  # B,256
        pos_loss += l.sum(0)
        tot += l.sum().item()
        pred = logits.argmax(-1)
        acc += (pred == x).sum().item()
        m = x != 0
        op_l += l[m].sum().item(); op_n += m.sum().item()
        tr_l += l[~m].sum().item(); tr_n += (~m).sum().item()
        wrong = (pred != x) & m & (pred != 0)
        rgb_err.append((pal_t[pred[wrong]] - pal_t[x[wrong]]).norm(dim=-1).cpu())
    n = len(idx) * 256
    rgb_err = torch.cat(rgb_err).numpy()
    return {
        "loss": tot / n, "bits_per_pixel": tot / n / math.log(2), "top1_acc": acc / n,
        "loss_opaque": op_l / op_n, "loss_transparent": tr_l / tr_n,
        "wrong_color_rgb_dist": {"p25": float(np.percentile(rgb_err, 25)), "p50": float(np.median(rgb_err)),
                                 "p75": float(np.percentile(rgb_err, 75)), "frac_lt_30": float((rgb_err < 30).mean())},
    }, (pos_loss / len(idx)).cpu().numpy()


t0 = time.time()
val_s, val_pos = nll_stats(val_idx)
val_u, _ = nll_stats(val_idx, use_text=False)
tr_sub = np.random.choice(train_idx, len(val_idx), replace=False)
tr_s, _ = nll_stats(tr_sub)
R["nll"] = {"val": val_s, "val_no_text": val_u, "train_subset": tr_s,
            "text_gain_bits_per_sprite": (val_u["loss"] - val_s["loss"]) * 256 / math.log(2),
            "per_row_loss": val_pos.reshape(16, 16).mean(1).round(3).tolist(),
            "per_pos_loss_16x16": val_pos.reshape(16, 16).round(2).tolist()}
log(json.dumps(R["nll"], indent=1), f"{time.time() - t0:.0f}s")

# ---------------------------------------------------------------- CLIP scorer
from transformers import CLIPModel  # noqa: E402

cm = CLIPModel.from_pretrained(T.CLIP, torch_dtype=torch.float16).to(dev).eval()
MEAN = torch.tensor([0.4815, 0.4578, 0.4082], device=dev)[:, None, None]
STD = torch.tensor([0.2686, 0.2613, 0.2758], device=dev)[:, None, None]


def to_rgba(img):
    idx = torch.as_tensor(img).long().to(dev).view(-1, 16, 16)
    rgb = pal_t[idx] / 255.0
    a = (idx != 0).float()[..., None]
    return rgb * a + 0.5 * (1 - a)  # gray background, B,16,16,3


def _feat(o):
    return o if torch.is_tensor(o) else o.pooler_output


@torch.no_grad()
def clip_embed_img(img):
    out = []
    for i in range(0, len(img), 256):
        x = to_rgba(img[i : i + 256]).permute(0, 3, 1, 2)
        x = F.interpolate(x, size=224, mode="nearest")
        out.append(F.normalize(_feat(cm.get_image_features(pixel_values=((x - MEAN) / STD).half())).float(), dim=-1))
    return torch.cat(out)


@torch.no_grad()
def clip_embed_txt(caps):
    ids = tokenizer(caps, padding=True, truncation=True, max_length=77, return_tensors="pt").to(dev)
    return F.normalize(_feat(cm.get_text_features(**ids)).float(), dim=-1)


def clip_metrics(img, caps):
    ie, te = clip_embed_img(img), clip_embed_txt(caps)
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
    ncol = np.array([len(np.unique(r[r != 0])) for r in x])
    border = np.concatenate([x[:, 0], x[:, -1], x[:, :, 0], x[:, :, -1]], 1)
    return {"opaque_frac": float(op.mean()), "isolated_px_frac": float(isolated.mean()),
            "noisy_sprites_frac": float((isolated > 0.1).mean()), "colors_per_sprite": float(ncol.mean()),
            "empty_sprites": int((op.sum((1, 2)) < 8).sum()), "border_opaque_frac": float((border != 0).mean())}


# ---------------------------------------------------------------- generation
N_EVAL = 256
ev = val_idx[np.random.choice(len(val_idx), N_EVAL, replace=False)]
ev_caps = [captions[i] for i in ev]
ev_th = text_h(text_ids[ev])
real = tokens[ev].astype(np.int64)
R["gen"] = {"real": {**clip_metrics(real, ev_caps), **pixel_stats(real)}}


def gen(th, cfg, temp, bs=128):
    out = []
    for i in range(0, len(th), bs):
        with torch.autocast("cuda", dtype=torch.float16):
            out.append(T.sample(model, th[i : i + bs], empty_h, cfg, temp).cpu())
    return torch.cat(out).numpy()


samples = {}
for cfg, temp in [(1.0, 1.0), (3.0, 1.0), (5.0, 1.0), (3.0, 0.8), (7.0, 1.0)]:
    t0 = time.time()
    g = gen(ev_th, cfg, temp)
    key = f"cfg{cfg:g}_t{temp:g}"
    samples[key] = g
    R["gen"][key] = {**clip_metrics(g, ev_caps), **pixel_stats(g), "sec_per_sprite": (time.time() - t0) / N_EVAL}
    log(key, R["gen"][key])

# grids: first 16 prompts, real row then each setting
order = ["real"] + list(samples)
rows = [real[:16]] + [samples[k][:16] for k in samples]
T.save_grid(torch.from_numpy(np.concatenate(rows)), palette, f"{OUT}/compare_settings.png", cols=16, scale=6)
with open(f"{OUT}/compare_settings.txt", "w") as f:
    f.write("rows: " + ", ".join(order) + "\n" + "\n".join(f"{i}: {c}" for i, c in enumerate(ev_caps[:16])))

# failure gallery at the default setting: lowest CLIP match
g = samples["cfg3_t1"]
ie, te = clip_embed_img(g), clip_embed_txt(ev_caps)
d = (ie * te).sum(1).cpu().numpy()
worst = np.argsort(d)[:16]
T.save_grid(torch.from_numpy(np.concatenate([g[worst], real[worst]])), palette, f"{OUT}/worst_cfg3.png", cols=16, scale=6)
R["gen"]["worst_cfg3_captions"] = [ev_caps[i] for i in worst]

# diversity: 4 samples for 64 prompts
th4 = ev_th[:64].repeat_interleave(4, 0)
g4 = gen(th4, 3.0, 1.0).reshape(64, 4, 256)
pair = [(g4[:, a] != g4[:, b]).mean() for a in range(4) for b in range(a + 1, 4)]
R["gen"]["diversity_hamming_same_prompt"] = float(np.mean(pair))
T.save_grid(torch.from_numpy(g4[:16].reshape(-1, 256)), palette, f"{OUT}/diversity.png", cols=16, scale=6)

# ---------------------------------------------------------------- color adherence
COLORS = {"red": 0, "orange": 30, "yellow": 55, "green": 120, "cyan": 185, "blue": 225, "purple": 275, "pink": 320}
OBJS = ["potion", "sword", "slime", "gem", "mushroom", "shield", "dragon", "flower"]
prompts = [f"{c} {o}" for c in COLORS for o in OBJS]
th = text_h(T.encode_text(tokenizer, [p for p in prompts for _ in range(4)]).numpy())
gc = gen(th, 3.0, 1.0)


def dominant_hue(r):
    px = palette[r[r != 0]] / 255.0
    if len(px) == 0:
        return None
    hsv = np.array([colorsys.rgb_to_hsv(*p) for p in px])
    w = hsv[:, 1] * hsv[:, 2]
    if w.sum() < 1e-6:
        return None
    ang = hsv[:, 0] * 2 * np.pi
    return (math.degrees(math.atan2((w * np.sin(ang)).sum(), (w * np.cos(ang)).sum())) + 360) % 360


names, hues = list(COLORS), np.array(list(COLORS.values()))
hit, per_color = 0, {c: 0 for c in COLORS}
for i, p in enumerate(prompts):
    c = p.split()[0]
    for r in gc[i * 4 : i * 4 + 4]:
        h = dominant_hue(r)
        if h is None:
            continue
        dist = np.minimum(abs(hues - h), 360 - abs(hues - h))
        if names[int(dist.argmin())] == c:
            hit += 1; per_color[c] += 1
R["color"] = {"acc": hit / len(gc), "per_color_acc": {c: v / (len(OBJS) * 4) for c, v in per_color.items()},
              "chance": 1 / len(COLORS)}
T.save_grid(torch.from_numpy(gc[::4]), palette, f"{OUT}/color_grid.png", cols=len(OBJS), scale=6)
log(R["color"])

# out-of-distribution / composition prompts
OOD = ["a red car", "a wooden house with a red roof", "a cat wearing a wizard hat", "a blue dragon breathing fire",
       "a golden crown with three rubies", "a skeleton holding a sword", "a cup of coffee", "a smiling sun",
       "a treasure chest full of gold coins", "an orange basketball", "a green frog with a crown", "a pixel heart"]
th = text_h(T.encode_text(tokenizer, [p for p in OOD for _ in range(4)]).numpy())
go = gen(th, 3.0, 1.0)
T.save_grid(torch.from_numpy(go), palette, f"{OUT}/ood_prompts.png", cols=8, scale=6)
R["ood_prompts"] = OOD
R["ood_clip"] = clip_metrics(go[::4], OOD)

# ---------------------------------------------------------------- memorization
tr_gpu = torch.from_numpy(tokens[train_idx].astype(np.int16)).to(dev)
q = torch.from_numpy(samples["cfg3_t1"].astype(np.int16)).to(dev)
best = torch.full((len(q),), 256, device=dev)
for i in range(0, len(tr_gpu), 20000):
    best = torch.minimum(best, (q[:, None] != tr_gpu[None, i : i + 20000]).sum(-1).min(1).values)
best = best.cpu().numpy() / 256
# reference: val sprites vs train (how close are genuinely new sprites)
qv = torch.from_numpy(real.astype(np.int16)).to(dev)
bv = torch.full((len(qv),), 256, device=dev)
for i in range(0, len(tr_gpu), 20000):
    bv = torch.minimum(bv, (qv[:, None] != tr_gpu[None, i : i + 20000]).sum(-1).min(1).values)
bv = bv.cpu().numpy() / 256
R["memo"] = {"gen_nn_hamming": {"p5": float(np.percentile(best, 5)), "p50": float(np.median(best)), "exact_copies": int((best == 0).sum()),
                                "near_copies_lt5pct": int((best < 0.05).sum())},
             "val_nn_hamming": {"p5": float(np.percentile(bv, 5)), "p50": float(np.median(bv)), "exact_copies": int((bv == 0).sum())}}
log(R["memo"])

with open(f"{OUT}/report.json", "w") as f:
    json.dump(R, f, indent=1, default=str)
log("eval done")
