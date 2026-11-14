"""Inference engine for UbaNova text-to-pixel-art generation.
Supports CPU and CUDA execution, checkpoint loading, and image generation.
"""
import base64
import io
import math
import os
from typing import List, Optional

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

VOCAB = 4097
TEXT_LEN = 32
D, LAYERS, HEADS = 512, 12, 8
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"


class Block(nn.Module):
    def __init__(self, d: int, heads: int):
        super().__init__()
        self.heads = heads
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(b, t, 3, self.heads, c // self.heads).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(y.transpose(1, 2).reshape(b, t, c))
        return x + self.mlp(self.ln2(x))


class UbaNova(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_proj = nn.Linear(512, D)
        self.tok = nn.Embedding(VOCAB + 1, D)
        self.pos = nn.Parameter(torch.randn(1, TEXT_LEN + 257, D) * 0.02)
        self.blocks = nn.ModuleList(Block(D, HEADS) for _ in range(LAYERS))
        self.ln_f = nn.LayerNorm(D)
        self.head = nn.Linear(D, VOCAB)

    def forward(self, text_h: torch.Tensor, prev: torch.Tensor) -> torch.Tensor:
        x = torch.cat([self.text_proj(text_h), self.tok(F.pad(prev, (1, 0), value=VOCAB))], 1)
        x = x + self.pos[:, : x.shape[1]]
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln_f(x[:, TEXT_LEN:]))


DEFAULT_CFG = 3.0
DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_P = 0.9  # best setting in our sampling study: CLIP score, color count and stray pixels match real sprites


@torch.no_grad()
def _prefill(model, th):
    """Runs [text | BOS] once and returns the first pixel's logits plus each layer's keys/values."""
    bos = torch.full((th.shape[0], 1), VOCAB, dtype=torch.long, device=th.device)
    x = torch.cat([model.text_proj(th), model.tok(bos)], 1) + model.pos[:, : TEXT_LEN + 1]
    kv = []
    for block in model.blocks:
        b, t, c = x.shape
        q, k, v = block.qkv(block.ln1(x)).view(b, t, 3, block.heads, c // block.heads).permute(2, 0, 3, 1, 4)
        x = x + block.proj(F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(b, t, c))
        x = x + block.mlp(block.ln2(x))
        kv.append((k, v))
    return model.head(model.ln_f(x[:, -1])).float(), kv


@torch.no_grad()
def _step(model, token, pos, kv):
    """Feeds one pixel at sequence position pos, attending to the cached keys/values."""
    x = model.tok(token) + model.pos[:, pos : pos + 1]
    for i, block in enumerate(model.blocks):
        b, t, c = x.shape
        q, k, v = block.qkv(block.ln1(x)).view(b, t, 3, block.heads, c // block.heads).permute(2, 0, 3, 1, 4)
        k, v = torch.cat([kv[i][0], k], 2), torch.cat([kv[i][1], v], 2)
        kv[i] = (k, v)
        x = x + block.proj(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, t, c))
        x = x + block.mlp(block.ln2(x))
    return model.head(model.ln_f(x[:, 0])).float()


def _pick(logits, temperature, top_p):
    probs = F.softmax(logits / max(temperature, 1e-4), -1)
    if top_p < 1.0:  # nucleus: drop the rare palette entries that show up as stray colored pixels
        sp, si = probs.sort(-1, descending=True)
        sp[(sp.cumsum(-1) - sp) > top_p] = 0
        probs = torch.zeros_like(probs).scatter_(-1, si, sp)
    return torch.multinomial(probs, 1)


@torch.no_grad()
def sample(model, text_h, empty_h, cfg=DEFAULT_CFG, temperature=DEFAULT_TEMPERATURE, top_p=DEFAULT_TOP_P):
    """KV-cached sampling with classifier-free guidance. Returns (B, 256) palette indices."""
    b = text_h.shape[0]
    logits, kv = _prefill(model, torch.cat([text_h, empty_h.expand(b, -1, -1)]))
    out = []
    for i in range(256):
        if i:
            logits = _step(model, torch.cat([out[-1], out[-1]]), TEXT_LEN + i, kv)
        cond, unc = logits.chunk(2)
        out.append(_pick(unc + cfg * (cond - unc), temperature, top_p))
    return torch.cat(out, 1)


HF_REPO = "ubanova/UbaNova-16"
WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")


def _resolve(path: Optional[str], name: str) -> str:
    """Explicit path, else weights/<name> next to this file, else download from the Hugging Face Hub."""
    if path:
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        return path
    local = os.path.join(WEIGHTS_DIR, name)
    if os.path.exists(local):
        return local
    from huggingface_hub import hf_hub_download

    return hf_hub_download(HF_REPO, name)


class SpriteGenerator:
    def __init__(
        self,
        model_path: Optional[str] = None,
        palette_path: Optional[str] = None,
        device: Optional[str] = None,
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but no GPU is available; use device='cpu'")
        # GPU: FP16 autocast (about 10x faster); CPU: plain FP32
        self.use_fp16 = self.device.type == "cuda"
        self.model_path = _resolve(model_path, "model.pt")
        self.palette_path = _resolve(palette_path, "palette.npy")

        self.palette = np.load(self.palette_path)
        self.model = self._load_model(self.model_path)
        self.tokenizer, self.clip = self._load_clip()
        self.empty_h = None

    def _load_model(self, path: str) -> UbaNova:
        model = UbaNova()
        raw = torch.load(path, map_location="cpu", weights_only=True)
        sd = raw["model"] if isinstance(raw, dict) and "model" in raw else raw
        cleaned_sd = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
        model.load_state_dict(cleaned_sd)
        model.to(self.device).eval()
        return model

    def _load_clip(self):
        from transformers import CLIPTextModel, CLIPTokenizerFast
        dtype = torch.float16 if self.use_fp16 else torch.float32
        clip = CLIPTextModel.from_pretrained(CLIP_MODEL_NAME, torch_dtype=dtype)
        clip = clip.to(self.device).eval().requires_grad_(False)
        tokenizer = CLIPTokenizerFast.from_pretrained(CLIP_MODEL_NAME)
        return tokenizer, clip

    def _get_empty_embedding(self) -> torch.Tensor:
        if self.empty_h is None:
            ids = self.tokenizer([""], padding="max_length", max_length=TEXT_LEN, truncation=True, return_tensors="np")
            empty_ids = torch.from_numpy(ids["input_ids"].astype(np.int32)).long().to(self.device)
            with torch.no_grad():
                self.empty_h = self.clip(input_ids=empty_ids).last_hidden_state
        return self.empty_h

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        n: int = 4,
        cfg: float = DEFAULT_CFG,
        temperature: float = DEFAULT_TEMPERATURE,
        seed: Optional[int] = None,
        top_p: float = DEFAULT_TOP_P,
    ) -> List[Image.Image]:
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        prompts = [prompt] * n
        ids = self.tokenizer(prompts, padding="max_length", max_length=TEXT_LEN, truncation=True, return_tensors="np")
        input_ids = torch.from_numpy(ids["input_ids"].astype(np.int32)).long().to(self.device)
        text_h = self.clip(input_ids=input_ids).last_hidden_state
        with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.use_fp16):
            tokens = sample(self.model, text_h, self._get_empty_embedding(), cfg, temperature, top_p)

        token_indices = tokens.cpu().numpy().reshape(-1, 16, 16)
        images = []
        for idx in token_indices:
            rgb = self.palette[idx]
            alpha = ((idx != 0) * 255).astype(np.uint8)[..., None]
            rgba = np.concatenate([rgb, alpha], axis=-1)
            images.append(Image.fromarray(rgba, mode="RGBA"))
        return images

    @staticmethod
    def create_grid(images: List[Image.Image], cols: Optional[int] = None, scale: int = 8) -> Image.Image:
        count = len(images)
        if count == 0:
            return Image.new("RGBA", (16 * scale, 16 * scale), (0, 0, 0, 0))
        cols = cols or math.ceil(math.sqrt(count))
        rows = math.ceil(count / cols)
        grid = Image.new("RGBA", (cols * 16, rows * 16), (0, 0, 0, 0))
        for i, img in enumerate(images):
            r, c = divmod(i, cols)
            grid.paste(img, (c * 16, r * 16))
        return grid.resize((cols * 16 * scale, rows * 16 * scale), Image.NEAREST)

    @staticmethod
    def to_data_uri(image: Image.Image, scale: int = 1) -> str:
        if scale > 1:
            w, h = image.size
            image = image.resize((w * scale, h * scale), Image.NEAREST)
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        encoded = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/png;base64,{encoded}"
