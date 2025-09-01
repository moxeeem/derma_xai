# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import math
import json
import random
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import Counter

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import sacrebleu
from rouge_score import rouge_scorer

import open_clip
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    CLIPImageProcessor as HFCLIPImageProcessor, 
    CLIPModel as HFCLIPModel, 
    CLIPTokenizer
)

try:
    from transformers.pytorch_utils import Conv1D as HFConv1D
except Exception:
    try:
        from transformers.modeling_utils import Conv1D as HFConv1D
    except Exception:
        HFConv1D = None

APPROACH_NAME = 'dermlip_vitb16' # 'dermlip_vitb16' или 'biomedclip_pubmedbert'


@dataclass
class InferenceCfg:
    CSV_PATH: str = '/root/clip_xai/dermaCAP/dermaCAP_v1_clean.csv'
    CKPT_PATHS: List[str] = None  # Список путей к чекпоинтам для мультимодельного инференса
    OUT_DIR: str = f'/root/clip_xai/final/infer_results_multi'
    
    SOURCE_FILTER: Optional[str] = None  # например 'skincap' или None
    SAMPLES: int = 128                    # 0 или отрицательное число - использовать весь датасет
    SEED: int = 42
    IMG_BATCH: int = 32
    PANEL_H: int = 780
    
    USE_AMP: bool = True

cfg = InferenceCfg()

# Если не задан список чекпоинтов, используем старую логику
if cfg.CKPT_PATHS is None:
    cfg.CKPT_PATHS = [f'/root/clip_xai/final/final_biomedclip_pubmedbert/final_captioner_gpt2-medium_TimmModel.pt']

# Можно раскомментировать для мультимодельного инференса:
# cfg.CKPT_PATHS = [
#     '/root/clip_xai/final/final_dermlip/final_captioner_gpt2-medium_VisionTransformer.pt',
#     '/root/clip_xai/final/final_pubmed_clip_b32/final_captioner_gpt2-medium_CLIPModel.pt', 
#     '/root/clip_xai/final/final_biomed/final_captioner_gpt2-medium_BiomedCLIP.pt',
# ]

CSV_PATH = cfg.CSV_PATH
CKPT_PATHS = cfg.CKPT_PATHS
OUT_DIR = cfg.OUT_DIR
SOURCE_FILTER = cfg.SOURCE_FILTER
SAMPLES = cfg.SAMPLES
SEED = cfg.SEED
IMG_BATCH = cfg.IMG_BATCH
PANEL_H = cfg.PANEL_H
USE_AMP = cfg.USE_AMP

HF_CLIP_PRESETS = {
    'dermlip_vitb16': {
        'repo': 'redlessone/DermLIP_ViT-B-16',
        'kind': 'open_clip',
    },
    'pubmed_clip_b32': {
        'repo': 'flaviagiammarino/pubmed-clip-vit-base-patch32',
        'kind': 'hf_clip',
    },
    'biomedclip_pubmedbert': {
        'repo': 'microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224',
        'kind': 'open_clip',
    },
}
PREF_TEXT_COLS = ['caption_final', 'caption']


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def pick_text_col(df: pd.DataFrame, fallback: str = 'caption') -> str:
    for c in PREF_TEXT_COLS:
        if c in df.columns: return c
    if fallback in df.columns: return fallback
    raise ValueError('Не найдена текстовая колонка (caption/caption_final)')


def sent_bleu(ref: str, hyp: str) -> float:
    if not sacrebleu:
        ref_tokens = ref.split(); hyp_tokens = hyp.split()
        if not hyp_tokens: return 0.0
        inter = sum((min(hyp_tokens.count(t), ref_tokens.count(t)) for t in set(hyp_tokens)))
        prec = inter / max(1, len(hyp_tokens))
        bp = math.exp(1 - len(ref_tokens) / len(hyp_tokens)) if len(hyp_tokens) < len(ref_tokens) else 1.0
        return 100.0 * prec * bp
    return float(sacrebleu.sentence_bleu(hyp, [ref]).score)


def rouge_l(ref: str, hyp: str) -> float:
    if not rouge_scorer: return 0.0
    rs = rouge_scorer.RougeScorer(['rougeLsum'], use_stemmer=True)
    return float(rs.score(ref, hyp)['rougeLsum'].fmeasure)


def _tok(s: str) -> List[str]:
    s = re.sub(r'[^a-z0-9 ]', ' ', str(s).lower())
    return [t for t in s.split() if t]


def _ngrams(tokens: List[str], n: int) -> Counter:
    return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)) if len(tokens) >= n else Counter()


def build_cider_idf(refs: List[str], nmax: int = 4) -> List[Dict[Tuple[str,...], float]]:
    N = max(1, len(refs))
    dfs = [Counter() for _ in range(nmax)]
    for r in refs:
        toks = _tok(r)
        for n in range(1, nmax+1):
            dfs[n-1].update(set(_ngrams(toks, n).keys()))
    idf = []
    for n in range(nmax):
        idf.append({g: math.log((N + 1.0) / (df + 1.0)) for g, df in dfs[n].items()})
    return idf


def _tfidf_vec(tokens: List[str], idf: List[Dict[Tuple[str,...], float]], 
               nmax: int = 4) -> List[Dict[Tuple[str,...], float]]:
    vecs = []
    for n in range(1, nmax+1):
        tf = _ngrams(tokens, n)
        L = sum(tf.values()) or 1
        v = {g: (c / L) * idf[n-1].get(g, 0.0) for g, c in tf.items()}
        vecs.append(v)
    return vecs


def ciderD_pair(hyp: str, ref: str, idf: List[Dict[Tuple[str,...], float]], 
                nmax: int = 4, sigma: float = 6.0) -> float:
    th, tr = _tok(hyp), _tok(ref)
    vh, vr = _tfidf_vec(th, idf, nmax), _tfidf_vec(tr, idf, nmax)
    score = 0.0
    for n in range(nmax):
        dot = 0.0; nh = 0.0; nr = 0.0
        for g, w in vh[n].items():
            nh += w*w
            if g in vr[n]: dot += w * vr[n][g]
        for w in vr[n].values(): nr += w*w
        sim = (dot / (math.sqrt(nh*nr) + 1e-8)) if nh > 0 and nr > 0 else 0.0
        score += sim
    score /= float(nmax)
    delta = math.exp(- (len(th) - len(tr))**2 / (2.0 * sigma * sigma))
    return 10.0 * score * delta


class CLIPBackend:
    def __init__(self, cfg: dict, device: str):
        preset = cfg.get('CLIP_PRESET', 'dermlip_vitb16')
        meta = HF_CLIP_PRESETS.get(preset, HF_CLIP_PRESETS['dermlip_vitb16'])
        self.kind = meta['kind']
        self.repo = meta['repo']
        self.device = device
        self.arch = None
        self.embed_dim = None

        if self.kind == 'open_clip':
            if 'BiomedCLIP' in self.repo:
                # Специальная загрузка для BiomedCLIP
                model_name = f'hf-hub:{self.repo}'
                self.model, self.preprocess, _ = open_clip.create_model_and_transforms(model_name)
            else:
                model_name = f'hf-hub:{self.repo}'
                self.model, self.preprocess, _ = open_clip.create_model_and_transforms(model_name)
            self.model = self.model.to(device).eval()
            self.processor = None
        else:
            # HF CLIP
            self.model = HFCLIPModel.from_pretrained(self.repo).to(device).eval()
            self.processor = HFCLIPImageProcessor.from_pretrained(self.repo)
            self.preprocess = None


    @torch.inference_mode()
    def encode_images(self, paths: List[str]) -> torch.Tensor:
        ims = []
        for p in paths:
            try:
                img = Image.open(p).convert('RGB')
            except Exception:
                img = Image.new('RGB', (224, 224), color=0)
            
            if self.kind == 'open_clip':
                ims.append(self.preprocess(img))
            else:
                ims.append(img)

        if self.kind == 'open_clip':
            x = torch.stack(ims).to(self.device)
            f = self.model.encode_image(x)
        elif 'BiomedCLIP' in self.repo:
            # BiomedCLIP также использует open_clip API
            x = torch.stack(ims).to(self.device)
            f = self.model.encode_image(x)
        else:
            proc = self.processor(images=ims, return_tensors='pt')
            x = proc['pixel_values'].to(self.device)
            f = self.model.get_image_features(pixel_values=x)
            
        return F.normalize(f, dim=-1)


    @torch.inference_mode()
    def clipscore(self, paths: List[str], texts: List[str]) -> List[float]:
        img_f = self.encode_images(paths)
        
        if self.kind == 'open_clip':
            if 'BiomedCLIP' in self.repo:
                # BiomedCLIP использует токенизатор из модели
                tok = open_clip.get_tokenizer(f'hf-hub:{self.repo}')
            else:
                tok = open_clip.get_tokenizer(f'hf-hub:{self.repo}')
            txt_tokens = tok(texts).to(self.device)
            with torch.amp.autocast('cuda', enabled=USE_AMP and img_f.is_cuda):
                txt_f = self.model.encode_text(txt_tokens)
        else:
            # HF CLIP
            tokenizer = CLIPTokenizer.from_pretrained(self.repo)
            txt_tokens = tokenizer(texts, return_tensors='pt', padding=True, 
                                   truncation=True, max_length=77).to(self.device)
            with torch.amp.autocast('cuda', enabled=USE_AMP and img_f.is_cuda):
                txt_f = self.model.get_text_features(**txt_tokens)
                
        txt_f = F.normalize(txt_f, dim=-1)
        scores = (img_f * txt_f).sum(dim=-1)
        return (scores * 100.0).detach().cpu().tolist()


class PrefixProjector(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, tokens: int, p_drop: float = 0.05):
        super().__init__()
        hidden = max(512, out_dim * 2)
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, out_dim * tokens)
        self.ln  = nn.LayerNorm(out_dim)
        self.tokens = tokens
        self.drop = nn.Dropout(p_drop)
        self.alpha = nn.Parameter(torch.tensor(0.5))
        nn.init.xavier_uniform_(self.fc1.weight, gain=1.0)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight, gain=0.5)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.gelu(self.fc1(x))
        y = self.fc2(y).view(y.size(0), self.tokens, -1)
        y = self.ln(y)
        return self.drop(self.alpha * y)


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16, dropout: float = 0.05):
        super().__init__()
        self.base = base
        for p in self.base.parameters(): p.requires_grad = False
        in_f, out_f = base.in_features, base.out_features
        self.r = r; self.scale = alpha / float(r)
        self.A = nn.Linear(in_f, r, bias=False)
        self.B = nn.Linear(r, out_f, bias=False)
        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B.weight)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.B(self.drop(self.A(x))) * self.scale


class LoRAConv1D(nn.Module):
    def __init__(self, base, r: int = 16, alpha: int = 16, dropout: float = 0.05):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        in_f = base.weight.shape[0]
        out_f = base.weight.shape[1]
        self.r = r
        self.scale = alpha / float(r)
        self.A = nn.Linear(in_f, r, bias=False)
        self.B = nn.Linear(r, out_f, bias=False)
        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B.weight)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, T, in_f]
        return self.base(x) + self.B(self.drop(self.A(x))) * self.scale


def _wrap_module_with_lora(module: nn.Module, name: str, r: int, alpha: int, 
                           dropout: float):
    if not hasattr(module, name):
        return
    base = getattr(module, name)
    if HFConv1D is not None and isinstance(base, HFConv1D):
        lora = LoRAConv1D(base, r=r, alpha=alpha, dropout=dropout)
    elif isinstance(base, nn.Linear):
        lora = LoRALinear(base, r=r, alpha=alpha, dropout=dropout)
    else:
        return
    lora.to(device=base.weight.device, dtype=base.weight.dtype)
    setattr(module, name, lora)


def inject_gpt2_lora(gpt2: nn.Module, last_k: int, r: int = 16, 
                     alpha: int = 16, dropout: float = 0.05):
    if last_k <= 0:
        return
    blocks = gpt2.transformer.h
    for layer in blocks[-last_k:]:
        _wrap_module_with_lora(layer.attn, 'c_attn', r, alpha, dropout)
        _wrap_module_with_lora(layer.attn, 'c_proj', r, alpha, dropout)
        _wrap_module_with_lora(layer.mlp,  'c_fc',   r, alpha, dropout)
        _wrap_module_with_lora(layer.mlp,  'c_proj', r, alpha, dropout)


class DermCaptioner(nn.Module):
    def __init__(self, cfg: dict, device: str):
        super().__init__()
        self.cfg = cfg
        self.device = device

        gpt2_name = cfg.get('GPT2_NAME', 'gpt2-medium')
        self.tok = AutoTokenizer.from_pretrained(gpt2_name)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.gpt2 = AutoModelForCausalLM.from_pretrained(gpt2_name).to(device)
        try: self.gpt2.gradient_checkpointing_enable()
        except Exception: pass

        kA = int(cfg.get('unfreeze_last_k_layers_A', 0))
        kB = int(cfg.get('unfreeze_last_k_layers_B', 0))
        self._lora_k = max(kA, kB)
        inject_gpt2_lora(self.gpt2, last_k=self._lora_k, r=16, alpha=16, dropout=0.05)

        self.clip_backend = CLIPBackend(cfg, device)
        self.img_dim = int(self.gpt2.config.n_embd) 
        self.txt_dim = int(self.gpt2.config.n_embd)
        
        if 'embed_dim' in cfg:
            in_dim = int(cfg['embed_dim'])
        else:
            if self.clip_backend.kind == 'open_clip':
                dummy_img = Image.new('RGB', (224, 224), color=0)
                dummy_tensor = self.clip_backend.preprocess(dummy_img).unsqueeze(0).to(device)
                with torch.no_grad():
                    dummy_feat = self.clip_backend.model.encode_image(dummy_tensor)
                in_dim = int(dummy_feat.shape[-1])
            else:
                in_dim = int(self.clip_backend.model.config.projection_dim)
        
        self.prefix = PrefixProjector(
            in_dim=in_dim,
            out_dim=self.txt_dim,
            tokens=int(cfg.get('prefix_tokens', 32)),
            p_drop=float(cfg.get('prefix_dropout_p', 0.05)),
        ).to(device)

        self.prompt = cfg.get('prompt_B') or cfg.get('prompt') or ''
        self.cliche_phrases = tuple(cfg.get('cliche_phrases', []))

        self.gen_max_new = int(min(cfg.get('gen_max_new', 80), 80))
        self.gen_beams   = int(max(1, cfg.get('gen_beams', 4)))
        self.no_repeat_ngram_size = int(cfg.get('no_repeat_ngram_size', 4))
        self.repetition_penalty   = float(cfg.get('repetition_penalty', 1.15))
        self.length_penalty       = float(cfg.get('length_penalty', 0.6))
        self.min_new_tokens       = int(cfg.get('min_new_tokens', 24))

    @torch.inference_mode()
    def generate(self, img_paths: List[str]) -> List[str]:
        c = self
        img_feat = self.clip_backend.encode_images(img_paths).detach().clone()
        pref = self.prefix(img_feat)

        prompt_ids = self.tok([c.prompt] * pref.size(0), return_tensors='pt', 
                              padding=True, truncation=True).to(self.device)
        prompt_emb = self.gpt2.transformer.wte(prompt_ids['input_ids'])
        inputs_embeds = torch.cat([pref, prompt_emb], dim=1)
        attn = torch.ones(inputs_embeds.size()[:-1], dtype=torch.long, device=self.device)

        bad_seqs = None
        if c.cliche_phrases:
            seqs = []
            for phrase in c.cliche_phrases:
                ids = self.tok.encode(str(phrase).strip(), add_special_tokens=False)
                if ids: seqs.append(ids)
            if seqs: bad_seqs = seqs

        gen = self.gpt2.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attn,
            max_new_tokens=c.gen_max_new,
            min_new_tokens=c.min_new_tokens,
            num_beams=c.gen_beams,
            do_sample=(c.gen_beams == 1),
            no_repeat_ngram_size=c.no_repeat_ngram_size,
            repetition_penalty=c.repetition_penalty,
            length_penalty=c.length_penalty,
            bad_words_ids=bad_seqs,
            pad_token_id=self.tok.eos_token_id,
            eos_token_id=self.tok.eos_token_id,
            early_stopping=True,
        )
        outs = self.tok.batch_decode(gen, skip_special_tokens=True)
        res = []
        for s in outs:
            cut = s.find(c.prompt)
            if cut >= 0: 
                s = s[cut + len(c.prompt):]
            res.append(' '.join(s.strip().split()))
        return res


def wrap_multiline(text: str, font: ImageFont.FreeTypeFont, max_width: int, 
                   draw: ImageDraw.ImageDraw) -> str:
    lines: List[str] = []
    for para in str(text).split('\n'):
        words = para.split(' ')
        cur = ''
        for w in words:
            test = (cur + ' ' + w).strip()
            if draw.textlength(test, font=font) <= max_width:
                cur = test
            else:
                if cur: 
                    lines.append(cur)
                cur = w
        if cur: 
            lines.append(cur)
    return '\n'.join(lines) if lines else ''


def make_card_multi_model(pil_img: Image.Image,
                         gt: str, 
                         model_preds: List[Tuple[str, str, Dict[str, float]]],  # [(model_name, pred, metrics)]
                         out_path: Path,
                         panel_h: int = PANEL_H,
                         font_cache: Dict[str, ImageFont.FreeTypeFont] | None = None) -> None:
    img = pil_img
    h = panel_h
    w_img = h
    w_text = h  # ширина для каждой модели
    
    ratio = img.width / max(1, img.height)
    new_w = int(h * ratio)
    img_resized = img.resize((new_w, h), Image.BICUBIC)

    # Левая панель с изображением и ground truth
    left = Image.new('RGB', (w_img, h), (255, 255, 255))
    left.paste(img_resized, ((w_img - new_w) // 2 if new_w <= w_img else 0, 0))

    if font_cache is None: 
        font_cache = {}
    try:
        font_title  = ImageFont.truetype('DejaVuSans-Bold.ttf', 22)
        font_text   = ImageFont.truetype('DejaVuSans.ttf', 20)
        font_metrics= ImageFont.truetype('DejaVuSansMono.ttf', 18)
        font_cache.update({'title': font_title, 'text': font_text, 'mono': font_metrics})
    except Exception:
        font_title = font_text = font_metrics = ImageFont.load_default()

    # Добавим ground truth на левую панель
    draw_left = ImageDraw.Draw(left)
    pad = 16
    max_text_w = w_img - 2 * pad
    
    # GT внизу левой панели
    gt_y = h - 180  # оставляем место для GT
    draw_left.text((pad, gt_y), 'Ground Truth', fill=(0, 0, 0), font=font_title)
    gt_y += int(font_title.size * 1.2)
    gt_w = wrap_multiline(gt or '', font_text, max_text_w, draw_left)
    draw_left.multiline_text((pad, gt_y), gt_w, fill=(20, 20, 20), font=font_text, spacing=4)

    # Создаем панели для моделей
    total_width = w_img + len(model_preds) * w_text
    canvas = Image.new('RGB', (total_width, h), (255, 255, 255))
    canvas.paste(left, (0, 0))

    for i, (model_name, pred, metrics) in enumerate(model_preds):
        x_offset = w_img + i * w_text
        panel = Image.new('RGB', (w_text, h), (240 + i * 5, 240 + i * 5, 240 + i * 5))
        draw = ImageDraw.Draw(panel)
        
        x = pad
        y = pad
        max_text_w = w_text - 2 * pad

        # Название модели
        draw.text((x, y), f'Model: {model_name}', fill=(0, 0, 0), font=font_title)
        y += int(font_title.size * 1.4)
        
        # Предсказание модели
        pred_w = wrap_multiline(pred or '', font_text, max_text_w, draw)
        draw.multiline_text((x, y), pred_w, fill=(10, 10, 10), font=font_text, spacing=4)
        
        # Определяем где начинать метрики
        pred_bbox = draw.multiline_textbbox((x, y), pred_w, font=font_text, spacing=4)
        y = pred_bbox[3] + int(font_title.size * 0.8)
        
        # Метрики
        draw.text((x, y), 'Metrics', fill=(0, 0, 0), font=font_title)
        y += int(font_title.size * 1.2)
        
        col = "\n".join([
            f"CLIP: {metrics.get('clip', 0):.1f}",
            f"BLEU: {metrics.get('bleu', 0):.1f}",
            f"ROUGE: {metrics.get('rougeL', 0):.3f}",
            f"CIDEr: {metrics.get('cider', 0):.2f}",
        ])
        draw.multiline_text((x, y), col, fill=(0, 0, 0), font=font_metrics, spacing=6)
        
        canvas.paste(panel, (x_offset, 0))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=95)


def make_card_pair(pil_img: Image.Image,
                   gt: str, model_pred: str,
                   metrics_model: Dict[str, float],
                   out_path: Path,
                   panel_h: int = PANEL_H,
                   font_cache: Dict[str, ImageFont.FreeTypeFont] | None = None) -> None:
    img = pil_img
    h = panel_h
    w_panel = h
    ratio = img.width / max(1, img.height)
    new_w = int(h * ratio)
    img_resized = img.resize((new_w, h), Image.BICUBIC)

    left = Image.new('RGB', (w_panel, h), (255, 255, 255))
    left.paste(img_resized, ((w_panel - new_w) // 2 if new_w <= w_panel else 0, 0))

    right = Image.new('RGB', (w_panel, h), (250, 250, 250))
    draw = ImageDraw.Draw(right)
    if font_cache is None: 
        font_cache = {}
    try:
        font_title  = font_cache.get('title') or ImageFont.truetype('DejaVuSans-Bold.ttf', 24)
        font_text   = font_cache.get('text')  or ImageFont.truetype('DejaVuSans.ttf', 18)
        font_metrics= font_cache.get('mono')  or ImageFont.truetype('DejaVuSansMono.ttf', 18)
        font_cache.update({'title': font_title, 'text': font_text, 'mono': font_metrics})
    except Exception:
        font_title = font_text = font_metrics = ImageFont.load_default()

    pad = 22
    x = pad
    y = pad
    max_text_w = w_panel - 2 * pad

    draw.text((x, y), 'Ground Truth', fill=(0, 0, 0), font=font_title)
    y += int(font_title.size * 1.4)
    gt_w = wrap_multiline(gt or '', font_text, max_text_w, draw)
    draw.multiline_text((x, y), gt_w, fill=(20, 20, 20), font=font_text, spacing=6)
    y = draw.multiline_textbbox((x, y), gt_w, font=font_text, spacing=6)[3] + int(font_title.size * 0.8)

    draw.text((x, y), 'Model', fill=(0, 0, 0), font=font_title)
    y += int(font_title.size * 1.4)
    md_w = wrap_multiline(model_pred or '', font_text, max_text_w, draw)
    draw.multiline_text((x, y), md_w, fill=(10, 10, 10), font=font_text, spacing=6)

    y = draw.multiline_textbbox((x, y), md_w, font=font_text, spacing=6)[3] + int(font_title.size * 0.8)
    draw.text((x, y), 'Metrics', fill=(0, 0, 0), font=font_title)
    y += int(font_title.size * 1.2)
    m = metrics_model
    col = "\n".join([
        f"CLIP: {m.get('clip', 0):.1f}",
        f"BLEU: {m.get('bleu', 0):.1f}",
        f"ROUGE-L: {m.get('rougeL', 0):.3f}",
        f"CIDEr-D: {m.get('cider', 0):.2f}",
    ])
    draw.multiline_text((x, y), col, fill=(0, 0, 0), font=font_metrics, spacing=8)

    canvas = Image.new('RGB', (w_panel * 2, h), (255, 255, 255))
    canvas.paste(left, (0, 0))
    canvas.paste(right, (w_panel, 0))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=95)


def compute_text_metrics_multi(hyps: List[str], refs: List[str], clip_backend=None, 
                               img_paths: List[str] = None) -> Dict[str, float]:
    """Вычисление текстовых метрик для мультимодельного инференса"""
    if not hyps or not refs or len(hyps) != len(refs):
        return {'BLEU': 0.0, 'BERT_P': 0.0, 'BERT_R': 0.0, 'BERT_F1': 0.0, 
                'ROUGE_L': 0.0, 'CIDEr_D': 0.0, 'CLIP': 0.0}
    
    filtered_pairs = [(h, r, i) for i, (h, r) in enumerate(zip(hyps, refs)) if h.strip() and r.strip()]
    if not filtered_pairs:
        return {'BLEU': 0.0, 'BERT_P': 0.0, 'BERT_R': 0.0, 'BERT_F1': 0.0, 
                'ROUGE_L': 0.0, 'CIDEr_D': 0.0, 'CLIP': 0.0}
    
    hyps_filtered, refs_filtered, valid_indices = zip(*filtered_pairs)
    
    try:
        bleu = sacrebleu.corpus_bleu(hyps_filtered, [refs_filtered]).score
    except:
        bleu = 0.0
    
    rouge_scores = [rouge_l(r, h) for r, h in zip(refs_filtered, hyps_filtered)]
    rouge_l_score = float(np.mean(rouge_scores)) if rouge_scores else 0.0
    
    cider_idf = build_cider_idf(list(refs_filtered))
    cider_scores = [ciderD_pair(h, r, cider_idf) for h, r in zip(hyps_filtered, refs_filtered)]
    cider_d_score = float(np.mean(cider_scores)) if cider_scores else 0.0
    
    clip_score = 0.0
    if clip_backend is not None and img_paths is not None:
        try:
            valid_paths = [img_paths[i] for i in valid_indices]
            clip_scores = clip_backend.clipscore(valid_paths, list(hyps_filtered))
            clip_score = float(np.mean(clip_scores)) if clip_scores else 0.0
        except Exception as e:
            clip_score = 0.0
    
    return {
        'BLEU': float(bleu), 
        'BERT_P': 0.0,  # Упрощено для скорости
        'BERT_R': 0.0, 
        'BERT_F1': 0.0,
        'ROUGE_L': rouge_l_score,
        'CIDEr_D': cider_d_score,
        'CLIP': clip_score
    }


def save_examples(df: pd.DataFrame, hypotheses: List[str], references: List[str], 
                  img_paths: List[str], out_path: Path, indices: List[int], title: str):
    """Сохраняет панель с примерами генерации"""
    examples = []
    for idx in indices:
        if idx >= len(img_paths):
            continue
        img_path = img_paths[idx]
        ref = references[idx] if idx < len(references) else ''
        hyp = hypotheses[idx] if idx < len(hypotheses) else ''
        
        # Загружаем и обрабатываем изображение
        try:
            img = Image.open(img_path).convert('RGB')
            img = img.resize((224, 224))
        except Exception as e:
            img = Image.new('RGB', (224, 224), color=0)
        
        examples.append((img, ref, hyp))
    
    if not examples:
        return
    
    # Создаем панель
    panel_w, panel_h = 224, PANEL_H
    num_examples = len(examples)
    canvas = Image.new('RGB', (panel_w * num_examples, panel_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    
    try:
        font = ImageFont.truetype('DejaVuSans.ttf', 12)
    except:
        font = ImageFont.load_default()
    
    for i, (img, ref, hyp) in enumerate(examples):
        # Вставляем изображение
        canvas.paste(img, (i * panel_w, 0))
        
        # Добавляем подписи
        y_text = 230
        if y_text < panel_h - 50:
            draw.text((i * panel_w + 5, y_text), "Reference:", fill=(0, 0, 0), font=font)
            y_text += 15
            ref_short = ref[:30] + "..." if len(ref) > 30 else ref
            draw.text((i * panel_w + 5, y_text), ref_short, fill=(0, 0, 0), font=font)
            y_text += 25
            draw.text((i * panel_w + 5, y_text), "Hypothesis:", fill=(0, 0, 0), font=font)
            y_text += 15
            hyp_short = hyp[:30] + "..." if len(hyp) > 30 else hyp
            draw.text((i * panel_w + 5, y_text), hyp_short, fill=(0, 0, 0), font=font)
    
    # Добавляем заголовок
    draw.text((10, panel_h - 30), title, fill=(0, 0, 0), font=font)
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=95)


@torch.no_grad()
def run_multi_model_infer(csv_path: Path, ckpt_paths: List[Path], out_dir: Path,
                         source_filter: Optional[str], n_samples: int,
                         device: torch.device) -> None:
    """Инференс на нескольких моделях одновременно"""
    set_seed(SEED)
    
    # Загружаем все модели
    models = []
    model_names = []
    clip_backends = []
    
    # Создаем выходную директорию
    out_dir.mkdir(parents=True, exist_ok=True)
    
    for i, ckpt_path in enumerate(ckpt_paths):
        print(f"Загрузка модели {i+1}/{len(ckpt_paths)}: {ckpt_path}")
        pack = torch.load(ckpt_path, map_location='cpu')
        state = pack.get('model', pack)
        cfg: dict = pack.get('cfg', {})
        meta: dict = pack.get('meta', {})

        if 'prompt' not in cfg:
            if 'prompt_B' in cfg: cfg['prompt'] = cfg['prompt_B']
            elif 'prompt_A' in cfg: cfg['prompt'] = cfg['prompt_A']
            else: cfg['prompt'] = ''

        if 'embed_dim' in meta:
            cfg['embed_dim'] = int(meta['embed_dim'])

        if 'CLIP_PRESET' not in cfg and 'clip_backend_kind' in meta:
            if meta.get('clip_backend_kind') == 'open_clip':
                if 'DermLIP_ViT-B-16' in meta.get('clip_repo', ''):
                    cfg['CLIP_PRESET'] = 'dermlip_vitb16'
                elif 'BiomedCLIP' in meta.get('clip_repo', ''):
                    cfg['CLIP_PRESET'] = 'BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
                elif 'DermLIP_PanDerm-base-w-PubMed-256' in meta.get('clip_repo', ''):
                    cfg['CLIP_PRESET'] = 'dermlip_panderm_pubmed256'
            elif meta.get('clip_backend_kind') == 'hf_clip':
                if 'pubmed-clip-vit-base-patch32' in meta.get('clip_repo', ''):
                    cfg['CLIP_PRESET'] = 'pubmed_clip_b32'
        
        # Обработка TimmModel для BiomedCLIP
        if 'TimmModel' in str(ckpt_path) and 'biomedclip' in str(ckpt_path).lower():
            cfg['CLIP_PRESET'] = 'biomedclip_pubmedbert'
        
        model = DermCaptioner(cfg, device=str(device)).to(device).eval()
        model.load_state_dict(state, strict=True)
        models.append((model, cfg))
        
        # Извлекаем имя модели из пути или конфигурации
        model_name = ckpt_path.stem
        if 'dermlip' in model_name.lower():
            model_name = 'DermLIP'
        elif 'pubmed' in model_name.lower():
            model_name = 'PubMedCLIP'
        elif 'biomed' in model_name.lower() or cfg.get('CLIP_PRESET', '').startswith('BiomedCLIP'):
            model_name = 'BiomedCLIP'
        else:
            model_name = f'Model_{i+1}'
        
        if hasattr(cfg, 'CLIP_PRESET'):
            model_name = f"{model_name}_{cfg['CLIP_PRESET']}"
        model_names.append(model_name)
        
        # Инициализируем CLIP backend для текущей модели
        clip_backend = model.clip_backend
        clip_backends.append(clip_backend)
        print(f"CLIP backend инициализирован для {model_name}")

    # Подготовка CSV
    df = pd.read_csv(csv_path, encoding='utf-8-sig', low_memory=False)
    col_img = models[0][1].get('COL_IMG', 'img_path')
    col_txt = models[0][1].get('COL_TXT', 'caption')

    if col_img not in df.columns: 
        raise ValueError(f"В CSV нет колонки: {col_img}")
    if col_txt not in df.columns: 
        col_txt = pick_text_col(df)

    if source_filter and models[0][1].get('COL_SRC') in df.columns:
        df = df[df[models[0][1]['COL_SRC']].astype(str).str.contains(str(source_filter), 
                                                                    case=False, 
                                                                    regex=False)]
    
    # Проверка существования изображений
    valid_mask = df[col_img].map(lambda p: Path(str(p)).exists())
    df = df[valid_mask].reset_index(drop=True)
    
    if len(df) == 0:
        raise ValueError("Не найдено ни одного существующего изображения")
    
    if n_samples > 0 and n_samples < len(df):
        df = df.sample(n=n_samples, random_state=SEED).reset_index(drop=True)
    
    # Подготовка данных
    img_paths = df[col_img].tolist()
    references = df[col_txt].tolist()
    
    print(f"Обработка {len(img_paths)} изображений")
    
    # Генерация гипотез для всех моделей
    all_hypotheses = []
    for i, (model, cfg) in enumerate(models):
        print(f"Генерация гипотез модель {i+1}/{len(models)}: {model_names[i]}")
        hypotheses = []
        for j in tqdm(range(0, len(img_paths), IMG_BATCH), desc=f"Model {i+1}"):
            batch_paths = img_paths[j:j+IMG_BATCH]
            batch_hyps = model.generate(batch_paths)
            hypotheses.extend(batch_hyps)
        all_hypotheses.append(hypotheses)
    
    # Вычисление метрик для каждой модели
    metrics_list = []
    for i in range(len(models)):
        print(f"Вычисление метрик для модели {i+1}/{len(models)}: {model_names[i]}")
        metrics = compute_text_metrics_multi(
            all_hypotheses[i], 
            references, 
            clip_backends[i], 
            img_paths
        )
        metrics_list.append(metrics)
        print(f"Метрики для {model_names[i]}: {metrics}")
    
    # Сохранение результатов
    results = []
    for i in range(len(img_paths)):
        result = {
            'image_path': img_paths[i],
            'reference': references[i]
        }
        for j in range(len(models)):
            result[f'hypothesis_{model_names[j]}'] = all_hypotheses[j][i]
        results.append(result)
    
    # Сохранение в CSV
    results_df = pd.DataFrame(results)
    results_csv = out_dir / 'multi_model_results.csv'
    results_df.to_csv(results_csv, index=False)
    print(f"Результаты сохранены в {results_csv}")
    
    # Создание сводной таблицы метрик
    metrics_summary = {}
    for metric in ['BLEU', 'BERT_P', 'BERT_R', 'BERT_F1', 'ROUGE_L', 'CIDEr_D', 'CLIP']:
        metrics_summary[metric] = {}
        for i in range(len(models)):
            metrics_summary[metric][model_names[i]] = metrics_list[i][metric]
    
    # Сохранение сводной таблицы
    metrics_df = pd.DataFrame(metrics_summary).T
    metrics_csv = out_dir / 'metrics_summary.csv'
    metrics_df.to_csv(metrics_csv)
    print(f"Сводная таблица метрик сохранена в {metrics_csv}")
    
    # Визуализация результатов
    if plt is not None:
        plt.figure(figsize=(12, 8))
        metrics_df.plot(kind='bar', figsize=(12, 8))
        plt.title('Сравнение метрик между моделями')
        plt.ylabel('Значение метрики')
        plt.xticks(rotation=0)
        plt.legend(title='Модели')
        plt.tight_layout()
        plt.savefig(out_dir / 'metrics_comparison.png', dpi=160)
        plt.close()
        print(f"График сравнения метрик сохранен")
    
    # Примеры лучших и худших результатов для каждой модели
    for i in range(len(models)):
        # Сортируем по CLIP оценке
        clip_scores = []
        for j in range(len(img_paths)):
            try:
                score = clip_backends[i].clipscore([img_paths[j]], [all_hypotheses[i][j]])[0]
                clip_scores.append(score)
            except:
                clip_scores.append(0.0)
        
        # Создаем индексы для сортировки
        indices = list(range(len(clip_scores)))
        indices.sort(key=lambda x: clip_scores[x], reverse=True)
        
        # Лучшие результаты
        best_indices = indices[:min(5, len(indices))]
        worst_indices = indices[-min(5, len(indices)):]
        
        # Сохраняем примеры
        save_examples(df, all_hypotheses[i], references, img_paths, 
                     out_dir / f'best_examples_{model_names[i]}.jpg', 
                     best_indices, f"Лучшие результаты {model_names[i]}")
        save_examples(df, all_hypotheses[i], references, img_paths, 
                     out_dir / f'worst_examples_{model_names[i]}.jpg', 
                     worst_indices, f"Худшие результаты {model_names[i]}")
    
    print("Мультимодельный инференс завершен успешно")


@torch.no_grad()
def run_infer(csv_path: Path, ckpt_path: Path, out_dir: Path,
              source_filter: Optional[str], n_samples: int, 
              device: torch.device) -> None:
    set_seed(SEED)

    pack = torch.load(ckpt_path, map_location='cpu')
    state = pack.get('model', pack)
    cfg: dict = pack.get('cfg', {})
    meta: dict = pack.get('meta', {})

    if 'prompt' not in cfg:
        if 'prompt_B' in cfg: cfg['prompt'] = cfg['prompt_B']
        elif 'prompt_A' in cfg: cfg['prompt'] = cfg['prompt_A']
        else: cfg['prompt'] = ''

    if 'embed_dim' in meta:
        cfg['embed_dim'] = int(meta['embed_dim'])

    if 'CLIP_PRESET' not in cfg and 'clip_backend_kind' in meta:
        if meta.get('clip_backend_kind') == 'open_clip':
            if 'DermLIP_ViT-B-16' in meta.get('clip_repo', ''):
                cfg['CLIP_PRESET'] = 'dermlip_vitb16'
            elif 'BiomedCLIP' in meta.get('clip_repo', ''):
                cfg['CLIP_PRESET'] = 'BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
            elif 'DermLIP_PanDerm-base-w-PubMed-256' in meta.get('clip_repo', ''):
                cfg['CLIP_PRESET'] = 'dermlip_panderm_pubmed256'
        elif meta.get('clip_backend_kind') == 'hf_clip':
            if 'pubmed-clip-vit-base-patch32' in meta.get('clip_repo', ''):
                cfg['CLIP_PRESET'] = 'pubmed_clip_b32'
    
    # Обработка TimmModel для BiomedCLIP
    if 'TimmModel' in str(ckpt_path) and 'biomedclip' in str(ckpt_path).lower():
        cfg['CLIP_PRESET'] = 'biomedclip_pubmedbert'
    
    model = DermCaptioner(cfg, device=str(device)).to(device).eval()
    missing, unexpected = model.load_state_dict(state, strict=True)

    df = pd.read_csv(csv_path, encoding='utf-8-sig', low_memory=False)
    col_img = cfg.get('COL_IMG', 'img_path')
    col_txt = cfg.get('COL_TXT', 'caption')

    if col_img not in df.columns: 
        raise ValueError(f"В CSV нет колонки: {col_img}")
    if col_txt not in df.columns: 
        col_txt = pick_text_col(df)

    if source_filter and cfg.get('COL_SRC') in df.columns:
        df = df[df[cfg['COL_SRC']].astype(str).str.contains(str(source_filter), 
                                                            case=False, 
                                                            regex=False)]
    
    df = df[df[col_img].map(lambda p: Path(str(p)).exists())].reset_index(drop=True)

    if n_samples <= 0 or n_samples > len(df): 
        n_samples = len(df)

    rows = df.sample(n=n_samples, random_state=SEED).reset_index(drop=True)

    out_dir.mkdir(parents=True, exist_ok=True)

    gt_for_idf = [str(rows.iloc[i][col_txt]) for i in range(len(rows))]
    cider_idf = build_cider_idf(gt_for_idf)

    results: List[Dict[str, object]] = []
    font_cache: Dict[str, ImageFont.FreeTypeFont] = {}

    all_clip, all_bleu, all_rouge, all_cider = [], [], [], []

    for i in tqdm(range(0, len(rows), IMG_BATCH), desc='Infer', dynamic_ncols=True):
        part  = rows.iloc[i:i+IMG_BATCH]
        paths = [str(Path(p)) for p in part[col_img].tolist()]
        gts   = [str(part.iloc[j][col_txt]) if col_txt in part.columns else '' for j in range(len(part))]

        preds = model.generate(paths)

        clip_gen = model.clip_backend.clipscore(paths, preds)
        bleus    = [sent_bleu(r, h) if r else 0.0 for r, h in zip(gts, preds)]
        rls      = [rouge_l(r, h)   if r else 0.0 for r, h in zip(gts, preds)]
        cids     = [ciderD_pair(h, r, cider_idf) if r else 0.0 for r, h in zip(gts, preds)]

        all_clip.extend(clip_gen)
        all_bleu.extend(bleus)
        all_rouge.extend(rls)
        all_cider.extend(cids)

        pil_imgs = [Image.open(Path(p)).convert('RGB') for p in paths]
        for j in range(len(part)):
            m_metrics = {'clip': clip_gen[j], 'bleu': bleus[j], 'rougeL': rls[j], 'cider': cids[j]}
            make_card_pair(
                pil_imgs[j], gts[j], preds[j],
                metrics_model=m_metrics,
                out_path=out_dir / f"preview_{i+j+1:04d}_{Path(paths[j]).stem}.png",
                panel_h=PANEL_H, font_cache=font_cache
            )
            results.append({
                'img_path': paths[j],
                'gt': gts[j],
                'gen': preds[j],
                'gen_clip': clip_gen[j],
                'gen_bleu': bleus[j],
                'gen_rougeL': rls[j],
                'gen_ciderD': cids[j],
            })

    tsv_path = out_dir / 'inference_results.tsv'
    pd.DataFrame(results).to_csv(tsv_path, sep='\t', index=False)
    print(f'[save] previews → {out_dir.resolve()}')
    print(f'[save] TSV     → {tsv_path.resolve()}')

    if len(all_clip):
        print(f"[summary] CLIP={np.mean(all_clip):.2f} | BLEU={np.mean(all_bleu):.2f} | "
              f"ROUGE-L={np.mean(all_rouge):.3f} | CIDEr-D={np.mean(all_cider):.3f}")

    if plt is not None and len(all_clip):
        xs = list(range(1, len(all_clip)+1))
        plt.figure(figsize=(10, 5))
        plt.plot(xs, all_clip, marker='o', label='Model (CLIPScore)')
        plt.xlabel('Sample #')
        plt.ylabel('CLIPScore')
        plt.title('CLIPScore: Model')
        plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
        plot_path = out_dir / 'summary_clip_model.png'
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f'[save] plot    → {plot_path.resolve()}')
    else:
        print('[plot] matplotlib недоступен или нет результатов для графика')


if __name__ == '__main__':
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Device:', device)
    
    # Проверяем, нужен ли мультимодельный инференс
    if len(CKPT_PATHS) > 1:
        print(f"Running multi-model inference with {len(CKPT_PATHS)} models:")
        for i, path in enumerate(CKPT_PATHS):
            print(f"  {i+1}. {path}")
        run_multi_model_infer(Path(CSV_PATH), 
                             [Path(p) for p in CKPT_PATHS],
                             Path(OUT_DIR), 
                             SOURCE_FILTER, 
                             SAMPLES, 
                             device)
    else:
        run_infer(Path(CSV_PATH), 
                  Path(CKPT_PATHS[0]), 
                  Path(OUT_DIR), 
                  SOURCE_FILTER, 
                  SAMPLES, 
                  device)
