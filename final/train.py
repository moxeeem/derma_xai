# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import logging as pylog
import math
import os
import random
import re
import sys
import time
import warnings
from collections import Counter, OrderedDict
from dataclasses import asdict, dataclass
from io import StringIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TextIO, Any

import matplotlib.pyplot as plt
import numpy as np
import open_clip
import pandas as pd
import sacrebleu
import torch
import torch.nn as nn
import torch.nn.functional as F
from bert_score import score as bert_score
from PIL import Image, ImageFile
from rouge_score import rouge_scorer
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class TeeLogger:
    def __init__(self, filename: str, mode: str = 'w', filter_progress: bool = True):
        self.terminal = sys.stdout
        self.file = open(filename, mode, encoding='utf-8', buffering=1)
        self.filter_progress = filter_progress
        self._last_line = ""
        self._is_progress_line = False
        
    def write(self, message: str):
        self.terminal.write(message)
        
        if self.filter_progress:
            if '\r' in message:
                self._is_progress_line = True
                self._last_line = message.rstrip('\r\n')
                return
            elif message.strip() and self._is_progress_line:
                if self._last_line:
                    self.file.write(self._last_line + '\n')
                    self._last_line = ""
                self._is_progress_line = False
        
        if not self._is_progress_line:
            self.file.write(message)
            
    def flush(self):
        if self._is_progress_line and self._last_line:
            self.file.write(self._last_line + '\n')
            self._last_line = ""
            self._is_progress_line = False
            
        self.terminal.flush()
        self.file.flush()
        
    def close(self):
        self.flush()
        self.file.close()
        
    def __del__(self):
        self.close()

try:
    from transformers.pytorch_utils import Conv1D as HFConv1D
except Exception:
    try:
        from transformers.modeling_utils import Conv1D as HFConv1D
    except Exception:
        HFConv1D = None

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    CLIPImageProcessor as HFCLIPImageProcessor,
    CLIPModel as HFCLIPModel,
    PreTrainedTokenizerBase,
    get_cosine_schedule_with_warmup,
    logging as hf_logging,
)

hf_logging.set_verbosity_error()
pylog.getLogger("transformers").setLevel(pylog.ERROR)
warnings.filterwarnings(
    "ignore",
    message=r"Some weights of *",
)

torch.backends.cudnn.benchmark = True
ImageFile.LOAD_TRUNCATED_IMAGES = True


APPROACH_NAME = 'dermlip_vitb16'


@dataclass
class TrainCfg:
    # CLIP пресеты: 'dermlip_vitb16' | 'pubmed_clip_b32' | 'biomedclip_pubmedbert'
    CLIP_PRESET: str = 'dermlip_vitb16'

    CSV_META: str = '/root/clip_xai/dermaCAP/dermaCAP_v1_clean.csv'
    OUT_DIR: str = f'/root/clip_xai/final/final_{APPROACH_NAME}'

    IMG_ROOT: str = ''
    COL_IMG: str = 'img_path'
    COL_TXT: str = 'caption'
    COL_SPLIT: str = 'split'
    COL_SRC: Optional[str] = 'source'
    SKINCAP_TAG: str = 'skincap'
    GPT2_NAME: str = 'gpt2-medium'

    seed: int = 42
    device: str = 'cuda'
    num_workers: int = 4
    patience: int = 3
    clip_grad_norm: float = 1.0
    
    
    batch_size_A: int = 24
    batch_size_B: int = 16
    
    lr_A: float = 1.2e-4         
    lr_B: float = 3.0e-5 
    
    epochs_A: int = 3
    epochs_B: int = 20
    
    unfreeze_last_k_layers_A: int = 0
    unfreeze_last_k_layers_B: int = 1 
    unfreeze_schedule_B: Tuple[int, int] = (3, 2)  # с 3-й эпохи B разморозим 2 блока GPT-2
    freeze_prefix_in_B: bool = True
    
    
    label_smoothing_A: float = 0.05
    label_smoothing_B: float = 0.00
    
    wd_A: float = 0.01
    wd_B: float = 0.00

    gen_beams: int = 4
    gen_max_new: int = 80
    no_repeat_ngram_size: int = 4
    repetition_penalty: float = 1.15
    length_penalty: float = 0.6
    min_new_tokens: int = 24

    prompt_A: str = (
        'In ≤25 words, describe primary lesion morphology, color, scale, border, and anatomical site. '
        'Conclude with the most likely diagnosis (1–3 words).'
    )
    prompt_B: str = (
        'Describe the skin lesion concisely (morphology, color, scale, border, location) in one sentence.'
        'Conclude with the most likely diagnosis (1–3 words).'
    )

    prefix_tokens: int = 32
    prefix_dropout_p: float = 0.05

    cliche_phrases: Tuple[str, ...] = (
        'seek medical attention',
        'consult a dermatologist',
        'promptly for further',
        'It is recommended to',
        'further evaluation is required',
        'should be considered for',
        'medical attention',
        'follow up',
        'consult your doctor',
        'should be evaluated'
    )
    
    amp_enabled: bool = True  # авто bf16 (если доступен), иначе fp16

    global_val_frac: float = 0.08
    global_val_min_per_source: int = 50

    clip_cache_enabled: bool = True
    clip_cache_max_gb: float = 6.0
    loss_plot_name: str = f'loss_{APPROACH_NAME}.png'
    examples_to_show: int = 2
    clipscore_batch: int = 64  # батчинг для расчёта CLIPScore, чтобы избежать OOM


cfg = TrainCfg()
random.seed(cfg.seed)
np.random.seed(cfg.seed)
torch.manual_seed(cfg.seed)


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


def _count_trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class CaptionDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_root: str, col_img: str, 
                 col_txt: str, tok: PreTrainedTokenizerBase, prompt: str):
        self.df = df.reset_index(drop=True)
        self.img_root = img_root
        self.col_img = col_img
        self.col_txt = col_txt
        self.tok = tok
        self.prompt = prompt.strip()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        img_path = str(row[self.col_img])
        if self.img_root and not os.path.isabs(img_path):
            img_path = os.path.join(self.img_root, img_path)
        caption = str(row[self.col_txt]).strip()
        return img_path, caption


def collate(batch, tok: PreTrainedTokenizerBase, prompt: str):
    batch = [(p, t) for p, t in batch if isinstance(t, str) and len(t.strip()) > 0]
    if not batch:
        return [], None, None
    paths, txts = zip(*batch)
    inputs = tok([prompt] * len(txts), return_tensors='pt', padding=True, 
                 truncation=True, max_length=64)
    labels = tok(list(txts), return_tensors='pt', padding=True, 
                 truncation=True, max_length=96)

    return list(paths), inputs, labels


def load_df(cfg: TrainCfg) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(cfg.CSV_META)
    if cfg.COL_SPLIT in df.columns:
        tr = df[df[cfg.COL_SPLIT].astype(str).str.lower().eq('train')]
        va = df[df[cfg.COL_SPLIT].astype(str).str.lower().eq('val')]
        te = df[df[cfg.COL_SPLIT].astype(str).str.lower().eq('test')]
    else:
        df = df.sample(frac=1.0, random_state=cfg.seed).reset_index(drop=True)
        n = len(df); a = int(0.8*n); b = int(0.9*n)
        tr, va, te = df.iloc[:a], df.iloc[a:b], df.iloc[b:]
    return tr.reset_index(drop=True), va.reset_index(drop=True), te.reset_index(drop=True)


def subset_by_source(df: pd.DataFrame, cfg: TrainCfg, tag: str) -> pd.DataFrame:
    if cfg.COL_SRC and (cfg.COL_SRC in df.columns):
        return df[df[cfg.COL_SRC].astype(str).str.contains(tag, case=False, regex=False)].reset_index(drop=True)
    return df


def make_global_validation(df_all: pd.DataFrame, cfg: TrainCfg) -> pd.DataFrame:
    if cfg.COL_SRC and (cfg.COL_SRC in df_all.columns):
        parts = []
        for src, grp in df_all.groupby(cfg.COL_SRC):
            k_by_frac = int(math.ceil(cfg.global_val_frac * len(grp)))
            k = max(cfg.global_val_min_per_source, k_by_frac)
            k = min(k, len(grp))
            parts.append(grp.sample(n=k, random_state=cfg.seed))
        gval = pd.concat(parts, axis=0).drop_duplicates().reset_index(drop=True)
    else:
        # fallback без источников
        frac = min(0.1, max(0.02, cfg.global_val_frac))
        gval = df_all.sample(frac=frac, random_state=cfg.seed).reset_index(drop=True)
    return gval


def make_loaders(df_train, tok, cfg, batch_size: int, prompt: str,
                 val_sets: Dict[str, pd.DataFrame]):
    ds_tr = CaptionDataset(df_train, cfg.IMG_ROOT, cfg.COL_IMG, cfg.COL_TXT, tok, prompt)
    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True,  num_workers=cfg.num_workers,
                       collate_fn=lambda b: collate(b, tok, prompt), pin_memory=True)
    val_loaders = {}
    for name, df_val in val_sets.items():
        ds_va = CaptionDataset(df_val, cfg.IMG_ROOT, cfg.COL_IMG, cfg.COL_TXT, tok, prompt)
        dl_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False, num_workers=cfg.num_workers,
                           collate_fn=lambda b: collate(b, tok, prompt), pin_memory=True)
        val_loaders[name] = dl_va
    return dl_tr, val_loaders


class CLIPBackend:
    def __init__(self, cfg: TrainCfg, device: str, out_dir: str):
        meta = HF_CLIP_PRESETS[cfg.CLIP_PRESET]
        self.kind = meta['kind']
        self.repo = meta['repo']
        self.device = device
        self.out_dir = out_dir
        self.bs = int(getattr(cfg, 'clipscore_batch', 64))

        self.cache_enabled = cfg.clip_cache_enabled
        self.cache_max_bytes = int(cfg.clip_cache_max_gb * (1024**3))
        self.cache: 'OrderedDict[str, Tuple[torch.Tensor,int]]' = OrderedDict()
        self.cache_bytes = 0

        if self.kind == 'open_clip':
            self.model_name = f'hf-hub:{self.repo}'
            self.model, self.preprocess, _ = open_clip.create_model_and_transforms(self.model_name)
            self.model = self.model.to(device).eval()
            self.processor = None
            self.arch = self.model.visual.__class__.__name__
            self.weight_path = self.model_name
            self.embed_dim = self._probe_embed_dim_openclip()
            # токенайзер OpenCLIP держим один раз
            self._oclip_tokenize = open_clip.get_tokenizer(self.model_name)
            self._hf_clip_tokenizer = None
        else:
            from transformers import CLIPTokenizer  # локальный импорт, чтобы не тянуть зря
            # HF CLIP (PubMedCLIP)
            self.model = HFCLIPModel.from_pretrained(self.repo).to(device).eval()
            self.processor = HFCLIPImageProcessor.from_pretrained(self.repo)
            self.preprocess = None
            self.arch = 'CLIPModel'
            self.weight_path = self.repo
            self.embed_dim = int(self.model.config.projection_dim)
            # токенайзер HF CLIP (держим в памяти)
            self._hf_clip_tokenizer = CLIPTokenizer.from_pretrained(self.repo)
            self._oclip_tokenize = None

    @torch.inference_mode()
    def _probe_embed_dim_openclip(self) -> int:
        img = Image.new('RGB', (224, 224), color=0)
        x = self.preprocess(img).unsqueeze(0).to(self.device)
        feat = self.model.encode_image(x)
        return int(feat.shape[-1])

    def _cache_get(self, path: str) -> Optional[torch.Tensor]:
        if not self.cache_enabled:
            return None
        v = self.cache.get(path)
        if v is None:
            return None
        tensor_cpu, nbytes = v
        self.cache.move_to_end(path)
        return tensor_cpu

    def _cache_put(self, path: str, tensor_cpu: torch.Tensor):
        if not self.cache_enabled:
            return
        nbytes = tensor_cpu.numel() * tensor_cpu.element_size()
        while self.cache_bytes + nbytes > self.cache_max_bytes and len(self.cache) > 0:
            _, (old_t, old_n) = self.cache.popitem(last=False)
            self.cache_bytes -= old_n
            del old_t
        self.cache[path] = (tensor_cpu, nbytes)
        self.cache.move_to_end(path)
        self.cache_bytes += nbytes

    def encode_images(self, paths: List[str]) -> torch.Tensor:
        cached_feats: Dict[int, torch.Tensor] = {}
        to_compute: List[Tuple[int, str]] = []

        for i, p in enumerate(paths):
            t_cpu = self._cache_get(p)
            if t_cpu is not None:
                cached_feats[i] = t_cpu.to(self.device, non_blocking=True)
            else:
                to_compute.append((i, p))

        new_feats: Dict[int, torch.Tensor] = {}
        if to_compute:
            with torch.no_grad():
                bs = max(1, self.bs)
                for s in range(0, len(to_compute), bs):
                    chunk = to_compute[s:s+bs]
                    ims: List[torch.Tensor] = []
                    idxs: List[int] = []
                    pil_batch: List[Image.Image] = []

                    for i, p in chunk:
                        try:
                            img = Image.open(p).convert('RGB')
                        except Exception:
                            img = Image.new('RGB', (224, 224), color=0)
                        if self.kind == 'open_clip':
                            ims.append(self.preprocess(img))
                        else:
                            pil_batch.append(img)
                        idxs.append(i)

                    if self.kind == 'open_clip':
                        x = torch.stack(ims).to(self.device, non_blocking=True)
                        feats = self.model.encode_image(x)
                    else:
                        proc = self.processor(images=pil_batch, return_tensors='pt')
                        x = proc['pixel_values'].to(self.device, non_blocking=True)
                        feats = self.model.get_image_features(pixel_values=x)

                    feats = F.normalize(feats, dim=-1).detach()

                    for j, i in enumerate(idxs):
                        f = feats[j]
                        new_feats[i] = f
                        self._cache_put(paths[i], f.detach().cpu())

        B = len(paths)
        out = [None] * B
        for i in range(B):
            t = cached_feats.get(i)
            if t is None:
                t = new_feats.get(i)
            assert t is not None, f'missing feature for index {i}'
            out[i] = t
        return torch.stack(out, dim=0)

    @torch.inference_mode()
    def clipscore(self, paths: List[str], texts: List[str]) -> List[float]:
        assert len(paths) == len(texts), "paths and texts must match in length"
        N = len(paths)
        if N == 0:
            return []

        scores: List[float] = []
        bs = max(1, self.bs)

        for s in range(0, N, bs):
            e = min(N, s + bs)
            p_chunk = paths[s:e]
            t_chunk = texts[s:e]

            img_f = self.encode_images(p_chunk)  # уже нормализованы
            if self.kind == 'open_clip':
                txt_tokens = self._oclip_tokenize(t_chunk).to(self.device, non_blocking=True)
                txt_f = self.model.encode_text(txt_tokens)
            else:
                # HF CLIP
                txt_tokens = self._hf_clip_tokenizer(
                    t_chunk, return_tensors='pt', padding=True, truncation=True, max_length=77
                ).to(self.device)
                txt_f = self.model.get_text_features(**txt_tokens)

            txt_f = F.normalize(txt_f, dim=-1)

            chunk_scores = (img_f * txt_f).sum(dim=-1) * 100.0
            scores.extend(chunk_scores.detach().cpu().tolist())

        return scores


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
        y = self.fc2(y)                      # [B, out_dim * tokens]
        y = y.view(y.size(0), self.tokens, -1)
        y = self.ln(y)
        y = self.drop(self.alpha * y)
        return y


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16, dropout: float = 0.05):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False

        in_f, out_f = base.in_features, base.out_features
        self.r = r
        self.scale = alpha / float(r)
        self.A = nn.Linear(in_f, r, bias=False)
        self.B = nn.Linear(r, out_f, bias=False)
        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B.weight)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.B(self.drop(self.A(x))) * self.scale


class LoRAConv1D(nn.Module):
    def __init__(self, base, r: int = 8, alpha: int = 16, dropout: float = 0.05):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        in_f = base.weight.shape[0]   # in_features
        out_f = base.weight.shape[1]  # out_features
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


def _wrap_module_with_lora(module: nn.Module, attr: str, r: int, alpha: int, dropout: float):
    if not hasattr(module, attr):
        return
    base = getattr(module, attr)

    if HFConv1D is not None and isinstance(base, HFConv1D):
        lora = LoRAConv1D(base, r=r, alpha=alpha, dropout=dropout)
    elif isinstance(base, nn.Linear):
        lora = LoRALinear(base, r=r, alpha=alpha, dropout=dropout)
    else:
        return

    lora.to(device=base.weight.device, dtype=base.weight.dtype)
    setattr(module, attr, lora)


def inject_gpt2_lora(gpt2: nn.Module, last_k: int, r: int = 8, alpha: int = 16, dropout: float = 0.05):
    if last_k <= 0:
        return
    blocks = gpt2.transformer.h
    for layer in blocks[-last_k:]:
        _wrap_module_with_lora(layer.attn, 'c_attn', r, alpha, dropout)
        _wrap_module_with_lora(layer.attn, 'c_proj', r, alpha, dropout)
        _wrap_module_with_lora(layer.mlp,  'c_fc',   r, alpha, dropout)
        _wrap_module_with_lora(layer.mlp,  'c_proj', r, alpha, dropout)


class DermCaptioner(nn.Module):
    def __init__(self, cfg: TrainCfg, device: str):
        super().__init__()
        self.cfg = cfg
        self.device = device

        self.tok = AutoTokenizer.from_pretrained(cfg.GPT2_NAME)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token

        self.gpt2 = AutoModelForCausalLM.from_pretrained(cfg.GPT2_NAME).to(device)
        try:
            self.gpt2.gradient_checkpointing_enable()
        except Exception:
            pass

        self._lora_k = max(cfg.unfreeze_last_k_layers_A, cfg.unfreeze_last_k_layers_B)
        inject_gpt2_lora(self.gpt2, last_k=self._lora_k, r=16, alpha=16, dropout=0.05)
        
        self.gpt2.to(device)

        self.clip_backend = CLIPBackend(cfg, device, cfg.OUT_DIR)

        self.img_dim = int(self.clip_backend.embed_dim)
        self.txt_dim = int(self.gpt2.config.n_embd)
        self.prefix = PrefixProjector(
            in_dim=self.img_dim,
            out_dim=self.txt_dim,
            tokens=cfg.prefix_tokens,
            p_drop=cfg.prefix_dropout_p
        ).to(device)


    def freeze_gpt2_except_last_k(self, k: int):
        for n, p in self.gpt2.named_parameters():
            p.requires_grad = False

        for n, p in self.gpt2.named_parameters():
            if 'A.weight' in n or 'B.weight' in n:  # LoRA-маркер
                p.requires_grad = True

        blocks = self.gpt2.transformer.h
        if k > 0:
            for layer in blocks[-k:]:
                for p in layer.parameters():
                    p.requires_grad = True

        for p in self.gpt2.lm_head.parameters():
            p.requires_grad = True
        for p in self.gpt2.transformer.wte.parameters():
            p.requires_grad = True


    @torch.inference_mode()
    def generate(self, img_paths: List[str], prompt: Optional[str] = None) -> List[str]:
        c = self.cfg
        if prompt is None:
            prompt = getattr(
                c, 'prompt',
                getattr(c, 'prompt_B', getattr(c, 'prompt_A', ''))
            )

        img_feat = self.clip_backend.encode_images(img_paths).detach().clone()
        pref = self.prefix(img_feat)

        prompt_ids = self.tok(
            [prompt] * pref.size(0),
            return_tensors='pt', padding=True, truncation=True
        ).to(self.device)
        prompt_emb = self.gpt2.transformer.wte(prompt_ids['input_ids'])
        inputs_embeds = torch.cat([pref, prompt_emb], dim=1)
        attn = torch.ones(inputs_embeds.size()[:-1], dtype=torch.long, device=self.device)

        # bad_words_ids
        bad_seqs = None
        if c.cliche_phrases and any(p.strip() for p in c.cliche_phrases):
            tmp = []
            for phrase in c.cliche_phrases:
                s = phrase.strip()
                if not s:
                    continue
                ids = self.tok.encode(s, add_special_tokens=False)
                if ids:
                    tmp.append(ids)
            if tmp:
                bad_seqs = tmp

        gen = self.gpt2.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attn,
            max_new_tokens=min(c.gen_max_new, 60),
            min_new_tokens=c.min_new_tokens,
            num_beams=max(4, min(6, c.gen_beams)),
            do_sample=False,
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
            cut = s.find(prompt)
            if cut >= 0:
                s = s[cut + len(prompt):]
            res.append(s.strip())
        return res


    def compute_loss(self, img_paths: List[str], inp_batch, lbl_batch) -> torch.Tensor:
        c = self.cfg
        img_feat = self.clip_backend.encode_images(img_paths).detach().clone()
        pref = self.prefix(img_feat)

        in_ids  = inp_batch['input_ids'].to(self.device)
        tgt_ids = lbl_batch['input_ids'].to(self.device)

        emb_prompt = self.gpt2.transformer.wte(in_ids)
        emb_target = self.gpt2.transformer.wte(tgt_ids)
        txt_embeds = torch.cat([emb_prompt, emb_target], dim=1)

        inputs_embeds = torch.cat([pref, txt_embeds], dim=1)
        attn = torch.ones(inputs_embeds.size()[:-1], dtype=torch.long, device=self.device)

        B, Tpref = pref.size(0), pref.size(1)
        Tp = in_ids.size(1); Tt = tgt_ids.size(1)
        labels = torch.full((B, Tpref + Tp + Tt), fill_value=-100, dtype=torch.long, device=self.device)
        labels[:, Tpref + Tp:] = tgt_ids

        out = self.gpt2(inputs_embeds=inputs_embeds, attention_mask=attn)
        logits = out.logits  # [B, T, V]

        # shift для next-token обучения
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
            label_smoothing=float(c.label_smoothing) if c.label_smoothing > 0 else 0.0,
        )
        return loss
    
    def set_prefix_trainable(self, flag: bool):
        for p in self.prefix.parameters():
            p.requires_grad = flag


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


def _tfidf_vec(tokens: List[str], idf: List[Dict[Tuple[str,...], float]], nmax: int = 4) -> List[Dict[Tuple[str,...], float]]:
    vecs = []
    for n in range(1, nmax+1):
        tf = _ngrams(tokens, n)
        L = sum(tf.values()) or 1
        v = {g: (c / L) * idf[n-1].get(g, 0.0) for g, c in tf.items()}
        vecs.append(v)
    return vecs


def ciderD_pair(hyp: str, ref: str, idf: List[Dict[Tuple[str,...], float]], nmax: int = 4, sigma: float = 6.0) -> float:
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


def compute_text_metrics(hyps: List[str], refs: List[str], clip_backend=None, 
                         img_paths: List[str] = None) -> Dict[str, float]:
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
    
    try:
        P, R, F = bert_score(
            hyps_filtered,
            refs_filtered,
            lang='en',
            rescale_with_baseline=True,
            model_type='microsoft/deberta-large-mnli', 
            verbose=False
        )

        bert_p = float(P.mean()) if len(P) > 0 else 0.0
        bert_r = float(R.mean()) if len(R) > 0 else 0.0
        bert_f1 = float(F.mean()) if len(F) > 0 else 0.0
    except:
        bert_p = bert_r = bert_f1 = 0.0
    
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
        'BERT_P': bert_p, 
        'BERT_R': bert_r, 
        'BERT_F1': bert_f1,
        'ROUGE_L': rouge_l_score,
        'CIDEr_D': cider_d_score,
        'CLIP': clip_score
    }


def train_stage(
    stage_name: str,
    model: DermCaptioner,
    df_train: pd.DataFrame,
    val_sets: Dict[str, pd.DataFrame],
    primary_val_name: str,
    epochs: int,
    lr: float,
    batch_size: int,
    unfreeze_k: int,
    out_ckpt: Path,
    cfg: TrainCfg,
    *,
    prompt_override: Optional[str] = None,
    label_smoothing_override: Optional[float] = None,
    weight_decay_override: Optional[float] = None,
    freeze_prefix: bool = False,
    unfreeze_schedule_k: Optional[Tuple[int, int]] = None) -> Dict[str, List[float]]:

    print(f'[{stage_name}] start | train={len(df_train)} | unfreeze_k={unfreeze_k}')
    for n, dfv in val_sets.items():
        print(f'[{stage_name}] will validate on: {n} (N={len(dfv)})')
    print(f'[{stage_name}] trainable_params={_count_trainable(model):,}')

    prev_prompt = cfg.prompt_A
    stage_prompt = prompt_override if prompt_override is not None else cfg.prompt_A

    if not hasattr(cfg, 'label_smoothing'):
        cfg.label_smoothing = cfg.label_smoothing_A
    prev_smooth = cfg.label_smoothing
    cfg.label_smoothing = (label_smoothing_override
                           if label_smoothing_override is not None else cfg.label_smoothing)

    if freeze_prefix:
        model.set_prefix_trainable(False)
    else:
        model.set_prefix_trainable(True)

    model.freeze_gpt2_except_last_k(unfreeze_k)
    print(f'[{stage_name}] after freeze_k={unfreeze_k} | trainable={_count_trainable(model):,}')

    params = list(model.parameters())
    wd = weight_decay_override if weight_decay_override is not None else cfg.wd_A
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=wd)

    steps_per_epoch = max(1, math.ceil(len(df_train) / max(1, batch_size)))
    num_training_steps = steps_per_epoch * epochs
    num_warmup_steps = max(10, int(0.06 * num_training_steps))
    scheduler = get_cosine_schedule_with_warmup(opt, num_warmup_steps, num_training_steps)

    amp_ok = (cfg.amp_enabled and torch.cuda.is_available())
    amp_dtype = torch.bfloat16 if (amp_ok and torch.cuda.is_bf16_supported()) else torch.float16
    use_scaler = (amp_ok and amp_dtype == torch.float16)
    scaler = torch.amp.GradScaler('cuda', enabled=use_scaler)

    dl_tr, val_loaders = make_loaders(df_train, model.tok, cfg, 
                                      batch_size=batch_size, prompt=stage_prompt, 
                                      val_sets=val_sets)

    best_loss = float('inf')
    best_stats = {}
    bad_epochs = 0
    train_losses: List[float] = []
    val_losses_per_set: Dict[str, List[float]] = {k: [] for k in val_loaders.keys()}

    for ep in range(1, epochs + 1):
        if unfreeze_schedule_k and ep == unfreeze_schedule_k[0]:
            model.freeze_gpt2_except_last_k(unfreeze_schedule_k[1])
            print(f'[{stage_name}] unfreezed last_k={unfreeze_schedule_k[1]} | trainable={_count_trainable(model):,}')

        t0 = time.time()
        model.train()
        total_loss, n_items = 0.0, 0
        start_loss = None
        progress_bar = tqdm(dl_tr, desc=f'[{stage_name}] train ep {ep}/{epochs}', leave=False)

        for paths, inp, lbl in progress_bar:
            if not paths:
                continue
            opt.zero_grad(set_to_none=True)
            if amp_ok:
                with torch.autocast(device_type='cuda', dtype=amp_dtype):
                    loss = model.compute_loss(paths, inp, lbl)
                if use_scaler:
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(params, cfg.clip_grad_norm)
                    scaler.step(opt)
                    scheduler.step()
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(params, cfg.clip_grad_norm)
                    opt.step()
                    scheduler.step()
            else:
                loss = model.compute_loss(paths, inp, lbl)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, cfg.clip_grad_norm)
                opt.step()
                scheduler.step()

            total_loss += loss.item() * len(paths)
            n_items += len(paths)
            if start_loss is None:
                start_loss = loss.item()
                progress_bar.set_description(f'[{stage_name}] train ep {ep}/{epochs} (start_loss={start_loss:.4f})')
            progress_bar.set_postfix(loss=loss.item())

        tr_loss = total_loss / max(1, n_items)
        train_losses.append(tr_loss)

        model.eval()
        metrics_by_set = {}
        for vname, dl_va in val_loaders.items():
            total_loss, n_items = 0.0, 0
            hyps_all, refs_all, paths_all = [], [], []
            with torch.inference_mode():
                val_start_loss = None
                val_progress = tqdm(dl_va, desc=f'[{stage_name}] valid[{vname}] ep {ep}/{epochs}', leave=False)
                for paths, inp, lbl in val_progress:
                    if not paths:
                        continue
                    if amp_ok:
                        with torch.autocast(device_type='cuda', dtype=amp_dtype):
                            loss = model.compute_loss(paths, inp, lbl)
                    else:
                        loss = model.compute_loss(paths, inp, lbl)
                    total_loss += loss.item() * len(paths)
                    n_items += len(paths)
                    if val_start_loss is None:
                        val_start_loss = loss.item()
                        val_progress.set_description(f'[{stage_name}] valid[{vname}] ep {ep}/{epochs} (start={val_start_loss:.4f})')
                    val_progress.set_postfix(loss=loss.item())

                    hyps = model.generate(paths, prompt=stage_prompt)
                    refs = model.tok.batch_decode(lbl["input_ids"], skip_special_tokens=True)
                    hyps_all.extend(hyps); refs_all.extend(refs); paths_all.extend(paths)

            v_loss = total_loss / max(1, n_items)
            val_losses_per_set[vname].append(v_loss)
            ppl = math.exp(v_loss) if v_loss < 20 else float('inf')
            text_metrics = compute_text_metrics(hyps_all, refs_all, model.clip_backend, paths_all)
            metrics_by_set[vname] = (v_loss, ppl, text_metrics)

        val_loss_primary, ppl_primary, text_metrics_primary = metrics_by_set[primary_val_name]
        took = time.time() - t0

        msg = [f'[{stage_name}] ep {ep:02d}/{epochs} | train_loss={tr_loss:.4f} | {took:.1f}s']
        for vname, (v_loss, ppl, tm) in metrics_by_set.items():
            msg.append(f'[{vname}] val={v_loss:.4f} PPL={ppl:.2f} '
                       f'BLEU={tm.get("BLEU",0):.1f} ROUGE-L={tm.get("ROUGE_L",0):.3f} '
                       f'CIDEr-D={tm.get("CIDEr_D",0):.2f} CLIP={tm.get("CLIP",0):.1f} '
                       f'BERT_F1={tm.get("BERT_F1",0):.3f}')
        print(' | '.join(msg))

        if val_loss_primary < best_loss - 1e-4:
            best_loss = val_loss_primary
            best_stats = {'val_loss': val_loss_primary, 'ppl': ppl_primary, 
                          **text_metrics_primary}
            torch.save({'model': model.state_dict(), 'cfg': asdict(cfg)}, 
                       out_ckpt)
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                print(f'[{stage_name}] early stop at epoch {ep}')
                break

    cfg.label_smoothing = prev_smooth
    print(f'[{stage_name}] best: val_loss={best_stats.get("val_loss", 0):.4f} | '
          f'PPL={best_stats.get("ppl", 0):.2f} | '
          f'BLEU={best_stats.get("BLEU", 0.0):.1f} | '
          f'ROUGE-L={best_stats.get("ROUGE_L", 0.0):.3f} | '
          f'CIDEr-D={best_stats.get("CIDEr_D", 0.0):.2f} | '
          f'CLIP={best_stats.get("CLIP", 0.0):.1f} | '
          f'BERT_F1={best_stats.get("BERT_F1", 0.0):.3f}')
    return {'train_losses': train_losses, 'val_losses': val_losses_per_set}


def main():
    os.makedirs(cfg.OUT_DIR, exist_ok=True)
    
    # Настраиваем логирование в файл
    log_file = os.path.join(cfg.OUT_DIR, f"train_log_{APPROACH_NAME}_{int(time.time())}.txt")
    sys.stdout = TeeLogger(log_file, mode='w', filter_progress=True)
    
    device = cfg.device if torch.cuda.is_available() else 'cpu'
    print(f"[device] {device} | GPT-2={cfg.GPT2_NAME} | CLIP_PRESET={cfg.CLIP_PRESET} | Log: {log_file}")

    model = DermCaptioner(cfg, device)

    # 1) Сплиты и объединение
    tr, va, te = load_df(cfg)
    df_all = pd.concat([tr, va, te], axis=0, ignore_index=True)

    # 2) Две валидации
    global_val = make_global_validation(df_all, cfg)
    print(f'[GLOBAL VAL] prepared N={len(global_val)} (stratified by source)')

    sk_hold = subset_by_source(te, cfg, cfg.SKINCAP_TAG)
    if len(sk_hold) < 50:
        sk_hold = subset_by_source(va, cfg, cfg.SKINCAP_TAG)
    print(f'[SKINCAP HOLDOUT] N={len(sk_hold)}')

    # 3) Исключаем ОБЕ валидации из train (без утечки)
    gv_paths = set(global_val[cfg.COL_IMG].astype(str)) if len(global_val) else set()
    sk_paths = set(sk_hold[cfg.COL_IMG].astype(str)) if len(sk_hold) else set()
    excluded_paths = gv_paths | sk_paths

    df_all_train = df_all[~df_all[cfg.COL_IMG].astype(str).isin(excluded_paths)].reset_index(drop=True)

    print(f"[SPLITS] train_common={len(df_all_train)} | global_val={len(global_val)} | skincap_holdout={len(sk_hold)}")
    if len(df_all_train) == 0:
        raise ValueError("Empty train after excluding GLOBAL ∪ SKINCAP_HOLDOUT")
    if len(global_val) == 0:
        raise ValueError("Empty GLOBAL validation")
    if len(sk_hold) == 0:
        raise ValueError("Empty SKINCAP_HOLDOUT validation")

    tr_A = df_all_train
    ckptA = Path(cfg.OUT_DIR) / 'best_stageA.pt'

    A_vals = {
        'GLOBAL': global_val,
        'SKINCAP_HOLDOUT': sk_hold,
    }
    stats_A = train_stage(
        'Stage A (META)', model, tr_A, A_vals,
        primary_val_name='GLOBAL',
        epochs=cfg.epochs_A, lr=cfg.lr_A, batch_size=cfg.batch_size_A,
        unfreeze_k=cfg.unfreeze_last_k_layers_A, out_ckpt=ckptA, cfg=cfg,
        prompt_override=cfg.prompt_A,
        label_smoothing_override=cfg.label_smoothing_A,
        weight_decay_override=cfg.wd_A,
        freeze_prefix=False,
        unfreeze_schedule_k=None
    )

    if ckptA.exists():
        sd = torch.load(ckptA, map_location='cpu')
        model.load_state_dict(sd['model'], strict=False)

    # Учим на SkinCAP-части только из уже очищенного train
    tr_B = subset_by_source(df_all_train, cfg, cfg.SKINCAP_TAG)
    if len(tr_B) < 50:
        tr_B = df_all_train

    ckptB = Path(cfg.OUT_DIR) / 'best_stageB.pt'

    B_vals = {
        'GLOBAL': global_val,
        'SKINCAP_HOLDOUT': sk_hold,
    }
    stats_B = train_stage(
        'Stage B (SKINCAP)', model, tr_B, B_vals,
        primary_val_name='SKINCAP_HOLDOUT',
        epochs=cfg.epochs_B, lr=cfg.lr_B, batch_size=cfg.batch_size_B,
        unfreeze_k=cfg.unfreeze_last_k_layers_B, out_ckpt=ckptB, cfg=cfg,
        prompt_override=cfg.prompt_B,
        label_smoothing_override=cfg.label_smoothing_B,
        weight_decay_override=cfg.wd_B,
        freeze_prefix=cfg.freeze_prefix_in_B,
        unfreeze_schedule_k=cfg.unfreeze_schedule_B
    )

    cfg.prompt = cfg.prompt_B


    if ckptB.exists():
        sd = torch.load(ckptB, map_location='cpu')
        model.load_state_dict(sd['model'], strict=False)

    final_dir = Path(cfg.OUT_DIR)
    final_dir.mkdir(parents=True, exist_ok=True)
    model_tag = cfg.GPT2_NAME.replace('/', '-')
    clip_tag  = model.clip_backend.arch.replace('/', '-')
    final_ckpt = final_dir / f'final_captioner_{model_tag}_{clip_tag}.pt'
    final_json = final_dir / f'final_captioner_{model_tag}_{clip_tag}.json'

    model_cpu = DermCaptioner(cfg, device='cpu')
    model_cpu.load_state_dict(model.state_dict(), strict=True)

    torch.save({
        'model': model_cpu.state_dict(),
        'cfg': asdict(cfg),
        'meta': {
            'gpt2_name': cfg.GPT2_NAME,
            'clip_backend_kind': model.clip_backend.kind,
            'clip_repo': model.clip_backend.repo,
            'clip_arch': model.clip_backend.arch,
            'clip_weight_path': model.clip_backend.weight_path,
            'embed_dim': model.clip_backend.embed_dim,
            'prefix_tokens': cfg.prefix_tokens,
            'prompt': cfg.prompt_B,
            'cliche_phrases': list(cfg.cliche_phrases),
        }
    }, final_ckpt)

    with open(final_json, 'w', encoding='utf-8') as f:
        json.dump({
            'gpt2_name': cfg.GPT2_NAME,
            'clip_backend_kind': model.clip_backend.kind,
            'clip_repo': model.clip_backend.repo,
            'clip_arch': model.clip_backend.arch,
            'clip_weight_path': model.clip_backend.weight_path,
            'embed_dim': model.clip_backend.embed_dim,
            'prefix_tokens': cfg.prefix_tokens,
            'prompt': cfg.prompt_B,
            'cliche_phrases': list(cfg.cliche_phrases),
            'columns': {
                'image_path': cfg.COL_IMG,
                'caption': cfg.COL_TXT,
                'split': cfg.COL_SPLIT,
                'source': cfg.COL_SRC,
            }
        }, f, ensure_ascii=False, indent=2)

    print(f"[save] Final model saved to: {final_ckpt}")
    print(f"[save] Inference config saved to: {final_json}")

    try:
        A_train = stats_A['train_losses']
        B_train = stats_B['train_losses']
        A_val = stats_A['val_losses']['GLOBAL'] if 'GLOBAL' in stats_A['val_losses'] else []
        B_val = stats_B['val_losses']['SKINCAP_HOLDOUT'] if 'SKINCAP_HOLDOUT' in stats_B['val_losses'] else []

        x_A = list(range(1, len(A_train)+1))
        x_B = list(range(len(A_train)+1, len(A_train)+len(B_train)+1))

        plt.figure(figsize=(7, 10))  # Узкая вертикальная фигура
        plt.rcParams.update({'font.size': 14})  # Увеличиваем размер шрифта
        
        plt.plot(x_A, A_train, label='Train (A)', linewidth=2.5)
    
        if A_val:
            plt.plot(x_A, A_val, label='Val GLOBAL (A)', linewidth=2.5)
        plt.plot(x_B, B_train, label='Train (B)', linewidth=2.5)
        if B_val:
            plt.plot(x_B, B_val, label='Val SKINCAP (B)', linewidth=2.5)

        if x_B:
            boundary = x_B[0] - 0.5
            plt.axvline(boundary, linestyle='--', linewidth=2.0, color='gray')
            plt.text(boundary+0.1, plt.ylim()[1]*0.95, 'Stage B', fontsize=16, weight='bold')

        title = f"Loss A→B | CLIP={cfg.CLIP_PRESET}"
        plt.title(title, fontsize=18, weight='bold', pad=20)
        plt.xlabel('Epoch (continuous)', fontsize=16, weight='bold')
        plt.ylabel('Loss', fontsize=16, weight='bold')
        
        all_x = x_A + x_B if x_B else x_A
        if all_x:
            plt.xticks(all_x, fontsize=14)
        plt.yticks(fontsize=14)
        
        plt.legend(fontsize=14, loc='upper right')
        plt.grid(True, alpha=0.3)
        out_plot = Path(cfg.OUT_DIR) / cfg.loss_plot_name
        plt.tight_layout()
        plt.savefig(out_plot, dpi=160)
        plt.close()
        print(f'[plot] saved loss curve -> {out_plot}')
    except Exception as e:
        print(f'[plot] failed: {e}')

    def _show_examples(name: str, df_src: pd.DataFrame, k: int):
        if len(df_src) == 0: 
            return
        k = min(k, len(df_src))
        samp = df_src.sample(n=k, random_state=cfg.seed)
        img_paths = [str(p) for p in samp[cfg.COL_IMG].tolist()]
        refs = [str(t) for t in samp[cfg.COL_TXT].tolist()]
        hyps = model.generate(img_paths, prompt=cfg.prompt_B)
        print(f'\n[Examples @ {name}]')
        for i, (p, h, r) in enumerate(zip(img_paths, hyps, refs), 1):
            print(f'  #{i} path={p}')
            print(f'     pred: {h}')
            print(f'     true: {r}')

    _show_examples('GLOBAL', global_val, cfg.examples_to_show)
    _show_examples('SKINCAP_HOLDOUT', sk_hold, cfg.examples_to_show)
    print('[done]')


if __name__ == '__main__':
    main()
