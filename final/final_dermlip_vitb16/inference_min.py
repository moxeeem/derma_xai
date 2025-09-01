# -*- coding: utf-8 -*-
from __future__ import annotations
import os, json, re
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

try:
    import open_clip
    HAS_OPENCLIP = True
except Exception:
    HAS_OPENCLIP = False

from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    CLIPImageProcessor as HFCLIPImageProcessor,
    CLIPModel as HFCLIPModel,
)

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
        y = self.fc2(y).view(x.size(0), self.tokens, -1)
        y = self.ln(y)
        y = self.drop(self.alpha * y)
        return y

class CLIPBackend:
    def __init__(self, repo_or_kind: str, device: str):
        self.device = device
        self.repo_or_kind = repo_or_kind
        
        # Определяем тип модели
        if 'BiomedCLIP' in repo_or_kind or 'microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224' in repo_or_kind:
            # BiomedCLIP через open_clip
            assert HAS_OPENCLIP, "open_clip is required for BiomedCLIP"
            if not repo_or_kind.startswith('microsoft/'):
                repo_or_kind = 'microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
            model_name = f'hf-hub:{repo_or_kind}'
            self.model, self.preprocess, _ = open_clip.create_model_and_transforms(model_name)
            self.model = self.model.to(device).eval()
            self.kind = "open_clip"
            self.processor = None
        elif "/" in repo_or_kind and 'pubmed-clip' in repo_or_kind:
            # PubMedCLIP через HF
            self.model = HFCLIPModel.from_pretrained(repo_or_kind).to(device).eval()
            self.processor = HFCLIPImageProcessor.from_pretrained(repo_or_kind)
            self.kind = "hf_clip"
            self.preprocess = None
        elif "/" in repo_or_kind or repo_or_kind.startswith('redlessone/'):
            # DermLIP через open_clip
            assert HAS_OPENCLIP, "open_clip is required for DermLIP"
            model_name = f"hf-hub:{repo_or_kind}"
            self.model, self.preprocess, _ = open_clip.create_model_and_transforms(model_name)
            self.model = self.model.to(device).eval()
            self.kind = "open_clip"
            self.processor = None
        else:
            # Fallback для других моделей, включая случаи когда передается просто тип модели
            try:
                # Пытаемся определить по названию
                if 'biomedclip' in repo_or_kind.lower() or 'biomed' in repo_or_kind.lower():
                    assert HAS_OPENCLIP, "open_clip is required for BiomedCLIP"
                    model_name = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
                    self.model, self.preprocess, _ = open_clip.create_model_and_transforms(model_name)
                    self.model = self.model.to(device).eval()
                    self.kind = "open_clip"
                    self.processor = None
                elif 'dermlip' in repo_or_kind.lower():
                    assert HAS_OPENCLIP, "open_clip is required for DermLIP"
                    model_name = "hf-hub:redlessone/DermLIP_ViT-B-16"
                    self.model, self.preprocess, _ = open_clip.create_model_and_transforms(model_name)
                    self.model = self.model.to(device).eval()
                    self.kind = "open_clip"
                    self.processor = None
                elif 'pubmed' in repo_or_kind.lower():
                    # PubMedCLIP через HF
                    repo_name = "flaviagiammarino/pubmed-clip-vit-base-patch32"
                    self.model = HFCLIPModel.from_pretrained(repo_name).to(device).eval()
                    self.processor = HFCLIPImageProcessor.from_pretrained(repo_name)
                    self.kind = "hf_clip"
                    self.preprocess = None
                else:
                    raise ValueError(f"Unknown model type: {repo_or_kind}")
            except Exception as e:
                # Последняя попытка - попробовать как HF модель
                try:
                    self.model = HFCLIPModel.from_pretrained(repo_or_kind).to(device).eval()
                    self.processor = HFCLIPImageProcessor.from_pretrained(repo_or_kind)
                    self.kind = "hf_clip"
                    self.preprocess = None
                except:
                    raise ValueError(f"Failed to load model {repo_or_kind}: {e}")
                
        # Определяем размер эмбеддинга
        if self.kind == "open_clip":
            with torch.no_grad():
                img = Image.new('RGB', (224, 224), color=0)
                x = self.preprocess(img).unsqueeze(0).to(device)
                feat = self.model.encode_image(x)
            self.embed_dim = int(feat.shape[-1])
        else:
            self.embed_dim = int(self.model.config.projection_dim)
            
    @torch.inference_mode()
    def encode_images(self, paths: List[str]) -> torch.Tensor:
        ims = []
        if self.kind == "open_clip":
            for p in paths:
                try:
                    im = Image.open(p).convert("RGB")
                except:
                    im = Image.new("RGB", (224, 224), color=0)
                ims.append(self.preprocess(im))
            x = torch.stack(ims).to(self.device)
            f = self.model.encode_image(x)
        else:
            # HF CLIP (PubMedCLIP)
            for p in paths:
                try:
                    im = Image.open(p).convert("RGB")
                except:
                    im = Image.new("RGB", (224, 224), color=0)
                ims.append(im)
            proc = self.processor(images=ims, return_tensors='pt')
            x = proc['pixel_values'].to(self.device)
            f = self.model.get_image_features(pixel_values=x)
        return F.normalize(f, dim=-1)

class Captioner(nn.Module):
    def __init__(self, gpt2_name: str, clip_repo: str, prefix_tokens: int, prompt: str, device: str):
        super().__init__()
        self.device = device
        self.prompt = prompt
        self.tok = AutoTokenizer.from_pretrained(gpt2_name)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.gpt2 = AutoModelForCausalLM.from_pretrained(gpt2_name).to(device).eval()
        self.clip = CLIPBackend(clip_repo, device)
        self.prefix = PrefixProjector(self.clip.embed_dim, int(self.gpt2.config.n_embd), prefix_tokens).to(device).eval()
        
    @torch.inference_mode()
    def generate(self, img_paths: List[str], prompt: Optional[str] = None) -> List[str]:
        pr = prompt or self.prompt or ""
        f = self.clip.encode_images(img_paths)
        pref = self.prefix(f)
        ids = self.tok([pr]*pref.size(0), return_tensors='pt', padding=True, truncation=True).to(self.device)
        emb_prompt = self.gpt2.transformer.wte(ids['input_ids'])
        inputs_embeds = torch.cat([pref, emb_prompt], dim=1)
        attn = torch.ones(inputs_embeds.size()[:-1], dtype=torch.long, device=self.device)
        gen = self.gpt2.generate(
            inputs_embeds=inputs_embeds, attention_mask=attn,
            max_new_tokens=60, min_new_tokens=24, num_beams=4,
            no_repeat_ngram_size=4, repetition_penalty=1.15, length_penalty=0.6,
            pad_token_id=self.tok.eos_token_id, eos_token_id=self.tok.eos_token_id, early_stopping=True
        )
        outs = self.tok.batch_decode(gen, skip_special_tokens=True)
        res = []
        for s in outs:
            cut = s.find(pr)
            if cut >= 0: s = s[cut+len(pr):]
            res.append(s.strip())
        return res

def load_model(repo_dir: str | os.PathLike) -> Captioner:
    repo_dir = Path(repo_dir)
    cfgs = sorted(repo_dir.glob("final_captioner_*.json"))
    if not cfgs:
        raise FileNotFoundError("final_captioner_*.json not found in repo snapshot")
    data = json.loads(cfgs[-1].read_text(encoding='utf-8'))
    gpt2 = data.get("gpt2_name", "gpt2-medium")
    
    # Определяем CLIP репозиторий с поддержкой TimmModel
    clip_repo = data.get("clip_weight_path", data.get("clip_repo", data.get("clip_backend_kind", "")))
    
    # Если информация о CLIP не найдена в JSON, пытаемся определить по имени файла
    if not clip_repo or clip_repo in ["open_clip", "hf_clip"]:
        ckpts = sorted(repo_dir.glob("final_captioner_*.pt"))
        if ckpts:
            ckpt_name = str(ckpts[-1])
            if "TimmModel" in ckpt_name:
                clip_repo = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
            elif "VisionTransformer" in ckpt_name:
                clip_repo = "redlessone/DermLIP_ViT-B-16"
            elif "CLIPModel" in ckpt_name:
                clip_repo = "flaviagiammarino/pubmed-clip-vit-base-patch32"
            elif "biomedclip" in ckpt_name.lower():
                clip_repo = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    
    prefix_tokens = int(data.get("prefix_tokens", 32))
    prompt = data.get("prompt", "Describe the skin lesion.")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Captioner(gpt2, clip_repo, prefix_tokens, prompt, device).to(device).eval()
    # подгрузим state_dict
    ckpts = sorted(repo_dir.glob("final_captioner_*.pt"))
    if not ckpts:
        raise FileNotFoundError("final_captioner_*.pt not found in repo snapshot")
    state = torch.load(ckpts[-1], map_location="cpu")
    sd = state.get("model", state)
    model.load_state_dict(sd, strict=False)
    return model

def generate(model: Captioner, img_paths: List[str], prompt: Optional[str] = None) -> List[str]:
    return model.generate(img_paths, prompt=prompt)
