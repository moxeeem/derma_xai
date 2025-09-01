# xai.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import os, json, argparse, importlib, logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Подавляем предупреждения от open_clip
logging.getLogger('root').setLevel(logging.ERROR)

import pandas as pd
from PIL import Image
from tqdm import tqdm

# ----------------------- Core models (shared) -----------------------

@dataclass
class RuntimeCfg:
    csv_path: str
    img_root: str
    col_img: str
    col_txt: str
    col_split: str
    ckpt_path: str
    gpt2_name: str
    clip_repo: str
    device: str
    seed: int
    prompt: str
    output_dir: str
    num_samples: int
    methods: List[str]
    method_params: Dict[str, Dict[str, Any]]

def set_seeds(seed: int):
    np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class PrefixProjector(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, tokens: int = 32, p_drop: float = 0.05):
        super().__init__()
        hidden = max(512, out_dim * 2)
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, out_dim * tokens)
        self.ln  = nn.LayerNorm(out_dim)
        self.tokens = tokens
        self.drop = nn.Dropout(p_drop)
        self.alpha = nn.Parameter(torch.tensor(0.5))
        nn.init.xavier_uniform_(self.fc1.weight, gain=1.0); nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight, gain=0.5); nn.init.zeros_(self.fc2.bias)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.gelu(self.fc1(x)); y = self.fc2(y)
        y = y.view(y.size(0), self.tokens, -1); y = self.ln(y); y = self.drop(self.alpha * y)
        return y

# transformers / CLIP / open_clip
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    CLIPModel, CLIPImageProcessor, CLIPTokenizer,
)

try:
    import open_clip
    HAS_OPENCLIP = True
except ImportError:
    HAS_OPENCLIP = False

class CLIPBackbone:
    """Wrapper expected by methods: model, processor, tokenizer, device, image_size, grid; encode_image/text."""
    def __init__(self, repo: str, device: str):
        self.device = device
        self.repo = repo
        
        if 'BiomedCLIP' in repo or 'microsoft/BiomedCLIP' in repo:
            # BiomedCLIP через open_clip с правильной загрузкой весов
            assert HAS_OPENCLIP, "open_clip is required for BiomedCLIP"
            if not repo.startswith('microsoft/'):
                repo = 'microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
            # корректная загрузка весов из HF Hub
            try:
                # пробуем новый API create_model_from_pretrained
                self.model, self.preprocess = open_clip.create_model_from_pretrained(f"hf-hub:{repo}")
            except (AttributeError, TypeError, RuntimeError):
                try:
                    # fallback: старый API с model_name и pretrained
                    arch = "ViT-B-16" if "16" in repo else "ViT-B-32"  
                    self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                        model_name=arch, pretrained=f"hf-hub:{repo}"
                    )
                except (TypeError, RuntimeError):
                    # последний fallback: загружаем архитектуру и веса отдельно
                    arch = "ViT-B-16" if "16" in repo else "ViT-B-32"
                    self.model, _, self.preprocess = open_clip.create_model_and_transforms(arch)
                    # Пытаемся загрузить веса вручную
                    try:
                        checkpoint = open_clip.download_pretrained(f"hf-hub:{repo}")
                        self.model.load_state_dict(checkpoint, strict=False)
                    except Exception:
                        print(f"Warning: Could not load pretrained weights for {repo}, using random weights")
            self.model = self.model.to(device).eval()
            self.processor = None
            self.tokenizer = None
            self.kind = "open_clip"
            # Получаем размеры изображения из конфигурации
            self.image_size = 224  # для ViT-B-16
            self.patch_size = 16
            self.grid = self.image_size // self.patch_size
        elif 'redlessone/DermLIP' in repo or 'DermLIP' in repo:
            # DermLIP через open_clip с правильной загрузкой весов
            assert HAS_OPENCLIP, "open_clip is required for DermLIP"
            try:
                # пробуем новый API create_model_from_pretrained
                self.model, self.preprocess = open_clip.create_model_from_pretrained(f"hf-hub:{repo}")
            except (AttributeError, TypeError, RuntimeError):
                try:
                    # fallback: старый API с model_name и pretrained
                    arch = "ViT-B-16" if "16" in repo else "ViT-B-32"  
                    self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                        model_name=arch, pretrained=f"hf-hub:{repo}"
                    )
                except (TypeError, RuntimeError):
                    # последний fallback: загружаем архитектуру и веса отдельно
                    arch = "ViT-B-16" if "16" in repo else "ViT-B-32"
                    self.model, _, self.preprocess = open_clip.create_model_and_transforms(arch)
                    # Пытаемся загрузить веса вручную
                    try:
                        checkpoint = open_clip.download_pretrained(f"hf-hub:{repo}")
                        self.model.load_state_dict(checkpoint, strict=False)
                    except Exception:
                        print(f"Warning: Could not load pretrained weights for {repo}, using random weights")
            self.model = self.model.to(device).eval()
            self.processor = None
            self.tokenizer = None
            self.kind = "open_clip"
            # Получаем размеры изображения из конфигурации
            self.image_size = 224  # для ViT-B-16
            self.patch_size = 16
            self.grid = self.image_size // self.patch_size
        else:
            # Стандартный HF CLIP
            try:
                self.model: CLIPModel = CLIPModel.from_pretrained(repo, attn_implementation="eager")
            except TypeError:
                self.model: CLIPModel = CLIPModel.from_pretrained(repo)
                try: self.model.config.attn_implementation = "eager"
                except Exception: pass
            self.model = self.model.to(device).eval()
            self.processor = CLIPImageProcessor.from_pretrained(repo)
            self.tokenizer = CLIPTokenizer.from_pretrained(repo)
            self.kind = "hf_clip"
            self.preprocess = None
            conf = self.model.vision_model.config
            self.image_size = conf.image_size
            self.patch_size = conf.patch_size
            self.grid = self.image_size // self.patch_size

    @torch.inference_mode()
    def encode_image(self, pil_images: List[Image.Image]) -> torch.Tensor:
        if self.kind == "open_clip":
            # BiomedCLIP и другие open_clip модели
            ims = [self.preprocess(img) for img in pil_images]
            px = torch.stack(ims).to(self.device)
            feats = self.model.encode_image(px)
        else:
            # HF CLIP
            px = self.processor(images=pil_images, return_tensors='pt')['pixel_values'].to(self.device)
            feats = self.model.get_image_features(pixel_values=px)
        return F.normalize(feats, dim=-1)

    @torch.inference_mode()
    def encode_text(self, texts: List[str]) -> torch.Tensor:
        if self.kind == "open_clip":
            # BiomedCLIP и другие open_clip модели
            if 'BiomedCLIP' in self.repo:
                # BiomedCLIP использует специальный токенизатор
                tok = open_clip.get_tokenizer(f'hf-hub:{self.repo}')
            else:
                # DermLIP и другие модели
                tok = open_clip.get_tokenizer(f'hf-hub:{self.repo}')
            txt_tokens = tok(texts).to(self.device)
            feats = self.model.encode_text(txt_tokens)
        else:
            # HF CLIP
            tok = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=77).to(self.device)
            feats = self.model.get_text_features(**tok)
        return F.normalize(feats, dim=-1)

class DermCaptioner(nn.Module):
    """Prefix-conditioned GPT-2 captioner compatible with your training checkpoint."""
    def __init__(self, clip: CLIPBackbone, device: str, gpt2_name: str = 'gpt2-medium',
                 prefix_tokens: int = 32, prefix_dropout_p: float = 0.05):
        super().__init__()
        self.device = device
        self.clip = clip
        
        # Очистка перед загрузкой токенизатора
        import gc
        torch.cuda.empty_cache()
        gc.collect()
        
        self.tok = AutoTokenizer.from_pretrained(gpt2_name)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
            
        # Очистка перед загрузкой GPT-2
        torch.cuda.empty_cache()
        gc.collect()
        
        self.gpt2 = AutoModelForCausalLM.from_pretrained(gpt2_name).to(device).eval()
        self.txt_dim = int(self.gpt2.config.n_embd)
        
        # Получаем embed_dim в зависимости от типа модели
        if clip.kind == "open_clip":
            # Для open_clip моделей пробуем получить размер через dummy input
            dummy_img = Image.new('RGB', (224, 224), color=0)
            with torch.no_grad():
                dummy_feat = clip.encode_image([dummy_img])
            self.img_dim = int(dummy_feat.shape[-1])
            # Очистка dummy tensor
            del dummy_feat
            torch.cuda.empty_cache()
        else:
            # Для HF CLIP
            self.img_dim = int(self.clip.model.config.projection_dim)
            
        self.prefix = PrefixProjector(self.img_dim, self.txt_dim, tokens=prefix_tokens,
                                      p_drop=prefix_dropout_p).to(device).eval()
        self.prompt: Optional[str] = None  # set from config
        
        # Финальная очистка
        torch.cuda.empty_cache()
        gc.collect()

    def load_ckpt(self, ckpt_path: str):
        # Очистка перед загрузкой checkpoint
        import gc
        torch.cuda.empty_cache()
        gc.collect()
        
        sd = torch.load(ckpt_path, map_location='cpu')
        self.load_state_dict(sd['model'], strict=False)
        self.eval()
        
        # Очистка после загрузки
        del sd
        torch.cuda.empty_cache()
        gc.collect()

    @torch.inference_mode()
    def generate(self, pil_images: List[Image.Image], prompt: Optional[str] = None, gen_cfg: Dict = None) -> List[str]:
        if gen_cfg is None: gen_cfg = {}
        if prompt is None: prompt = self.prompt or "Describe the image."
        
        # Очистка перед генерацией
        import gc
        torch.cuda.empty_cache()
        
        # Улучшенные параметры генерации для стабильности
        img_feats = self.clip.encode_image(pil_images)     # [B,D]
        pref = self.prefix(img_feats)                      # [B,P,txt_dim]
        prompt_ids = self.tok([prompt]*pref.size(0), return_tensors='pt', padding=True, truncation=True).to(self.device)
        prompt_emb = self.gpt2.transformer.wte(prompt_ids['input_ids'])    # [B,Tp,txt_dim]
        inputs_embeds = torch.cat([pref, prompt_emb], dim=1)
        attn = torch.ones(inputs_embeds.size()[:-1], dtype=torch.long, device=self.device)
        
        try:
            # Более консервативные параметры для стабильной генерации
            gen = self.gpt2.generate(
                inputs_embeds=inputs_embeds, 
                attention_mask=attn,
                max_new_tokens=min(gen_cfg.get('gen_max_new', 50), 80),
                min_new_tokens=gen_cfg.get('min_new_tokens', 15),
                num_beams=gen_cfg.get('gen_beams', 3),
                do_sample=False, 
                no_repeat_ngram_size=gen_cfg.get('no_repeat_ngram_size', 3),
                repetition_penalty=gen_cfg.get('repetition_penalty', 1.1),
                length_penalty=gen_cfg.get('length_penalty', 0.8),
                bad_words_ids=None,
                pad_token_id=self.tok.eos_token_id, 
                eos_token_id=self.tok.eos_token_id,
                early_stopping=True
            )
        finally:
            # Очистка промежуточных tensors
            del img_feats, pref, prompt_ids, prompt_emb, inputs_embeds, attn
            torch.cuda.empty_cache()
            gc.collect()
            
        outs = self.tok.batch_decode(gen, skip_special_tokens=True)
        res = []
        for s in outs:
            cut = s.find(prompt)
            if cut >= 0: s = s[cut + len(prompt):]
            s = s.strip()
            # Ограничиваем длину для стабильности
            if len(s) > 200:
                s = s[:200].rsplit(' ', 1)[0] + '...'
            res.append(s)
            
        # Очистка результатов генерации
        del gen, outs
        torch.cuda.empty_cache()
        gc.collect()
        
        return res

# ----------------------- IO helpers -----------------------

def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)

def load_csv_samples(csv_path: str, img_col: str, split_col: str, img_root: str,
                     max_n: int, ref_col: Optional[str] = None) -> List[Tuple[str, Optional[str]]]:
    df = pd.read_csv(csv_path)
    if split_col in df.columns:
        df = df[df[split_col].astype(str).str.lower().isin(['test','val'])].reset_index(drop=True)
    df = df.head(max_n).copy()
    out = []
    for _, row in df.iterrows():
        p = str(row[img_col]); ref = str(row[ref_col]) if (ref_col and (ref_col in df.columns)) else None
        if img_root and not os.path.isabs(p): p = os.path.join(img_root, p)
        out.append((p, ref))
    return out

def load_config(path: str) -> RuntimeCfg:
    with open(path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    if cfg.get("device", "") not in ("cuda", "cpu"):
        cfg["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    return RuntimeCfg(
        csv_path=cfg["csv_path"],
        img_root=cfg.get("img_root",""),
        col_img=cfg.get("col_img","img_path"),
        col_txt=cfg.get("col_txt","caption"),
        col_split=cfg.get("col_split","split"),
        ckpt_path=cfg["ckpt_path"],
        gpt2_name=cfg.get("gpt2_name","gpt2-medium"),
        clip_repo=cfg["clip_repo"],
        device=cfg["device"],
        seed=int(cfg.get("seed",42)),
        prompt=cfg["prompt"],
        output_dir=cfg["output_dir"],
        num_samples=int(cfg.get("num_samples",10)),
        methods=list(cfg["methods"]),
        method_params=cfg.get("method_params", {}),
    )

# ----------------------- Orchestrator -----------------------

def main():
    parser = argparse.ArgumentParser(description="XAI runner")
    parser.add_argument("--config", type=str, default="/root/clip_xai/final/xai_config.json", help="Path to JSON config")
    args = parser.parse_args()

    rcfg = load_config(args.config)
    set_seeds(rcfg.seed)
    ensure_dir(Path(rcfg.output_dir))

    # Агрессивная очистка памяти перед загрузкой моделей
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        # Сброс накопленной памяти
        try:
            torch.cuda.reset_peak_memory_stats()
        except:
            pass

    # Models
    print(f"[XAI] Loading CLIP model: {rcfg.clip_repo}")
    clip = CLIPBackbone(repo=rcfg.clip_repo, device=rcfg.device)
    
    # Очистка после загрузки CLIP
    torch.cuda.empty_cache()
    gc.collect()
    
    print(f"[XAI] Loading captioner model: {rcfg.gpt2_name}")
    cap  = DermCaptioner(clip=clip, device=rcfg.device, gpt2_name=rcfg.gpt2_name)
    
    # Очистка после создания captioner
    torch.cuda.empty_cache()
    gc.collect()
    
    print(f"[XAI] Loading checkpoint: {rcfg.ckpt_path}")
    cap.load_ckpt(rcfg.ckpt_path)
    cap.prompt = rcfg.prompt
    
    # Финальная очистка после загрузки checkpoint
    torch.cuda.empty_cache()
    gc.collect()

    # Data
    samples = load_csv_samples(
        csv_path=rcfg.csv_path, img_col=rcfg.col_img, split_col=rcfg.col_split,
        img_root=rcfg.img_root, max_n=rcfg.num_samples, ref_col=rcfg.col_txt
    )

    # Run methods
    for method_name in rcfg.methods:
        print(f"[XAI] Running method: {method_name}")
        method_mod = importlib.import_module(f"methods.{method_name}")
        out_dir = Path(rcfg.output_dir) / method_name
        ensure_dir(out_dir)
        cfg_dict = rcfg.method_params.get(method_name, {})

        # each method must provide its own Config dataclass named <Method>Config or expose defaults via **cfg_dict
        cfg_ctor = getattr(method_mod, f"{method_name.title().replace('_', '')}Config", None)
        for i, (img_path, ref_text) in enumerate(tqdm(samples, desc=f"{method_name}")):
            try:
                if cfg_ctor:
                    method_mod.run(clip, cap, image_path=img_path, out_dir=out_dir,
                                   ref_caption=ref_text, config=cfg_ctor(**cfg_dict))
                else:
                    method_mod.run(clip, cap, image_path=img_path, out_dir=out_dir,
                                   ref_caption=ref_text, config=cfg_dict)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                # Агрессивная очистка памяти после ошибки
                torch.cuda.empty_cache()
                gc.collect()
                continue
            
            # Очистка памяти после каждого образца
            if i % 5 == 0:  # Более частая очистка
                torch.cuda.empty_cache()
                gc.collect()

if __name__ == "__main__":
    main()
