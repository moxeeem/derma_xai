from __future__ import annotations
import os, re, json, textwrap, hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

from huggingface_hub import (
    HfApi, HfFolder, create_repo, upload_file, upload_folder,
    whoami, snapshot_download
)

# нужен scope Write
HF_TOKEN = ""
HF_REPO_ID = None
HF_PRIVATE = False
OUT_DIR = None

UPLOAD_GLOBS = [
    "final_captioner_*.pt",
    "best_stageA.pt",
    "best_stageB.pt",
    "final_captioner_*.json",
    "loss_*.png",
]

INFER_FILE_NAME = "inference_min.py"



def _detect_model_type_and_repo_name(out_dir: Path, final_json: Dict) -> Tuple[str, str]:
    clip_preset = final_json.get("clip_backend_kind", "")
    clip_repo = final_json.get("clip_repo", "")
    clip_weight_path = final_json.get("clip_weight_path", "")
    
    pt_files = list(out_dir.glob("final_captioner_*.pt"))
    json_files = list(out_dir.glob("final_captioner_*.json"))
    
    model_type = "unknown"
    repo_suffix = "captioner"
    
    # Сначала проверяем имена файлов - они наиболее надежные
    if any("TimmModel" in str(f) for f in pt_files):
        model_type = "biomedclip"
        repo_suffix = "biomedclip-gpt2-captioner"
    elif any("VisionTransformer" in str(f) for f in pt_files):
        model_type = "dermlip"
        repo_suffix = "dermlip-gpt2-captioner"
    elif any("CLIPModel" in str(f) for f in pt_files):
        model_type = "pubmedclip"
        repo_suffix = "pubmedclip-gpt2-captioner"
    elif any("BiomedCLIP" in str(f) for f in pt_files):
        model_type = "biomedclip"
        repo_suffix = "biomedclip-gpt2-captioner"
    # Проверяем директорию
    elif "biomedclip" in str(out_dir).lower() or "biomed" in str(out_dir).lower():
        model_type = "biomedclip"
        repo_suffix = "biomedclip-gpt2-captioner"
    elif "dermlip" in str(out_dir).lower():
        model_type = "dermlip"
        repo_suffix = "dermlip-gpt2-captioner"
    elif "pubmed" in str(out_dir).lower():
        model_type = "pubmedclip"
        repo_suffix = "pubmedclip-gpt2-captioner"
    # Проверяем JSON конфиг
    elif any("dermlip" in str(f).lower() for f in pt_files + json_files) or \
       "DermLIP" in clip_repo or "dermlip" in clip_weight_path.lower():
        model_type = "dermlip"
        repo_suffix = "dermlip-gpt2-captioner"
    elif any("pubmed" in str(f).lower() for f in pt_files + json_files) or \
         "pubmed-clip" in clip_repo or "pubmed" in clip_weight_path.lower():
        model_type = "pubmedclip"
        repo_suffix = "pubmedclip-gpt2-captioner"
    elif any("biomed" in str(f).lower() for f in pt_files + json_files) or \
         "BiomedCLIP" in clip_repo or "BiomedCLIP" in clip_weight_path:
        model_type = "biomedclip"
        repo_suffix = "biomedclip-gpt2-captioner"
    
    return model_type, repo_suffix


def _get_model_description(model_type: str, final_json: Dict) -> Tuple[str, str, str]:
    clip_repo = final_json.get("clip_repo", final_json.get("clip_weight_path", ""))
    gpt2_name = final_json.get("gpt2_name", "gpt2-medium")
    
    if model_type == "dermlip":
        title = "DermLIP + GPT-2 Dermatology Captioner"
        clip_name = "DermLIP (ViT-B/16)"
        description = (
            "A dermatology image captioning model combining DermLIP vision encoder "
            f"with {gpt2_name} language model. Trained on dermatological images for "
            "generating clinical descriptions of skin lesions."
        )
    elif model_type == "pubmedclip":
        title = "PubMedCLIP + GPT-2 Dermatology Captioner"
        clip_name = "PubMedCLIP (ViT-B/32)"
        description = (
            "A dermatology image captioning model combining PubMedCLIP vision encoder "
            f"with {gpt2_name} language model. Trained on dermatological images for "
            "generating clinical descriptions of skin lesions."
        )
    elif model_type == "biomedclip":
        title = "BiomedCLIP + GPT-2 Dermatology Captioner"
        clip_name = "BiomedCLIP (ViT-B/16)"
        description = (
            "A dermatology image captioning model combining BiomedCLIP vision encoder "
            f"with {gpt2_name} language model. Trained on dermatological images for "
            "generating clinical descriptions of skin lesions."
        )
    else:
        title = "CLIP + GPT-2 Dermatology Captioner"
        clip_name = "CLIP"
        description = (
            "A dermatology image captioning model combining CLIP vision encoder "
            f"with {gpt2_name} language model. Trained on dermatological images for "
            "generating clinical descriptions of skin lesions."
        )
    return title, clip_name, description


def _assert_env():
    if not HF_TOKEN or not HF_TOKEN.startswith("hf_"):
        raise RuntimeError("Укажи корректный HF_TOKEN со scope 'write'.")

    if not OUT_DIR.exists():
        raise FileNotFoundError(f"OUT_DIR не найден: {OUT_DIR}")

    pts = list(OUT_DIR.glob("*.pt")) + list(OUT_DIR.glob("final_captioner_*.pt"))
    if not pts:
        raise FileNotFoundError(f"В {OUT_DIR} не найдено файлов *.pt")


def _collect_files() -> List[Path]:
    files: List[Path] = []
    for pat in UPLOAD_GLOBS:
        files.extend(OUT_DIR.glob(pat))
    files = sorted(set(files))
    return files


def _find_latest_log() -> Optional[Path]:
    cand = sorted(OUT_DIR.glob("train_log_*.txt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return cand[0] if cand else None


def _parse_best_metrics(log_path: Path) -> Dict[str, Dict[str, float]]:
    if not log_path or not log_path.exists():
        return {}

    patt = re.compile(
        r"^\[(?P<stage>Stage[^\]]+)\]\s*best:\s*"
        r"val_loss=(?P<val_loss>[-+0-9\.eE]+)\s*\|\s*"
        r"PPL=(?P<ppl>[-+0-9\.eEinf]+)\s*\|\s*"
        r"BLEU=(?P<bleu>[-+0-9\.eE]+)\s*\|\s*"
        r"ROUGE-L=(?P<rouge>[-+0-9\.eE]+)\s*\|\s*"
        r"CIDEr-D=(?P<cider>[-+0-9\.eE]+)\s*\|\s*"
        r"CLIP=(?P<clip>[-+0-9\.eE]+)\s*\|\s*"
        r"BERT_F1=(?P<bertf1>[-+0-9\.eE]+)",
        re.IGNORECASE
    )

    results: Dict[str, Dict[str, float]] = {}
    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = patt.search(line.strip())
            if m:
                stage = m.group("stage").strip()
                # парсим числа
                def _flt(x: str) -> float:
                    try:
                        if x.lower() == "inf":
                            return float("inf")
                        return float(x)
                    except Exception:
                        return 0.0
                results[stage] = {
                    "val_loss": _flt(m.group("val_loss")),
                    "PPL": _flt(m.group("ppl")),
                    "BLEU": _flt(m.group("bleu")),
                    "ROUGE_L": _flt(m.group("rouge")),
                    "CIDEr_D": _flt(m.group("cider")),
                    "CLIP": _flt(m.group("clip")),
                    "BERT_F1": _flt(m.group("bertf1")),
                }
    return results


def _read_final_json() -> Dict:
    js = {}
    cand = sorted(OUT_DIR.glob("final_captioner_*.json"))
    if cand:
        try:
            js = json.loads(cand[-1].read_text(encoding="utf-8"))
        except Exception:
            js = {}
    return js


def _fmt_size(nbytes: int) -> str:
    for u in ["B","KB","MB","GB","TB"]:
        if nbytes < 1024:
            return f"{nbytes:.0f} {u}"
        nbytes /= 1024.0
    return f"{nbytes:.1f} PB"


def _build_readme(repo_id: str,
                  files: List[Path],
                  metrics_by_stage: Dict[str, Dict[str, float]],
                  final_json: Dict) -> str:
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    model_type, _ = _detect_model_type_and_repo_name(OUT_DIR, final_json)
    title, clip_name, description = _get_model_description(model_type, final_json)

    rows = []
    for p in files:
        size = _fmt_size(p.stat().st_size)
        sha = hashlib.sha256(p.read_bytes()[:1024*1024]).hexdigest()[:12]  # по первым 1МБ для быстроты
        rows.append(f"| `{p.name}` | {size} | sha256[:12]={sha} |")
    files_table = "\n".join(["| File | Size | Check |", "|---|---:|---|", *rows])

    # Метрики
    def fmt_stage(name: str) -> str:
        m = metrics_by_stage.get(name, {})
        if not m:
            return f"**{name}**: _no data_"
        return (
            f"**{name}**  \n"
            f"val_loss={m.get('val_loss',0):.4f} • PPL={m.get('PPL',0):.2f}  \n"
            f"BLEU={m.get('BLEU',0):.1f} • ROUGE-L={m.get('ROUGE_L',0):.3f} • "
            f"CIDEr-D={m.get('CIDEr_D',0):.2f} • CLIP={m.get('CLIP',0):.1f} • "
            f"BERT_F1={m.get('BERT_F1',0):.3f}"
        )

    stages_md = []
    for key in ["Stage A (META)", "Stage B (SKINCAP)"]:
        if key in metrics_by_stage:
            stages_md.append(fmt_stage(key))
    if not stages_md and metrics_by_stage:
        stages_md = [fmt_stage(k) for k in metrics_by_stage.keys()]

    # Инфа из json
    gpt2_name = final_json.get("gpt2_name", "gpt2-medium")
    clip_repo = final_json.get("clip_weight_path", final_json.get("clip_repo", ""))
    prompt = final_json.get("prompt", "Describe the skin lesion...")

    header = textwrap.dedent(f"""\
    ---
    tags:
      - image-to-text
      - image-captioning
      - CLIP
      - GPT-2
      - dermatology
      - {model_type}
    library_name: transformers
    license: other
    language:
      - en
    pipeline_tag: image-to-text
    ---

    # {title}

    {description}

    **Architecture**: {clip_name} → learnable prefix → GPT-2 (`{gpt2_name}`).
    Trained in two stages: Stage A (META) for generalization and Stage B (SkinCAP) for style/terminology.
    """)

    metrics_md = "## Metrics\n" + ("\n\n".join(stages_md) if stages_md else "_No metrics parsed from logs._")

    usage = textwrap.dedent(f"""\
    ## Inference

    > Minimal example uses `inference_min.py` included in this repo.  
    > Requires: `pip install torch transformers open_clip_torch pillow huggingface_hub`

    ```python
    from huggingface_hub import snapshot_download
    from inference_min import load_model, generate

    # 1) download repo snapshot
    repo_dir = snapshot_download("{repo_id}", allow_patterns=["*.pt","*.json","inference_min.py"])

    # 2) load model from saved config/weights
    model = load_model(repo_dir)  # builds CLIP backend + GPT-2 + prefix projector

    # 3) run generation
    img_paths = ["/path/to/derma_image.jpg"]  # local test images
    caps = generate(model, img_paths, prompt={json.dumps(prompt)})
    for c in caps:
        print(c)
    ```
    """)

    files_md = "## Files\n" + files_table

    details = textwrap.dedent(f"""\
    ## Details

    - **Vision Encoder**: {clip_name}
    - **Language Model**: GPT-2 (`{gpt2_name}`)
    - **CLIP weights**: `{clip_repo}`
    - **Prefix tokens**: {final_json.get("prefix_tokens", "N/A")}  
    - **Training prompt**: `{prompt}`

    ### Model Type Detection
    - Detected as: `{model_type}`
    - Repository: `{repo_id}`

    _Auto-generated on {now}._
    """)

    return "\n\n".join([header, metrics_md, usage, files_md, details])


def _inference_min_py() -> str:
    return r'''# -*- coding: utf-8 -*-
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
'''


def main():
    import sys
    if len(sys.argv) != 2:
        print("Usage: python to_hf.py <out_dir>")
        print("Example: python to_hf.py /root/clip_xai/final/final_biomedclip_pubmedbert")
        sys.exit(1)
    
    global OUT_DIR, HF_REPO_ID
    OUT_DIR = Path(sys.argv[1])
    
    print(f"[start] Uploading from {OUT_DIR}")
    _assert_env()
    
    final_json = _read_final_json()
    print(f"[info] final_json keys: {list(final_json.keys())}")
    
    model_type, repo_suffix = _detect_model_type_and_repo_name(OUT_DIR, final_json)
    print(f"[detect] Model type: {model_type}, repo suffix: {repo_suffix}")
    
    try:
        api = HfApi(token=HF_TOKEN)
        user_info = whoami(token=HF_TOKEN)
        username = user_info['name']
        HF_REPO_ID = f"{username}/{repo_suffix}"
        print(f"[repo] Will create/update repo: {HF_REPO_ID}")
    except Exception as e:
        print(f"[error] Failed to get user info: {e}")
        HF_REPO_ID = f"user/{repo_suffix}"  # fallback
    
    files = _collect_files()
    print(f"[files] Found {len(files)} files to upload:")
    for f in files:
        print(f"  - {f.name} ({_fmt_size(f.stat().st_size)})")
    
    log_path = _find_latest_log()
    print(f"[log] Latest log: {log_path}")
    metrics_by_stage = _parse_best_metrics(log_path) if log_path else {}
    if metrics_by_stage:
        print("[metrics] Parsed from log:")
        for stage, metrics in metrics_by_stage.items():
            print(f"  {stage}: BLEU={metrics.get('BLEU', 0):.1f}, CLIP={metrics.get('CLIP', 0):.1f}")
    
    readme_content = _build_readme(HF_REPO_ID, files, metrics_by_stage, final_json)
    inference_content = _inference_min_py()
    
    try:
        api = HfApi(token=HF_TOKEN)
        create_repo(HF_REPO_ID, private=HF_PRIVATE, token=HF_TOKEN, exist_ok=True)
        print(f"[repo] Created/updated repo: https://huggingface.co/{HF_REPO_ID}")
    except Exception as e:
        print(f"[warning] Repo creation issue: {e}")
    
    try:
        readme_path = OUT_DIR / "README.md"
        readme_path.write_text(readme_content, encoding="utf-8")
        upload_file(
            path_or_fileobj=str(readme_path),
            path_in_repo="README.md",
            repo_id=HF_REPO_ID,
            token=HF_TOKEN
        )
        print("[upload] README.md")
        
        inference_path = OUT_DIR / INFER_FILE_NAME
        inference_path.write_text(inference_content, encoding="utf-8")
        upload_file(
            path_or_fileobj=str(inference_path),
            path_in_repo=INFER_FILE_NAME,
            repo_id=HF_REPO_ID,
            token=HF_TOKEN
        )
        print(f"[upload] {INFER_FILE_NAME}")
        
        for f in files:
            upload_file(
                path_or_fileobj=str(f),
                path_in_repo=f.name,
                repo_id=HF_REPO_ID,
                token=HF_TOKEN
            )
            print(f"[upload] {f.name}")
            
    except Exception as e:
        print(f"[error] Upload failed: {e}")
        sys.exit(1)
    
    print(f"[done] All files uploaded to: https://huggingface.co/{HF_REPO_ID}")
    print(f"[info] Model type: {model_type}")
    title, clip_name, description = _get_model_description(model_type, final_json)
    print(f"[info] Title: {title}")
    print(f"[info] CLIP backend: {clip_name}")


if __name__ == "__main__":
    main()
