---
tags:
  - image-to-text
  - image-captioning
  - CLIP
  - GPT-2
  - dermatology
  - biomedclip
library_name: transformers
license: other
language:
  - en
pipeline_tag: image-to-text
---

# BiomedCLIP + GPT-2 Dermatology Captioner

A dermatology image captioning model combining BiomedCLIP vision encoder with gpt2-medium language model. Trained on dermatological images for generating clinical descriptions of skin lesions.

**Architecture**: BiomedCLIP (ViT-B/16) → learnable prefix → GPT-2 (`gpt2-medium`).
Trained in two stages: Stage A (META) for generalization and Stage B (SkinCAP) for style/terminology.


## Metrics
**Stage A (META)**  
val_loss=1.1222 • PPL=3.07  
BLEU=36.6 • ROUGE-L=0.521 • CIDEr-D=0.10 • CLIP=34.7 • BERT_F1=0.526

**Stage B (SKINCAP)**  
val_loss=1.1997 • PPL=3.32  
BLEU=9.3 • ROUGE-L=0.267 • CIDEr-D=0.12 • CLIP=40.5 • BERT_F1=0.348

## Inference

> Minimal example uses `inference_min.py` included in this repo.  
> Requires: `pip install torch transformers open_clip_torch pillow huggingface_hub`

```python
from huggingface_hub import snapshot_download
from inference_min import load_model, generate

# 1) download repo snapshot
repo_dir = snapshot_download("moxeeeem/biomedclip-gpt2-captioner", allow_patterns=["*.pt","*.json","inference_min.py"])

# 2) load model from saved config/weights
model = load_model(repo_dir)  # builds CLIP backend + GPT-2 + prefix projector

# 3) run generation
img_paths = ["/path/to/derma_image.jpg"]  # local test images
caps = generate(model, img_paths, prompt="Describe the skin lesion concisely (morphology, color, scale, border, location) in one sentence.Conclude with the most likely diagnosis (1\u20133 words).")
for c in caps:
    print(c)
```


## Files
| File | Size | Check |
|---|---:|---|
| `best_stageA.pt` | 2 GB | sha256[:12]=4c3f773c92f9 |
| `best_stageB.pt` | 2 GB | sha256[:12]=2b1b7e6aaa8f |
| `final_captioner_gpt2-medium_TimmModel.json` | 899 B | sha256[:12]=40f3fb250eab |
| `final_captioner_gpt2-medium_TimmModel.pt` | 2 GB | sha256[:12]=799a4d1b831d |
| `loss_biomedclip_pubmedbert.png` | 112 KB | sha256[:12]=92c3b8dfdf7f |

## Details

- **Vision Encoder**: BiomedCLIP (ViT-B/16)
- **Language Model**: GPT-2 (`gpt2-medium`)
- **CLIP weights**: `hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`
- **Prefix tokens**: 32  
- **Training prompt**: `Describe the skin lesion concisely (morphology, color, scale, border, location) in one sentence.Conclude with the most likely diagnosis (1–3 words).`

### Model Type Detection
- Detected as: `biomedclip`
- Repository: `moxeeeem/biomedclip-gpt2-captioner`

_Auto-generated on 2025-08-30 06:08 UTC._
