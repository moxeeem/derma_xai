---
tags:
  - image-to-text
  - image-captioning
  - CLIP
  - GPT-2
  - dermatology
  - dermlip
library_name: transformers
license: other
language:
  - en
pipeline_tag: image-to-text
---

# DermLIP + GPT-2 Dermatology Captioner

A dermatology image captioning model combining DermLIP vision encoder with gpt2-medium language model. Trained on dermatological images for generating clinical descriptions of skin lesions.

**Architecture**: DermLIP (ViT-B/16) → learnable prefix → GPT-2 (`gpt2-medium`).
Trained in two stages: Stage A (META) for generalization and Stage B (SkinCAP) for style/terminology.


## Metrics
**Stage A (META)**  
val_loss=1.1070 • PPL=3.03  
BLEU=38.6 • ROUGE-L=0.550 • CIDEr-D=0.17 • CLIP=24.4 • BERT_F1=0.565

**Stage B (SKINCAP)**  
val_loss=1.1903 • PPL=3.29  
BLEU=10.0 • ROUGE-L=0.278 • CIDEr-D=0.13 • CLIP=25.9 • BERT_F1=0.363

## Inference

> Minimal example uses `inference_min.py` included in this repo.  
> Requires: `pip install torch transformers open_clip_torch pillow huggingface_hub`

```python
from huggingface_hub import snapshot_download
from inference_min import load_model, generate

# 1) download repo snapshot
repo_dir = snapshot_download("moxeeeem/dermlip-gpt2-captioner", allow_patterns=["*.pt","*.json","inference_min.py"])

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
| `best_stageA.pt` | 2 GB | sha256[:12]=3219636f48b0 |
| `best_stageB.pt` | 2 GB | sha256[:12]=69bded2dcad1 |
| `final_captioner_gpt2-medium_VisionTransformer.json` | 849 B | sha256[:12]=e157402c9fe2 |
| `final_captioner_gpt2-medium_VisionTransformer.pt` | 2 GB | sha256[:12]=536ae07811c9 |
| `loss_dermlip_vitb16.png` | 110 KB | sha256[:12]=a04b1e5832d9 |

## Details

- **Vision Encoder**: DermLIP (ViT-B/16)
- **Language Model**: GPT-2 (`gpt2-medium`)
- **CLIP weights**: `hf-hub:redlessone/DermLIP_ViT-B-16`
- **Prefix tokens**: 32  
- **Training prompt**: `Describe the skin lesion concisely (morphology, color, scale, border, location) in one sentence.Conclude with the most likely diagnosis (1–3 words).`

### Model Type Detection
- Detected as: `dermlip`
- Repository: `moxeeeem/dermlip-gpt2-captioner`

_Auto-generated on 2025-08-30 09:25 UTC._
