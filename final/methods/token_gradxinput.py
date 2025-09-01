# final/methods/token_gradxinput.py
# -*- coding: utf-8 -*-
"""
Token Grad×Input for CLIP Text — вертикальная узкая карточка без matplotlib.

Метод:
  score_i = |grad(token_embed_i) * token_embed_i| по токенам → агрегация в слова (max/mean).
  HF-CLIP: точные BPE-токены → слова.
  open_clip: hook на token_embedding → токен-градиенты; слова восстанавливаем из подписи
             (без показа id_***), распределяя токен-оценки по словам пропорционально длине.

Карточка (PIL):
[Original image] → [Generated caption (подсветка слов)] → [Legend] → [Metrics] → [Top words]
"""

from __future__ import annotations
from dataclasses import dataclass, asdict, fields
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Union
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import torch
import torch.nn.functional as F


# ============================= Config ==================================

@dataclass
class TGxIConfig:
    # Верстка (узкая вертикаль)
    canvas_width_px: int = 720
    canvas_height_px: int = 1400
    margins_px: int = 16
    block_gap_px: int = 10
    image_block_height_px: int = 360

    # Типографика — единый шрифт
    font_path_regular: str = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    fs_header: int = 22
    fs_text: int = 20
    fs_small: int = 16
    fs_text_min: int = 14
    line_pad_px: int = 6

    # Аггрегация и метрики
    word_agg: str = "max"     # "max" | "mean"
    tau: float = 0.60
    topk: int = 12
    show_reference: bool = False

    # Очистка токенов (HF)
    drop_special_tokens: bool = True
    merge_bpe_suffix: str = "</w>"

    # Поведение рендера
    auto_shrink_text: bool = True
    auto_expand_canvas: bool = True
    canvas_height_max_px: int = 4000

def _coerce_cfg(config: Union[TGxIConfig, Dict[str, Any], None]) -> TGxIConfig:
    if isinstance(config, TGxIConfig):
        return config
    d = dict(config or {})
    valid = {f.name for f in fields(TGxIConfig)}
    d = {k: v for k, v in d.items() if k in valid}
    return TGxIConfig(**d)


# ============================= Utils ===================================

def normalize01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    mn, mx = float(x.min()), float(x.max())
    if mx <= mn + 1e-8: return np.zeros_like(x)
    return (x - mn) / (mx - mn + 1e-8)

def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = F.normalize(a, dim=-1); b = F.normalize(b, dim=-1)
    return (a * b).sum(dim=-1)

def load_image(path: str, fallback_hw: int) -> Image.Image:
    try: return Image.open(path).convert('RGB')
    except Exception: return Image.new('RGB', (fallback_hw, fallback_hw), (245,245,245))

def _find_font(path: str, size: int):
    try: return ImageFont.truetype(path, size=size)
    except Exception:
        try: return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size=size)
        except Exception: return ImageFont.load_default()

def _text_size(drw: ImageDraw.ImageDraw, text: str, font) -> Tuple[int,int]:
    try:
        b = drw.textbbox((0,0), text, font=font); return b[2]-b[0], b[3]-b[1]
    except Exception:
        return drw.textsize(text, font=font)

def _fit_cover(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    sw, sh = target_w / img.width, target_h / img.height
    s = max(sw, sh)
    nw, nh = max(1,int(round(img.width*s))), max(1,int(round(img.height*s)))
    img2 = img.resize((nw, nh), Image.LANCZOS)
    ox, oy = (nw - target_w)//2, (nh - target_h)//2
    return img2.crop((ox, oy, ox + target_w, oy + target_h))

# --- BPE → слова (HF путь) ---
def _words_from_tokens(tokens: List[str]) -> Tuple[List[str], List[List[int]]]:
    if not tokens: return [], []
    words, groups = [], []
    cur = ""; idxs: List[int] = []
    def is_word_start(tok: str) -> bool:
        return tok.startswith("Ġ") or tok.startswith("▁") or tok in [".", ",", ":", ";", "!", "?", "(", ")", "/", "-"]
    for i, t in enumerate(tokens):
        plain = t.lstrip("Ġ▁").replace("</w>", "")
        if i == 0 or is_word_start(t):
            if cur:
                words.append(cur); groups.append(idxs)
            cur = plain; idxs = [i]
        else:
            cur += plain; idxs.append(i)
    if cur: words.append(cur); groups.append(idxs)
    out_w, out_g = [], []
    for w, g in zip(words, groups):
        if w.strip(): out_w.append(w); out_g.append(g)
    return out_w, out_g

# --- open_clip: распределим токен-оценки по словам подписи ---
def _map_token_scores_to_words_by_length(caption: str, tok_scores: np.ndarray) -> Tuple[List[str], np.ndarray]:
    words = caption.split()
    if len(words) == 0 or tok_scores.size == 0:
        return words, np.zeros(len(words), dtype=np.float32)
    lens = np.array([max(1, len(w)) for w in words], dtype=np.float32)
    parts = np.maximum(1, np.round(lens / lens.sum() * tok_scores.size).astype(int))
    # скорректируем, чтобы суммы совпали
    diff = int(tok_scores.size - parts.sum())
    if diff != 0:
        # отдадим/заберём по 1 у самых длинных слов (по модулю diff)
        order = np.argsort(-lens)
        for idx in order[:abs(diff)]:
            parts[idx] += 1 if diff > 0 else -1
    # разрез токенов
    scores_w = []
    p0 = 0
    for p in parts:
        block = tok_scores[p0:p0+p]
        scores_w.append(float(block.max()) if block.size else 0.0)
        p0 += p
    return words, np.array(scores_w, dtype=np.float32)


# ================== Grad×Input text features (HF/open_clip) ==================

def _text_embeds_for_grad_hf(clip, texts: List[str]):
    """HF CLIP путь: вернём text_embeds(norm), tok_emb(requires_grad), attn_mask, input_ids, raw_tokens."""
    device = clip.device
    
    # Очистка памяти в начале
    torch.cuda.empty_cache()
    
    tok = clip.tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=77)
    input_ids = tok['input_ids'].to(device)
    attn_mask = tok['attention_mask'].to(device)
    tm = clip.model.text_model

    # 1) Через inputs_embeds, чтобы сразу иметь градиенты
    try:
        tok_emb = tm.embeddings.token_embedding(input_ids)  # [B,T,dim]
        tok_emb = tok_emb.detach().requires_grad_(True)
        out = tm(inputs_embeds=tok_emb, attention_mask=attn_mask,
                 output_hidden_states=True, return_dict=True)
        last_hidden = out.last_hidden_state
    except TypeError:
        # 2) Старый путь — hook
        store = {}
        def fwd_hook(_m, _inp, out):
            store['tok_emb'] = out
            out.retain_grad()
        h = tm.embeddings.token_embedding.register_forward_hook(fwd_hook)
        out = tm(input_ids=input_ids, attention_mask=attn_mask,
                 output_hidden_states=True, return_dict=True)
        h.remove()
        last_hidden = out.last_hidden_state
        tok_emb = store['tok_emb']  # [B,T,dim]

    eos_id = getattr(clip.tokenizer, "eos_token_id", None)
    if eos_id is None:
        eos_pos = (attn_mask.sum(dim=1) - 1).clamp(min=0).long()
    else:
        eos_pos = (input_ids == eos_id).int().argmax(dim=1)

    idx = eos_pos.view(-1,1,1).expand(-1,1,last_hidden.size(-1))
    pooled = last_hidden.gather(1, idx).squeeze(1)
    pooled = tm.final_layer_norm(pooled)
    text_embeds = pooled @ clip.model.text_projection.weight.T
    text_embeds = F.normalize(text_embeds, dim=-1)

    # сырые BPE-токены
    raw_tokens = clip.tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
    
    # Очистка промежуточных результатов
    torch.cuda.empty_cache()
    
    return text_embeds, tok_emb, attn_mask, input_ids, raw_tokens

def _text_embeds_for_grad_openclip(clip, text: str):
    """open_clip путь: hook на token_embedding; вернём tfeat(norm), tok_emb(requires_grad), mask(≈!=0), ids."""
    device = clip.device
    
    # Очистка памяти в начале
    torch.cuda.empty_cache()
    
    try:
        import open_clip
        tok_fn = open_clip.get_tokenizer(getattr(clip, "repo", "ViT-B-32"))
    except Exception:
        # на крайний случай — простой токенайзер open_clip
        import open_clip
        tok_fn = open_clip.get_tokenizer("ViT-B-32")

    toks = tok_fn([text]).to(device)  # [1,77] int64
    store = {}
    def fwd_hook(_m, _inp, out):
        store['tok_emb'] = out
        out.retain_grad()
    h = clip.model.token_embedding.register_forward_hook(fwd_hook)
    
    try:
        tfeat = clip.model.encode_text(toks)  # [1,D]
    finally:
        h.remove()
        # Очистка памяти после forward pass
        torch.cuda.empty_cache()
    
    tok_emb: torch.Tensor = store['tok_emb']  # [1,T,dim]
    # бинарная маска: всё, что >0 (паддинги у open_clip обычно 0)
    mask = (toks.squeeze(0) > 0).to(device)
    return F.normalize(tfeat, dim=-1), tok_emb, mask, toks.squeeze(0)


# ============================ Core method ==============================

def compute_token_gradxinput(clip, image: Image.Image, text: str, cfg: TGxIConfig
                             ) -> Tuple[List[str], List[float], Dict[str, Any]]:
    """
    Возвращает:
      words (для отображения), scores_word ∈ [0,1], meta (cos, покрытие и т.д.).
    """
    # Очистка CUDA кэша перед началом
    torch.cuda.empty_cache()
    
    # image features (без градиента)
    with torch.no_grad():
        if hasattr(clip, "encode_image"):
            img_feat = clip.encode_image([image])
        else:
            px = clip.processor(images=[image], return_tensors='pt')['pixel_values'].to(clip.device)
            img_feat = clip.model.get_image_features(pixel_values=px)
        img_feat = F.normalize(img_feat, dim=-1)

    # text side
    raw_tokens: List[str] = []
    used_hf = hasattr(clip, "model") and hasattr(clip.model, "text_model")
    if used_hf:
        text_embeds, tok_emb, attn_mask, input_ids, raw_tokens = _text_embeds_for_grad_hf(clip, [text])
        pad_mask = attn_mask.squeeze(0).bool()
        ids_list = input_ids.squeeze(0)[pad_mask].tolist()
    else:
        tfeat, tok_emb, pad_mask, ids = _text_embeds_for_grad_openclip(clip, text)
        text_embeds = tfeat
        ids_list = ids[pad_mask].tolist()

    # цель и backward
    sim = cosine_sim(img_feat, text_embeds)  # [1]
    clip.model.zero_grad(set_to_none=True)
    sim.backward()

    grads = tok_emb.grad  # [1,T,dim]
    if grads is None:
        # Очистка перед возвратом ошибки
        torch.cuda.empty_cache()
        raise RuntimeError("No gradients on token embeddings; cannot compute grad×input.")
    gi = (grads * tok_emb).sum(dim=-1).squeeze(0)  # [T]
    gi = gi.abs()

    # применим маску паддингов
    gi = gi[pad_mask]

    # очистка спец токенов (HF)
    if used_hf and cfg.drop_special_tokens and len(raw_tokens) > 0:
        eos_id = getattr(clip.tokenizer, "eos_token_id", None)
        keep_scores, keep_tokens = [], []
        for s, i, t in zip(gi.detach().cpu().tolist(), ids_list, [raw_tokens[k] for k, m in enumerate(pad_mask.tolist()) if m]):
            if eos_id is not None and i == eos_id:
                continue
            keep_scores.append(s); keep_tokens.append(t)
        gi = torch.tensor(keep_scores, dtype=tok_emb.dtype, device=tok_emb.device) if keep_scores else torch.zeros(0, device=tok_emb.device)
        raw_tokens = keep_tokens

    # нормировка токен-скор
    arr_tok = gi.detach().cpu().numpy().astype(np.float32)
    tok_scores01 = normalize01(arr_tok)

    # ---- агрегируем в слова ----
    if used_hf:
        words, groups = _words_from_tokens(raw_tokens)
        if len(words) == 0:
            words = text.split()
            scores_word = np.zeros(len(words), dtype=np.float32)
        else:
            scores_word = np.zeros(len(words), dtype=np.float32)
            for wi, idxs in enumerate(groups):
                if cfg.word_agg == "mean":
                    scores_word[wi] = float(tok_scores01[idxs].mean())
                else:
                    scores_word[wi] = float(tok_scores01[idxs].max())
    else:
        # open_clip: распределим токены по словам подписи
        words, scores_word = _map_token_scores_to_words_by_length(text, tok_scores01)

    scores_word = normalize01(scores_word)

    meta = {
        "cos": float(sim.detach().cpu().item()),
        "words_count": int(len(words)),
        "tokens_count": int(arr_tok.size),
        "token_max": float(scores_word.max()) if scores_word.size else 0.0,
        "token_cov@tau": float((scores_word >= cfg.tau).mean()) if scores_word.size else 0.0,
        "word_agg": cfg.word_agg,
        "backend": "hf" if used_hf else "open_clip"
    }
    
    # Финальная очистка памяти
    torch.cuda.empty_cache()
    
    return words, scores_word.tolist(), meta


# ============================== Renderer ===============================

def _wrap_words_draw(drw, x: int, y: int, max_w: int, words: List[str], scores: List[float], font, gap: int) -> int:
    xx, yy = x, y
    line_h = (font.size if hasattr(font, "size") else 20) + 8
    for word, sc in zip(words, scores):
        piece = word + " "
        tw, th = _text_size(drw, piece, font)
        if xx + tw > x + max_w:
            yy += line_h; xx = x
        c = float(max(0.0, min(1.0, sc)))
        if c > 0.05:
            v = int(255 * (1.0 - 0.8 * c))  # синий градиент
            drw.rectangle([xx - 2, yy - 2, xx + tw, yy + th + 2], fill=(v, v, 255), outline=None)
        drw.text((xx, yy), piece, font=font, fill=(0,0,0))
        xx += tw
    return yy + line_h

def _draw_gradient_bar_blue(drw, x: int, y: int, w: int, h: int):
    for i in range(w):
        t = i / max(1, w - 1)
        v = int(255 * (1 - 0.8 * t))
        drw.line([(x + i, y), (x + i, y + h)], fill=(v, v, 255))
    drw.rectangle([x, y, x + w, y + h], outline=(0, 0, 0), width=1)

def _measure_total_height(W: int, cfg: TGxIConfig, gen_caption: str, ref_caption: Optional[str],
                          f_hdr, f_txt, f_sml) -> int:
    tmp = Image.new("RGB", (W, 200), (255,255,255))
    drw = ImageDraw.Draw(tmp)
    M, GAP = cfg.margins_px, cfg.block_gap_px
    col_w = W - 2*M
    total = 0
    total += _text_size(drw, "Original image", f_hdr)[1] + 6
    total += cfg.image_block_height_px + GAP
    total += _text_size(drw, "Generated caption (Grad×Input)", f_hdr)[1] + 6
    # грубая оценка высоты подписи по словам
    words = gen_caption.split()
    xx, yy = 0, 0
    line_h = (f_txt.size if hasattr(f_txt, "size") else 20) + 8
    for w in words:
        piece = w + " "
        tw, th = _text_size(drw, piece, f_txt)
        if xx + tw > col_w: yy += line_h; xx = 0
        xx += tw
    total += yy + line_h + GAP
    if cfg.show_reference and ref_caption:
        total += _text_size(drw, "Reference caption", f_hdr)[1] + 6
        xx, yy = 0, 0
        for w in (ref_caption or "").split():
            piece = w + " "; tw, th = _text_size(drw, piece, f_txt)
            if xx + tw > col_w: yy += line_h; xx = 0
            xx += tw
        total += yy + line_h + GAP
    total += _text_size(drw, "Token importance (low → high)", f_sml)[1] + 4
    total += 14 + 26 + GAP
    total += _text_size(drw, "Metrics (explained)", f_hdr)[1] + 6
    for lbl in ["Косинус(изобр., полный текст)", f"Покрытие ≥ {cfg.tau:.2f}", "Макс. важность слова",
                "Слова/токены (features)"]:
        total += _text_size(drw, f"{lbl}: 0.000", f_sml)[1] + 4
    total += _text_size(drw, "Top words", f_hdr)[1] + 6
    total += cfg.topk * (max(14, f_sml.size + 4) + 6)
    total += _text_size(drw, "Grad×Input • CLIP text attribution", f_sml)[1] + 8
    return total

def _render_card_vertical(out_png: Path, image: Image.Image,
                          gen_caption: str, ref_caption: Optional[str],
                          words: List[str], scores: List[float],
                          meta: Dict[str, Any], cfg: TGxIConfig):
    W = cfg.canvas_width_px
    M, GAP = cfg.margins_px, cfg.block_gap_px
    col_w = W - 2*M

    f_hdr = _find_font(cfg.font_path_regular, cfg.fs_header)
    f_txt = _find_font(cfg.font_path_regular, cfg.fs_text)
    f_sml = _find_font(cfg.font_path_regular, cfg.fs_small)

    # авто-высота/кегль
    fs_caption = cfg.fs_text
    total = _measure_total_height(W, cfg, gen_caption, ref_caption if cfg.show_reference else None,
                                  f_hdr, f_txt, f_sml)
    if cfg.auto_shrink_text:
        while total + 2*M > cfg.canvas_height_max_px and fs_caption > cfg.fs_text_min:
            fs_caption -= 1
            f_txt = _find_font(cfg.font_path_regular, fs_caption)
            total = _measure_total_height(W, cfg, gen_caption, ref_caption if cfg.show_reference else None,
                                          f_hdr, f_txt, f_sml)

    H = max(cfg.canvas_height_px, total + 2*M)
    if cfg.auto_expand_canvas:
        H = min(max(H, cfg.canvas_height_px), cfg.canvas_height_max_px)

    canvas = Image.new("RGB", (W, H), (255,255,255))
    drw = ImageDraw.Draw(canvas)

    x = M; y = M

    # 1) Original
    drw.text((x, y), "Original image", font=f_hdr, fill=(0,0,0))
    y += _text_size(drw, "Original image", f_hdr)[1] + 6
    canvas.paste(_fit_cover(image, col_w, cfg.image_block_height_px), (x, y))
    y += cfg.image_block_height_px + GAP

    # 2) Caption with highlight
    title = "Generated caption (Grad×Input)"
    drw.text((x, y), title, font=f_hdr, fill=(0,0,0))
    y += _text_size(drw, title, f_hdr)[1] + 6
    y = _wrap_words_draw(drw, x, y, col_w, words if words else gen_caption.split(),
                         scores if scores else [0.0]*len(gen_caption.split()), f_txt, cfg.line_pad_px)
    y += GAP

    if cfg.show_reference and ref_caption:
        drw.text((x, y), "Reference caption", font=f_hdr, fill=(0,0,0))
        y += _text_size(drw, "Reference caption", f_hdr)[1] + 6
        y = _wrap_words_draw(drw, x, y, col_w, ref_caption.split(), [0.0]*len(ref_caption.split()), f_txt, cfg.line_pad_px)
        y += GAP

    # 3) Colorbar
    drw.text((x, y), "Token importance (low → high)", font=f_sml, fill=(0,0,0))
    y += _text_size(drw, "Token importance (low → high)", f_sml)[1] + 4
    _draw_gradient_bar_blue(drw, x, y, col_w, 14)
    drw.text((x, y + 18), "Low", font=f_sml, fill=(0,0,0))
    drw.text((x + col_w - _text_size(drw, "High", f_sml)[0], y + 18), "High", font=f_sml, fill=(0,0,0))
    y += 14 + 26 + GAP

    # 4) Metrics
    drw.text((x, y), "Metrics (explained)", font=f_hdr, fill=(0,0,0))
    y += _text_size(drw, "Metrics (explained)", f_hdr)[1] + 6

    def kv(k: str, v: str):
        nonlocal y
        line = f"{k}: {v}"
        drw.text((x, y), line, font=f_sml, fill=(0,0,0))
        y += _text_size(drw, line, f_sml)[1] + 4

    kv("Косинус(изобр., полный текст)", f"{float(meta.get('cos',0.0)):.3f}")
    kv(f"Покрытие ≥ {cfg.tau:.2f}", f"{float(meta.get('token_cov@tau',0.0)):.3f}")
    kv("Макс. важность слова", f"{float(meta.get('token_max',0.0)):.3f}")
    kv("Слова/токены (features)", f"{int(meta.get('words_count',0))} / {int(meta.get('tokens_count',0))}")
    y += GAP

    # 5) Top words (с цифрами)
    drw.text((x, y), "Top words", font=f_hdr, fill=(0,0,0))
    y += _text_size(drw, "Top words", f_hdr)[1] + 6
    pairs = [(w, float(s)) for w, s in zip(words, scores)]
    pairs.sort(key=lambda t: t[1], reverse=True)
    pairs = pairs[:max(1, int(cfg.topk))]
    bar_h = max(14, f_sml.size + 4)
    for name, sc in pairs:
        c = max(0.0, min(1.0, sc))
        nm = name[:28] + ("…" if len(name) > 28 else "")
        tw, _ = _text_size(drw, nm, f_sml)
        drw.text((x, y), nm, font=f_sml, fill=(0,0,0))
        bx = x + tw + 8
        avail = col_w - (tw + 8)
        bw = max(0, int(avail * c))
        v = int(255 * (1 - 0.8 * c))
        drw.rectangle([bx, y, bx + bw, y + bar_h], fill=(v, v, 255), outline=(0,0,0))
        sval = f"{c:.3f}"
        svw, _ = _text_size(drw, sval, f_sml)
        if bx + bw + 6 + svw <= x + col_w:
            drw.text((bx + bw + 6, y), sval, font=f_sml, fill=(0,0,0))
        else:
            drw.text((x + col_w - svw, y), sval, font=f_sml, fill=(0,0,0))
        y += bar_h + 6

    # футер
    footer = "Grad×Input • CLIP text attribution • cos(image, text)"
    fw, fh = _text_size(drw, footer, f_sml)
    drw.text((W - fw - M, H - fh - 6), footer, font=f_sml, fill=(90,90,90))

    canvas.save(str(out_png), format="PNG")


# ============================== Public API =============================

def run(clip, captioner, *, image_path: str, out_dir: Path,
        ref_caption: Optional[str] = None,
        config: Optional[Union[TGxIConfig, Dict[str, Any]]] = None) -> Dict[str, Any]:
    cfg = _coerce_cfg(config)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    image = load_image(image_path, fallback_hw=getattr(clip, "image_size", 224))
    try:
        gen_caption = captioner.generate([image], prompt=getattr(captioner, "prompt", None) or "Describe the image.")[0]
    except Exception:
        gen_caption = ""

    words, scores, meta = compute_token_gradxinput(clip, image, gen_caption, cfg)

    # карточка
    stem = Path(image_path).stem
    out_png = out_dir / f"{stem}.png"
    _render_card_vertical(out_png, image, gen_caption, (ref_caption if cfg.show_reference else None),
                          words, scores, meta, cfg)

    # JSON
    top_list = [{"word": w, "score": float(s)} for w, s in sorted(zip(words, scores), key=lambda z: -z[1])[:max(1,int(cfg.topk))]]
    data = {
        "image_path": image_path,
        "method": "token_gradxinput_wordlevel",
        "config": asdict(cfg),
        "caption_generated": gen_caption,
        "caption_reference": (ref_caption if cfg.show_reference else None),
        "display_units": words,
        "scores": scores,
        "meta": meta,
        "top": top_list,
        "outputs": {"png": str(out_png)}
    }
    out_json = out_dir / f"{stem}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return data
