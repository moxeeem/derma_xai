# final/methods/attentionxgradient.py
# -*- coding: utf-8 -*-
"""
Attention × Gradient для CLIP-текста.
score_i = (grad×input)_i * Attn_last(EOS→i), усреднение по головам.
Если attention недоступен (open_clip и т.п.) — fallback на чистый Grad×Input.

Вертикальная узкая карточка (PIL):
[Original image] → [Generated caption (подсветка слов)] → [Legend] → [Metrics] → [Top words]
Единый шрифт: DejaVu Sans Regular.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict, fields
from typing import Optional, Dict, Any, Union, Tuple, List
from pathlib import Path
import json, re
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import torch
import torch.nn.functional as F


# ----------------------------- Config ---------------------------------

@dataclass
class AxGConfig:
    # Верстка
    canvas_width_px: int = 720
    canvas_height_px: int = 1400
    margins_px: int = 16
    block_gap_px: int = 10
    image_block_height_px: int = 360

    # Типографика — один шрифт
    font_path_regular: str = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    fs_header: int = 22
    fs_text: int = 20
    fs_small: int = 16
    fs_text_min: int = 14
    line_pad_px: int = 6

    # Метрики/визуал
    tau: float = 0.60                 # порог для coverage по словам
    topk: int = 12                    # Top words
    show_reference: bool = False

    # Режим агрегации токенов в слово
    word_agg: str = "max"             # "max" | "mean"

    # Рендер-поведение
    auto_shrink_text: bool = True
    auto_expand_canvas: bool = True
    canvas_height_max_px: int = 4000


def _coerce_cfg(config: Optional[Union[AxGConfig, Dict[str, Any]]]) -> AxGConfig:
    if isinstance(config, AxGConfig): 
        return config
    d = dict(config or {})
    valid = {f.name for f in fields(AxGConfig)}
    d = {k: v for k, v in d.items() if k in valid}
    return AxGConfig(**d)


# ----------------------------- Utils ----------------------------------

def load_image(path: str, fallback_hw: int) -> Image.Image:
    try: return Image.open(path).convert("RGB")
    except Exception: return Image.new("RGB", (fallback_hw, fallback_hw), (245,245,245))

def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = F.normalize(a, dim=-1); b = F.normalize(b, dim=-1)
    return (a * b).sum(dim=-1)

def normalize01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    mn, mx = float(x.min()), float(x.max())
    if mx <= mn + 1e-8: return np.zeros_like(x)
    return (x - mn) / (mx - mn + 1e-8)

def _find_font(path: str, size: int):
    try: return ImageFont.truetype(path, size=size)
    except Exception:
        try: return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size=size)
        except Exception: return ImageFont.load_default()

def _text_size(drw: ImageDraw.ImageDraw, text: str, font) -> tuple[int,int]:
    try:
        b = drw.textbbox((0,0), text, font=font); return b[2]-b[0], b[3]-b[1]
    except Exception:
        return drw.textsize(text, font=font)

# --- BPE→слова (как в других методах) ---
def _words_from_tokens(tokens: List[str]) -> Tuple[List[str], List[List[int]]]:
    if not tokens: return [], []
    words, groups = [], []
    cur = ""; idxs = []
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
        if w.strip():
            out_w.append(w); out_g.append(g)
    return out_w, out_g

def _filter_special_tokens(tokenizer, ids: List[int], toks: List[str], scores: torch.Tensor):
    """Убираем BOS/EOS/PAD/UNK из отображения и метрик."""
    try:
        mask = torch.tensor(
            tokenizer.get_special_tokens_mask(ids, already_has_special_tokens=True),
            dtype=torch.bool, device=scores.device
        )
        keep = ~mask
    except Exception:
        # Fallback: убираем известные спец-токены
        special_tokens = {"<|startoftext|>", "<|endoftext|>", "<pad>", "<unk>", "<bos>", "<eos>"}
        keep = torch.tensor([tok not in special_tokens for tok in toks], dtype=torch.bool, device=scores.device)
    
    ids = [i for i, k in zip(ids, keep.tolist()) if k]
    toks = [t for t, k in zip(toks, keep.tolist()) if k]
    scores = scores[keep]
    return ids, toks, scores

def _token_scores_to_words(text: str, toks: List[str], scores: torch.Tensor, tokenizer, agg: str = "max"):
    """
    Распределяем токен-важности по символам исходного текста, а затем агрегируем по словам.
    Надёжно работает даже без offset_mapping.
    """
    # нормализуем пробелы
    text_norm = re.sub(r"\s+", " ", text.strip())
    if not text_norm:
        return [], np.array([], dtype=np.float32)

    # пометим важность каждого символа
    ch = np.zeros(len(text_norm), dtype=np.float32)
    cursor = 0
    for t, sc in zip(toks, scores.detach().cpu().tolist()):
        try:
            piece = tokenizer.convert_tokens_to_string([t])
        except Exception:
            piece = t.replace("Ġ", " ")
        piece = re.sub(r"\s+", " ", piece)
        if not piece:
            continue

        # ищем следующий матч от текущей позиции
        pos = text_norm.find(piece, cursor)
        if pos < 0:
            # попробуем без ведущего пробела
            pos = text_norm.find(piece.lstrip(), cursor)
        if pos < 0:
            continue

        ch[pos:pos+len(piece)] += float(sc)
        cursor = pos + len(piece)

    # теперь собираем слова (по \S+)
    words = []
    w_scores = []
    for m in re.finditer(r"\S+", text_norm):
        s, e = m.span()
        words.append(m.group())
        seg = ch[s:e]
        if seg.size == 0:
            w_scores.append(0.0)
        elif agg == "mean":
            w_scores.append(float(seg.mean()))
        else:
            w_scores.append(float(seg.max()))

    w_scores = np.array(w_scores, dtype=np.float32)
    w_scores = (w_scores - w_scores.min()) / (w_scores.max() - w_scores.min() + 1e-8)
    return words, w_scores


# --------------- Core: Attention × Gradient (with robust fallback) -----

def _encode_image(clip, pil: Image.Image) -> torch.Tensor:
    # Унифицированный путь
    if hasattr(clip, "encode_image"):
        return clip.encode_image([pil])
    # Fallback на HF API
    px = clip.processor(images=[pil], return_tensors='pt')['pixel_values'].to(clip.device)
    return clip.model.get_image_features(pixel_values=px)

def _hf_axg(clip, text: str):
    """
    Возвращает:
      tokens_str, input_ids, tok_emb(requires_grad), last_attn (или None),
      eos_pos, tfeat (градиентный!).
    """
    device = clip.device
    tok = clip.tokenizer([text], return_tensors='pt', padding=True, truncation=True, max_length=77).to(device)
    input_ids, attn_mask = tok['input_ids'], tok['attention_mask']
    tm = clip.model.text_model

    store = {}
    def fwd_hook(_m, _inp, out):
        store['tok_emb'] = out
        out.retain_grad()
    h = tm.embeddings.token_embedding.register_forward_hook(fwd_hook)

    # один-единственный forward с включённым hook
    out = tm(input_ids=input_ids, attention_mask=attn_mask,
             output_hidden_states=True, output_attentions=True, return_dict=True)
    last_hidden = out.last_hidden_state                      # [1,T,C]
    attns = out.attentions                                   # tuple(L)[1,H,T,T] или None
    h.remove()

    tok_emb: torch.Tensor = store['tok_emb']                 # [1,T,C]

    eos_id = clip.tokenizer.eos_token_id
    eos_pos = (input_ids == eos_id).int().argmax(dim=1)      # [1]
    idx = eos_pos.view(-1,1,1).expand(-1,1,last_hidden.size(-1))
    pooled = last_hidden.gather(1, idx).squeeze(1)           # [1,C]
    pooled = tm.final_layer_norm(pooled)                     # [1,C]
    # та же проекция, что в CLIPModel
    tfeat = pooled @ clip.model.text_projection.weight.T     # [1,D]

    toks = clip.tokenizer.convert_ids_to_tokens(input_ids.squeeze(0).tolist())
    last_attn = attns[-1] if attns else None                 # [1,H,T,T] или None
    return toks, input_ids, tok_emb, last_attn, eos_pos, tfeat

def _openclip_axg(clip, text: str):
    """
    open_clip путь: attention недоступен → используем Grad×Input без attention.
    """
    device = clip.device
    # токенайзер
    try:
        import open_clip
        tokenizer_name = f"hf-hub:{clip.repo}" if hasattr(clip, "repo") else "ViT-B-32"
        tok_fn = open_clip.get_tokenizer(tokenizer_name)
        toks = tok_fn([text]).to(device)                           # [1,77]
        
        # Пробуем загрузить словарь токенов для преобразования в текст
        try:
            # Для DermLIP нужно подготовить токены для отображения
            if 'DermLIP' in clip.repo or 'dermlip' in clip.repo.lower():
                from transformers import CLIPTokenizer
                # Пробуем загрузить CLIP токенизатор для декодирования
                temp_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch16")
                tokens_str = []
                for t in toks.squeeze(0).tolist():
                    # Преобразуем ID в текст, если возможно
                    if t < temp_tokenizer.vocab_size:
                        tokens_str.append(temp_tokenizer.decode([t]))
                    else:
                        tokens_str.append(f"id_{t}")
            else:
                # Для других моделей просто используем ID как строки
                tokens_str = [f"id_{int(x)}" for x in toks.squeeze(0).tolist()]
        except Exception:
            # Fallback: просто ID как строки
            tokens_str = [f"id_{int(x)}" for x in toks.squeeze(0).tolist()]
    except Exception as e:
        raise RuntimeError(f"open_clip tokenizer not available: {e}")

    # hook на token_embedding
    store = {}
    def fwd_hook(_m, _inp, out):
        store['tok_emb'] = out
        out.retain_grad()
    h = clip.model.token_embedding.register_forward_hook(fwd_hook)

    tfeat = clip.model.encode_text(toks)                           # [1,D]
    h.remove()
    tok_emb: torch.Tensor = store['tok_emb']                       # [1,T,dim]

    eos_pos = torch.tensor([toks.size(1)-1], device=device)        # приблизительно (последний токен)
    return tokens_str, toks, tok_emb, None, eos_pos, tfeat


def compute_attentionxgradient(clip, image: Image.Image, text: str, cfg: AxGConfig
                               ) -> Tuple[List[str], List[float], Dict[str, Any]]:
    """
    Возвращает: words (для отображения), scores_word ∈ [0,1], meta.
    """
    device = clip.device

    # image feat (без градиента)
    with torch.no_grad():
        ifeat = _encode_image(clip, image)
        ifeat = F.normalize(ifeat, dim=-1)

    used_attn = False

    # --- текстовая ветка (HF предпочтительно) ---
    if hasattr(clip.model, "text_model"):   # HF CLIP
        tokens_str, input_ids, tok_emb, last_attn, eos_pos, tfeat = _hf_axg(clip, text)
        used_attn = last_attn is not None
    else:
        # open_clip: fallback без attention
        tokens_str, input_ids, tok_emb, last_attn, eos_pos, tfeat = _openclip_axg(clip, text)
        used_attn = False

    tfeat = F.normalize(tfeat, dim=-1)

    # Cosine + backward → grad×input
    sim = cosine_sim(ifeat, tfeat)                  # [1]
    clip.model.zero_grad(set_to_none=True)
    sim.backward(retain_graph=True)

    grads = tok_emb.grad                             # [1,T,dim]
    gi = (grads * tok_emb).sum(dim=-1).squeeze(0)    # [T]
    gi = gi.abs()

    if torch.all(gi == 0) or torch.isnan(gi).any():
        # запасной план: чистый attention (если есть) либо равномерно
        if used_attn:
            A = last_attn.mean(dim=1).squeeze(0)                 # [T,T]
            att_row = A[eos_pos.item(), :]
            sc_tok = att_row / (att_row.sum() + 1e-8)
        else:
            sc_tok = torch.ones_like(gi) / gi.numel()
    else:
        if used_attn:
            # Среднее по головам с последнего слоя, строка EOS
            A = last_attn.mean(dim=1).squeeze(0)         # [T,T]
            att_row = A[eos_pos.item(), :]
            att_row = att_row / (att_row.sum() + 1e-8)
            sc_tok = gi * att_row
        else:
            sc_tok = gi

    # Маска паддингов
    if hasattr(clip, "tokenizer") and hasattr(clip.tokenizer, "pad_token_id"):
        pad_id = clip.tokenizer.pad_token_id
        mask = (input_ids.squeeze(0) != pad_id).bool()
    else:
        mask = torch.ones_like(input_ids.squeeze(0), dtype=torch.bool)

    sc_tok = sc_tok[mask]
    ids_masked = input_ids.squeeze(0)[mask].tolist()
    toks_clean = tokens_str[:len(ids_masked)]

    # убираем спец-токены из визуала
    ids_masked, toks_clean, sc_tok = _filter_special_tokens(clip.tokenizer, ids_masked, toks_clean, sc_tok)

    # агрегируем по словам исходного текста (а не по BPE-кусочкам)
    words, word_scores = _token_scores_to_words(text, toks_clean, sc_tok, clip.tokenizer, agg=cfg.word_agg)
    scores = (word_scores ** 0.75)  # чуть повысим контраст
    display_units = words
    tokens_used = len(toks_clean)

    meta = {
        "cos": float(sim.detach().cpu().item()),
        "used_attention": bool(used_attn),
        "aggregation": cfg.word_agg,
        "words_count": int(len(display_units)),
        "tokens_count": int(tokens_used),
        "token_max": float(scores.max()) if len(scores) else 0.0,
        "token_cov@tau": float((scores >= cfg.tau).mean()) if len(scores) else 0.0,
    }
    return display_units, scores.tolist(), meta


# ---------------------- Renderer (vertical like others) ----------------

def _wrap_words_height(drw, max_w: int, words: List[str], font) -> int:
    if not words: return 0
    line_h = (font.size if hasattr(font, "size") else 20) + 8
    xx, yy = 0, 0
    for w in words:
        piece = w + " "
        tw, th = _text_size(drw, piece, font)
        if xx + tw > max_w:
            yy += line_h; xx = 0
        xx += tw
    return yy + line_h

def _wrap_words_draw(drw, x: int, y: int, max_w: int, words: List[str], scores: List[float], font, gap: int) -> int:
    xx, yy = x, y
    line_h = (font.size if hasattr(font, "size") else 20) + 8
    for word, sc in zip(words, scores):
        piece = word + " "
        tw, th = _text_size(drw, piece, font)
        if xx + tw > x + max_w:
            yy += line_h; xx = x
        c = float(max(0.0, min(1.0, sc)))
        if c > 0.02:
            v = int(255 * (1.0 - 0.8 * c))      # синий градиент (как в других методах)
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

def _measure_total_height(W: int, cfg: AxGConfig, gen_caption: str, ref_caption: Optional[str],
                          f_hdr, f_txt, f_sml) -> int:
    tmp = Image.new("RGB", (W, 200), (255,255,255))
    drw = ImageDraw.Draw(tmp)
    M, GAP = cfg.margins_px, cfg.block_gap_px
    col_w = W - 2*M
    total = 0
    total += _text_size(drw, "Original image", f_hdr)[1] + 6
    total += cfg.image_block_height_px + GAP
    total += _text_size(drw, "Generated caption (Attention×Gradient)", f_hdr)[1] + 6
    total += _wrap_words_height(drw, col_w, gen_caption.split(), f_txt) + GAP
    if cfg.show_reference and ref_caption:
        total += _text_size(drw, "Reference caption", f_hdr)[1] + 6
        total += _wrap_words_height(drw, col_w, ref_caption.split(), f_txt) + GAP
    total += _text_size(drw, "Token importance (low → high)", f_sml)[1] + 4
    total += 14 + 26 + GAP
    total += _text_size(drw, "Metrics (explained)", f_hdr)[1] + 6
    for lbl in ["Косинус(изобр., полный текст)", f"Покрытие ≥ {cfg.tau:.2f}", "Макс. важность слова",
                "Слова/токены (features)", "Attention использован"]:
        total += _text_size(drw, f"{lbl}: 0.000", f_sml)[1] + 4
    total += _text_size(drw, "Top words", f_hdr)[1] + 6
    bar_h = max(14, f_sml.size + 4)
    total += cfg.topk * (bar_h + 6)
    total += _text_size(drw, "Attention×Gradient • CLIP text attribution", f_sml)[1] + 8
    return total

def _render_card_vertical(
    out_png: Path,
    image: Image.Image,
    gen_caption: str,
    ref_caption: Optional[str],
    words: List[str],
    scores: List[float],
    metrics: Dict[str, Any],
    cfg: AxGConfig,
):
    W = cfg.canvas_width_px
    M, GAP = cfg.margins_px, cfg.block_gap_px
    col_w = W - 2*M

    f_hdr = _find_font(cfg.font_path_regular, cfg.fs_header)
    f_txt = _find_font(cfg.font_path_regular, cfg.fs_text)
    f_sml = _find_font(cfg.font_path_regular, cfg.fs_small)

    # Автоподбор высоты/кегля
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

    # 1) ОРИГИНАЛ
    drw.text((x, y), "Original image", font=f_hdr, fill=(0,0,0))
    y += _text_size(drw, "Original image", f_hdr)[1] + 6
    # cover-кроп
    sw, sh = (W - 2*M)/image.width, cfg.image_block_height_px/image.height
    s = max(sw, sh)
    nw, nh = max(1,int(round(image.width*s))), max(1,int(round(image.height*s)))
    img2 = image.resize((nw, nh), Image.LANCZOS)
    ox, oy = (nw - (W - 2*M))//2, (nh - cfg.image_block_height_px)//2
    img_fit = img2.crop((ox, oy, ox + (W - 2*M), oy + cfg.image_block_height_px))
    canvas.paste(img_fit, (x, y))
    y += cfg.image_block_height_px + GAP

    # 2) CAPTION с подсветкой слов
    title = "Generated caption (Attention×Gradient)"
    drw.text((x, y), title, font=f_hdr, fill=(0,0,0))
    y += _text_size(drw, title, f_hdr)[1] + 6
    words_to_show = words if words else gen_caption.split()
    scores_to_show = scores if scores else [0.0]*len(words_to_show)
    y = _wrap_words_draw(drw, x, y, col_w, words_to_show, scores_to_show, f_txt, cfg.line_pad_px)
    y += GAP

    if cfg.show_reference and ref_caption:
        drw.text((x, y), "Reference caption", font=f_hdr, fill=(0,0,0))
        y += _text_size(drw, "Reference caption", f_hdr)[1] + 6
        y = _wrap_words_draw(drw, x, y, col_w, ref_caption.split(), [0.0]*len(ref_caption.split()), f_txt, cfg.line_pad_px)
        y += GAP

    # 3) ШКАЛА
    drw.text((x, y), "Token importance (low → high)", font=f_sml, fill=(0,0,0))
    y += _text_size(drw, "Token importance (low → high)", f_sml)[1] + 4
    _draw_gradient_bar_blue(drw, x, y, col_w, 14)
    drw.text((x, y + 18), "Low", font=f_sml, fill=(0,0,0))
    drw.text((x + col_w - _text_size(drw, "High", f_sml)[0], y + 18), "High", font=f_sml, fill=(0,0,0))
    y += 14 + 26 + GAP

    # 4) МЕТРИКИ
    drw.text((x, y), "Metrics (explained)", font=f_hdr, fill=(0,0,0))
    y += _text_size(drw, "Metrics (explained)", f_hdr)[1] + 6

    def kv(k: str, v: str):
        nonlocal y
        line = f"{k}: {v}"
        drw.text((x, y), line, font=f_sml, fill=(0,0,0))
        y += _text_size(drw, line, f_sml)[1] + 4

    kv("Косинус(изобр., полный текст)", f"{float(metrics.get('cos',0.0)):.3f}")
    kv(f"Покрытие ≥ {cfg.tau:.2f}", f"{float(metrics.get('token_cov@tau',0.0)):.3f}")
    kv("Макс. важность слова", f"{float(metrics.get('token_max',0.0)):.3f}")
    kv("Слова/токены (features)", f"{int(metrics.get('words_count',0))} / {int(metrics.get('tokens_count',0))}")
    kv("Attention использован", "да" if bool(metrics.get('used_attention', False)) else "нет")
    y += GAP

    # 5) TOP WORDS (с цифрами)
    drw.text((x, y), "Top words", font=f_hdr, fill=(0,0,0))
    y += _text_size(drw, "Top words", f_hdr)[1] + 6
    pairs = [(w, float(s)) for w, s in zip(words_to_show, scores_to_show)]
    pairs.sort(key=lambda t: t[1], reverse=True)
    pairs = pairs[:cfg.topk]
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
        # бар
        drw.rectangle([bx, y, bx + bw, y + bar_h], fill=(v, v, 255), outline=(0,0,0))
        sval = f"{c:.3f}"
        svw, _ = _text_size(drw, sval, f_sml)
        if bx + bw + 6 + svw <= x + col_w:
            drw.text((bx + bw + 6, y), sval, font=f_sml, fill=(0,0,0))
        else:
            tx = max(bx + 2, x + col_w - svw)
            drw.text((tx, y), sval, font=f_sml, fill=(0,0,0))
        y += bar_h + 6

    # футер
    footer = "Attention×Gradient • CLIP text attribution • cos(image, text)"
    fw, fh = _text_size(drw, footer, f_sml)
    drw.text((W - fw - M, H - fh - 6), footer, font=f_sml, fill=(90,90,90))

    canvas.save(str(out_png), format="PNG")


# ----------------------------- Public API ------------------------------

def run(clip, captioner, *, image_path: str, out_dir: Path,
        ref_caption: Optional[str] = None,
        config: Optional[Union[AxGConfig, Dict[str, Any]]] = None) -> Dict[str, Any]:

    cfg = _coerce_cfg(config)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    image = load_image(image_path, fallback_hw=getattr(clip, "image_size", 224))
    try:
        gen_caption = captioner.generate([image], prompt=getattr(captioner, "prompt", None) or "Describe the image.")[0]
    except Exception:
        gen_caption = ""

    words, scores, meta = compute_attentionxgradient(clip, image, gen_caption, cfg)

    stem = Path(image_path).stem
    out_png = out_dir / f"{stem}.png"
    _render_card_vertical(out_png, image, gen_caption, (ref_caption if cfg.show_reference else None),
                          words, scores, meta, cfg)

    out_json = out_dir / f"{stem}.json"
    data = {
        "image_path": image_path,
        "method": "attentionxgradient_wordlevel",
        "config": asdict(cfg),
        "caption_generated": gen_caption,
        "caption_reference": (ref_caption if cfg.show_reference else None),
        "display_units": words,
        "scores": scores,
        "meta": meta,
        "outputs": {"png": str(out_png)}
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return data

# Добавить перед вызовом run в xai.py
