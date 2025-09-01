# final/methods/token_occlusion.py
# -*- coding: utf-8 -*-
"""
Token Occlusion for CLIP text encoder (non-gradient).
Идея: важность слова = падение cos(image, text) при удалении слова из подписи.

Карточка — ВЕРТИКАЛЬНАЯ узкая (как в прошлой задаче):
[Original image] → [Generated caption (подсветка)] → [Legend] → [Metrics] → [Top words + цифры]

xai.py ожидает вызов:
    method_mod.run(clip, captioner, image_path=..., out_dir=..., ref_caption=..., config=method_params["token_occlusion"])
"""

from __future__ import annotations
from dataclasses import dataclass, asdict, fields
from typing import List, Tuple, Optional, Dict, Any, Union
from pathlib import Path
import json, math, re
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import torch
import torch.nn.functional as F


# ----------------------------- Config ---------------------------------

@dataclass
class TOccConfig:
    # Узкий формат (ширина фикс), высота — авто
    canvas_width_px: int = 720
    canvas_height_px: int = 1400            # старт; при auto_expand_canvas высота подгоняется под контент
    margins_px: int = 16
    block_gap_px: int = 10

    # Блок изображения
    image_block_height_px: int = 360        # фиксированная высота (cover-crop)

    # Шрифт — единый, как в matplotlib
    font_path_regular: str = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    fs_header: int = 22
    fs_text: int = 20
    fs_small: int = 16
    fs_text_min: int = 14                   # минимальный кегль для длинного caption
    line_pad_px: int = 6

    # Occlusion
    unit: str = "word"                      # "word" | "token"  (по умолчанию — слова)
    casefold: bool = False                 # приводить слова к нижнему регистру
    normalize: str = "minmax"               # "minmax" | "softmax" | "zscore"
    batch_size: int = 64
    tau: float = 0.60                       # порог для coverage

    # Топ-список
    topk: int = 12

    # Вспомогательные опции рендера
    auto_shrink_text: bool = True
    auto_expand_canvas: bool = True
    canvas_height_max_px: int = 4000
    show_reference: bool = False            # если включите — выведем реф-подпись ниже модели


def _coerce_cfg(config: Optional[Union[TOccConfig, Dict[str, Any]]]) -> TOccConfig:
    if isinstance(config, TOccConfig):
        return config
    d = dict(config or {})
    # Удаляем старые/неиспользуемые параметры
    old_params = ['dpi', 'fig_width', 'fig_height', 'text_col_width_ratio', 'merge_bpe', 'occlude_special']
    for param in old_params:
        d.pop(param, None)
    valid = {f.name for f in fields(TOccConfig)}
    d = {k: v for k, v in d.items() if k in valid}
    return TOccConfig(**d)


# --------------------------- Utils ------------------------------------

def _find_font(path: str, size: int):
    try:
        return ImageFont.truetype(path, size=size)
    except Exception:
        try:
            return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size=size)
        except Exception:
            return ImageFont.load_default()

def _text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont) -> Tuple[int,int]:
    try:
        b = draw.textbbox((0,0), text, font=font)
        return b[2]-b[0], b[3]-b[1]
    except Exception:
        return draw.textsize(text, font=font)

def load_image(path: str, fallback_hw: int) -> Image.Image:
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return Image.new("RGB", (fallback_hw, fallback_hw), (245,245,245))

def _fit_cover(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    sw, sh = target_w / img.width, target_h / img.height
    s = max(sw, sh)
    nw, nh = max(1, int(round(img.width * s))), max(1, int(round(img.height * s)))
    img2 = img.resize((nw, nh), Image.LANCZOS)
    ox, oy = (nw - target_w) // 2, (nh - target_h) // 2
    return img2.crop((ox, oy, ox + target_w, oy + target_h))

def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = F.normalize(a, dim=-1); b = F.normalize(b, dim=-1)
    return (a * b).sum(dim=-1)

_WORD_RE = re.compile(r"[A-Za-z]+(?:[-'][A-Za-z]+)*|\d+(?:\.\d+)?")
def split_words_from_text(text: str) -> List[str]:
    return _WORD_RE.findall(text or "")

def _normalize_scores(x: np.ndarray, mode: str = "minmax") -> np.ndarray:
    x = x.astype(np.float32)
    if mode == "softmax":
        e = np.exp(x - x.max()); return e / (e.sum() + 1e-8)
    if mode == "zscore":
        mu, sd = float(x.mean()), float(x.std() + 1e-8)
        z = (x - mu) / sd
        z = (z - z.min()) / (z.max() - z.min() + 1e-8)
        return z
    mn, mx = float(x.min()), float(x.max())
    if mx <= mn + 1e-8: return np.zeros_like(x)
    return (x - mn) / (mx - mn + 1e-8)


# --------------------------- Core: Token Occlusion ---------------------

@torch.inference_mode()
def _encode_image(clip, pil: Image.Image) -> torch.Tensor:
    # Унифицированный путь: если у backend есть .encode_image — используем его
    if hasattr(clip, "encode_image"):
        return clip.encode_image([pil])
    # Fallbacks (почти не понадобятся)
    if hasattr(clip, "preprocess") and hasattr(clip, "model"):
        px = clip.preprocess(pil).unsqueeze(0).to(clip.device)
        return clip.model.encode_image(px)
    raise RuntimeError("CLIP backend must provide encode_image")

@torch.inference_mode()
def _encode_texts_batch(clip, texts: List[str], batch_size: int = 32) -> torch.Tensor:
    if hasattr(clip, "encode_text"):
        outs = []
        for i in range(0, len(texts), batch_size):
            outs.append(clip.encode_text(texts[i:i+batch_size]))
        return torch.cat(outs, dim=0) if outs else torch.zeros((0, getattr(clip, "text_dim", 512)), device=clip.device)
    # Fallback ветка для «сырых» моделей
    if hasattr(clip, "tokenizer") and hasattr(clip, "model"):
        outs = []
        for i in range(0, len(texts), batch_size):
            inputs = clip.tokenizer(texts[i:i+batch_size], return_tensors='pt', padding=True, truncation=True, max_length=77).to(clip.device)
            outs.append(clip.model.get_text_features(**inputs))
        return torch.cat(outs, dim=0)
    raise RuntimeError("CLIP backend must provide encode_text or tokenizer+model")

def compute_token_occlusion(clip, image: Image.Image, text: str, cfg: TOccConfig
                            ) -> Tuple[List[str], List[float], Dict[str, Any]]:
    """
    Возвращает:
      words_or_tokens: список отображаемых единиц (слова или токены),
      scores: важности в [0,1],
      meta: словарь с метриками и топ-K.
    """
    device = clip.device
    # Базовый cos
    img_feat = _encode_image(clip, image)                      # [1, D]
    base_text_feat = _encode_texts_batch(clip, [text])         # [1, D]
    base_cos = float(cosine_sim(img_feat, base_text_feat)[0].item())

    # ----- По СЛОВАМ (по умолчанию) -----
    if cfg.unit.lower() == "word":
        words_full = split_words_from_text(text)
        if cfg.casefold:
            words_display = [w.lower() for w in words_full]
        else:
            words_display = words_full[:]

        if not words_display:
            return [], [], {"note": "empty caption", "base_cos": base_cos, "unit": "word"}

        # Варианты «без слова i»
        variants = []
        for i in range(len(words_display)):
            if len(words_display) == 1:
                variants.append("")  # пустой текст
            else:
                variants.append(" ".join(words_display[:i] + words_display[i+1:]))

        # Считаем падение cos
        drops = np.zeros(len(words_display), dtype=np.float32)
        bs = max(1, int(cfg.batch_size))
        for s in range(0, len(variants), bs):
            tfeat = _encode_texts_batch(clip, variants[s:s+bs])             # [B, D]
            sims = cosine_sim(img_feat.expand_as(tfeat), tfeat).detach().cpu().numpy()
            for j, sim_val in enumerate(sims):
                drop = max(0.0, base_cos - float(sim_val))
                idx = s + j
                if idx < len(drops):
                    drops[idx] = max(drops[idx], drop)

        # Нормируем
        scores = _normalize_scores(drops, cfg.normalize)
        # Метрики
        meta = {
            "base_cos": base_cos,
            "unit": "word",
            "normalize": cfg.normalize,
            "words_count": int(len(words_display)),
            "masks_sampled": int(len(variants)),
            "token_max": float(scores.max()) if scores.size else 0.0,
            "token_cov@tau": float((scores >= cfg.tau).mean()) if scores.size else 0.0,
        }
        # Топ-K
        order = scores.argsort()[::-1][: max(1, int(cfg.topk))]
        meta["top"] = [{"unit": words_display[i], "score": float(scores[i])} for i in order]
        return words_display, scores.tolist(), meta

    # ----- По ТОКЕНАМ (реже нужен; оставлен для совместимости) -----
    # Требует clip.tokenizer
    if not hasattr(clip, "tokenizer"):
        raise RuntimeError("Token mode requires `clip.tokenizer`")

    toks = clip.tokenizer(text, return_tensors='pt', padding=False, truncation=True, max_length=77)
    ids = toks['input_ids'][0].tolist()
    # выбрасываем BOS/EOS из эксперимента
    occ_indices = list(range(1, len(ids)-1)) if len(ids) >= 2 else list(range(len(ids)))

    variants = []
    for i in occ_indices:
        ids_i = ids[:i] + ids[i+1:]
        variants.append(clip.tokenizer.decode(ids_i, skip_special_tokens=True))

    drops = np.zeros(len(ids), dtype=np.float32)
    if variants:
        bs = max(1, int(cfg.batch_size))
        for s in range(0, len(variants), bs):
            tfeat = _encode_texts_batch(clip, variants[s:s+bs])
            sims = cosine_sim(img_feat.expand_as(tfeat), tfeat).detach().cpu().numpy()
            for j, sim_val in enumerate(sims):
                k = occ_indices[s + j]
                drop = max(0.0, base_cos - float(sim_val))
                drops[k] = max(drops[k], drop)

    # нормируем только по исследуемым индексам
    scores = drops.copy()
    if occ_indices:
        scores[occ_indices] = _normalize_scores(drops[occ_indices], cfg.normalize)
    # отображаемые «токены»
    words_display = clip.tokenizer.convert_ids_to_tokens(ids)
    words_display = [t.lstrip("Ġ▁").replace("</w>", "") for t in words_display]
    # метрики
    sel = scores[occ_indices] if occ_indices else scores
    meta = {
        "base_cos": base_cos,
        "unit": "token",
        "normalize": cfg.normalize,
        "words_count": int(len(words_display)),
        "masks_sampled": int(len(variants)),
        "token_max": float(sel.max()) if sel.size else 0.0,
        "token_cov@tau": float((sel >= cfg.tau).mean()) if sel.size else 0.0,
    }
    order = sel.argsort()[::-1][: max(1, int(cfg.topk))]
    top = [{"unit": words_display[occ_indices[i]], "score": float(sel[i])} for i in order]
    meta["top"] = top
    return words_display, scores.tolist(), meta


# --------------- Measuring & Rendering (vertical like previous) --------

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
        if c > 0.05:
            v = int(255 * (1.0 - 0.8 * c))      # синяя подсветка — как раньше
            drw.rectangle([xx - 2, yy - 2, xx + tw, yy + th + 2], fill=(v, v, 255), outline=None)
        drw.text((xx, yy), piece, font=font, fill=(0,0,0))
        xx += tw
    return yy + line_h

def _measure_kv_block_height(drw, items: List[Tuple[str, str]], font) -> int:
    h = 0
    for k, v in items:
        h += _text_size(drw, f"{k}: {v}", font)[1] + 4
    return h

def _render_card_vertical(
    out_png: Path,
    image: Image.Image,
    gen_caption: str,
    ref_caption: Optional[str],
    words: List[str],
    scores: List[float],
    metrics: Dict[str, Any],
    cfg: TOccConfig,
):
    W = cfg.canvas_width_px
    M, GAP = cfg.margins_px, cfg.block_gap_px
    col_w = W - 2 * M

    # Предварительные шрифты для измерений
    f_hdr = _find_font(cfg.font_path_regular, cfg.fs_header)
    f_txt = _find_font(cfg.font_path_regular, cfg.fs_text)
    f_sml = _find_font(cfg.font_path_regular, cfg.fs_small)

    tmp = Image.new("RGB", (W, 200), (255,255,255))
    mdrw = ImageDraw.Draw(tmp)

    def measure_total(fs_caption: int) -> int:
        nonlocal f_hdr, f_txt, f_sml
        f_hdr = _find_font(cfg.font_path_regular, cfg.fs_header)
        f_txt = _find_font(cfg.font_path_regular, fs_caption)
        f_sml = _find_font(cfg.font_path_regular, cfg.fs_small)
        total = 0
        total += _text_size(mdrw, "Original image", f_hdr)[1] + 6
        total += cfg.image_block_height_px + GAP
        total += _text_size(mdrw, "Generated caption (Token Occlusion, words)", f_hdr)[1] + 6
        total += _wrap_words_height(mdrw, col_w, words if words else gen_caption.split(), f_txt) + GAP
        if cfg.show_reference and ref_caption:
            total += _text_size(mdrw, "Reference caption", f_hdr)[1] + 6
            total += _wrap_words_height(mdrw, col_w, ref_caption.split(), f_txt) + GAP
        total += _text_size(mdrw, "Token importance (low → high)", f_sml)[1] + 4
        total += 14 + 26 + GAP
        items = [
            ("Косинус(изобр., полный текст)", f"{float(metrics.get('base_cos', 0.0)):.3f}"),
            ("Максимальная важность слова", f"{float(metrics.get('token_max', 0.0)):.3f}"),
            (f"Покрытие ≥ {cfg.tau:.2f}", f"{float(metrics.get('token_cov@tau', 0.0)):.3f}"),
            ("N — число слов (features)", str(int(metrics.get("words_count", 0)))),
            ("M — число сэмплированных масок", str(int(metrics.get("masks_sampled", 0)))),
        ]
        total += _text_size(mdrw, "Metrics (explained)", f_hdr)[1] + 6
        total += _measure_kv_block_height(mdrw, items, f_sml) + GAP
        total += _text_size(mdrw, "Top words", f_hdr)[1] + 6
        bar_h = max(14, f_sml.size + 4)
        total += min(cfg.topk, len(words)) * (bar_h + 6)
        footer = "Token Occlusion • CLIP text attribution (word-level) • cos(image, text)"
        total += _text_size(mdrw, footer, f_sml)[1] + 8
        return total

    fs_caption = cfg.fs_text
    total = measure_total(fs_caption)
    if cfg.auto_shrink_text:
        while total + 2 * M > cfg.canvas_height_max_px and fs_caption > cfg.fs_text_min:
            fs_caption -= 1
            total = measure_total(fs_caption)

    H = max(cfg.canvas_height_px, total + 2 * M)
    if cfg.auto_expand_canvas:
        H = min(max(H, cfg.canvas_height_px), cfg.canvas_height_max_px)

    # ---- RENDER ----
    canvas = Image.new("RGB", (W, H), (255,255,255))
    drw = ImageDraw.Draw(canvas)
    f_hdr = _find_font(cfg.font_path_regular, cfg.fs_header)
    f_txt = _find_font(cfg.font_path_regular, fs_caption)
    f_sml = _find_font(cfg.font_path_regular, cfg.fs_small)

    x = M; y = M

    # 1) ОРИГИНАЛ
    drw.text((x, y), "Original image", font=f_hdr, fill=(0,0,0))
    y += _text_size(drw, "Original image", f_hdr)[1] + 6
    img_fit = _fit_cover(image, col_w, cfg.image_block_height_px)
    canvas.paste(img_fit, (x, y))
    y += cfg.image_block_height_px + GAP

    # 2) CAPTION c подсветкой
    title = "Generated caption (Token Occlusion, words)"
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
    for i in range(col_w):
        t = i / max(1, col_w - 1)
        v = int(255 * (1 - 0.8 * t))
        drw.line([(x + i, y), (x + i, y + 14)], fill=(v, v, 255))
    drw.rectangle([x, y, x + col_w, y + 14], outline=(0,0,0), width=1)
    drw.text((x, y + 18), "Low", font=f_sml, fill=(0,0,0))
    drw.text((x + col_w - _text_size(drw, "High", f_sml)[0], y + 18), "High", font=f_sml, fill=(0,0,0))
    y += 14 + 26 + GAP

    # 4) МЕТРИКИ
    drw.text((x, y), "Metrics (explained)", font=f_hdr, fill=(0,0,0))
    y += _text_size(drw, "Metrics (explained)", f_hdr)[1] + 6

    def kv(k: str, v: str):
        nonlocal y
        drw.text((x, y), f"{k}: {v}", font=f_sml, fill=(0,0,0))
        y += _text_size(drw, f"{k}: {v}", f_sml)[1] + 4

    kv("Косинус(изобр., полный текст)", f"{float(metrics.get('base_cos', 0.0)):.3f}")
    kv("Максимальная важность слова", f"{float(metrics.get('token_max', 0.0)):.3f}")
    kv(f"Покрытие ≥ {cfg.tau:.2f}", f"{float(metrics.get('token_cov@tau', 0.0)):.3f}")
    kv("N — число слов (features)", str(int(metrics.get("words_count", 0))))
    kv("M — число сэмплированных масок", str(int(metrics.get("masks_sampled", 0))))
    y += GAP

    # 5) TOP WORDS (с цифрами)
    drw.text((x, y), "Top words", font=f_hdr, fill=(0,0,0))
    y += _text_size(drw, "Top words", f_hdr)[1] + 6
    pairs = [(w, float(s)) for w, s in zip(words_to_show, scores_to_show)]
    pairs.sort(key=lambda t: t[1], reverse=True)
    pairs = pairs[: cfg.topk]
    bar_h = max(14, f_sml.size + 4)
    for name, sc in pairs:
        c = max(0.0, min(1.0, sc))
        nm = name[:28] + ("…" if len(name) > 28 else "")
        tw, th = _text_size(drw, nm, f_sml)
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
            tx = max(bx + 2, x + col_w - svw)
            drw.text((tx, y), sval, font=f_sml, fill=(0,0,0))
        y += bar_h + 6

    # футер
    footer = "Token Occlusion • CLIP text attribution (word-level) • cos(image, text)"
    fw, fh = _text_size(drw, footer, f_sml)
    drw.text((W - fw - M, H - fh - 6), footer, font=f_sml, fill=(90,90,90))

    canvas.save(str(out_png), format="PNG")


# --------------------------- Public API --------------------------------

def run(clip, captioner, *, image_path: str, out_dir: Path,
        ref_caption: Optional[str] = None,
        config: Optional[Union[TOccConfig, Dict[str, Any]]] = None) -> Dict[str, Any]:

    cfg = _coerce_cfg(config)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # Проверяем, что модель работает
    print(f"[DEBUG] Processing image: {Path(image_path).name}")
    print(f"[DEBUG] CLIP device: {clip.device}, kind: {getattr(clip, 'kind', 'unknown')}")
    
    image = load_image(image_path, fallback_hw=getattr(clip, "image_size", 224))
    try:
        print(f"[DEBUG] Generating caption with prompt: {captioner.prompt[:50]}...")
        gen_caption = captioner.generate([image], prompt=getattr(captioner, "prompt", None) or "Describe the image.")[0]
        print(f"[DEBUG] Generated caption: {gen_caption}")
        
        if not gen_caption or len(gen_caption.strip()) < 3:
            print(f"[WARNING] Empty or very short caption generated: '{gen_caption}'")
            gen_caption = "A medical image showing skin lesion."
            
    except Exception as e:
        print(f"[ERROR] Caption generation failed: {e}")
        gen_caption = "A medical image showing skin lesion."

    # считаем occlusion важности
    try:
        print(f"[DEBUG] Computing token occlusion for caption: '{gen_caption}'")
        units, scores, meta = compute_token_occlusion(clip, image, gen_caption, cfg)
        print(f"[DEBUG] Found {len(units)} units with scores, base_cos: {meta.get('base_cos', 0):.3f}")
    except Exception as e:
        print(f"[ERROR] Token occlusion computation failed: {e}")
        # Fallback данные
        units = gen_caption.split()[:10]
        scores = [0.1] * len(units)
        meta = {"base_cos": 0.0, "unit": "word", "error": str(e)}

    # рендер карточки
    stem = Path(image_path).stem
    out_png = out_dir / f"{stem}.png"
    try:
        _render_card_vertical(out_png, image, gen_caption, ref_caption if cfg.show_reference else None,
                              units, scores, meta, cfg)
        print(f"[DEBUG] Rendered card: {out_png}")
    except Exception as e:
        print(f"[ERROR] Card rendering failed: {e}")

    # json вывод
    out_json = out_dir / f"{stem}.json"
    data = {
        "image_path": image_path,
        "method": "token_occlusion_wordlevel" if cfg.unit.lower()=="word" else "token_occlusion_tokenlevel",
        "config": asdict(cfg),
        "caption_generated": gen_caption,
        "caption_reference": (ref_caption if cfg.show_reference else None),
        "display_units": units,
        "scores": scores,
        "meta": meta,
        "outputs": {"png": str(out_png)}
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return data
