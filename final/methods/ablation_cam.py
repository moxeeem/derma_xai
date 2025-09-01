# final/methods/ablation_cam.py
# -*- coding: utf-8 -*-
"""
Ablation-CAM for CLIP Vision: затыкаем регионы и меряем падение cos(img, text).
Вертикальная узкая карточка:
[Original image] → [Ablation-CAM overlay] → [Legend] → [Metrics (explained)]
Без SciPy/Matplotlib. Единый шрифт DejaVu Sans Regular.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict, fields
from typing import Optional, Dict, Any, Union, Tuple, List
from pathlib import Path
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

import torch
import torch.nn.functional as F


# ----------------------------- Config ---------------------------------

@dataclass
class AblCamConfig:
    # Узкий формат; высота — авто
    canvas_width_px: int = 720
    canvas_height_px: int = 1400
    margins_px: int = 16
    block_gap_px: int = 10

    # Блоки изображений
    image_block_height_px: int = 360   # фикс-высота (cover-crop) для Original и Overlay

    # Шрифты — один и тот же, как в matplotlib
    font_path_regular: str = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    fs_header: int = 22
    fs_text: int = 20
    fs_small: int = 16
    fs_text_min: int = 14
    line_pad_px: int = 6

    # Overlay
    overlay_alpha: float = 0.45
    alpha_mode: str = "scaled"       # scaled | uniform
    gamma: float = 0.9               # <1 → горячее
    p_low: float = 80.0              # percentile stretch
    p_high: float = 99.5
    tau: float = 0.60                # для coverage ≥ τ

    # Ablation
    patch_px: int = 64               # размер окна
    stride_px: int = 32              # шаг окна
    baseline: str = "blur"           # blur | mean | zero
    blur_sigma_px: float = 8.0
    batch_vision: int = 32

    # Рендер-поведение
    auto_shrink_text: bool = True
    auto_expand_canvas: bool = True
    canvas_height_max_px: int = 4000
    show_reference: bool = False      # если True — покажем реф-подпись под модельной


def _coerce_cfg(config: Optional[Union[AblCamConfig, Dict[str, Any]]]) -> AblCamConfig:
    if isinstance(config, AblCamConfig):
        return config
    d = dict(config or {})
    valid = {f.name for f in fields(AblCamConfig)}
    d = {k: v for k, v in d.items() if k in valid}
    return AblCamConfig(**d)


# ----------------------------- Utils ----------------------------------
def enhance_heatmap(heat: np.ndarray, enhance_factor: float = 1.2) -> np.ndarray:
    """Улучшает контраст тепловой карты для более чёткого выделения областей"""
    heat_enhanced = heat ** enhance_factor
    return normalize01(heat_enhanced)


def load_image(path: str, fallback_hw: int) -> Image.Image:
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return Image.new("RGB", (fallback_hw, fallback_hw), (245, 245, 245))

def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = F.normalize(a, dim=-1); b = F.normalize(b, dim=-1)
    return (a * b).sum(dim=-1)

def normalize01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    mn, mx = float(x.min()), float(x.max())
    if mx <= mn + 1e-8:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn + 1e-8)

def _fit_cover(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    sw, sh = target_w / img.width, target_h / img.height
    s = max(sw, sh)
    nw, nh = max(1, int(round(img.width * s))), max(1, int(round(img.height * s)))
    img2 = img.resize((nw, nh), Image.LANCZOS)
    ox, oy = (nw - target_w) // 2, (nh - target_h) // 2
    return img2.crop((ox, oy, ox + target_w, oy + target_h))

def _apply_colormap_reds(gray01: np.ndarray) -> np.ndarray:
    g = np.clip(gray01, 0, 1).astype(np.float32)
    r = (0.9 * g + 0.1) * 255.0
    gb = (1.0 - 0.85 * g) * 255.0
    return np.clip(np.stack([r, gb, gb], axis=-1), 0, 255).astype(np.uint8)

def _overlay_with_heat(base: Image.Image, heat01: np.ndarray, alpha: float, alpha_mode: str = "scaled") -> Image.Image:
    """Накладываем тепловую карту (0..1) поверх base (уже нужного размера)."""
    W, H = base.size
    hm_rgb = _apply_colormap_reds(heat01)
    hm_img = Image.fromarray(hm_rgb).resize((W, H), Image.BICUBIC)
    if alpha_mode.lower() == "uniform":
        a_img = Image.new("L", (W, H), int(np.clip(alpha, 0, 1) * 255))
    else:
        a = (normalize01(heat01) * (alpha * 255)).astype(np.uint8)
        a_img = Image.fromarray(a, "L").resize((W, H), Image.BICUBIC)
    out = base.copy()
    out.paste(hm_img, (0, 0), a_img)
    return out

def _find_font(path: str, size: int):
    try:
        return ImageFont.truetype(path, size=size)
    except Exception:
        try:
            return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size=size)
        except Exception:
            return ImageFont.load_default()

def _text_size(drw: ImageDraw.ImageDraw, text: str, font) -> Tuple[int, int]:
    try:
        b = drw.textbbox((0, 0), text, font=font)
        return b[2] - b[0], b[3] - b[1]
    except Exception:
        return drw.textsize(text, font=font)

def _wrap_text(drw, x: int, y: int, max_w: int, text: str, font, gap=6) -> int:
    if not text:
        return y
    words, line, yy = text.split(" "), "", y
    for w in words:
        t = w if not line else (line + " " + w)
        tw, th = _text_size(drw, t, font)
        if tw <= max_w:
            line = t
        else:
            drw.text((x, yy), line, font=font, fill=(0, 0, 0)); yy += th + gap
            line = w
    if line:
        drw.text((x, yy), line, font=font, fill=(0, 0, 0))
        yy += _text_size(drw, line, font)[1]
    return yy

def _draw_gradient_bar_reds(drw, x: int, y: int, w: int, h: int):
    for i in range(w):
        t = i / max(1, w - 1)
        # белый → насыщенно-красный
        r = int(255 * (0.1 + 0.9 * t))
        gb = int(255 * (1 - 0.85 * t))
        drw.line([(x + i, y), (x + i, y + h)], fill=(r, gb, gb))
    drw.rectangle([x, y, x + w, y + h], outline=(0, 0, 0), width=1)

def heatmap_stats(hm01: np.ndarray, tau: float) -> Dict[str, float]:
    h = normalize01(hm01)
    cov = float((h >= tau).mean())
    flat = np.sort(h.reshape(-1))[::-1]
    k = max(1, int(0.10 * flat.size))
    mass_top10 = float(flat[:k].sum() / (flat.sum() + 1e-8))
    entropy = float(-np.sum(h * np.log(h + 1e-8)) / h.size)
    return {"coverage@tau": cov, "mass@top10%": mass_top10, "entropy": entropy}


# ------------------------- Core: Ablation-CAM --------------------------

def _iter_windows(W: int, H: int, patch: int, stride: int):
    for y0 in range(0, H, stride):
        for x0 in range(0, W, stride):
            x1, y1 = min(W, x0 + patch), min(H, y0 + patch)
            yield x0, y0, x1, y1

@torch.inference_mode()
def compute_ablation_cam(clip, image: Image.Image, text: str, cfg: AblCamConfig
                         ) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    """
    Возвращает heatmap (H×W, 0..1), базовый cos(img, text), meta.
    """
    # Эмбеддинги (унифицированный API backend'а)
    tfeat = clip.encode_text([text])           # [1, D]
    ifeat = clip.encode_image([image])         # [1, D]
    base_cos = float(cosine_sim(ifeat, tfeat)[0].item())

    W, H = image.size

    # Baseline для «затычки»
    if cfg.baseline == "blur":
        base_img = image.filter(ImageFilter.GaussianBlur(radius=float(cfg.blur_sigma_px)))
    elif cfg.baseline == "mean":
        arr = np.asarray(image).astype(np.float32)
        mean_color = tuple(int(c) for c in arr.reshape(-1, 3).mean(axis=0))
        base_img = Image.new("RGB", (W, H), mean_color)
    else:
        base_img = Image.new("RGB", (W, H), (0, 0, 0))

    heat = np.zeros((H, W), dtype=np.float32)
    cnt = np.zeros((H, W), dtype=np.float32)

    # Оптимизация: кэширование базового изображения
    base_img_cache = {}
    
    # Оптимизация: параллельная обработка в более крупных батчах
    batch: List[Tuple[Image.Image, Tuple[int,int,int,int]]] = []
    all_windows = list(_iter_windows(W, H, int(cfg.patch_px), int(cfg.stride_px)))
    
    # Расчет и прогресс
    total_windows = len(all_windows)
    batch_size = min(int(cfg.batch_vision), total_windows)
    
    # Более эффективное построение батчей
    for i in range(0, total_windows, batch_size):
        current_windows = all_windows[i:i+batch_size]
        batch = []
        for x0, y0, x1, y1 in current_windows:
            # Повторное использование объектов для экономии памяти
            region_key = (x0, y0, x1, y1)
            if region_key not in base_img_cache:
                base_img_cache[region_key] = base_img.crop((x0, y0, x1, y1))
            
            im = image.copy()
            im.paste(base_img_cache[region_key], (x0, y0))
            batch.append((im, (x0, y0, x1, y1)))
        
        _flush_batch(clip, batch, tfeat, base_cos, heat, cnt)
        
        # Очистка кэша, если он слишком большой
        if len(base_img_cache) > 100:
            base_img_cache.clear()
    # if batch:
    #     _flush_batch(clip, batch, tfeat, base_cos, heat, cnt)

    # Усреднение по покрытию + нормировка
    heat /= (cnt + 1e-6)
    heat = normalize01(heat)

    # Percentile stretch + gamma
    try:
        lo = np.percentile(heat, float(cfg.p_low))
        hi = np.percentile(heat, float(cfg.p_high))
        if hi > lo:
            heat = np.clip((heat - lo) / (hi - lo + 1e-8), 0, 1)
    except Exception:
        pass
    if cfg.gamma and cfg.gamma != 1.0:
        heat = np.power(heat, float(max(0.05, cfg.gamma)))
        heat = normalize01(heat)

    # Улучшение контраста тепловой карты
    heat = enhance_heatmap(heat)

    meta = {
        "patch_px": int(cfg.patch_px),
        "stride_px": int(cfg.stride_px),
        "baseline": str(cfg.baseline),
        "overlay_alpha": float(cfg.overlay_alpha),
        "alpha_mode": str(cfg.alpha_mode),
        "gamma": float(cfg.gamma),
        "p_low": float(cfg.p_low),
        "p_high": float(cfg.p_high),
    }
    return heat, base_cos, meta

def _flush_batch(clip, batch: List[Tuple[Image.Image, Tuple[int,int,int,int]]],
                 tfeat: torch.Tensor, base_cos: float,
                 heat: np.ndarray, cnt: np.ndarray):
    if not batch:
        return
        
    # Оптимизация: предварительно выделяем память под списки
    batch_size = len(batch)
    ims = [None] * batch_size
    boxes = [None] * batch_size
    
    # Заполняем списки
    for i, (im, box) in enumerate(batch):
        ims[i] = im
        boxes[i] = box
    
    # Обработка на GPU в одном батче - используем новый API
    with torch.amp.autocast('cuda', enabled=True):  # Обновленный синтаксис для смешанной точности
        feats = clip.encode_image(ims)  # [B, D]
        sims = cosine_sim(feats, tfeat.expand_as(feats))
    
    # Перенос на CPU только когда нужно
    drops = np.maximum(0.0, base_cos - sims.detach().cpu().numpy())
    
    # Быстрое обновление тепловой карты
    for (x0, y0, x1, y1), d in zip(boxes, drops):
        heat[y0:y1, x0:x1] += float(d)
        cnt[y0:y1, x0:x1] += 1.0


# --------------------------- Renderer (vertical) -----------------------

def _measure_total_height(W: int, cfg: AblCamConfig, image: Image.Image,
                          gen_caption: str, ref_caption: Optional[str],
                          f_hdr, f_txt, f_sml) -> int:
    tmp = Image.new("RGB", (W, 200), (255, 255, 255))
    drw = ImageDraw.Draw(tmp)
    M, GAP = cfg.margins_px, cfg.block_gap_px
    col_w = W - 2 * M
    total = 0
    # Original
    total += _text_size(drw, "Original image", f_hdr)[1] + 6
    total += cfg.image_block_height_px + GAP
    # Overlay
    total += _text_size(drw, "Ablation-CAM overlay", f_hdr)[1] + 6
    total += cfg.image_block_height_px + GAP
    # Caption
    total += _text_size(drw, "Generated caption (for CLIP score)", f_hdr)[1] + 6
    total += _wrap_text(drw, 0, 0, col_w, gen_caption or "(empty)", f_txt, cfg.line_pad_px) - 0 + GAP
    if cfg.show_reference and ref_caption:
        total += _text_size(drw, "Reference caption", f_hdr)[1] + 6
        total += _wrap_text(drw, 0, 0, col_w, ref_caption, f_txt, cfg.line_pad_px) - 0 + GAP
    # Legend
    total += _text_size(drw, "Heat (low → high)", f_sml)[1] + 4
    total += 14 + 26 + GAP
    # Metrics
    total += _text_size(drw, "Metrics (explained)", f_hdr)[1] + 6
    items = [
        ("Косинус(изобр., полный текст)", ""),  # значения нарисуем на рендере
        (f"Покрытие ≥ {cfg.tau:.2f}", ""),
        ("Масса топ-10% областей", ""),
        ("Энтропия карты", ""),
        (f"Окно/шаг, baseline", ""),
        ("Stretch P-low..P-high; γ", ""),
    ]
    for k, _ in items:
        total += _text_size(drw, f"{k}: 0.000", f_sml)[1] + 4
    # Footer
    total += _text_size(drw, "Ablation-CAM • CLIP vision attribution", f_sml)[1] + 8
    return total

def _render_card_vertical(out_png: Path, image: Image.Image, heat01: np.ndarray,
                          gen_caption: str, ref_caption: Optional[str],
                          base_cos: float, meta: Dict[str, Any], cfg: AblCamConfig):
    W = cfg.canvas_width_px
    M, GAP = cfg.margins_px, cfg.block_gap_px
    col_w = W - 2 * M

    # Шрифты
    f_hdr = _find_font(cfg.font_path_regular, cfg.fs_header)
    f_txt = _find_font(cfg.font_path_regular, cfg.fs_text)
    f_sml = _find_font(cfg.font_path_regular, cfg.fs_small)

    # Авто-высота / авто-кегль
    fs_caption = cfg.fs_text
    total = _measure_total_height(W, cfg, image, gen_caption, ref_caption if cfg.show_reference else None,
                                  f_hdr, f_txt, f_sml)
    if cfg.auto_shrink_text:
        while total + 2 * M > cfg.canvas_height_max_px and fs_caption > cfg.fs_text_min:
            fs_caption -= 1
            f_txt = _find_font(cfg.font_path_regular, fs_caption)
            total = _measure_total_height(W, cfg, image, gen_caption, ref_caption if cfg.show_reference else None,
                                          f_hdr, f_txt, f_sml)

    H = max(cfg.canvas_height_px, total + 2 * M)
    if cfg.auto_expand_canvas:
        H = min(max(H, cfg.canvas_height_px), cfg.canvas_height_max_px)

    # ---- RENDER ----
    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    drw = ImageDraw.Draw(canvas)

    x = M; y = M

    # 1) Original
    drw.text((x, y), "Original image", font=f_hdr, fill=(0, 0, 0))
    y += _text_size(drw, "Original image", f_hdr)[1] + 6
    img_fit = _fit_cover(image, col_w, cfg.image_block_height_px)
    canvas.paste(img_fit, (x, y))
    y += cfg.image_block_height_px + GAP

    # 2) Overlay (heat resized под блок)
    drw.text((x, y), "Ablation-CAM overlay", font=f_hdr, fill=(0, 0, 0))
    y += _text_size(drw, "Ablation-CAM overlay", f_hdr)[1] + 6
    base_block = _fit_cover(image, col_w, cfg.image_block_height_px)
    # Приводим heat к геометрии исходника, затем ресайзим под блок
    overlay = _overlay_with_heat(base_block, heat01, float(cfg.overlay_alpha), cfg.alpha_mode)
    canvas.paste(overlay, (x, y))
    y += cfg.image_block_height_px + GAP

    # 3) Caption
    drw.text((x, y), "Generated caption (for CLIP score)", font=f_hdr, fill=(0, 0, 0))
    y += _text_size(drw, "Generated caption (for CLIP score)", f_hdr)[1] + 6
    y = _wrap_text(drw, x, y, col_w, gen_caption or "(empty)", f_txt, cfg.line_pad_px) + GAP
    if cfg.show_reference and ref_caption:
        drw.text((x, y), "Reference caption", font=f_hdr, fill=(0, 0, 0))
        y += _text_size(drw, "Reference caption", f_hdr)[1] + 6
        y = _wrap_text(drw, x, y, col_w, ref_caption, f_txt, cfg.line_pad_px) + GAP

    # 4) Legend
    drw.text((x, y), "Heat (low → high)", font=f_sml, fill=(0, 0, 0))
    y += _text_size(drw, "Heat (low → high)", f_sml)[1] + 4
    _draw_gradient_bar_reds(drw, x, y, col_w, 14)
    drw.text((x, y + 18), "Low", font=f_sml, fill=(0, 0, 0))
    drw.text((x + col_w - _text_size(drw, "High", f_sml)[0], y + 18), "High", font=f_sml, fill=(0, 0, 0))
    y += 14 + 26 + GAP

    # 5) Metrics
    drw.text((x, y), "Metrics (explained)", font=f_hdr, fill=(0, 0, 0))
    y += _text_size(drw, "Metrics (explained)", f_hdr)[1] + 6

    def kv(k: str, v: str):
        nonlocal y
        line = f"{k}: {v}"
        drw.text((x, y), line, font=f_sml, fill=(0, 0, 0))
        y += _text_size(drw, line, f_sml)[1] + 4

    stats = heatmap_stats(heat01, tau=float(cfg.tau))
    kv("Косинус(изобр., полный текст)", f"{base_cos:.3f}")
    kv(f"Покрытие ≥ {cfg.tau:.2f}", f"{stats['coverage@tau']:.3f}")
    kv("Масса топ-10% областей", f"{stats['mass@top10%']:.3f}")
    kv("Энтропия карты", f"{stats['entropy']:.3f}")
    kv("Окно/шаг, baseline", f"{int(meta['patch_px'])}/{int(meta['stride_px'])} px; {meta['baseline']}")
    kv("Stretch P-low..P-high; γ", f"P{meta['p_low']:.1f}..P{meta['p_high']:.1f}; {meta['gamma']:.2f}")

    # футер
    footer = "Ablation-CAM • CLIP vision attribution • cos(image, text)"
    fw, fh = _text_size(drw, footer, f_sml)
    drw.text((W - fw - M, H - fh - 6), footer, font=f_sml, fill=(90, 90, 90))

    canvas.save(str(out_png), format="PNG")


# ----------------------------- Public API ------------------------------

def run(clip, captioner, *, image_path: str, out_dir: Path,
        ref_caption: Optional[str] = None,
        config: Optional[Union[AblCamConfig, Dict[str, Any]]] = None) -> Dict[str, Any]:

    cfg = _coerce_cfg(config)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    image = load_image(image_path, fallback_hw=getattr(clip, "image_size", 224))
    try:
        gen_caption = captioner.generate([image], prompt=getattr(captioner, "prompt", None) or "Describe the image.")[0]
    except Exception:
        gen_caption = ""

    heat, base_cos, meta = compute_ablation_cam(clip, image, gen_caption, cfg)

    stem = Path(image_path).stem
    out_png = out_dir / f"{stem}.png"
    _render_card_vertical(out_png, image, heat, gen_caption,
                          (ref_caption if cfg.show_reference else None),
                          base_cos, meta, cfg)

    stats = heatmap_stats(heat, tau=float(cfg.tau))
    metrics = {"cos": base_cos, **stats, **meta}

    out_json = out_dir / f"{stem}.json"
    data = {
        "image_path": image_path,
        "method": "ablation_cam",
        "config": asdict(cfg),
        "caption_generated": gen_caption,
        "caption_reference": (ref_caption if cfg.show_reference else None),
        "meta": metrics,
        "outputs": {"png": str(out_png)}
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return data
