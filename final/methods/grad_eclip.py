# methods/score_cam.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass, asdict, fields
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import json, numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import torch, torch.nn.functional as F
import math

# ----------------------------- Config ---------------------------------
@dataclass
class ScoreCAMConfig:
    # Верстка (как у твоего идеального Score-CAM)
    canvas_width_px: int = 720
    canvas_height_px: int = 1400
    margins_px: int = 16
    block_gap_px: int = 10
    image_block_height_px: int = 360
    font_path_regular: str = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    fs_header: int = 22; fs_text: int = 20; fs_small: int = 16; fs_text_min: int = 14
    line_pad_px: int = 6

    # Совместимость полей (рендер это ожидает; для grad-eCLIP они «декоративные»)
    grid_n: int = 0
    work_resize_px: int = 0
    baseline: str = "n/a"
    blur_sigma_px: float = 0.0
    batch_vision: int = 0
    mask_shape: str = "grad-eclip"
    stride_frac: float = 0.0
    sigma_frac: float = 0.0
    edge_soften_px: int = 0

    # Порог для coverage
    tau: float = 0.70
    
    overlay_alpha: float = 0.65  # Было 0.45 - увеличиваем непрозрачность
    alpha_mode: str = "scaled"   # "scaled" - прозрачность зависит от интенсивности, "uniform" - одинаковая

    # Постобработка теплокарты - усиливаем контраст и фокус
    p_low: float = 85.0  # Повышаем с 60.0 до 85.0 для удаления шума
    p_high: float = 98.0  # Оставляем как есть
    gamma: float = 0.5    # Уменьшаем с 0.7 до 0.5 для большего контраста
    heat_smooth_sigma_px: float = 0.8  # Уменьшаем с 1.0 для меньшего размытия

    # Дополнительные параметры для усиления фокуса
    focus_threshold: float = 0.4  # Порог для удаления слабых сигналов
    focus_boost: float = 1.3      # Коэффициент усиления для значимых областей

    # Overlay - делаем более заметным
    overlay_alpha: float = 0.65  # Было 0.45 - увеличиваем непрозрачность

    # Рендер
    auto_shrink_text: bool = True
    auto_expand_canvas: bool = True
    canvas_height_max_px: int = 4000
    show_reference: bool = True

    # Настройки градиентной карты
    grad_aggregate: str = "gradxinput"  # Оставляем как есть, но можно попробовать "absgrad"
    grad_pool: str = "sum"  # Было "mean" - меняем на "sum" для усиления сигнала
    safe_float32: bool = True            # насильно считать в fp32 (меньше сюрпризов)

def _coerce_cfg(config: Optional[Dict[str, Any]]) -> ScoreCAMConfig:
    if isinstance(config, ScoreCAMConfig): return config
    d = dict(config or {}); valid = {f.name for f in fields(ScoreCAMConfig)}
    return ScoreCAMConfig(**{k:v for k,v in d.items() if k in valid})

# --------------------------- Utils ------------------------------------
def normalize01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32); mn, mx = float(x.min()), float(x.max())
    return np.zeros_like(x) if mx <= mn + 1e-8 else (x - mn) / (mx - mn + 1e-8)

def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = F.normalize(a, dim=-1); b = F.normalize(b, dim=-1); return (a*b).sum(dim=-1)

def load_image(path: str, fallback_hw: int) -> Image.Image:
    try: return Image.open(path).convert('RGB')
    except Exception: return Image.new('RGB', (fallback_hw, fallback_hw), (245,245,245))

def _find_font(path: str, size: int):
    try: return ImageFont.truetype(path, size=size)
    except Exception:
        try: return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size=size)
        except Exception: return ImageFont.load_default()

def _text_size(drw, text: str, font) -> Tuple[int,int]:
    try: b = drw.textbbox((0,0), text, font=font); return b[2]-b[0], b[3]-b[1]
    except Exception: return drw.textsize(text, font=font)

def _fit_cover(img: Image.Image, W: int, H: int) -> Image.Image:
    s = max(W/img.width, H/img.height)
    nw, nh = max(1,int(round(img.width*s))), max(1,int(round(img.height*s)))
    img2 = img.resize((nw,nh), Image.LANCZOS); ox, oy = (nw-W)//2, (nh-H)//2
    return img2.crop((ox,oy,ox+W,oy+H))

def _fit_cover_gray(gray01: np.ndarray, W: int, H: int) -> np.ndarray:
    h,w = gray01.shape; s = max(W/w, H/h)
    nw, nh = max(1,int(round(w*s))), max(1,int(round(h*s)))
    pil = Image.fromarray((normalize01(gray01)*255).astype(np.uint8),'L').resize((nw,nh), Image.BICUBIC)
    ox, oy = (nw-W)//2, (nh-H)//2
    return np.asarray(pil.crop((ox,oy,ox+W,oy+H))).astype(np.float32)/255.0

def _apply_colormap_reds(g: np.ndarray) -> np.ndarray:
    g = np.clip(g,0,1).astype(np.float32)
    # Усиливаем красный канал и ослабляем зеленый и синий для более ярких тепловых карт
    r = (0.95*g+0.05)*255.0  # Больший контраст в красном (было 0.9*g+0.1)
    gb = (1.0-0.95*g)*255.0  # Сильнее подавляем зеленый и синий (было 0.85)
    return np.clip(np.stack([r,gb,gb],-1),0,255).astype(np.uint8)

def _overlay_with_heat(base: Image.Image, heat01: np.ndarray, alpha: float, alpha_mode: str="scaled") -> Image.Image:
    W,H = base.size; heat = normalize01(heat01)
    if heat.shape != (H,W):
        heat = np.array(Image.fromarray((heat*255).astype(np.uint8),'L').resize((W,H), Image.BICUBIC)).astype(np.float32)/255.0
    hm_img = Image.fromarray(_apply_colormap_reds(heat))
    if str(alpha_mode).lower() == "uniform":
        a_img = Image.new("L",(W,H),int(np.clip(alpha,0,1)*255))
    else:
        a = np.clip(heat, 0.0, 1.0) * float(np.clip(alpha, 0.0, 1.0))
        a_img = Image.fromarray((a*255).astype(np.uint8), "L")
    out = base.copy(); out.paste(hm_img,(0,0),a_img); return out

def heatmap_stats(hm: np.ndarray, tau: float=0.6) -> Dict[str,float]:
    h = normalize01(hm); cov = float((h>=tau).mean())
    flat = np.sort(h.reshape(-1))[::-1]; k = max(1,int(0.10*flat.size))
    mass = float(flat[:k].sum()/(flat.sum()+1e-8))
    ent = float(-np.sum(h*np.log(h+1e-8))/h.size)
    return {"coverage@τ": cov, "mass@top10%": mass, "entropy": ent}

# ----------------------- Grad-eCLIP core -------------------------------
def _get_text_features_const(clip, text: str) -> torch.Tensor:
    with torch.no_grad():
        return F.normalize(clip.encode_text([text]), dim=-1)

def _hf_image_grad(clip, image: Image.Image, cfg: ScoreCAMConfig):
    """HF-путь: pixel_values с градиентом. Надёжно проверяем, что processor вызываемый."""
    device = clip.device
    proc = getattr(clip, "processor", None)
    if not callable(proc):
        raise RuntimeError("HF path requested but clip.processor is not callable")
    pv = proc(images=[image], return_tensors="pt")["pixel_values"].to(device)
    if cfg.safe_float32: pv = pv.to(torch.float32)
    pv.requires_grad_(True)
    if hasattr(clip.model, "get_image_features"):
        ifeat = clip.model.get_image_features(pixel_values=pv)
    else:
        # очень редкий случай: берём фичи через vision_model -> pool -> projection
        out = clip.model.vision_model(pixel_values=pv, output_hidden_states=False, return_dict=True)
        pooled = out.pooler_output if hasattr(out, "pooler_output") else out[1]
        ifeat = pooled @ clip.model.visual_projection.weight.T
    return pv, ifeat

def _openclip_image_grad(clip, image: Image.Image, cfg: ScoreCAMConfig):
    """open_clip / чекпоинт: нужен callable preprocess; иначе градиенты по пикселям невозможны."""
    device = clip.device if hasattr(clip, "device") else ("cuda" if torch.cuda.is_available() else "cpu")
    pre = getattr(clip, "preprocess", None)
    if not callable(pre):
        # попробуем внезапно processor, если он есть и вызываемый (бывает в обёртках)
        proc = getattr(clip, "processor", None)
        if callable(proc):
            pv = proc(images=[image], return_tensors="pt")["pixel_values"].to(device)
            if cfg.safe_float32: pv = pv.to(torch.float32)
            pv.requires_grad_(True)
            if hasattr(clip, "model") and hasattr(clip.model, "encode_image"):
                ifeat = clip.model.encode_image(pv)
            else:
                ifeat = clip.encode_image(pv)
            return pv, ifeat
        raise RuntimeError("No callable preprocess/processor on CLIP object")
    x = pre(image)                          # CHW float tensor
    if cfg.safe_float32: x = x.to(torch.float32)
    x = x.unsqueeze(0).to(device).requires_grad_(True)
    if hasattr(clip, "model") and hasattr(clip.model, "encode_image"):
        ifeat = clip.model.encode_image(x)
    else:
        ifeat = clip.encode_image(x)
    return x, ifeat

def _make_heat_from_grad(input_tensor: torch.Tensor, grad: torch.Tensor,
                         aggregate: str = "gradxinput", pool: str = "sum") -> np.ndarray:
    """
    input_tensor, grad: [1,3,H,W] (в пространстве препроцесса)
    Возвращает heat_small в [0..1] размера H×W.
    """
    if input_tensor.dim() != 4 or grad is None:
        return np.zeros((max(1,int(input_tensor.size(-2))), max(1,int(input_tensor.size(-1)))), dtype=np.float32)
    
    # Уменьшаем шум в градиентах (опционально)
    # Отсекаем слабые градиенты (нижние 10%)
    if grad.numel() > 0:
        grad_abs = grad.abs()
        grad_threshold = torch.quantile(grad_abs.flatten(), 0.1)
        grad = torch.where(grad_abs > grad_threshold, grad, torch.zeros_like(grad))
    
    if aggregate.lower() == "absgrad":
        sal = grad.abs()
    else:
        sal = (grad * input_tensor).abs()  # Grad×Input
        
    if pool.lower() == "sum":
        sal = sal.sum(dim=1, keepdim=False)
    else:
        sal = sal.mean(dim=1, keepdim=False)  # [1,H,W]
    
    heat = sal.squeeze(0).detach().cpu().numpy().astype(np.float32)
    
    # Дополнительное удаление слабых сигналов
    if heat.size > 0:
        # Сглаживание шума и усиление локальных максимумов
        from scipy import ndimage
        heat = ndimage.gaussian_filter(heat, sigma=0.5)
        
    return normalize01(heat)

def _postprocess_heat(h: np.ndarray, cfg: ScoreCAMConfig) -> np.ndarray:
    try:
        # Применяем перцентильное отсечение
        lo = np.percentile(h, float(cfg.p_low))
        hi = np.percentile(h, float(cfg.p_high))
        
        if hi > lo: 
            h = np.clip((h - lo) / (hi - lo + 1e-8), 0, 1)
            
            # Усиление фокуса: подавление слабых сигналов и усиление сильных
            if hasattr(cfg, 'focus_threshold') and hasattr(cfg, 'focus_boost'):
                threshold = float(cfg.focus_threshold)
                boost = float(cfg.focus_boost)
                
                # Подавление слабых сигналов
                h = np.where(h > threshold, h, h * 0.3)
                
                # Нелинейное усиление важных областей
                h = np.where(h > 0.7, np.minimum(h * boost, 1.0), h)
            
            # Степенное преобразование для контраста
            if cfg.gamma and cfg.gamma != 1.0:
                h = np.power(h, float(max(0.05, cfg.gamma)))
            
    except Exception as e:
        print(f"Ошибка в постобработке тепловой карты: {e}")
        
    # Применяем мягкое размытие, если задано
    if cfg.heat_smooth_sigma_px and cfg.heat_smooth_sigma_px > 0:
        hm = Image.fromarray((h*255).astype(np.uint8),'L').filter(
            ImageFilter.GaussianBlur(radius=float(cfg.heat_smooth_sigma_px)))
        h = np.asarray(hm).astype(np.float32)/255.0
    
    # Финальная нормализация
    return normalize01(h)

def compute_score_cam(clip, image: Image.Image, text: str, cfg: ScoreCAMConfig) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    """
    Совместимая с Score-CAM сигнатура: heat(H×W,0..1), base_cos, meta.
    Внутри — grad-eCLIP (|grad×input| или |grad|).
    """
    # 1) Текст и базовый cos без градиентов
    tfeat = _get_text_features_const(clip, text)
    with torch.no_grad():
        try:
            base_ifeat = clip.encode_image([image])
        except TypeError:
            # некоторые обёртки ждут батч тензоров — на всякий случай
            pre = getattr(clip, "preprocess", None)
            if callable(pre):
                base_ifeat = clip.encode_image(pre(image).unsqueeze(0).to(tfeat.device))
            else:
                base_ifeat = clip.encode_image([image])
        base_cos = float(cosine_sim(F.normalize(base_ifeat, dim=-1), tfeat)[0].item())

    # 2) Надёжно определяем путь к пикселям для градиента
    proc = getattr(clip, "processor", None)
    use_hf = callable(proc) and hasattr(clip, "model")
    try:
        if use_hf:
            x, ifeat = _hf_image_grad(clip, image, cfg); backend = "hf"
        else:
            x, ifeat = _openclip_image_grad(clip, image, cfg); backend = "open_clip"
    except Exception as e:
        # нет доступа к препроцессу → возвращаем «пустую» карту, но не падаем
        H0, W0 = image.height, image.width
        empty = np.zeros((H0, W0), dtype=np.float32)
        meta = {
            "rollout_backend": "grad-eclip", "grad_backend": "none",
            "error": f"{type(e).__name__}: {e}",
            "layers_total": 0, "layers_used": [0, 0], "head_fuse": "-",
            "add_residual": False, "row_normalize": False,
            "mask_shape": "grad-eclip", "baseline": "n/a",
            "p_low": float(cfg.p_low), "p_high": float(cfg.p_high),
            "gamma": float(cfg.gamma), "heat_smooth_sigma_px": float(cfg.heat_smooth_sigma_px),
            "overlay_alpha": float(cfg.overlay_alpha), "alpha_mode": str(cfg.alpha_mode),
        }
        return empty, base_cos, meta

    # 3) cos(img, text) с градиентом только по входу
    ifeat = F.normalize(ifeat, dim=-1)
    sim = cosine_sim(ifeat, tfeat)  # [1]
    # обнулим градиенты на модели, где это возможно (мягко)
    mdl = getattr(clip, "model", None)
    try:
        if mdl is not None: mdl.zero_grad(set_to_none=True)
    except TypeError:
        if mdl is not None: mdl.zero_grad()
    try:
        clip.zero_grad(set_to_none=True)  # если сам объект — nn.Module
    except Exception:
        pass

    sim.backward()  # retain_graph=False

    grad = x.grad  # [1,3,H,W]
    heat_small = _make_heat_from_grad(x, grad, cfg.grad_aggregate, cfg.grad_pool)

    # 4) Апсемпл к исходному и постобработка
    H0, W0 = image.height, image.width
    heat = Image.fromarray((heat_small * 255).astype(np.uint8), 'L').resize((W0, H0), Image.BICUBIC)
    heat = _postprocess_heat(np.asarray(heat).astype(np.float32) / 255.0, cfg)

    meta = {
        "rollout_backend": "grad-eclip",
        "layers_total": 0, "layers_used": [0, 0],
        "head_fuse": "-", "add_residual": False, "row_normalize": False,
        "grad_backend": backend, "grad_aggregate": cfg.grad_aggregate, "grad_pool": cfg.grad_pool,
        "input_tensor_shape": list(x.shape[-2:]),
        "mask_shape": "grad-eclip", "baseline": "n/a",
        "p_low": float(cfg.p_low), "p_high": float(cfg.p_high),
        "gamma": float(cfg.gamma), "heat_smooth_sigma_px": float(cfg.heat_smooth_sigma_px),
        "overlay_alpha": float(cfg.overlay_alpha), "alpha_mode": str(cfg.alpha_mode),
    }

    # очистка
    try:
        del x, ifeat, grad
        torch.cuda.empty_cache()
    except Exception:
        pass

    return heat, base_cos, meta

# --------------------------- Renderer (как было) -----------------------
def _draw_gradient_bar_reds(drw, x: int, y: int, w: int, h: int):
    for i in range(w):
        t = i / max(1, w - 1)
        r = int(255 * (0.1 + 0.9 * t))
        gb = int(255 * (1 - 0.85 * t))
        drw.line([(x + i, y), (x + i, y + h)], fill=(r, gb, gb))
    drw.rectangle([x, y, x + w, y + h], outline=(0, 0, 0), width=1)

def _wrap_text(drw, x: int, y: int, max_w: int, text: str, font, gap=6) -> int:
    if not text: return y
    words, line, yy = text.split(" "), "", y
    for w in words:
        t = w if not line else (line + " " + w)
        tw, th = _text_size(drw, t, font)
        if tw <= max_w:
            line = t
        else:
            drw.text((x, yy), line, font=font, fill=(0, 0, 0)); yy += th + gap; line = w
    if line:
        drw.text((x, yy), line, font=font, fill=(0, 0, 0)); yy += _text_size(drw, line, font)[1]
    return yy

def _measure_total_height(W: int, cfg: ScoreCAMConfig, gen_caption: str, ref_caption: Optional[str],
                          f_hdr, f_txt, f_sml) -> int:
    tmp = Image.new("RGB", (W, 200), (255, 255, 255))
    drw = ImageDraw.Draw(tmp)
    M, GAP = cfg.margins_px, cfg.block_gap_px
    col_w = W - 2 * M
    total = 0
    total += _text_size(drw, "Original image", f_hdr)[1] + 6
    total += cfg.image_block_height_px + GAP
    total += _text_size(drw, "Grad-eCLIP overlay", f_hdr)[1] + 6
    total += cfg.image_block_height_px + GAP
    total += _text_size(drw, "Generated caption (for CLIP score)", f_hdr)[1] + 6
    total += _wrap_text(drw, 0, 0, col_w, gen_caption or "(empty)", f_txt, cfg.line_pad_px) + GAP
    if cfg.show_reference and ref_caption:
        total += _text_size(drw, "Reference caption", f_hdr)[1] + 6
        total += _wrap_text(drw, 0, 0, col_w, ref_caption, f_txt, cfg.line_pad_px) + GAP
    total += _text_size(drw, "Heat (low → high)", f_sml)[1] + 4
    total += 14 + 26 + GAP
    total += _text_size(drw, "Metrics (explained)", f_hdr)[1] + 6
    for lbl in [
        "Косинус(изобр., полный текст)",
        f"Покрытие ≥ {cfg.tau:.2f}",
        "Масса топ-10% областей",
        "Энтропия карты",
        "Маски / Rollout",
        "Stretch P-low..P-high; γ",
    ]:
        total += _text_size(drw, f"{lbl}: 0.000", f_sml)[1] + 4
    total += _text_size(drw, "eCLIP", f_sml)[1] + 8
    return total

def _render_card_vertical(out_png: Path, image: Image.Image, heat01: np.ndarray,
                          gen_caption: str, ref_caption: Optional[str],
                          base_cos: float, meta: Dict[str, Any], cfg: ScoreCAMConfig):
    W = cfg.canvas_width_px
    M, GAP = cfg.margins_px, cfg.block_gap_px
    col_w = W - 2 * M

    f_hdr = _find_font(cfg.font_path_regular, cfg.fs_header)
    f_txt = _find_font(cfg.font_path_regular, cfg.fs_text)
    f_sml = _find_font(cfg.font_path_regular, cfg.fs_small)

    fs_caption = cfg.fs_text
    total = _measure_total_height(W, cfg, gen_caption, ref_caption if cfg.show_reference else None,
                                  f_hdr, f_txt, f_sml)
    if cfg.auto_shrink_text:
        while total + 2 * M > cfg.canvas_height_max_px and fs_caption > cfg.fs_text_min:
            fs_caption -= 1
            f_txt = _find_font(cfg.font_path_regular, fs_caption)
            total = _measure_total_height(W, cfg, gen_caption, ref_caption if cfg.show_reference else None,
                                          f_hdr, f_txt, f_sml)

    H = max(cfg.canvas_height_px, total + 2 * M)
    if cfg.auto_expand_canvas:
        H = min(max(H, cfg.canvas_height_px), cfg.canvas_height_max_px)

    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    drw = ImageDraw.Draw(canvas)

    x = M; y = M

    # 1) Original
    drw.text((x, y), "Original image", font=f_hdr, fill=(0, 0, 0))
    y += _text_size(drw, "Original image", f_hdr)[1] + 6
    img_fit = _fit_cover(image, col_w, cfg.image_block_height_px)
    canvas.paste(img_fit, (x, y))
    y += cfg.image_block_height_px + GAP

    # 2) Overlay (heat cover-кроп под блок)
    drw.text((x, y), "Grad-eCLIP overlay", font=f_hdr, fill=(0, 0, 0))
    y += _text_size(drw, "Grad-eCLIP overlay", f_hdr)[1] + 6
    base_block = _fit_cover(image, col_w, cfg.image_block_height_px)
    heat_block = _fit_cover_gray(heat01, col_w, cfg.image_block_height_px)
    overlay = _overlay_with_heat(base_block, heat_block, float(cfg.overlay_alpha), cfg.alpha_mode)
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
    kv(f"Покрытие ≥ {cfg.tau:.2f}", f"{stats['coverage@τ']:.3f}")
    kv("Масса топ-10% областей", f"{stats['mass@top10%']:.3f}")
    kv("Энтропия карты", f"{stats['entropy']:.3f}")

    # Маски / Rollout — переиспользуем тот же блок (backend отдадим как 'grad-eclip')
    if meta.get("mask_shape","") == "rollout" or "rollout_backend" in meta:
        s,e = meta.get("layers_used",[0,0])
        L = int(meta.get("layers_total",0))
        backend = meta.get("rollout_backend","grad-eclip")
        fuse = meta.get("head_fuse","-")
        rn = "yes" if meta.get("row_normalize", False) else "no"
        rs = "yes" if meta.get("add_residual", False) else "no"
        extra = f", grad={meta.get('grad_backend','?')}, agg={meta.get('grad_aggregate','gradxinput')}/{meta.get('grad_pool','mean')}"
        kv("Маски / Rollout", f"{backend}; layers {int(s)}..{int(e)} / {L}; fuse={fuse}; residual={rs}; row-norm={rn}{extra}")
    else:
        kv("Маски, baseline", f"{meta.get('mask_shape','n/a')}; {meta.get('baseline','n/a')}")

    kv("Stretch P-low..P-high; γ", f"P{float(meta.get('p_low',0.0)):.1f}..P{float(meta.get('p_high',0.0)):.1f}; {float(meta.get('gamma',1.0)):.2f}")

    footer = "Grad ECLIP"
    fw, fh = _text_size(drw, footer, f_sml)
    drw.text((W - fw - M, H - fh - 6), footer, font=f_sml, fill=(90, 90, 90))

    canvas.save(str(out_png), format="PNG")

# --------------------------- Public API --------------------------------
def run(clip, captioner, *, image_path: str, out_dir: Path,
        ref_caption: Optional[str] = None, config: Optional[ScoreCAMConfig] = None) -> Dict[str, Any]:
    cfg = _coerce_cfg(config)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    image = load_image(image_path, fallback_hw=getattr(clip, "image_size", 224))
    try:
        gen_caption = captioner.generate([image], prompt=getattr(captioner, "prompt", None) or "Describe the image.")[0]
    except Exception:
        gen_caption = ""

    heat, base_cos, meta = compute_score_cam(clip, image, gen_caption, cfg)

    stem = Path(image_path).stem
    out_png = out_dir / f"{stem}.png"
    _render_card_vertical(out_png, image, heat, gen_caption,
                          (ref_caption if cfg.show_reference else None),
                          base_cos, meta, cfg)

    stats = heatmap_stats(heat, tau=float(cfg.tau))
    data = {
        "image_path": image_path, "method": "grad_eclip",
        "config": asdict(cfg), "caption_generated": gen_caption,
        "caption_reference": (ref_caption if cfg.show_reference else None),
        "clip_cosine": base_cos, "heatmap_stats": stats, "meta": meta,
        "outputs": {"png": str(out_png)}
    }
    with open(out_dir / f"{stem}.json","w",encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return data
