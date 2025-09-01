# methods/score_cam.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass, asdict, fields
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Union
import json, numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import torch, torch.nn.functional as F
import math

# ----------------------------- Config ---------------------------------
@dataclass
class ScoreCAMConfig:
    # --- верстка как было ---
    canvas_width_px: int = 720
    canvas_height_px: int = 1400
    margins_px: int = 16
    block_gap_px: int = 10
    image_block_height_px: int = 360
    font_path_regular: str = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    fs_header: int = 22; fs_text: int = 20; fs_small: int = 16; fs_text_min: int = 14
    line_pad_px: int = 6

    # --- поля ниже оставляем для совместимости JSON/рендерера ---
    # (в rollout они не используются, но renderer ожидает эти ключи)
    grid_n: int = 9
    work_resize_px: int = 0
    baseline: str = "n/a"
    blur_sigma_px: float = 0.0
    batch_vision: int = 0

    mask_shape: str = "rollout"   # НЕ 'gauss' → renderer покажет описание rollout
    stride_frac: float = 0.0
    sigma_frac: float = 0.0
    edge_soften_px: int = 0

    # Нормировки/порог
    score_norm: str = "softmax"
    score_centering: str = "base"
    tau: float = 0.60

    # Постобработка теплокарты
    p_low: float = 80.0; p_high: float = 99.5
    gamma: float = 0.9
    heat_smooth_sigma_px: float = 0.0

    # Overlay
    overlay_alpha: float = 0.45
    alpha_mode: str = "scaled"

    # Рендер
    auto_shrink_text: bool = True
    auto_expand_canvas: bool = True
    canvas_height_max_px: int = 4000
    show_reference: bool = True

    # --- УЛУЧШЕННЫЕ ПАРАМЕТРЫ ROLLOUT ---
    head_fuse: str = "max"           # "max" вместо "mean" для более четких результатов
    add_residual: bool = True        # I + A
    row_normalize: bool = True
    start_layer: int = 0
    end_layer: int = -1              # -1 → до конца
    proxy_temp: float = 0.25         # снижена температура для более четких фокусов
    rollout_blur_ksize: int = 3      # добавлено небольшое сглаживание для подавления шума
    rollout_blur_sigma: float = 0.8
    
    # Улучшенные параметры постобработки
    p_low: float = 85.0              # увеличено для отсечения большего шума
    p_high: float = 99.8             # увеличено для сохранения ярких участков
    gamma: float = 1.2               # увеличено для более контрастной визуализации
    overlay_alpha: float = 0.55      # увеличена прозрачность для лучшей видимости

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
    img2 = img.resize((nw,nh), Image.LANCZOS)
    ox, oy = (nw-W)//2, (nh-H)//2
    return img2.crop((ox,oy,ox+W,oy+H))

def _fit_cover_gray(gray01: np.ndarray, W: int, H: int) -> np.ndarray:
    h,w = gray01.shape; s = max(W/w, H/h)
    nw, nh = max(1,int(round(w*s))), max(1,int(round(h*s)))
    pil = Image.fromarray((normalize01(gray01)*255).astype(np.uint8),'L').resize((nw,nh), Image.BICUBIC)
    ox, oy = (nw-W)//2, (nh-H)//2
    return np.asarray(pil.crop((ox,oy,ox+W,oy+H))).astype(np.float32)/255.0

def _apply_colormap_reds(g: np.ndarray) -> np.ndarray:
    g = np.clip(g,0,1).astype(np.float32); r = (0.9*g+0.1)*255.0; gb = (1.0-0.85*g)*255.0
    return np.clip(np.stack([r,gb,gb],-1),0,255).astype(np.uint8)

def _overlay_with_heat(base: Image.Image, heat01: np.ndarray, alpha: float, alpha_mode: str="scaled") -> Image.Image:
    W,H = base.size; heat = normalize01(heat01)
    
    # Улучшенная интерполяция для карты внимания
    if heat.shape != (H,W):
        pil_heat = Image.fromarray((heat*255).astype(np.uint8),'L')
        pil_heat = pil_heat.resize((W,H), Image.BICUBIC)
        heat = np.array(pil_heat).astype(np.float32)/255.0
    
    # Более насыщенная карта цветов
    hm_img = Image.fromarray(_apply_colormap_reds(heat))
    
    amode = str(alpha_mode).lower()
    if amode == "uniform":
        a_img = Image.new("L",(W,H),int(np.clip(alpha,0,1)*255))
    else:
        # Нелинейная альфа для лучшего контраста
        a = np.power(np.clip(heat, 0.0, 1.0), 0.8) * float(np.clip(alpha, 0.0, 1.0))
        a_img = Image.fromarray((a*255).astype(np.uint8), "L")
    
    # Применение маски с улучшенной прозрачностью
    out = base.copy(); out.paste(hm_img,(0,0),a_img); 
    return out

def heatmap_stats(hm: np.ndarray, tau: float=0.6) -> Dict[str,float]:
    h = normalize01(hm); cov = float((h>=tau).mean())
    flat = np.sort(h.reshape(-1))[::-1]; k = max(1,int(0.10*flat.size))
    mass = float(flat[:k].sum()/(flat.sum()+1e-8))
    ent = float(-np.sum(h*np.log(h+1e-8))/h.size)
    return {"coverage@τ": cov, "mass@top10%": mass, "entropy": ent}

# ----------------------- Attention Rollout core -----------------------
@torch.inference_mode()
def _rollout_hf(clip, image: Image.Image, cfg: ScoreCAMConfig) -> Tuple[np.ndarray, Dict[str,Any]]:
    proc = getattr(clip, "processor", None)
    if not callable(proc): raise RuntimeError("HF path requires clip.processor")
    pv = proc(images=[image], return_tensors="pt")["pixel_values"].to(clip.device)
    try:
        out = clip.model.vision_model(pixel_values=pv, output_attentions=True, return_dict=True)
        atts = out.attentions
    except TypeError:
        out = clip.model.vision_model(pixel_values=pv, output_attentions=True)
        atts = out[-1] if isinstance(out,(list,tuple)) else getattr(out, "attentions", None)
    if not isinstance(atts,(list,tuple)) or len(atts)==0:
        raise RuntimeError("No attentions from HF vision_model")

    L = len(atts); s = max(0, int(cfg.start_layer))
    e = L if cfg.end_layer in (-1,None) else min(L, int(cfg.end_layer))
    use = atts[s:e] if s<e else atts

    joint = None
    for A in use:
        if A.dim()==3: A = A.unsqueeze(0)            # [1,H,T,T]
        A = A.max(dim=1).values if cfg.head_fuse.lower()=="max" else A.mean(dim=1)  # [B,T,T]
        if cfg.add_residual:
            I = torch.eye(A.size(-1), device=A.device, dtype=A.dtype).unsqueeze(0)
            A = A + I
        if cfg.row_normalize:
            A = A / A.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        joint = A if joint is None else joint @ A

    rel = joint[:,0,1:]                                  # CLS→patch
    T_p = rel.size(-1); g = int(math.sqrt(T_p))
    if g*g != T_p: g = int(math.floor(math.sqrt(T_p))); rel = rel[:, :g*g]
    rel = rel.view(1,1,g,g)
    if int(cfg.rollout_blur_ksize)>1:
        rel = _gaussian_blur_compat(rel.to(torch.float32), k=int(cfg.rollout_blur_ksize), sigma=float(cfg.rollout_blur_sigma))
    H,W = image.height, image.width
    heat = F.interpolate(rel, size=(H,W), mode="bilinear", align_corners=False).squeeze().cpu().numpy()

    try:
        lo = np.percentile(heat, float(cfg.p_low)); hi = np.percentile(heat, float(cfg.p_high))
        if hi>lo: heat = np.clip((heat-lo)/(hi-lo+1e-8), 0, 1)
    except Exception: pass
    if cfg.gamma and cfg.gamma!=1.0:
        heat = normalize01(np.power(heat, float(max(0.05, cfg.gamma))))

    meta = {
        "rollout_backend": "hf_attn",
        "layers_total": L,
        "layers_used": [int(s), int(e if cfg.end_layer not in (-1,None) else L)],
        "head_fuse": cfg.head_fuse,
        "add_residual": bool(cfg.add_residual),
        "row_normalize": bool(cfg.row_normalize),
        "p_low": float(cfg.p_low), "p_high": float(cfg.p_high), "gamma": float(cfg.gamma),
        "blur_ksize": int(cfg.rollout_blur_ksize), "blur_sigma": float(cfg.rollout_blur_sigma),
    }
    return heat, meta

@torch.inference_mode()
def _rollout_proxy(clip, image: Image.Image, cfg: ScoreCAMConfig) -> Tuple[np.ndarray, Dict[str,Any]]:
    # Очистка памяти перед работой с hooks
    torch.cuda.empty_cache()
    
    visual = getattr(getattr(clip,"model",clip), "visual", getattr(clip,"model",clip))
    feats: List[torch.Tensor] = []
    hooks = []

    def _keep(o: torch.Tensor):
        if isinstance(o, torch.Tensor) and o.dim()==3: feats.append(o.detach())

    # LN1 / norm1 если есть
    for name, module in visual.named_modules():
        if "blocks" in name or "resblocks" in name or "encoder.layers" in name:
            if hasattr(module, "ln_1"): hooks.append(module.ln_1.register_forward_hook(lambda m,i,o: _keep(o)))
            elif hasattr(module, "norm1"): hooks.append(module.norm1.register_forward_hook(lambda m,i,o: _keep(o)))
    # если ничего не поймали — на сами блоки
    if not hooks:
        for name, module in visual.named_modules():
            if "blocks" in name or "resblocks" in name or "encoder.layers" in name:
                hooks.append(module.register_forward_hook(lambda m,i,o: _keep(o)))

    try:
        clip.encode_image([image])  # вызовет forward блоков
    finally:
        # Улучшенная очистка hooks
        for h in hooks:
            try: 
                h.remove()
            except: 
                pass
        hooks = []
        torch.cuda.empty_cache()

    if not feats:
        raise RuntimeError("Proxy rollout: no token features captured (ln_1/block outputs).")

    # одинаковая длина токенов
    hist: Dict[int,int] = {}
    for f in feats: hist[f.shape[1]] = hist.get(f.shape[1],0)+1
    T_common = max(hist, key=hist.get)
    layers = [f for f in feats if f.shape[1]==T_common]

    L = len(layers); s = max(0, int(cfg.start_layer))
    e = L if cfg.end_layer in (-1,None) else min(L, int(cfg.end_layer))
    layers = layers[s:e] if s<e else layers

    # Оптимизация обработки слоев
    joint = None
    Tsoft = float(max(1e-3, cfg.proxy_temp))
    for i, X in enumerate(layers):
        # Преобразование в float32 для улучшения численной стабильности
        Xn = F.normalize(X.to(torch.float32), dim=-1)                  # [B,T,C]
        
        # Ускоренное вычисление матрицы внимания с использованием bmm
        X_t = Xn.transpose(1, 2)  # [B,C,T]
        A = torch.bmm(Xn, X_t) / Tsoft  # [B,T,T]
        
        A = F.softmax(A, dim=-1)
        if cfg.add_residual:
            I = torch.eye(A.size(-1), device=A.device, dtype=A.dtype).unsqueeze(0)
            A = A + I
        if cfg.row_normalize:
            A = A / A.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        
        # Очистка промежуточных тензоров для экономии памяти
        del Xn, X_t
        
        joint = A if joint is None else joint @ A
        
        # Периодическая очистка для длинных последовательностей слоев
        if i % 4 == 3:
            torch.cuda.empty_cache()

    rel = joint[:,0,1:]
    T_p = rel.size(-1); g = int(math.sqrt(T_p))
    if g*g != T_p: g = int(math.floor(math.sqrt(T_p))); rel = rel[:, :g*g]
    rel = rel.view(1,1,g,g)

    if int(cfg.rollout_blur_ksize)>1:
        rel = _gaussian_blur_compat(rel, k=int(cfg.rollout_blur_ksize), sigma=float(cfg.rollout_blur_sigma))

    H,W = image.height, image.width
    heat = F.interpolate(rel, size=(H,W), mode="bilinear", align_corners=False).squeeze().cpu().numpy()

    try:
        lo = np.percentile(heat, float(cfg.p_low)); hi = np.percentile(heat, float(cfg.p_high))
        if hi>lo: heat = np.clip((heat-lo)/(hi-lo+1e-8), 0, 1)
    except Exception: pass
    if cfg.gamma and cfg.gamma!=1.0:
        heat = normalize01(np.power(heat, float(max(0.05, cfg.gamma))))

    meta = {
        "rollout_backend": "proxy",
        "layers_total": L,
        "layers_used": [int(s), int(e if cfg.end_layer not in (-1,None) else L)],
        "head_fuse": "proxy/mean",
        "add_residual": bool(cfg.add_residual),
        "row_normalize": bool(cfg.row_normalize),
        "proxy_temp": float(cfg.proxy_temp),
        "p_low": float(cfg.p_low), "p_high": float(cfg.p_high), "gamma": float(cfg.gamma),
        "blur_ksize": int(cfg.rollout_blur_ksize), "blur_sigma": float(cfg.rollout_blur_sigma),
    }
    return heat, meta

def _gaussian_blur_compat(x: torch.Tensor, k: int = 5, sigma: float = 1.0) -> torch.Tensor:
    if k <= 1: return x
    r = (k - 1)//2
    coords = torch.arange(-r, r+1, device=x.device, dtype=x.dtype)
    g = torch.exp(-(coords**2)/(2*sigma*sigma)); g = g/(g.sum()+1e-8)
    ker2 = (g[:,None] @ g[None,:]).unsqueeze(0).unsqueeze(0)
    xpad = F.pad(x, (r,r,r,r), mode='reflect')
    return F.conv2d(xpad, ker2)

# --------------------- Rollout wrapper (Score-CAM API) -----------------
@torch.inference_mode()
def compute_score_cam(clip, image: Image.Image, text: str, cfg: ScoreCAMConfig) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    """
    Сигнатура и выходы как у Score-CAM:
      heat(H×W,0..1), base_cos (cos(img,text)), meta (включая rollout_*).
    """
    # CLIP score (для метрик/подписи)
    tfeat = clip.encode_text([text])
    ifeat = clip.encode_image([image])
    base_cos = float(cosine_sim(ifeat, tfeat)[0].item())

    # Heatmap по Attention Rollout (HF → иначе proxy)
    try:
        heat, meta_roll = _rollout_hf(clip, image, cfg)
    except Exception:
        heat, meta_roll = _rollout_proxy(clip, image, cfg)

    # Соберём meta, чтобы renderer остался неизменным
    meta: Dict[str, Any] = {
        # совместимость (renderer читает эти поля)
        "mask_shape": "rollout",
        "grid_n": int(cfg.grid_n),
        "stride_frac": float(cfg.stride_frac),
        "sigma_frac": float(cfg.sigma_frac),
        "edge_soften_px": int(cfg.edge_soften_px),
        "baseline": "n/a",
        "blur_sigma_px": float(cfg.blur_sigma_px),
        "score_norm": str(cfg.score_norm),
        "score_centering": str(cfg.score_centering),
        "p_low": float(cfg.p_low), "p_high": float(cfg.p_high),
        "gamma": float(cfg.gamma),
        "work_resize_px": int(cfg.work_resize_px or 0),
        "heat_smooth_sigma_px": float(cfg.heat_smooth_sigma_px),
        "overlay_alpha": float(cfg.overlay_alpha),
        "alpha_mode": str(cfg.alpha_mode),
        # rollout детали
        **meta_roll,
    }
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
    total += _text_size(drw, "Attention Rollout overlay", f_hdr)[1] + 6
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
    total += _text_size(drw, "Score-CAM • CLIP vision attribution", f_sml)[1] + 8
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

    # 2) Overlay (heat cover-кроп под блок, как и прежде)
    drw.text((x, y), "Attention Rollout overlay", font=f_hdr, fill=(0, 0, 0))
    y += _text_size(drw, "Attention Rollout overlay", f_hdr)[1] + 6
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

    # 5) Metrics (RU)
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

    # Маски / Rollout (адаптивно)
    if meta.get("mask_shape","") == "rollout" or "rollout_backend" in meta:
        s,e = meta.get("layers_used",[0,0])
        L = int(meta.get("layers_total",0))
        backend = meta.get("rollout_backend","proxy")
        fuse = meta.get("head_fuse","mean")
        rn = "yes" if meta.get("row_normalize", True) else "no"
        rs = "yes" if meta.get("add_residual", True) else "no"
        extra = (f", T={meta.get('proxy_temp',0.0):.2f}" if backend=="proxy" else "")
        kv("Маски / Rollout", f"{backend}; layers {int(s)}..{int(e)} / {L}; fuse={fuse}; residual={rs}; row-norm={rn}{extra}")
    else:
        if meta.get('mask_shape','rect') == 'gauss':
            mask_desc = f"{int(meta['grid_n'])}×{int(meta['grid_n'])} Gaussian; stride={float(meta.get('stride_frac', 0.0)):.2f}, σ={float(meta.get('sigma_frac', 0.0)):.2f}"
        else:
            mask_desc = f"{int(meta['grid_n'])}×{int(meta['grid_n'])} rect / soften={int(meta.get('edge_soften_px', 0))} px"
        kv("Маски, baseline", f"{mask_desc}; {meta.get('baseline','n/a')}")

    kv("Stretch P-low..P-high; γ", f"P{float(meta['p_low']):.1f}..P{float(meta['p_high']):.1f}; {float(meta['gamma']):.2f}")

    # Footer (оставим как было)
    footer = "Attention Rollout"
    fw, fh = _text_size(drw, footer, f_sml)
    drw.text((W - fw - M, H - fh - 6), footer, font=f_sml, fill=(90, 90, 90))

    canvas.save(str(out_png), format="PNG")

# --------------------------- Public API --------------------------------
def run(clip, captioner, *, image_path: str, out_dir: Path,
        ref_caption: Optional[str] = None, config: Optional[ScoreCAMConfig] = None) -> Dict[str, Any]:
    cfg = _coerce_cfg(config)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    image = load_image(image_path, fallback_hw=getattr(clip, "image_size", 224))

    # подпись для CLIP score (CPU-safe)
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
        "image_path": image_path, "method": "attention_rollout",  # честно укажем метод
        "config": asdict(cfg), "caption_generated": gen_caption,
        "caption_reference": (ref_caption if cfg.show_reference else None),
        "clip_cosine": base_cos, "heatmap_stats": stats, "meta": meta,
        "outputs": {"png": str(out_png)}
    }
    with open(out_dir / f"{stem}.json","w",encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return data
