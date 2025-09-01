# methods/score_cam.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass, asdict, fields
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import json, numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import torch, torch.nn.functional as F

# ----------------------------- Config ---------------------------------
@dataclass
class ScoreCAMConfig:
    canvas_width_px: int = 720
    canvas_height_px: int = 1400
    margins_px: int = 16
    block_gap_px: int = 10
    image_block_height_px: int = 360
    font_path_regular: str = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    fs_header: int = 22; fs_text: int = 20; fs_small: int = 16; fs_text_min: int = 14
    line_pad_px: int = 6

    # Маскирование
    grid_n: int = 9               # базовая плотность по стороне
    work_resize_px: int = 448     # ускорение; 0 = исходный размер
    baseline: str = "blur"        # "blur"|"mean"|"zero"
    blur_sigma_px: float = 6.0
    batch_vision: int = 32

    # НОВОЕ: перекрывающиеся Гауссовы маски
    mask_shape: str = "gauss"     # "gauss" | "rect"
    stride_frac: float = 0.5      # шаг центров в долях размера «ячейки» (0.5 = 50% перекрытия)
    sigma_frac: float = 0.6       # σ в долях размера «ячейки» для гаусса
    edge_soften_px: int = 1       # только для rect

    # Веса
    score_norm: str = "softmax"   # "minmax"|"softmax"
    score_centering: str = "base" # "min"|"median"|"base"
    tau: float = 0.60

    # Постобработка теплокарты
    p_low: float = 80.0; p_high: float = 99.5
    gamma: float = 0.9
    heat_smooth_sigma_px: float = 2.0

    # Overlay
    overlay_alpha: float = 0.45
    alpha_mode: str = "scaled"

    # Рендер
    auto_shrink_text: bool = True
    auto_expand_canvas: bool = True
    canvas_height_max_px: int = 4000
    show_reference: bool = True

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
    if heat.shape != (H,W):
        heat = np.array(Image.fromarray((heat*255).astype(np.uint8),'L').resize((W,H), Image.BICUBIC)).astype(np.float32)/255.0
    hm_img = Image.fromarray(_apply_colormap_reds(heat))
    amode = str(alpha_mode).lower()
    if amode == "uniform":
        a_img = Image.new("L",(W,H),int(np.clip(alpha,0,1)*255))
    else:
        # Масштабируем альфу значениями теплокарты без постоянного сдвига
        a = np.clip(heat, 0.0, 1.0) * float(np.clip(alpha, 0.0, 1.0))
        a_img = Image.fromarray((a*255).astype(np.uint8), "L")
    out = base.copy(); out.paste(hm_img,(0,0),a_img); return out

def heatmap_stats(hm: np.ndarray, tau: float=0.6) -> Dict[str,float]:
    h = normalize01(hm); cov = float((h>=tau).mean())
    flat = np.sort(h.reshape(-1))[::-1]; k = max(1,int(0.10*flat.size))
    mass = float(flat[:k].sum()/(flat.sum()+1e-8))
    ent = float(-np.sum(h*np.log(h+1e-8))/h.size)
    return {"coverage@τ": cov, "mass@top10%": mass, "entropy": ent}

# -------------------- Masks (overlapping Gaussian) --------------------
def _build_gauss_masks(H:int, W:int, grid_n:int, stride_frac:float, sigma_frac:float):
    """
    Возвращает:
      - masks_u8: список uint8 масок (H,W) в [0..255]
      - cover_map: float32 карта суммарного покрытия Σ M_i (H,W) в [0..N]
    Центры равномерные, шаг = stride_frac * размер «ячейки». σ = sigma_frac * размер «ячейки».
    """
    grid_n = max(2, int(grid_n))
    cell_w = W / grid_n; cell_h = H / grid_n
    step_x = max(1.0, cell_w * max(0.1, float(stride_frac)))
    step_y = max(1.0, cell_h * max(0.1, float(stride_frac)))
    xs = np.arange(cell_w/2, W, step_x, dtype=np.float32)
    ys = np.arange(cell_h/2, H, step_y, dtype=np.float32)
    sigma_x = max(1.0, cell_w * float(sigma_frac))
    sigma_y = max(1.0, cell_h * float(sigma_frac))

    X = np.arange(W, dtype=np.float32)[None, :]
    Y = np.arange(H, dtype=np.float32)[:, None]

    masks_u8: List[np.ndarray] = []
    cover = np.zeros((H, W), dtype=np.float32)
    inv2sx2 = 1.0 / (2.0 * sigma_x * sigma_x)
    inv2sy2 = 1.0 / (2.0 * sigma_y * sigma_y)

    for cy in ys:
        dy2 = (Y - cy) ** 2
        gy = np.exp(-dy2 * inv2sy2)  # H,1
        for cx in xs:
            gx = np.exp(- (X - cx) ** 2 * inv2sx2)  # 1,W
            m = (gy * gx).astype(np.float32)  # H,W
            # мягкий обрез «далёкого хвоста»
            m[m < 1e-3] = 0.0
            cover += m
            masks_u8.append((np.clip(m,0,1) * 255.0).astype(np.uint8))
    return masks_u8, cover

def _build_rect_masks_fast(H:int, W:int, grid_n:int, soften_px:int) -> List[np.ndarray]:
    xs = np.linspace(0, W, grid_n + 1).astype(int)
    ys = np.linspace(0, H, grid_n + 1).astype(int)
    masks: List[np.ndarray] = []
    blur = max(0, int(soften_px))
    for gy in range(grid_n):
        for gx in range(grid_n):
            x0,x1 = xs[gx], xs[gx+1]; y0,y1 = ys[gy], ys[gy+1]
            m = Image.new("L",(W,H),0); d = ImageDraw.Draw(m)
            d.rectangle([x0,y0,max(x0+1,x1-1),max(y0+1,y1-1)], fill=255)
            if blur>0: m = m.filter(ImageFilter.GaussianBlur(radius=float(blur)))
            masks.append(np.asarray(m).astype(np.float32)/255.0)
    return masks

# --------------------------- Score-CAM core ----------------------------
@torch.inference_mode()
def compute_score_cam(clip, image: Image.Image, text: str, cfg: ScoreCAMConfig) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    # Фичи baseline
    tfeat = clip.encode_text([text])
    ifeat = clip.encode_image([image])
    base_cos = float(cosine_sim(ifeat, tfeat)[0].item())

    W0, H0 = image.size
    if cfg.work_resize_px and int(cfg.work_resize_px) > 0:
        s = float(cfg.work_resize_px) / float(max(W0, H0))
        Ww, Hw = max(32, int(round(W0*s))), max(32, int(round(H0*s)))
    else:
        Ww, Hw = W0, H0

    # Baseline
    if cfg.baseline == "blur":
        base_img_big = image.filter(ImageFilter.GaussianBlur(radius=float(cfg.blur_sigma_px)))
    elif cfg.baseline == "mean":
        arr = np.asarray(image).astype(np.float32)
        mean_color = tuple(int(c) for c in arr.reshape(-1,3).mean(axis=0))
        base_img_big = Image.new("RGB",(W0,H0), mean_color)
    else:
        base_img_big = Image.new("RGB",(W0,H0),(0,0,0))

    img_small  = image.resize((Ww,Hw), Image.BILINEAR)
    base_small = base_img_big.resize((Ww,Hw), Image.BILINEAR)
    orig_np = np.asarray(img_small).astype(np.float32)/255.0
    base_np = np.asarray(base_small).astype(np.float32)/255.0

    # Маски
    if str(cfg.mask_shape).lower() == "gauss":
        masks_u8, cover = _build_gauss_masks(Hw, Ww, int(cfg.grid_n), float(cfg.stride_frac), float(cfg.sigma_frac))
        masks = [m.astype(np.float32)/255.0 for m in masks_u8]
        cover = np.maximum(cover.astype(np.float32), 1e-6)
    else:
        masks = _build_rect_masks_fast(Hw, Ww, int(cfg.grid_n), int(cfg.edge_soften_px))
        cover = np.maximum(
            np.sum(np.stack(masks, axis=0).astype(np.float32), axis=0),
            1e-6
        )

    # Батч-инференс
    sims_list: List[torch.Tensor] = []; batch_imgs: List[Image.Image] = []
    def flush():
        nonlocal batch_imgs, sims_list
        if not batch_imgs: return
        feats = clip.encode_image(batch_imgs)
        sims = cosine_sim(feats, tfeat.expand_as(feats)).detach().cpu()
        sims_list.append(sims); batch_imgs = []

    for m in masks:
        m3 = np.repeat(m[...,None], 3, axis=-1)
        arr = (orig_np*m3 + base_np*(1.0 - m3)).clip(0,1)
        batch_imgs.append(Image.fromarray((arr*255.0).astype(np.uint8)))
        if len(batch_imgs) >= int(cfg.batch_vision): flush()
    flush()

    sims_all = torch.cat(sims_list, dim=0) if sims_list else torch.zeros(len(masks))

    # Центровка → ReLU
    mode = cfg.score_centering.lower()
    if mode == "median":
        sims_all = sims_all - sims_all.median()
    elif mode == "base":
        sims_all = sims_all - torch.tensor(base_cos, dtype=sims_all.dtype)
    else:
        sims_all = sims_all - sims_all.min()
    sims_all = torch.relu(sims_all)

    # Нормировка весов
    if cfg.score_norm.lower() == "softmax":
        sims_all = torch.softmax(sims_all, dim=0)
    else:
        sims_all = sims_all / (sims_all.max() + 1e-8)

    # Анти-«сеточные круги»: нормализация по покрытию
    heat_small = np.zeros((Hw, Ww), dtype=np.float32)
    for w, m in zip(sims_all.cpu().numpy().tolist(), masks):
        heat_small += m.astype(np.float32) * float(w)
    heat_small = heat_small / cover
    # Доп. защита от NaN/Inf
    heat_small = np.nan_to_num(heat_small, nan=0.0, posinf=0.0, neginf=0.0)
    heat_small = normalize01(heat_small)

    # Stretch + gamma
    try:
        lo = np.percentile(heat_small, float(cfg.p_low))
        hi = np.percentile(heat_small, float(cfg.p_high))
        if hi > lo:
            heat_small = np.clip((heat_small - lo) / (hi - lo + 1e-8), 0, 1)
    except Exception:
        pass
    if cfg.gamma and cfg.gamma != 1.0:
        heat_small = normalize01(np.power(heat_small, float(max(0.05, cfg.gamma))))

    # Финальное сглаживание
    if cfg.heat_smooth_sigma_px and cfg.heat_smooth_sigma_px > 0:
        hm = Image.fromarray((heat_small*255).astype(np.uint8),'L').filter(
            ImageFilter.GaussianBlur(radius=float(cfg.heat_smooth_sigma_px)))
        heat_small = np.asarray(hm).astype(np.float32)/255.0

    # Апсемпл к исходному размеру
    heat = Image.fromarray((heat_small*255).astype(np.uint8),'L').resize((W0,H0), Image.BICUBIC)
    heat = normalize01(np.asarray(heat).astype(np.float32)/255.0)

    meta = {
        "grid_n": int(cfg.grid_n),
        "mask_shape": str(cfg.mask_shape),
        "stride_frac": float(cfg.stride_frac),
        "sigma_frac": float(cfg.sigma_frac),
        "edge_soften_px": int(cfg.edge_soften_px),
        "baseline": str(cfg.baseline),
        "blur_sigma_px": float(cfg.blur_sigma_px),
        "score_norm": str(cfg.score_norm),
        "score_centering": str(cfg.score_centering),
        "p_low": float(cfg.p_low), "p_high": float(cfg.p_high),
        "gamma": float(cfg.gamma),
        "work_resize_px": int(cfg.work_resize_px or 0),
        "heat_smooth_sigma_px": float(cfg.heat_smooth_sigma_px),
        "overlay_alpha": float(cfg.overlay_alpha),
        "alpha_mode": str(cfg.alpha_mode)
    }
    return heat, base_cos, meta

# --------------------------- Renderer (как было) -----------------------
# --------------------------- Renderer (vertical) -----------------------

def _draw_gradient_bar_reds(drw, x: int, y: int, w: int, h: int):
    for i in range(w):
        t = i / max(1, w - 1)
        r = int(255 * (0.1 + 0.9 * t))
        gb = int(255 * (1 - 0.85 * t))
        drw.line([(x + i, y), (x + i, y + h)], fill=(r, gb, gb))
    drw.rectangle([x, y, x + w, y + h], outline=(0, 0, 0), width=1)

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
    total += _text_size(drw, "Score-CAM overlay", f_hdr)[1] + 6
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
        "Маски, baseline",
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

    # 2) Overlay
    drw.text((x, y), "Score-CAM overlay", font=f_hdr, fill=(0, 0, 0))
    y += _text_size(drw, "Score-CAM overlay", f_hdr)[1] + 6
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

    # Описание масок и baseline
    if meta.get('mask_shape', 'rect') == 'gauss':
        mask_desc = f"{int(meta['grid_n'])}×{int(meta['grid_n'])} Gaussian; stride={float(meta.get('stride_frac', 0.0)):.2f}, σ={float(meta.get('sigma_frac', 0.0)):.2f}"
    else:
        mask_desc = f"{int(meta['grid_n'])}×{int(meta['grid_n'])} rect / soften={int(meta.get('edge_soften_px', 0))} px"
    kv("Маски, baseline", f"{mask_desc}; {meta['baseline']}")
    kv("Stretch P-low..P-high; γ", f"P{float(meta['p_low']):.1f}..P{float(meta['p_high']):.1f}; {float(meta['gamma']):.2f}")

    # Footer
    footer = "Score-CAM • CLIP vision attribution • cos(image, text)"
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
        "image_path": image_path, "method": "score_cam_gauss_overlap",
        "config": asdict(cfg), "caption_generated": gen_caption,
        "caption_reference": (ref_caption if cfg.show_reference else None),
        "clip_cosine": base_cos, "heatmap_stats": stats, "meta": meta,
        "outputs": {"png": str(out_png)}
    }
    with open(out_dir / f"{stem}.json","w",encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return data
