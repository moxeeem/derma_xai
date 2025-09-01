# final/methods/grad_cam.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass, asdict, fields
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, Union
import json, math, numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------- Config ---------------------------------
@dataclass
class GradCAMConfig:
    canvas_width_px: int = 720
    canvas_height_px: int = 1400
    canvas_height_max_px: int = 4000
    margins_px: int = 16
    block_gap_px: int = 10
    image_block_height_px: int = 360

    font_path_regular: str = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    fs_header: int = 22; fs_text: int = 20; fs_small: int = 16; fs_text_min: int = 14
    line_pad_px: int = 6

    # Постобработка
    p_low: float = 80.0; p_high: float = 99.5
    gamma: float = 0.9
    heat_smooth_sigma_px: float = 2.0

    # Overlay
    overlay_alpha: float = 0.45
    alpha_mode: str = "scaled"

    # Метрики
    tau: float = 0.60

    # Хинт для выбора слоя
    target_layer_hint: Optional[str] = None
    show_reference: bool = True
    auto_shrink_text: bool = True
    auto_expand_canvas: bool = True

def _coerce_cfg(config: Optional[Union[GradCAMConfig, Dict[str, Any]]]) -> GradCAMConfig:
    if isinstance(config, GradCAMConfig): return config
    d = dict(config or {}); valid = {f.name for f in fields(GradCAMConfig)}
    return GradCAMConfig(**{k:v for k,v in d.items() if k in valid})

# ----------------------------- Utils ----------------------------------
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
    if str(alpha_mode).lower()=="uniform":
        a_img = Image.new("L",(W,H),int(np.clip(alpha,0,1)*255))
    else:
        a = np.clip(heat,0.0,1.0) * float(np.clip(alpha,0.0,1.0))
        a_img = Image.fromarray((a*255).astype(np.uint8),"L")
    out = base.copy(); out.paste(hm_img,(0,0),a_img); return out

def heatmap_stats(hm: np.ndarray, tau: float=0.6) -> Dict[str,float]:
    h = normalize01(hm); cov = float((h>=tau).mean())
    flat = np.sort(h.reshape(-1))[::-1]; k = max(1,int(0.10*flat.size))
    mass = float(flat[:k].sum()/(flat.sum()+1e-8))
    ent = float(-np.sum(h*np.log(h+1e-8))/h.size)
    return {"coverage@τ": cov, "mass@top10%": mass, "entropy": ent}

# ----------------------- Backend-specific glue -------------------------
def _locate_backend(clip):
    if hasattr(clip, "model") and hasattr(clip.model, "vision_model") and callable(getattr(clip, "processor", None)):
        vm = clip.model.vision_model
        dev = getattr(clip, "device", next(vm.parameters()).device)
        proc = clip.processor
        return "hf", vm, proc, dev
    visual = getattr(getattr(clip, "model", None), "visual", None) or getattr(clip, "visual", None)
    pre = getattr(clip, "preprocess", None)
    if visual is not None and callable(pre):
        dev = getattr(clip, "device", next(visual.parameters()).device)
        return "openclip", visual, pre, dev
    raise RuntimeError("Cannot locate CLIP visual + preprocess/processor")

def _is_container(m: nn.Module) -> bool:
    return isinstance(m, (nn.ModuleList, nn.Sequential))

def _select_last_block(visual: nn.Module, hint: Optional[str]) -> Tuple[nn.Module, str]:
    # 1) Если есть хинт — ищем модуль с совпадением,
    #    но игнорируем контейнеры; для контейнера берём его последний элемент.
    if hint:
        found: Optional[Tuple[nn.Module,str]] = None
        for name, m in visual.named_modules():
            if hint in name:
                if _is_container(m):
                    try:
                        last = m[-1]
                        return last, f"{name}[-1]"
                    except Exception:
                        continue
                else:
                    found = (m, name)
        if found is not None:
            return found

    # 2) Дефолтные пути для популярных реализаций
    if hasattr(visual, "transformer") and hasattr(visual.transformer, "resblocks"):
        blocks = visual.transformer.resblocks
        if _is_container(blocks) and len(blocks):
            return blocks[-1], "visual.transformer.resblocks[-1]"
    if hasattr(visual, "blocks"):
        blocks = visual.blocks
        if _is_container(blocks) and len(blocks):
            return blocks[-1], "visual.blocks[-1]"
    if hasattr(visual, "encoder") and hasattr(visual.encoder, "layers"):
        layers = visual.encoder.layers
        if _is_container(layers) and len(layers):
            return layers[-1], "vision_model.encoder.layers[-1]"
    return visual, visual.__class__.__name__

def _tokens_to_cam_generic(acts: torch.Tensor, grads: torch.Tensor) -> torch.Tensor:
    """
    acts/grads: [B,T,C] или [B,C,T].
    Вернёт CAM для ViT-токенов: [B,1,g,g], где g×g — число patch-токенов.
    """
    assert acts.dim() == 3 and grads.dim() == 3
    # приводим к [B,T,C]
    if acts.shape[1] > acts.shape[2]:   # [B,C,T] -> [B,T,C]
        A = acts.transpose(1, 2).contiguous()
        G = grads.transpose(1, 2).contiguous()
    else:
        A, G = acts, grads

    B, T, C = A.shape
    # выбрасываем CLS и приводим к квадрату
    Tp = T - 1
    g = int(math.floor(math.sqrt(Tp)))
    if g <= 0:
        raise RuntimeError("Too few tokens for CAM")
    A = A[:, 1:1 + g * g, :]   # [B,g^2,C]
    G = G[:, 1:1 + g * g, :]   # [B,g^2,C]

    # Вариант 1 (классический Grad-CAM по ViT-токенам):
    # веса = средний положительный градиент по токенам
    w_pos = F.relu(G).mean(dim=1, keepdim=True)      # [B,1,C]
    S1 = (A * w_pos).sum(dim=2, keepdim=True)        # [B,g^2,1]
    S1 = F.relu(S1).transpose(1, 2).contiguous()     # [B,1,g^2]

    # Вариант 2 (elementwise): ReLU(grad) ⊙ act -> суммируем по каналам
    S2 = (F.relu(G) * A).sum(dim=2, keepdim=True)    # [B,g^2,1]
    S2 = F.relu(S2).transpose(1, 2).contiguous()     # [B,1,g^2]

    # берём тот, где сигнал сильнее; если оба вырождены — мягкий fallback |w|
    max1 = S1.max().item() if S1.numel() else 0.0
    max2 = S2.max().item() if S2.numel() else 0.0
    S = S1 if max1 >= max2 else S2

    if (S.max() - S.min()).item() <= 1e-12:
        w_abs = G.abs().mean(dim=1, keepdim=True)    # [B,1,C]
        S = (A * w_abs).sum(dim=2, keepdim=True).transpose(1, 2).contiguous()  # [B,1,g^2]

    return S.view(B, 1, g, g)


def _post_heat(heat: np.ndarray, cfg) -> np.ndarray:
    """
    Percentile stretch → gamma → (опц.) Gaussian blur → normalize to [0,1]
    """
    h = np.asarray(heat, dtype=np.float32)
    # stretch
    try:
        lo = np.percentile(h, float(cfg.p_low))
        hi = np.percentile(h, float(cfg.p_high))
        if hi > lo:
            h = np.clip((h - lo) / (hi - lo + 1e-8), 0.0, 1.0)
    except Exception:
        h = np.clip(h, 0.0, 1.0)

    # gamma
    g = float(getattr(cfg, "gamma", 1.0) or 1.0)
    if abs(g - 1.0) > 1e-6:
        h = np.power(np.clip(h, 0.0, 1.0), max(0.05, g))
        # re-norm
        mn, mx = float(h.min()), float(h.max())
        if mx > mn + 1e-8:
            h = (h - mn) / (mx - mn + 1e-8)
        else:
            h[:] = 0.0

    # smooth
    sig = float(getattr(cfg, "heat_smooth_sigma_px", 0.0) or 0.0)
    if sig > 0:
        im = Image.fromarray((np.clip(h, 0, 1) * 255).astype(np.uint8), mode="L")
        im = im.filter(ImageFilter.GaussianBlur(radius=sig))
        h = np.asarray(im).astype(np.float32) / 255.0

    # final clamp
    return np.clip(h, 0.0, 1.0)


def compute_grad_cam(clip, image: Image.Image, text: str, cfg: GradCAMConfig
                     ) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    # 0) базовый cos для метрик
    with torch.no_grad():
        tfeat = clip.encode_text([text])
        base_img_feat = clip.encode_image([image])
        base_cos = float(cosine_sim(base_img_feat, tfeat)[0].item())

    backend, visual, preproc, device = _locate_backend(clip)

    # 1) соберём список кандидатов-слоёв
    candidates: list[tuple[nn.Module, str]] = []
    # пост-норм перед пуллингом — лучший вариант для ViT
    if hasattr(visual, "ln_post") and isinstance(getattr(visual, "ln_post"), nn.Module):
        candidates.append((visual.ln_post, "visual.ln_post"))
    if hasattr(visual, "norm") and isinstance(getattr(visual, "norm"), nn.Module):
        candidates.append((visual.norm, "visual.norm"))

    # последние два трансформер-блока
    tr = getattr(visual, "transformer", None)
    if tr is not None and hasattr(tr, "resblocks"):
        blocks = tr.resblocks
        if isinstance(blocks, (nn.Sequential, nn.ModuleList)) and len(blocks) > 0:
            candidates.append((blocks[-1], "visual.transformer.resblocks[-1]"))
            if len(blocks) >= 2:
                candidates.append((blocks[-2], "visual.transformer.resblocks[-2]"))

    # timm/hf-style
    if hasattr(visual, "blocks"):
        blocks = visual.blocks
        if isinstance(blocks, (nn.Sequential, nn.ModuleList)) and len(blocks) > 0:
            nm = "visual.blocks"
            if (blocks[-1], f"{nm}[-1]") not in candidates:
                candidates.append((blocks[-1], f"{nm}[-1]"))
            if len(blocks) >= 2:
                candidates.append((blocks[-2], f"{nm}[-2]"))

    # запасной вариант — последний блок через нашу старую евристику
    if not candidates:
        tgt, tgt_name = _select_last_block(visual, cfg.target_layer_hint)
        candidates.append((tgt, tgt_name))

    # 2) временно включим градиенты у визуалки
    prev_req = [p.requires_grad for p in visual.parameters()]
    for p in visual.parameters():
        p.requires_grad_(True)

    # хранилища активаций/градов для всех кандидатов
    stores = []
    hooks = []
    for mod, name in candidates:
        st = {"name": name, "acts": None, "grads": None}
        def _make_fhook(store_ref):
            def fh(_m, _inp, out):
                if isinstance(out, torch.Tensor):
                    store_ref["acts"] = out
                    out.retain_grad()
                    out.register_hook(lambda g: store_ref.__setitem__("grads", g))
            return fh
        hooks.append(mod.register_forward_hook(_make_fhook(st)))
        stores.append(st)

    try:
        p0 = next(visual.parameters())
        dtype = p0.dtype

        # 3) прямой прогон через визуалку (не encode_image, чтобы точно был граф)
        if backend == "openclip":
            x = preproc(image).unsqueeze(0).to(device=device, dtype=dtype, non_blocking=True)
            visual.eval()
            feats = visual(x)                      # [B,D]
            ifeat = F.normalize(feats, dim=-1)
        else:
            proc = preproc(images=[image], return_tensors='pt')
            pv = proc['pixel_values'].to(device=device, dtype=dtype)
            out = visual(pixel_values=pv, output_attentions=False, return_dict=True)
            hidden = out.last_hidden_state         # [B,T,C]
            pooled = hidden[:, 0, :]
            proj = getattr(getattr(clip, "model", None), "visual_projection", None)
            feats = proj(pooled) if proj is not None else pooled
            ifeat = F.normalize(feats, dim=-1)

        # 4) цель — cos(ifeat, tfeat)
        sim = cosine_sim(ifeat, tfeat)
        if hasattr(clip, "model") and isinstance(clip.model, nn.Module):
            clip.model.zero_grad(set_to_none=True)
        visual.zero_grad(set_to_none=True)
        sim.mean().backward()

        # 5) построим CAM по каждому кандидату
        H0, W0 = image.height, image.width
        cams: list[Tuple[np.ndarray, str, float]] = []  # (heat, name, variance)

        for st in stores:
            A, G = st["acts"], st["grads"]
            if A is None or G is None:
                continue

            # ViT-токены [B,T,C]
            if A.dim() == 3 and G.dim() == 3:
                L = _tokens_to_cam_generic(A.to(torch.float32), G.to(torch.float32))   # [B,1,g,g]
                h = F.interpolate(L, size=(H0, W0), mode="bilinear", align_corners=False)
                heat = h.squeeze().detach().cpu().numpy().astype(np.float32)
            # CNN [B,C,h,w]
            elif A.dim() == 4 and G.dim() == 4:
                Gp = F.relu(G.to(torch.float32)).mean(dim=(2, 3), keepdim=True)
                L1 = F.relu((Gp * A.to(torch.float32)).sum(dim=1, keepdim=True))
                L2 = F.relu((F.relu(G.to(torch.float32)) * A.to(torch.float32)).sum(dim=1, keepdim=True))
                L = L1 if (L1.max() >= L2.max()) else L2
                h = F.interpolate(L, size=(H0, W0), mode="bilinear", align_corners=False)
                heat = h.squeeze().detach().cpu().numpy().astype(np.float32)
            else:
                continue

            heat = _post_heat(heat, cfg)
            cams.append((heat, st["name"], float(np.var(heat))))

        # 6) выбор/агрегация
        if not cams:
            heat = np.zeros((H0, W0), dtype=np.float32)
            chosen = "n/a"
        else:
            # берём карту с наибольшей дисперсией (наиболее «контрастную»)
            cams.sort(key=lambda x: x[2], reverse=True)
            best_heat, chosen, best_var = cams[0]
            # если сигнал совсем слабый — усредним топ-3
            if best_var < 1e-6 and len(cams) >= 2:
                k = min(3, len(cams))
                heat = normalize01(np.mean([c[0] for c in cams[:k]], axis=0))
                chosen = "+".join([c[1] for c in cams[:k]])
            else:
                heat = best_heat

        meta = {
            "backend": backend,
            "target_layer": chosen,
            "p_low": float(cfg.p_low), "p_high": float(cfg.p_high),
            "gamma": float(cfg.gamma),
            "heat_smooth_sigma_px": float(cfg.heat_smooth_sigma_px),
            "overlay_alpha": float(cfg.overlay_alpha),
            "alpha_mode": str(cfg.alpha_mode),
        }

    except Exception as e:
        H0, W0 = image.height, image.width
        heat = np.zeros((H0, W0), dtype=np.float32)
        # Попробуем хотя бы что-то разумное показать при аварии — нули безопаснее
        meta = {"backend": f"error:{backend}",
                "target_layer": candidates[0][1] if candidates else "n/a",
                "error": f"{type(e).__name__}: {e}"}
    finally:
        for h in hooks:
            try: h.remove()
            except: pass
        for p, flag in zip(visual.parameters(), prev_req):
            p.requires_grad_(flag)
        try:
            torch.cuda.empty_cache(); torch.cuda.ipc_collect()
        except: pass

    return heat, base_cos, meta


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
        if tw <= max_w: line = t
        else:
            drw.text((x, yy), line, font=font, fill=(0, 0, 0)); yy += th + gap; line = w
    if line:
        drw.text((x, yy), line, font=font, fill=(0, 0, 0)); yy += _text_size(drw, line, font)[1]
    return yy

def _measure_total_height(W: int, cfg: GradCAMConfig, gen_caption: str, ref_caption: Optional[str],
                          f_hdr, f_txt, f_sml) -> int:
    tmp = Image.new("RGB", (W, 200), (255, 255, 255)); drw = ImageDraw.Draw(tmp)
    M, GAP = cfg.margins_px, cfg.block_gap_px; col_w = W - 2 * M
    total = 0
    total += _text_size(drw, "Original image", f_hdr)[1] + 6
    total += cfg.image_block_height_px + GAP
    total += _text_size(drw, "Grad-CAM overlay", f_hdr)[1] + 6
    total += cfg.image_block_height_px + GAP
    total += _text_size(drw, "Generated caption (for CLIP score)", f_hdr)[1] + 6
    total += _wrap_text(drw, 0, 0, col_w, gen_caption or "(empty)", f_txt, cfg.line_pad_px) + GAP
    if cfg.show_reference and ref_caption:
        total += _text_size(drw, "Reference caption", f_hdr)[1] + 6
        total += _wrap_text(drw, 0, 0, col_w, ref_caption, f_txt, cfg.line_pad_px) + GAP
    total += _text_size(drw, "Heat (low → high)", f_sml)[1] + 4
    total += 14 + 26 + GAP
    total += _text_size(drw, "Metrics (explained)", f_hdr)[1] + 6
    for lbl in ["Косинус(изобр., полный текст)", f"Покрытие ≥ {cfg.tau:.2f}",
                "Масса топ-10% областей", "Энтропия карты", "Слой/бэкенд",
                "Stretch P-low..P-high; γ"]:
        total += _text_size(drw, f"{lbl}: 0.000", f_sml)[1] + 4
    total += _text_size(drw, "Grad-CAM • CLIP vision attribution", f_sml)[1] + 8
    return total

def _render_card_vertical(out_png: Path, image: Image.Image, heat01: np.ndarray,
                          gen_caption: str, ref_caption: Optional[str],
                          base_cos: float, meta: Dict[str, Any], cfg: GradCAMConfig):
    W = cfg.canvas_width_px; M, GAP = cfg.margins_px, cfg.block_gap_px; col_w = W - 2 * M
    f_hdr = _find_font(cfg.font_path_regular, cfg.fs_header)
    f_txt = _find_font(cfg.font_path_regular, cfg.fs_text)
    f_sml = _find_font(cfg.font_path_regular, cfg.fs_small)

    fs_caption = cfg.fs_text
    total = _measure_total_height(W, cfg, gen_caption, ref_caption if cfg.show_reference else None, f_hdr, f_txt, f_sml)
    if cfg.auto_shrink_text:
        while total + 2 * M > cfg.canvas_height_max_px and fs_caption > cfg.fs_text_min:
            fs_caption -= 1; f_txt = _find_font(cfg.font_path_regular, fs_caption)
            total = _measure_total_height(W, cfg, gen_caption, ref_caption if cfg.show_reference else None, f_hdr, f_txt, f_sml)

    H = max(cfg.canvas_height_px, total + 2 * M)
    if cfg.auto_expand_canvas: H = min(max(H, cfg.canvas_height_px), cfg.canvas_height_max_px)

    canvas = Image.new("RGB", (W, H), (255, 255, 255)); drw = ImageDraw.Draw(canvas)
    x = M; y = M

    # 1) Original
    drw.text((x, y), "Original image", font=f_hdr, fill=(0, 0, 0))
    y += _text_size(drw, "Original image", f_hdr)[1] + 6
    img_fit = _fit_cover(image, col_w, cfg.image_block_height_px); canvas.paste(img_fit, (x, y))
    y += cfg.image_block_height_px + GAP

    # 2) Overlay
    drw.text((x, y), "Grad-CAM overlay", font=f_hdr, fill=(0, 0, 0))
    y += _text_size(drw, "Grad-CAM overlay", f_hdr)[1] + 6
    base_block = _fit_cover(image, col_w, cfg.image_block_height_px)
    heat_block = _fit_cover_gray(heat01, col_w, cfg.image_block_height_px)
    overlay = _overlay_with_heat(base_block, heat_block, float(cfg.overlay_alpha), cfg.alpha_mode)
    canvas.paste(overlay, (x, y)); y += cfg.image_block_height_px + GAP

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
        line = f"{k}: {v}"; drw.text((x, y), line, font=f_sml, fill=(0, 0, 0))
        y += _text_size(drw, line, f_sml)[1] + 4

    stats = heatmap_stats(heat01, tau=float(cfg.tau))
    kv("Косинус(изобр., полный текст)", f"{float(base_cos):.3f}")
    kv(f"Покрытие ≥ {cfg.tau:.2f}", f"{stats['coverage@τ']:.3f}")
    kv("Масса топ-10% областей", f"{stats['mass@top10%']:.3f}")
    kv("Энтропия карты", f"{stats['entropy']:.3f}")
    kv("Слой/бэкенд", f"{meta.get('target_layer','-')}; {meta.get('backend','-')}")
    kv("Stretch P-low..P-high; γ", f"P{float(cfg.p_low):.1f}..P{float(cfg.p_high):.1f}; {float(cfg.gamma):.2f}")

    footer = "Grad-CAM • CLIP vision attribution • cos(image, text)"
    fw, fh = _text_size(drw, footer, f_sml)
    drw.text((W - fw - M, H - fh - 6), footer, font=f_sml, fill=(90, 90, 90))

    canvas.save(str(out_png), format="PNG")

# --------------------------- Public API --------------------------------
def run(clip, captioner, *, image_path: str, out_dir: Path,
        ref_caption: Optional[str] = None, config: Optional[GradCAMConfig] = None) -> Dict[str, Any]:
    cfg = _coerce_cfg(config)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    image = load_image(image_path, fallback_hw=getattr(clip, "image_size", 224))
    try:
        gen_caption = captioner.generate([image], prompt=getattr(captioner, "prompt", None) or "Describe the image.")[0]
    except Exception:
        gen_caption = ""

    heat, base_cos, meta = compute_grad_cam(clip, image, gen_caption, cfg)

    stem = Path(image_path).stem
    out_png = out_dir / f"{stem}.png"
    _render_card_vertical(out_png, image, heat, gen_caption,
                          (ref_caption if cfg.show_reference else None),
                          base_cos, meta, cfg)

    stats = heatmap_stats(heat, tau=float(cfg.tau))
    data = {
        "image_path": image_path, "method": "grad_cam",
        "config": asdict(cfg), "caption_generated": gen_caption,
        "caption_reference": (ref_caption if cfg.show_reference else None),
        "clip_cosine": base_cos, "heatmap_stats": stats, "meta": meta,
        "outputs": {"png": str(out_png)}
    }
    with open(out_dir / f"{stem}.json","w",encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return data
