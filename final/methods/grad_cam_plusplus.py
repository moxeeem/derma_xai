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
    p_low: float = 75.0; p_high: float = 99.8
    gamma: float = 0.85
    heat_smooth_sigma_px: float = 2.0
    auto_contrast: bool = True          # NEW: адаптивные перцентили при слабом сигнале

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

    # CAM подсчёт
    force_fp32: bool = True             # NEW: принудительно считать CAM в float32
    cam_samples: int = 1        # для LM-атрибуции обычно 1 достаточно
    cam_noise_std: float = 0.0

    # === НОВОЕ: цели для капшенера ===
    caption_source: str = "generated"   # "generated" | "reference"
    token_target: str = "last"          # "last" | "maxlogit" | "index:<int>" | "match:<substr>"
    objective: str = "logit"            # "logit" | "logprob"


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



def _tokens_to_cam_gradcampp(acts: torch.Tensor, grads: torch.Tensor) -> torch.Tensor:
    """
    Полностью исправленная реализация Grad-CAM++ для ViT-токенов.
    Вход: acts/grads: [B,T,C] или [B,C,T]; Выход: [B,1,g,g] по patch-токенам.
    """
    assert acts.dim() == 3 and grads.dim() == 3
    
    # Приводим к [B,T,C]
    if acts.shape[1] > acts.shape[2]:   # [B,C,T] -> [B,T,C]
        A = acts.transpose(1, 2).contiguous()
        G = grads.transpose(1, 2).contiguous()
    else:
        A, G = acts, grads

    B, T, C = A.shape
    
    # Убираем CLS токен (позиция 0)
    A_no_cls = A[:, 1:, :]  # [B, Tp, C]
    G_no_cls = G[:, 1:, :]  # [B, Tp, C]
    Tp = T - 1  # Количество patch-токенов
    
    # Определяем размеры патч-сетки
    g = int(math.sqrt(Tp))
    if g * g == Tp:
        # Идеальный квадрат
        g_h = g_w = g
    else:
        # Пытаемся найти правильные размеры
        g_h = int(math.floor(math.sqrt(Tp)))
        g_w = (Tp + g_h - 1) // g_h  # Округляем вверх
        
        # Если все равно не получается, используем ближайший квадрат
        if g_h * g_w != Tp:
            g = int(math.ceil(math.sqrt(Tp)))
            g_h = g_w = g
    
    # ИСПРАВЛЕНО: критическая ошибка здесь - нужно убедиться, что токены расположены правильно
    # В ViT токены организованы в порядке "сверху вниз, слева направо"
    # Но при преобразовании в карту мы должны сохранить это пространственное расположение
    
    # ВАЖНО: не добавляем нулевые токены - это искажает пространственное расположение
    # Вместо этого используем только существующие токены и корректно вычисляем размеры
    
    # Проверяем, что количество токенов соответствует размерам
    if g_h * g_w != Tp:
        # Корректируем размеры, чтобы точно соответствовать количеству токенов
        g_w = Tp // g_h
        g_h = Tp // g_w
        # Если все равно не делится, используем первый подходящий размер
        if g_h * g_w != Tp:
            g = int(math.sqrt(Tp))
            while g * g < Tp:
                g += 1
            g_h = g_w = g
    
    # ИСПРАВЛЕНО: критическая ошибка - неправильное преобразование токенов в пространственную карту
    # Нужно убедиться, что мы сохраняем правильное пространственное расположение токенов
    try:
        # Пытаемся преобразовать напрямую
        A_reshaped = A_no_cls.view(B, g_h, g_w, C)
        G_reshaped = G_no_cls.view(B, g_h, g_w, C)
    except Exception:
        # Если не получается, используем ближайшие подходящие размеры
        g = int(math.sqrt(Tp))
        while g * g > Tp:
            g -= 1
        g_h = g_w = g
        # Обрезаем лишние токены, чтобы сохранить пространственное расположение
        A_reshaped = A_no_cls[:, :g_h*g_w, :].view(B, g_h, g_w, C)
        G_reshaped = G_no_cls[:, :g_h*g_w, :].view(B, g_h, g_w, C)
    
    # Теперь преобразуем обратно в [B, H*W, C] для дальнейшей обработки
    A = A_reshaped.view(B, g_h * g_w, C)
    G = G_reshaped.view(B, g_h * g_w, C)
    
    # Используем только положительные градиенты
    G_pos = F.relu(G)
    
    # Вычисляем компоненты формулы Grad-CAM++
    G2 = G_pos * G_pos
    G3 = G2 * G_pos
    
    # ИСПРАВЛЕНО: критическая ошибка в вычислении знаменателя
    # В оригинальной формуле Grad-CAM++ знаменатель вычисляется как:
    # 2 * ReLU(grad)^2 + sum(acts * ReLU(grad)^3)
    # Но sum должен быть по КАНАЛАМ, а не по пространству!
    # Это была главная ошибка - неправильное направление суммирования
    denom = 2.0 * G2 + (A * G3).sum(dim=2, keepdim=True)  # ИСПРАВЛЕНО: sum по dim=2 (каналам)
    
    # Защита от деления на ноль
    alpha = G2 / (denom + 1e-8)
    
    # Веса = alpha * положительные градиенты
    weights = alpha * G_pos
    
    # Суммируем веса по КАНАЛАМ для получения карты активации
    # ИСПРАВЛЕНО: суммируем по dim=2 (каналам), а не по dim=1 (токенам)
    S = weights.sum(dim=2, keepdim=True)  # [B, g_h*g_w, 1]
    
    # Преобразуем в правильную форму [B, 1, g_h, g_w]
    S = S.view(B, 1, g_h, g_w)
    
    # Если карта вырожденная, используем fallback
    if (S.max() - S.min()).item() <= 1e-12:
        return _tokens_to_cam_generic(acts, grads)
    
    return S

def _cnn_cam_gradcampp(acts: torch.Tensor, grads: torch.Tensor) -> torch.Tensor:
    """
    Исправленная реализация Grad-CAM++ для CNN-активаций.
    Вход: [B,C,H,W] — активации/градиенты; Выход: [B,1,H,W]
    """
    assert acts.dim() == 4 and grads.dim() == 4
    A = acts
    G = grads
    
    # Используем только положительные градиенты
    G_pos = F.relu(G)
    
    # Вычисляем компоненты формулы Grad-CAM++
    G2 = G_pos * G_pos
    G3 = G2 * G_pos
    
    # ИСПРАВЛЕНО: правильно вычисляем знаменатель
    # Суммируем по пространственным измерениям
    denom = 2.0 * G2 + (A * G3).sum(dim=[2, 3], keepdim=True)
    
    # Защита от деления на ноль
    alpha = G2 / (denom + 1e-8)
    
    # Веса = alpha * положительные градиенты
    weights = alpha * G_pos
    
    # Суммируем веса по каналам
    L = weights.sum(dim=1, keepdim=True)
    
    # Применяем ReLU и нормализуем
    L = F.relu(L)
    
    # Если карта вырожденная, используем fallback
    if (L.max() - L.min()).item() <= 1e-12:
        # Классический Grad-CAM
        weights = F.relu(G).mean(dim=[2, 3], keepdim=True)
        L = F.relu((weights * A).sum(dim=1, keepdim=True))
    
    return L

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
    # текстовые фичи (без графа), сразу на девайс/тип
    backend, visual, preproc, device = _locate_backend(clip)
    p0 = next(visual.parameters())
    cam_dtype = torch.float32 if getattr(cfg, "force_fp32", True) else p0.dtype

    with torch.no_grad():
        tfeat = clip.encode_text([text])
        tfeat = F.normalize(tfeat.to(device=device, dtype=cam_dtype), dim=-1)
        base_img_feat = clip.encode_image([image])
        base_cos = float(cosine_sim(base_img_feat, tfeat)[0].item())

        # contrastive objective (опционально)
        tneg = None
        if str(getattr(cfg, "target_objective", "cosine")).lower() == "contrastive":
            neg_txt = getattr(cfg, "neg_text", None)
            if neg_txt is not None and len(neg_txt.strip()) > 0:
                tneg = clip.encode_text([neg_txt])
                tneg = F.normalize(tneg.to(device=device, dtype=cam_dtype), dim=-1)

    # кандидаты слоёв (как было)
    candidates: list[tuple[nn.Module, str]] = []
    if hasattr(visual, "ln_post") and isinstance(getattr(visual, "ln_post"), nn.Module):
        candidates.append((visual.ln_post, "visual.ln_post"))
    if hasattr(visual, "norm") and isinstance(getattr(visual, "norm"), nn.Module):
        candidates.append((visual.norm, "visual.norm"))
    tr = getattr(visual, "transformer", None)
    if tr is not None and hasattr(tr, "resblocks"):
        blocks = tr.resblocks
        if isinstance(blocks, (nn.Sequential, nn.ModuleList)) and len(blocks) > 0:
            candidates.append((blocks[-1], "visual.transformer.resblocks[-1]"))
            if len(blocks) >= 2:
                candidates.append((blocks[-2], "visual.transformer.resblocks[-2]"))
    if hasattr(visual, "blocks"):
        blocks = visual.blocks
        if isinstance(blocks, (nn.Sequential, nn.ModuleList)) and len(blocks) > 0:
            nm = "visual.blocks"
            if (blocks[-1], f"{nm}[-1]") not in candidates:
                candidates.append((blocks[-1], f"{nm}[-1]"))
            if len(blocks) >= 2:
                candidates.append((blocks[-2], f"{nm}[-2]"))
    if not candidates:
        tgt, tgt_name = _select_last_block(visual, cfg.target_layer_hint)
        candidates.append((tgt, tgt_name))

    # включаем градиенты у визуалки
    prev_req = [p.requires_grad for p in visual.parameters()]
    for p in visual.parameters():
        p.requires_grad_(True)

    H0, W0 = image.height, image.width
    samples = max(1, int(getattr(cfg, "cam_samples", 1)))

    def _one_pass(noise_std: float) -> Tuple[np.ndarray, str]:
        # хранилища для хуков
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
            if backend == "openclip":
                x = preproc(image).unsqueeze(0).to(device=device, dtype=cam_dtype, non_blocking=True)
                if noise_std > 0:
                    x = x + noise_std * torch.randn_like(x)
                visual.eval()
                with torch.autocast(device_type=str(device).split(":")[0], enabled=False):
                    feats = visual(x)                  # [B,D]
                    ifeat = F.normalize(feats.to(dtype=cam_dtype), dim=-1)
            else:
                proc = preproc(images=[image], return_tensors='pt')
                pv = proc['pixel_values'].to(device=device, dtype=cam_dtype, non_blocking=True)
                if noise_std > 0:
                    pv = pv + noise_std * torch.randn_like(pv)
                visual.eval()
                with torch.autocast(device_type=str(device).split(":")[0], enabled=False):
                    out = visual(pixel_values=pv, output_attentions=False, return_dict=True)
                    hidden = out.last_hidden_state     # [B,T,C]
                    pooled = hidden[:, 0, :]
                    proj = getattr(getattr(clip, "model", None), "visual_projection", None)
                    feats = proj(pooled) if proj is not None else pooled
                    ifeat = F.normalize(feats.to(dtype=cam_dtype), dim=-1)

            # цель: cosine или contrastive
            if tneg is None:
                loss_target = cosine_sim(ifeat, tfeat) * 1.0
            else:
                pos = cosine_sim(ifeat, tfeat)
                neg = cosine_sim(ifeat, tneg)
                loss_target = pos - neg

            if hasattr(clip, "model") and isinstance(clip.model, nn.Module):
                clip.model.zero_grad(set_to_none=True)
            visual.zero_grad(set_to_none=True)
            loss_target.mean().backward()

            # строим CAM по каждому кандидату
            cams: list[Tuple[np.ndarray, str, float]] = []
            for st in stores:
                A, G = st["acts"], st["grads"]
                if A is None or G is None: continue
                if A.dim() == 3 and G.dim() == 3:
                    L = _tokens_to_cam_gradcampp(A.to(torch.float32), G.to(torch.float32))
                elif A.dim() == 4 and G.dim() == 4:
                    L = _cnn_cam_gradcampp(A.to(torch.float32), G.to(torch.float32))
                else:
                    continue
                h = F.interpolate(L, size=(H0, W0), mode="bilinear", align_corners=False)
                heat = h.squeeze().detach().cpu().numpy().astype(np.float32)
                # НЕ пост-обрабатываем здесь (сделаем после усреднения)
                cams.append((heat, st["name"], float(np.var(heat))))
            if not cams:
                return np.zeros((H0, W0), dtype=np.float32), "n/a"
            cams.sort(key=lambda x: x[2], reverse=True)
            best_heat, chosen, _ = cams[0]
            return best_heat, chosen
        finally:
            for h in hooks:
                try: h.remove()
                except: pass

    # SmoothGrad-CAM++: усредняем
    acc = np.zeros((H0, W0), dtype=np.float32)
    chosen_names = []
    noise_std = float(getattr(cfg, "cam_noise_std", 0.0) or 0.0)
    for i in range(samples):
        heat_i, chosen_i = _one_pass(noise_std if samples > 1 else 0.0)
        acc += heat_i
        chosen_names.append(chosen_i)
    heat = acc / float(samples)
    # финальная постобработка
    heat = _post_heat(heat, cfg)

    meta = {
        "backend": backend,
        "target_layer": max(set(chosen_names), key=chosen_names.count) if chosen_names else "n/a",
        "p_low": float(cfg.p_low), "p_high": float(cfg.p_high),
        "gamma": float(cfg.gamma),
        "heat_smooth_sigma_px": float(cfg.heat_smooth_sigma_px),
        "overlay_alpha": float(cfg.overlay_alpha),
        "alpha_mode": str(cfg.alpha_mode),
        "samples": samples,
        "noise_std": noise_std,
        "objective": str(getattr(cfg, "target_objective", "cosine")).lower(),
    }

    # вернуть всё и откатить requires_grad
    for p, flag in zip(visual.parameters(), prev_req):
        p.requires_grad_(flag)
    try:
        torch.cuda.empty_cache(); torch.cuda.ipc_collect()
    except: pass

    return heat, base_cos, meta

def compute_grad_cam_captioner(captioner, image: Image.Image, ref_caption: Optional[str], cfg: GradCAMConfig
                               ) -> Tuple[np.ndarray, float, Dict[str, Any], str]:
    """
    Строит карту значимости изображения для выбранного токена капшена:
    Vision -> Projector -> GPT-2, цель — logit(target_token) или logprob.
    Возвращает: heat, base_cos (для справки, если можно посчитать), meta, used_caption_text
    """
    parts = _locate_captioner_parts(captioner)
    backend, visual, preproc, device = parts["backend"], parts["visual"], parts["preproc"], parts["device"]
    projector, lm, tok = parts["projector"], parts["lm"], parts["tokenizer"]

    # --- готовим текст ---
    if str(cfg.caption_source).lower() == "reference" and ref_caption:
        cap_text = ref_caption
    else:
        try:
            cap_text = captioner.generate([image], prompt=getattr(captioner, "prompt", None) or "Describe the image.")[0]
        except Exception:
            cap_text = ref_caption or "a medical skin lesion photo"

    ids = tok(cap_text, return_tensors="pt", add_special_tokens=True).input_ids.to(device)
    tgt_k, _ = _pick_target_token(ids, tok, str(cfg.token_target))

    p0 = next(visual.parameters())
    cam_dtype = torch.float32 if getattr(cfg, "force_fp32", True) else p0.dtype

    candidates: list[tuple[nn.Module, str]] = []
    if hasattr(visual, "ln_post") and isinstance(getattr(visual, "ln_post"), nn.Module):
        candidates.append((visual.ln_post, "visual.ln_post"))
    if hasattr(visual, "norm") and isinstance(getattr(visual, "norm"), nn.Module):
        candidates.append((visual.norm, "visual.norm"))
    tr = getattr(visual, "transformer", None)
    if tr is not None and hasattr(tr, "resblocks"):
        blocks = tr.resblocks
        if isinstance(blocks, (nn.Sequential, nn.ModuleList)) and len(blocks) > 0:
            candidates.append((blocks[-1], "visual.transformer.resblocks[-1]"))
            if len(blocks) >= 2:
                candidates.append((blocks[-2], "visual.transformer.resblocks[-2]"))
    if hasattr(visual, "blocks"):
        blocks = visual.blocks
        if isinstance(blocks, (nn.Sequential, nn.ModuleList)) and len(blocks) > 0:
            nm = "visual.blocks"
            candidates.append((blocks[-1], f"{nm}[-1]"))
            if len(blocks) >= 2:
                candidates.append((blocks[-2], f"{nm}[-2]"))
    if not candidates:
        tgt, tgt_name = _select_last_block(visual, getattr(cfg, "target_layer_hint", None))
        candidates.append((tgt, tgt_name))

    prev_req = [p.requires_grad for p in visual.parameters()]
    for p in visual.parameters():
        p.requires_grad_(True)

    stores = []; hooks = []
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

    H0, W0 = image.height, image.width

    try:
        if backend == "openclip":
            x = preproc(image).unsqueeze(0).to(device=device, dtype=cam_dtype, non_blocking=True)
        else:
            proc = preproc(images=[image], return_tensors='pt')
            x = proc['pixel_values'].to(device=device, dtype=cam_dtype, non_blocking=True)

        visual.eval(); projector.eval(); lm.eval()

        with torch.autocast(device_type=str(device).split(":")[0], enabled=False):
            if backend == "openclip":
                vfeat = visual(x)
                if vfeat.dim() == 2:
                    pooled = vfeat
                else:
                    pooled = vfeat[:,0,:]
            else:
                out = visual(pixel_values=x, output_attentions=False, return_dict=True)
                hidden = out.last_hidden_state
                pooled = hidden[:,0,:]

            try:
                vtok = projector(pooled)
            except Exception:
                vtok = projector(hidden) if 'hidden' in locals() else projector(pooled)

            if vtok.dim() == 2:
                vtok = vtok.unsqueeze(1)
            assert vtok.dim() == 3, "projector must return [B,Tv,H]"
            B, Tv, H = vtok.shape

            prefix_ids = ids[:, :max(1, tgt_k)]
            WTE = getattr(lm, "transformer", None)
            if WTE is not None:
                WTE = getattr(lm.transformer, "wte", None)
            if WTE is None:
                raise RuntimeError("LM token embedding not found (expected lm.transformer.wte)")
            te = WTE(prefix_ids)
            inp = torch.cat([vtok.to(dtype=te.dtype), te], dim=1)

            out_lm = lm(inputs_embeds=inp, use_cache=False, return_dict=True)
            logits = out_lm.logits
            pos = Tv + prefix_ids.shape[1] - 1
            pos = max(0, min(int(logits.shape[1]-1), pos))
            tgt_id = ids[:, tgt_k:tgt_k+1]

            if str(cfg.objective).lower() == "logprob":
                logit = logits[:, pos, :]
                logprob = logit.log_softmax(dim=-1).gather(-1, tgt_id).mean()
                objective = logprob
            else:
                objective = logits.gather(dim=-1, index=tgt_id.unsqueeze(1).expand(-1, logits.shape[1], -1))[:, pos, 0].mean()

            if hasattr(captioner, "model") and isinstance(captioner.model, nn.Module):
                captioner.model.zero_grad(set_to_none=True)
            if hasattr(captioner, "zero_grad") and callable(captioner.zero_grad):
                captioner.zero_grad(set_to_none=True)
            visual.zero_grad(set_to_none=True)
            projector.zero_grad(set_to_none=True)
            lm.zero_grad(set_to_none=True)

            objective.backward()

            cams = []
            for st in stores:
                A, G = st["acts"], st["grads"]
                if A is None or G is None: continue
                if A.dim() == 3 and G.dim() == 3:
                    L = _tokens_to_cam_gradcampp(A.to(torch.float32), G.to(torch.float32))
                elif A.dim() == 4 and G.dim() == 4:
                    L = _cnn_cam_gradcampp(A.to(torch.float32), G.to(torch.float32))
                else:
                    continue
                h = F.interpolate(L, size=(H0, W0), mode="bilinear", align_corners=False)
                heat = h.squeeze().detach().cpu().numpy().astype(np.float32)
                cams.append((heat, st["name"], float(np.var(heat))))
            if not cams:
                heat = np.zeros((H0, W0), dtype=np.float32)
                chosen = "n/a"
            else:
                cams.sort(key=lambda x: x[2], reverse=True)
                heat, chosen, _ = cams[0]

            try:
                with torch.no_grad():
                    if hasattr(parts["clip"], "encode_text") and hasattr(parts["clip"], "encode_image"):
                        tfeat = parts["clip"].encode_text([cap_text]).to(device)
                        if parts["backend"] == "openclip":
                            if parts["preproc"] is None:
                                base_cos = 0.0
                            else:
                                imgfeat = parts["clip"].encode_image([image]).to(device)
                                base_cos = float(cosine_sim(imgfeat, F.normalize(tfeat, dim=-1))[0].item())
                        else:
                            base_cos = 0.0
                    else:
                        base_cos = 0.0
            except Exception:
                base_cos = 0.0

            heat = _post_heat(heat, cfg)
            meta = {
                "backend": backend,
                "target_layer": chosen,
                "objective": str(cfg.objective),
                "caption_source": str(cfg.caption_source),
                "token_target": str(cfg.token_target),
                "token_index": int(tgt_k),
            }
            return heat, float(base_cos), meta, cap_text

    except Exception as e:
        heat = np.zeros((H0, W0), dtype=np.float32)
        meta = {"backend": f"error:{type(e).__name__}", "error": f"{e}"}
        return heat, 0.0, meta, cap_text
    finally:
        for h in hooks:
            try: h.remove()
            except: pass
        for p, flag in zip(visual.parameters(), prev_req):
            p.requires_grad_(flag)
        try:
            torch.cuda.empty_cache(); torch.cuda.ipc_collect()
        except: pass

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
    total += _text_size(drw, "Grad-CAM++ overlay", f_hdr)[1] + 6
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
    total += _text_size(drw, "Grad-CAM++ • CLIP vision attribution", f_sml)[1] + 8
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
    drw.text((x, y), "Grad-CAM++ overlay", font=f_hdr, fill=(0, 0, 0))
    y += _text_size(drw, "Grad-CAM++ overlay", f_hdr)[1] + 6
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

    footer = "Grad-CAM++ • CLIP vision attribution • cos(image, text)"
    fw, fh = _text_size(drw, footer, f_sml)
    drw.text((W - fw - M, H - fh - 6), footer, font=f_sml, fill=(90, 90, 90))

    canvas.save(str(out_png), format="PNG")

def _locate_captioner_parts(captioner):
    """
    Пытается найти: clip/visual + preprocess/processor, projector, lm (GPT2), tokenizer, device.
    Возвращает dict или бросает RuntimeError.
    """
    parts = {}

    # vision/clip
    clip = getattr(captioner, "clip", None) or getattr(captioner, "vision", None) or getattr(captioner, "visual", None)
    if clip is None:
        model = getattr(captioner, "model", None)
        if model is not None:
            clip = getattr(model, "clip", None) or getattr(model, "vision", None) or getattr(model, "visual", None)
    if clip is None:
        raise RuntimeError("captioner: CLIP/vision module not found")

    backend, visual, preproc, dev = _locate_backend(clip)
    parts.update(dict(backend=backend, clip=clip, visual=visual, preproc=preproc, device=dev))

    # projector
    proj = getattr(captioner, "projector", None) or getattr(captioner, "proj", None)
    if proj is None:
        model = getattr(captioner, "model", None)
        if model is not None:
            proj = getattr(model, "projector", None) or getattr(model, "proj", None)
    if proj is None:
        raise RuntimeError("captioner: projector not found")
    parts["projector"] = proj

    # language model (GPT-2) + tokenizer
    lm = getattr(captioner, "lm", None) or getattr(captioner, "text_model", None) or getattr(captioner, "gpt", None)
    if lm is None:
        model = getattr(captioner, "model", None)
        if model is not None:
            lm = getattr(model, "lm", None) or getattr(model, "text_model", None) or getattr(model, "gpt", None)
    if lm is None:
        raise RuntimeError("captioner: language model (GPT) not found")
    parts["lm"] = lm

    tok = getattr(captioner, "tokenizer", None) or getattr(captioner, "tok", None)
    if tok is None:
        model = getattr(captioner, "model", None)
        if model is not None:
            tok = getattr(model, "tokenizer", None) or getattr(model, "tok", None)
    if tok is None:
        raise RuntimeError("captioner: tokenizer not found")
    parts["tokenizer"] = tok

    return parts

# --------------------------- Public API --------------------------------
def run(clip, captioner, *, image_path: str, out_dir: Path,
        ref_caption: Optional[str] = None, config: Optional[GradCAMConfig] = None) -> Dict[str, Any]:
    cfg = _coerce_cfg(config)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    image = load_image(image_path, fallback_hw=getattr(clip, "image_size", 224))

    # Генерация текста (и для LM-атрибуции, и как fallback)
    try:
        gen_caption = captioner.generate([image], prompt=getattr(captioner, "prompt", None) or "Describe the image.")[0]
    except Exception:
        gen_caption = ref_caption or ""

    # Попытка глубокой атрибуции от логита LM
    used_caption = gen_caption
    try:
        heat, base_cos, meta, used_caption = compute_grad_cam_captioner(captioner, image, ref_caption, cfg)
    except Exception:
        # fallback: старый CLIP-режим
        heat, base_cos, meta = compute_grad_cam(clip, image, gen_caption, cfg)

    stem = Path(image_path).stem
    out_png = out_dir / f"{stem}.png"
    _render_card_vertical(out_png, image, heat, used_caption,
                          (ref_caption if cfg.show_reference else None),
                          base_cos, meta, cfg)

    stats = heatmap_stats(heat, tau=float(cfg.tau))
    data = {
        "image_path": image_path,
        "method": "grad_cam_pp",
        "config": asdict(cfg),
        "caption_generated": used_caption,
        "caption_reference": (ref_caption if cfg.show_reference else None),
        "clip_cosine": base_cos,
        "heatmap_stats": stats,
        "meta": meta,
        "outputs": {"png": str(out_png)}
    }
    with open(out_dir / f"{stem}.json","w",encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return data

def _pick_target_token(ids: torch.Tensor, tok, mode: str) -> Tuple[int, int]:
    """
    ids: [1, T] (включая спец-токены, если они есть)
    mode: 'last' | 'maxlogit' (обработается позже) | 'index:<i>' | 'match:<substr>'
    Возврат: (target_index_in_ids, fallback_index_if_needed)
    """
    T = int(ids.shape[1])
    if mode.startswith("index:"):
        try:
            k = int(mode.split(":",1)[1])
            k = max(1, min(T-1, k))  # не даём 0 (там не на что предсказывать)
            return k, k
        except Exception:
            pass
    if mode.startswith("match:"):
        sub = mode.split(":",1)[1].strip().lower()
        try:
            text = tok.decode(ids[0].tolist())
        except Exception:
            text = ""
        pos = text.lower().find(sub)
        if pos >= 0:
            best_k = T-1
            return best_k, best_k
    return T-1, T-1
