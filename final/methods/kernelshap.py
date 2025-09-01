# final/methods/kernelshap.py
# -*- coding: utf-8 -*-
"""
KernelSHAP по словам (word-level) для вклада текста CLIP в cos(img, text).
Вертикальная узкая карточка без пустых мест:
- Фиксированная высота блока изображения (cover-кроп).
- Автовысота холста и/или уменьшение кегля caption, чтобы всё влезало.
- Подсветка ВАЖНОСТИ по словам исходного caption (никаких <id_…>).
- Метрики русским языком с расшифровкой.
- Один шрифт — DejaVu Sans Regular, чтобы совпадать с matplotlib.
Без matplotlib / SciPy — чистый PIL.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict, fields
from typing import Optional, Dict, Any, Union, List, Tuple
from pathlib import Path
import json, math
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import torch
import torch.nn.functional as F


# ----------------------------- Config ---------------------------------

@dataclass
class KernelSHAPConfig:
    # Узкий формат (ширина фикс), высота — авто (см. auto_expand_canvas)
    canvas_width_px: int = 720
    canvas_height_px: int = 1400           # старт; если auto_expand_canvas=True — пересчитаем
    margins_px: int = 16
    block_gap_px: int = 10

    # Блок изображения
    image_block_height_px: int = 360       # фиксированная высота блока с cover-кропом

    # SHAP
    max_tokens: int = 60
    num_samples: int = 240
    ridge_lambda: float = 1e-3
    batch_text: int = 32
    tau: float = 0.6

    # Типографика (единый шрифт DejaVu Sans Regular)
    fs_header: int = 22
    fs_text: int = 20
    fs_small: int = 16
    fs_text_min: int = 14                 # минимальный кегль, если длинный caption
    line_pad_px: int = 6

    # Шрифт — только DejaVu Sans Regular (как в matplotlib)
    font_path_regular: str = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

    # Управление компоновкой
    auto_shrink_text: bool = True          # при нехватке места уменьшаем кегль caption
    auto_expand_canvas: bool = True        # при нехватке места увеличиваем высоту холста
    canvas_height_max_px: int = 4000       # ограничитель «на всякий»

    show_reference: bool = False


def _coerce_cfg(config: Optional[Union[KernelSHAPConfig, Dict[str, Any]]]) -> KernelSHAPConfig:
    if isinstance(config, KernelSHAPConfig):
        return config
    d = dict(config or {})
    valid = {f.name for f in fields(KernelSHAPConfig)}
    d = {k: v for k, v in d.items() if k in valid}
    return KernelSHAPConfig(**d)


# ----------------------------- Utils ----------------------------------

def _find_font(path: str, size: int):
    try:
        return ImageFont.truetype(path, size=size)
    except Exception:
        # последний шанс — системный DejaVu Sans
        try:
            return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size=size)
        except Exception:
            return ImageFont.load_default()

def _text_size(drw: ImageDraw.ImageDraw, text: str, font) -> tuple[int, int]:
    try:
        b = drw.textbbox((0, 0), text, font=font)
        return b[2] - b[0], b[3] - b[1]
    except Exception:
        return drw.textsize(text, font=font)

def load_image(path: str, fallback_hw: int) -> Image.Image:
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return Image.new("RGB", (fallback_hw, fallback_hw), (245, 245, 245))

def _fit_cover(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    """Масштабируем под cover и центр-кропим: фиксированная высота блока без плясок по аспекту."""
    sw = target_w / img.width
    sh = target_h / img.height
    s = max(sw, sh)
    nw, nh = max(1, int(round(img.width * s))), max(1, int(round(img.height * s)))
    img2 = img.resize((nw, nh), Image.LANCZOS)
    ox = (nw - target_w) // 2
    oy = (nh - target_h) // 2
    return img2.crop((ox, oy, ox + target_w, oy + target_h))

def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = F.normalize(a, dim=-1); b = F.normalize(b, dim=-1)
    return (a * b).sum(dim=-1)

def normalize01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    mn, mx = float(x.min()), float(x.max())
    if mx <= mn + 1e-8:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn + 1e-8)


# ------------------------ Word-level SHAP core -------------------------

@torch.inference_mode()
def _batch_text_features(clip, texts: List[str], bs: int) -> torch.Tensor:
    outs = []
    for i in range(0, len(texts), bs):
        chunk = texts[i:i+bs]
        outs.append(clip.encode_text(chunk))
    return torch.cat(outs, dim=0) if outs else torch.zeros((0, getattr(clip, "text_dim", 512)), device=clip.device)

def _kernel_weight(n: int, k: int) -> float:
    if k == 0 or k == n:
        return 1000.0  # Увеличиваем вес для крайних случаев
    try:
        return (n - 1) / (math.comb(n, k) * k * (n - k))
    except (OverflowError, ZeroDivisionError):
        return 1e-6

def _split_words(caption: str) -> List[str]:
    return caption.split()

def compute_kernelshap(clip, image_path: str, caption: str, cfg: KernelSHAPConfig):
    pil_img = Image.open(image_path).convert("RGB")
    ifeat = clip.encode_image([pil_img])

    words_full = _split_words(caption)
    if not words_full:
        return [], [], {"note": "empty caption", "words_count": 0, "masks_sampled": 0, "base_sim": 0.0}

    if len(words_full) > cfg.max_tokens:
        mid = len(words_full) // 2
        half = cfg.max_tokens // 2
        words = words_full[max(0, mid - half): max(0, mid - half) + cfg.max_tokens]
    else:
        words = words_full[:]
    N = len(words)

    M = int(cfg.num_samples)
    rng = np.random.default_rng(42)
    
    # Улучшенное сэмплирование масок без дублирования
    masks_set = set()
    Z = []
    
    # Обязательные случаи
    empty_mask = tuple([0] * N)
    full_mask = tuple([1] * N)
    masks_set.add(empty_mask)
    masks_set.add(full_mask)
    Z.append(list(empty_mask))
    Z.append(list(full_mask))
    
    # Добавляем сингулярные маски (только один признак)
    for i in range(N):
        mask = [0] * N
        mask[i] = 1
        mask_tuple = tuple(mask)
        if mask_tuple not in masks_set and len(Z) < M:
            masks_set.add(mask_tuple)
            Z.append(mask)
    
    # Случайное сэмплирование для оставшихся масок
    attempts = 0
    max_attempts = M * 10  # Ограничиваем количество попыток
    
    while len(Z) < M and attempts < max_attempts:
        attempts += 1
        
        # Генерируем случайную маску
        if N <= 2:
            mask = rng.integers(0, 2, size=N).tolist()
        else:
            # Предпочитаем маски средней кардинальности
            if rng.random() < 0.7:  # 70% времени используем умное сэмплирование
                target_k = rng.choice([max(1, N//4), N//2, min(N-1, 3*N//4)])
                mask = [0] * N
                indices = rng.choice(N, size=min(target_k, N), replace=False)
                for idx in indices:
                    mask[idx] = 1
            else:  # 30% времени полностью случайно
                mask = rng.integers(0, 2, size=N).tolist()
        
        mask_tuple = tuple(mask)
        if mask_tuple not in masks_set:
            masks_set.add(mask_tuple)
            Z.append(mask)
    
    # Если не удалось набрать нужное количество уникальных масок, добавляем случайные
    while len(Z) < M:
        mask = rng.integers(0, 2, size=N).tolist()
        Z.append(mask)
    
    Z = np.array(Z[:M], dtype=np.int32)

    texts = []
    ks = []
    for z in Z:
        sel = [w for z_i, w in zip(z.tolist(), words) if z_i == 1]
        texts.append(" ".join(sel))
        ks.append(int(z.sum()))
    ks = np.asarray(ks, dtype=np.int32)

    tfeats = _batch_text_features(clip, texts, cfg.batch_text)
    sims = cosine_sim(ifeat.expand_as(tfeats), tfeats).detach().cpu().numpy()

    # Улучшенные веса и регуляризация
    w = np.array([_kernel_weight(N, int(k)) for k in ks], dtype=np.float64)
    
    # Нормализация весов
    w = w / (w.sum() + 1e-12)
    w = w * len(w)  # Восстанавливаем масштаб
    
    Zf = Z.astype(np.float64)
    sqrtw = np.sqrt(w + 1e-12)[:, None]
    A = sqrtw * Zf
    b = (sqrtw[:, 0] * sims)
    
    # Адаптивная регуляризация
    reg = max(float(cfg.ridge_lambda), 1e-6 * np.trace(A.T @ A) / N)
    AtA = A.T @ A + reg * np.eye(N, dtype=np.float64)
    Atb = A.T @ b
    
    try:
        phi = np.linalg.solve(AtA, Atb)
    except np.linalg.LinAlgError:
        # Fallback к псевдообращению
        phi = np.linalg.pinv(AtA) @ Atb

    s = phi.astype(np.float32)
    
    # Улучшенная постобработка
    # Проверяем знак - если большинство отрицательные, возможно нужно инвертировать
    s_pos = np.maximum(0.0, s)
    s_neg = np.maximum(0.0, -s)
    
    if s_neg.sum() > s_pos.sum() and s_neg.max() > s_pos.max():
        # Используем абсолютные значения, если отрицательные доминируют
        s_vis = normalize01(np.abs(s))
    else:
        if s_pos.max() <= 1e-8:
            s_pos = np.abs(s)
        s_vis = normalize01(s_pos)

    # косинус по ПОЛНОМУ caption (а не усечённому окну)
    base_sim = float(cosine_sim(ifeat, _batch_text_features(clip, [caption], 1)).item())

    meta = {
        "phi_raw": s.tolist(),
        "words_count": int(N),                 # N — число слов (признаков) в расчёте
        "masks_sampled": int(len(Z)),          # Фактическое количество масок
        "masks_unique": int(len(masks_set)),   # Количество уникальных масок
        "base_sim": base_sim,                  # cos(img, полный caption)
        "token_max": float(s_vis.max()) if s_vis.size else 0.0,
        "token_cov@tau": float((s_vis >= cfg.tau).mean()) if s_vis.size else 0.0,
        "reg_used": float(reg),                # Фактически использованная регуляризация
        "weight_range": [float(w.min()), float(w.max())],  # Диапазон весов
    }
    return words, s_vis.tolist(), meta


# --------------- Measuring & Rendering (no overflow, no gaps) ----------

def _wrap_words_height(drw, max_w: int, words: List[str], font) -> int:
    if not words:
        return 0
    line_h = (font.size if hasattr(font, "size") else 20) + 8
    xx, yy = 0, 0
    for word in words:
        piece = word + " "
        w, h = _text_size(drw, piece, font)
        if xx + w > max_w:
            yy += line_h
            xx = 0
        xx += w
    return yy + line_h

def _wrap_words_draw(drw, x: int, y: int, max_w: int, words: List[str], scores: List[float], font, gap: int) -> int:
    xx, yy = x, y
    line_h = (font.size if hasattr(font, "size") else 20) + 8
    for word, sc in zip(words, scores):
        piece = word + " "
        w, h = _text_size(drw, piece, font)
        if xx + w > x + max_w:
            yy += line_h
            xx = x
        c = float(max(0.0, min(1.0, sc)))
        if c > 0.05:
            v = int(255 * (1.0 - 0.8 * c))
            drw.rectangle([xx - 2, yy - 2, xx + w, yy + h + 2], fill=(v, v, 255), outline=None)
        drw.text((xx, yy), piece, font=font, fill=(0, 0, 0))
        xx += w
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
    cfg: KernelSHAPConfig,
):
    W = cfg.canvas_width_px
    M, GAP = cfg.margins_px, cfg.block_gap_px

    # Шрифты: один и тот же файл (DejaVu Sans Regular), разные кегли
    f_hdr = _find_font(cfg.font_path_regular, cfg.fs_header)
    f_txt = _find_font(cfg.font_path_regular, cfg.fs_text)
    f_sml = _find_font(cfg.font_path_regular, cfg.fs_small)

    # Черновой canvas для измерений
    tmp = Image.new("RGB", (W, 200), (255, 255, 255))
    mdrw = ImageDraw.Draw(tmp)
    col_w = W - 2 * M

    # ---- MEASURE ----
    def measure_total(fs_caption: int) -> int:
        nonlocal f_hdr, f_txt, f_sml
        f_hdr = _find_font(cfg.font_path_regular, cfg.fs_header)
        f_txt = _find_font(cfg.font_path_regular, fs_caption)
        f_sml = _find_font(cfg.font_path_regular, cfg.fs_small)
        total = 0
        total += _text_size(mdrw, "Original image", f_hdr)[1] + 6
        total += cfg.image_block_height_px + GAP
        total += _text_size(mdrw, "Generated caption (KernelSHAP by WORDS)", f_hdr)[1] + 6
        cap_h = _wrap_words_height(mdrw, col_w, words if words else gen_caption.split(), f_txt)
        total += cap_h + GAP
        if cfg.show_reference and ref_caption:
            total += _text_size(mdrw, "Reference caption", f_hdr)[1] + 6
            total += _wrap_words_height(mdrw, col_w, ref_caption.split(), f_txt) + GAP
        total += _text_size(mdrw, "Token importance (low → high)", f_sml)[1] + 4
        total += 14 + 26 + GAP
        metrics_items = [
            ("Косинус(изобр., полный текст)", f"{float(metrics.get('base_sim', 0.0)):.3f}"),
            ("Максимальная важность слова", f"{float(metrics.get('token_max', 0.0)):.3f}"),
            (f"Покрытие ≥ {cfg.tau:.2f}", f"{float(metrics.get('token_cov@tau', 0.0)):.3f}"),
            ("N — число слов (features)", str(int(metrics.get("words_count", 0)))),
            ("M — число уникальных масок", str(int(metrics.get("masks_unique", 0)))),
        ]
        total += _text_size(mdrw, "Metrics (explained)", f_hdr)[1] + 6
        total += _measure_kv_block_height(mdrw, metrics_items, f_sml) + GAP
        total += _text_size(mdrw, "Top words", f_hdr)[1] + 6
        top_k = min(12, len(words))
        bar_h = max(14, f_sml.size + 4)
        total += top_k * (bar_h + 6)
        footer = "KernelSHAP • CLIP text attribution (word-level) • cos(image, text)"
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
    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    drw = ImageDraw.Draw(canvas)

    # Пересоздаём финальные шрифты с учётом fs_caption
    f_hdr = _find_font(cfg.font_path_regular, cfg.fs_header)
    f_txt = _find_font(cfg.font_path_regular, fs_caption)
    f_sml = _find_font(cfg.font_path_regular, cfg.fs_small)

    x = M
    y = M
    col_w = W - 2 * M

    # 1) ОРИГИНАЛ
    drw.text((x, y), "Original image", font=f_hdr, fill=(0, 0, 0))
    y += _text_size(drw, "Original image", f_hdr)[1] + 6
    img_fit = _fit_cover(image, col_w, cfg.image_block_height_px)
    canvas.paste(img_fit, (x, y))
    y += cfg.image_block_height_px + GAP

    # 2) CAPTION
    cap_title = "Generated caption (KernelSHAP by WORDS)"
    drw.text((x, y), cap_title, font=f_hdr, fill=(0, 0, 0))
    y += _text_size(drw, cap_title, f_hdr)[1] + 6
    words_to_show = words if words else gen_caption.split()
    scores_to_show = scores if scores else [0.0] * len(words_to_show)
    y = _wrap_words_draw(drw, x, y, col_w, words_to_show, scores_to_show, f_txt, cfg.line_pad_px)
    y += GAP

    if cfg.show_reference and ref_caption:
        drw.text((x, y), "Reference caption", font=f_hdr, fill=(0, 0, 0))
        y += _text_size(drw, "Reference caption", f_hdr)[1] + 6
        y = _wrap_words_draw(drw, x, y, col_w, ref_caption.split(), [0.0]*len(ref_caption.split()), f_txt, cfg.line_pad_px)
        y += GAP

    # 3) ШКАЛА
    label = "Token importance (low → high)"
    drw.text((x, y), label, font=f_sml, fill=(0, 0, 0))
    y += _text_size(drw, label, f_sml)[1] + 4
    for i in range(col_w):
        t = i / max(1, col_w - 1)
        v = int(255 * (1 - 0.8 * t))
        drw.line([(x + i, y), (x + i, y + 14)], fill=(v, v, 255))
    drw.rectangle([x, y, x + col_w, y + 14], outline=(0, 0, 0), width=1)
    drw.text((x, y + 18), "Low", font=f_sml, fill=(0, 0, 0))
    drw.text((x + col_w - _text_size(drw, "High", f_sml)[0], y + 18), "High", font=f_sml, fill=(0, 0, 0))
    y += 14 + 26 + GAP

    # 4) МЕТРИКИ (РАСШИФРОВАНО)
    drw.text((x, y), "Metrics (explained)", font=f_hdr, fill=(0, 0, 0))
    y += _text_size(drw, "Metrics (explained)", f_hdr)[1] + 6

    def kv(k: str, v: str):
        nonlocal y
        drw.text((x, y), f"{k}: {v}", font=f_sml, fill=(0, 0, 0))
        y += _text_size(drw, f"{k}: {v}", f_sml)[1] + 4

    kv("Косинус(изобр., полный текст)", f"{float(metrics.get('base_sim', 0.0)):.3f}")
    kv("Максимальная важность слова", f"{float(metrics.get('token_max', 0.0)):.3f}")
    kv(f"Покрытие ≥ {cfg.tau:.2f}", f"{float(metrics.get('token_cov@tau', 0.0)):.3f}")
    kv("N — число слов (features)", str(int(metrics.get("words_count", 0))))
    kv("M — число уникальных масок", str(int(metrics.get("masks_unique", 0))))
    y += GAP

    # 5) TOP WORDS (с ЦИФРАМИ)
    drw.text((x, y), "Top words", font=f_hdr, fill=(0, 0, 0))
    y += _text_size(drw, "Top words", f_hdr)[1] + 6
    
    # Агрегируем одинаковые слова, берем максимальную важность
    word_scores = {}
    for w, s in zip(words_to_show, scores_to_show):
        word_lower = w.lower()  # Приводим к нижнему регистру для сравнения
        if word_lower in word_scores:
            word_scores[word_lower] = max(word_scores[word_lower], float(s))
        else:
            word_scores[word_lower] = float(s)
    
    # Сортируем по важности и берем топ
    pairs = [(word, score) for word, score in word_scores.items()]
    pairs.sort(key=lambda t: t[1], reverse=True)
    pairs = pairs[:12]
    bar_h = max(14, f_sml.size + 4)
    for name, sc in pairs:
        c = max(0.0, min(1.0, sc))
        nm = name[:28] + ("…" if len(name) > 28 else "")
        # текст слева
        tw, th = _text_size(drw, nm, f_sml)
        drw.text((x, y), nm, font=f_sml, fill=(0, 0, 0))
        # бар
        bx = x + tw + 8
        avail = col_w - (tw + 8)
        bw = max(0, int(avail * c))
        v = int(255 * (1 - 0.8 * c))
        drw.rectangle([bx, y, bx + bw, y + bar_h], fill=(v, v, 255), outline=(0, 0, 0))
        # значение (три знака после запятой)
        sval = f"{c:.3f}"
        svw, svh = _text_size(drw, sval, f_sml)
        # Если число не помещается справа — рисуем ВНУТРИ бара справа; иначе — справа от бара
        if bx + bw + 6 + svw <= x + col_w:
            drw.text((bx + bw + 6, y), sval, font=f_sml, fill=(0, 0, 0))
        else:
            # внутри бара, справа, с тёмным текстом (фон светлый)
            tx = max(bx + 2, x + col_w - svw)
            drw.text((tx, y), sval, font=f_sml, fill=(0, 0, 0))
        y += bar_h + 6

    # футер
    footer = "KernelSHAP • CLIP text attribution (word-level) • cos(image, text)"
    fw, fh = _text_size(drw, footer, f_sml)
    drw.text((W - fw - M, H - fh - 6), footer, font=f_sml, fill=(90, 90, 90))

    canvas.save(str(out_png), format="PNG")


# ----------------------------- Public API ------------------------------

def run(clip, captioner, *, image_path: str, out_dir: Path,
        ref_caption: Optional[str] = None,
        config: Optional[Union[KernelSHAPConfig, Dict[str, Any]]] = None) -> Dict[str, Any]:

    cfg = _coerce_cfg(config)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    image = load_image(image_path, fallback_hw=getattr(clip, "image_size", 224))
    try:
        gen_caption = captioner.generate([image], prompt=getattr(captioner, "prompt", None) or "Describe the image.")[0]
    except Exception:
        gen_caption = ""

    words, scores, meta = compute_kernelshap(clip, image_path, gen_caption, cfg)

    stem = Path(image_path).stem
    out_png = out_dir / f"{stem}.png"
    _render_card_vertical(out_png, image, gen_caption, ref_caption, words, scores, meta, cfg)

    out_json = out_dir / f"{stem}.json"
    data = {
        "image_path": image_path,
        "method": "kernelshap_wordlevel",
        "config": asdict(cfg),
        "caption_generated": gen_caption,
        "caption_reference": ref_caption if cfg.show_reference else None,
        "meta": meta,
        "outputs": {"png": str(out_png)}
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return data
