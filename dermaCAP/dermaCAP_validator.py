from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

import pandas as pd
import cv2
from tqdm import tqdm

CSV_PATH      = Path("/root/clip_xai/dermaCAP/dermaCAP_v1.csv")
IMG_ROOT      = Path("/root/clip_xai/dermaCAP/dermaCAP_img")
EXPECTED_ROWS = 22483
OUTPUT_CSV    = CSV_PATH.with_name(CSV_PATH.stem + "_clean.csv")

MAX_WORKERS = min(32, (os.cpu_count() or 8) * 2)
FAST_READ_FLAG = getattr(cv2, "IMREAD_REDUCED_GRAYSCALE_2", cv2.IMREAD_GRAYSCALE)


def index_img_root(root: Path) -> dict[str, Path]:
    name2path: dict[str, Path] = {}
    for p in root.rglob("*"):
        if p.is_file():
            name = p.name
            if name not in name2path:
                name2path[name] = p
    return name2path


def quick_read_ok(path: Path) -> bool:
    img = cv2.imread(str(path), FAST_READ_FLAG)
    if img is not None and img.size > 0:
        return True
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    return img is not None and img.size > 0


def main() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV не найден: {CSV_PATH}")
    if not IMG_ROOT.exists():
        raise FileNotFoundError(f"Каталог с изображениями не найден: {IMG_ROOT}")

    df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")

    required_cols = {"img_path", "caption", "source", "img_name"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Отсутствуют обязательные колонки: {', '.join(sorted(missing))}")

    n_rows = len(df)
    if n_rows != EXPECTED_ROWS:
        print(f"[WARN] ожидалось {EXPECTED_ROWS} строк, в CSV: {n_rows}")

    name2path = index_img_root(IMG_ROOT)

    img_names = df["img_name"].astype(str).str.strip()
    mapped_paths = img_names.map(lambda n: name2path.get(n, None))

    mask_exists = mapped_paths.notna()
    missing_files = int((~mask_exists).sum())

    to_check_indices = df.index[mask_exists].tolist()
    to_check_paths = mapped_paths[mask_exists].tolist()

    ok_mask = pd.Series(False, index=df.index)
    corrupted = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(quick_read_ok, p): i for i, p in zip(to_check_indices, to_check_paths)}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Проверка", dynamic_ncols=True):
            i = futures[fut]
            ok = False
            try:
                ok = fut.result()
            except Exception:
                ok = False
            if ok:
                ok_mask.at[i] = True
            else:
                corrupted += 1

    keep_mask = mask_exists & ok_mask
    df_clean = df.loc[keep_mask].copy()
    df_clean.loc[:, "img_path"] = mapped_paths[keep_mask].astype(str).values

    df_clean.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

    print("-" * 50)
    print(f"Исходных строк:        {n_rows}")
    print(f"Удалено (нет файла):   {missing_files}")
    print(f"Удалено (битые):       {corrupted}")
    print(f"Итого оставлено:       {len(df_clean)}")
    print(f"Сохранено в:           {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
