"""照片前處理：縮放並裁切成目標尺寸，另存新檔，不改動原圖。"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

from app.core.models import UserFacingError


def prepare_photo(source: Path, dest: Path, width: int, height: int) -> list[str]:
    """裁成 width×height 存成 PNG，回傳警告清單。

    直式照片裁成橫式時，裁切框偏上，保留人物的頭與麥克風。
    """
    try:
        image = ImageOps.exif_transpose(Image.open(source)).convert("RGB")
    except FileNotFoundError:
        raise UserFacingError("找不到人像照片", "請到輸入頁重新選擇照片")
    except (UnidentifiedImageError, OSError) as e:
        raise UserFacingError(f"無法讀取照片：{e}", "請改用 PNG 或 JPG 格式的照片")
    warnings = []
    if image.width < width or image.height < height:
        warnings.append(f"照片只有 {image.width}×{image.height}，小於影片的 {width}×{height}，畫面會比較模糊")
    source_ratio, target_ratio = image.width / image.height, width / height
    if abs(source_ratio - target_ratio) / target_ratio > 0.02:
        kept = min(source_ratio, target_ratio) / max(source_ratio, target_ratio)
        if kept < 0.6:
            warnings.append(f"照片長寬比與影片差很多，裁切後只保留約 {kept:.0%} 的畫面；"
                            "建議改用接近 16:9 的橫式照片")
    vertical = 0.3 if source_ratio < target_ratio else 0.5
    ImageOps.fit(image, (width, height), Image.Resampling.LANCZOS, centering=(0.5, vertical)).save(dest, "PNG")
    return warnings
