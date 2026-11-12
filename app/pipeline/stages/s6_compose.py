"""階段 6：字幕合成。用 FFmpeg 把 lyrics.srt 燒入 video_nosub.mp4，產出 output/final_mv.mp4。

Windows 路徑在字幕濾鏡裡需要跳脫（磁碟代號的冒號會被當成選項分隔），所以執行時把工作目錄
設為專案資料夾，濾鏡內只寫相對路徑；字型資料夾才用跳脫過的絕對路徑。
"""
from __future__ import annotations

import os
from pathlib import Path

from app.core.models import Problem, UserFacingError
from app.pipeline.stage_base import Reporter, StageContext, StageResult
from app.services import ffmpeg_service as F
from app.services import fonts
from app.storage.app_config import AppConfig

STYLE_KEYS = ("font", "size_percent", "margin_percent", "color", "outline_color", "outline_percent", "bold",
              "italic", "position", "background", "back_color", "back_opacity")

# libass 把 SRT 當成高度 288 的畫布來排版，樣式裡的數字都以它為單位
_ASS_HEIGHT = 288
# force_style 的 Alignment 用的是舊版 SSA 的編號（實測）：2 置底置中、6 置頂置中、10 畫面正中
_ALIGNMENT = {"bottom": 2, "top": 6, "middle": 10}
# 陰影的位移、底框的留白：佔畫面高度的百分比
_SHADOW_PERCENT = 0.5
_BOX_PADDING_PERCENT = 1.0


def font_dirs() -> list[Path]:
    dirs = [Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"]
    local = os.environ.get("LOCALAPPDATA")
    if local:
        dirs.append(Path(local) / "Microsoft" / "Windows" / "Fonts")     # 使用者自己安裝的字型
    return [d for d in dirs if d.is_dir()]


def find_font_dir(config: AppConfig, font: str, dirs: list[Path] | None = None) -> Path | None:
    """字型有安裝就回傳它所在的資料夾；不認得或沒安裝回傳 None。"""
    pattern = config.get("subtitle.font_files", {}).get(font)
    if not pattern:
        return None
    for folder in (dirs if dirs is not None else font_dirs()):
        if any(folder.glob(pattern)):
            return folder
    return None


def subtitle_style(config: AppConfig, project) -> dict:
    """字幕樣式：設定檔的預設值，被專案自己的設定覆蓋。"""
    defaults = {key: config.get(f"subtitle.{key}") for key in STYLE_KEYS}
    return {**defaults, **project.data.get("params", {}).get("subtitle_style", {})}


def resolve_font(config: AppConfig, font: str) -> tuple[bool, Path | None]:
    """（字型能不能用, 要告訴 FFmpeg 的字型資料夾）。

    設定檔列出檔名樣式的字型，直接指到它所在的資料夾；其他字型只要系統查得到就用名稱指定，
    另外附上使用者自己安裝字型的資料夾（FFmpeg 不一定會自己去找那裡）。
    """
    folder = find_font_dir(config, font)
    if folder is not None:
        return True, folder
    if font in config.get("subtitle.font_files", {}) or not fonts.is_installed(font):
        return False, None
    user_dirs = font_dirs()[1:]
    return True, user_dirs[0] if user_dirs else None


def ass_color(value: str, opacity: float = 100.0) -> str:
    """#RRGGBB → ASS 的 &HAABBGGRR（AA 是透明度：00 不透明、FF 全透明）。"""
    text = value.strip().lstrip("#")
    if len(text) != 6 or any(c not in "0123456789abcdefABCDEF" for c in text):
        raise UserFacingError(f"字幕顏色「{value}」不是 #RRGGBB 格式", "請在字幕樣式重新選擇顏色")
    alpha = round(255 * (1 - min(max(float(opacity), 0.0), 100.0) / 100))
    return f"&H{alpha:02X}{text[4:6]}{text[2:4]}{text[0:2]}".upper()


def escape_filter_path(path: Path) -> str:
    """濾鏡參數中的絕對路徑：反斜線改正斜線，冒號加跳脫。"""
    return str(path).replace("\\", "/").replace(":", "\\:")


def build_filter(style: dict, font: str, font_dir: Path | None, srt_relative: str) -> str:
    # 文字背後：none 只有描邊、shadow 加陰影、box 加半透明底框（BorderStyle=4，底框與陰影都用 BackColour）
    background = style.get("background") or "none"
    back = ass_color(style.get("back_color") or "#000000", style.get("back_opacity", 60))
    if background == "box":
        border, shadow = 4, _BOX_PADDING_PERCENT / 100 * _ASS_HEIGHT
    else:
        border, shadow = 1, (_SHADOW_PERCENT / 100 * _ASS_HEIGHT if background == "shadow" else 0)
    position = style.get("position") or "bottom"
    force = ",".join([
        f"FontName={font}",
        f"FontSize={style['size_percent'] / 100 * _ASS_HEIGHT:.1f}",
        f"PrimaryColour={ass_color(style['color'])}",
        f"OutlineColour={ass_color(style['outline_color'])}",
        f"BackColour={back}",
        f"BorderStyle={border}",
        f"Outline={style['outline_percent'] / 100 * _ASS_HEIGHT:.2f}",
        f"Shadow={shadow:.2f}",
        f"Bold={-1 if style.get('bold') else 0}",
        f"Italic={-1 if style.get('italic') else 0}",
        f"Alignment={_ALIGNMENT.get(position, 2)}",
        # 邊距是字幕到畫面上緣或下緣的距離；放在正中時用不到
        f"MarginV={0 if position == 'middle' else style['margin_percent'] / 100 * _ASS_HEIGHT:.0f}",
    ])
    parts = [f"subtitles={srt_relative}"]
    if font_dir is not None:
        parts.append(f"fontsdir='{escape_filter_path(font_dir)}'")
    parts.append(f"force_style='{force}'")
    return ":".join(parts)


def choose_font(config: AppConfig, style: dict) -> tuple[str, Path | None, list[str]]:
    """決定實際使用的字型：指定的字型不能用就改用備用字型，回傳（字型, 字型資料夾, 警告）。"""
    font = style["font"]
    usable, font_dir = resolve_font(config, font)
    if usable:
        return font, font_dir, []
    fallback = config.get("subtitle.fallback_font", "Microsoft JhengHei")
    usable, fallback_dir = resolve_font(config, fallback)
    if not usable:
        raise UserFacingError(f"找不到字幕字型「{font}」，備用的「{fallback}」也沒有安裝",
                              "請安裝 Noto Sans TC 或微軟正黑體，或在字幕樣式改用已安裝的中文字型")
    return fallback, fallback_dir, [f"找不到字幕字型「{font}」，改用「{fallback}」"]


PREVIEW_SIZE = (896, 512)


def render_preview(config: AppConfig, project, style: dict, text: str, work_dir: Path) -> Path:
    """把一行範例字幕燒在一張畫面上，回傳圖檔路徑；字幕樣式視窗用它即時預覽。

    畫面依序取自：已生成的對嘴影片、處理過的照片、原始照片；都沒有就用灰色底。
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    font, font_dir, _ = choose_font(config, style)
    (work_dir / "preview.srt").write_text(f"1\n00:00:00,000 --> 00:00:30,000\n{text}\n", encoding="utf-8")
    width, height = PREVIEW_SIZE
    video = project.path("output", "video_nosub.mp4")
    photos = [project.path("input", "photo_prepared.png")]
    if project.data.get("inputs", {}).get("photo"):
        photos.append(project.path(project.data["inputs"]["photo"]))
    photo = next((p for p in photos if p.exists()), None)
    if video.exists():
        source = ["-ss", f"{F.duration_of(video) / 2:.2f}", "-i", str(video)]
    elif photo is not None:
        source = ["-i", str(photo)]
    else:
        source = ["-f", "lavfi", "-i", f"color=c=0x606060:s={width}x{height}"]
    fit = f"scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height}"
    F._run(["ffmpeg", "-y", "-v", "error", *source, "-vf",
            f"{fit},{build_filter(style, font, font_dir, 'preview.srt')}", "-frames:v", "1", "preview.png"],
           "產生字幕預覽", cwd=work_dir, timeout=30)
    return work_dir / "preview.png"


class ComposeStage:
    id = "s6_compose"

    def check_inputs(self, ctx: StageContext) -> list[Problem]:
        problems = []
        if not ctx.project.path("output", "video_nosub.mp4").exists():
            problems.append(Problem("還沒有對嘴影片", "請先完成對嘴影片階段"))
        if not ctx.project.path("lyrics", "lyrics.srt").exists():
            problems.append(Problem("還沒有字幕檔", "請先完成字幕對齊階段"))
        return problems

    def outputs(self, ctx: StageContext) -> list[Path]:
        return [ctx.project.path("output", "final_mv.mp4")]

    def run(self, ctx: StageContext, reporter: Reporter) -> StageResult:
        project, config = ctx.project, ctx.config
        style = subtitle_style(config, project)
        font, font_dir, warnings = choose_font(config, style)

        source = project.path("output", "video_nosub.mp4")
        output = project.path("output", "final_mv.mp4")
        partial = project.path("output", "final_mv.partial.mp4")
        reporter.progress(0.05, "燒入字幕")
        reporter.log(f"燒入字幕：字型 {font}，字級為畫面高度的 {style['size_percent']}%")
        video_filter = build_filter(style, font, font_dir, "lyrics/lyrics.srt")
        F._run(["ffmpeg", "-y", "-v", "error", "-i", "output/video_nosub.mp4", "-vf", video_filter,
                "-c:v", "libx264", "-preset", "medium", "-crf", str(config.get("subtitle.crf", 18)),
                "-pix_fmt", "yuv420p", "-c:a", "copy", "-movflags", "+faststart", f"output/{partial.name}"],
               "燒入字幕", cwd=project.folder)
        before, after = F.probe_video(source), F.probe_video(partial)
        if before["frames"] != after["frames"]:
            partial.unlink(missing_ok=True)
            raise UserFacingError(f"燒入字幕後影格數從 {before['frames']} 變成 {after['frames']}", "請再試一次")
        os.replace(partial, output)
        project.stage(self.id)["style"] = {**style, "font_used": font}
        reporter.progress(1.0, "完成")
        return StageResult(True, warnings)

    resume = run
