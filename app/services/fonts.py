"""列出這台電腦已安裝的字型（Windows GDI），不依賴介面層，燒字幕階段與字幕樣式視窗共用。"""
from __future__ import annotations

import ctypes
import re
import subprocess
import sys
import tempfile
from ctypes import wintypes
from pathlib import Path

DEFAULT_CHARSET = 1
CHINESEBIG5_CHARSET = 136      # 繁體中文
_TRUETYPE = 4                  # 列舉回呼的字型種類：TrueType／OpenType
_MARK_MISSING = 1              # GGI_MARK_NONEXISTING_GLYPHS：沒有的字回傳 0xFFFF


class _LOGFONTW(ctypes.Structure):
    _fields_ = [("lfHeight", wintypes.LONG), ("lfWidth", wintypes.LONG), ("lfEscapement", wintypes.LONG),
                ("lfOrientation", wintypes.LONG), ("lfWeight", wintypes.LONG), ("lfItalic", wintypes.BYTE),
                ("lfUnderline", wintypes.BYTE), ("lfStrikeOut", wintypes.BYTE), ("lfCharSet", wintypes.BYTE),
                ("lfOutPrecision", wintypes.BYTE), ("lfClipPrecision", wintypes.BYTE),
                ("lfQuality", wintypes.BYTE), ("lfPitchAndFamily", wintypes.BYTE),
                ("lfFaceName", wintypes.WCHAR * 32)]


def _libraries():
    """user32 與 gdi32，並宣告用到的函式型別：控制代碼是 64 位元，不宣告的話會被當成 32 位元整數而出錯。"""
    user32, gdi32 = ctypes.windll.user32, ctypes.windll.gdi32
    user32.GetDC.argtypes = [wintypes.HWND]
    user32.GetDC.restype = wintypes.HDC
    user32.ReleaseDC.argtypes = [wintypes.HWND, wintypes.HDC]
    gdi32.CreateFontW.argtypes = [ctypes.c_int] * 5 + [wintypes.DWORD] * 8 + [wintypes.LPCWSTR]
    gdi32.CreateFontW.restype = wintypes.HFONT
    gdi32.SelectObject.argtypes = [wintypes.HDC, wintypes.HGDIOBJ]
    gdi32.SelectObject.restype = wintypes.HGDIOBJ
    gdi32.DeleteObject.argtypes = [wintypes.HGDIOBJ]
    gdi32.GetGlyphIndicesW.argtypes = [wintypes.HDC, wintypes.LPCWSTR, ctypes.c_int,
                                       ctypes.POINTER(wintypes.WORD), wintypes.DWORD]
    return user32, gdi32


def _enumerate(face: str, charset: int) -> list[str]:
    if sys.platform != "win32":
        return []
    names: list[str] = []
    callback_type = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.POINTER(_LOGFONTW), ctypes.c_void_p,
                                       wintypes.DWORD, wintypes.LPARAM)

    def collect(logfont, _metric, kind, _param) -> int:
        if kind & _TRUETYPE:                       # 點陣字型不能用來燒字幕
            names.append(logfont.contents.lfFaceName)
        return 1

    user32, gdi32 = _libraries()
    gdi32.EnumFontFamiliesExW.argtypes = [wintypes.HDC, ctypes.POINTER(_LOGFONTW), callback_type,
                                          wintypes.LPARAM, wintypes.DWORD]
    query = _LOGFONTW()
    query.lfCharSet = charset
    query.lfFaceName = face[:31]
    dc = user32.GetDC(None)
    try:
        gdi32.EnumFontFamiliesExW(dc, ctypes.byref(query), callback_type(collect), 0, 0)
    finally:
        user32.ReleaseDC(None, dc)
    return names


def _has_glyphs(dc, name: str, text: str) -> bool:
    """字型檔裡是不是真的有這些字（系統會替沒有中文的字型借字，所以不能只看它宣稱支援的字集）。"""
    _, gdi32 = _libraries()
    font = gdi32.CreateFontW(0, 0, 0, 0, 0, 0, 0, 0, DEFAULT_CHARSET, 0, 0, 0, 0, name)
    old = gdi32.SelectObject(dc, font)
    try:
        indices = (wintypes.WORD * len(text))()
        count = gdi32.GetGlyphIndicesW(dc, text, len(text), indices, _MARK_MISSING)
        return count == len(text) and all(i != 0xFFFF for i in indices)
    finally:
        gdi32.SelectObject(dc, old)
        gdi32.DeleteObject(font)


def chinese_fonts(sample: str = "風歌體驗") -> list[str]:
    """字型檔內含繁體中文字的已安裝字型名稱（不含直書用的 @ 字型），依名稱排序。"""
    names = sorted({name for name in _enumerate("", CHINESEBIG5_CHARSET) if not name.startswith("@")})
    if not names:
        return []
    user32, _ = _libraries()
    dc = user32.GetDC(None)
    try:
        return [name for name in names if _has_glyphs(dc, name, sample)]
    finally:
        user32.ReleaseDC(None, dc)


_usable_cache: dict[tuple, list[str]] = {}


def usable_for_subtitles(names: list[str], fonts_dir: Path | None = None) -> list[str]:
    """從 names 挑出 FFmpeg 燒字幕時真的找得到的字型。

    系統列出的名稱有些 FFmpeg（libass）不認得，例如同一套字型的各種粗細（Noto Sans TC Light），
    指定了只會悄悄換成別的字型。這裡把所有名稱寫進一個字幕檔讓 FFmpeg 實際選一次，看它選到什麼。
    檢查失敗時原樣回傳，不擋使用者。
    """
    key = (tuple(names), str(fonts_dir))
    if key in _usable_cache:
        return _usable_cache[key]
    result = list(names)
    try:
        with tempfile.TemporaryDirectory(prefix="mv_fonts_") as folder:
            styles = "".join(f"Style: s{i},{name},20,&H00FFFFFF,&H00000000,&H00000000,0,0,1,1,0,2,0,0,0,1\n"
                             for i, name in enumerate(names))
            events = "".join(f"Dialogue: 0,0:00:00.00,0:00:05.00,s{i},風歌\n" for i in range(len(names)))
            Path(folder, "fonts.ass").write_text(
                "[Script Info]\nScriptType: v4.00+\nPlayResX: 384\nPlayResY: 288\n\n[V4+ Styles]\n"
                "Format: Name, Fontname, Fontsize, PrimaryColour, OutlineColour, BackColour, Bold, Italic, "
                "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n" + styles
                + "\n[Events]\nFormat: Layer, Start, End, Style, Text\n" + events, encoding="utf-8")
            video_filter = "subtitles=fonts.ass"
            if fonts_dir is not None:
                video_filter += ":fontsdir='" + str(fonts_dir).replace("\\", "/").replace(":", "\\:") + "'"
            done = subprocess.run(
                ["ffmpeg", "-v", "verbose", "-f", "lavfi", "-i", "color=s=384x288:d=1", "-vf", video_filter,
                 "-frames:v", "1", "-f", "null", "-"], cwd=folder, capture_output=True, timeout=60,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            chosen: dict[str, str] = {}
            for line in done.stderr.decode("utf-8", errors="replace").splitlines():
                match = re.search(r"fontselect: \((.+), \d+, \d+\) -> (.+?), \d+, ", line)
                if match:
                    chosen.setdefault(match.group(1), match.group(2))     # 第一筆才是依名稱找的結果
            if chosen:
                # 找不到指定的字型時 libass 會先退回 Arial
                result = [n for n in names if n in chosen
                          and (not chosen[n].lower().startswith("arial") or n.lower().startswith("arial"))]
    except (OSError, subprocess.SubprocessError):
        pass
    _usable_cache[key] = result
    return result


def is_installed(name: str) -> bool:
    """字型有沒有安裝；中文名稱或英文名稱都認得。"""
    return bool(name) and bool(_enumerate(name, DEFAULT_CHARSET))
