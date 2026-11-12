"""強制對齊子程序：把已知的歌詞逐行對到人聲音軌上的時間。

獨立程序執行，結束就釋放 VRAM，當掉也不會拖垮介面。用法：
    python -m app.services.align_worker <vocals.wav> <lyrics.txt> <輸出.json> [--model large-v3] [--language zh]

輸出 JSON：{"lines": [{"start": 秒, "end": 秒, "words": [{"word", "start", "end", "probability"}]}, ...]}
行數與 lyrics.txt 的非空行一一對應；完全對不上的行 start、end 為 null。
歌詞文字不會被修改，辨識結果只用來定位。
"""
from __future__ import annotations

import argparse
import json
import os
import sys


def align(audio: str, lines: list[str], model_name: str, language: str) -> list[dict]:
    import stable_whisper

    cache = os.path.join(os.getenv("XDG_CACHE_HOME", os.path.join(os.path.expanduser("~"), ".cache")), "whisper")
    if os.path.sep not in model_name and not os.path.exists(os.path.join(cache, f"{model_name}.pt")):
        print("PROGRESS 0.02 下載對齊模型（約 3 GB，只需下載一次，視網路速度可能要數十分鐘）", flush=True)
    else:
        print("PROGRESS 0.05 載入對齊模型", flush=True)
    model = stable_whisper.load_model(model_name)
    print("PROGRESS 0.35 對齊中", flush=True)
    # original_split=True：依換行切分，一行歌詞對應一個 segment
    result = model.align(audio, "\n".join(lines), language=language, original_split=True,
                         suppress_silence=True, verbose=None)
    segments = [s for s in result.segments if s.text.strip()]
    output = []
    for index, _ in enumerate(lines):
        if index >= len(segments):
            output.append({"start": None, "end": None, "words": []})
            continue
        segment = segments[index]
        words = [{"word": w.word, "start": round(float(w.start), 3), "end": round(float(w.end), 3),
                  "probability": round(float(w.probability), 4) if w.probability is not None else None}
                 for w in (segment.words or [])]
        output.append({"start": round(float(segment.start), 3), "end": round(float(segment.end), 3),
                       "words": words})
    return output


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio")
    parser.add_argument("lyrics")
    parser.add_argument("output")
    parser.add_argument("--model", default="large-v3")
    parser.add_argument("--language", default="zh")
    args = parser.parse_args(argv)
    with open(args.lyrics, encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    try:
        aligned = align(args.audio, lines, args.model, args.language)
    except Exception as e:      # 把原因寫到 stderr，主程式會轉成看得懂的訊息
        print(f"ALIGN_ERROR {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        return 1
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({"lines": aligned, "model": args.model, "language": args.language}, f, ensure_ascii=False, indent=1)
    print("PROGRESS 1.0 完成", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
