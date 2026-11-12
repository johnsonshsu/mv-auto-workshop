"""階段 1：硬體與環境偵測。

判定分三級：通過、可執行但需降規、不通過。不通過時不允許開始製作。
偵測（收集事實）與判定（evaluate）分開，判定是純函式，方便測試。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from app.core.models import DEGRADE, FAIL, PASS, CheckItem, Problem
from app.pipeline.stage_base import Reporter, StageContext, StageResult
from app.services import hardware_probe
from app.services.comfy_client import ComfyClient
from app.services.hardware_probe import GpuInfo
from app.services.workflow_template import load_requires
from app.storage.app_config import AppConfig

_ORDER = {PASS: 0, DEGRADE: 1, FAIL: 2}


def worst(*levels: str) -> str:
    return max(levels, key=_ORDER.__getitem__) if levels else PASS


def parse_version(text: str) -> tuple[int, ...]:
    """把 "v0.44.2"、"0.36.0-rc1" 轉成可比較的整數組。"""
    return tuple(int(n) for n in re.findall(r"\d+", text.split("-")[0].split("+")[0])[:3])


def combo_options(object_info: dict, node: str, input_name: str) -> list[str]:
    """取出節點下拉選單的選項；新舊兩種 /object_info 格式都支援。"""
    inputs = object_info.get(node, {}).get("input", {})
    spec = inputs.get("required", {}).get(input_name) or inputs.get("optional", {}).get(input_name)
    if not spec:
        return []
    if isinstance(spec[0], list):                       # 舊格式：[[選項…], {…}]
        return [str(o) for o in spec[0]]
    if len(spec) > 1 and isinstance(spec[1], dict):     # 新格式：["COMBO", {"options": […]}]
        return [str(o) for o in spec[1].get("options", [])]
    return []


@dataclass
class EnvFacts:
    """偵測到的事實。"""
    gpus: list[GpuInfo] = field(default_factory=list)
    ram_gb: float = 0.0
    projects_root: str = ""
    disk_free_gb: float | None = None
    ffmpeg_version: str | None = None
    comfy_url: str | None = None
    comfy_version: str = ""
    object_info: dict | None = None
    # ComfyUI 的啟動參數；None 表示查不到（不檢查）
    comfy_argv: list[str] | None = None


def collect_facts(config: AppConfig, reporter: Reporter | None = None) -> EnvFacts:
    reporter = reporter or Reporter()
    facts = EnvFacts()
    reporter.progress(0.05, "偵測顯示卡與記憶體")
    facts.gpus = hardware_probe.probe_gpus()
    facts.ram_gb = hardware_probe.probe_ram_gb()
    facts.projects_root = str(config.get("paths.projects_root"))
    facts.disk_free_gb = hardware_probe.probe_disk_free_gb(Path(facts.projects_root))
    facts.ffmpeg_version = hardware_probe.probe_ffmpeg()

    reporter.progress(0.3, "尋找 ComfyUI")
    facts.comfy_url = ComfyClient.detect(
        config.get("comfyui.host"), config.get("comfyui.ports"),
        config.get("comfyui.last_url", ""), config.get("comfyui.connect_timeout", 3.0))
    if facts.comfy_url:
        client = ComfyClient(facts.comfy_url)
        try:
            system = client.system_stats().get("system", {})
            facts.comfy_version = system.get("comfyui_version", "")
            facts.comfy_argv = system.get("argv")
            reporter.progress(0.5, "讀取 ComfyUI 節點清單")
            facts.object_info = client.object_info()
        except Exception as e:      # 連得上但查詢失敗：當作未連線，detail 會說明
            reporter.log(f"查詢 ComfyUI 失敗：{e}")
            facts.object_info = None
        finally:
            client.close()
        if config.get("comfyui.last_url") != facts.comfy_url:
            config.set("comfyui.last_url", facts.comfy_url)
    reporter.progress(0.9, "判定結果")
    return facts


# ---- 判定 ----

def check_requires_option(req: dict, object_info: dict) -> tuple[bool, str, str]:
    """檢查一組需求，回傳（是否齊全、缺少的說明、補救方式）。"""
    missing_nodes = [n for n in req.get("nodes", []) if n not in object_info]
    missing_models = []
    for model in req.get("models", []):
        options = combo_options(object_info, model["loader_node"], model["loader_input"])
        if "pattern" in model:
            found = any(re.search(model["pattern"], o) for o in options)
        else:
            found = model["file"] in options
        if not found:
            missing_models.append(model)
    if not missing_nodes and not missing_models:
        return True, "", ""
    detail, fix = [], []
    if missing_nodes:
        detail.append("缺少節點：" + "、".join(missing_nodes))
        packs = req.get("custom_node_packs") or []
        fix.append("請安裝自訂節點套件：" + "、".join(packs) if packs else "請更新 ComfyUI 到最新版")
    for model in missing_models:
        detail.append(f"缺少模型：{model['file']}")
        fix.append(f"把 {model['file']} 放到 ComfyUI 的 {model['folder']}/，下載：{model['url']}")
    return False, "；".join(detail), "\n".join(fix)


def resolve_models(req: dict, object_info: dict) -> dict[str, str]:
    """需求清單中帶有 param 的模型 → 要注入工作流的檔名。

    指定的檔名不在本機時，改用符合 pattern 的第一個（例如只有 fp32 版、或檔名帶 _fp16）。
    """
    params = {}
    for model in req.get("models", []):
        if "param" not in model:
            continue
        options = combo_options(object_info, model["loader_node"], model["loader_input"])
        params[model["param"]] = model["file"] if model["file"] in options else next(
            (o for o in options if re.search(model.get("pattern", ""), o)), model["file"])
    return params


def check_argv(req: dict, argv: list[str] | None) -> tuple[str, str]:
    """檢查 ComfyUI 是否帶有工作流需要的啟動參數，回傳（說明、補救方式）；沒問題回傳空字串。"""
    if argv is None:
        return "", ""
    for item in req.get("requires_argv", []):
        if item["flag"] not in argv:
            return f"ComfyUI 啟動時沒有加 {item['flag']}：{item['reason']}", item["fix"]
    return "", ""


def check_requires(req: dict, object_info: dict) -> tuple[str, str, str]:
    """回傳（等級、說明、補救方式）。any_of：第一組齊全為通過，只有後面的組齊全為降規。"""
    if "any_of" not in req:
        ok, detail, fix = check_requires_option(req, object_info)
        return (PASS, "節點與模型齊全", "") if ok else (FAIL, detail, fix)
    first_detail = first_fix = ""
    for index, option in enumerate(req["any_of"]):
        ok, detail, fix = check_requires_option(option, object_info)
        if ok and index == 0:
            return PASS, f"使用 {option.get('label', '')}，節點與模型齊全", ""
        if ok:
            return (DEGRADE, f"{req['any_of'][0].get('label', '首選')}不可用（{first_detail}），"
                             f"改用 {option.get('label', '')}", first_fix)
        if index == 0:
            first_detail, first_fix = detail, fix
    return FAIL, first_detail, first_fix


def check_threshold(th: dict, vram_gb: float, ram_gb: float) -> tuple[str, str]:
    """依門檻表判定硬體等級，回傳（等級、說明）。"""
    # 標稱 24 GB 的顯示卡實際回報約 23.9 GiB，四捨五入後再比較
    vram_gb, ram_gb = round(vram_gb), round(ram_gb)
    if vram_gb < th["min_vram"]:
        return FAIL, f"VRAM {vram_gb:.0f} GB 低於最低需求 {th['min_vram']} GB"
    if ram_gb < th["min_ram"]:
        return FAIL, f"系統記憶體 {ram_gb:.0f} GB 低於最低需求 {th['min_ram']} GB"
    short = []
    if vram_gb < th["recommended_vram"]:
        short.append(f"VRAM {vram_gb:.0f} GB 低於建議的 {th['recommended_vram']} GB")
    if ram_gb < th["recommended_ram"]:
        short.append(f"系統記憶體 {ram_gb:.0f} GB 低於建議的 {th['recommended_ram']} GB")
    if short:
        return DEGRADE, "；".join(short) + "。" + th.get("degrade_hint", "")
    return PASS, ""


def _workflow_item(item_id: str, key: str, workflow: str, facts: EnvFacts, config: AppConfig) -> CheckItem:
    """一個工作流的檢查 = 需求清單（節點、模型）＋ 硬體門檻。"""
    req = load_requires(config.workflows_dir, workflow)
    label = req.get("label", workflow)
    if facts.object_info is None:
        return CheckItem(item_id, label, FAIL, "尚未連線到 ComfyUI，無法檢查", "請先啟動 ComfyUI 再重新檢查")
    level, detail, fix = check_requires(req, facts.object_info)
    argv_detail, argv_fix = check_argv(req, facts.comfy_argv)
    if argv_detail and level != FAIL:
        level, detail, fix = FAIL, argv_detail, argv_fix
    th = config.thresholds.get(key)
    if th and level != FAIL and facts.gpus:
        vram = max(g.vram_gb for g in facts.gpus)
        th_level, th_detail = check_threshold(th, vram, facts.ram_gb)
        if th_level != PASS:
            detail = th_detail if level == PASS else f"{detail}；{th_detail}"
            if th_level == FAIL:
                fix = "這台電腦的硬體不足以執行這個模型"
            level = worst(level, th_level)
    return CheckItem(item_id, label, level, detail, fix)


def evaluate(facts: EnvFacts, config: AppConfig) -> dict:
    items: list[CheckItem] = []

    if facts.gpus:
        gpu = max(facts.gpus, key=lambda g: g.vram_gb)
        items.append(CheckItem("gpu", "顯示卡", PASS,
                               f"{gpu.name} · {gpu.vram_gb:.0f} GB VRAM · 驅動 {gpu.driver}"))
    else:
        items.append(CheckItem("gpu", "顯示卡", FAIL, "找不到 NVIDIA 顯示卡或 nvidia-smi",
                               "本 APP 需要 NVIDIA 顯示卡；請安裝或更新 NVIDIA 驅動程式"))

    items.append(CheckItem("ram", "系統記憶體", PASS, f"{facts.ram_gb:.0f} GB"))

    min_free = config.get("disk.min_free_gb", 20)
    if facts.disk_free_gb is None:
        items.append(CheckItem("disk", "磁碟空間", FAIL, f"專案資料夾所在的磁碟不存在：{facts.projects_root}",
                               "請在設定頁把專案資料夾改到存在的磁碟"))
    elif facts.disk_free_gb < min_free:
        items.append(CheckItem("disk", "磁碟空間", FAIL,
                               f"{facts.projects_root} 可用 {facts.disk_free_gb:.0f} GB，低於需求 {min_free} GB",
                               "請清出空間，或在設定頁把專案資料夾改到其他磁碟"))
    else:
        items.append(CheckItem("disk", "磁碟空間", PASS,
                               f"{facts.projects_root} 可用 {facts.disk_free_gb:.0f} GB"))

    if facts.ffmpeg_version:
        items.append(CheckItem("ffmpeg", "FFmpeg", PASS, facts.ffmpeg_version))
    else:
        items.append(CheckItem("ffmpeg", "FFmpeg", FAIL, "找不到 ffmpeg 指令",
                               "請安裝 FFmpeg 並加入 PATH：https://www.gyan.dev/ffmpeg/builds/"))

    min_version = config.get("comfyui.min_version")
    if not facts.comfy_url:
        ports = "、".join(str(p) for p in config.get("comfyui.ports"))
        items.append(CheckItem("comfyui", "ComfyUI", FAIL, f"連接埠 {ports} 都沒有回應",
                               "請啟動 ComfyUI Desktop；若使用其他連接埠，請在設定頁指定位址"))
    elif facts.object_info is None:
        items.append(CheckItem("comfyui", "ComfyUI", FAIL, f"{facts.comfy_url} 有回應但查詢節點清單失敗",
                               "請確認 ComfyUI 已完成啟動後再重新檢查"))
    elif parse_version(facts.comfy_version) < parse_version(min_version):
        items.append(CheckItem("comfyui", "ComfyUI", FAIL,
                               f"{facts.comfy_url} · v{facts.comfy_version}，低於需求 v{min_version}",
                               "請更新 ComfyUI（YuE2 需要 v0.36.0 以上）"))
    else:
        items.append(CheckItem("comfyui", "ComfyUI", PASS, f"{facts.comfy_url} · v{facts.comfy_version}"))

    items.append(_workflow_item("wf_yue2", "yue2", config.get("yue2.workflow"), facts, config))
    items.append(_workflow_item("wf_vocal_separation", "vocal_separation",
                                config.get("vocal_separation.workflow"), facts, config))

    engines: dict[str, str] = {}
    for engine_id, engine in config.get("engines", {}).items():
        item = _workflow_item(f"engine_{engine_id}", engine_id, engine["workflow"], facts, config)
        engines[engine_id] = item.level
        items.append(item)

    base_level = worst(*(i.level for i in items if not i.id.startswith("engine_")))
    best_engine = min(engines.values(), key=_ORDER.__getitem__) if engines else FAIL
    if base_level == FAIL or best_engine == FAIL:
        overall = FAIL
    else:
        overall = worst(base_level, *engines.values()) if FAIL not in engines.values() else DEGRADE
    return {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "comfy_url": facts.comfy_url,
        "comfy_version": facts.comfy_version,
        "overall": overall,
        "base_ok": base_level != FAIL,
        "engines": engines,
        "items": [i.to_dict() for i in items],
    }


def can_start(report: dict, engine_id: str) -> bool:
    """基本項目都沒有不通過，而且選用的對嘴引擎可用，才允許開始製作。"""
    return bool(report.get("base_ok")) and report.get("engines", {}).get(engine_id, FAIL) != FAIL


class EnvStage:
    id = "s1_env"

    def check_inputs(self, ctx: StageContext) -> list[Problem]:
        return []

    def outputs(self, ctx: StageContext) -> list[Path]:
        return [ctx.project.path("env_report.json")] if ctx.project else []

    def run(self, ctx: StageContext, reporter: Reporter) -> StageResult:
        facts = collect_facts(ctx.config, reporter)
        report = evaluate(facts, ctx.config)
        self.report = report
        if ctx.project is not None:
            ctx.project.path("env_report.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        reporter.progress(1.0, "完成")
        warnings = [f"{i['label']}：{i['detail']}" for i in report["items"] if i["level"] == DEGRADE]
        if report["overall"] == FAIL:
            failed = "、".join(i["label"] for i in report["items"] if i["level"] == FAIL)
            return StageResult(False, warnings, f"環境檢查不通過：{failed}")
        return StageResult(True, warnings)

    resume = run


def load_report(project_folder: Path) -> dict | None:
    try:
        return json.loads((Path(project_folder) / "env_report.json").read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None
