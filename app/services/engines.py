"""對嘴引擎：把「一張起始影像 + 一段音訊 → 一段影片」包成同一個介面，日後可再加引擎。

引擎只負責組出要送給 ComfyUI 的工作流；分段、續跑、接合由影片階段處理。
"""
from __future__ import annotations

from dataclasses import dataclass

from app.core.models import UserFacingError
from app.core.segmenter import MOTION, PREV
from app.pipeline.stages.s1_env import resolve_models
from app.services.workflow_template import WorkflowTemplate, load_requires
from app.storage.app_config import AppConfig


JOIN_ANCHOR, JOIN_PHOTO, JOIN_PREV = "anchor", "photo", "prev"


@dataclass
class SegmentRequest:
    image: str               # 已上傳到 ComfyUI 的起始影像名稱
    audio: str               # 已上傳的音訊片段名稱
    frames: int              # 要生成的影格數
    width: int
    height: int
    prompt: str
    seed: int
    filename_prefix: str
    low_vram: bool = False   # 顯示記憶體不足後的重試：改用更省 VRAM 的設定
    anchor_image: str = ""   # 已上傳的原始照片；有值時這一段的結尾會回到照片的樣子
    motion_video: str = ""   # 已上傳的上一段尾端影片；有值時這一段接續它的動作


class LipSyncEngine:
    """各引擎共用的部分：讀設定、載入工作流範本、決定影片尺寸。"""
    id = ""
    # 一段影片的生成過程中，取樣節點會跑幾輪（用來把節點進度換算成這一段的進度）
    sampler_passes = 1
    # 延續段怎麼接：PREV 以上一段最後一幀起始；MOTION 以上一段的尾端當動作參考
    continuation = PREV
    # 以 MOTION 接續時，要給模型上一段最後幾幀
    motion_tail_frames = 0
    # 輸出的畫面比音訊早幾幀（模型解碼方式造成的固定偏移）；送音訊時往前多取這幾幀來對齊
    audio_lead_frames = 0

    def passes(self, frames: int) -> int:
        """生成 frames 幀時取樣節點會跑幾輪。"""
        return self.sampler_passes

    def __init__(self, config: AppConfig):
        self.config = config
        self.settings: dict = dict(config.get(f"engines.{self.id}", {}))     # 複本：專案的設定會覆蓋其中幾項
        self.label: str = self.settings.get("label", self.id)
        self.fps: float = float(self.settings.get("fps", 25))
        self.frame_step: int = int(self.settings.get("frame_step", 1))
        self.frame_offset: int = int(self.settings.get("frame_offset", 0))
        self.max_segment_seconds: float = float(self.settings.get("max_segment_seconds", 60))
        self.min_segment_seconds: float = float(self.settings.get("min_segment_seconds", 3))
        self.reset_every: int = int(self.settings.get("reset_every", 4))
        self.default_prompt: str = self.settings.get("default_prompt", "")
        self.template = WorkflowTemplate.load(config.workflows_dir, self.settings["workflow"])
        self._requires = load_requires(config.workflows_dir, self.settings["workflow"])
        # 結尾引導：有設定專用工作流的引擎才支援
        self.anchor_template = (WorkflowTemplate.load(config.workflows_dir, self.settings["anchor_workflow"])
                                if self.settings.get("end_anchor") and self.settings.get("anchor_workflow") else None)

        self.join_mode = self.default_join()["join_mode"]

    @property
    def end_anchor(self) -> bool:
        return self.anchor_template is not None

    @property
    def supports_anchor(self) -> bool:
        """這個引擎有沒有「段尾回到照片」用的工作流。"""
        return bool(self.settings.get("anchor_workflow"))

    def default_join(self) -> dict:
        """設定檔決定的分段銜接方式，格式與專案輸入的 join_* 欄位相同。"""
        mode = JOIN_ANCHOR if self.end_anchor else JOIN_PHOTO if self.reset_every > 0 else JOIN_PREV
        return {"join_mode": mode, "join_every": max(self.reset_every, 1),
                "anchor_strength": float(self.settings.get("anchor_strength", 0.8))}

    def configure_join(self, inputs: dict) -> None:
        """套用專案選擇的分段銜接方式；專案沒有指定的項目沿用設定檔。

        - anchor：每一段都以上一段的最後一幀起始，每 N 段讓結尾回到照片的樣子（引擎不支援時改用 photo）
        - photo：每 N 段改以原始照片起始，其餘以上一段的最後一幀起始
        - prev：一律以上一段的最後一幀起始，不做任何校正
        """
        join = {**self.default_join(), **{k: inputs[k] for k in ("join_mode", "join_every", "anchor_strength")
                                          if inputs.get(k) not in (None, "")}}
        mode = join["join_mode"]
        if mode == JOIN_ANCHOR and not self.supports_anchor:
            mode = JOIN_PHOTO
        if mode == JOIN_ANCHOR:
            if self.anchor_template is None:
                self.anchor_template = WorkflowTemplate.load(self.config.workflows_dir, self.settings["anchor_workflow"])
            self.settings["anchor_strength"] = min(max(float(join["anchor_strength"]), 0.0), 1.0)
        else:
            self.anchor_template = None
        self.reset_every = 0 if mode == JOIN_PREV else max(int(join["join_every"]), 1)
        self.join_mode = mode

    def size(self, resolution: str) -> tuple[int, int]:
        """影片尺寸：引擎有自己的尺寸表就用它的（模型對長寬有倍數限制），否則用共用的。"""
        table = self.settings.get("resolutions") or self.config.get("video.resolutions", {})
        entry = table.get(resolution) or self.config.get(f"video.resolutions.{resolution}")
        if not entry:
            raise UserFacingError(f"設定檔沒有定義解析度「{resolution}」", "請到輸入頁重新選擇解析度")
        return int(entry["width"]), int(entry["height"])

    def check(self, object_info: dict) -> None:
        problems = self.template.check_against(object_info)
        if self.anchor_template:
            problems += self.anchor_template.check_against(object_info)
        if problems:
            raise UserFacingError(f"{self.label} 工作流範本與本機 ComfyUI 不相容：" + "；".join(problems),
                                  "請更新 workflows 資料夾內的範本與對應表")

    def build(self, request: SegmentRequest, object_info: dict) -> dict:
        raise NotImplementedError

    @property
    def output_node(self) -> str:
        return self.template.output_node("video")


class InfiniteTalkEngine(LipSyncEngine):
    """InfiniteTalk（kijai ComfyUI-WanVideoWrapper，單人模式）。嘴型穩定，但很慢。"""
    id = "infinitetalk"

    def build(self, request: SegmentRequest, object_info: dict) -> dict:
        s = self.settings
        blocks = s.get("oom_blocks_to_swap", 40) if request.low_vram else s.get("blocks_to_swap", 20)
        return self.template.inject({
            **resolve_models(self._requires, object_info),
            "image": request.image,
            "audio": request.audio,
            "num_frames": request.frames,
            "fps": self.fps,
            "width": request.width,
            "height": request.height,
            "positive_prompt": request.prompt,
            "negative_prompt": s.get("negative_prompt", ""),
            "seed": request.seed,
            "steps": int(s.get("steps", 6)),
            "blocks_to_swap": int(blocks),
            "frame_window_size": int(s.get("frame_window_size", 81)),
            "motion_frame": int(s.get("motion_frame", 9)),
            "filename_prefix": request.filename_prefix,
        })


class LTX23Engine(LipSyncEngine):
    """LTX-2.3（官方「圖片＋音訊轉影片」範本）。先以一半解析度生成，再放大兩倍精修。快，但單段長度有限。"""
    id = "ltx23"
    sampler_passes = 2

    def build(self, request: SegmentRequest, object_info: dict) -> dict:
        s = self.settings
        template, anchor = self.template, {}
        if request.anchor_image and self.anchor_template:
            template = self.anchor_template
            anchor = {"anchor_image": request.anchor_image,
                      "anchor_strength": float(s.get("anchor_strength", 0.8))}
        return template.inject({
            **resolve_models(self._requires, object_info),
            **anchor,
            "image": request.image,
            "audio": request.audio,
            "audio_seconds": request.frames / self.fps,
            "num_frames": request.frames,
            "fps": self.fps,
            "width": request.width,
            "height": request.height,
            "half_width": request.width // 2,
            "half_height": request.height // 2,
            "positive_prompt": request.prompt,
            "negative_prompt": s.get("negative_prompt", ""),
            "seed": request.seed,
            "decode_tile_size": int(s.get("oom_decode_tile_size", 512) if request.low_vram
                                    else s.get("decode_tile_size", 768)),
            "filename_prefix": request.filename_prefix,
        })


def _renumber(nodes: dict, tag: str) -> dict:
    """把一組節點的編號加上後綴（連到同組節點的線一併改），回傳新的一組。"""
    def link(value):
        return [f"{value[0]}{tag}", value[1]] if isinstance(value, list) and len(value) == 2 \
            and str(value[0]) in nodes else value
    return {f"{node_id}{tag}": {"class_type": node["class_type"],
                                "inputs": {k: link(v) for k, v in node["inputs"].items()}}
            for node_id, node in nodes.items()}


class Wan22S2VEngine(LipSyncEngine):
    """Wan 2.2 S2V（官方「音訊＋照片轉影片」範本，4 步 LoRA）。

    模型一次生成一個區塊（77 幀，約 5 秒），更長的片段在同一個工作流裡一個區塊接一個區塊延伸。
    照片是每個區塊的參考影像，所以人物不會越接越不像；跨段時把上一段的尾端當成動作參考來接續。
    """
    id = "wan22_s2v"
    continuation = MOTION
    motion_tail_frames = 73
    audio_lead_frames = 2
    CHUNK_STEP = 80          # 每個區塊實際推進的影格數（77 幀在潛空間是 20 格，解碼成 80 幀）

    def chunks(self, frames: int) -> int:
        """要幾個區塊才夠 frames 幀：K 個區塊解碼後是 80K − 2 幀。"""
        return max(-(-(frames + 2) // self.CHUNK_STEP), 1)

    def passes(self, frames: int) -> int:
        return self.chunks(frames)

    def build(self, request: SegmentRequest, object_info: dict) -> dict:
        s = self.settings
        workflow = self.template.inject({
            **resolve_models(self._requires, object_info),
            "image": request.image,
            "audio": request.audio,
            "width": request.width,
            "height": request.height,
            "chunk_frames": int(s.get("chunk_frames", 77)),
            "fps": self.fps,
            "positive_prompt": request.prompt,
            "negative_prompt": s.get("negative_prompt", ""),
            "seed": request.seed,
            "steps": int(s.get("steps", 4)),
            "shift": float(s.get("shift", 8.0)),
            "filename_prefix": request.filename_prefix,
        })
        first, sampler = workflow["12"], workflow["13"]
        if request.motion_video:
            workflow["20"] = {"class_type": "LoadVideo", "inputs": {"file": request.motion_video}}
            workflow["21"] = {"class_type": "GetVideoComponents", "inputs": {"video": ["20", 0]}}
            first["inputs"]["ref_motion"] = ["21", 0]
        # 延伸區塊：每個都接在目前為止的潛空間影片後面（官方範本的 Video S2V Extend 子圖）
        total = ["13", 0]
        for i in range(1, self.chunks(request.frames)):
            extend, sample, concat = f"e{i}", f"k{i}", f"c{i}"
            workflow[extend] = {"class_type": "WanSoundImageToVideoExtend", "inputs": {
                "positive": first["inputs"]["positive"], "negative": first["inputs"]["negative"],
                "vae": first["inputs"]["vae"], "length": first["inputs"]["length"], "video_latent": total,
                "audio_encoder_output": first["inputs"]["audio_encoder_output"], "ref_image": first["inputs"]["ref_image"]}}
            workflow[sample] = {"class_type": "KSampler", "inputs": {
                **sampler["inputs"], "positive": [extend, 0], "negative": [extend, 1], "latent_image": [extend, 2],
                "seed": request.seed + i}}
            workflow[concat] = {"class_type": "LatentConcat", "inputs": {"samples1": total, "samples2": [sample, 0], "dim": "t"}}
            total = [concat, 0]
        workflow["30"]["inputs"]["samples"] = total
        workflow["31"]["inputs"]["samples2"] = total
        return workflow


class InfiniteTalkNativeEngine(LipSyncEngine):
    """InfiniteTalk 原生版（ComfyUI 內建的 WanInfiniteTalkToVideo，官方範本）。

    與外掛版是同一個模型。一次生成一個視窗（81 幀），更長的片段在同一個工作流裡一個視窗接一個視窗：
    後面的視窗拿前面所有畫面當 previous_frames，開頭有幾幀與前面重疊，接合前要去掉。
    """
    id = "infinitetalk_native"
    WINDOW_NODES = ("12", "13", "15", "16", "17", "18")       # 每個視窗各自一份的節點

    def windows(self, frames: int) -> int:
        window, motion = int(self.settings.get("frame_window_size", 81)), int(self.settings.get("motion_frame", 9))
        return 1 + max(-(-(frames - window) // (window - motion)), 0)

    def passes(self, frames: int) -> int:
        return self.windows(frames)

    def build(self, request: SegmentRequest, object_info: dict) -> dict:
        s = self.settings
        workflow = self.template.inject({
            **resolve_models(self._requires, object_info),
            "image": request.image,
            "audio": request.audio,
            "width": request.width,
            "height": request.height,
            "window_frames": int(s.get("frame_window_size", 81)),
            "motion_frame": int(s.get("motion_frame", 9)),
            "fps": self.fps,
            "positive_prompt": request.prompt,
            "seed": request.seed,
            "steps": int(s.get("steps", 6)),
            "filename_prefix": request.filename_prefix,
        })
        base = {node_id: workflow[node_id] for node_id in self.WINDOW_NODES}
        total = ["18", 0]
        for i in range(1, self.windows(request.frames)):
            tag = f"w{i}"
            window = _renumber(base, tag)
            window[f"12{tag}"]["inputs"]["previous_frames"] = total
            window[f"16{tag}"]["inputs"]["noise_seed"] = request.seed + i
            workflow.update(window)
            # 視窗開頭與前面重疊的影格數由節點的 trim_image 輸出決定
            workflow[f"t{i}"] = {"class_type": "ImageFromBatch", "inputs": {
                "image": [f"18{tag}", 0], "batch_index": [f"12{tag}", 4], "length": 4096}}
            workflow[f"b{i}"] = {"class_type": "ImageBatch", "inputs": {"image1": total, "image2": [f"t{i}", 0]}}
            total = [f"b{i}", 0]
        workflow[self.output_node]["inputs"]["images"] = total
        return workflow


ENGINES = {"infinitetalk": InfiniteTalkEngine, "infinitetalk_native": InfiniteTalkNativeEngine,
           "ltx23": LTX23Engine, "wan22_s2v": Wan22S2VEngine}


def make_engine(config: AppConfig, engine_id: str) -> LipSyncEngine:
    if engine_id not in ENGINES:
        label = config.get(f"engines.{engine_id}.label", engine_id)
        raise UserFacingError(f"對嘴引擎 {label} 尚未實作", "請到輸入頁改用其他引擎")
    return ENGINES[engine_id](config)
