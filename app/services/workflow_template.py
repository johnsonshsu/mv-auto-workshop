"""工作流範本：API 格式 JSON ＋ 對應表（參數名稱 → 節點 ID 與輸入欄位）＋ 需求清單。

程式只認參數名稱，節點 ID 只存在於對應表；換工作流時改對應表即可。
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from app.core.models import UserFacingError


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise UserFacingError(f"找不到工作流檔案：{path.name}", f"請確認 {path.parent} 內有這個檔案")
    except json.JSONDecodeError as e:
        raise UserFacingError(f"工作流檔案 {path.name} 不是合法的 JSON（{e}）", "請重新匯出或修正這個檔案")


def load_requires(workflows_dir: Path, name: str) -> dict:
    return _read_json(Path(workflows_dir) / f"{name}.requires.json")


class WorkflowTemplate:
    def __init__(self, name: str, workflow: dict, mapping: dict):
        self.name = name
        self.workflow = workflow
        self.params: dict[str, list[dict]] = mapping.get("params", {})
        self.outputs: dict[str, dict] = mapping.get("outputs", {})
        # 節點類型 → 不檢查的欄位。有些節點的欄位會隨其他選項動態出現，/object_info 查不到
        self.unchecked: dict[str, list[str]] = mapping.get("unchecked_fields", {})
        self._validate()

    @classmethod
    def load(cls, workflows_dir: Path, name: str) -> "WorkflowTemplate":
        workflows_dir = Path(workflows_dir)
        mapping = _read_json(workflows_dir / f"{name}.map.json")
        workflow = _read_json(workflows_dir / mapping.get("template", f"{name}.api.json"))
        return cls(name, workflow, mapping)

    def _validate(self) -> None:
        """對應表指到的節點與欄位都必須存在於範本中。"""
        if "nodes" in self.workflow and "links" in self.workflow:
            raise UserFacingError(
                f"工作流 {self.name} 是一般儲存格式，不是 API 格式",
                "請在 ComfyUI 用 File → Export Workflow (API) 重新匯出")
        for param, targets in self.params.items():
            for i, target in enumerate(targets, start=1):
                self._check_target(f"參數「{param}」第 {i} 項", target)
        for key, target in self.outputs.items():
            node_id = str(target.get("node"))
            if node_id not in self.workflow:
                raise UserFacingError(
                    f"工作流 {self.name} 的對應表：輸出「{key}」指向不存在的節點 {node_id}",
                    f"請修正 {self.name}.map.json")

    def _check_target(self, where: str, target: dict) -> None:
        node_id, field = str(target.get("node")), target.get("input")
        node = self.workflow.get(node_id)
        if node is None:
            raise UserFacingError(
                f"工作流 {self.name} 的對應表：{where}指向不存在的節點 {node_id}",
                f"請修正 {self.name}.map.json")
        if field not in node.get("inputs", {}):
            raise UserFacingError(
                f"工作流 {self.name} 的對應表：{where}指向節點 {node_id}"
                f"（{node.get('class_type')}）不存在的欄位「{field}」",
                f"請修正 {self.name}.map.json")

    def inject(self, values: dict[str, Any]) -> dict:
        """回傳注入參數後的新工作流；不會改動範本本身。"""
        unknown = [k for k in values if k not in self.params]
        if unknown:
            raise UserFacingError(
                f"工作流 {self.name} 的對應表沒有定義參數：{'、'.join(unknown)}",
                f"請在 {self.name}.map.json 的 params 補上對應")
        workflow = copy.deepcopy(self.workflow)
        for param, value in values.items():
            for target in self.params[param]:
                workflow[str(target["node"])]["inputs"][target["input"]] = value
        return workflow

    def output_node(self, key: str) -> str:
        if key not in self.outputs:
            raise UserFacingError(
                f"工作流 {self.name} 的對應表沒有定義輸出「{key}」", f"請在 {self.name}.map.json 的 outputs 補上")
        return str(self.outputs[key]["node"])

    def check_against(self, object_info: dict) -> list[str]:
        """以本機 /object_info 驗證範本用到的節點類型與欄位仍然存在，回傳問題清單。"""
        problems = []
        for node_id, node in self.workflow.items():
            class_type = node.get("class_type")
            info = object_info.get(class_type)
            if info is None:
                problems.append(f"節點 {node_id}：本機 ComfyUI 沒有「{class_type}」")
                continue
            known = set(info.get("input", {}).get("required", {})) | set(info.get("input", {}).get("optional", {}))
            for field in node.get("inputs", {}):
                if field not in known and field not in self.unchecked.get(class_type, []):
                    problems.append(f"節點 {node_id}（{class_type}）：欄位「{field}」已不存在")
        return problems
