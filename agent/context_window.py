"""模型上下文窗口探测器。

查找优先级：
    1. config 手动指定（context_token_limit > 0）
    2. 本地持久化缓存 / 进程内缓存（仅接受远端 metadata 已确认的值）
    3. provider models 元数据中的上下文字段
    4. 默认值 200000（200k）

说明：
    - 不允许基于模型名、usage、报错文本做推断。
    - 如果远端 models 接口不暴露上下文窗口，就只能回退到保守默认值或由用户手动覆盖。
"""

from __future__ import annotations

import json
import os
import threading
from typing import Any

_DEFAULT = 200000  # 200k，未知模型时的保守回退值
_CACHE_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".context_window_cache.json")
_CACHE_LOCK = threading.Lock()
_CACHE_LOADED = False
_DETECTED_WINDOWS: dict[str, int] = {}

_CONTEXT_KEYS = (
    "context_window",
    "context_length",
    "max_context_tokens",
    "max_input_tokens",
    "input_token_limit",
    "max_prompt_tokens",
    "max_sequence_length",
    "max_position_embeddings",
)


def _normalize_base_url(base_url: str | None) -> str:
    return (base_url or "default").rstrip("/").lower()


def _cache_key(model: str, base_url: str | None) -> str:
    return f"{_normalize_base_url(base_url)}::{(model or '').strip().lower()}"


def _load_cache():
    global _CACHE_LOADED
    if _CACHE_LOADED:
        return
    with _CACHE_LOCK:
        if _CACHE_LOADED:
            return
        if os.path.exists(_CACHE_FILE):
            try:
                with open(_CACHE_FILE, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                if isinstance(payload, dict):
                    for key, value in payload.items():
                        if isinstance(value, dict):
                            source = value.get("source", "")
                            window = value.get("context_window", 0)
                            if source == "remote_metadata" and isinstance(window, int) and window > 0:
                                _DETECTED_WINDOWS[key] = window
            except Exception:
                pass
        _CACHE_LOADED = True


def _save_cache():
    with _CACHE_LOCK:
        try:
            with open(_CACHE_FILE, "w", encoding="utf-8") as f:
                payload = {
                    key: {"context_window": value, "source": "remote_metadata"}
                    for key, value in _DETECTED_WINDOWS.items()
                    if isinstance(value, int) and value > 0
                }
                json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
        except Exception:
            pass


def _extract_context_value(payload: Any) -> int:
    if isinstance(payload, dict):
        for key, value in payload.items():
            lowered = key.lower()
            if lowered in _CONTEXT_KEYS and isinstance(value, int) and value > 0:
                return value
            detected = _extract_context_value(value)
            if detected > 0:
                return detected
    elif isinstance(payload, list):
        for item in payload:
            detected = _extract_context_value(item)
            if detected > 0:
                return detected
    return 0


def _to_dict(item: Any) -> dict[str, Any]:
    if item is None:
        return {}
    if hasattr(item, "to_dict"):
        try:
            return item.to_dict()
        except Exception:
            return {}
    if isinstance(item, dict):
        return item
    if hasattr(item, "__dict__"):
        return dict(item.__dict__)
    return {}


def _detect_from_metadata(model: str, client=None) -> int:
    if client is None or not model:
        return 0

    try:
        payload = _to_dict(client.models.retrieve(model))
        detected = _extract_context_value(payload)
        if detected > 0:
            return detected
    except Exception:
        pass

    try:
        listing = client.models.list()
        for item in getattr(listing, "data", []) or []:
            payload = _to_dict(item)
            if str(payload.get("id", "")).lower() != model.lower():
                continue
            detected = _extract_context_value(payload)
            if detected > 0:
                return detected
            break
    except Exception:
        pass

    return 0


def remember_context_window(model: str, context_window: int, base_url: str | None = None) -> int:
    if not model or context_window <= 0:
        return 0
    _load_cache()
    key = _cache_key(model, base_url)
    existing = _DETECTED_WINDOWS.get(key, 0)
    if context_window <= existing:
        return existing
    _DETECTED_WINDOWS[key] = context_window
    _save_cache()
    return context_window


def get_context_window(model: str, config_override: int = 0, client=None, base_url: str | None = None) -> int:
    """返回指定模型的上下文窗口大小（token 数）。"""
    if config_override and config_override > 0:
        return config_override

    _load_cache()
    key = _cache_key(model, base_url)
    cached = _DETECTED_WINDOWS.get(key, 0)
    if cached > 0:
        return cached

    detected = _detect_from_metadata(model, client=client)
    if detected > 0:
        return remember_context_window(model, detected, base_url=base_url)

    return _DEFAULT
