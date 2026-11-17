"""运行阶段跟踪器：供底部状态栏显示 Agent 当前正在做什么。

线程安全：埋点发生在 worker 线程（Agent.chat），渲染发生在
prompt_toolkit 主循环（bottom_toolbar / 直画），全程持锁。
"""

import threading
import time


class ActivityTracker:
    # key → 状态栏显示文案
    LABELS = {
        "idle": "待命",
        "prepare": "构建上下文",
        "memory_search": "检索情景记忆",
        "memory_rerank": "记忆重排序",
        "memory_tool": "记忆操作",
        "llm_wait": "等待LLM回复",
        "reasoning": "思考中",
        "llm_stream": "LLM输出中",
        "cmd_exec": "执行命令",
        "read_file": "读取文件",
        "mcp": "MCP调用",
        "delegate": "子任务分派",
        "skill": "加载Skill",
        "summarize": "压缩历史",
        "confirm_wait": "等待确认命令",
        "extract": "记忆提取",
    }

    def __init__(self):
        self._lock = threading.Lock()
        self._phase = "idle"
        self._detail = ""
        self._since = time.monotonic()
        self._background = ""  # 后台任务标志（如记忆提取），独立于主阶段

    def set(self, phase: str, detail: str = ""):
        """更新当前阶段。phase 见 LABELS；detail 为可选细节（如命令前缀）。"""
        with self._lock:
            if (phase, detail) != (self._phase, self._detail):
                self._phase = phase
                self._detail = detail
                self._since = time.monotonic()

    def get(self) -> tuple[str, str, float]:
        """返回 (phase, detail, 该阶段已持续秒数)。"""
        with self._lock:
            return self._phase, self._detail, time.monotonic() - self._since

    def set_background(self, name: str):
        """标记后台任务进行中（如记忆提取）。与主阶段互不影响。"""
        with self._lock:
            self._background = name

    def clear_background(self):
        with self._lock:
            self._background = ""

    def get_background(self) -> str:
        with self._lock:
            return self._background

    def label(self) -> str:
        """返回当前阶段的显示文案（未知 key 原样返回）。"""
        phase, _, _ = self.get()
        return self.LABELS.get(phase, phase)
