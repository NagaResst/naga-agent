# Naga Agent

一个运行在本地终端的智能代理，支持任意 OpenAI 兼容模型，具备工具调用、智能路由、分层记忆和上下文压缩能力。

## 特性

- **模型无关**：连接任意 OpenAI 兼容接口（OpenAI、Anthropic、DeepSeek、Qwen、MiniMax、本地 Ollama 等）
- **智能路由**：根据任务复杂度自动选择 low / medium / high 档模型，节省费用
- **工具调用**：Shell 命令执行、文件读写编辑、脚本生成、网页搜索与抓取、记忆管理
- **三层记忆**：SQLite 核心记忆 + mem0 向量情节记忆 + 历史摘要，跨会话持久化，无需外部服务
- **上下文压缩**：零 LLM 调用的机械压缩管线，自动维持 token 预算
- **Skill 系统**：按需加载 `.md` 格式的专项能力提示词
- **思维链支持**：透明展示模型推理过程（支持 MiniMax M2.7 reasoning_split、Qwen enable_thinking 等）

---

## 快速开始

### 依赖

- Python 3.11+
- SQLite（Python 标准库内置，无需额外安装）

### 安装

```bash
git clone <repo-url>
cd naga-agent
pip install -r requirements.txt
```

### 配置

**1. 环境变量**（复制 `.env.example` 为 `.env`）

```bash
cp .env.example .env
```

编辑 `.env`：

```env
OPENAI_API_KEY=sk-...          # API Key
OPENAI_BASE_URL=               # 留空使用 OpenAI 官方，填入第三方兼容服务地址
BOCHA_API_KEY=                 # 博查搜索 API Key，web_search 工具使用
```

**MiniMax M2.7 快速配置**：

```env
OPENAI_API_KEY=your_minimax_api_key
OPENAI_BASE_URL=https://api.minimaxi.com/v1
```

当前 `config.toml` 已预配置 MiniMax M2.7 作为默认模型，支持 reasoning_split 思维链拆分。
详见 [MINIMAX_SETUP.md](MINIMAX_SETUP.md) 和 [QUICKSTART_MINIMAX.md](QUICKSTART_MINIMAX.md)。

**2. 模型与行为配置**（`config.toml`）

关键配置项：

```toml
[storage]
db_path = "naga.db"            # SQLite 数据库路径（相对于项目根目录）

[model]
default = "gpt-4o-mini"        # 默认模型
available = ["gpt-4o-mini", "gpt-4o"]
base_url = ""                  # 同 OPENAI_BASE_URL，config 优先级更高

[agent]
context_token_limit = 0        # 0 = 自动检测，非零时强制覆盖
compress_threshold = 0.60      # token 用量达上下文窗口 60% 时触发压缩
tool_output_max_chars = 300    # 旧轮次工具输出最大保留字符数

[memory]
backend = "chroma"             # chroma（内嵌）/ qdrant
```

### 运行

```bash
python main.py
```

启动后选择模型和会话，进入交互 REPL。

---

## 工具系统

代理内置 7 个工具，启动时自动注册：

| 工具 | 说明 |
|------|------|
| `execute_command` | 执行本地 Shell 命令，含黑名单拦截和危险命令强制确认 |
| `read_file` | 读取本地文件内容，支持行范围指定 |
| `edit_file` | 精确字符串替换编辑文件，编辑前需先 read_file 确认 |
| `generate_script` | 在 `generated_scripts/` 目录生成脚本文件 |
| `fetch_url` | 抓取网页正文内容（HTML 转纯文本） |
| `web_search` | 使用博查 API 搜索中文互联网，返回结构化结果 |
| `memory` | 显式读写 Layer1 核心记忆（save / recall / delete / list） |

自定义工具：在 `tools/` 目录下新建 `.py` 文件，实现 `TOOL_DEFINITION` 和 `execute(args)` 即可自动注册。

---

## 分层记忆系统

记忆分三层，由 mem0 + SQLite + ChromaDB 共同支撑：

```
Layer1  核心记忆（SQLite KV）
        ──────────────────────────────────────────────
        用户的长期偏好、固定配置、身份信息。
        每轮对话开始时全量注入 system prompt。
        通过 memory 工具的 save/recall 操作显式管理。
        进程内缓存预热，冷启动后精确查找仍可用。

Layer2  情节记忆（mem0 + 向量库）
        ──────────────────────────────────────────────
        过往对话中的关键事件和操作结论。
        系统每 N 轮自动提取（可配置 extract_every_n_turns）。
        每轮根据当前话题语义检索 Top-K 条注入 system prompt。

Layer3  历史摘要（SQLite 字符串，只读）
        ──────────────────────────────────────────────
        极早期对话的旧式压缩摘要（遗留兼容）。
        只读，不再写入新摘要。
```

**支持的向量库后端**（通过 `config.toml [memory] backend` 切换）：

| 后端 | 模式 | 说明 |
|------|------|------|
| `chroma` | local | 内嵌，无需独立服务（默认） |
| `chroma` | server | 独立部署 Chroma HTTP 服务 |
| `qdrant` | local | 内嵌文件模式 |
| `qdrant` | server | 独立部署 Qdrant |

**嵌入模型**（通过 `config.toml [memory.embedder]` 配置）：

- `openai`：`text-embedding-3-small`（默认），支持自定义 `base_url`
- `ollama`：本地部署，如 `nomic-embed-text`

---

## 智能路由

开启路由（`[routing] enabled = true`）后，每条用户消息会先被分类为三档复杂度，再映射到对应费用等级的模型：

| 复杂度 | 示例 | 模型档位 |
|--------|------|----------|
| `simple` | 问候、单一事实查询 | low |
| `medium` | 需要推理或知识整合 | medium |
| `complex` | 代码生成、系统设计、深度分析 | high |

分类结果进程内缓存 7 天，相同输入不重复分类。使用 `/model <名称>` 可锁定模型跳过路由。

---

## 上下文压缩

当消息历史的 token 估算值达到模型上下文窗口的 `compress_threshold`（默认 60%）时，自动触发三步压缩管线：

```
Step 1  slim_tool_outputs
        对非最新 3 轮的工具输出截断至 300 字符

Step 2  priority_trim
        按优先级丢弃旧消息（P1=工具调用 > P2=长用户消息 > P3=长助手消息 > P4=其他）

Step 3  dual_track_compress（兜底）
        最旧 1/3 消息折叠为 [已折叠]，中间 1/3 截断
```

**零 LLM 调用，不影响记忆系统。** 压缩前 mem0 会先同步提取一次情节记忆。

---

## Skill 系统

在 `skills/` 目录下放置 `.md` 文件，使用 YAML frontmatter 声明元数据：

```markdown
---
name: ops-expert
description: 运维专家模式，强化 K8s/Linux 操作指引
keywords: ["k8s", "kubernetes", "linux 运维", "故障排查"]
examples: ["帮我排查 k8s pod 重启", "分析这台 Linux 机器的 CPU 飙高原因"]
enabled: false
---

你是一个资深运维工程师，处理以下任务时...
```

推荐为每个 Skill 补充 `keywords` 和 `examples`：

- `keywords`：放 5~12 个高区分度短语，优先写用户真实会说的话，不要只写泛词。
- `examples`：放 2~5 条典型请求，帮助匹配器识别近似表达。
- `description`：保留一段简洁摘要，不要把整套执行流程塞进 frontmatter。

| 操作 | 命令 |
|------|------|
| 查看所有 Skill | `/skill list` |
| 激活 Skill | `/skill on ops-expert` |
| 停用 Skill | `/skill off ops-expert` |

也可在 `config.toml` 中预设激活列表：

```toml
[skills]
enabled = ["ops-expert", "code-reviewer"]
match_top_k = 2
summary_top_k = 6
min_match_score = 2.2
```

匹配策略说明：Agent 会综合 `name`、`description`、`keywords`、`examples` 做相关度排序，只把 Top-K 命中的 Skill 全文注入 prompt，其余仅保留摘要，避免无关 Skill 污染上下文。

如果你想在对话里强制使用某个 Skill，可以在用户消息里显式写 `@skill(skill-name)`，例如 `@skill(fund-deep-research) 帮我分析 007119`。这样该 Skill 会被直接提升到注入优先级最高。

---

## 内置命令速查

| 命令 | 说明 |
|------|------|
| `/help` | 显示所有命令 |
| `/model <名称>` | 切换并锁定模型 |
| `/model auto` | 恢复自动路由 |
| `/thinking on/off` | 切换思维链模式 |
| `/thinking show/hide` | 显示/隐藏思考过程 |
| `/skill list` | 查看所有 Skill |
| `/skill on/off <名称>` | 激活/停用 Skill |
| `/session` | 当前会话信息（含 token 和费用） |
| `/session new` | 创建并切换到新会话 |
| `/session list` | 列出最近 10 条历史会话 |
| `/session switch <N>` | 切换到 list 中编号为 N 的会话 |
| `/token` | 详细 token 用量明细 |
| `/history` | 最近 5 条对话摘要 |
| `/prompt` | 查看当前 system prompt |
| `/prompt set <内容>` | 设置 system prompt |
| `/prompt clear` | 清除 system prompt |
| `/confirm on/off` | 开关命令自动确认 |
| `/routing on/off` | 开关智能路由 |
| `/clear` | 清除当前会话历史 |
| `/quit` | 退出 |

---

## 配置参考

### `[model]`

| 字段 | 说明 |
|------|------|
| `default` | 默认使用的模型名 |
| `available` | 可用模型列表（启动时展示选择） |
| `base_url` | 第三方兼容服务地址，留空使用 OpenAI 官方 |
| `tiers` | 模型费用等级标记（low/medium/high） |
| `extra_params.<model>` | 模型专属扩展参数，注入 `extra_body`（如 Qwen3 思维链参数） |

### `[routing]`

| 字段 | 说明 |
|------|------|
| `enabled` | 是否启用智能路由 |
| `classifier_model` | 用于分类的轻量模型 |
| `show_routing_decision` | 是否在界面显示路由决策 |

### `[agent]`

| 字段 | 默认值 | 说明 |
|------|--------|------|
| `max_history` | 50 | 会话最大保留消息条数 |
| `auto_confirm` | true | 命令自动确认（false 时危险命令需手动确认） |
| `context_token_limit` | 0 | 0=自动检测，非零=强制覆盖上下文窗口大小 |
| `compress_threshold` | 0.60 | 触发压缩的 token 占比阈值 |
| `tool_output_max_chars` | 300 | 旧轮次工具输出截断字符数 |
| `show_thinking` | true | 是否展示思维链内容 |
| `system_prompt_file` | — | 系统提示词文件路径 |

### `[memory]`

| 字段 | 默认值 | 说明 |
|------|--------|------|
| `backend` | chroma | 向量库后端（chroma/qdrant） |
| `extract_every_n_turns` | 3 | 自动提取情节记忆的间隔轮数 |
| `episodic_top_k` | 3 | 情节记忆语义检索返回条数 |
| `core_max_items` | 20 | 核心记忆注入 system prompt 的最大条数 |

---

## 项目结构

```
naga-agent/
├── main.py                   # CLI 入口，REPL 主循环
├── config.toml               # 全局配置
├── requirements.txt
├── .env.example
├── agent/
│   ├── core.py               # Agent 主类，消息构建，工具调用循环
│   ├── config.py             # 配置加载器
│   ├── router.py             # 智能模型路由
│   ├── summarizer.py         # 零LLM上下文压缩管线
│   ├── context_window.py     # 模型上下文窗口查找表
│   ├── token_tracker.py      # tiktoken token 计数
│   └── skill_registry.py     # Skill 扫描与加载
├── memory/
│   └── manager.py            # 三层记忆管理器（SQLite Layer1 + mem0 向量层）
├── session/
│   └── sqlite_session.py     # SQLite 会话持久化（会话、消息、记忆、摘要）
├── tools/                    # 工具模块（自动注册）
│   ├── execute_command.py
│   ├── edit_file.py
│   ├── read_file.py
│   ├── generate_script.py
│   ├── fetch_url.py
│   ├── web_search.py
│   └── memory.py
├── skills/                   # Skill 提示词（.md 格式）
└── generated_scripts/        # 生成的脚本和提示词文件
```
