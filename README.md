# Naga Agent

一个运行在本地终端的智能代理，支持 OpenAI 兼容模型，具备工具调用、按需分派低成本子任务、分层记忆和 Token 用量统计功能。

当前实现的模块职责、功能流程与已知差异参见 [设计文档](docs/design.md)。

## 特性

- **模型无关**：连接任意 OpenAI 兼容接口（OpenAI、Anthropic、DeepSeek、Qwen、MiniMax、本地 Ollama 等）
- **模型驱动分派**：主模型直接处理请求，必要时将独立的简单只读任务交给低成本子 Agent
- **工具调用**：Shell 命令执行、文件读写编辑、脚本生成、网页搜索与抓取、记忆管理
- **三层记忆**：SQLite 核心记忆 + mem0 向量情节记忆 + 历史摘要，跨会话持久化，无需外部服务
- **上下文预算**：自动探测模型窗口，按完整对话轮次裁剪历史消息与超长工具输出
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
SUBAGENT_MODEL=                # 子 Agent 使用的低成本模型；留空时不分派
SUBAGENT_BASE_URL=             # 可选：独立的 OpenAI 兼容地址
SUBAGENT_API_KEY=              # 使用独立地址时必填
```

主模型与低成本子模型可以共用 OpenAI 兼容地址；使用不同服务时，须同时设置 `SUBAGENT_BASE_URL` 和 `SUBAGENT_API_KEY`。未配置子模型时仍可使用主 Agent 对话。

**2. 模型与行为配置**（`config.toml`）

关键配置项：

```toml
[storage]
db_path = "naga.db"            # SQLite 数据库路径（相对于项目根目录）

[model]
default = "GLM-5.3"    # 默认主模型，需与服务端支持的名称一致
available = ["GLM-5.3"]
base_url = "https://your-provider.example/v1"

[delegation]
enabled = false               # 当前提供商未列出独立子模型
model = ""                     # 有可用子模型时再配置并启用
max_tasks = 3                 # 单次最多 3 个子任务
max_rounds = 3                # 子 Agent 每个任务最多 3 轮只读工具调用
max_calls = 2                 # 主 Agent 单轮最多分派 2 次

[agent]
context_token_limit = 0        # 0 = 自动检测，非零时强制覆盖

[memory]
backend = "qdrant"             # 需启动 localhost:6333 的 Qdrant 服务
```

### 运行

```bash
python3 main.py
```

启动后选择模型和会话，进入交互 REPL。

---

## 工具系统

代理内置 8 个自动注册工具；配置低成本子模型时额外提供 `delegate_tasks` 分派工具：

| 工具 | 说明 |
|------|------|
| `execute_command` | 执行本地 Shell 命令，含黑名单拦截和危险命令强制确认 |
| `read_file` | 读取本地文件内容，支持行范围指定 |
| `edit_file` | 精确字符串替换编辑文件，编辑前需先 read_file 确认 |
| `generate_script` | 在 `generated_scripts/` 目录生成脚本文件 |
| `fetch_url` | 抓取网页正文内容（HTML 转纯文本） |
| `web_search` | 博查搜索，服务不可用时可回退至 DDG |
| `memory` | 显式读写核心及情景记忆 |
| `skill_manager` | 加载、删除或重扫描 Skill |
| `delegate_tasks` | 按需交给低成本子 Agent，未配置子模型时不提供 |

自定义工具：在 `tools/` 目录下新建 `.py` 文件，实现 `TOOL_DEFINITION` 和 `execute(args)` 即可自动注册。

---

## 分层记忆系统

记忆分三层，由 mem0 + SQLite + Qdrant 共同支撑：

```
Layer1  核心记忆（SQLite KV）
        ──────────────────────────────────────────────
        用户的长期偏好、固定配置、身份信息。
        按剩余预算作为参考数据提供，不构成 system 指令。
        通过 memory 工具的 save/list/delete 操作显式管理。
        进程内缓存预热，冷启动后精确查找仍可用。

Layer2  情节记忆（mem0 + 向量库）
        ──────────────────────────────────────────────
        过往对话中的关键事件和操作结论。
        系统每 N 轮自动提取（可配置 extract_every_n_turns）。
        每轮根据当前话题语义检索 Top-K 条作为参考数据提供。

Layer3  历史摘要（SQLite，带最后覆盖消息 ID）
        ──────────────────────────────────────────────
        输入接近预算上限时按完整旧轮次生成短摘要，原始消息保留。
        恢复会话时从覆盖 ID 后继续读取原文；摘要仅是参考信息。
```

当前配置需要 `localhost:6333` 的 Qdrant 服务。`[memory.embedder]` 使用远程 `Qwen3-Embedding-8B`（OpenAI 兼容 `/v1/embeddings`），`[memory.reranker]` 使用 `Qwen3-Reranker-8B`（`/v1/rerank`），接口不可用时召回会保留向量排序。嵌入维度设为 4096，写入新集合 `mem0_qwen3_embedding_8b`；旧 768 维 `mem0` 集合的数据不会自动迁移。
情景记忆的写入、检索和列表统一使用 `[memory]` 的 `agent_id`（默认 `naga-agent`），跨会话共享。更改该值后，旧作用域的记忆不再被当前查询返回；此前未设置归属 ID 的旧记忆也不会自动迁移。Qdrant 客户端版本不匹配提示已在 mem0 初始化时单独隐藏，但客户端与服务端版本并未因此变得兼容。

在环境变量或不提交的 `.env` 中设置 `VECTOR_API_KEY`；未设置时，嵌入和重排会复用主模型的 `OPENAI_API_KEY`。不要把密钥写入 `config.toml`。当前配置不启动本地嵌入 daemon。

---

## 子任务分派

每条用户消息直接由当前主模型处理，不再先请求分类模型。启用 `[delegation]` 且配置低成本 `model`（或 `SUBAGENT_MODEL`）后，主模型可按需调用 `delegate_tasks`，一次提交最多 `max_tasks` 个独立任务。子任务顺序执行、上下文互不共享，仅可使用 `read_file`、`fetch_url`、`web_search`，结果返回主模型，由主模型完成最终判断和回复。子任务不会再创建子 Agent；其模型用量单独记入会话统计。

`/delegate on/off` 控制当前进程的分派功能；`/model auto` 仅恢复配置中的默认主模型。没有配置子模型时不展示分派工具，不影响主模型直接调用原有工具。

---

## 上下文预算

`context_token_limit = 0` 时，Agent 按服务地址和模型名从 provider 元数据探测窗口；已确认值一天后重新验证，可向下更新，探测失败时暂用上次可信值。接口不提供可靠值时回退到 131072 token（128k），实际窗口更小时须手动指定正数覆盖。system 仅包含配置和代码中的固定指令；工具 schema 独立传给模型并先计入预算。启用 Skill 的 frontmatter 构成参考目录，匹配或 `@skill(name)` 指定的正文按预算加载；主题、运行状态、历史摘要、核心与情景记忆都是带标签的参考数据，不作为 system 指令。记忆最多占窗口的 10% 且不超过 2048 token。默认仅最近 3 批配对工具调用和结果进入上下文，旧批次在压缩前成对剔除，不写进摘要；用户请求和助手结论继续保留。思维链只作流式展示，从不写入会话或再次送给模型。输入达到可用预算约 75% 时归档已经完成的旧轮次，并在 SQLite 保存摘要覆盖 ID；原始工具结果仍留在 SQLite 供追溯，清空会话时删除。过长工具输出仅截断发送副本；最新用户输入仍放不下时会明确报错。子 Agent 使用相同预算算法，其窗口可用 `[delegation].context_token_limit` 覆盖。`/token` 显示最近一次上下文估算和摘要进度；估算与 API usage 校准，不保证精确。

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

匹配策略说明：Agent 以 `name`、`keywords`、`examples` 计算匹配分数，按 `min_match_score` 和 `match_top_k` 选取正文；启用 Skill 的 frontmatter 以目录形式保留，正文受独立 token 限额约束，过长时提示按需读取入口文件。

如果你想在对话里强制使用某个 Skill，可以在用户消息里显式写 `@skill(skill-name)`，例如 `@skill(fund-deep-research) 帮我分析 007119`。这样该 Skill 会被直接提升到注入优先级最高。

---

## 内置命令速查

| 命令 | 说明 |
|------|------|
| `/help` | 显示所有命令 |
| `/model <名称>` | 切换主模型 |
| `/model auto` | 恢复默认主模型 |
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
| `/delegate on/off` | 开关低成本子任务分派 |
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

### `[delegation]`

| 字段 | 说明 |
|------|------|
| `enabled` | 是否向主模型提供分派工具 |
| `model` | 低成本子模型；可用 `SUBAGENT_MODEL` 覆盖 |
| `max_tasks` | 每次分派的任务数量上限 |
| `max_rounds` | 子 Agent 的只读工具轮次上限 |
| `max_calls` | 主 Agent 单轮分派次数上限 |
| `context_token_limit` | 子模型上下文窗口；0 为自动探测 |

### `[agent]`

| 字段 | 默认值 | 说明 |
|------|--------|------|
| `max_history` | 50 | 请求中最多保留的历史消息数，按完整对话轮次裁剪（SQLite 不删） |
| `auto_confirm` | true | 命令自动确认（false 时危险命令需手动确认） |
| `context_token_limit` | 0 | 0=自动检测，非零=强制覆盖上下文窗口大小 |
| `context_compact_ratio` | 0.75 | 达到输入预算比例时压缩完整旧轮次 |
| `tool_result_keep_rounds` | 3 | 上下文中保留的最近配对工具批次数，0 为全部丢弃 |
| `show_thinking` | true | 是否展示思维链内容；不会持久化或回送 |
| `system_prompt_file` | — | 系统提示词文件路径 |

### `[memory]`

| 字段 | 默认值 | 说明 |
|------|--------|------|
| `backend` | chroma | 向量库后端（chroma/qdrant） |
| `extract_every_n_turns` | 3 | 自动提取情节记忆的间隔轮数 |
| `episodic_top_k` | 3 | 情节记忆语义检索返回条数 |
| `core_max_items` | 20 | 核心记忆参考条目的最大条数 |

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
│   ├── subagent.py           # 低成本子任务执行器
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
