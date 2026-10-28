# Project Guidelines

## Project Map

- This is a Python 3.11+ terminal agent. `main.py` owns the interactive CLI and commands; `agent/core.py` owns prompt construction, model calls, tool execution, context management, and memory integration.
- `agent/config.py` loads `config.toml` and environment variables; change defaults there and documented settings in `config.toml` together. The main model handles requests directly; `agent/subagent.py` executes bounded read-only tasks with a configured low-cost model. `agent/skill_registry.py` discovers optional skills from `skills/`.
- `session/sqlite_session.py` persists sessions, messages, token usage, and core memories in SQLite. `memory/manager.py` manages the core-memory cache and optional mem0/Qdrant episodic memory. Keep the two layers' storage and failure behavior distinct.
- `tools/registry.py` imports modules in `tools/` automatically. New tools expose an OpenAI-style `TOOL_DEFINITION` and `execute(args)`; check how `agent/core.py` dispatches tool calls before adding special arguments or state.
- `scripts/vector_daemon.py` serves local embeddings; `generated_scripts/` contains generated/user scripts and prompts, not application modules.

## Development

- Work from the repository root. Use Python 3.11+ (`python3` on macOS): install dependencies with `python3 -m pip install -r requirements.txt`; copy `.env.example` to `.env`, configure a compatible API key, then start with `python3 main.py`. Never commit `.env`, API keys, SQLite databases, or downloaded models.
- The current `config.toml` selects GLM-5.3-Coding as the main model, Qdrant on `localhost:6333`, and a local embedding daemon on port 8000. Configure `SUBAGENT_MODEL` (and `SUBAGENT_BASE_URL` / `SUBAGENT_API_KEY` for another provider) to enable delegation. `Agent` may launch the daemon and wait for its health check; starting the full CLI can download a model, contact external services, and write persistent data. Use isolated paths and mocks for tests.
- Read `README.md` for user-facing commands and `docs/design.md` for the current architecture and known implementation gaps; verify details against code and `config.toml` when changing behavior.
- Keep changes localized to the owning module, preserve CLI commands and persisted SQLite schema compatibility, and avoid reformatting or overwriting existing user changes. Follow the existing Python style and use the standard library where appropriate.

## Verification

- Run `python3 -m unittest discover -s tests` for the focused regression suite; there is no configured lint runner. For Python changes, run a syntax check on affected files and add focused tests for behavior that changes. For storage changes, exercise fresh and existing database schemas with temporary databases; for tool changes, verify registration and executor behavior without invoking unsafe commands.
- Avoid using the interactive CLI as the default automated check: it requires credentials, can block for input or daemon startup, and may mutate local session or vector data.