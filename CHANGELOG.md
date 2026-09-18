# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.10] - 2026-09-17

### Fixed

- Memory recall no longer changes the system prompt. Recall is snapshotted on each
  new user message and replayed on API copies across tool calls, checkpoint resumes,
  and later turns; displayed/captured user content stays clean. Empty results and
  failures are frozen for the turn, and edited messages cannot replay stale recall.
  Automatic and manual compaction token estimates include the persisted recall.
  Regression coverage exercises real SQLite and opt-in PostgreSQL checkpoint
  reopen, interrupted tool resume, recall, and clean raw capture.
- Test workspace initialization now isolates the import-time provider-template path
  as well as runtime home lookups, preventing overwrite tests from touching the
  developer's real provider template.

## [1.0.9] - 2026-09-13

## [1.0.8] - 2026-09-11

### Added

- ``ProviderConfig.session_header``: optional HTTP header name filled with the
  current invocation ``thread_id`` on each LLM request (openai / anthropic).
  Hosts that need per-conversation affinity (e.g. OpenCode Go
  ``x-opencode-session``) set the name; other providers are unchanged.
- Optional ``opensandbox`` backend (``type: "opensandbox"``, extra
  ``orcakit-harness-agent[opensandbox]``). Creates a remote sandbox at resolve
  time and destroys it on ``close()``. E2B is not included yet.
- Peer agents: ``HarnessAgentConfig.team_enabled`` (default ``True``) and
  ``team_peers`` (optional name/id allowlist; ``None`` = all agents owned by
  the current user). ``PeerAgentMiddleware`` mounts ``agent_list`` /
  ``ask_agent`` and injects a live roster plus ``@name`` usage hint (only
  when the latest user message contains ``@agent``, appended as a
  system-prompt suffix via ``append_to_system_message``) when the agent is
  created through ``HarnessAgentManager``. The roster includes each peer's
  description and optional guidance cards (``metadata["quick_prompts"]``).
  Hosts can bind ``TeamManager.bind_peer_enrich`` to refresh that metadata
  on each ``list_peers`` / ``agent_list`` call. ``agent_list`` returns the
  same card titles and short descriptions (no ``prompt`` text).
  ``ask_agent`` continues the callee on a stable thread derived from the
  caller's ``thread_id`` (``{caller}~{callee}``). Hosts may bind
  ``TeamManager.bind_peer_session`` to persist that thread on their
  session table without going through an inbound gateway turn.

### Changed

- ``BootstrapMiddleware`` appends ``BOOTSTRAP.md`` after the compiled
  system prompt instead of replacing it, so the working-directory
  directive (and media-tool policy) survive first-run onboarding.
  Onboarding text stays last so it outranks slash-skill rules.
- The compiled working-directory directive tells the model that ``/`` is
  the backend root, not the workspace, and to prefer relative tool paths.
- Runtime diagnostic logging is process-wide: ``log_dir`` / ``log_level`` /
  rotation live on ``HarnessAgentManager`` (or ``setup_logging``), not
  ``HarnessAgentConfig``. ``config.debug`` does not change the process log
  level. ``[agent=<id>]`` is stamped from ``HarnessAgent.agent_id``.
  ``HarnessAgentConfig(log_dir=…)`` still accepts the old kwargs (ignored)
  so hosts such as Octop do not break; pass ``log_dir`` on the manager.
- ``@mention`` is no longer a pre-invoke intercept. Hosts should put
  ``@Name`` in the user message; the model is instructed to match an
  expert first (``ask_agent``), otherwise use ``task`` for a subagent
  already listed in the system prompt. ``configurable.target_agent_ids``
  is ignored.
- Rename ``TeamMiddleware`` to ``PeerAgentMiddleware``
  (``harness_agent.middleware.peer``). Import aliases remain for one release.
- The compiled system prompt always includes a static ``/name`` skill-invoke
  rule (not a path like ``/root/file``). ``SkillFilterMiddleware`` does not
  mutate the system prompt per turn. Unknown ``/foo`` is ordinary chat.
  Per-turn ``ChatRequest.skills`` remain for API allow-lists; composers
  should put ``/slug`` in the user text instead of that filter.

### Fixed

- Manual ``/compact`` (``acompact_conversation``) now passes
  ``session_id`` to deepagents ``_aoffload_to_backend`` and persists
  ``_summarization_session_id``, matching the auto-summarization path.
  ``CompactResult.display_path`` is a POSIX workspace-relative path (Windows
  / macOS / Linux host paths stripped). Offload failure no longer applies a
  summary. Force compact reuses the graph middleware when present.
- Restore ``HarnessAgent.aappend_messages()`` from the unmerged 1.0.4
  branch so hosts can append canonical, timestamped thread messages
  without invoking the model.
- Restore ``TeamManager.team_tools()`` so hosts that still inject
  ``agent_list`` / ``ask_agent`` via ``config.tools`` (e.g. Octop) keep
  working. Prefer ``PeerAgentMiddleware`` and skip the extra inject to avoid
  duplicate tool names.

## [1.0.4] - 2026-09-03

### Added

- Add ``HarnessAgent.aappend_messages()`` for atomically appending canonical,
  timestamped conversation messages without invoking the model.

## [1.0.3] - 2026-09-02

### Fixed

- Graph recompiles no longer leak a maintenance beat. ``MemoryRuntime.build_middleware``
  replaced ``MemoryMiddleware`` on every recompile (MCP tool injection, subagent
  reload, bootstrap) without shutting the old one down; its ``threading.Timer``
  kept the dropped instance alive and re-armed it forever, so each rebuild added
  a permanent hourly GC + vacuum pass — and an immediate one 1s later. The
  running instance is now reused when nothing it depends on changed, and retired
  properly when it does.
- Idle maintenance ticks log at DEBUG. A pass that reclaims nothing
  (``gc_rows=0 pages=0``) was an unconditional INFO line, so an idle host with N
  agents wrote N lines an hour plus a burst on every restart — on one deployment
  40% of the log. Only a pass that reclaimed something, or failed, logs at INFO.
- Lifecycle GC over a Postgres namespace whose tables were never created is a
  skip, not an hourly ``UndefinedTable`` warning traceback.
- Agents sharing one store (same namespace + DSN) coalesce their reclaim passes
  instead of each running GC over the same rows back to back.
- ``HarnessAgent.close`` / ``aclose`` now defer while ``call`` / ``stream`` /
  ``resume_hitl`` is in flight, releasing once the last one finishes. Hosts
  hot-reload an agent by dropping their reference and closing it (new MCP
  servers, refreshed connector credentials, a changed provider); that used to
  release the memory backend and the LangGraph checkpointer pool under a running
  turn, so the end-of-turn write flush failed on a dead pool
  (``psycopg_pool.PoolClosed``) after the answer had already streamed. Because
  the agent tracks this itself, every caller is covered — including peer
  ``@mention`` calls that reach ``entry.agent.call`` directly. Idle closes are
  unchanged, and ``HarnessAgentManager._rebuild_agent`` now goes through
  ``remove_agent`` so there is a single close path.
- ``HarnessAgentManager.aremove_agent`` awaits ``aclose`` on the calling loop
  instead of running the sync ``close`` through ``asyncio.to_thread``. aiosqlite
  is bound to that loop and the sync-close fallback drives it with
  ``asyncio.run`` from the worker, so the worker waited on a loop that was
  itself blocked waiting for the worker — the hang a host hits when a chat
  reloads its MCP servers mid-turn.
- Hourly maintenance no longer logs a warning traceback when the memory backend
  is not SQLite: lifecycle GC is SQLite-only, so a Postgres backend is recorded
  as a skip. Reclaim still runs vacuum.

### Changed

- Rebuilds reuse the process-shared ``Memory`` (backend + checkpointer pool)
  for the same namespace + SQLite path / Postgres DSN. MCP / policy reloads
  compile a new graph and tools; they no longer open a second store or close
  the one a running turn is still writing to. Last agent close still releases
  the connections.

## [1.0.2] - 2026-09-01

### Added

- Built-in ``ask_user_question`` tool with a typed question schema and a
  respond-only HITL interrupt; disable it for unattended agents with
  ``HarnessAgentConfig.ask_user_enabled=False``.
- ``browser_use(..., action="close_session", kill=True)`` forwards to
  harness-browser so the agent can stop the local Chrome process while
  keeping the on-disk profile.

### Changed

- Refresh Tencent Cloud Token Plan with its current seven canonical models,
  and add separate enterprise-China and Hy Token Plan presets without
  duplicating documented model aliases.
- Split host ``HarnessLocalShellBackend`` into ``backends/local_shell.py``; ``bwrap_shell`` remains the Linux jail and still re-exports the host backend.

### Fixed

- Accept agent-facing rootfs ``workspace_dir`` values (e.g. ``/.octop/workspaces/<id>``) on
  Windows, where they carry no drive letter. ``HarnessAgent`` already mapped them onto the
  backend ``root_dir``; only the config-time absolute-path check rejected them.
- Align scoped local-shell execution with virtual filesystem paths when bwrap is unavailable:
  conservatively translate command and environment paths onto `root_dir`, execute from the
  workspace cwd, and present host mount paths back in agent-facing form. Bubblewrap execution
  now also starts in the virtual workspace.

## [1.0.1] - 2026-08-30

### Added

- Skill catalog summaries expose optional presentation metadata from ``metadata.octop``
  (``label``, ``summary``, ``icon_url``) plus compatible ``emoji`` / ``display_name``
  without changing the identity ``name``.
- Builtin ``SKILL.md`` files include bilingual ``label`` / ``summary`` for host UIs.

## [1.0.0] - 2026-08-28

### Added

- **deepagents 0.7** (`>=0.7,<0.8`): default `todos_enabled=True` preserves `write_todos` for Octop Chat Todo.
- **`ModelAccess`**: protocol-aware model invoke/stream (`protocol` defaults to `langgraph`); obtain via `ChatModelFactory.get` / `get_for`.
- **`ChatModelFactory`**: `resolve_ref`, `resolve_for_turn`, `list_refs`; `get` returns `ModelAccess`; `get_chat_model` escape hatch.
- **Plugin `ctx.model_factory`**: bind after agent build; use `ctx.get_model_access(spec)` or `factory.get_for(spec, get_protocol=ctx.get_protocol)`.
- **`HarnessAgentConfig`**: `todos_enabled`, `role_bindings`; **`ModelConfig.capabilities`**.

### Changed

- **`delete`** tool: FilesystemGuard / SkillFilter coverage; default HITL tools list includes `delete`.
- Cloud/docker backends: `grep` / `agrep` accept `max_count` (deepagents 0.7 protocol).

### Changed (prior unreleased)

- Automatic memory maintenance no longer runs a full compact/VACUUM at startup or after long idle.

### Fixed

- Deepagents conversation-history / media offloads workspace scoping (see 0.9.26 notes).

## [0.9.26] - 2026-08-26

### Changed

- Refresh the Volcengine Ark Coding Plan preset with live-validated model aliases, modalities, and token limits.

## [0.9.25] - 2026-08-23

### Added

- ``HarnessAgentConfig.tools_disabled`` denylist with ``set_tools_disabled`` hot update and ``ToolsFilterMiddleware`` (strip matching tools from each model request; runs after MCP/user middleware and before tool search).

### Changed

- Plugin tools default to **enabled** when a plugin is globally on and the agent omits per-tool ``enabled`` (opt-out via ``enabled: false``).

## [0.9.24] - 2026-08-22

### Fixed

- ``BackendWorkspace.skill_paths()`` / ``memory_paths()`` 在 ``virtual_mode`` 下改为向 deepagents 传入 agent-facing virtual key（``_backend_storage_key``），不再经 ``resolve_path`` 拼 host ``root_dir``。此前 Skills middleware ``backend.ls`` 会对 ``/home/…/.octop/workspaces/…`` 报 ``path_not_found``，系统提示误称无 skills。

### Changed

- Hatch wheel/sdist 固定 ``core-metadata-version = "2.3"``，避免新版元数据导致上传/安装问题。

## [0.9.23] - 2026-08-21

### Fixed

- Context usage 估算补齐 CJK / JSON / content blocks；``/compact`` 立即 shrink snapshot。

### Changed

- ``BubbledLocalShellBackend`` 仅 Linux+bwrap jail；非 jail 用 ``HarnessLocalShellBackend``。
- Idle memory maintenance 改为 ``run_gc`` + ``nudge_vacuum``，不再走会 prune checkpoint 的 ``run_idle_maintenance``。

### Removed

- 回合结束不再 ``prune_checkpoints``（避免截断 transcript）。

## [0.9.22] - 2026-08-20

### Added

- ``HarnessAgentConfig.system_files_path`` optionally relocates system artifacts
  (``skills/``, ``agents/``, ``sessions/``, ``.env``, ``.bootstrapped``,
  checkpoints / memory sqlite) under a workspace-relative prefix such as
  ``.octop/``. Persona markdown (``AGENTS.md``, ``SOUL.md``, …) stays at the
  workspace root; empty prefix keeps the legacy layout. Reads still fall back
  to legacy root ``skills/`` and ``agents/`` when the prefix is set.
- Model presets now carry distinct ``context_window`` and
  ``max_output_tokens`` metadata alongside the prompt-side
  ``max_input_tokens`` cap; legacy one-field configurations remain compatible.
- LangGraph streams now emit one normalized ``usage`` event per model call,
  keyed by ``call_id`` and split into uncached input, cache read, cache write,
  output, and reasoning token buckets.
- Memory middleware runs cheap SQLite slimming on a dedicated timer (1s after start, then hourly) and again when idle extract fires: checkpoint prune, lifecycle GC, incremental vacuum. Legacy ``auto_vacuum=NONE`` files get one ``compact`` on quiet startup / long idle with no size cap (skip and retry later if free disk is below file+WAL). Short idle still only compact ≤200MB. Idle windows also ``wal_checkpoint(TRUNCATE)`` when ``-wal`` is at least 2MB. Periodic full compact is not scheduled. Startup slims are process-wide serialized. ``maintenance_status()`` / ``HarnessAgent.memory_maintenance_status()`` expose queued / pruning / compacting for host UIs.

### Changed

- Scoped virtual backends may take an agent-facing ``workspace_dir``
  (e.g. ``/.octop/workspaces/<id>``); harness maps it onto ``{root_dir}/…`` for
  local persistence, presents the rootfs path in the system prompt, and
  rewrites host mount prefixes out of ``execute`` output. Workspace ``.env``
  for local execute is read through ``BackendWorkspace`` (same as env-file tools).
- Context occupancy keeps the routed model's context-window limit and the
  provider-reported input total. Segment breakdowns remain explicitly
  heuristic instead of being scaled to imply tokenizer-level precision.
- Bundled provider limits are refreshed for current DeepSeek V4, GLM 5.2,
  Kimi K2.5/K2.6/K2.7/K3, MiniMax M2.5/M2.7, MiMo V2/V2.5, and Qwen 3.x
  routes, while provider-specific hosted limits remain independent.
- Explicit per-agent ``max_tokens`` now reserves output capacity from the total
  context window before DeepAgents computes prompt summarization thresholds.
- After a finished chat turn, memory middleware prunes that thread to the latest parent checkpoint and drops completed ``tools:*`` subgraph streams (delayed 2s; skipped on HITL / in-flight). Hourly maintenance only drops subgraphs when the agent is quiet. Requires harness-memory ADR-029.
- Memory extract uses one model: optional ``aux_model``, then the live chat model, then ``default_model``. ``tier`` only changes timeout; light/heavy no longer pick different refs.
- Disabled skills are dropped from the system prompt and blocked on filesystem
  tools plus ``execute``/``bash`` when args name ``skills/<slug>`` or
  ``_builtin_skills/<slug>``. Unrelated paths that only share the slug are not
  blocked.
- Shell, Docker, ACP, and MCP stdio each merge workspace ``.env`` with a
  documented inherit policy (host execute keeps process env; Docker/MCP do not
  copy the full host environment; ``OCTOP_*`` / identity keys stay locked).
- Cloud backends persist binary uploads as base64 envelopes, create listable
  directory markers via ``mkdir_path``, and map relative workspace paths to
  virtual ``/…`` keys instead of host-absolute object keys.
- Require ``harness-memory>=0.9.6`` so recall injection can pass ``session_id``.

## [0.9.21] - 2026-08-18

### Added

- Provider-neutral built-in `generate_image` and `generate_video` tools with a
  Volcengine Ark adapter (Seedream/Seedance), workspace-backed artifact storage,
  WorkBuddy-compatible result envelopes, and deferred loading by default.
- Serializable `MediaGenerationConfig` for media model IDs, Ark connection,
  output directory, polling, and timeout controls.
- Media-generation credentials stay out of serialized configs, and image/video
  tools can be enabled independently.

- Provider-agnostic client-side progressive tool loading. The ordinary
  `tool_search` function works with any tool-calling chat model, persists loaded
  tools per checkpointed thread, exposes deferred tool names and descriptions as
  lightweight references, and keeps real execution on the existing DeepAgents
  ToolNode.
- Serializable `HarnessAgentConfig.tool_search_mode` with `client` (default),
  `native`, and `eager` strategies.
- Provider-native deferred tool loading for OpenAI Responses hosted
  `tool_search` and Anthropic hosted `defer_loading` / `tool_reference`, while
  preserving real tool calls through the existing DeepAgents execution,
  security, and HITL path.
- `ModelConfig.native_tool_search` plus serializable
  `HarnessAgentConfig.deferred_tools`, `defer_mcp_tools`, and
  `tool_search_fallback` controls. Unsupported routed models fall back to
  eager schemas by default or can be rejected in strict mode.

### Changed

- Deferred-tool activation and media provider failures now return structured,
  model-visible tool errors with retry safety and remediation guidance instead
  of terminating the agent stream for expected operational failures.
- Seedream 5 image tools default to a valid 2K output and normalize undersized
  explicit dimensions; generated video audio is opt-in for low-cost model compatibility.
- Minimum provider integration versions now match the native tool-search
  contract: LangChain 1.3.14, langchain-openai 1.2.2, and
  langchain-anthropic 1.5.4.

## [0.9.20] - 2026-08-10

### Changed

- Runtime diagnostic logs default to ``~/.harness-agent/logs`` (not ``{workspace}/logs``). Host apps pass ``log_dir`` (Octop: ``~/.octop/logs``). Lines include ``[agent=<id>]``; session JSONL is unchanged.

- `DockerSandbox`: image ensure/pull (anonymous pull fallback when credential
  helpers fail); named-container reuse; async `als`/`aglob` use mapped sync
  APIs; `execute` honors `command_timeout` (exit 124).
- `HarnessAgent.close` / `aclose` call `backend.close()` when present.
- `spec_supports_execution` treats `docker` like `local_shell`.
- `SecurityPolicy.apply_to_config` keeps `permissions` for execution backends;
  `HarnessAgent` mounts `FilesystemGuardMiddleware` and only passes
  `permissions` to deepagents when the backend is non-execution.
- Dependency `deepagents` tightened to `>=0.6.12,<0.7`.

### Fixed

- Internal conversation-summarization LLM tokens are no longer exposed as user-visible output by LangGraph, OpenAI, or MCP streaming protocols.
- ACP inbound server no longer returns JSON ``null`` for ``initialize`` / ``session/new``: ``_Agent`` now subclasses ``(HarnessACPAgent, Agent)`` so ``acp.Agent`` Protocol stubs do not shadow the real implementation (MRO). Shared via ``_acp_agent_cls`` for ``run_harness_acp_server`` and ``build_harness_acp_agent``.
- Documented ``root_dir`` vs ``workspace_dir`` as different dimensions in ``AGENTS.md`` (backend mount vs agent working directory); L1 agent content I/O goes through ``BackendWorkspace``, while memory databases, logs, JSONL conversations, and checkpoints remain direct local runtime persistence; non-host ``virtual_mode`` shell is bwrap jail or path rewrite + workspace materialize failback.
- When ``virtual_mode`` maps absolute paths onto a non-host ``root_dir``, ``BackendWorkspace`` mkdir/delete/move mutations now consistently resolve through the backend mount; host-rooted backends still keep relative mutations in the workspace.
- ``BackendWorkspace.aexists`` now has the same root-map → original/workspace read failback as sync reads; relative failback rejects workspace escapes, and local ``write_text(force=False)`` no longer overwrites existing L1 content.
- ``BackendWorkspace`` rejects local moves into their own descendants and preserves native async backend uploads in ``aupload_many``.
- ``send_file_to_user`` keeps caller ``path`` unchanged but emits an RFC 8089 ``file://`` URI from the **materialized** local file (avoids ``file://rel`` netloc traps for relative paths).
- Shell virtual-path rewrite maps colon-separated abs values (``/a:/b``) segment-wise; preserves ``https://`` URLs; quotes bare paths when ``root_dir`` contains spaces; ``BubbledLocalShellBackend`` documents that non-bwrap rewrite is alignment, not isolation.
- ``BubbledLocalShellBackend`` bwrap jail adds PID/IPC/UTS isolation (not network); falls back to path rewrite when ``bwrap`` fails to start.
- ``BackendWorkspace`` read/exists no longer let stale workspace files shadow remote backends (S3/COS) without a local mount.
- ``HarnessAgent.stream`` holds ``logging_scope`` only while fetching each chunk so agent log tags do not leak across ``yield``.
- `SecurityPolicy.apply_to_config` no longer instantiates a backend just to test shell capability; it reads the spec via `spec_supports_execution`, removing one backend build (and one host-root warning) per `create_agent` / `rebuild_all_agents`.
- `HarnessAgentManager.create_agent(init_workspace=True)` only recompiles the graph when seeding actually wrote templates, skills, or subagent manifests — an already-seeded workspace no longer pays for a second compile.
- Long-thread turns no longer fail on strict OpenAI-compatible backends with `System message must be at the beginning`. `BootstrapMiddleware.before_agent` was appending `SystemMessage` into checkpoint state via LangGraph's `add_messages` reducer; after onboarding completed those residuals sat mid-list. Bootstrap now injects `BOOTSTRAP.md` only as `request.system_message` (never into conversation `messages`).
- `SummarizationMiddleware` fraction thresholds (85% trigger / 10% keep) now follow the turn's `configurable["model"]` via a turn-aware seed `profile`, matching UI model overrides instead of staying stuck on the compile-time default window.
- Host-rooted backends (`root_dir="/"`, including `DEFAULT_BACKEND_SPEC`) resolved with a `workspace_dir` are wrapped in `MountedCompositeBackend` so deepagents conversation history, media, and large tool-result offloads land under the workspace instead of the often-unwritable host root.
- `ModelConfig.from_dict` accepts provider-catalog `context_window` as an alias for `max_input_tokens` when the latter is missing.
- Context-window usage stamps now use the routed request model's `max_input_tokens` (not only the agent default / 128k fallback).
- `/compact` success copy clarifies that the chat UI keeps full history while model context is compacted.

- HITL no longer asks for approval on filesystem tools when `SecurityPolicy`
  filesystem deny rules already block the path — interrupt `when` skips those
  calls so `FilesystemGuard` / deepagents permissions return permission denied
  directly.

### Added

- `HarnessAgent.acompact_conversation` / `force_compact_thread` — force one SummarizationMiddleware cycle (offload + summary + `_summarization_event`) without waiting for the auto trigger or starting a new thread; uses an aggressive keep policy and the turn-selected summary model, reusing the graph middleware's backend when present.
- `TurnAwareProfile` / `install_turn_aware_profile` — dynamic seed-model profile for deepagents summarization limits (shared turn model-ref rules with ModelRouter).
- `resolve_turn_model_ref` / `max_input_tokens_from_model` — shared turn model helpers used by router, profile, and context-usage stamping.
- `spec_supports_execution(spec)` — predict whether a backend spec resolves to a shell-capable backend without building it.
- `MountedCompositeBackend` — composite backend that exposes the default backend's local mount as `cwd` so `BackendWorkspace` mkdir/move/delete keep working.
- Context-window usage estimation: `ContextUsageMiddleware`, `HarnessAgent.aget_context_usage`, and `HarnessAgentManager.get_context_usage` (segment breakdown persisted on AI messages).
- Subagent catalog `emoji` field with default `🤖` when frontmatter omits it.
- Optional dependency aggregates: `object-storage`, `web-search-all`, and `all` (library feature extras only; excludes `cli` — use `[cli,all]` when needed).
- `ModelPreset.reasoning` — optional JSON boolean marking reasoning / extended-thinking models; serialized for dashboards via `serialize_model_preset`.

- `ProviderPreset.vendor` / `vendor_name` / `variant` — optional JSON fields to group multi-site provider presets (one `id` per site).
- `serialize_provider_preset` / `serialize_model_preset` — dashboard/API serialization with `input` and vendor aliases (`provider_group`, `provider_variant`).
- Bundled `provider_template.json` multi-site rows for Kimi, MiniMax, Zhipu, Aliyun, Volcengine, and SiliconFlow; Tencent presets gain `vendor` grouping.
- `ModelPreset.input` / `is_multimodal` — provider template JSON `input` modalities are parsed and propagated through env detection and the CLI setup wizard into `ModelConfig`.
- Multimodal `input: ["text", "image"]` annotations on bundled preset models.
- `@register("model")` for the REPL `/model` slash command.

- `DockerSandbox` backend (`type: "docker"`, optional `[docker]` extra):
  container FS I/O via Docker SDK; default in-container workspace root mirrors
  host `workspace_dir` (same absolute path, **not** bind-mounted); host path
  remains sessions/memory/checkpoints. Naming via `sandbox_scope` /
  `sandbox_prefix`; optional `volumes`; `previewable` defaults true only
  for `fixed` (host UI browse flag). `destroy()` removes the container.
- `FilesystemGuardMiddleware` — enforce `SecurityPolicy` filesystem deny rules
  on execution backends where deepagents rejects `permissions`.
- `ModelSettingsMiddleware` — merge `configurable.model_settings` /
  `configurable.max_input_tokens` into each model call.

### Changed

- `HarnessAgent.list_subagent_summaries` / `subagents.catalog.list_subagent_summaries` are now `async` (same shape as `list_skill_summaries`); callers must `await`.
- Package version is static in `pyproject.toml` `[project].version` (runtime via `importlib.metadata`); dropped hatch-vcs generated `_version.py`.
- Bump version to `0.9.15`.
- Manual `/compact` (`acompact_conversation`) keeps only the last 6 messages (vs auto ~10% fraction), requires enough history to summarize ≥2 messages, and always generates the summary with the turn-selected model while reusing the graph middleware backend for workspace-scoped offload.
- Forced compaction runs history offload and summary LLM concurrently (after optional inline-media offload), matching deepagents auto-summarization.

### Removed

- Legacy single-site preset ids `moonshot`, `minimax`, `zhipu`, `dashscope`, `volces`, and `silicon` from the bundled template (replaced by vendor-grouped site ids such as `kimi-cn`, `minimax-cn`, `siliconflow-cn`).
- Bundled `web-search` built-in skill (en/zh). Use optional `web_search_tools` (`tavily_search`, `searchfree_search`, etc.) instead.

### Fixed

- Recall injection now passes an overridden `ModelRequest` instead of assigning to the read-only `system_prompt` property.
- MCP tool args: omit nullables from LLM-facing JSON Schema and strip explicit `null`/`None` before the MCP call (servers such as Notion Zod reject null for optional fields).
- Import `harness_memory.llm` instead of the non-existent `harness_memory.ports.llm` (fixes pytest collection errors).
- Provider template `input` modalities are validated against `text` / `image` / `audio` / `video`.
- REPL `/model` is routed only via `slash_router` (removed duplicate `COMMANDS` registration).
- Test harness closes `HarnessAgent` instances explicitly; dropped broad SQLite unraisable-warning ignore.
- `HarnessAgent.aclose()` for async teardown (`async with` / running event loop); `HarnessAgentManager.aclose()` delegates to it.
- Close memory SQLite backends in `MemoryRuntime.close()` / `HarnessAgent.close()` to reduce connection leaks in tests.
- Use `inspect.iscoroutinefunction` in MCP tool wrapping (Python 3.14 compatibility).
- Ignore pytest unraisable warnings for SQLite GC teardown on Python 3.14.

## [0.9.18] - 2026-08-01

### 修复

- 将依赖 `mcp` 限制为 `>=1.27.1,<2`，避免 mcp 2.x 移除 `RequestContext` 导致与 `langchain-mcp-adapters` 不兼容、安装验证失败
