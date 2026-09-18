"""Slash command registry and dispatch for the REPL."""

from __future__ import annotations

import uuid
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Console

if TYPE_CHECKING:
    from harness_agent.cli.repl.state import SessionState
    from harness_agent.cli.ui.theme import Theme

# Type alias for command handlers
CommandHandler = Callable[["SessionState", Console, "Theme", str], None]

# Registry: command name → handler function
COMMANDS: dict[str, CommandHandler] = {}

# Alias map: alias → canonical name
_ALIASES: dict[str, str] = {}


def register(name: str, aliases: list[str] | None = None) -> Callable[[CommandHandler], CommandHandler]:
    """Decorator to register a slash command."""

    def decorator(fn: CommandHandler) -> CommandHandler:
        COMMANDS[name] = fn
        if aliases:
            for alias in aliases:
                _ALIASES[alias] = name
        return fn

    return decorator


def dispatch_command(text: str, state: SessionState, console: Console, theme: Theme) -> None:
    """Parse and dispatch a slash command string."""
    parts = text.split(maxsplit=1)
    cmd_name = parts[0].lstrip("/").lower()
    args = parts[1].strip() if len(parts) > 1 else ""

    # Resolve alias
    canonical = _ALIASES.get(cmd_name, cmd_name)
    handler = COMMANDS.get(canonical)

    if handler:
        handler(state, console, theme, args)
    else:
        console.print(f"  [{theme.dot_error}]●[/] Unknown command: /{cmd_name}. Type /help")


# ---------------------------------------------------------------------------
# Built-in commands
# ---------------------------------------------------------------------------


@register("help")
def cmd_help(state: SessionState, console: Console, theme: Theme, args: str) -> None:
    """Show available commands."""
    from harness_agent.cli.repl.slash_router import runtime_command_names

    runtime = sorted(runtime_command_names())
    help_text = (
        "  Commands:\n"
        "  [dim]Agent (runtime):[/]\n"
        f"    {', '.join(f'/{n}' for n in runtime if n not in ('cancel', 'models'))}\n"
        "  [dim]CLI (host):[/]\n"
        "    /help              Show this help\n"
        "    /exit              Exit the session (aliases: /quit, /q)\n"
        "    /new               Start a new session\n"
        "    /clear             Clear the terminal\n"
        "    /agent list        List agent profiles\n"
        "    /agent create <n>  Create a new agent profile\n"
        "    /agent remove <n>  Remove an agent profile\n"
        "    /agent switch <n>  Switch to a different agent\n"
        "    /agent current     Show current agent\n"
        "    /config            Show current config\n"
        "    /env set K V       Set env var (saved to ~/.harness-agent/.env)\n"
        "    /env get K         Show env var value\n"
        "    /env list          List vars in ~/.harness-agent/.env\n"
        "    /env unset K       Remove env var\n"
        "    /env reload        Reload .env into current session\n"
        "\n"
        "  Keys: Enter=send, Escape+Enter=newline, Ctrl+C=stop/quit"
    )
    console.print(f"  [{theme.dot_info}]●[/] {help_text}", highlight=False)


@register("exit", aliases=["quit", "q"])
def cmd_exit(state: SessionState, console: Console, theme: Theme, args: str) -> None:
    """Exit the REPL."""
    raise SystemExit(0)


@register("new")
def cmd_new(state: SessionState, console: Console, theme: Theme, args: str) -> None:
    """Start a new session."""
    state.session_id = uuid.uuid4().hex
    state.input_tokens = 0
    state.output_tokens = 0
    state.last_elapsed = 0.0
    console.print(f"  [{theme.dot_info}]●[/] New session: {state.session_id[:8]}...")


@register("clear")
def cmd_clear(state: SessionState, console: Console, theme: Theme, args: str) -> None:
    """Clear the terminal."""
    console.clear()


@register("config")
def cmd_config(state: SessionState, console: Console, theme: Theme, args: str) -> None:
    """Show current agent configuration."""
    import json

    formatted = json.dumps(state.agent_cfg, indent=2, ensure_ascii=False)
    console.print(f"  [{theme.dot_info}]●[/] Config:\n{formatted}")


@register("agent")
def cmd_agent(state: SessionState, console: Console, theme: Theme, args: str) -> None:
    """Agent management: /agent list | /agent switch <name> | /agent current"""
    parts = args.split(maxsplit=1)
    subcmd = parts[0] if parts else ""
    subargs = parts[1].strip() if len(parts) > 1 else ""

    if subcmd == "list":
        from harness_agent.cli.agents.manager import CliAgentManager
        from harness_agent.cli.config.paths import CliPaths

        paths = CliPaths(project_dir=Path.cwd())
        mgr = CliAgentManager(paths.global_config_file)
        agents = mgr.list()
        default_name = mgr.default_agent_name
        lines = []
        for a in agents:
            marker = " ●" if a.name == state.current_agent else ""
            default_marker = " (default)" if a.name == default_name else ""
            model_info = f"{a.provider}/{a.model}" if a.provider and a.model else "(inherit)"
            lines.append(f"    {a.name}{marker}{default_marker} — {model_info}")
        console.print(f"  [{theme.dot_info}]●[/] Agents:\n" + "\n".join(lines))

    elif subcmd == "switch" and subargs:
        from harness_agent.cli.agents.manager import CliAgentManager
        from harness_agent.cli.config.paths import CliPaths

        paths = CliPaths(project_dir=Path.cwd())
        mgr = CliAgentManager(paths.global_config_file)
        profile = mgr.get(subargs)
        if profile is None:
            console.print(f"  [{theme.dot_error}]●[/] Agent '{subargs}' not found. Use /agent list")
            return
        state._switch_to = subargs  # type: ignore[attr-defined]
        console.print(f"  [{theme.dot_info}]●[/] Switching to agent: {subargs}")

    elif subcmd == "create":
        from rich.prompt import Prompt

        from harness_agent.cli.agents.manager import CliAgentManager
        from harness_agent.cli.agents.profile import AgentProfile
        from harness_agent.cli.config.paths import CliPaths

        # Name: first token of subargs, or prompt for it
        tokens = subargs.split() if subargs else []
        name = tokens[0] if tokens else Prompt.ask("  Agent name")
        if not name:
            console.print(f"  [{theme.dot_error}]●[/] Name is required.")
            return

        paths = CliPaths(project_dir=Path.cwd())
        mgr = CliAgentManager(paths.global_config_file)

        # Check if name already exists
        if mgr.get(name) is not None:
            console.print(f"  [{theme.dot_error}]●[/] Agent '{name}' already exists.")
            return

        # Interactive: pick provider from configured providers
        providers_cfg = state.agent_cfg.get("providers", {})
        provider_names = list(providers_cfg.keys())

        if not provider_names:
            console.print(f"  [{theme.dot_error}]●[/] No providers configured. Run setup wizard first.")
            return

        # Select provider
        console.print()
        for i, pname in enumerate(provider_names, 1):
            console.print(f"    [cyan]{i:>3}[/] {pname}")
        console.print()
        provider_choice = Prompt.ask("  Select provider", default="1")
        try:
            provider_idx = int(provider_choice) - 1
            if not (0 <= provider_idx < len(provider_names)):
                raise ValueError
            provider_key = provider_names[provider_idx]
        except (ValueError, IndexError):
            console.print(f"  [{theme.dot_error}]●[/] Invalid selection.")
            return

        # Select model from that provider
        provider_cfg = providers_cfg[provider_key]
        models = provider_cfg.get("models", [])
        model_ids = [m["id"] if isinstance(m, dict) else m.id for m in models]

        if len(model_ids) == 1:
            model_id = model_ids[0]
        elif model_ids:
            console.print()
            for i, mid in enumerate(model_ids, 1):
                console.print(f"    [cyan]{i:>3}[/] {mid}")
            console.print()
            model_choice = Prompt.ask("  Select model", default="1")
            try:
                model_idx = int(model_choice) - 1
                if not (0 <= model_idx < len(model_ids)):
                    raise ValueError
                model_id = model_ids[model_idx]
            except (ValueError, IndexError):
                console.print(f"  [{theme.dot_error}]●[/] Invalid selection.")
                return
        else:
            model_id = Prompt.ask("  Model ID")
            if not model_id:
                console.print(f"  [{theme.dot_error}]●[/] Model ID is required.")
                return

        # Optional: workspace_dir
        workspace_dir = Prompt.ask("  Workspace dir", default="(inherit)") or None
        if workspace_dir == "(inherit)":
            workspace_dir = None

        profile = AgentProfile(
            name=name,
            provider=provider_key,
            model=model_id,
            workspace_dir=workspace_dir,
        )
        try:
            mgr.create(profile)
            console.print(f"\n  [{theme.dot_info}]●[/] Agent '{name}' created: {provider_key}/{model_id}")
        except ValueError as exc:
            console.print(f"  [{theme.dot_error}]●[/] {exc}")

    elif subcmd == "remove" and subargs:
        from harness_agent.cli.agents.manager import CliAgentManager
        from harness_agent.cli.config.paths import CliPaths

        paths = CliPaths(project_dir=Path.cwd())
        mgr = CliAgentManager(paths.global_config_file)
        try:
            mgr.remove(subargs)
            console.print(f"  [{theme.dot_info}]●[/] Agent '{subargs}' removed.")
        except ValueError as exc:
            console.print(f"  [{theme.dot_error}]●[/] {exc}")

    elif subcmd in ("current", ""):
        console.print(
            f"  [{theme.dot_info}]●[/] Current agent: {state.current_agent} ({state.model})",
            highlight=False,
        )

    else:
        console.print(
            f"  [{theme.dot_info}]●[/] Usage: /agent list | /agent create <name> [--provider X --model Y] "
            "| /agent remove <name> | /agent switch <name> | /agent current"
        )


# ---------------------------------------------------------------------------
# /env command
# ---------------------------------------------------------------------------


def _mask_secret(value: str) -> str:
    """Mask a value if it looks like a secret (len>12, mixed alnum).

    Returns the masked string or the original if short/not secret-like.
    """
    if len(value) <= 12:
        return value
    has_digit = any(c.isdigit() for c in value)
    has_alpha = any(c.isalpha() for c in value)
    if has_digit and has_alpha:
        return f"{value[:4]}{'••••'}{value[-4:]}"
    return value


@register("env")
def cmd_env(state: SessionState, console: Console, theme: Theme, args: str) -> None:
    """Manage environment variables: /env set KEY val | get KEY | list | unset KEY | reload"""
    parts = args.split(maxsplit=2)
    subcmd = parts[0] if parts else ""
    rest = parts[1:] if len(parts) > 1 else []

    global_env_file = Path.home() / ".harness-agent" / ".env"

    if subcmd == "set":
        if len(rest) < 2:
            console.print(
                f"  [{theme.dot_error}]●[/] Usage: /env set KEY value",
                highlight=False,
            )
            return
        import os

        from dotenv import set_key

        key, value = rest[0], rest[1]
        global_env_file.parent.mkdir(parents=True, exist_ok=True)
        set_key(str(global_env_file), key, value)
        os.environ[key] = value
        console.print(
            f"  [{theme.dot_info}]●[/] {key} saved to ~/.harness-agent/.env",
            highlight=False,
        )

    elif subcmd == "get":
        if not rest:
            console.print(f"  [{theme.dot_error}]●[/] Usage: /env get KEY", highlight=False)
            return
        import os

        key = rest[0]
        val = os.environ.get(key)
        if val is None:
            console.print(f"  [{theme.dot_info}]●[/] {key} is not set", highlight=False)
        else:
            console.print(f"  [{theme.dot_info}]●[/] {key}={val}", highlight=False)

    elif subcmd == "list":
        if not global_env_file.is_file():
            console.print(
                f"  [{theme.dot_info}]●[/] No .env found at ~/.harness-agent/.env",
                highlight=False,
            )
            return
        from dotenv import dotenv_values
        from rich.table import Table

        entries = dotenv_values(str(global_env_file))
        if not entries:
            console.print(
                f"  [{theme.dot_info}]●[/] ~/.harness-agent/.env is empty",
                highlight=False,
            )
            return
        table = Table(show_header=True, header_style="bold", box=None, padding=(0, 2))
        table.add_column("Key", style="cyan")
        table.add_column("Value")
        for k, v in entries.items():
            table.add_row(k, _mask_secret(v or ""))
        console.print(table)

    elif subcmd == "unset":
        if not rest:
            console.print(f"  [{theme.dot_error}]●[/] Usage: /env unset KEY", highlight=False)
            return
        import os

        from dotenv import unset_key

        key = rest[0]
        if global_env_file.is_file():
            unset_key(str(global_env_file), key)
        os.environ.pop(key, None)
        console.print(
            f"  [{theme.dot_info}]●[/] {key} removed from ~/.harness-agent/.env",
            highlight=False,
        )

    elif subcmd == "reload":
        from harness_agent.cli import load_dotenv_files

        loaded = load_dotenv_files(Path.cwd(), override=True)
        if loaded:
            console.print(
                f"  [{theme.dot_info}]●[/] Reloaded .env from {loaded}",
                highlight=False,
            )
        else:
            console.print(
                f"  [{theme.dot_info}]●[/] No .env file found to reload",
                highlight=False,
            )

    else:
        console.print(
            f"  [{theme.dot_info}]●[/] Usage: /env set KEY value | /env get KEY "
            "| /env list | /env unset KEY | /env reload",
            highlight=False,
        )
