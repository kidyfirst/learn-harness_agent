"""Click CLI entry point: the `harness` command group."""

from __future__ import annotations

from pathlib import Path

from harness_agent.cli import _check_cli_deps

try:
    _check_cli_deps()
except ImportError as exc:
    raise SystemExit(str(exc)) from None

import click

from harness_agent.cli import __version__, load_dotenv_files


@click.group(invoke_without_command=True)
@click.version_option(version=__version__, prog_name="harness-agent-cli")
@click.pass_context
def cli(ctx: click.Context) -> None:
    """Harness Agent CLI — chat with AI agents from your terminal."""
    load_dotenv_files(Path.cwd())
    if ctx.invoked_subcommand is None:
        from harness_agent.cli.commands.chat import chat

        ctx.invoke(chat)


def _register_commands() -> None:
    """Register all subcommands."""
    from harness_agent.cli.commands.agent_cmd import agent
    from harness_agent.cli.commands.chat import chat
    from harness_agent.cli.commands.config_cmd import config
    from harness_agent.cli.commands.init_cmd import init
    from harness_agent.cli.commands.skill import skill
    from harness_agent.cli.commands.update import update

    cli.add_command(agent)
    cli.add_command(chat)
    cli.add_command(config)
    cli.add_command(skill)
    cli.add_command(init)
    cli.add_command(update)


_register_commands()
