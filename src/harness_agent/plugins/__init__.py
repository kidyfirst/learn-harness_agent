"""Octop / harness-agent plugin system."""

from __future__ import annotations

from langchain.agents.middleware import AgentMiddleware

from harness_agent.plugins.context import PluginContext
from harness_agent.plugins.loader import discover_plugin_dirs, load_all, load_plugin_dir, unload_plugin
from harness_agent.plugins.manifest import PluginManifest
from harness_agent.plugins.registry import LoadedPlugin, PluginRegistry
from harness_agent.plugins.tools import build_plugin_tools, collect_plugin_tool_configs, get_tool_config

__all__ = [
    "AgentMiddleware",
    "LoadedPlugin",
    "PluginContext",
    "PluginManifest",
    "PluginRegistry",
    "build_plugin_tools",
    "collect_plugin_tool_configs",
    "discover_plugin_dirs",
    "get_tool_config",
    "load_all",
    "load_plugin_dir",
    "unload_plugin",
]
