"""Environment variable detection for provider configuration."""

from __future__ import annotations

import os
import re
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from harness_agent.config import ModelConfig, ProtocolName, ProviderConfig
    from harness_agent.providers import ModelPreset, ProviderPreset

_HARNESS_KEY_PATTERN = re.compile(r"^HARNESS_PROVIDER_([A-Z0-9]+)_API_KEY$")


def _model_config_from_preset(model: ModelPreset) -> ModelConfig:
    """Build a :class:`ModelConfig` from a bundled provider preset model."""
    from harness_agent.config import InputModality, ModelConfig

    return ModelConfig(
        id=model.id,
        name=model.name,
        input=[cast(InputModality, modality) for modality in model.input],
        max_input_tokens=model.max_input_tokens,
        context_window=model.effective_context_window,
        max_output_tokens=model.max_output_tokens,
    )


def _build_provider(
    provider_id: str,
    api_key: str,
    base_url: str,
    protocol: str = "openai",
    models: list[ModelConfig] | None = None,
) -> ProviderConfig:
    """Construct a ProviderConfig from raw values."""
    from harness_agent.config import ProviderConfig

    valid_protocols = ("openai", "anthropic", "bedrock")
    safe_protocol = cast("ProtocolName", protocol if protocol in valid_protocols else "openai")
    return ProviderConfig(
        id=provider_id,
        api_key=api_key,
        base_url=base_url,
        protocol=safe_protocol,
        models=list(models or []),
    )


def _load_template_presets() -> list[ProviderPreset]:
    """Load provider presets from builtin JSON template. Returns [] on failure."""
    try:
        from harness_agent.backends.utils import DEFENSIVE_OP_ERRORS
        from harness_agent.providers import load_provider_templates

        return load_provider_templates()
    except DEFENSIVE_OP_ERRORS:
        return []


def _find_preset(presets: list[ProviderPreset], provider_id: str) -> ProviderPreset | None:
    for preset in presets:
        if preset.id == provider_id:
            return preset
    return None


def detect_providers_from_env() -> tuple[list[ProviderConfig], str | None]:
    """Scan environment variables and return (providers_list, default_model_ref)."""
    providers_dict: dict[str, ProviderConfig] = {}
    presets = _load_template_presets()

    for key, value in os.environ.items():
        m = _HARNESS_KEY_PATTERN.match(key)
        if not m or not value:
            continue
        raw_name = m.group(1)
        name = raw_name.lower()

        base_url = os.environ.get(f"HARNESS_PROVIDER_{raw_name}_BASE_URL", "")
        protocol = os.environ.get(f"HARNESS_PROVIDER_{raw_name}_PROTOCOL", "openai")

        preset = _find_preset(presets, name)
        if not base_url and preset is not None:
            base_url = preset.base_url
            if not protocol or protocol == "openai":
                protocol = preset.protocol

        if not base_url:
            continue

        providers_dict[name] = _build_provider(name, value, base_url, protocol)

    openai_key = os.environ.get("OPENAI_API_KEY", "")
    if openai_key and "openai" not in providers_dict:
        base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        openai_preset = _find_preset(presets, "openai")
        openai_models = (
            [_model_config_from_preset(m) for m in openai_preset.models] if openai_preset is not None else None
        )
        providers_dict["openai"] = _build_provider("openai", openai_key, base_url, "openai", models=openai_models)

    for preset in presets:
        if not preset.api_key_env or preset.id in providers_dict:
            continue
        key_val = os.environ.get(preset.api_key_env, "")
        if not key_val:
            continue
        providers_dict[preset.id] = _build_provider(
            provider_id=preset.id,
            api_key=key_val,
            base_url=preset.base_url,
            protocol=preset.protocol,
            models=[_model_config_from_preset(m) for m in preset.models],
        )

    providers: list[ProviderConfig] = list(providers_dict.values())

    default_model: str | None = os.environ.get("HARNESS_DEFAULT_MODEL")

    if default_model is None:
        openai_model_name = os.environ.get("OPENAI_MODEL_NAME")
        if openai_model_name and "openai" in providers_dict:
            default_model = f"openai/{openai_model_name}"
            provider = providers_dict["openai"]
            if not any(m.id == openai_model_name for m in provider.models):
                from harness_agent.config import ModelConfig

                provider.models = [
                    *provider.models,
                    ModelConfig(id=openai_model_name, name=openai_model_name),
                ]

    if default_model is None and len(providers) == 1:
        sole = providers[0]
        if sole.models:
            default_model = f"{sole.id}/{sole.models[0].id}"

    return providers, default_model


__all__ = ["detect_providers_from_env"]
