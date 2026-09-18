"""Media-generation provider protocol."""

from __future__ import annotations

from typing import Protocol

from harness_agent.media.models import GeneratedMedia, ImageGenerationRequest, VideoGenerationRequest


class MediaGenerationProvider(Protocol):
    """Adapter contract implemented by image/video generation vendors."""

    @property
    def provider_id(self) -> str:
        """Stable provider identifier used in tool results and logs."""
        ...

    @property
    def image_model(self) -> str:
        """Configured image model identifier."""
        ...

    @property
    def video_model(self) -> str:
        """Configured video model identifier."""
        ...

    async def generate_image(self, request: ImageGenerationRequest) -> list[GeneratedMedia]:
        """Generate one or more image artifacts."""
        ...

    async def generate_video(self, request: VideoGenerationRequest) -> list[GeneratedMedia]:
        """Generate one or more video artifacts."""
        ...
