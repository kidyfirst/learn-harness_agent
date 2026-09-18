"""Provider-neutral request and result models for media generation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ImageGenerationRequest:
    """Normalized image-generation request passed to a media provider."""

    prompt: str
    reference_images: tuple[str, ...] = ()
    size: str | None = None
    count: int = 1
    seed: int | None = None
    watermark: bool = False


@dataclass(frozen=True)
class VideoGenerationRequest:
    """Normalized video-generation request passed to a media provider."""

    prompt: str
    first_frame: str | None = None
    last_frame: str | None = None
    reference_images: tuple[str, ...] = ()
    duration: int = 5
    aspect_ratio: str = "16:9"
    resolution: str = "720p"
    generate_audio: bool = False
    seed: int | None = None
    watermark: bool = False


@dataclass(frozen=True)
class GeneratedMedia:
    """One generated binary artifact returned by a media provider."""

    data: bytes
    media_type: str
    extension: str


@dataclass(frozen=True)
class StoredMedia:
    """One generated artifact persisted through ``BackendWorkspace``."""

    path: str
    local_path: str
    media_type: str
    size_bytes: int
