"""Workspace-aware orchestration for provider-neutral media generation."""

from __future__ import annotations

import asyncio
import base64
import mimetypes
from datetime import UTC, datetime
from pathlib import PurePosixPath
from urllib.parse import urlsplit
from uuid import uuid4

from harness_agent.backends.workspace import BackendWorkspace
from harness_agent.media.models import (
    GeneratedMedia,
    ImageGenerationRequest,
    StoredMedia,
    VideoGenerationRequest,
)
from harness_agent.media.provider import MediaGenerationProvider


class MediaGenerationService:
    """Normalize references, call a provider, and persist generated artifacts."""

    def __init__(
        self,
        provider: MediaGenerationProvider,
        workspace: BackendWorkspace,
        *,
        output_dir: str,
    ) -> None:
        self._provider = provider
        self._workspace = workspace
        self._output_dir = output_dir.strip("/")

    async def generate_image(
        self,
        *,
        prompt: str,
        reference_images: list[str] | None = None,
        size: str | None = None,
        count: int = 1,
        seed: int | None = None,
        watermark: bool = False,
    ) -> dict[str, object]:
        prompt = prompt.strip()
        if not prompt:
            raise ValueError("prompt must not be empty")
        if not 1 <= count <= 4:
            raise ValueError("count must be between 1 and 4")
        references = await self._normalize_references(reference_images or [])
        request = ImageGenerationRequest(
            prompt=prompt,
            reference_images=tuple(references),
            size=size,
            count=count,
            seed=seed,
            watermark=watermark,
        )
        generated = await self._provider.generate_image(request)
        stored = await self._store(generated, kind="images", prefix="image")
        return {
            "schema_version": 1,
            "type": "image_gen_tool_result",
            "tool": "generate_image",
            "prompt": prompt,
            "status": "completed",
            "is_error": False,
            "message": f"Generated {len(stored)} image artifact(s).",
            "provider": self._provider.provider_id,
            "model": self._provider.image_model,
            "error": None,
            "remediation": None,
            "execution": {
                "provider": self._provider.provider_id,
                "model": self._provider.image_model,
                "fallback_used": False,
            },
            "images": [_stored_payload(item) for item in stored],
        }

    async def generate_video(
        self,
        *,
        prompt: str,
        first_frame: str | None = None,
        last_frame: str | None = None,
        reference_images: list[str] | None = None,
        duration: int = 5,
        aspect_ratio: str = "16:9",
        resolution: str = "720p",
        generate_audio: bool = False,
        seed: int | None = None,
        watermark: bool = False,
    ) -> dict[str, object]:
        prompt = prompt.strip()
        if not prompt:
            raise ValueError("prompt must not be empty")
        if not 4 <= duration <= 15:
            raise ValueError("duration must be between 4 and 15 seconds")
        normalized_first = await self._normalize_optional_reference(first_frame)
        normalized_last = await self._normalize_optional_reference(last_frame)
        references = await self._normalize_references(reference_images or [])
        request = VideoGenerationRequest(
            prompt=prompt,
            first_frame=normalized_first,
            last_frame=normalized_last,
            reference_images=tuple(references),
            duration=duration,
            aspect_ratio=aspect_ratio,
            resolution=resolution,
            generate_audio=generate_audio,
            seed=seed,
            watermark=watermark,
        )
        generated = await self._provider.generate_video(request)
        stored = await self._store(generated, kind="videos", prefix="video")
        return {
            "schema_version": 1,
            "type": "video_gen_tool_result",
            "tool": "generate_video",
            "prompt": prompt,
            "status": "completed",
            "is_error": False,
            "message": f"Generated {len(stored)} video artifact(s).",
            "provider": self._provider.provider_id,
            "model": self._provider.video_model,
            "error": None,
            "remediation": None,
            "execution": {
                "provider": self._provider.provider_id,
                "model": self._provider.video_model,
                "fallback_used": False,
            },
            "videos": [_stored_payload(item) for item in stored],
        }

    async def _normalize_optional_reference(self, reference: str | None) -> str | None:
        if reference is None:
            return None
        normalized = await self._normalize_references([reference])
        return normalized[0]

    async def _normalize_references(self, references: list[str]) -> list[str]:
        normalized: list[str] = []
        for reference in references:
            raw = reference.strip()
            if not raw:
                raise ValueError("reference image paths and URLs must not be empty")
            if urlsplit(raw).scheme in {"http", "https", "data"}:
                normalized.append(raw)
                continue
            data = await self._workspace.adownload_bytes(raw)
            if data is None:
                raise FileNotFoundError(f"reference image not found: {raw}")
            media_type, _ = mimetypes.guess_type(raw)
            if media_type is None or not media_type.startswith("image/"):
                media_type = "image/png"
            encoded = base64.b64encode(data).decode("ascii")
            normalized.append(f"data:{media_type};base64,{encoded}")
        return normalized

    async def _store(
        self,
        generated: list[GeneratedMedia],
        *,
        kind: str,
        prefix: str,
    ) -> list[StoredMedia]:
        if not generated:
            raise RuntimeError("media provider returned no artifacts")
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        stored: list[StoredMedia] = []
        for artifact in generated:
            extension = artifact.extension.strip(".") or "bin"
            filename = f"{prefix}-{timestamp}-{uuid4().hex[:12]}.{extension}"
            path = str(PurePosixPath(self._output_dir) / kind / filename)
            await self._workspace.aupload_bytes(path, artifact.data)
            local = await asyncio.to_thread(self._workspace.materialize_local, path)
            stored.append(
                StoredMedia(
                    path=path,
                    local_path=str(local) if local is not None else path,
                    media_type=artifact.media_type,
                    size_bytes=len(artifact.data),
                )
            )
        return stored


def _stored_payload(item: StoredMedia) -> dict[str, object]:
    return {
        "localPath": item.local_path,
        "path": item.path,
        "mediaType": item.media_type,
        "sizeBytes": item.size_bytes,
    }
