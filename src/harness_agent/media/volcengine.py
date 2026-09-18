"""Volcengine Ark adapter for Seedream image and Seedance video generation."""

from __future__ import annotations

import asyncio
import base64
import re
import time
from collections.abc import Mapping
from typing import Any, cast
from urllib.parse import urlsplit

import httpx

from harness_agent.config import MediaGenerationConfig
from harness_agent.media.errors import MediaGenerationError
from harness_agent.media.models import GeneratedMedia, ImageGenerationRequest, VideoGenerationRequest

_SEEDREAM_5_MIN_PIXELS = 3_686_400
_EXPLICIT_IMAGE_SIZE = re.compile(r"^(\d+)[xX](\d+)$")


def _seedream_image_size(model: str, requested: str | None) -> str | None:
    """Return an Ark-compatible size, upgrading undersized Seedream 5 requests."""
    if not model.startswith("doubao-seedream-5-0"):
        return requested
    size = (requested or "2K").strip()
    match = _EXPLICIT_IMAGE_SIZE.fullmatch(size)
    if match and int(match.group(1)) * int(match.group(2)) < _SEEDREAM_5_MIN_PIXELS:
        return "2K"
    return size


class VolcengineMediaProvider:
    """Call the public Volcengine Ark Seedream and Seedance HTTP APIs."""

    def __init__(
        self,
        config: MediaGenerationConfig,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._config = config
        self._client = client

    @property
    def provider_id(self) -> str:
        return "volcengine"

    @property
    def image_model(self) -> str:
        return self._config.image_model

    @property
    def video_model(self) -> str:
        return self._config.video_model

    async def generate_image(self, request: ImageGenerationRequest) -> list[GeneratedMedia]:
        if self._client is not None:
            return await self._generate_image(self._client, request)
        async with httpx.AsyncClient(timeout=self._config.request_timeout_seconds) as client:
            return await self._generate_image(client, request)

    async def generate_video(self, request: VideoGenerationRequest) -> list[GeneratedMedia]:
        if self._client is not None:
            return await self._generate_video(self._client, request)
        async with httpx.AsyncClient(timeout=self._config.request_timeout_seconds) as client:
            return await self._generate_video(client, request)

    async def _generate_image(
        self,
        client: httpx.AsyncClient,
        request: ImageGenerationRequest,
    ) -> list[GeneratedMedia]:
        body: dict[str, Any] = {
            "model": self.image_model,
            "prompt": request.prompt,
            "response_format": "b64_json",
            "watermark": request.watermark,
            "sequential_image_generation": "auto" if request.count > 1 else "disabled",
        }
        if request.reference_images:
            body["image"] = list(request.reference_images)
        size = _seedream_image_size(self.image_model, request.size)
        if size:
            body["size"] = size
        if request.seed is not None:
            body["seed"] = request.seed
        if request.count > 1:
            body["sequential_image_generation_options"] = {"max_images": request.count}

        payload = await self._request_json(
            client,
            "POST",
            "/images/generations",
            json=body,
        )
        error = payload.get("error")
        if isinstance(error, Mapping) and error:
            raise _provider_payload_error(error, fallback="image generation failed")
        rows = payload.get("data")
        if not isinstance(rows, list) or not rows:
            raise MediaGenerationError("Volcengine image generation returned no images")

        generated: list[GeneratedMedia] = []
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            artifact = await self._image_from_row(client, row)
            if artifact is not None:
                generated.append(artifact)

        if not generated:
            raise MediaGenerationError("Volcengine image generation returned no usable images")
        return generated

    async def _image_from_row(
        self,
        client: httpx.AsyncClient,
        row: Mapping[object, object],
    ) -> GeneratedMedia | None:
        encoded = row.get("b64_json")
        if isinstance(encoded, str) and encoded:
            try:
                data = base64.b64decode(encoded, validate=True)
            except ValueError as exc:
                raise MediaGenerationError("Volcengine returned invalid base64 image data") from exc
        else:
            url = row.get("url")
            if not isinstance(url, str) or not url:
                return None
            data = await self._download(client, url)
        media_type, extension = _detect_media(data, fallback_type="image/png", fallback_extension="png")
        return GeneratedMedia(data=data, media_type=media_type, extension=extension)

    async def _generate_video(
        self,
        client: httpx.AsyncClient,
        request: VideoGenerationRequest,
    ) -> list[GeneratedMedia]:
        content: list[dict[str, Any]] = [{"type": "text", "text": request.prompt}]
        if request.first_frame:
            content.append(_image_content(request.first_frame, role="first_frame"))
        if request.last_frame:
            content.append(_image_content(request.last_frame, role="last_frame"))
        content.extend(_image_content(image, role="reference_image") for image in request.reference_images)

        body: dict[str, Any] = {
            "model": self.video_model,
            "content": content,
            "duration": request.duration,
            "ratio": request.aspect_ratio,
            "resolution": request.resolution,
            "generate_audio": request.generate_audio,
            "watermark": request.watermark,
            "output_format": "mp4",
        }
        if request.seed is not None:
            body["seed"] = request.seed

        created = await self._request_json(
            client,
            "POST",
            "/contents/generations/tasks",
            json=body,
        )
        task_id = created.get("id")
        if not isinstance(task_id, str) or not task_id:
            raise MediaGenerationError("Volcengine video generation did not return a task id")

        deadline = time.monotonic() + self._config.video_timeout_seconds
        try:
            while True:
                task = await self._request_json(
                    client,
                    "GET",
                    f"/contents/generations/tasks/{task_id}",
                    provider_task_id=task_id,
                )
                status = task.get("status")
                if status == "succeeded":
                    break
                if status in {"failed", "cancelled"}:
                    error = task.get("error")
                    provider_error = _provider_payload_error(
                        error,
                        fallback=f"task {task_id} {status}",
                        provider_task_id=task_id,
                    )
                    if status == "cancelled":
                        provider_error.code = "generation_cancelled"
                    raise provider_error
                if time.monotonic() >= deadline:
                    await self._cancel_task(client, task_id)
                    raise MediaGenerationError(
                        f"Video generation timed out after {self._config.video_timeout_seconds:g} seconds",
                        code="provider_timeout",
                        retryable=True,
                        safe_to_resubmit=False,
                        remediation="check_task_status",
                        model_instruction=(
                            "Do not submit another video. Tell the user the provider task timed out "
                            "and its final state could not be confirmed."
                        ),
                        provider_task_id=task_id,
                    )
                await asyncio.sleep(self._config.video_poll_interval_seconds)
        except asyncio.CancelledError:
            await asyncio.shield(self._cancel_task(client, task_id))
            raise

        task_content = task.get("content")
        if not isinstance(task_content, Mapping):
            raise MediaGenerationError("Volcengine video task completed without content")
        raw_url = task_content.get("video_url") or task_content.get("file_url")
        if not isinstance(raw_url, str) or not raw_url:
            raise MediaGenerationError("Volcengine video task completed without a download URL")
        data = await self._download(client, raw_url)
        media_type, extension = _detect_media(data, fallback_type="video/mp4", fallback_extension="mp4")
        return [GeneratedMedia(data=data, media_type=media_type, extension=extension)]

    async def _request_json(
        self,
        client: httpx.AsyncClient,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        provider_task_id: str | None = None,
    ) -> dict[str, Any]:
        try:
            api_key = self._config.resolve_api_key()
        except ValueError as exc:
            raise MediaGenerationError(
                "Media generation credentials are not configured.",
                code="credential_missing",
                category="configuration",
                remediation="configure_credentials",
                model_instruction=("Do not retry. Tell the user to configure media-generation credentials."),
            ) from exc
        try:
            response = await client.request(
                method,
                f"{self._config.base_url.rstrip('/')}{path}",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=json,
            )
        except httpx.TimeoutException as exc:
            raise MediaGenerationError(
                "The media provider request timed out.",
                code="provider_timeout",
                retryable=True,
                safe_to_resubmit=False,
                remediation="retry_later",
                model_instruction=(
                    "Do not retry this tool call manually because the provider may have accepted it. "
                    "Tell the user the request timed out."
                ),
                provider_task_id=provider_task_id,
            ) from exc
        except httpx.RequestError as exc:
            raise MediaGenerationError(
                "The media provider could not be reached.",
                code="provider_unavailable",
                retryable=True,
                safe_to_resubmit=False,
                remediation="retry_later",
                model_instruction=(
                    "Do not retry this tool call manually. Tell the user the provider is temporarily unavailable."
                ),
                provider_task_id=provider_task_id,
            ) from exc
        if response.is_error:
            raise _response_error(response, provider_task_id=provider_task_id)
        try:
            payload = response.json()
        except ValueError as exc:
            raise MediaGenerationError(
                "The media provider returned invalid JSON.",
                code="provider_response_invalid",
            ) from exc
        if not isinstance(payload, dict):
            raise MediaGenerationError("Volcengine API returned an unexpected response")
        return cast(dict[str, Any], payload)

    async def _cancel_task(self, client: httpx.AsyncClient, task_id: str) -> None:
        """Best-effort cancellation for a task that this call will no longer poll."""
        try:
            await client.delete(
                f"{self._config.base_url.rstrip('/')}/contents/generations/tasks/{task_id}",
                headers={"Authorization": f"Bearer {self._config.resolve_api_key()}"},
            )
        except httpx.HTTPError:
            return

    @staticmethod
    async def _download(client: httpx.AsyncClient, url: str) -> bytes:
        if urlsplit(url).scheme not in {"http", "https"}:
            raise MediaGenerationError("Volcengine returned an unsupported download URL")
        # Deliberately omit the Ark Authorization header: generated artifacts
        # are commonly hosted on a different TOS origin.
        try:
            response = await client.get(url, follow_redirects=True)
        except httpx.TimeoutException as exc:
            raise MediaGenerationError(
                "Generated media download timed out.",
                code="provider_timeout",
                retryable=True,
                safe_to_resubmit=False,
                remediation="retry_download",
                model_instruction=("Do not regenerate the media. Tell the user the result download timed out."),
            ) from exc
        except httpx.RequestError as exc:
            raise MediaGenerationError(
                "Generated media could not be downloaded.",
                code="provider_unavailable",
                retryable=True,
                safe_to_resubmit=False,
                remediation="retry_download",
                model_instruction=(
                    "Do not regenerate the media. Tell the user the result download is temporarily unavailable."
                ),
            ) from exc
        if response.is_error:
            raise MediaGenerationError(
                f"Generated media download failed with HTTP {response.status_code}.",
                code="download_failed",
                retryable=response.status_code >= 500,
                safe_to_resubmit=False,
                remediation="retry_download",
                model_instruction=(
                    "Do not regenerate the media. Tell the user to retry downloading the existing result."
                ),
                http_status=response.status_code,
            )
        if not response.content:
            raise MediaGenerationError(
                "Generated media download returned an empty file.",
                code="download_failed",
                safe_to_resubmit=False,
                remediation="retry_download",
                model_instruction=(
                    "Do not regenerate the media. Tell the user that the provider returned an empty file."
                ),
            )
        return response.content


def _image_content(url: str, *, role: str) -> dict[str, Any]:
    return {"type": "image_url", "image_url": {"url": url}, "role": role}


def _provider_error_message(value: object, *, fallback: str) -> str:
    if isinstance(value, Mapping):
        code = value.get("code")
        message = value.get("message")
        parts = [str(item) for item in (code, message) if item]
        if parts:
            return ": ".join(parts)
    return fallback


def _provider_payload_error(
    value: object,
    *,
    fallback: str,
    provider_task_id: str | None = None,
) -> MediaGenerationError:
    message = _provider_error_message(value, fallback=fallback)
    provider_code: str | None = None
    if isinstance(value, Mapping) and value.get("code"):
        provider_code = str(value["code"])
    return _classified_provider_error(
        message,
        provider_code=provider_code,
        provider_task_id=provider_task_id,
    )


def _classified_provider_error(  # noqa: PLR0911 - explicit categories keep remediation auditable
    message: str,
    *,
    provider_code: str | None = None,
    provider_task_id: str | None = None,
    http_status: int | None = None,
    retry_after_seconds: float | None = None,
) -> MediaGenerationError:
    haystack = f"{provider_code or ''} {message}".lower()
    if http_status == 401 or any(token in haystack for token in ("invalid api key", "unauthorized")):
        return MediaGenerationError(
            "The media provider rejected the configured credentials.",
            code="credential_invalid",
            category="authentication",
            remediation="configure_credentials",
            model_instruction="Do not retry. Tell the user to check or replace the provider API key.",
            provider_code=provider_code,
            provider_task_id=provider_task_id,
            http_status=http_status,
            retry_after_seconds=retry_after_seconds,
        )
    if any(token in haystack for token in ("unsafe", "safety", "content policy", "moderation")):
        return MediaGenerationError(
            message,
            code="content_rejected",
            category="safety",
            remediation="adjust_prompt",
            model_instruction=("Do not switch providers to bypass this rejection. Ask the user to revise the request."),
            provider_code=provider_code,
            provider_task_id=provider_task_id,
            http_status=http_status,
            retry_after_seconds=retry_after_seconds,
        )
    if http_status == 429 or any(token in haystack for token in ("rate limit", "too many requests")):
        return MediaGenerationError(
            "The media provider is rate limited.",
            code="rate_limited",
            category="rate_limit",
            retryable=True,
            safe_to_resubmit=provider_task_id is None,
            remediation="retry_later",
            model_instruction=(
                "Do not retry this tool call manually. Tell the user to retry later or use a configured fallback."
            ),
            provider_code=provider_code,
            provider_task_id=provider_task_id,
            http_status=http_status,
            retry_after_seconds=retry_after_seconds,
        )
    if any(token in haystack for token in ("insufficient", "balance", "quota")):
        return MediaGenerationError(
            message,
            code="quota_exhausted",
            category="quota",
            remediation="check_billing",
            model_instruction="Do not retry. Tell the user to check provider quota or billing.",
            provider_code=provider_code,
            provider_task_id=provider_task_id,
            http_status=http_status,
            retry_after_seconds=retry_after_seconds,
        )
    if http_status in {403, 404} or any(
        token in haystack for token in ("model not found", "model access", "not activated", "permission")
    ):
        return MediaGenerationError(
            message,
            code="model_access_required" if http_status == 403 else "model_not_found",
            category="authorization",
            remediation="configure_model",
            model_instruction=(
                "Do not retry. Tell the user to enable the configured model or select another available model."
            ),
            provider_code=provider_code,
            provider_task_id=provider_task_id,
            http_status=http_status,
            retry_after_seconds=retry_after_seconds,
        )
    if http_status is not None and http_status >= 500:
        return MediaGenerationError(
            "The media provider is temporarily unavailable.",
            code="provider_unavailable",
            retryable=True,
            safe_to_resubmit=provider_task_id is None,
            remediation="retry_later",
            model_instruction=(
                "Do not retry this tool call manually. Tell the user the provider is temporarily unavailable."
            ),
            provider_code=provider_code,
            provider_task_id=provider_task_id,
            http_status=http_status,
            retry_after_seconds=retry_after_seconds,
        )
    return MediaGenerationError(
        message,
        provider_code=provider_code,
        provider_task_id=provider_task_id,
        http_status=http_status,
        retry_after_seconds=retry_after_seconds,
    )


def _response_error(
    response: httpx.Response,
    *,
    provider_task_id: str | None,
) -> MediaGenerationError:
    detail = _response_error_message(response)
    provider_code: str | None = None
    try:
        payload = response.json()
    except ValueError:
        payload = None
    if isinstance(payload, Mapping) and isinstance(payload.get("error"), Mapping):
        raw_code = payload["error"].get("code")
        if raw_code:
            provider_code = str(raw_code)
    retry_after: float | None = None
    raw_retry_after = response.headers.get("retry-after")
    if raw_retry_after:
        try:
            retry_after = float(raw_retry_after)
        except ValueError:
            retry_after = None
    return _classified_provider_error(
        detail,
        provider_code=provider_code,
        provider_task_id=provider_task_id,
        http_status=response.status_code,
        retry_after_seconds=retry_after,
    )


def _response_error_message(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text[:500] or response.reason_phrase
    if isinstance(payload, Mapping):
        error = payload.get("error")
        if error:
            return _provider_error_message(error, fallback=str(error))
    return str(payload)[:500]


def _detect_media(data: bytes, *, fallback_type: str, fallback_extension: str) -> tuple[str, str]:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png", "png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg", "jpg"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif", "gif"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "image/webp", "webp"
    if len(data) >= 12 and data[4:8] == b"ftyp":
        return "video/mp4", "mp4"
    return fallback_type, fallback_extension
