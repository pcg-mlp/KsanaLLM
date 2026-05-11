# Copyright 2026 Tencent Inc.  All rights reserved.
#
# ==============================================================================
"""ResourceManager: resolve network URLs / base64 data URIs to local temp files.

Moved from python/ksana_plugin/plugin_utils.py so that request-layer resource
normalization is available uniformly to all plugins via
ksana_llm.KsanaPlugin.preprocess, rather than being embedded inside a single
plugin (arc_hunyuan_video).

Behavior is unchanged from the previous location; see
docs/superpowers/specs/2026-04-23-resource-manager-relocation-design.md.
"""

import base64
import copy
import mimetypes
import os
import re
import tempfile
import urllib.request
from typing import Dict, List
from urllib.parse import urlparse


__all__ = ["ResourceManager"]


# Mapping from MIME type to file extension for common media types.
_MIME_TO_EXT = {
    "video/mp4": ".mp4",
    "video/webm": ".webm",
    "video/avi": ".avi",
    "video/quicktime": ".mov",
    "audio/mpeg": ".mp3",
    "audio/mp3": ".mp3",
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/ogg": ".ogg",
    "audio/flac": ".flac",
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/webp": ".webp",
    "image/gif": ".gif",
}

# data URI regex:  data:[<mediatype>][;base64],<data>
_DATA_URI_RE = re.compile(
    r"^data:(?P<mime>[^;,]+)?(?:;base64)?,(?P<data>.+)$", re.DOTALL
)

# Message content types whose nested URL field should be resolved.
# Mapping: content_type -> nested_key_containing_url
_URL_CONTENT_TYPES: Dict[str, str] = {
    "video_url": "video_url",
    "audio_url": "audio_url",
    "image_url": "image_url",
}


def _ext_from_mime(mime: str) -> str:
    """Return a file extension for *mime*, falling back to mimetypes stdlib."""
    ext = _MIME_TO_EXT.get(mime)
    if ext:
        return ext
    ext = mimetypes.guess_extension(mime)
    return ext if ext else ""


def _ext_from_url(url: str) -> str:
    """Best-effort extension extracted from URL path (e.g. .mp4)."""
    path = urlparse(url).path
    _, ext = os.path.splitext(path)
    # Only keep short, sane extensions
    if ext and len(ext) <= 6:
        return ext
    return ""


def _is_loopback(hostname: str) -> bool:
    """Return True if *hostname* resolves to a loopback address."""
    return hostname in ("localhost", "127.0.0.1", "::1") or hostname.startswith("127.")


def _build_proxy_dict_bypassing_loopback() -> dict:
    """Return a proxy dict that uses the environment proxies but adds loopback
    addresses to the no_proxy set so that internal services are never proxied.

    If no proxy environment variables are set, returns ``{}`` (direct connect).
    """
    http_proxy = os.environ.get("http_proxy") or os.environ.get("HTTP_PROXY", "")
    https_proxy = os.environ.get("https_proxy") or os.environ.get("HTTPS_PROXY", "")

    if not http_proxy and not https_proxy:
        return {}  # no proxy in env → direct connection

    # Ensure loopback is in no_proxy
    no_proxy = os.environ.get("no_proxy") or os.environ.get("NO_PROXY", "")
    loopback_entries = {"127.0.0.1", "localhost", "::1"}
    current_entries = {e.strip() for e in no_proxy.split(",") if e.strip()}
    if not loopback_entries.issubset(current_entries):
        merged = ",".join(current_entries | loopback_entries)
        os.environ["no_proxy"] = merged
        os.environ["NO_PROXY"] = merged

    proxies: dict = {}
    if http_proxy:
        proxies["http"] = http_proxy
    if https_proxy:
        proxies["https"] = https_proxy
    return proxies


class ResourceManager:
    """Lightweight manager that resolves remote / base64 resources in OpenAI-style
    messages to local temporary files, and cleans them up on release.

    Typical usage::

        rm = ResourceManager()
        messages = rm.resolve(messages)
        # ... process messages with local paths ...
        # rm cleaned up automatically when reference count drops to zero

    Also supports ``with`` statements and explicit ``cleanup()`` calls.
    """

    def __init__(self, *, timeout: float = 120):
        self._temp_files: List[str] = []
        self._timeout = timeout
        self._cleaned = False

    # -- context manager protocol ------------------------------------------

    def __enter__(self) -> "ResourceManager":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.cleanup()

    def __del__(self) -> None:
        self.cleanup()

    # -- public API --------------------------------------------------------

    def resolve(self, messages: List[Dict]) -> List[Dict]:
        """Return a **shallow copy** of *messages* with remote URLs / base64
        data URIs replaced by local temporary file paths.

        Only non-``None`` URL values are processed.  Local paths are left
        unchanged.
        """
        messages = copy.copy(messages)
        for i, msg in enumerate(messages):
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            # shallow copy message & content so we don't mutate the caller's data
            msg = dict(msg)
            content = list(content)
            msg["content"] = content
            messages[i] = msg

            for j, item in enumerate(content):
                item_type = item.get("type")
                nested_key = _URL_CONTENT_TYPES.get(item_type)
                if nested_key is None:
                    continue
                url_obj = item.get(nested_key)
                if url_obj is None:
                    continue
                url_value = url_obj.get("url") if isinstance(url_obj, dict) else None
                if not url_value:
                    continue
                local_path = self._resolve_url(url_value)
                if local_path != url_value:
                    # shallow copy the item & nested dict
                    item = dict(item)
                    item[nested_key] = dict(url_obj, url=local_path)
                    content[j] = item
        return messages

    def cleanup(self) -> None:
        """Delete all temporary files created by this manager."""
        if self._cleaned:
            return
        self._cleaned = True
        for path in self._temp_files:
            try:
                if os.path.isfile(path):
                    os.remove(path)
            except OSError as exc:
                print(f"[W] ResourceManager: failed to remove {path}: {exc}")
        self._temp_files.clear()

    # -- internal helpers --------------------------------------------------

    def _resolve_url(self, url: str) -> str:
        """Dispatch to the appropriate resolver based on URL scheme."""
        if _DATA_URI_RE.match(url):
            return self._resolve_base64(url)
        parsed = urlparse(url)
        if parsed.scheme in ("http", "https"):
            return self._resolve_http(url)
        # Treat everything else (no scheme, file://, /) as a local path.
        return url

    def _resolve_http(self, url: str) -> str:
        """Download *url* to a temp file and return its path."""
        ext = _ext_from_url(url)
        fd, path = tempfile.mkstemp(suffix=ext, prefix="ksana_res_")
        os.close(fd)
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "KsanaLLM-ResourceManager/1.0"},
            )
            # Build a proxy-aware opener: honour the system no_proxy list but
            # also always bypass the proxy for loopback addresses, which are
            # commonly used for internal encoder / file servers.
            opener = urllib.request.build_opener(
                urllib.request.ProxyHandler(_build_proxy_dict_bypassing_loopback())
            )
            with opener.open(req, timeout=self._timeout) as resp:
                # Try to refine extension from Content-Type if we had none.
                if not ext:
                    ct = resp.headers.get("Content-Type", "")
                    mime = ct.split(";")[0].strip()
                    better_ext = _ext_from_mime(mime)
                    if better_ext:
                        new_path = path + better_ext
                        os.rename(path, new_path)
                        path = new_path
                with open(path, "wb") as f:
                    while True:
                        chunk = resp.read(1 << 20)  # 1 MiB
                        if not chunk:
                            break
                        f.write(chunk)
        except Exception:
            # Clean up on failure and re-raise.
            try:
                os.remove(path)
            except OSError:
                pass
            raise
        self._temp_files.append(path)
        return path

    def _resolve_base64(self, data_uri: str) -> str:
        """Decode a ``data:`` URI to a temp file and return its path."""
        m = _DATA_URI_RE.match(data_uri)
        if not m:
            return data_uri
        mime = m.group("mime") or ""
        raw_data = m.group("data")
        ext = _ext_from_mime(mime)
        fd, path = tempfile.mkstemp(suffix=ext, prefix="ksana_res_")
        try:
            decoded = base64.b64decode(raw_data)
            os.write(fd, decoded)
        except Exception:
            os.close(fd)
            try:
                os.remove(path)
            except OSError:
                pass
            raise
        os.close(fd)
        self._temp_files.append(path)
        return path
