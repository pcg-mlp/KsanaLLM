#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LLM_kernels 跨机器编译产物缓存（单文件入口）。

只打包可重定位的静态库与清单，不缓存整棵 build 树（含绝对路径，换机/换目录
不可复用）。主工程先完整 ``add_subdirectory(LLM_kernels)``（由其自行 init/
patch），结束后再算 key / try-download；HIT 则用包内 .a 替换 STATIC 产物并
跳过 NVCC，MISS 则正常编译，编完再 pack/upload。Triton cubin 与 .o 不进包。

产物布局（按 cache_key 隔离）::

    ${LLM_KERNELS_CACHE_DIR}/${cache_key}/
      lib/*.a              libllm_kernels*、prefill_kernels 等静态库
      generated_md5/       构建期 *.md5sums（可选）
      manifest.json        static/interface 目标表、key_fields、bundle_sha256
                           （侧车权威；包内 manifest 无法自指 sha）
      bundle.tar.zst       上述内容的压缩包，供远程 PUT/GET

cache_key 算法::

    fields = compute_cache_key_fields()   # 不含 cache_key 自身
    key_material = json.dumps(fields, sort_keys=True, separators=(',',':'),
                              default=str)
    cache_key = sha256(key_material).hexdigest()

fields 由三块输入拼成（任一变化即得到新 key，旧远程包 miss）:

  * kernel_src — 仅 ``3rdparty/LLM_kernels/csrc/kernels``。对
    ``git ls-files -s`` 的 path+blob 排序拼接，再叠 dirty 指纹
    （``git status --porcelain``；有文件则 hash 内容，否则用状态码）。
    无扩展名过滤：CMakeLists、头文件、脚本等凡 git 跟踪的都计入。
  * deps — c3x/c4x cutlass、FlashMLA、half 的工作区 HEAD + dirty，以及
    ``cmake/flashinfer.cmake`` 中的 GIT_TAG。主工程在 ``add_subdirectory``
    （LLM_kernels 自行 init/patch）之后再 try-download，保证与编完 upload 一致。
  * toolchain — SM、由 SM 解析的 HAS_SM*、固定优先级探测的 CUDA/nvcc、
    torch、g++、CMAKE_BUILD_TYPE、WITH_TESTING、cxx_std。

不进入 fields、因此不影响 key 的包括：主工程 ``csrc/`` ``cmake/`` ``python/``
等非上述路径、本缓存脚本自身、``DISABLE_GENERATE_TRITON_KERNEL``。

用法::

    export LLM_KERNELS_REMOTE_CACHE=1            # 默认关；=1 完整 cmake 后再 try-download
    export LLM_KERNELS_REMOTE_CACHE_UPLOAD=1     # 默认关；=1 源码编完 try-upload
    export LLM_KERNELS_CACHE_MIRROR_BASE=...     # 必填（远程）
    export LLM_KERNELS_CACHE_REPO=...            # 必填（远程）
    export LLM_KERNELS_CACHE_USER=...            # 必填（远程）
    export LLM_KERNELS_CACHE_TOKEN=...           # 必填（远程）
    # 以下可不设：
    # LLM_KERNELS_CACHE_DIR            默认 ~/.cache/KsanaLLM/llm_kernels_build_cache
    # LLM_KERNELS_CACHE_REMOTE_PREFIX  默认空（URL 无额外 path 段）
    # LLM_KERNELS_CACHE_INSECURE_SSL   默认 auto（校验失败再降级）；可强制 1/0
    # LLM_KERNELS_CACHE_BUNDLE_NAME    默认 bundle.tar.zst
    # LLM_KERNELS_CACHE_MANIFEST_NAME  默认 manifest.json
    # SM / CMAKE_BUILD_TYPE 由 cmake 注入 try-download/upload，不必手动 export

子命令见 --help；环境变量说明亦见 help-env。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import ssl
import subprocess
import sys
import tarfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

SCRIPT_PATH = Path(__file__).resolve()
SOURCE_ROOT = SCRIPT_PATH.parent.parent

# urllib 网络与 SSL 相关失败（用于 insecure 降级，勿用裸 Exception）
_HTTP_NET_EXCS = (
    urllib.error.URLError,
    urllib.error.HTTPError,
    subprocess.CalledProcessError,
    TimeoutError,
    OSError,
    ssl.SSLError,
)
# http_download 会把 HTTPError 包成 RuntimeError；download() miss 路径一并捕获
_DOWNLOAD_MISS_EXCS = _HTTP_NET_EXCS + (RuntimeError,)


# =============================================================================
# 基础工具
# =============================================================================


def _run(cmd: List[str], cwd: Optional[Path] = None, check: bool = True) -> str:
    result = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        check=check,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.stdout.strip()


def _sha256_hex(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _log(msg: str) -> None:
    # 走 stderr：try-download 的 stdout 只能有 export 行，才能被 shell eval
    print(f"[llm_kernels_cache] {msg}", file=sys.stderr)


def _env_truthy(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).upper() in ("1", "ON", "TRUE", "YES")


# =============================================================================
# Cache key
#
# fields → json.dumps(sort_keys, 紧凑分隔) → sha256 → cache_key。
# 增删 fields 定义会使全网旧包失效，属预期。
#
# kernel_src_fingerprint
#   cwd=3rdparty/LLM_kernels；git ls-files -s -- csrc/kernels
#   每行取 blob+path，排序后 \\n 拼接，再 +"\\nDIRTY\\n"+dirty_fingerprint。
#   dirty：git status --porcelain -uall -- csrc/kernels；有文件 sha256 内容，
#   无文件用状态码两字符；空列表则 dirty_fingerprint="clean"。
#
# deps_fingerprint
#   子模块工作区 HEAD + dirty（c3x/c4x cutlass、FlashMLA、half）+ flashinfer
#   GIT_TAG。主工程 add_subdirectory(LLM_kernels) 完成后再 try-download，与
#   编完 upload 共用同一工作区；勿只读 gitlink。dirty 忽略 egg-info 等构建噪声。
#
# sm / HAS_SM* / cuda_version / torch_version / cxx_compiler /
# cmake_build_type / with_testing / cxx_std
#   工具链与配置；nvcc 按 CMAKE_CUDA_COMPILER → CUDA_NVCC_EXECUTABLE →
#   CUDA_HOME|CUDA_PATH → /usr/local/cuda → PATH 的固定顺序探测。
# =============================================================================


def _kernel_dirty_fingerprint(llm_kernels_dir: Path) -> Dict[str, Any]:
    """csrc/kernels 工作区脏文件指纹，避免未提交改动命中干净树远程包。"""
    try:
        porcelain = _run(
            ["git", "status", "--porcelain", "-uall", "--", "csrc/kernels"],
            cwd=llm_kernels_dir,
        )
    except subprocess.CalledProcessError:
        return {"dirty": False, "fingerprint": "clean", "entries": []}

    entries: List[str] = []
    for line in porcelain.splitlines():
        if not line or len(line) < 4:
            continue
        path_part = line[3:]
        if " -> " in path_part:
            path_part = path_part.split(" -> ", 1)[1]
        rel = path_part.strip()
        if not rel:
            continue
        abs_path = llm_kernels_dir / rel
        if abs_path.is_file():
            entries.append(f"{rel}\0{_sha256_file(abs_path)}")
        else:
            entries.append(f"{rel}\0{line[:2]}")
    entries.sort()
    if not entries:
        return {"dirty": False, "fingerprint": "clean", "entries": []}
    return {
        "dirty": True,
        "fingerprint": _sha256_hex("\n".join(entries)),
        "entries": [e.split("\0", 1)[0] for e in entries],
    }


def _kernel_src_fingerprint(llm_kernels_dir: Path) -> Dict[str, Any]:
    """用 index blob（ls-files -s）做干净树指纹，不用 HEAD:path。"""
    files = _run(["git", "ls-files", "--", "csrc/kernels"], cwd=llm_kernels_dir).splitlines()
    files = [f for f in files if f]
    if not files:
        raise RuntimeError(f"no git-indexed files under csrc/kernels in {llm_kernels_dir}")
    indexed = _run(["git", "ls-files", "-s", "--", "csrc/kernels"], cwd=llm_kernels_dir).splitlines()
    pairs: List[str] = []
    for line in sorted(indexed):
        parts = line.split(None, 3)
        if len(parts) < 4:
            continue
        blob, rel_path = parts[1], parts[3]
        pairs.append(f"{rel_path}\0{blob}")
    dirty = _kernel_dirty_fingerprint(llm_kernels_dir)
    digest = _sha256_hex("\n".join(pairs) + "\nDIRTY\n" + dirty["fingerprint"])
    return {
        "fingerprint": digest,
        "file_count": len(files),
        "dirty": dirty["dirty"],
        "dirty_fingerprint": dirty["fingerprint"],
        "dirty_files": dirty["entries"],
    }


def _submodule_commit(llm_kernels_dir: Path, rel_path: str) -> str:
    """读取子模块工作区 HEAD；未检出返回 missing。

    空目录上 ``git rev-parse HEAD`` 会向上解析到主仓，必须以 .git 存在为准。
    正常路径下 try-download 在 add_subdirectory 之后，此处应为真实 HEAD。
    """
    path = llm_kernels_dir / rel_path
    if not path.is_dir() or not (path / ".git").exists():
        return "missing"
    try:
        return _run(["git", "rev-parse", "HEAD"], cwd=path)
    except subprocess.CalledProcessError:
        return "missing"


def _submodule_dirty_fingerprint(llm_kernels_dir: Path, rel_path: str) -> str:
    """子模块工作区脏文件指纹；干净为 clean。用于检出后的本地改动进 key。

    忽略构建噪声（egg-info / __pycache__ 等），避免 pip install -e 导致跨机 key 漂移。
    """
    path = llm_kernels_dir / rel_path
    if not path.is_dir() or not (path / ".git").exists():
        return "missing"
    try:
        porcelain = _run(["git", "status", "--porcelain", "-uall"], cwd=path)
    except subprocess.CalledProcessError:
        return "missing"
    ignore_substrings = (
        ".egg-info/",
        "__pycache__/",
        ".pytest_cache/",
        ".mypy_cache/",
        ".pyc",
    )
    entries: List[str] = []
    for line in porcelain.splitlines():
        if not line or len(line) < 4:
            continue
        path_part = line[3:]
        if " -> " in path_part:
            path_part = path_part.split(" -> ", 1)[1]
        rel = path_part.strip().strip('"')
        if not rel:
            continue
        if any(s in rel for s in ignore_substrings):
            continue
        abs_path = path / rel
        if abs_path.is_file():
            entries.append(f"{rel}\0{_sha256_file(abs_path)}")
        else:
            entries.append(f"{rel}\0{line[:2]}")
    entries.sort()
    if not entries:
        return "clean"
    return _sha256_hex("\n".join(entries))


def _deps_fingerprint(llm_kernels_dir: Path) -> Dict[str, Any]:
    """deps：工作区子模块 HEAD + dirty + flashinfer GIT_TAG（非仅 gitlink）。"""
    submodules = {
        "c3x_cutlass": "3rdparty/c3x/cutlass",
        "c4x_cutlass": "3rdparty/c4x/cutlass",
        "FlashMLA": "3rdparty/FlashMLA",
        "half": "3rdparty/half",
    }
    commits = {name: _submodule_commit(llm_kernels_dir, path) for name, path in submodules.items()}
    dirty = {
        name: _submodule_dirty_fingerprint(llm_kernels_dir, path)
        for name, path in submodules.items()
    }
    flashinfer_tag = "unknown"
    flashinfer_cmake = llm_kernels_dir / "cmake" / "flashinfer.cmake"
    if flashinfer_cmake.is_file():
        match = re.search(r"GIT_TAG\s+([0-9a-f]{7,40})", flashinfer_cmake.read_text(encoding="utf-8"))
        if match:
            flashinfer_tag = match.group(1)
    payload = json.dumps(
        {
            "submodules": commits,
            "submodules_dirty": dirty,
            "flashinfer_git_tag": flashinfer_tag,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return {
        "fingerprint": _sha256_hex(payload),
        "submodules": commits,
        "submodules_dirty": dirty,
        "flashinfer_git_tag": flashinfer_tag,
    }


def _tool_version(cmd: List[str]) -> str:
    try:
        out = _run(cmd)
        return out.splitlines()[0] if out else "unknown"
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _nvcc_candidates() -> List[str]:
    # CMAKE_CUDA_COMPILER → CUDA_NVCC_EXECUTABLE → $CUDA_HOME|/CUDA_PATH/bin/nvcc
    # → /usr/local/cuda/bin/nvcc → which nvcc
    candidates: List[str] = []
    seen: Set[str] = set()

    def add(path: Optional[str]) -> None:
        if not path:
            return
        normalized = os.path.expanduser(path)
        if normalized not in seen:
            seen.add(normalized)
            candidates.append(normalized)

    add(os.environ.get("CMAKE_CUDA_COMPILER"))
    add(os.environ.get("CUDA_NVCC_EXECUTABLE"))
    for env_name in ("CUDA_HOME", "CUDA_PATH"):
        root = os.environ.get(env_name)
        if root:
            add(os.path.join(root, "bin", "nvcc"))
    add("/usr/local/cuda/bin/nvcc")
    add(shutil.which("nvcc"))
    return candidates or ["nvcc"]


def _cuda_version() -> str:
    version_re = re.compile(r"release\s+([0-9.]+)", re.IGNORECASE)
    for nvcc in _nvcc_candidates():
        if os.path.basename(nvcc) != nvcc and not os.path.isfile(nvcc):
            continue
        try:
            out = _run([nvcc, "--version"])
            match = version_re.search(out)
            if match:
                return match.group(1)
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue
    return "unknown"


def _torch_version(python: str) -> str:
    try:
        return _run([python, "-c", "import torch; print(torch.__version__)"])
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _detect_sm(sm_arg: Optional[str], repo_root: Path) -> str:
    # 参数 > 环境变量 SM > tools/get_nvidia_gpu_properties.py > unknown
    if sm_arg:
        return sm_arg
    env_sm = os.environ.get("SM")
    if env_sm:
        return env_sm
    script = repo_root / "tools" / "get_nvidia_gpu_properties.py"
    if script.is_file():
        try:
            return _run([sys.executable, str(script)], cwd=repo_root)
        except subprocess.CalledProcessError:
            pass
    return "unknown"


def _has_sm_flags(sm: str) -> Dict[str, bool]:
    # 与 LLM_kernels 按 arch 裁剪编译对齐；单独成字段便于 key --json 对照
    sm_list = [s.strip() for s in sm.split(",") if s.strip()]
    return {
        "HAS_SM80": "80" in sm_list,
        "HAS_SM86": "86" in sm_list,
        "HAS_SM89": "89" in sm_list,
        "HAS_SM90": "90" in sm_list,
        "HAS_SM90A": "90a" in sm_list,
    }


def compute_cache_key_fields(
    source_root: Path,
    sm: Optional[str] = None,
    build_type: Optional[str] = None,
    with_testing: Optional[bool] = None,
    cxx_std: str = "17",
    python: Optional[str] = None,
) -> Dict[str, Any]:
    """收集计入 hash 的 fields，并写入 fields['cache_key']。

    序列化：json.dumps(..., sort_keys=True, separators=(',',':'), default=str)，
    再 sha256。返回值同时写入侧车 manifest.key_fields，供 ``key --json`` 对照。

    计入 hash 的键（cache_key 本身不参与）：
      kernel_src_fingerprint, kernel_src_file_count,
      kernel_src_dirty, kernel_src_dirty_fingerprint,
      deps_fingerprint, deps_submodules, deps_submodules_dirty, flashinfer_git_tag,
      sm, cuda_version, torch_version, cxx_compiler,
      cmake_build_type, with_testing, cxx_std,
      HAS_SM80, HAS_SM86, HAS_SM89, HAS_SM90, HAS_SM90A
    """
    llm_kernels_dir = source_root / "3rdparty" / "LLM_kernels"
    if not llm_kernels_dir.is_dir():
        raise FileNotFoundError(f"LLM_kernels not found: {llm_kernels_dir}")

    python_bin = python or os.environ.get("PYTHON_PATH", "python3")
    sm_value = _detect_sm(sm, source_root)
    build_type_value = build_type or os.environ.get("CMAKE_BUILD_TYPE", "Release")
    with_testing_value = (
        with_testing
        if with_testing is not None
        else _env_truthy("WITH_TESTING", "OFF")
    )

    kernel_src = _kernel_src_fingerprint(llm_kernels_dir)
    deps = _deps_fingerprint(llm_kernels_dir)
    fields: Dict[str, Any] = {
        "kernel_src_fingerprint": kernel_src["fingerprint"],
        "kernel_src_file_count": kernel_src["file_count"],
        "kernel_src_dirty": kernel_src["dirty"],
        "kernel_src_dirty_fingerprint": kernel_src["dirty_fingerprint"],
        "deps_fingerprint": deps["fingerprint"],
        "deps_submodules": deps["submodules"],
        "deps_submodules_dirty": deps["submodules_dirty"],
        "flashinfer_git_tag": deps["flashinfer_git_tag"],
        "sm": sm_value,
        "cuda_version": _cuda_version(),
        "torch_version": _torch_version(python_bin),
        "cxx_compiler": _tool_version(["g++", "--version"]),
        "cmake_build_type": build_type_value,
        "with_testing": with_testing_value,
        "cxx_std": cxx_std,
        **_has_sm_flags(sm_value),
    }
    key_material = json.dumps(fields, sort_keys=True, separators=(",", ":"), default=str)
    fields["cache_key"] = _sha256_hex(key_material)
    return fields


# =============================================================================
# 路径 / 远程 URL
# =============================================================================


def default_cache_dir() -> Path:
    return Path(os.environ.get("LLM_KERNELS_CACHE_DIR") or Path.home() / ".cache/KsanaLLM/llm_kernels_build_cache")


def key_dir(cache_key: str) -> Path:
    return default_cache_dir() / cache_key


def bundle_name() -> str:
    return os.environ.get("LLM_KERNELS_CACHE_BUNDLE_NAME", "bundle.tar.zst")


def manifest_name() -> str:
    return os.environ.get("LLM_KERNELS_CACHE_MANIFEST_NAME", "manifest.json")


def require_remote_env() -> None:
    missing = [
        v
        for v in (
            "LLM_KERNELS_CACHE_MIRROR_BASE",
            "LLM_KERNELS_CACHE_REPO",
            "LLM_KERNELS_CACHE_USER",
            "LLM_KERNELS_CACHE_TOKEN",
        )
        if not os.environ.get(v)
    ]
    if "LLM_KERNELS_CACHE_DIR" not in os.environ:
        os.environ["LLM_KERNELS_CACHE_DIR"] = str(default_cache_dir())
    if missing:
        raise SystemExit(f"[llm_kernels_cache] ERROR: env not set: {', '.join(missing)}")


def remote_url(cache_key: str, filename: str) -> str:
    base = os.environ["LLM_KERNELS_CACHE_MIRROR_BASE"].rstrip("/")
    repo = os.environ["LLM_KERNELS_CACHE_REPO"]
    prefix = os.environ.get("LLM_KERNELS_CACHE_REMOTE_PREFIX", "").strip("/")
    parts = [base, repo]
    if prefix:
        parts.append(prefix)
    parts.extend([cache_key, filename])
    url = "/".join(parts)
    return re.sub(r"(?<!:)/{2,}", "/", url)


def _ssl_insecure_mode() -> str:
    """返回 force_on / force_off / auto。

    默认 auto：先校验证书，遇 SSL/CA 失败再自动降级并告警。
    显式设 LLM_KERNELS_CACHE_INSECURE_SSL=1/0 可强制开/关。
    """
    raw = os.environ.get("LLM_KERNELS_CACHE_INSECURE_SSL", "").strip().lower()
    if raw in ("1", "true", "yes", "on"):
        return "force_on"
    if raw in ("0", "false", "no", "off"):
        return "force_off"
    return "auto"


def _ssl_error_hint(exc: BaseException) -> bool:
    parts = [str(exc)]
    if isinstance(exc, subprocess.CalledProcessError):
        if exc.stderr:
            parts.append(exc.stderr if isinstance(exc.stderr, str) else exc.stderr.decode("utf-8", "replace"))
        if exc.stdout:
            parts.append(exc.stdout if isinstance(exc.stdout, str) else exc.stdout.decode("utf-8", "replace"))
    text = "\n".join(parts).lower()
    needles = (
        "certificate",
        "ssl",
        "tls",
        "self signed",
        "self-signed",
        "unable to get local issuer",
        "certificate verify failed",
        "not trusted",
        "handshake failure",
    )
    return any(n in text for n in needles)


def _warn_insecure_ssl_fallback(reason: str) -> None:
    _log(
        f"SSL verify failed ({reason}); retrying without certificate check. "
        "Set LLM_KERNELS_CACHE_INSECURE_SSL=0 to disable this fallback."
    )


def _http_auth_opener(insecure: bool = False) -> urllib.request.OpenerDirector:
    """用 HTTPBasicAuthHandler 传凭据，避免 wget/curl 把 token 暴露在进程列表。"""
    user = os.environ["LLM_KERNELS_CACHE_USER"]
    token = os.environ["LLM_KERNELS_CACHE_TOKEN"]
    password_mgr = urllib.request.HTTPPasswordMgrWithDefaultRealm()
    # 对整个 mirror 注册
    base = os.environ["LLM_KERNELS_CACHE_MIRROR_BASE"]
    password_mgr.add_password(None, base, user, token)
    auth_handler = urllib.request.HTTPBasicAuthHandler(password_mgr)
    handlers: List[Any] = [auth_handler]
    if insecure:
        ctx = ssl._create_unverified_context()
        handlers.append(urllib.request.HTTPSHandler(context=ctx))
    return urllib.request.build_opener(*handlers)


def http_download(url: str, dest: Path) -> None:
    # 仅用 urllib（凭据不进 argv）；SSL 失败时按模式自动/强制降级
    mode = _ssl_insecure_mode()
    prefer_insecure = mode == "force_on"
    allow_fallback = mode == "auto"

    def _try(insecure: bool) -> None:
        opener = _http_auth_opener(insecure=insecure)
        with opener.open(url, timeout=600) as resp, dest.open("wb") as out:
            shutil.copyfileobj(resp, out)

    try:
        _try(prefer_insecure)
        return
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code} downloading {url}") from exc
    except _HTTP_NET_EXCS as first_exc:
        if prefer_insecure or not allow_fallback or not _ssl_error_hint(first_exc):
            raise
        if dest.exists():
            dest.unlink(missing_ok=True)
        _warn_insecure_ssl_fallback(str(first_exc))
        try:
            _try(True)
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"HTTP {exc.code} downloading {url}") from exc


def http_exists(url: str) -> bool:
    mode = _ssl_insecure_mode()
    prefer_insecure = mode == "force_on"
    allow_fallback = mode == "auto"

    def _probe(insecure: bool) -> Tuple[bool, Optional[BaseException]]:
        opener = _http_auth_opener(insecure=insecure)
        try:
            req = urllib.request.Request(url, method="HEAD")
            with opener.open(req, timeout=30):
                return True, None
        except _HTTP_NET_EXCS as head_exc:
            # 部分仓库不支持 HEAD，退回 GET 前几个字节失败即 miss
            try:
                with opener.open(url, timeout=30) as resp:
                    resp.read(1)
                return True, None
            except _HTTP_NET_EXCS as get_exc:
                return False, get_exc if _ssl_error_hint(get_exc) else head_exc

    ok, err = _probe(prefer_insecure)
    if ok:
        return True
    if prefer_insecure or not allow_fallback:
        return False
    if err is not None and not _ssl_error_hint(err):
        return False
    ok2, _ = _probe(True)
    if ok2:
        _warn_insecure_ssl_fallback(str(err) if err else "existence probe")
        return True
    return False


def http_upload(url: str, local_file: Path) -> None:
    # 仅用 urllib PUT，避免 curl -u 把 token 暴露在进程列表
    mode = _ssl_insecure_mode()
    prefer_insecure = mode == "force_on"
    allow_fallback = mode == "auto"

    def _urllib_put(insecure: bool) -> None:
        data = local_file.read_bytes()
        req = urllib.request.Request(url, data=data, method="PUT")
        opener = _http_auth_opener(insecure=insecure)
        with opener.open(req, timeout=600):
            pass

    try:
        _urllib_put(prefer_insecure)
        return
    except _HTTP_NET_EXCS as first_exc:
        if prefer_insecure or not allow_fallback or not _ssl_error_hint(first_exc):
            raise
        _warn_insecure_ssl_fallback(str(first_exc))
        _urllib_put(True)


# =============================================================================
# pack / unpack
#
# 从 --build-dir（CMAKE_BINARY_DIR）收集 .a，写入
# ${CACHE_DIR}/${cache_key}/{lib,generated_md5,manifest.json,bundle.tar.zst}。
# 先打 tar 再算 sha 写侧车；下载以侧车 bundle_sha256 为准，unpack 后覆盖目录内
# manifest。不收 Triton cubin / .o。
#
# gemm/humming 源码侧是 INTERFACE 聚合多个 STATIC；pack 按已有 .a 名推断 links，
# generate-cmake 用 ';' 连接（空格会被链接器拆成错误的 -l）。
# _REQUIRED_STATIC_TARGETS 对应主工程无条件链接的库；旧 build 缺 dspark 时硬失败，
# 不要在 CMakeLists 里改成 if(TARGET) 掩盖。
# =============================================================================


def _archive_name_to_target(archive_name: str) -> str:
    name = archive_name
    if name.startswith("lib"):
        name = name[3:]
    if name.endswith(".a"):
        name = name[:-2]
    return name


def _find_static_archives(build_dir: Path) -> Dict[str, Path]:
    archives: Dict[str, Path] = {}
    patterns = ("libllm_kernels*.a", "libprefill_kernels.a", "prefill_kernels.a")
    search_roots = [
        build_dir / "lib",
        build_dir / "3rdparty" / "LLM_kernels" / "lib",
        build_dir / "3rdparty" / "LLM_kernels",
    ]
    for root in search_roots:
        if not root.is_dir():
            continue
        iterator = root.glob if root.name == "lib" else root.rglob
        for pattern in patterns:
            for path in iterator(pattern):
                if path.is_file():
                    archives[_archive_name_to_target(path.name)] = path
    return archives


# 主工程 csrc/kernels/CMakeLists.txt 无条件链接的库；缺失通常是旧 build 未 reconfigure
_REQUIRED_STATIC_TARGETS = (
    "llm_kernels_nvidia_kernel_dspark",
)


def _validate_required_archives(archives: Dict[str, Path], build_dir: Path) -> None:
    missing = [name for name in _REQUIRED_STATIC_TARGETS if name not in archives]
    if not missing:
        return
    hint = (
        "rebuild after a clean cmake configure so LLM_kernels targets are regenerated "
        f"(missing under {build_dir}/lib)"
    )
    raise RuntimeError(
        "incomplete LLM_kernels build for pack; required static libs missing: "
        f"{', '.join(missing)}. Likely stale build tree before those targets existed; {hint}"
    )


def _infer_interface_targets(static_names: Set[str]) -> List[Dict[str, Any]]:
    interfaces: List[Dict[str, Any]] = []
    # 与 humming 一致：按前缀收集，避免新增 gemm_* STATIC 时静默漏链
    gemm_prefix = "llm_kernels_nvidia_kernel_gemm_"
    gemm_links = sorted(n for n in static_names if n.startswith(gemm_prefix))
    if gemm_links:
        interfaces.append(
            {"name": "llm_kernels_nvidia_kernel_gemm", "type": "INTERFACE", "links": gemm_links}
        )
    humming_prefix = "llm_kernels_nvidia_kernel_humming_"
    humming_links = sorted(n for n in static_names if n.startswith(humming_prefix))
    if humming_links:
        interfaces.append(
            {"name": "llm_kernels_nvidia_kernel_humming", "type": "INTERFACE", "links": humming_links}
        )
    elif "llm_kernels_nvidia_kernel_humming_stubs" in static_names:
        interfaces.append(
            {
                "name": "llm_kernels_nvidia_kernel_humming",
                "type": "INTERFACE",
                "links": ["llm_kernels_nvidia_kernel_humming_stubs"],
            }
        )
    return interfaces


def _gzip_fallback_path(out_path: Path) -> Path:
    """无 zstd 时的临时/目标 gzip 路径。

    不能对 ``bundle.tar.zst`` 用 ``Path.with_suffix('.tar.gz')``：
    会变成 ``bundle.tar.tar.gz``，再 rename 回 ``.zst`` 后无法按 gzip 解压。
    """
    if out_path.name.endswith(".tar.gz"):
        return out_path
    name = out_path.name
    if name.endswith(".tar.zst"):
        return out_path.with_name(name[: -len(".tar.zst")] + ".tar.gz")
    if name.endswith(".zst"):
        return out_path.with_name(name[: -len(".zst")] + ".tar.gz")
    return out_path.with_name(out_path.name + ".tar.gz")


def _create_bundle(staging: Path, out_path: Path, members: List[str]) -> str:
    if out_path.exists():
        out_path.unlink()
    if shutil.which("zstd"):
        # bundle.tar.zst -> 中间文件 bundle.tar（勿用 with_suffix('.tar') 变成 bundle.tar.tar）
        if out_path.name.endswith(".tar.zst"):
            tar_path = out_path.with_name(out_path.name[: -len(".tar.zst")] + ".tar")
        else:
            tar_path = out_path.with_suffix(".tar")
        try:
            with tarfile.open(tar_path, "w") as tar:
                for name in members:
                    src = staging / name
                    if src.exists():
                        tar.add(src, arcname=name)
            subprocess.run(
                ["zstd", "-q", "-T0", "-f", str(tar_path), "-o", str(out_path)],
                check=True,
            )
        finally:
            tar_path.unlink(missing_ok=True)
        return "zstd"
    gz_path = _gzip_fallback_path(out_path)
    with tarfile.open(gz_path, "w:gz") as tar:
        for name in members:
            src = staging / name
            if src.exists():
                tar.add(src, arcname=name)
    if gz_path != out_path:
        shutil.move(str(gz_path), str(out_path))
    return "gzip"


def pack_bundle(
    source_root: Path,
    build_dir: Path,
    cache_key: str,
    output_dir: Path,
    sm: Optional[str] = None,
) -> Path:
    llm_build_dir = build_dir / "3rdparty" / "LLM_kernels"
    if not llm_build_dir.is_dir():
        raise FileNotFoundError(f"LLM_kernels build dir not found: {llm_build_dir}")
    key_fields = compute_cache_key_fields(source_root=source_root, sm=sm)
    if key_fields["cache_key"] != cache_key:
        raise ValueError(
            f"cache_key mismatch: arg={cache_key}, computed={key_fields['cache_key']}"
        )

    key_dir_path = output_dir
    if key_dir_path.exists():
        shutil.rmtree(key_dir_path)
    key_dir_path.mkdir(parents=True)
    (key_dir_path / "lib").mkdir()
    (key_dir_path / "generated_md5").mkdir()

    archives = _find_static_archives(build_dir)
    if not archives:
        raise RuntimeError(f"no static archives under {build_dir}")
    _validate_required_archives(archives, build_dir)

    static_targets: List[Dict[str, str]] = []
    static_names: Set[str] = set()
    for target_name in sorted(archives):
        src = archives[target_name]
        rel = f"lib/{src.name}"
        shutil.copy2(src, key_dir_path / rel)
        static_targets.append({"name": target_name, "type": "STATIC", "archive": rel})
        static_names.add(target_name)

    for md5_file in sorted(llm_build_dir.rglob("*.md5sums")):
        rel = md5_file.relative_to(llm_build_dir)
        dst = key_dir_path / "generated_md5" / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(md5_file, dst)

    bname = bundle_name()
    payload: Dict[str, Any] = {
        "format_version": 1,
        "cache_key": cache_key,
        "key_fields": key_fields,
        "static_targets": static_targets,
        "interface_targets": _infer_interface_targets(static_names),
        "bundle_name": bname,
        "note": "bundle_sha256 lives in side-car manifest.json next to the archive",
    }
    _write_json(key_dir_path / "manifest.json", payload)
    bundle_path = key_dir_path / bname
    compression = _create_bundle(key_dir_path, bundle_path, ["lib", "generated_md5", "manifest.json"])
    side = dict(payload)
    side["bundle_compression"] = compression
    side["bundle_name"] = bundle_path.name
    side["bundle_sha256"] = _sha256_file(bundle_path)
    _write_json(key_dir_path / "manifest.json", side)
    return bundle_path


def _safe_extractall(tar: tarfile.TarFile, dest_dir: Path) -> None:
    """解压前校验成员路径不逃逸 dest；Py3.12+ 再启用 filter='data'。"""
    dest_resolved = dest_dir.resolve()
    for member in tar.getmembers():
        member_path = (dest_resolved / member.name).resolve()
        if member_path != dest_resolved and not str(member_path).startswith(
            str(dest_resolved) + os.sep
        ):
            raise RuntimeError(f"unsafe path in archive: {member.name!r}")
    # Python 3.12+：拒绝特殊文件等；更早版本仅靠上面的路径校验
    if sys.version_info >= (3, 12):
        tar.extractall(dest_dir, filter="data")
    else:
        tar.extractall(dest_dir)


def _extract_bundle(bundle_path: Path, dest_dir: Path) -> None:
    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    dest_dir.mkdir(parents=True)
    if bundle_path.suffix == ".zst" or bundle_path.name.endswith(".tar.zst"):
        tar_path = dest_dir.parent / "_bundle_tmp.tar"
        try:
            subprocess.run(
                ["zstd", "-q", "-d", "-f", str(bundle_path), "-o", str(tar_path)],
                check=True,
            )
            with tarfile.open(tar_path, "r") as tar:
                _safe_extractall(tar, dest_dir)
        finally:
            tar_path.unlink(missing_ok=True)
    elif bundle_path.suffix == ".gz" or bundle_path.name.endswith(".tar.gz"):
        with tarfile.open(bundle_path, "r:gz") as tar:
            _safe_extractall(tar, dest_dir)
    else:
        with tarfile.open(bundle_path, "r") as tar:
            _safe_extractall(tar, dest_dir)


def unpack_bundle(bundle_path: Path, dest_dir: Path, cache_key: Optional[str] = None) -> Path:
    manifest_side = bundle_path.parent / "manifest.json"
    if not manifest_side.is_file():
        raise FileNotFoundError(f"manifest.json not found next to bundle: {bundle_path}")
    manifest = json.loads(manifest_side.read_text(encoding="utf-8"))
    if cache_key and manifest.get("cache_key") != cache_key:
        raise ValueError(
            f"manifest cache_key mismatch: expected {cache_key}, got {manifest.get('cache_key')}"
        )
    expected_sha = manifest.get("bundle_sha256")
    if expected_sha:
        actual = _sha256_file(bundle_path)
        if actual != expected_sha:
            raise ValueError(f"bundle sha256 mismatch: expected {expected_sha}, got {actual}")
    _extract_bundle(bundle_path, dest_dir)
    shutil.copy2(manifest_side, dest_dir / "manifest.json")
    return dest_dir


# =============================================================================
# download / upload / status
# =============================================================================


def _manifest_bundle_name(manifest: Dict[str, Any], *, where: str) -> str:
    """读取侧车/包内 manifest 的 bundle_name；缺失时给出可读错误而非 KeyError。"""
    bname = manifest.get("bundle_name")
    if not bname or not isinstance(bname, str):
        raise RuntimeError(f"manifest missing valid bundle_name ({where})")
    return bname


def download(cache_key: str) -> Path:
    require_remote_env()
    dest = key_dir(cache_key)
    cache_root = default_cache_dir()
    work = cache_root / f".tmp_{cache_key}"

    if (dest / "manifest.json").is_file() and (dest / "lib").is_dir():
        local_key = json.loads((dest / "manifest.json").read_text(encoding="utf-8")).get("cache_key")
        if local_key == cache_key:
            _log(f"local cache hit: {dest}")
            return dest

    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)

    mname = manifest_name()
    try:
        try:
            _log("downloading manifest")
            http_download(remote_url(cache_key, mname), work / mname)
        except _DOWNLOAD_MISS_EXCS as exc:
            _log(f"remote cache miss for key={cache_key}: {exc}")
            raise SystemExit(2) from exc

        manifest = json.loads((work / mname).read_text(encoding="utf-8"))
        bname = _manifest_bundle_name(manifest, where=str(work / mname))
        _log(f"downloading bundle {bname}")
        http_download(remote_url(cache_key, bname), work / bname)
        unpack_bundle(work / bname, dest, cache_key)
        _log(f"unpacked to {dest}")
        return dest
    finally:
        # unpack 失败 / miss 退出时都清掉临时目录，避免残留
        shutil.rmtree(work, ignore_errors=True)


def upload(cache_key: str, build_dir: Path, sm: Optional[str] = None) -> None:
    require_remote_env()
    out = key_dir(cache_key)
    _log(f"packing build dir={build_dir}, key={cache_key}")
    pack_bundle(SOURCE_ROOT, build_dir, cache_key, out, sm=sm)
    mfile = out / manifest_name()
    bname = _manifest_bundle_name(
        json.loads(mfile.read_text(encoding="utf-8")),
        where=str(mfile),
    )
    for local in (mfile, out / bname):
        _log(f"uploading {local.name}")
        http_upload(remote_url(cache_key, local.name), local)
    _log(f"upload done for key={cache_key}")


def status(cache_key: Optional[str] = None) -> None:
    fields = compute_cache_key_fields(
        SOURCE_ROOT,
        sm=os.environ.get("SM"),
        build_type=os.environ.get("CMAKE_BUILD_TYPE"),
    )
    key = cache_key or fields["cache_key"]
    dest = key_dir(key)
    print(f"cache_key={key}")
    print(f"local={'HIT (' + str(dest) + ')' if (dest / 'manifest.json').is_file() else 'MISS'}")
    try:
        require_remote_env()
        print("remote=HIT" if http_exists(remote_url(key, manifest_name())) else "remote=MISS")
    except SystemExit:
        print("remote=SKIP (env not configured)")


# =============================================================================
# generate-cmake
# =============================================================================


def generate_prebuilt_cmake(manifest_path: Path, prebuilt_dir: Path, output: Path) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    prebuilt_dir = prebuilt_dir.resolve()
    lines = [
        "# Auto-generated by llm_kernels_cache.py",
        f'set(LLM_KERNELS_PREBUILT_CACHE_KEY "{manifest.get("cache_key", "")}")',
        "",
    ]
    for item in manifest.get("static_targets", []):
        name = item["name"]
        archive = prebuilt_dir / item["archive"]
        if not archive.is_file():
            raise FileNotFoundError(f"missing prebuilt archive: {archive}")
        loc = str(archive).replace("\\", "/")
        lines.extend(
            [
                f"if(NOT TARGET {name})",
                f"  add_library({name} STATIC IMPORTED GLOBAL)",
                f"  set_target_properties({name} PROPERTIES",
                f'    IMPORTED_LOCATION "{loc}"',
                "    POSITION_INDEPENDENT_CODE ON",
                "  )",
                "endif()",
                "",
            ]
        )
    for item in manifest.get("interface_targets", []):
        name = item["name"]
        link_expr = ";".join(item.get("links", []))
        lines.extend(
            [
                f"if(NOT TARGET {name})",
                f"  add_library({name} INTERFACE IMPORTED GLOBAL)",
                f'  set_property(TARGET {name} PROPERTY INTERFACE_LINK_LIBRARIES "{link_expr}")',
                "endif()",
                "",
            ]
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


# =============================================================================
# self-test
# =============================================================================


def cmd_self_test(rounds: int = 3, skip_upload: bool = False) -> int:
    require_remote_env()
    os.environ.setdefault("SM", "90a")
    os.environ.setdefault("CMAKE_BUILD_TYPE", "Release")
    os.environ["LLM_KERNELS_REMOTE_CACHE"] = "1"
    os.environ.setdefault("DISABLE_GENERATE_TRITON_KERNEL", "1")
    test_cache = Path(os.environ.get("SELF_TEST_CACHE_DIR", "/tmp/llm_kernels_cache_self_test"))
    os.environ["LLM_KERNELS_CACHE_DIR"] = str(test_cache)
    test_cache.mkdir(parents=True, exist_ok=True)

    def pass_(msg: str) -> None:
        _log(f"PASS: {msg}")

    # key 稳定性
    _log("=== test_key_stability ===")
    old_path = os.environ.get("PATH", "")
    os.environ["PATH"] = "/usr/local/cuda/bin:" + old_path
    key1 = compute_cache_key_fields(SOURCE_ROOT, sm=os.environ.get("SM"))["cache_key"]
    cuda1 = compute_cache_key_fields(SOURCE_ROOT, sm=os.environ.get("SM"))["cuda_version"]
    os.environ["PATH"] = ":".join(p for p in old_path.split(":") if "/cuda/bin" not in p)
    key2 = compute_cache_key_fields(SOURCE_ROOT, sm=os.environ.get("SM"))["cache_key"]
    cuda2 = compute_cache_key_fields(SOURCE_ROOT, sm=os.environ.get("SM"))["cuda_version"]
    os.environ["PATH"] = "/usr/local/cuda/bin:" + old_path
    if key1 != key2 or cuda1 != cuda2 or cuda1 == "unknown":
        _log(f"FAIL: key/cuda drift key1={key1} key2={key2} cuda={cuda1}/{cuda2}")
        return 1
    pass_(f"key stable ({key1[:12]}...), cuda_version={cuda1}")

    # remote
    _log(f"=== ensure_remote_bundle key={key1[:12]}... ===")
    if not http_exists(remote_url(key1, manifest_name())):
        if skip_upload:
            _log("FAIL: remote miss and SELF_TEST_SKIP_UPLOAD")
            return 1
        build_dir = Path(
            os.environ.get(
                "DPV3_BUILD",
                str(SOURCE_ROOT.parent / "dpv3_baseline_build"),
            )
        )
        if not (build_dir / "lib").is_dir():
            _log(f"FAIL: build dir missing lib/: {build_dir}")
            return 1
        upload(key1, build_dir, sm=os.environ.get("SM"))
    pass_("remote has bundle")

    for i in range(1, rounds + 1):
        _log(f"=== try_download round {i}/{rounds} ===")
        for child in test_cache.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
        dest = download(key1)
        if not (dest / "manifest.json").is_file():
            _log(f"FAIL: round {i} missing manifest")
            return 1
        pass_(f"round {i}: download ok -> {dest}")
    _log("ALL TESTS PASSED")
    return 0


# =============================================================================
# CLI
# =============================================================================


def _default_build_dir() -> Path:
    return Path(os.environ.get("DPV3_BUILD", str(SOURCE_ROOT.parent / "dpv3_baseline_build")))


def _resolve_key(sm: Optional[str], build_type: Optional[str]) -> Tuple[str, Dict[str, Any]]:
    fields = compute_cache_key_fields(
        SOURCE_ROOT,
        sm=sm or os.environ.get("SM"),
        build_type=build_type or os.environ.get("CMAKE_BUILD_TYPE"),
    )
    return fields["cache_key"], fields


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="LLM_kernels remote build cache (single-file tool)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_key = sub.add_parser("key", help="打印 cache key")
    p_key.add_argument("--json", action="store_true")
    p_key.add_argument("--sm")
    p_key.add_argument("--build-type")

    sub.add_parser("try-download", help="LLM_KERNELS_REMOTE_CACHE=1 时自动下载；stdout 输出 export 行")
    p_up = sub.add_parser("try-upload", help="LLM_KERNELS_REMOTE_CACHE_UPLOAD=1 时自动上传")
    p_up.add_argument("--build-dir", type=Path, default=None)

    p_dl = sub.add_parser("download", help="下载指定 key")
    p_dl.add_argument("cache_key")

    p_ul = sub.add_parser("upload", help="打包并上传指定 key")
    p_ul.add_argument("cache_key")
    p_ul.add_argument("--build-dir", type=Path, default=None)
    p_ul.add_argument("--sm")

    p_st = sub.add_parser("status", help="本地/远程命中状态")
    p_st.add_argument("cache_key", nargs="?")

    p_pack = sub.add_parser("pack", help="仅打包")
    p_pack.add_argument("--build-dir", type=Path, required=True)
    p_pack.add_argument("--cache-key", required=True)
    p_pack.add_argument("--output-dir", type=Path, default=None)
    p_pack.add_argument("--sm")

    p_un = sub.add_parser("unpack", help="仅解压")
    p_un.add_argument("--bundle", type=Path, required=True)
    p_un.add_argument("--dest-dir", type=Path, required=True)
    p_un.add_argument("--cache-key")

    p_gen = sub.add_parser("generate-cmake", help="从 manifest 生成 IMPORTED cmake（供 cmake 调用）")
    p_gen.add_argument("--manifest", type=Path, required=True)
    p_gen.add_argument("--prebuilt-dir", type=Path, required=True)
    p_gen.add_argument("--output", type=Path, required=True)

    p_test = sub.add_parser("self-test", help="自动化自测")
    p_test.add_argument("--rounds", type=int, default=3)
    p_test.add_argument("--skip-upload", action="store_true")

    sub.add_parser("help-env", help="打印环境变量示例")

    args = parser.parse_args(argv)

    if args.cmd == "help-env":
        print(
            """export LLM_KERNELS_REMOTE_CACHE=1            # 默认关；=1 完整 cmake 后再 try-download
export LLM_KERNELS_REMOTE_CACHE_UPLOAD=1     # 默认关；=1 源码编完 try-upload
export LLM_KERNELS_CACHE_MIRROR_BASE=...     # 必填（远程）
export LLM_KERNELS_CACHE_REPO=...            # 必填（远程）
export LLM_KERNELS_CACHE_USER=...            # 必填（远程）
export LLM_KERNELS_CACHE_TOKEN=...           # 必填（远程）
# 以下可不设：
# LLM_KERNELS_CACHE_DIR            默认 ~/.cache/KsanaLLM/llm_kernels_build_cache
# LLM_KERNELS_CACHE_REMOTE_PREFIX  默认空（URL 无额外 path 段）
# LLM_KERNELS_CACHE_INSECURE_SSL   默认 auto（校验失败再降级）；可强制 1/0
# LLM_KERNELS_CACHE_BUNDLE_NAME    默认 bundle.tar.zst
# LLM_KERNELS_CACHE_MANIFEST_NAME  默认 manifest.json
# SM / CMAKE_BUILD_TYPE 由 cmake 注入 try-download/upload，不必手动 export
"""
        )
        return 0

    if args.cmd == "key":
        key, fields = _resolve_key(args.sm, args.build_type)
        print(json.dumps(fields, indent=2, sort_keys=True) if args.json else key)
        return 0

    if args.cmd == "try-download":
        if not _env_truthy("LLM_KERNELS_REMOTE_CACHE"):
            return 0
        key, _ = _resolve_key(None, None)
        try:
            dest = download(key)
        except SystemExit as exc:
            return int(exc.code) if isinstance(exc.code, int) else 1
        # 仅 stdout 输出 export，供 eval
        print(f"export LLM_KERNELS_PREBUILT_DIR={dest}")
        print(f"export LLM_KERNELS_CACHE_KEY={key}")
        return 0

    if args.cmd == "try-upload":
        if not _env_truthy("LLM_KERNELS_REMOTE_CACHE_UPLOAD"):
            return 0
        key, _ = _resolve_key(None, None)
        upload(key, args.build_dir or _default_build_dir(), sm=os.environ.get("SM"))
        return 0

    if args.cmd == "download":
        try:
            print(download(args.cache_key))
        except SystemExit as exc:
            return int(exc.code) if isinstance(exc.code, int) else 1
        return 0

    if args.cmd == "upload":
        upload(args.cache_key, args.build_dir or _default_build_dir(), sm=args.sm)
        return 0

    if args.cmd == "status":
        status(args.cache_key)
        return 0

    if args.cmd == "pack":
        out = args.output_dir or (default_cache_dir() / args.cache_key)
        print(pack_bundle(SOURCE_ROOT, args.build_dir, args.cache_key, out, sm=args.sm))
        return 0

    if args.cmd == "unpack":
        unpack_bundle(args.bundle, args.dest_dir, args.cache_key)
        return 0

    if args.cmd == "generate-cmake":
        generate_prebuilt_cmake(args.manifest, args.prebuilt_dir, args.output)
        return 0

    if args.cmd == "self-test":
        return cmd_self_test(rounds=args.rounds, skip_upload=args.skip_upload)

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
