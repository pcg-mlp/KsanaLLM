#!/usr/bin/env python3
"""Run queued Triton/TileLang kernel gens with exclusive GPU allocation.

Each job is single-GPU (default) or multi-GPU (`multi_gpu: true`), which
occupies the whole --gpus pool until it exits. Single-GPU jobs pack onto
whatever cards are free.

Job output goes to <log-dir>/<name>.log. Since a single gen can run for tens
of minutes without printing anything here, a heartbeat reports every running
job's elapsed time, log path, log growth and last log line, so a quiet build
is distinguishable from a hung one. Interval: KSANA_TRITON_GEN_HEARTBEAT_SEC
(seconds, 0 disables).

On success a job writes stamps under <artifact_dir>/md5_sum/, then if
live_dir is set and differs, promotes that staging tree over the live
job directory so a failed gen does not wipe last-known-good cubins.
cmake skip requires every listed file to exist
under the live job dir and still hash to that digest. Creators needing
their own log dir should use KSANA_TRITON_GEN_LOG_DIR rather than writing
into artifact_dir.

Job JSON lines:
  name, cwd, cmd (argv list), env (optional dict), md5_file, md5_value,
  artifact_dir (staging output dir), live_dir (final job dir, optional),
  multi_gpu (optional bool, default false)

On the first job failure the runner stops dispatching queued jobs, signals
in-flight gens to exit, and prints a copy-paste shell command (cwd + GPU +
job env + argv) so the failure can be reproduced outside cmake.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

HEARTBEAT_ENV = "KSANA_TRITON_GEN_HEARTBEAT_SEC"
DEFAULT_HEARTBEAT_SEC = 60.0

# Filesystem mtimes can lag the recorded start time slightly (coarse timestamps,
# NFS clock skew), so widen the window a little when attributing artifacts.
ARTIFACT_MTIME_SLACK_SEC = 2.0

ARTIFACT_SKIP_SUFFIXES = frozenset({".log", ".tmp"})

# Creators that write their own per-task logs put them here instead of in the
# kernel output dir.
LOG_DIR_ENV = "KSANA_TRITON_GEN_LOG_DIR"

_print_lock = threading.Lock()


def emit(text: str) -> None:
    with _print_lock:
        print(text, flush=True)


def clock() -> str:
    return time.strftime("%H:%M:%S")


def parse_gpus(spec: str) -> list[str]:
    gpus = [x.strip() for x in spec.split(",") if x.strip()]
    return gpus or ["0"]


def heartbeat_interval() -> float:
    raw = os.environ.get(HEARTBEAT_ENV, "")
    if not raw.strip():
        return DEFAULT_HEARTBEAT_SEC
    try:
        return max(0.0, float(raw))
    except ValueError:
        return DEFAULT_HEARTBEAT_SEC


def fmt_dur(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(seconds, 60)
    if m < 60:
        return f"{int(m)}m {s:04.1f}s"
    h, m = divmod(int(m), 60)
    return f"{h}h {m}m {s:04.1f}s"


def fmt_size(num_bytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(num_bytes) < 1024 or unit == "GB":
            return f"{num_bytes:.0f}{unit}" if unit == "B" else f"{num_bytes:.1f}{unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f}GB"


def cmd_preview(cmd: list) -> str:
    text = " ".join(str(x) for x in cmd)
    if len(text) > 120:
        return text[:117] + "..."
    return text


def format_reproduce_command(job: dict, gpu_spec: str) -> str:
    """One-line shell command matching how the runner launched this job."""
    cwd = str(job.get("cwd") or os.getcwd())
    argv = [str(x) for x in (job.get("cmd") or [])]
    env_parts = [f"CUDA_VISIBLE_DEVICES={shlex.quote(gpu_spec)}"]
    extra = job.get("env") or {}
    for key in sorted(extra):
        env_parts.append(f"{key}={shlex.quote(str(extra[key]))}")
    cmd_s = " ".join(shlex.quote(a) for a in argv)
    return f"cd {shlex.quote(cwd)} && {' '.join(env_parts)} {cmd_s}"


def emit_reproduce(job: dict, gpu_spec: str, rc: int, log_path) -> None:
    emit(f"[triton-gen] job {job.get('name')} failed rc={rc}; log={log_path}")
    emit("[triton-gen] ---- reproduce (copy-paste) ----")
    emit(format_reproduce_command(job, gpu_spec))
    emit("[triton-gen] --------------------------------")


def terminate_process_group(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            proc.terminate()
        except OSError:
            pass


def job_num_gpus(job: dict, pool_size: int) -> int:
    if job.get("multi_gpu"):
        return pool_size
    return 1


def log_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def log_tail_line(path: Path, max_bytes: int = 8192, max_len: int = 110) -> str:
    """Last non-empty line of the job log, for liveness reporting."""
    try:
        with path.open("rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            fh.seek(max(0, size - max_bytes))
            chunk = fh.read()
    except OSError:
        return ""
    for line in reversed(chunk.decode("utf-8", errors="replace").splitlines()):
        line = line.strip()
        if line:
            return line if len(line) <= max_len else line[: max_len - 3] + "..."
    return ""


def is_artifact(rel: Path) -> bool:
    """Reject build byproducts that some creators drop into the kernel dir.

    Logs and hidden scratch dirs are not consumed at runtime, so recording them
    would tie cache validity to files nobody is expected to keep.
    """
    if rel.suffix in ARTIFACT_SKIP_SUFFIXES:
        return False
    if any(part == "md5_sum" for part in rel.parts):
        return False
    return not any(part.startswith(".") for part in rel.parts)


def file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_artifacts(artifact_dir: Path, stamp_dir: Path, since: float) -> list[str]:
    """Kernel files touched since `since`, relative to this job's artifact_dir."""
    cutoff = since - ARTIFACT_MTIME_SLACK_SEC
    found = []
    for path in artifact_dir.rglob("*"):
        try:
            if not path.is_file() or path.stat().st_mtime < cutoff:
                continue
        except OSError:
            continue
        # Stamps and manifests live under the kernel dir; they are not artifacts.
        if stamp_dir == path.parent or stamp_dir in path.parents:
            continue
        rel = path.relative_to(artifact_dir)
        if not is_artifact(rel):
            continue
        found.append(rel.as_posix())
    return sorted(found)


def promote_staging(staging: Path, live: Path) -> None:
    """Replace live job dir with staging. On failure, restore live if it was moved."""
    live.parent.mkdir(parents=True, exist_ok=True)
    if not staging.is_dir():
        raise FileNotFoundError("staging directory missing: %s" % staging)
    try:
        if live.exists() and staging.resolve() == live.resolve():
            return
    except OSError:
        pass
    prev = live.parent / (live.name + ".prev")
    if prev.exists():
        shutil.rmtree(prev)
    try:
        if live.exists():
            live.rename(prev)
        try:
            staging.rename(live)
        except OSError:
            shutil.move(str(staging), str(live))
    except Exception:
        if prev.exists() and not live.exists():
            prev.rename(live)
        raise
    if prev.exists():
        shutil.rmtree(prev, ignore_errors=True)


def commit_success(job: dict, t0: float) -> int:
    """Write a non-empty content-md5 manifest, then the source md5 stamp.

    Artifacts are collected from artifact_dir (staging). Stamp is written
    under that tree; if live_dir is set, staging is then promoted over live
    so a failed run leaves the previous job directory intact.
    """
    artifact_dir = job.get("artifact_dir")
    root = Path(artifact_dir) if artifact_dir else None
    md5_file = Path(job["md5_file"])
    stamp_dir = (root / "md5_sum") if root is not None else md5_file.parent
    stamp_file = stamp_dir / md5_file.name

    artifacts: list[str] = []
    if root and root.is_dir():
        artifacts = collect_artifacts(root, stamp_dir, t0)

    if not artifacts:
        emit(
            f"[triton-gen] no artifacts recorded under {artifact_dir or '<none>'} "
            f"for {job.get('name')}; md5 stamp not written"
        )
        return 0

    stamp_dir.mkdir(parents=True, exist_ok=True)
    lines = []
    for rel in artifacts:
        lines.append(f"{file_md5(root / rel)}  {rel}\n")
    stamp_file.with_name(stamp_file.name + ".files").write_text("".join(lines))
    stamp_file.write_text(job["md5_value"])

    live_raw = job.get("live_dir")
    if live_raw and root is not None:
        live = Path(live_raw)
        if live.resolve() != root.resolve():
            promote_staging(root, live)
            emit(f"[triton-gen] promoted {root} -> {live}")
    return len(artifacts)


def run_job(
    job: dict,
    gpu_spec: str,
    log_path: Path,
    t0: float,
    proc_slot: dict | None = None,
) -> tuple[str, int, float, str, str]:
    name = job["name"]
    env = os.environ.copy()
    env[LOG_DIR_ENV] = str(log_path.parent)
    env.update(job.get("env") or {})
    env["CUDA_VISIBLE_DEVICES"] = gpu_spec
    cmd = job["cmd"]
    cwd = job["cwd"]
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        log.write(f"CUDA_VISIBLE_DEVICES={gpu_spec}\ncwd={cwd}\ncmd={cmd}\n\n")
        log.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        if proc_slot is not None:
            proc_slot["proc"] = proc
        rc = proc.wait()
    elapsed = time.time() - t0
    if rc == 0:
        try:
            n_artifacts = commit_success(job, t0)
        except Exception as exc:
            emit(
                f"[triton-gen] [{clock()}] FAIL  {name} GPU={gpu_spec} {fmt_dur(elapsed)} "
                f"rc=1 stamp-error={type(exc).__name__}: {exc} log={log_path}"
            )
            return name, 1, elapsed, str(log_path), gpu_spec
        emit(
            f"[triton-gen] [{clock()}] OK    {name} GPU={gpu_spec} {fmt_dur(elapsed)} "
            f"artifacts={n_artifacts}"
        )
    else:
        emit(
            f"[triton-gen] [{clock()}] FAIL  {name} GPU={gpu_spec} {fmt_dur(elapsed)} rc={rc} "
            f"log={log_path}"
        )
    return name, rc, elapsed, str(log_path), gpu_spec


def duplicate_job_names(jobs: list[dict]) -> list[str]:
    seen: set[str] = set()
    dups: list[str] = []
    for job in jobs:
        name = job.get("name")
        if not name:
            continue
        if name in seen and name not in dups:
            dups.append(name)
        seen.add(name)
    return dups


def schedule_jobs(jobs: list[dict], gpus: list[str], log_dir: Path) -> list[tuple]:
    """Exclusive GPU packing: multi-GPU jobs take the pool, others fill free cards."""
    pool_size = len(gpus)
    pending = list(jobs)
    orig_rank = {id(j): i for i, j in enumerate(jobs)}
    free = list(gpus)
    running: dict = {}
    results: list = []
    cv = threading.Condition()
    abort = threading.Event()
    started = 0
    wall_t0 = time.time()

    def cancel_inflight(except_id=None) -> None:
        with cv:
            leftover = list(pending)
            pending.clear()
            snapshot = [(jid, info.get("proc")) for jid, info in running.items()]
            cv.notify_all()
        if leftover:
            names = ", ".join(j.get("name", "?") for j in leftover)
            emit(f"[triton-gen] skip queued jobs after failure: {names}")
        for jid, proc in snapshot:
            if except_id is not None and jid == except_id:
                continue
            if proc is None:
                continue
            terminate_process_group(proc)

    def start_job(job: dict, gpu_list: list[str]) -> None:
        nonlocal started
        name = job["name"]
        gpu_spec = ",".join(gpu_list)
        started += 1
        job_id = started
        log_path = log_dir / f"{job_id:02d}-{name}.log"
        t0 = time.time()
        running[job_id] = {
            "t0": t0,
            "gpu": gpu_spec,
            "log": log_path,
            "last_size": 0,
            "proc": None,
            "job": job,
            "name": name,
        }
        emit(
            f"[triton-gen] [{clock()}] START {name} GPU={gpu_spec} "
            f"({len(results)} done / {started} started, {len(pending)} queued)  log={log_path}"
        )

        def target() -> None:
            slot = running.get(job_id) or {}
            result = (name, 1, 0.0, str(log_path), gpu_spec)
            try:
                result = run_job(job, gpu_spec, log_path, t0, proc_slot=slot)
            except Exception as exc:
                elapsed = time.time() - t0
                try:
                    with log_path.open("a", encoding="utf-8", errors="replace") as log:
                        log.write(f"\n[triton-gen] worker exception: {exc!r}\n")
                        traceback.print_exc(file=log)
                except OSError:
                    pass
                emit(
                    f"[triton-gen] [{clock()}] FAIL  {name} GPU={gpu_spec} "
                    f"{fmt_dur(elapsed)} rc=1 exception={type(exc).__name__}: {exc} "
                    f"log={log_path}"
                )
                result = (name, 1, elapsed, str(log_path), gpu_spec)
            finally:
                _name, rc, _elapsed, _log, _gpu = result
                cancelled = False
                if rc != 0 and not abort.is_set():
                    abort.set()
                    emit_reproduce(job, gpu_spec, rc, log_path)
                    cancel_inflight(except_id=job_id)
                elif rc != 0:
                    cancelled = True
                    emit(
                        f"[triton-gen] [{clock()}] CANCEL {name} "
                        f"(stopped after another job failed) log={log_path}"
                    )
                with cv:
                    running.pop(job_id, None)
                    free.extend(gpu_list)
                    results.append((_name, rc, _elapsed, _log, _gpu, cancelled))
                    cv.notify()

        threading.Thread(target=target, name=f"triton-{job_id}-{name}", daemon=True).start()

    def try_dispatch() -> None:
        if abort.is_set():
            if pending:
                cancel_inflight()
            return
        # Prefer wider jobs so a multi-GPU gen is not starved by 1-GPU packing.
        pending.sort(key=lambda j: (-job_num_gpus(j, pool_size), orig_rank[id(j)]))
        progressed = True
        while progressed:
            progressed = False
            if abort.is_set():
                return
            for job in list(pending):
                need = job_num_gpus(job, pool_size)
                if len(free) < need:
                    continue
                gpu_list = [free.pop(0) for _ in range(need)]
                pending.remove(job)
                start_job(job, gpu_list)
                progressed = True

    def report_progress() -> None:
        with cv:
            snapshot = sorted(running.items(), key=lambda kv: kv[1]["t0"])
            done, queued = len(results), len(pending)
            n_started = started
            growth = {}
            for jid, info in snapshot:
                size = log_size(info["log"])
                growth[jid] = size - info["last_size"]
                info["last_size"] = size
        now = time.time()
        emit(
            f"[triton-gen] [{clock()}] still running, elapsed {fmt_dur(now - wall_t0)} | "
            f"done {done}/{n_started} started | running {len(snapshot)} | queued {queued}"
            f"{' | aborting' if abort.is_set() else ''}"
        )
        for jid, info in snapshot:
            disp = info.get("name", jid)
            log_path = info["log"]
            emit(
                f"[triton-gen]     {disp:<32} GPU={info['gpu']:<16} "
                f"{fmt_dur(now - info['t0']):>10}  log={log_path} "
                f"({fmt_size(log_size(log_path))}, +{fmt_size(max(0, growth[jid]))} since last check)"
            )
            tail = log_tail_line(log_path)
            if tail:
                emit(f"[triton-gen]       last: {tail}")

    interval = heartbeat_interval()
    stop_heartbeat = threading.Event()

    def heartbeat_loop() -> None:
        while not stop_heartbeat.wait(interval):
            with cv:
                idle = not running and not pending
            if idle:
                continue
            report_progress()

    hb_thread = None
    if interval > 0:
        hb_thread = threading.Thread(target=heartbeat_loop, name="triton-heartbeat", daemon=True)
        hb_thread.start()

    try:
        with cv:
            try_dispatch()
            while pending or running:
                if abort.is_set() and pending:
                    cancel_inflight()
                if pending and not running and not free:
                    raise RuntimeError("triton-gen scheduler deadlock: pending jobs but no GPUs")
                cv.wait(timeout=0.5)
                try_dispatch()
            # Workers append results in the same lock that releases GPUs.
            while len(results) < started:
                cv.wait(timeout=0.5)
    finally:
        stop_heartbeat.set()
        if hb_thread is not None:
            hb_thread.join(timeout=1.0)
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs-file", required=True)
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--gpus", required=True, help="comma-separated CUDA_VISIBLE_DEVICES")
    args = parser.parse_args()

    jobs_path = Path(args.jobs_file)
    if not jobs_path.exists():
        emit(f"[triton-gen] no jobs file: {jobs_path}")
        return 0

    jobs = []
    with jobs_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                jobs.append(json.loads(line))
    if not jobs:
        emit("[triton-gen] queue empty")
        return 0

    dups = duplicate_job_names(jobs)
    if dups:
        emit(f"[triton-gen] duplicate job NAME(s) in queue: {', '.join(dups)}")
        emit("[triton-gen] each ksana_enqueue_triton_gen NAME must be unique")
        return 1

    gpus = parse_gpus(args.gpus)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    interval = heartbeat_interval()

    emit(f"[triton-gen] [{clock()}] {len(jobs)} job(s), GPU pool={len(gpus)}: {','.join(gpus)}")
    emit(
        "[triton-gen] schedule=exclusive "
        "(MULTI_GPU jobs take the whole pool; single-GPU jobs pack on free cards)"
    )
    emit(f"[triton-gen] per-job logs: {log_dir}/<index>-<name>.log   (follow with: tail -f <path>)")
    if interval > 0:
        emit(
            f"[triton-gen] progress heartbeat every {int(interval)}s "
            f"(set {HEARTBEAT_ENV}=0 to silence, or a smaller value for more detail)"
        )
    emit("[triton-gen] note: a single kernel gen can run 10-30min with no output here")
    emit("[triton-gen] ---- queued jobs ----")
    for i, job in enumerate(jobs):
        mode = "multi-gpu" if job.get("multi_gpu") else "single-gpu"
        emit(
            f"[triton-gen]   [{i + 1:02d}] {job['name']:<32} "
            f"{mode}  cwd={job.get('cwd', '')}  cmd={cmd_preview(job.get('cmd') or [])}"
        )
    emit("[triton-gen] ---- start concurrent ----")

    wall_t0 = time.time()
    results = schedule_jobs(jobs, gpus, log_dir)
    wall = time.time() - wall_t0

    results.sort(key=lambda x: x[2], reverse=True)
    sum_cpu = sum(elapsed for _n, _rc, elapsed, _p, _g, *_rest in results)
    emit("[triton-gen] ---- timing summary ----")
    emit(
        f"[triton-gen] wall {fmt_dur(wall)}  "
        f"sum-of-jobs {fmt_dur(sum_cpu)}  "
        f"jobs={len(results)}  pool={len(gpus)}"
    )
    for name, rc, elapsed, log_path, gpu, *rest in results:
        cancelled = rest[0] if rest else False
        if cancelled:
            status = "CANCEL"
        elif rc == 0:
            status = "OK"
        else:
            status = f"FAIL(rc={rc})"
        emit(
            f"[triton-gen]   {fmt_dur(elapsed):>10}  {status:<12}  GPU={gpu:<16}  "
            f"{name:<32} log={log_path}"
        )
    emit("[triton-gen] ---- end summary ----")

    failures = []
    for name, rc, elapsed, log_path, gpu, *rest in results:
        cancelled = rest[0] if rest else False
        if rc != 0 and not cancelled:
            failures.append((name, rc, log_path, gpu))
    if failures:
        jobs_by_name = {j["name"]: j for j in jobs}
        for name, rc, log_path, gpu in failures:
            emit(f"[triton-gen] ----- {name} log ({log_path}) -----")
            try:
                text = Path(log_path).read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                emit(f"[triton-gen] cannot read log: {exc}")
            else:
                lines = text.splitlines()
                tail = lines[-80:] if len(lines) > 80 else lines
                emit("\n".join(tail))
            job = jobs_by_name.get(name)
            if job is not None:
                emit_reproduce(job, gpu, rc, log_path)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
