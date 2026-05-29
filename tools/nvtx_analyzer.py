#!/usr/bin/env python3
"""
NVTX Event Analyzer: 统计指定父事件下子事件的 GPU 执行时间。

原理:
  通过 NVTX scope → CUPTI_ACTIVITY_KIND_RUNTIME (correlationId) → CUPTI_ACTIVITY_KIND_KERNEL
  的关联链，获取每个 NVTX 事件对应的 GPU kernel 实际执行时间。

  注意: 由于 GPU kernel 异步执行，一个 NVTX scope 内 launch 的 kernel 可能在该 scope
  结束后很久才开始/完成执行。本脚本统计的是 NVTX scope 内 launch 的所有 kernel 的
  GPU 执行时间之和。

功能:
  1. 根据事件名（子串匹配）找到父事件
  2. 按持续时间阈值过滤父事件（基于 CPU 时间）
  3. 统计父事件内各子事件对应的 GPU kernel 执行时间
  4. 支持聚合统计和单事件明细输出
  5. 支持 --cpu-time 模式回退到 CPU 侧 NVTX 时间统计

用法:
  python nvtx_analyzer.py <nsys-rep-or-sqlite> [options]

参数:
  input                 nsys-rep 或 sqlite 文件路径
  --parent PATTERN      父事件名（子串匹配），必选
  --children P1,P2,...  子事件名列表（逗号分隔，子串匹配），不指定则输出所有
  --child PATTERN       只显示包含该子事件的父事件
  --min-dur MS          父事件最小持续时间（ms，默认按 GPU 时间过滤；--cpu-time 模式按 CPU 时间），默认 0
  --max-dur MS          父事件最大持续时间（ms，同上），默认 0（不限）
  --limit N             最多分析多少个父事件（默认 20）
  --top N               输出前 N 个父事件详情（默认 5）
  --detail              输出每个父事件的子事件明细
  --aggregate           聚合所有匹配父事件的子事件统计
  --cpu-time            使用 CPU 侧 NVTX 时间（而非 GPU kernel 时间）

示例:

  # 1. 分析 decode iter 下各组件 GPU 执行时间
  python nvtx_analyzer.py report23.sqlite \\
      --parent "Exec.HandleExecute.iter" --min-dur 23 --max-dur 25 \\
      --children "MOE,CommonShareMlp,expert_gate,add_layer" \\
      --aggregate

  # 2. 对比多流并行版本的 GPU 时间
  python nvtx_analyzer.py report23-ms.sqlite \\
      --parent "Exec.HandleExecute.iter" --min-dur 25 --max-dur 28 \\
      --children "MOE,CommonShareMlp,expert_gate,add_layer" \\
      --aggregate

  # 3. 分析 prefill（seq_len=64）下各组件 GPU 时间
  python nvtx_analyzer.py report23-ms.sqlite \\
      --parent "DS_CommonMlp_seq_len_64" \\
      --children "MOE,CommonShareMlp,expert_gate" --aggregate

  # 4. 使用 CPU 时间模式（快速，不需要关联 kernel）
  python nvtx_analyzer.py report23.sqlite \\
      --parent "Exec.HandleExecute.iter" --min-dur 23 --max-dur 25 \\
      --children "MOE,CommonShareMlp" --aggregate --cpu-time

  # 5. 查看单个事件的详细 GPU kernel 明细
  python nvtx_analyzer.py report23-ms.sqlite \\
      --parent "DS_CommonMlp_seq_len_64" --detail --top 3
"""

import argparse
import bisect
import os
import re
import sqlite3
import subprocess
import sys
from collections import defaultdict


def ensure_sqlite(input_path):
    """Convert .nsys-rep to .sqlite if needed, return sqlite path."""
    if input_path.endswith('.sqlite'):
        return input_path
    sqlite_path = input_path.replace('.nsys-rep', '.sqlite')
    if not os.path.exists(sqlite_path):
        print(f"Converting {input_path} to SQLite...")
        subprocess.run(['nsys', 'stats', '--report', 'nvtx_sum', '--format', 'none',
                       '-o', sqlite_path.replace('.sqlite', ''), input_path],
                      capture_output=True)
        if not os.path.exists(sqlite_path):
            subprocess.run(['nsys', 'export', '--type', 'sqlite', '-o', sqlite_path, input_path],
                          capture_output=True)
    if not os.path.exists(sqlite_path):
        print(f"Error: Cannot generate {sqlite_path}")
        sys.exit(1)
    return sqlite_path


def ensure_indices(cur):
    """Create indices on large tables to avoid full table scans."""
    print("  Building indices (first run may take a few seconds)...")
    cur.execute("""
    CREATE INDEX IF NOT EXISTS idx_runtime_tid_start
    ON CUPTI_ACTIVITY_KIND_RUNTIME(globalTid, start)
    """)
    cur.execute("""
    CREATE INDEX IF NOT EXISTS idx_kernel_corrid_pid
    ON CUPTI_ACTIVITY_KIND_KERNEL(correlationId, globalPid)
    """)
    cur.execute("""
    CREATE INDEX IF NOT EXISTS idx_nvtx_textid_start
    ON NVTX_EVENTS(textId, start)
    """)
    cur.execute("""
    CREATE INDEX IF NOT EXISTS idx_nvtx_tid_start
    ON NVTX_EVENTS(globalTid, start)
    """)
    print("  Indices ready.")


def build_pid_map(cur):
    """Build sorted globalPid list for tid-to-pid mapping.

    In nsys multi-process traces, correlationId is NOT unique across processes.
    To correctly match RUNTIME -> KERNEL, we must filter KERNEL by globalPid.
    The mapping: for a given globalTid, globalPid = max(all globalPids <= globalTid).
    """
    cur.execute("SELECT DISTINCT globalPid FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE globalPid IS NOT NULL")
    pids = sorted([r[0] for r in cur.fetchall()])
    return pids


def tid_to_pid(global_pids, tid):
    """Map a globalTid to its owning process's globalPid."""
    if not global_pids:
        return None
    idx = bisect.bisect_right(global_pids, tid) - 1
    if idx < 0:
        return global_pids[0]
    return global_pids[idx]


def build_text_map(cur):
    """Build textId -> text mapping from StringIds table."""
    text_map = {}
    cur.execute("SELECT id, value FROM StringIds")
    for row in cur.fetchall():
        text_map[row[0]] = row[1]
    return text_map


def get_text_id(cur, text_map, pattern):
    """Find textId(s) matching a pattern (substring match)."""
    matches = []

    for tid, text in text_map.items():
        if pattern in text:
            #print(f"tid: {tid} text: {text}" )
            matches.append((tid, text))
    return matches


def find_parent_events(cur, text_map, parent_pattern, min_dur_ms, max_dur_ms, limit,
                       start_time_s=0, tid_filter=None, **kwargs):
    """Find parent NVTX events matching pattern and duration filter.

    min_dur/max_dur always filter by CPU time (NVTX scope duration).
    CPU time filtering is reliable in multi-process traces (no correlationId issues).
    GPU time is used only for child event analysis within correctly selected parents.
    start_time_s: only consider events with start >= this time (in seconds).
    tid_filter: if specified, only consider events from this OS TID (low 24 bits of globalTid).
    """
    parent_ids = get_text_id(cur, text_map, parent_pattern)
    if not parent_ids:
        print(f"Error: No NVTX event matching '{parent_pattern}'")
        print("Available events (sample):")
        for tid, text in sorted(text_map.items())[:30]:
            print(f"{tid}  {text}")
        sys.exit(1)

    id_list = ','.join(str(tid) for tid, _ in parent_ids)
    start_time_ns = int(start_time_s * 1_000_000_000) if start_time_s > 0 else 0

    # Always filter by CPU time (NVTX scope duration) for parent event selection.
    # GPU time filtering is unreliable for parent events in multi-process traces
    # due to correlationId collisions across processes.
    query = f"""
    SELECT start, end, (end-start)/1000000.0 as dur_ms, globalTid, textId
    FROM NVTX_EVENTS
    WHERE textId IN ({id_list})
    """
    conditions = []
    if start_time_ns > 0:
        conditions.append(f"start >= {start_time_ns}")
    if min_dur_ms > 0:
        conditions.append(f"(end-start) >= {int(min_dur_ms * 1000000)}")
    if max_dur_ms > 0:
        conditions.append(f"(end-start) <= {int(max_dur_ms * 1000000)}")
    if tid_filter is not None:
        conditions.append(f"(globalTid & 16777215) = {tid_filter}")
    if conditions:
        query += " AND " + " AND ".join(conditions)
    query += f" ORDER BY start LIMIT {limit}"

    cur.execute(query)
    events = cur.fetchall()
    #print(f"events: {events}")

    return events, {tid: text for tid, text in parent_ids}


def get_gpu_time_for_parent(cur, nvtx_start, nvtx_end, nvtx_tid, global_pid):
    """Get GPU wall-clock time for all kernels launched within a parent NVTX scope.

    Returns (gpu_wall_clock_ns, kernel_count).
    GPU wall-clock = last_kernel_end - first_kernel_start (matches nsys GUI "Duration").
    """
    cur.execute("""
    SELECT correlationId FROM CUPTI_ACTIVITY_KIND_RUNTIME
    WHERE globalTid = ? AND start >= ? AND start < ?
    """, (nvtx_tid, nvtx_start, nvtx_end))
    corr_ids = [r[0] for r in cur.fetchall()]

    if not corr_ids:
        return 0, 0

    first_start = None
    last_end = None
    count = 0
    chunk_size = 900
    for i in range(0, len(corr_ids), chunk_size):
        chunk = corr_ids[i:i + chunk_size]
        placeholders = ','.join('?' * len(chunk))
        cur.execute(f"""
        SELECT start, end FROM CUPTI_ACTIVITY_KIND_KERNEL
        WHERE correlationId IN ({placeholders}) AND globalPid = ?
        """, chunk + [global_pid])
        for row in cur.fetchall():
            count += 1
            if first_start is None or row[0] < first_start:
                first_start = row[0]
            if last_end is None or row[1] > last_end:
                last_end = row[1]

    if first_start is None:
        return 0, 0

    return last_end - first_start, count


def batch_get_gpu_times(cur, parent_start, parent_end, parent_tid, global_pid):
    """Batch fetch all correlationId -> kernel durations for a parent time range.

    Returns (corr_to_runtime_start, corr_to_duration) where:
      corr_to_runtime_start: dict correlationId -> runtime API start timestamp
      corr_to_duration: dict correlationId -> total kernel duration_ns (summed if multiple kernels)

    IMPORTANT: filters KERNEL table by globalPid to avoid cross-process correlationId collisions.
    """
    # Step 1: get all runtime correlationIds in the parent range (one query)
    cur.execute("""
    SELECT correlationId, start FROM CUPTI_ACTIVITY_KIND_RUNTIME
    WHERE globalTid = ? AND start >= ? AND start < ?
    ORDER BY start
    """, (parent_tid, parent_start, parent_end))
    runtime_rows = cur.fetchall()

    if not runtime_rows:
        return {}

    corr_ids = [r[0] for r in runtime_rows]
    # Map correlationId -> runtime_start (for associating with child NVTX ranges)
    corr_to_runtime_start = {r[0]: r[1] for r in runtime_rows}

    # Step 2: batch query kernel durations, filtered by globalPid
    # NOTE: A single correlationId can map to multiple kernels (e.g. cuBLAS sub-kernels),
    # so we SUM durations per correlationId.
    corr_to_duration = defaultdict(int)
    chunk_size = 900
    for i in range(0, len(corr_ids), chunk_size):
        chunk = corr_ids[i:i + chunk_size]
        placeholders = ','.join('?' * len(chunk))
        cur.execute(f"""
        SELECT correlationId, (end - start) FROM CUPTI_ACTIVITY_KIND_KERNEL
        WHERE correlationId IN ({placeholders}) AND globalPid = ?
        """, chunk + [global_pid])
        for row in cur.fetchall():
            corr_to_duration[row[0]] += row[1]

    return corr_to_runtime_start, dict(corr_to_duration)


def _get_direct_children(cur, parent_start, parent_end, parent_tid, child_patterns=None):
    """Get direct children (nesting depth=1) of a parent NVTX event.

    Uses stack-based depth calculation. If child_patterns is specified and no
    matching events are found at depth<=1, progressively searches deeper levels
    until matches are found.
    Returns list of (textId, start, end).
    """
    cur.execute("""
    SELECT textId, start, end
    FROM NVTX_EVENTS
    WHERE start >= ? AND end <= ?
    AND globalTid = ?
    AND textId IS NOT NULL
    ORDER BY start, (start - end)
    """, (parent_start, parent_end, parent_tid))
    all_events = cur.fetchall()

    if not all_events:
        return []

    # Sort by (start ASC, duration DESC) so wider ranges come before narrower at same start
    all_events.sort(key=lambda e: (e[1], -(e[2] - e[1])))

    # Compute depth for each event using a stack of end times
    stack = []
    events_with_depth = []  # (depth, textId, start, end)

    for text_id, s, e in all_events:
        while stack and stack[-1] <= s:
            stack.pop()
        depth = len(stack)

        # Skip the parent event itself
        if depth == 0 and s == parent_start and e == parent_end:
            stack.append(e)
            continue

        events_with_depth.append((depth, text_id, s, e))
        stack.append(e)

    if not events_with_depth:
        return []

    # Default: return depth<=1 (direct children)
    direct_children = [(tid, s, e) for d, tid, s, e in events_with_depth if d <= 1]

    if not child_patterns:
        return direct_children

    # Find the depth that covers ALL specified patterns (or as many as possible).
    # Strategy: expand depth until all patterns are matched, or max depth reached.
    cur.execute("SELECT id, value FROM StringIds")
    text_map_local = {r[0]: r[1] for r in cur.fetchall()}

    max_depth = max(d for d, _, _, _ in events_with_depth)
    best_depth = 1
    best_match_count = 0

    for target_depth in range(1, max_depth + 1):
        candidates = [(tid, s, e) for d, tid, s, e in events_with_depth if d <= target_depth]
        # Count how many patterns have at least one match at this depth
        matched_patterns = set()
        for tid, s, e in candidates:
            text = text_map_local.get(tid, "")
            for p in child_patterns:
                if p in text:
                    matched_patterns.add(p)
        if len(matched_patterns) > best_match_count:
            best_match_count = len(matched_patterns)
            best_depth = target_depth
        # If all patterns matched, stop
        if len(matched_patterns) == len(child_patterns):
            break

    if best_match_count == 0:
        # No match at any depth, return all
        return [(tid, s, e) for _, tid, s, e in events_with_depth]

    return [(tid, s, e) for d, tid, s, e in events_with_depth if d <= best_depth]


def analyze_children_gpu(cur, text_map, parent_start, parent_end, parent_tid,
                         child_patterns=None, global_pid=None):
    """Analyze direct child events' GPU kernel time within a parent event.

    Only counts depth=1 (direct children) to avoid double-counting nested events.
    Uses batch query to avoid per-child SQL roundtrips.
    Filters by globalPid for multi-process correctness.
    """
    # Batch fetch: all runtime correlationIds + kernel durations in parent range
    batch_result = batch_get_gpu_times(cur, parent_start, parent_end, parent_tid, global_pid)
    if not batch_result:
        return {}
    corr_to_runtime_start, corr_to_duration = batch_result

    # Get only direct children (depth=1), expanding deeper if patterns not found
    rows = _get_direct_children(cur, parent_start, parent_end, parent_tid, child_patterns)

    # Pre-sort runtime starts for binary search
    sorted_runtime = sorted(corr_to_runtime_start.items(), key=lambda x: x[1])
    runtime_starts = [r[1] for r in sorted_runtime]
    runtime_corr_ids = [r[0] for r in sorted_runtime]

    # Group by text and compute GPU time for each child
    # child_patterns is used for depth search AND output filtering
    groups = defaultdict(list)  # text -> list of gpu_time_us
    for text_id, ns, ne in rows:
        text = text_map.get(text_id, f"textId={text_id}")

        # Filter: only output events matching child_patterns
        if child_patterns:
            if not any(p in text for p in child_patterns):
                continue

        # Find runtime calls within [ns, ne) using binary search
        left = bisect.bisect_left(runtime_starts, ns)
        right = bisect.bisect_left(runtime_starts, ne)

        gpu_ns = 0
        for idx in range(left, right):
            cid = runtime_corr_ids[idx]
            gpu_ns += corr_to_duration.get(cid, 0)

        groups[text].append(gpu_ns / 1000.0)  # convert to us

    return dict(groups)


def analyze_children_gpu_with_kernels(cur, text_map, parent_start, parent_end, parent_tid,
                                      child_patterns=None, global_pid=None):
    """Like analyze_children_gpu but also returns per-child kernel details.

    Returns (groups, kernel_details) where:
      groups: dict text -> list of gpu_time_us (same as analyze_children_gpu)
      kernel_details: dict text -> dict kernel_name -> (total_ns, count)
    """
    # Batch fetch runtime correlationIds
    cur.execute("""
    SELECT correlationId, start FROM CUPTI_ACTIVITY_KIND_RUNTIME
    WHERE globalTid = ? AND start >= ? AND start < ?
    ORDER BY start
    """, (parent_tid, parent_start, parent_end))
    runtime_rows = cur.fetchall()

    if not runtime_rows:
        return {}, {}

    corr_ids = [r[0] for r in runtime_rows]
    corr_to_runtime_start = {r[0]: r[1] for r in runtime_rows}

    # Batch fetch kernel info: correlationId -> list of (shortName_id, duration_ns)
    corr_to_kernels = defaultdict(list)
    chunk_size = 900
    for i in range(0, len(corr_ids), chunk_size):
        chunk = corr_ids[i:i + chunk_size]
        placeholders = ','.join('?' * len(chunk))
        cur.execute(f"""
        SELECT correlationId, shortName, (end - start)
        FROM CUPTI_ACTIVITY_KIND_KERNEL
        WHERE correlationId IN ({placeholders}) AND globalPid = ?
        """, chunk + [global_pid])
        for row in cur.fetchall():
            corr_to_kernels[row[0]].append((row[1], row[2]))

    # Build kernel name map (shortName id -> string)
    kernel_name_ids = set()
    for kernels in corr_to_kernels.values():
        for name_id, _ in kernels:
            kernel_name_ids.add(name_id)
    if kernel_name_ids:
        ph = ','.join('?' * len(kernel_name_ids))
        cur.execute(f"SELECT id, value FROM StringIds WHERE id IN ({ph})", list(kernel_name_ids))
        kernel_name_map = {r[0]: r[1] for r in cur.fetchall()}
    else:
        kernel_name_map = {}

    # Get direct children
    rows = _get_direct_children(cur, parent_start, parent_end, parent_tid, child_patterns)

    # Pre-sort runtime starts for binary search
    sorted_runtime = sorted(corr_to_runtime_start.items(), key=lambda x: x[1])
    runtime_starts = [r[1] for r in sorted_runtime]
    runtime_corr_ids = [r[0] for r in sorted_runtime]

    groups = defaultdict(list)
    kernel_details = defaultdict(lambda: defaultdict(lambda: [0, 0]))  # text -> kernel_name -> [total_ns, count]

    for text_id, ns, ne in rows:
        text = text_map.get(text_id, f"textId={text_id}")

        if child_patterns:
            if not any(p in text for p in child_patterns):
                continue

        left = bisect.bisect_left(runtime_starts, ns)
        right = bisect.bisect_left(runtime_starts, ne)

        gpu_ns = 0
        for idx in range(left, right):
            cid = runtime_corr_ids[idx]
            for name_id, dur_ns in corr_to_kernels.get(cid, []):
                gpu_ns += dur_ns
                kname = kernel_name_map.get(name_id, f"kernel_{name_id}")
                kernel_details[text][kname][0] += dur_ns
                kernel_details[text][kname][1] += 1

        groups[text].append(gpu_ns / 1000.0)

    return dict(groups), dict(kernel_details)


def print_summary_with_kernels(children_stats, kernel_details, parent_dur_ms, time_label="GPU",
                               csv_file=None):
    """Print summary with per-child kernel breakdown. Optionally write CSV."""
    if not children_stats:
        print("    (no matching child events)")
        return

    sorted_items = sorted(children_stats.items(), key=lambda x: -sum(x[1]))

    # Write CSV if requested
    if csv_file:
        csv_file.write("event,kernel,count,total_ms,pct%,avg_us,min_us,max_us\n")
        for text, durs in sorted_items:
            total_us = sum(durs)
            avg_us = total_us / len(durs) if durs else 0
            min_us = min(durs) if durs else 0
            max_us = max(durs) if durs else 0
            pct = total_us / (parent_dur_ms * 1000) * 100 if parent_dur_ms > 0 else 0
            csv_file.write(f"{text},,{len(durs)},{total_us/1000:.3f},{pct:.1f},{avg_us:.1f},\
                {min_us:.1f},{max_us:.1f}\n")
            if text in kernel_details:
                kd = kernel_details[text]
                sorted_kernels = sorted(kd.items(), key=lambda x: -x[1][0])
                for kname, (ktotal_ns, kcount) in sorted_kernels:
                    kavg_us = ktotal_ns / kcount / 1000.0 if kcount else 0
                    kpct = (ktotal_ns / 1000.0) / (parent_dur_ms * 1000) * 100 if parent_dur_ms > 0 else 0
                    csv_file.write(f"{text},{kname},{kcount},{ktotal_ns/1e6:.3f},{kpct:.1f},{kavg_us:.1f},,\n")

    # Always print to terminal in human-readable format
    for text, durs in sorted_items:
        total_us = sum(durs)
        avg_us = total_us / len(durs) if durs else 0
        min_us = min(durs) if durs else 0
        max_us = max(durs) if durs else 0
        pct = total_us / (parent_dur_ms * 1000) * 100 if parent_dur_ms > 0 else 0
        print(f"    {text:<50} count={len(durs):>5}  "
              f"{time_label}_total={total_us/1000:>8.2f}ms ({pct:>5.1f}%)  "
              f"avg={avg_us:>8.1f}us  min={min_us:.1f}us  max={max_us:.1f}us")
        if text in kernel_details:
            kd = kernel_details[text]
            sorted_kernels = sorted(kd.items(), key=lambda x: -x[1][0])
            for kname, (ktotal_ns, kcount) in sorted_kernels:
                kavg_us = ktotal_ns / kcount / 1000.0 if kcount else 0
                kpct = (ktotal_ns / 1000.0) / (parent_dur_ms * 1000) * 100 if parent_dur_ms > 0 else 0
                print(f"      └─ {kname:<55} count={kcount:>4}  "
                      f"total={ktotal_ns/1e6:>6.2f}ms ({kpct:>4.1f}%)  avg={kavg_us:.1f}us")


def analyze_children_cpu(cur, text_map, parent_start, parent_end, parent_tid,
                         child_patterns=None, global_pid=None):
    """Analyze direct child events' CPU time within a parent event.

    Only counts depth=1 (direct children) to avoid double-counting nested events.
    If child_patterns not found at depth=1, searches deeper.
    """
    rows = _get_direct_children(cur, parent_start, parent_end, parent_tid, child_patterns)

    # child_patterns is used for depth search AND output filtering
    groups = defaultdict(list)
    for text_id, s, e in rows:
        text = text_map.get(text_id, f"textId={text_id}")
        if child_patterns:
            if not any(p in text for p in child_patterns):
                continue
        groups[text].append((e - s) / 1000.0)  # ns -> us

    return dict(groups)


def print_summary(children_stats, parent_dur_ms, time_label="GPU"):
    """Print formatted summary of children stats."""
    if not children_stats:
        print("    (no matching child events)")
        return

    sorted_items = sorted(children_stats.items(), key=lambda x: -sum(x[1]))
    for text, durs in sorted_items:
        total_us = sum(durs)
        avg_us = total_us / len(durs) if durs else 0
        min_us = min(durs) if durs else 0
        max_us = max(durs) if durs else 0
        pct = total_us / (parent_dur_ms * 1000) * 100 if parent_dur_ms > 0 else 0
        print(f"    {text:<50} count={len(durs):>5}  "
              f"{time_label}_total={total_us/1000:>8.2f}ms ({pct:>5.1f}%)  "
              f"avg={avg_us:>8.1f}us  min={min_us:.1f}us  max={max_us:.1f}us")


def main():
    parser = argparse.ArgumentParser(
        description="NVTX Event Analyzer: 统计指定父事件下子事件的 GPU 执行时间",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('input', help='nsys-rep 或 sqlite 文件路径')
    parser.add_argument('--parent', default=None,
                        help='父事件名 (子串匹配), --list-events 时可省略')
    parser.add_argument('--children', default=None,
                        help='子事件名列表, 逗号分隔 (子串匹配). 不指定则输出所有子事件')
    parser.add_argument('--child', default=None,
                        help='单个子事件过滤 (只显示包含该子事件的父事件)')
    parser.add_argument('--min-dur', type=float, default=0,
                        help='父事件最小持续时间 (ms, 默认GPU时间; --cpu-time时按CPU时间)')
    parser.add_argument('--max-dur', type=float, default=0,
                        help='父事件最大持续时间 (ms, 同上), 0=不限')
    parser.add_argument('--start-time', type=float, default=0,
                        help='只分析 start >= 该时间(秒)的事件, 用于跳过 warmup')
    parser.add_argument('--tid', type=int, default=None,
                        help='只分析指定 TID (OS线程ID, 与 nsys GUI 中显示的一致)')
    parser.add_argument('--kernels', action='store_true',
                        help='列出每个子事件对应的 GPU kernel 名称及耗时')
    parser.add_argument('--limit', type=int, default=20,
                        help='最多分析多少个父事件 (default: 20)')
    parser.add_argument('--top', type=int, default=5,
                        help='输出前 N 个父事件的详细结果 (default: 5)')
    parser.add_argument('--detail', action='store_true',
                        help='输出每个父事件的子事件明细')
    parser.add_argument('--aggregate', action='store_true',
                        help='聚合所有匹配父事件的子事件统计')
    parser.add_argument('--cpu-time', action='store_true',
                        help='使用 CPU 侧 NVTX 时间 (默认使用 GPU kernel 时间)')
    parser.add_argument('--list-events', action='store_true',
                        help='列出所有 NVTX 事件名称 (用于查看可用的 --parent 值)')
    parser.add_argument('--list-children', action='store_true',
                        help='列出 parent event 下所有直接子事件名 (去重, 去尾部数字)')
    parser.add_argument('--csv', type=str, default=None, metavar='FILE',
                        help='将结果输出到 CSV 文件 (同时在终端保留原格式输出)')

    args = parser.parse_args()

    # Prepare sqlite
    sqlite_path = ensure_sqlite(args.input)
    db = sqlite3.connect(sqlite_path)
    cur = db.cursor()

    # Build indices for fast queries (critical for large profiles)
    if not args.cpu_time:
        ensure_indices(cur)

    # Build process mapping for multi-process traces (correlationId is NOT unique across processes)
    global_pids = None
    if not args.cpu_time:
        global_pids = build_pid_map(cur)
        if len(global_pids) > 1:
            print(f"  Multi-process trace detected: {len(global_pids)} GPU processes")

    # Build text map
    text_map = build_text_map(cur)
    #print(f"text_map {text_map}")

    # --list-events: print all NVTX event names and exit
    if args.list_events:
        # Only show text entries that are actually used as NVTX event textIds
        cur.execute("SELECT DISTINCT textId FROM NVTX_EVENTS WHERE textId IS NOT NULL")
        used_text_ids = set(r[0] for r in cur.fetchall())
        nvtx_names = sorted(set(text_map[tid] for tid in used_text_ids if tid in text_map))
        print(f"NVTX event names ({len(nvtx_names)} entries):")
        for name in nvtx_names:
            print(f"  {name}")
        db.close()
        return

    # --parent is required for --list-children and analysis
    if not args.parent:
        print("Error: --parent is required (use --list-events to see available event names)")
        db.close()
        sys.exit(1)

    # --list-children: list all direct child event names under parent (deduplicated, trailing numbers stripped)
    if args.list_children:
        parent_events, parent_names = find_parent_events(
            cur, text_map, args.parent, args.min_dur, args.max_dur, args.limit,
            start_time_s=args.start_time, tid_filter=args.tid)
        if not parent_events:
            print(f"No events matching '{args.parent}'")
            db.close()
            return

        # Collect all direct children names from sampled parent events
        all_child_names = set()
        sample_count = min(5, len(parent_events))  # sample a few parents
        for evt in parent_events[:sample_count]:
            children = _get_direct_children(cur, evt[0], evt[1], evt[3])
            for text_id, _, _ in children:
                name = text_map.get(text_id, f"textId={text_id}")
                all_child_names.add(name)

        # Strip trailing numbers (and underscores before them)
        # e.g. "DS_CommonMlp_seq_len_8_hidden_units_4096" → "DS_CommonMlp_seq_len_hidden_units"
        #      "expert_gate" → "expert_gate" (no trailing number)
        def strip_trailing_numbers(name):
            return re.sub(r'(_\d+)+$', '', name)

        stripped_names = sorted(set(strip_trailing_numbers(n) for n in all_child_names))

        parent_name_str = ', '.join(f"'{v}'" for v in parent_names.values())
        print(f"\nDirect children of '{parent_name_str}' ({len(stripped_names)} unique names):")
        for name in stripped_names:
            print(f"  {name}")
        db.close()
        return

    # Parse child patterns
    child_patterns = None
    if args.children:
        child_patterns = [p.strip() for p in args.children.split(',')]

    # Select analysis function
    if args.cpu_time:
        analyze_fn = analyze_children_cpu
        time_label = "CPU"
    else:
        analyze_fn = analyze_children_gpu
        time_label = "GPU"

    # Find parent events (always filtered by CPU time for reliability)
    parent_events, parent_names = find_parent_events(
        cur, text_map, args.parent, args.min_dur, args.max_dur, args.limit,
        start_time_s=args.start_time, tid_filter=args.tid)

    if not parent_events:
        print(f"No events matching '{args.parent}' with {time_label} dur in [{args.min_dur}, {args.max_dur}]ms")
        db.close()
        return

    #  开始时间 结束时间 持续时间 globalTid textId
    #print(f"parent_events: {parent_events}")

    parent_name_str = ', '.join(f"'{v}'" for v in parent_names.values())
    first_start_ns = parent_events[0][0]
    print(f"\n{'='*80}")
    print(f"Parent event: {parent_name_str}")
    print(f"First event start: {first_start_ns/1e9:.6f}s (raw: {first_start_ns} ns)")
    print(f"Filter: CPU dur >= {args.min_dur}ms" + (f", dur <= {args.max_dur}ms" if args.max_dur > 0 else ""))
    print(f"Time mode: {time_label}")
    print(f"Found: {len(parent_events)} events")
    print(f"{'='*80}")
    print(f"parent_names: {parent_names}")

    # If --child filter is specified, only show parents containing that child
    if args.child:
        filtered_parents = []
        for evt in parent_events:
            gpid = tid_to_pid(global_pids, evt[3]) if global_pids else None
            children = analyze_fn(cur, text_map, evt[0], evt[1], evt[3], [args.child],
                                  global_pid=gpid)
            if children and any(durs for durs in children.values()):
                filtered_parents.append(evt)
        parent_events = filtered_parents
        print(f"parent_events2: {parent_events}")
        print(f"After filtering by child '{args.child}': {len(parent_events)} events")

    # Open CSV file if requested
    csv_file = None
    if args.csv and args.kernels:
        csv_file = open(args.csv, 'w')
        print(f"  CSV output: {args.csv}")

    # Aggregate mode
    if args.aggregate:
        print(f"\n--- Aggregated {time_label} stats across {len(parent_events)} parent events ---")
        all_children = defaultdict(list)
        all_kernel_details = defaultdict(lambda: defaultdict(lambda: [0, 0]))
        total_parent_dur_ms = 0
        total_parent_gpu_ms = 0
        for evt in parent_events:
            parent_dur_ms = evt[2]
            total_parent_dur_ms += parent_dur_ms
            gpid = tid_to_pid(global_pids, evt[3]) if global_pids else None
            if args.kernels and not args.cpu_time:
                children, kd = analyze_children_gpu_with_kernels(
                    cur, text_map, evt[0], evt[1], evt[3], child_patterns, global_pid=gpid)
                for text, kinfo in kd.items():
                    for kname, (ktotal, kcnt) in kinfo.items():
                        all_kernel_details[text][kname][0] += ktotal
                        all_kernel_details[text][kname][1] += kcnt
            else:
                children = analyze_fn(cur, text_map, evt[0], evt[1], evt[3], child_patterns,
                                      global_pid=gpid)
            for text, durs in children.items():
                all_children[text].extend(durs)
            # Compute parent GPU time for percentage
            if not args.cpu_time and gpid is not None:
                gpu_ns, _ = get_gpu_time_for_parent(cur, evt[0], evt[1], evt[3], gpid)
                total_parent_gpu_ms += gpu_ns / 1_000_000.0
        pct_base_ms = total_parent_gpu_ms if (not args.cpu_time and total_parent_gpu_ms > 0) else total_parent_dur_ms
        print(f"  Total parent CPU duration: {total_parent_dur_ms:.2f}ms")
        if not args.cpu_time:
            print(f"  Total parent GPU duration: {total_parent_gpu_ms:.2f}ms")
        if args.kernels and not args.cpu_time:
            print_summary_with_kernels(all_children, all_kernel_details, pct_base_ms, time_label,
                                       csv_file=csv_file)
        else:
            print_summary(all_children, pct_base_ms, time_label)

    # Per-event detail
    show_count = min(args.top, len(parent_events))
    if args.detail or not args.aggregate:
        print(f"\n--- Top {show_count} events ---")
        for i, evt in enumerate(parent_events[:show_count]):
            parent_dur_ms = evt[2]
            gpid = tid_to_pid(global_pids, evt[3]) if global_pids else None
            parent_text = text_map.get(evt[4], f"textId={evt[4]}")
            # Compute parent GPU time
            parent_gpu_ms = 0
            if not args.cpu_time and gpid is not None:
                gpu_ns, _ = get_gpu_time_for_parent(cur, evt[0], evt[1], evt[3], gpid)
                parent_gpu_ms = gpu_ns / 1_000_000.0
            pct_base_ms = parent_gpu_ms if (not args.cpu_time and parent_gpu_ms > 0) else parent_dur_ms
            print(f"\n  [{i+1}] {parent_text}: start={evt[0]/1e9:.6f}s, CPU dur={parent_dur_ms:.2f}ms, \
                GPU dur={parent_gpu_ms:.2f}ms, TID={evt[3] & 0xFFFFFF}, textId={evt[4]}")
            if args.kernels and not args.cpu_time:
                children, kd = analyze_children_gpu_with_kernels(
                    cur, text_map, evt[0], evt[1], evt[3], child_patterns, global_pid=gpid)
                print_summary_with_kernels(children, kd, pct_base_ms, time_label,
                                           csv_file=csv_file)
            else:
                children = analyze_fn(cur, text_map, evt[0], evt[1], evt[3], child_patterns,
                                      global_pid=gpid)
                print_summary(children, pct_base_ms, time_label)

    # Close CSV file
    if csv_file:
        csv_file.close()
        print(f"\n  CSV written to: {args.csv}")

    # Summary statistics
    if len(parent_events) > 1:
        durs = [e[2] for e in parent_events]
        print(f"\n--- Parent event CPU duration summary (N={len(durs)}) ---")
        print(f"  avg={sum(durs)/len(durs):.2f}ms, "
              f"min={min(durs):.2f}ms, max={max(durs):.2f}ms, "
              f"med={sorted(durs)[len(durs)//2]:.2f}ms")

    db.close()


if __name__ == '__main__':
    main()
