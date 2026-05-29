# nvtx_analyzer.py 使用指南

## 主要功能

支持从nsys profiling结果，可以根据nvtx event分析推理各个阶段的性能，并且结合配置参数，能够找出单独分析Prfill阶段、Decode阶段的kernel性能。

## 基本用法

```bash
python nvtx_analyzer.py <sqlite或nsys-rep文件> --parent <父事件名> [选项]
```

## 常用选项

| 选项                     | 说明                                                                            |
| ------------------------ | ------------------------------------------------------------------------------- |
| `--parent PATTERN`     | 父事件名（子串匹配），必选，用于指出要分析的最上层执行阶段，例如DS_LayerForward |
| `--children P1,P2,...` | 只输出匹配的子事件（也用于确定搜索深度）                                        |
| `--min-dur MS`         | 父事件最小 GPU 持续时间                                                         |
| `--max-dur MS`         | 父事件最大 GPU 持续时间，用于筛选decode或者prefill的执行阶段                    |
| `--tid TID`            | 只分析指定线程（与 nsys GUI 中 TID 一致）                                       |
| `--start-time S`       | 跳过该时间(秒)之前的事件                                                        |
| `--limit N`            | 最多分析 N 个父事件（默认 20）                                                  |
| `--aggregate`          | 聚合所有父事件的统计结果                                                        |
| `--detail`             | 输出每个父事件的明细                                                            |
| `--kernels`            | 列出每个子事件内的 GPU kernel 名及耗时                                          |
| `--csv FILE`           | 输出 CSV 文件（需配合 --kernels）                                               |
| `--cpu-time`           | 使用 CPU 侧 NVTX 时间（不关联 kernel）                                          |
| `--list-events`        | 列出所有 NVTX 事件名，协助找到某个事件下的子事件                                |
| `--list-children`      | 列出父事件的直接子事件名（去尾部数字），协助找到某个事件下的子事件              |
| --top N                  | 统计父事件的次数，默认是20                                                      |

## 示例

### 查看DS4模型中，父事件DS_LayerForward下的kernel执行时间，关注decode阶段

```bash


python KsanaLLM/tools/nvtx_analyzer.py \   
  report25.nsys-rep     \   
  --min-dur 10 --max-dur 28  \   # decode的耗时在20ms左右   
  --top 100 \   
  --parent "DS_LayerForward"  \   
  --children "MOE,CommonSharedMlp,expert_gate,add_layer,Mhc_PreForwardWithNorm,dsv4_mqa_layer,\   
  TpCommunicator_AllReduce,Mhc_PostForward,Mhc_PreForwardWithNorm,two_layered"   \  
  --aggregate \   
  --kernels \ 
  --csv perf.csv

```


### 定位目标事件发生的时间，去掉--aggregate,会列出每个父事件发生的时间，用于排查细节或者结果正确性

```bash
python KsanaLLM/tools/nvtx_analyzer.py \   
  report25.nsys-rep     \   
  --min-dur 10 --max-dur 28  \   # decode的耗时在20ms左右   
  --top 100 \   
  --parent "DS_LayerForward"  \   
  --children "MOE,CommonSharedMlp,expert_gate,add_layer,Mhc_PreForwardWithNorm,dsv4_mqa_layer,\   
  TpCommunicator_AllReduce,Mhc_PostForward,Mhc_PreForwardWithNorm,two_layered"   \  
  --limit 2 \   
  --kernels \ 
  --csv perf.csv
```

### 查看有哪些事件可用

```
python nvtx_analyzer.py report.sqlite --list-events
```

### 查看某个父事件下有哪些子事件

python nvtx_analyzer.py report.sqlite --parent "DS_LayerForward" --list-children

### 分析 decode 阶段各组件 GPU 时间（聚合）

python nvtx_analyzer.py report.sqlite 
    --parent "DS_LayerForward" --min-dur 20 --max-dur 25
    --children "MOE,CommonShareMlp,expert_gate,TpCommunicator_AllReduce,dsv4_mqa_layer"
    --aggregate --limit 10

### 指定 TID，查看单个事件详情 + kernel 明细

python nvtx_analyzer.py report.sqlite 
    --parent "DS_LayerForward" --tid 2389941 --min-dur 20 --max-dur 22
    --children "MOE,CommonShareMlp,expert_gate"
    --kernels --limit 1

### 输出到 CSV 文件

python nvtx_analyzer.py report.sqlite
    --parent "DS_LayerForward" --min-dur 20 --max-dur 22
    --children "MOE,CommonShareMlp,expert_gate"
    --kernels --aggregate --csv result.csv

## 输出说明

- **GPU dur**：parent 的 GPU wall-clock 时间（first kernel start → last kernel end），与 nsys GUI 一致
- **pct%**：子事件 GPU 时间 / parent GPU wall-clock 时间
- **`--children` 行为**：用于确定搜索深度（自动穿透嵌套层级），同时过滤输出
- 不指定 `--children` 则输出所有直接子事件（depth=1）

## 注意事项

- 多进程 trace 自动处理 correlationId 跨进程冲突（通过 globalPid 过滤）
- 首次运行会创建索引（几秒），后续查询很快
- `--children` 中的名称是子串匹配，注意拼写（如 `CommonShareMlp` 非 `CommonSharedMlp`）
