## Summary Card【必读】
- 改动范围：commit `6a1fe8d724e5a35c405dbb4d6cf84aa2875906b2`，89 文件，+5868/−651。模块：global_cache_connector、put worker、cache_manager、batch_scheduler fetcher/tracker/query hook、IPC、schedule parser、serving bootstrap、README/yaml
- 调用链摘要：hook+endpoint 启用 → Query 走 IPC long-task → connector Get（FA exist → SWA 对齐边界 → 洞/边界内拉取）→ DropUnbackedSwaBlocks；完整 prefill 后异步 Put（快照 + Prefix hold）→ 跨卡累加 → EndSwaPutHold
- 兼容风险：默认关闭（无 endpoint / 无 hook）。开启后 yaml 端口必须与 CLI `--mooncake-*-port` 一致。结构体 vs yaml 的 `swa_tail_block_num` 默认不一致（4 vs 2）
- 测试前置：单测不需真 Store（mock）；E2E 需 `ENABLE_GLOBAL_CACHE_HOOK=1` + master。本会话未跑 compile/unit
- 需回源：clarification §E I2–I9；change-plan §G 默认值分叉与 hold 单测缺口

## Verification Handoff【必填】
- 本地自检能力：无编译条件，未验证（本会话只补文档，未在 `dpv3_mn_test` 重跑）
- 增量单测范围（unit_test_targets）：
  - global_cache_connector_test
  - global_cache_put_worker_test
  - global_cache_put_tracker_test
  - global_cache_fetcher_test
  - global_cache_query_callable_test
  - schedule_config_parser_test
  - prefix_cache_manager_test
  - direct_cache_manager_test
- 覆盖缺口对照：
  - T01–T08：对应单测已在提交内 — 本会话未执行
  - T09 serving bootstrap：仅 manual，无单测
  - BeginSwaPutHold 真实语义：无单测（tracker 只 mock End 被调用）
- 是否需要集成测试：是；`pending_integration_decision`（`scripts/e2e/mooncake_cache_stages/` 人工触发）
- 已知不确定项：结构体/yaml `swa_tail_block_num` 默认分叉可能导致「直接构造 config 的测试」与 serving 行为不一致（可能导致测试失败语义偏差，非编译失败）
