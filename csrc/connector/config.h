/* Copyright 2025 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <string>
#include <vector>

namespace ksana_llm {
/**
 * @enum CommunicationType
 * @brief Defines the available communication protocols for the connector
 */
enum class CommunicationType {
  NCCL, /**< NVIDIA Collective Communications Library for GPU communication */
  TCP,  /**< Transmission Control Protocol for general network communication */
  MPI,  /**< Message Passing Interface for distributed computing */
  SHM,  /**< Shared Memory for inter-process communication on the same machine */
  ZMQ   /**< Remote Procedure Call for service-oriented communication */
};

/**
 * @enum GroupRole
 * @brief Defines the role of a node in the prefill-decode pipeline
 */
enum class GroupRole {
  PREFILL, /**< Node responsible for prefill operations */
  DECODE,  /**< Node responsible for decode operations */
  BOTH,    /**< Node performs both prefill and decode operations */
  NONE     /**< No specific role assigned */
};

/**
 * @brief Convert GroupRole enum to string representation
 * @param role The GroupRole enum value to convert
 * @return String representation of the role
 */
inline std::string GroupRoleToString(GroupRole role) {
  switch (role) {
    case GroupRole::NONE:
      return "none";
    case GroupRole::PREFILL:
      return "prefill";
    case GroupRole::DECODE:
      return "decode";
    default:
      return "none";
  }
}
/**
 * @struct ConnectorConfig
 * @brief Configuration parameters for connector initialization
 *
 * Contains all the necessary parameters to set up a communication connector
 * between prefill and decode components in the KsanaLLM framework.
 */
struct ConnectorConfig {
  /** @brief Router address (e.g., "http://localhost:8000") */
  std::string router_addr;

  /**
   * @brief Cluster name for this connector
   *
   * All prefill and decode nodes under the same cluster name will be grouped together
   * and every prefill node will communicate with decode nodes in the same cluster.
   */
  std::string cluster_name;

  /** @brief Connector role (prefill, decode, or both) */
  GroupRole group_role = GroupRole::NONE;

  // ============================================================
  // DEPRECATED pd_v2 fields below.
  //
  // pd_v2 callers must read PdV2RuntimeConfig
  // (csrc/pd_v2/configure/pd_v2_runtime_config.h) via
  // Environment::GetPdV2RuntimeConfig. ConnectorConfig is V1-only.
  //
  // Mixing them caused memory K7 (V1 `is_decode_only = (group_role ==
  // DECODE)` derivation propagated into pd_v2 Decode and corrupted
  // forwarding buffers under local-bypass prefill).
  //
  // The fields are still populated by ScheduleConfigParser for
  // back-compat (existing callers / tests reading them are not yet
  // migrated). Deletion in a follow-up PR.
  // ============================================================

  /**
   * @brief Whether this deployment runs pd_v2 (token-based PD disaggregation
   * over Mooncake). Mirrors `setting.connector.backend == "v2"` in YAML.
   * pd_v2 code paths (e.g. executor KV pool contiguous allocation) gate on
   * this flag rather than env vars. Default false preserves V1 behavior.
   */
  bool is_pd_v2 = false;

  /**
   * @brief Mooncake metadata server connection string (pd_v2 only).
   * Mirrors `setting.connector.metadata_addr` in YAML. Forwarded to
   * mooncake::TransferEngine::init() at connector Initialize. Empty
   * string defaults to "P2PHANDSHAKE" (single-machine, no metadata
   * server needed). See PdV2ConnectorConfig.metadata_addr docstring
   * for supported schemes.
   */
  std::string metadata_addr;

  /**
   * @brief Redis password for metadata_addr=redis://... (pd_v2 only).
   * Forwarded by InferenceEngine/InferenceExecutor bootstrap to env var
   * MC_REDIS_PASSWORD which Mooncake reads at TransferEngine::init time
   * and uses for AUTH against the Redis server's default user. Mirrors
   * YAML `setting.connector.redis_password`. Empty = no AUTH issued.
   */
  std::string redis_password;

  /**
   * @brief pd_v2 transfer chunk size (number of model layers per
   * Mooncake submitTransfer batch). 0 = transfer everything at the
   * model's final layer only (one big batch). N>0 = a batch fires
   * every N layers (smaller batches, lower latency, more overhead).
   * Default 1 (per-layer) — matches the design doc's reference
   * configuration. Mirrors YAML
   * `setting.connector.transfer_layer_chunk_size`. Ignored when
   * is_pd_v2 == false.
   */
  uint32_t transfer_layer_chunk_size = 1;

  /**
   * @brief pd_v2 Executor Mooncake RPC base port. Per-rank port is
   *        assigned as `base + executor_rank` so multi-rank TP
   *        deployments (one executor process per GPU on a host)
   *        don't collide on a single port. Default 20000; mirrors
   *        YAML `setting.connector.pd_v2_executor_rpc_base_port`.
   *        Ignored when is_pd_v2 == false.
   */
  uint32_t pd_v2_executor_rpc_base_port = 20000;

  /**
   * @brief pd_v2 Decode-side "remaining prefill ≤ N stays local" threshold
   * (tokens). When (input_tokens.size() − prefix_cache_len) ≤ this value,
   * Decode bypasses Prefill and runs the residual prefill + decode locally.
   * Default 0 disables the fast-path (always forward). Mirrors YAML
   * `setting.connector.pd_v2_local_complete_threshold`. Ignored when
   * is_pd_v2 == false or group_role != DECODE.
   */
  int pd_v2_local_complete_threshold = 0;

  /**
   * @brief pd_v2 Decode-side peer-selection policy. 0=kRoundRobin (default),
   * 1=kLinUCB. kLinUCB enables the contextual-bandit LOCAL-vs-REMOTE router
   * (remote peer still chosen by the round-robin path). Mirrors YAML
   * `setting.connector.pd_v2_selector_policy`.
   * Ignored when is_pd_v2 == false or group_role != DECODE.
   */
  int pd_v2_selector_policy = 0;

  /** @brief LinUCB exploration weight (selector_policy==kLinUCB only).
   *  YAML `setting.connector.pd_v2_linucb_alpha`. */
  double pd_v2_linucb_alpha = 1.0;
  /** @brief LinUCB TPOT budget in us. YAML `pd_v2_linucb_tpot_target_us`. */
  double pd_v2_linucb_tpot_target_us = 50000.0;
  /** @brief LinUCB LOCAL gate aims at tpot_target_us * this (headroom for the
   *  prefill-local TPOT bump). <1 = TPOT prioritized harder. Default 0.85.
   *  YAML `pd_v2_linucb_tpot_safety_factor`. */
  double pd_v2_linucb_tpot_safety_factor = 0.85;
  /** @brief LinUCB TTFT budget in us. YAML `pd_v2_linucb_ttft_threshold_us`. */
  double pd_v2_linucb_ttft_threshold_us = 1500000.0;
  /** @brief LinUCB "fully busy" running_reqs reference. YAML `pd_v2_linucb_running_ref`. */
  double pd_v2_linucb_running_ref = 16.0;
  /** @brief LinUCB "too big to keep local" remaining-token reference.
   *  YAML `pd_v2_linucb_remaining_ref_tokens`. */
  double pd_v2_linucb_remaining_ref_tokens = 4096.0;
  /** @brief LinUCB round-trip latency (us) mapping to REMOTE reward 0.
   *  YAML `pd_v2_linucb_remote_lat_ref_us`. */
  double pd_v2_linucb_remote_lat_ref_us = 200000.0;

  /** @brief Type of communication protocol to use */
  CommunicationType communication_type = CommunicationType::NCCL;

  /** @brief Heartbeat interval in milliseconds */
  int heartbeat_interval_ms = 5000;

  std::string coordinator_addr = ""; /**< @brief Address for coordinator service */

  std::string inference_addr = ""; /**< @brief Inference address (IP:port format) */

  int node_rank = -1;
  int device_count = -1;           /**< @brief Number of devices in the group, used for attention. */
  int world_size = -1;             /**< @brief Total number of devices for group prefill and decode. */
  int transfer_batch = 1048576;    /**< @brief Batch size for transfer */
  int send_thread_num = 4;         /**< @brief Number of threads for sending tasks */
  int connector_waiting_sec = 3;   /**< @brief Timeout for connector operations in milliseconds */
  int task_expire_sec = 300;       /**< @brief Expiration time for tasks in seconds */
  int circular_bucket_size = 8192; /**< @brief Size of the circular buffer for task keys */
  int circular_bucket_num = 4;     /**< @brief Number of buckets for task management, used for circular buffer */
  int circular_thread_num = 4;     /**< @brief Number of threads for TBB task arena, used for parallel operations */

  /**
   * @brief Convert CommunicationType enum to string representation
   * @param type The CommunicationType enum value to convert
   * @return String representation of the communication type
   */
  static std::string CommunicationTypeToString(CommunicationType type) {
    switch (type) {
      case CommunicationType::NCCL:
        return "NCCL";
      case CommunicationType::ZMQ:
        return "ZMQ";
      default:
        return "unknown";
    }
  }

  /**
   * @brief Convert the configuration to a string representation for debugging
   * @return String representation of the configuration
   */
  std::string toString() const {
    std::string result = "ConnectorConfig {\n";
    result += "  router_addr: " + router_addr + "\n";
    result += "  cluster_name: " + cluster_name + "\n";
    result += "  communication_type: " + CommunicationTypeToString(communication_type) + "\n";
    result += "  heartbeat_interval_ms: " + std::to_string(heartbeat_interval_ms) + "\n";
    result += "  coordinator_addr: " + coordinator_addr + "\n";
    result += "  inference_addr: " + inference_addr + "\n";
    result += "  node_rank: " + std::to_string(node_rank) + "\n";
    result += "  device_count: " + std::to_string(device_count) + "\n";
    result += "  group_role: " + GroupRoleToString(group_role) + "\n";
    result += "  connector_waiting_sec: " + std::to_string(connector_waiting_sec) + "\n";
    result += "  task_expire_sec: " + std::to_string(task_expire_sec) + "\n";
    result += "  send_thread_num: " + std::to_string(send_thread_num) + "\n";
    result += "  world_size: " + std::to_string(world_size) + "\n";
    result += "  transfer_batch: " + std::to_string(transfer_batch) + "\n";
    result += "}";
    return result;
  }
};

}  // namespace ksana_llm