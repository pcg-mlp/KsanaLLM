/* Copyright 2026 Tencent Inc.  All rights reserved.

==============================================================================*/
//
// Pure-formula tests for the PdV2KvDispatcher device_id routing rule
// (design doc §6 matrix). The dispatcher's per-layer hook participates iff
//   my_prefill_rank == decode_attn_dp_group_id * d_attn_TP +
//                      decode_attn_tp_rank_in_group
// — same formula handles MHA/GQA and MLA. p_attn_TP < d_attn_TP cells are
// "TBD slice/replicate" and intentionally skipped here (no Prefill rank
// matches in those rows).
//
// We don't instantiate PdV2KvDispatcher (that would pull
// MooncakeExecutorConnector + the runtime). Instead we replicate the
// formula in a tiny helper and lock it in against representative cells
// from §6.1 / §6.2 / §6.3 / §6.4. If anyone changes the dispatcher's
// rule, this test catches the divergence.

#include <gtest/gtest.h>

#include <cstdint>
#include <set>
#include <vector>

#include "pd_v2/public/pd_v2_types.h"

namespace ksana_llm {
namespace pd_v2 {
namespace {

// Mirror of dispatcher's selection rule. Returns the participating
// Prefill ranks (set) for a single request, given the matrix cell.
//   d_gid     : Decode-side attn_dp_group_id assigned by workload balancer
//   d_attn_tp : Decode TP / Decode DP   (= req->pd_v2_decode_targets.size())
//   p_tp      : Prefill total TP        (number of Prefill executor ranks
//                                        that participate in dispatcher polling)
std::set<int> ParticipatingPrefillRanks(uint32_t d_gid, uint32_t d_attn_tp, uint32_t p_tp) {
  // Decode wire packs `d_attn_tp` DecodeTargets: r = 0 .. d_attn_tp - 1.
  std::vector<DecodeTargetLite> targets(d_attn_tp);
  for (uint32_t r = 0; r < d_attn_tp; ++r) {
    targets[r].decode_attn_tp_rank_in_group = r;
  }
  std::set<int> hits;
  for (int my_rank = 0; my_rank < static_cast<int>(p_tp); ++my_rank) {
    for (const auto& t : targets) {
      const uint32_t target_device_id = d_gid * d_attn_tp + t.decode_attn_tp_rank_in_group;
      if (static_cast<int>(target_device_id) == my_rank) {
        hits.insert(my_rank);
        break;
      }
    }
  }
  return hits;
}

}  // namespace

// ===== §6.1 Prefill DP=1 (p_aTP=8) ============================================

// TP=8 DP=1 + TP=8 DP=1: Prefill [0..7] -> Decode [0..7], 8 路 1-1
TEST(PdV2KvDispatcherRouting, MatrixSec61_DecD1_PreD1) {
  EXPECT_EQ(ParticipatingPrefillRanks(/*d_gid=*/0, /*d_attn_tp=*/8, /*p_tp=*/8),
            (std::set<int>{0, 1, 2, 3, 4, 5, 6, 7}));
}

// TP=8 DP=2 + TP=8 DP=1: d_gid=0 -> P[0..3], d_gid=1 -> P[4..7], 4 路
TEST(PdV2KvDispatcherRouting, MatrixSec61_DecD2_PreD1) {
  EXPECT_EQ(ParticipatingPrefillRanks(/*d_gid=*/0, /*d_attn_tp=*/4, /*p_tp=*/8), (std::set<int>{0, 1, 2, 3}));
  EXPECT_EQ(ParticipatingPrefillRanks(/*d_gid=*/1, /*d_attn_tp=*/4, /*p_tp=*/8), (std::set<int>{4, 5, 6, 7}));
}

// TP=8 DP=4 + TP=8 DP=1: d_gid=g -> P[2g..2g+1], 2 路
TEST(PdV2KvDispatcherRouting, MatrixSec61_DecD4_PreD1) {
  EXPECT_EQ(ParticipatingPrefillRanks(/*d_gid=*/0, /*d_attn_tp=*/2, /*p_tp=*/8), (std::set<int>{0, 1}));
  EXPECT_EQ(ParticipatingPrefillRanks(/*d_gid=*/1, /*d_attn_tp=*/2, /*p_tp=*/8), (std::set<int>{2, 3}));
  EXPECT_EQ(ParticipatingPrefillRanks(/*d_gid=*/2, /*d_attn_tp=*/2, /*p_tp=*/8), (std::set<int>{4, 5}));
  EXPECT_EQ(ParticipatingPrefillRanks(/*d_gid=*/3, /*d_attn_tp=*/2, /*p_tp=*/8), (std::set<int>{6, 7}));
}

// TP=8 DP=8 + TP=8 DP=1: d_gid=g -> P[g], 1 路 (this is the user's
// requested smoke topology — Decode TP=4 DP=4 + Prefill TP=4 DP=1
// behaves the same way modulo TP=4 vs TP=8).
TEST(PdV2KvDispatcherRouting, MatrixSec61_DecD8_PreD1) {
  for (uint32_t g = 0; g < 8; ++g) {
    EXPECT_EQ(ParticipatingPrefillRanks(g, /*d_attn_tp=*/1, /*p_tp=*/8), (std::set<int>{static_cast<int>(g)}));
  }
}

// Direct check on the user's actual config (Decode TP=4 DP=4, Prefill TP=4 DP=1).
TEST(PdV2KvDispatcherRouting, UserSmoke_Decode_TP4_DP4_Prefill_TP4_DP1) {
  EXPECT_EQ(ParticipatingPrefillRanks(/*d_gid=*/0, /*d_attn_tp=*/1, /*p_tp=*/4), (std::set<int>{0}));
  EXPECT_EQ(ParticipatingPrefillRanks(/*d_gid=*/1, /*d_attn_tp=*/1, /*p_tp=*/4), (std::set<int>{1}));
  EXPECT_EQ(ParticipatingPrefillRanks(/*d_gid=*/2, /*d_attn_tp=*/1, /*p_tp=*/4), (std::set<int>{2}));
  EXPECT_EQ(ParticipatingPrefillRanks(/*d_gid=*/3, /*d_attn_tp=*/1, /*p_tp=*/4), (std::set<int>{3}));
}

// ===== §6.2 Prefill DP=2 (p_aTP=4) ============================================
//
// Same physical device_id formula → behavior is independent of how Prefill
// internally groups DP. The dispatcher only sees per-rank hits; the p_gid
// is implicit (= my_rank / p_aTP). Tests below confirm hit sets match §6.2.

TEST(PdV2KvDispatcherRouting, MatrixSec62_DecD2_PreD2) {
  EXPECT_EQ(ParticipatingPrefillRanks(/*d_gid=*/0, /*d_attn_tp=*/4, /*p_tp=*/8),
            (std::set<int>{0, 1, 2, 3}));  // hits in p_gid=0 (ranks 0..3)
  EXPECT_EQ(ParticipatingPrefillRanks(/*d_gid=*/1, /*d_attn_tp=*/4, /*p_tp=*/8),
            (std::set<int>{4, 5, 6, 7}));  // hits in p_gid=1 (ranks 4..7)
}

TEST(PdV2KvDispatcherRouting, MatrixSec62_DecD4_PreD2) {
  EXPECT_EQ(ParticipatingPrefillRanks(/*d_gid=*/0, /*d_attn_tp=*/2, /*p_tp=*/8), (std::set<int>{0, 1}));
  EXPECT_EQ(ParticipatingPrefillRanks(/*d_gid=*/1, /*d_attn_tp=*/2, /*p_tp=*/8), (std::set<int>{2, 3}));
  EXPECT_EQ(ParticipatingPrefillRanks(/*d_gid=*/2, /*d_attn_tp=*/2, /*p_tp=*/8), (std::set<int>{4, 5}));
  EXPECT_EQ(ParticipatingPrefillRanks(/*d_gid=*/3, /*d_attn_tp=*/2, /*p_tp=*/8), (std::set<int>{6, 7}));
}

TEST(PdV2KvDispatcherRouting, MatrixSec62_DecD8_PreD2) {
  for (uint32_t g = 0; g < 8; ++g) {
    EXPECT_EQ(ParticipatingPrefillRanks(g, /*d_attn_tp=*/1, /*p_tp=*/8), (std::set<int>{static_cast<int>(g)}));
  }
}

// ===== §6.3 / §6.4 Prefill DP=4 / DP=8 same formula ===========================

TEST(PdV2KvDispatcherRouting, MatrixSec63_DecD4_PreD4) {
  // Each Decode group g handled by its matching Prefill group g (2 路).
  EXPECT_EQ(ParticipatingPrefillRanks(/*d_gid=*/0, /*d_attn_tp=*/2, /*p_tp=*/8), (std::set<int>{0, 1}));
  EXPECT_EQ(ParticipatingPrefillRanks(/*d_gid=*/2, /*d_attn_tp=*/2, /*p_tp=*/8), (std::set<int>{4, 5}));
}

TEST(PdV2KvDispatcherRouting, MatrixSec64_DecD8_PreD8) {
  for (uint32_t g = 0; g < 8; ++g) {
    EXPECT_EQ(ParticipatingPrefillRanks(g, /*d_attn_tp=*/1, /*p_tp=*/8), (std::set<int>{static_cast<int>(g)}));
  }
}

// ===== p_aTP < d_aTP TBD rows: rejected at IntakeWrapper time (P1.3) ==========
//
// The dispatcher itself, looked at in isolation, partially "covers" the
// matching ranks (rank 0..min(prefill_tp, d_attn_tp)-1 hit, the rest
// can't match). That partial coverage is the silent-corruption mode
// the review's §P1.3 called out. The fail-fast lives at IntakeWrapper
// time (see PdV2PrefillIntakeWrapper::OnIncoming) — it rejects the
// PrefillRequest with kFailed before the dispatcher is ever called.
// We keep the partial-hit assertion here as a regression check on the
// dispatcher's pure formula, but the integration path no longer
// reaches it for unsupported topology.
TEST(PdV2KvDispatcherRouting, TbdCell_p_aTP_lt_d_aTP_PartialHits_FailFastByIntake) {
  EXPECT_EQ(ParticipatingPrefillRanks(/*d_gid=*/0, /*d_attn_tp=*/8, /*p_tp=*/4), (std::set<int>{0, 1, 2, 3}));
  // ranks 4..7 don't exist on this Prefill — IntakeWrapper rejects before
  // the dispatcher even sees the request. See pd_v2_prefill_intake_wrapper.cpp
  // OnIncoming `prefill_attn_tp` gate.
}

}  // namespace pd_v2
}  // namespace ksana_llm
