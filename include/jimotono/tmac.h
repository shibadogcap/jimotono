#pragma once
#ifndef JIMOTONO_TMAC_H
#define JIMOTONO_TMAC_H
// T-MAC LUT-based mpGEMM kernel (papers.md §1準拠, Phase 1).
//
// 方式: CPU動的LUT。activationからオンラインでQLUT+scales/biasesを生成し、
// 重みbit-planeのg-bitをインデックスとして参照する。逆量子化なし
// (int8参照→int32累積→最後にscale乗算1回)。
//
// 固定仮定 (Phase 1決め打ち):
// - g=4固定 (LUT粒度。g>=5はテーブル肥大で遅いため。papers.md §1 C実装2)。
// - mirror consolidation: 素朴16エントリ→8エントリ (符号反転で復元、ロスレス)。
// - act_group=32仮定: 連続32 activation (= g=4グループ×8) で1組のscale/biasを
//   共有する動的量子化粒度。T-MAC原典は g=4で8値量子化のみ明示のため、
//   32はJIMOTONO仮定として明記する (papers.md §1 C実装2)。
// - weight scale粒度 group_size=128はP1では単一w_scaleに縮退
//   (将来per-block化。独自バイナリはbit-plane済みで焼く前提)。
//
// レイアウト (全てrow-major, little-endian前提):
// - act: [N][K] float。
// - qlut: [N][ngroups][8] int8。ngroups=K/4。
//   qlut[g*8+i] はパターン p=8+i (MSB=1, LSB-first: bit j ↔ a_j) の量子化値。
//   p<8の参照時はミラー対 15-p (格納index 7-p) の符号反転で復元。
// - scales/biases: [N][nblocks] float。nblocks=K/32。
//   block bbは groups [bb*8, bb*8+7] (= activation 32個) を共有。
// - idx: [(bits)][ngroups] uint8。下位4bitのみ使用 (0..15)、上位nibble無視。
//   将来のnibble分割2-lookup拡張用に予約。
//
// bit-serial線形変換: 重みbit 0/1を s=-1/+1に写像しfloat乗算を加減算のみに
// する。LUT値域を最小化し量子化誤差を削減する (papers.md §1核心5)。
//
// fast aggregationは実装しない (デフォルトOFF推奨のため)。
// 理由は src/tmac.c の注記を参照 (NMSE 2.5倍劣化)。
//
// 規約: C11, restrict, errno+goto cleanup (実装側), 64B整列推奨
// (jt_arena_alloc推奨。スカラーP1核は非整列でも動作するが将来SIMDで必須)。
// サイズ計算はarena.cのjt_align_up方式 (2の冪検査+オーバーフロー番兵)。

#include <stddef.h>
#include <stdint.h>

#include "jimotono/common.h"

#define JT_TMAC_G 4
#define JT_TMAC_LUT_HALF 8
#define JT_TMAC_LUT_FULL 16
#define JT_TMAC_ACT_GROUP 32
#define JT_TMAC_ALIGN 64

// 必要バイト数 (オーバーフロー時はJT_ERR_NOMEM/EINVAL + errno)。
// qlut: N*(K/4)*8 byte。param: N*(K/32)*sizeof(float) (scales/biases各1配列分)。
// 注意(MINOR-2): qlut_bytes単体はK%4、ctorはK%32を要求。K=4等では qlut_bytes=OK でも ctor=INVAL になる。param_bytesもK%32要求のため実害なし。
int jt_tmac_qlut_bytes(size_t n, size_t k, size_t *restrict out);
int jt_tmac_param_bytes(size_t n, size_t k, size_t *restrict out);

// LUT構築: act[N][K] float → qlut + scales/biases。
// 戻り値: JT_OK / JT_ERR_INVAL (NULL, n==0/k==0, K%4!=0, K%32!=0,
//          NaN/Inf混入) / JT_ERR_NOMEM (内部サイズ積オーバーフロー)。
int jt_tmac_lut_ctor(const float *restrict act, size_t n, size_t k,
                     int8_t *restrict qlut, float *restrict scales, float *restrict biases);

// LUT参照累積: 1行分のQLUT (ngroups*8) と1列分の重みindex (bits*ngroups) から
// ドット積1スカラーを求める。GEMM時は呼び出し側で(n,m)ループする。
// - ngroups>0, nblocks>0, ngroups==nblocks*8 (act_group=32仮定) を要求。
// - bits ∈ {1,2,3,4} (bit-serialパス数)。結果は Σ_b 2^b * plane_b。
// - w_scale: 重みblock scale (P1は単一値に縮退)。
// - int32累積→最後にscale/bias乗算 (ブロック毎に1回ずつ。逆量子化回避)。
// 戻り値: JT_OK / JT_ERR_INVAL。
int jt_tmac_lookup_accum(const int8_t *restrict qlut, const uint8_t *restrict idx,
                         const float *restrict scales, const float *restrict biases,
                         size_t ngroups, size_t nblocks, int bits, float w_scale,
                         float *restrict out);

#endif  // JIMOTONO_TMAC_H
