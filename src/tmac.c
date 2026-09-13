// jt_tmac: T-MAC LUT-based mpGEMM, P1 scalar core.
//
// papers.md §1準拠。CPU動的LUT: activationからオンラインでQLUT+scales/biasesを
// 生成し、重みbit-planeのg-bitをインデックスとして参照する。逆量子化なし
// (int8参照→int32累積→最後にscale/bias乗算1回/ブロック/プレーン)。
//
// 固定仮定 (tmac.h参照):
// - g=4固定、mirror consolidation (16→8、符号反転で復元、ロスレス)。
// - act_group=32 (連続32 activation = 8グループで1組のscale/bias共有)。
// - weight scaleは単一w_scaleに縮退 (将来per-block化)。
//
// 量子化設計 (P1決め打ち、tmac.hと対):
// - グループ内線形結合 F[g][p] = Σ_j s_j(p)*a_j、s=+1 (bit=1) / -1 (bit=0)。
//   LSB-first: bit j ↔ a_j。F(15-p) = -F(p) (15=0b1111より厳密成立)。
// - ブロック (32 acts = 8 groups) 毎に max_abs = max|F|、scale = max_abs/127、
//   bias = Σ 32 acts (bit-serial線形変換のバイアス一括補正用。papers.md §1核心5)。
//   対称量子化のためmirrorの符号反転はint8レベルで厳密 (端-128はint32で復元)。
// - q = clamp(round(F/scale), -128, 127)。max_abs==0 (全ゼロ) はscale=1/bias=0/q=0。
// - 参照: plane寄与 = (scale*acc + bias)/2 (0/1→±1の逆変換)、
//   out = w_scale * Σ_b 2^b Σ_bb (scale[bb]*acc_bb_b + bias[bb])/2。
//   accはint32 (ブロック内8加算、範囲±1024でオーバーフロー不能)。
//
// fast aggregationは実装しない (NMSE 2.5倍劣化のためデフォルトOFF。papers.md §1)。
//
// 規約: C11、restrict積極使用、errno+goto cleanup、リトルエンディアン前提
// (common.hでビッグエンディアンをコンパイルエラー)、64B整列推奨
// (jt_arena_alloc推奨。スカラーP1核は非整列でも動作、将来SIMDで必須)。
//
// 将来SIMD拡張点:
// - LUT参照は本質的にshuffle (g=4/int8で16B=NEON 128bitに丁度)。
// - 下記#ifdefは将来の置換点を明示するもので、P1は全経路スカラー核に落とす。
#if defined(__AVX512__)
/* TODO(P2): AVX-512 VBMI _mm512_permutexvar_epi8 で64B一括lookup。 */
#elif defined(__AVX2__)
/* TODO(P2): AVX2 _mm256_shuffle_epi8 (lane複製要) で32B一括lookup。 */
#elif defined(__ARM_NEON)
/* TODO(P2): NEON vqtbl1q_u8 で16B常駐lookup。 */
#endif
/* P1: ポータブルなスカラー核 (アライメント非依存、K-first走査は呼び出し側)。 */

#include "jimotono/tmac.h"

#include <errno.h>
#include <float.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>

#include "jimotono/common.h"

// size_t積のオーバーフロー検査。成功時*outに積、失敗時1。
static int jt_size_mul_ov(size_t a, size_t b, size_t *restrict out) {
    if (a == 0 || b == 0) {
        *out = 0;
        return 0;
    }
    if (a > SIZE_MAX / b) {
        return 1;
    }
    *out = a * b;
    return 0;
}

int jt_tmac_qlut_bytes(size_t n, size_t k, size_t *restrict out) {
    if (out == NULL) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    *out = 0;
    if (n == 0 || k == 0) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    if ((k % (size_t)JT_TMAC_G) != 0) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    {
        size_t ngroups = k / (size_t)JT_TMAC_G;
        size_t t = 0;
        size_t bytes = 0;
        if (jt_size_mul_ov(n, ngroups, &t)) {
            errno = ENOMEM;
            return JT_ERR_NOMEM;
        }
        if (jt_size_mul_ov(t, (size_t)JT_TMAC_LUT_HALF, &bytes)) {
            errno = ENOMEM;
            return JT_ERR_NOMEM;
        }
        *out = bytes;
        return JT_OK;
    }
}

int jt_tmac_param_bytes(size_t n, size_t k, size_t *restrict out) {
    if (out == NULL) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    *out = 0;
    if (n == 0 || k == 0) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    if ((k % (size_t)JT_TMAC_ACT_GROUP) != 0) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    {
        size_t nblocks = k / (size_t)JT_TMAC_ACT_GROUP;
        size_t t = 0;
        size_t bytes = 0;
        if (jt_size_mul_ov(n, nblocks, &t)) {
            errno = ENOMEM;
            return JT_ERR_NOMEM;
        }
        if (jt_size_mul_ov(t, sizeof(float), &bytes)) {
            errno = ENOMEM;
            return JT_ERR_NOMEM;
        }
        *out = bytes;
        return JT_OK;
    }
}

int jt_tmac_lut_ctor(const float *restrict act, size_t n, size_t k,
                     int8_t *restrict qlut, float *restrict scales, float *restrict biases) {
    int rc = JT_OK;
    size_t ngroups = 0;
    size_t nblocks = 0;

    if (act == NULL || qlut == NULL || scales == NULL || biases == NULL) {
        errno = EINVAL;
        rc = JT_ERR_INVAL;
        goto cleanup;
    }
    if (n == 0 || k == 0) {
        errno = EINVAL;
        rc = JT_ERR_INVAL;
        goto cleanup;
    }
    if ((k % (size_t)JT_TMAC_G) != 0 || (k % (size_t)JT_TMAC_ACT_GROUP) != 0) {
        errno = EINVAL;
        rc = JT_ERR_INVAL;
        goto cleanup;
    }
    ngroups = k / (size_t)JT_TMAC_G;
    nblocks = k / (size_t)JT_TMAC_ACT_GROUP;
    {
        // インデックス積のオーバーフロー番兵 (arena.cのjt_align_up方式に準拠)。
        size_t t = 0;
        size_t u = 0;
        if (jt_size_mul_ov(n, k, &t)) {
            errno = ENOMEM;
            rc = JT_ERR_NOMEM;
            goto cleanup;
        }
        if (jt_size_mul_ov(n, ngroups, &t) || jt_size_mul_ov(t, (size_t)JT_TMAC_LUT_HALF, &u)) {
            errno = ENOMEM;
            rc = JT_ERR_NOMEM;
            goto cleanup;
        }
        if (jt_size_mul_ov(n, nblocks, &t)) {
            errno = ENOMEM;
            rc = JT_ERR_NOMEM;
            goto cleanup;
        }
    }

    for (size_t r = 0; r < n; r++) {
        const float *restrict arow = act + r * k;
        int8_t *restrict qrow = qlut + r * ngroups * (size_t)JT_TMAC_LUT_HALF;
        float *restrict srow = scales + r * nblocks;
        float *restrict brow = biases + r * nblocks;
        for (size_t bb = 0; bb < nblocks; bb++) {
            const float *restrict abase = arow + bb * (size_t)JT_TMAC_ACT_GROUP;
            int8_t *restrict qbase = qrow + bb * 8u * (size_t)JT_TMAC_LUT_HALF;
            double fmat[8][16];
            double mn = 0.0;
            double mx = 0.0;
            double maxabs = 0.0;
            double asum = 0.0;
            int first = 1;
            // 入力有限検査 + 全結合値の列挙 (8 groups × 16 patterns)。
            for (size_t gi = 0; gi < 8u; gi++) {
                for (size_t p = 0; p < 16u; p++) {
                    double s = 0.0;
                    for (size_t j = 0; j < 4u; j++) {
                        float av = abase[gi * 4u + j];
                        if (!isfinite((double)av)) {
                            errno = EINVAL;
                            rc = JT_ERR_INVAL;
                            goto cleanup;
                        }
                        s += (((p >> j) & 1u) != 0u ? 1.0 : -1.0) * (double)av;
                    }
                    fmat[gi][p] = s;
                    if (first) {
                        mn = s;
                        mx = s;
                        first = 0;
                    } else {
                        if (s < mn) {
                            mn = s;
                        }
                        if (s > mx) {
                            mx = s;
                        }
                    }
                    {
                        double a = s >= 0.0 ? s : -s;
                        if (a > maxabs) {
                            maxabs = a;
                        }
                    }
                }
            }
            for (size_t i = 0; i < (size_t)JT_TMAC_ACT_GROUP; i++) {
                asum += (double)abase[i];
            }
            if (!isfinite(mn) || !isfinite(mx) || !isfinite(maxabs) || !isfinite(asum)) {
                errno = EINVAL;
                rc = JT_ERR_INVAL;
                goto cleanup;
            }
            if (maxabs == 0.0) {
                // 全ゼロ: 除算回避。biasはasum (==0) をそのまま格納。
                float bias_f = (float)asum;
                if (!isfinite((double)bias_f)) {
                    errno = ERANGE;
                    rc = JT_ERR_INVAL;
                    goto cleanup;
                }
                for (size_t gi = 0; gi < 8u; gi++) {
                    for (size_t ii = 0; ii < 8u; ii++) {
                        qbase[gi * 8u + ii] = 0;
                    }
                }
                srow[bb] = 1.0f;
                brow[bb] = bias_f;
            } else {
                double scale_d = maxabs / 127.0;
                float scale_f = 0.0f;
                float bias_f = 0.0f;
                if (!isfinite(scale_d) || scale_d <= 0.0) {
                    errno = ERANGE;
                    rc = JT_ERR_INVAL;
                    goto cleanup;
                }
                scale_f = (float)scale_d;
                bias_f = (float)asum;
                if (!isfinite((double)scale_f) || !isfinite((double)bias_f)) {
                    // floatに収まらない極端なレンジ (オーバーフロー)。
                    // アンダーフローによる0は許容する (誤差は無視可能域)。
                    errno = ERANGE;
                    rc = JT_ERR_INVAL;
                    goto cleanup;
                }
                for (size_t gi = 0; gi < 8u; gi++) {
                    for (size_t ii = 0; ii < 8u; ii++) {
                        double v = fmat[gi][8u + ii];
                        double qd = round(v / scale_d);
                        int qi = 0;
                        if (qd > 127.0) {
                            qd = 127.0;
                        } else if (qd < -128.0) {
                            qd = -128.0;
                        }
                        qi = (int)qd;
                        qbase[gi * 8u + ii] = (int8_t)qi;
                    }
                }
                srow[bb] = scale_f;
                brow[bb] = bias_f;
            }
        }
    }

cleanup:
    return rc;
}

int jt_tmac_lookup_accum(const int8_t *restrict qlut, const uint8_t *restrict idx,
                         const float *restrict scales, const float *restrict biases,
                         size_t ngroups, size_t nblocks, int bits, float w_scale,
                         float *restrict out) {
    int rc = JT_OK;
    double total = 0.0;

    if (qlut == NULL || idx == NULL || scales == NULL || biases == NULL || out == NULL) {
        errno = EINVAL;
        rc = JT_ERR_INVAL;
        goto cleanup;
    }
    if (ngroups == 0 || nblocks == 0) {
        errno = EINVAL;
        rc = JT_ERR_INVAL;
        goto cleanup;
    }
    if (nblocks > SIZE_MAX / 8u) {
        errno = ENOMEM;
        rc = JT_ERR_NOMEM;
        goto cleanup;
    }
    if (ngroups != nblocks * 8u) {
        // act_group=32仮定 (ngroups == nblocks*8) を要求。
        errno = EINVAL;
        rc = JT_ERR_INVAL;
        goto cleanup;
    }
    if (bits != 1 && bits != 2 && bits != 4) {
        errno = EINVAL;
        rc = JT_ERR_INVAL;
        goto cleanup;
    }
    if (ngroups > SIZE_MAX / (size_t)JT_TMAC_LUT_HALF) {
        errno = ENOMEM;
        rc = JT_ERR_NOMEM;
        goto cleanup;
    }
    if (ngroups > SIZE_MAX / (size_t)bits) {
        errno = ENOMEM;
        rc = JT_ERR_NOMEM;
        goto cleanup;
    }
    if (!isfinite((double)w_scale)) {
        errno = EINVAL;
        rc = JT_ERR_INVAL;
        goto cleanup;
    }
    for (size_t bb = 0; bb < nblocks; bb++) {
        if (!isfinite((double)scales[bb]) || !isfinite((double)biases[bb])) {
            errno = EINVAL;
            rc = JT_ERR_INVAL;
            goto cleanup;
        }
    }

    // int32累積→ブロック毎にscale/bias乗算 (逆量子化回避)。
    // 将来SIMD: 内側giループがshuffle+加算に置換される位置。
    for (size_t bb = 0; bb < nblocks; bb++) {
        double sc = (double)scales[bb];
        double bi = (double)biases[bb];
        const int8_t *restrict qblk = qlut + bb * 8u * (size_t)JT_TMAC_LUT_HALF;
        for (int b = 0; b < bits; b++) {
            const uint8_t *restrict plane = idx + (size_t)b * ngroups + bb * 8u;
            int32_t acc = 0;
            for (size_t gi = 0; gi < 8u; gi++) {
                uint8_t p = (uint8_t)(plane[gi] & 0x0Fu);
                int qv = 0;
                if (p >= 8u) {
                    qv = (int)qblk[gi * 8u + (size_t)(p - 8u)];
                } else {
                    // mirror: F(p) = -F(15-p)。格納indexは (15-p)-8 = 7-p。
                    qv = -(int)qblk[gi * 8u + (size_t)(7u - p)];
                }
                acc += (int32_t)qv;  // ±1024以内で安全
            }
            {
                double contrib = (sc * (double)acc + bi) * 0.5;
                total += contrib * (double)(1 << b);
                if (!isfinite(total)) {
                    errno = ERANGE;
                    rc = JT_ERR_INVAL;
                    goto cleanup;
                }
            }
        }
    }
    total *= (double)w_scale;
    if (!isfinite(total)) {
        errno = ERANGE;
        rc = JT_ERR_INVAL;
        goto cleanup;
    }
    {
        float f = (float)total;
        if (!isfinite((double)f)) {
            errno = ERANGE;
            rc = JT_ERR_INVAL;
            goto cleanup;
        }
        *out = f;
    }

cleanup:
    return rc;
}
