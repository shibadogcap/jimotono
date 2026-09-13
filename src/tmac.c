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
// - P2: LUT参照(gather的8 lookup)はスカラー維持、集計FMAのみベクトル化。
//   shuffle/gather回避。int8 8要素のshuffle化はlane複製・mirror符号復元
//   分岐が増え、8要素ではペイしないため。16B=NEON 128bit一致は将来の
//   lookup完全ベクトル化 (vqtbl1q_u8 / _mm256_shuffle_epi8) の予約位置。
// - 集計 (sc*acc+bi)*0.5*2^b のブロック内bits方向 (最大4) をベクトル化。
//   __m256d (4xdouble) / float64x2_t (2xdouble) で1ブロック分を一括計算し、
//   レーン合算は b=0..bits-1 のスカラー順序で加算してbit同一性を保つ。
//   FMA (fmadd/vfmaq) は使わない: 1丸め化でスカラー (mul→add 2丸め) と
//   bit同一にならないため。mul/add分離でIEEE丸め順序を保存する。
// - AVX-512見送り理由: ZMM使用で最大30%ダウンクロック (DESIGN.MD §2.3、
//   ARCHITECTURE.MD §1)。1C1T帯域律速設計では周波数低下がそのままtok/sに
//   直結する。bits<=4は256bit (4xdouble) に丁度収まり512bitの追加スループット
//   はない。LUT参照がgather律速で演算律速でない以上ZMMの利益はなく、
//   N100/i5-8th baselineの移植性 (CMakeLists: SIMDはTU毎opt-in、global強制不可)
//   を優先し、AVX-512パスは設けない。将来演算律速部が出れば再検討。
// - アライメント: 全経路アライメント非依存 (P1互換)。SIMD側は非整列
//   load/store (_mm256_storeu、vst1q)のみ、gather部はスカラーのため
//   64B整列は推奨のまま必須化しない。fail-closed・量子化境界は不変。
#if defined(__AVX512__)
/* P2: AVX-512パスなし (上記ダウンクロック懸念のため意図的に見送り)。 */
#endif
#ifdef __AVX2__
/* P2: 下部 jt_tmac_block_sum_avx2 で集計FMAをベクトル化。LUT参照はスカラー。 */
#endif
#ifdef __ARM_NEON
/* P2: 下部 jt_tmac_block_sum_neon で集計FMAをベクトル化。LUT参照はスカラー。 */
#endif
/* P1: ポータブルなスカラー核 (アライメント非依存、K-first走査は呼び出し側)。フォールバックは常時スカラー。 */

#include "jimotono/tmac.h"

#include <errno.h>
#include <float.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>

#include "jimotono/common.h"

#ifdef __AVX2__
#include <immintrin.h>
#endif
#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

// 1プレーン分のLUT参照+int32累積 (全経路共通スカラー。8加算で±1024以内)。
// mirror復元: p>=8は格納値そのまま、p<8は F(p)=-F(15-p) より格納index (7-p) の符号反転。
// -128の符号反転はint域で行いint32 accに加算するため厳密 (P1互換)。
static int32_t jt_tmac_plane_acc(const int8_t *restrict qblk, const uint8_t *restrict plane) {
    int32_t acc = 0;
    for (size_t gi = 0; gi < 8u; gi++) {
        uint8_t p = (uint8_t)(plane[gi] & 0x0Fu);
        int qv = 0;
        if (p >= 8u) {
            qv = (int)qblk[gi * 8u + (size_t)(p - 8u)];
        } else {
            qv = -(int)qblk[gi * 8u + (size_t)(7u - p)];
        }
        acc += qv;
    }
    return acc;
}

#ifdef __AVX2__
// 1ブロック分の集計 (sc*acc+bi)*0.5*2^b を__m256d 1本で処理。
// 要素毎の演算順序はスカラーと同一 (mul→add→mul(0.5)→mul(pow2))、
// レーン合算のみb昇順スカラー加算 → スカラーとbit同一。FMA不使用。
static double jt_tmac_block_sum_avx2(double sc, double bi, const int32_t *restrict acc,
                                     int bits) {
    __m256d vacc = _mm256_setr_pd((double)acc[0], bits > 1 ? (double)acc[1] : 0.0,
                                  bits > 2 ? (double)acc[2] : 0.0,
                                  bits > 3 ? (double)acc[3] : 0.0);
    __m256d vsc = _mm256_set1_pd(sc);
    __m256d vbi = _mm256_set1_pd(bi);
    __m256d v = _mm256_add_pd(_mm256_mul_pd(vsc, vacc), vbi);
    __m256d vhalf = _mm256_set1_pd(0.5);
    __m256d vpow = _mm256_setr_pd(1.0, 2.0, 4.0, 8.0);
    double lane[4];
    double bsum = 0.0;
    v = _mm256_mul_pd(v, vhalf);
    v = _mm256_mul_pd(v, vpow);
    _mm256_storeu_pd(lane, v);
    for (int b = 0; b < bits; b++) {
        bsum += lane[b];
    }
    return bsum;
}
#endif

#ifdef __ARM_NEON
// NEON版 (float64x2_tで2要素ずつ、要素内順序はスカラー同一、合算はb昇順)。
// 16B=128bit整列は要求しない (vst1qのみ・gather部スカラーのため非整列可)。
// FMA (vfmaq_f64) 不使用、mul/add分離。
static double jt_tmac_block_sum_neon(double sc, double bi, const int32_t *restrict acc,
                                     int bits) {
    double bsum = 0.0;
    float64x2_t vsc = vdupq_n_f64(sc);
    float64x2_t vbi = vdupq_n_f64(bi);
    float64x2_t vhalf = vdupq_n_f64(0.5);
    for (int b = 0; b < bits; b += 2) {
        int n = (bits - b) >= 2 ? 2 : 1;
        float64x2_t vacc = vdupq_n_f64(0.0);
        float64x2_t vpow = vdupq_n_f64(0.0);
        float64x2_t v = vdupq_n_f64(0.0);
        double lane[2];
        vacc = vsetq_lane_f64((double)acc[b], vacc, 0);
        vpow = vsetq_lane_f64((double)(1 << b), vpow, 0);
        if (n == 2) {
            vacc = vsetq_lane_f64((double)acc[b + 1], vacc, 1);
            vpow = vsetq_lane_f64((double)(1 << (b + 1)), vpow, 1);
        }
        v = vaddq_f64(vmulq_f64(vsc, vacc), vbi);
        v = vmulq_f64(v, vhalf);
        v = vmulq_f64(v, vpow);
        vst1q_f64(lane, v);
        for (int k = 0; k < n; k++) {
            bsum += lane[k];
        }
    }
    return bsum;
}
#endif

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
    if (bits < 1 || bits > 4) {
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
    // P2: LUT参照はスカラー維持 (jt_tmac_plane_acc)、集計FMAのみベクトル化。
    for (size_t bb = 0; bb < nblocks; bb++) {
        double sc = (double)scales[bb];
        double bi = (double)biases[bb];
        const int8_t *restrict qblk = qlut + bb * 8u * (size_t)JT_TMAC_LUT_HALF;
        int32_t accs[4] = {0, 0, 0, 0};
        for (int b = 0; b < bits; b++) {
            const uint8_t *restrict plane = idx + (size_t)b * ngroups + bb * 8u;
            accs[b] = jt_tmac_plane_acc(qblk, plane);
        }
#ifdef __AVX2__
        {
            double bsum = jt_tmac_block_sum_avx2(sc, bi, accs, bits);
            total += bsum;
            if (!isfinite(total)) {
                errno = ERANGE;
                rc = JT_ERR_INVAL;
                goto cleanup;
            }
        }
#else
#ifdef __ARM_NEON
        {
            double bsum = jt_tmac_block_sum_neon(sc, bi, accs, bits);
            total += bsum;
            if (!isfinite(total)) {
                errno = ERANGE;
                rc = JT_ERR_INVAL;
                goto cleanup;
            }
        }
#else
        for (int b = 0; b < bits; b++) {
            double contrib = (sc * (double)accs[b] + bi) * 0.5;
            total += contrib * (double)(1 << b);
            if (!isfinite(total)) {
                errno = ERANGE;
                rc = JT_ERR_INVAL;
                goto cleanup;
            }
        }
#endif
#endif
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
