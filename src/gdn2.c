// jt_gdn2: Gated DeltaNet-2 デコード逐次核 (Phase 1, C11 スカラー)。
// AGENTS.MD 7.1: restrict積極使用、errnoベース + goto cleanup、
// クロスプラットフォーム (Linux/macOS/Windows)、llama.cpp等への依存なし。
//
// 更新式 (papers.md §2):
//   S_t = (I - k_t (b_t⊙k_t)^T) D_t S_{t-1} + k_t (w_t⊙v_t)^T
//   o_t = S_t^T q_t
//
// 精度方針:
// - S・log-decay累積・q/k正規化は fp32。S の量子化はしない。
// - b_t (erase, key側): チャネル性の寄与が大きい (Ablation) ため fp32 維持。
//   INT2/INT4 化の削減は w_t (write, value側) から行うこと:
//   候補順 (1) w のスカラー化 (全チャネル共通βw)、(2) w の量子化、
//   (3) b の量子化は最後の手段。P1 では (1)〜(3) いずれも行わない。
// - alpha は核外で fp32 累積・exp 済みの値を受ける (核内では exp しない)。
//
// 性能 NOTE (P1 スカラー核 + P2 への道):
// - デコードはトークン逐次で i 方向 (dk) に依存連鎖があるため、素朴な
//   1ストリーム FMA ではレイテンシ (~4cyc) を隠せない。
// - 対策として dv 方向を4ストリームに分割し、独立な4本の FMA チェーンで
//   実行する (下記各ループの j/j+1/j+2/j+3 分割がそれ)。
// - P2 SIMD 対応表:
//     #if defined(__AVX512F__) → _mm512_fmadd_ps (16-wide) で各ストリーム拡張
//     #elif defined(__AVX2__)  → _mm256_fmadd_ps (8-wide)
//     #elif defined(__ARM_NEON) → vfmaq_f32 (4-wide)
//   head 方向の並列は呼び出し側 (1C1T では層パイプライン側) で行う。

#include "jimotono/gdn2.h"

#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#include <malloc.h>
#endif

// ENOSYS が無い処理系 (MSVC) 用の代替定義。スタブ返却の識別専用。
#ifndef ENOSYS
#define ENOSYS 38
#endif

static int jt_gdn2_valid_dims(int dk, int dv) {
    return dk > 0 && dv > 0 && dk <= JT_GDN2_MAX_D && dv <= JT_GDN2_MAX_D;
}

// MAJOR-2: fail-closed 有限検査用。全要素が有限なら1、1つでも NaN/Inf なら0。
static int jt_gdn2_all_finite(const float *restrict x, int n) {
    for (int i = 0; i < n; i++) {
        if (!isfinite(x[i])) {
            return 0;
        }
    }
    return 1;
}

int jt_gdn2_state_bytes(int dk, int dv, size_t *restrict out_bytes) {
    if (out_bytes == NULL) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    *out_bytes = 0;
    if (!jt_gdn2_valid_dims(dk, dv)) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    *out_bytes =
        (size_t)(unsigned int)dk * (size_t)(unsigned int)dv * sizeof(float);
    return JT_OK;
}

int jt_gdn2_state_alloc(float **restrict out_S, int dk, int dv) {
    if (out_S == NULL) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    *out_S = NULL;
    size_t bytes = 0;
    int rc = jt_gdn2_state_bytes(dk, dv, &bytes);
    if (rc != JT_OK) {
        return rc;  // errno は下位で設定済み
    }
    // aligned_alloc は size % alignment == 0 を要求するため切り上げ。
    size_t alloc =
        (bytes + (size_t)(JT_CACHELINE - 1)) & ~(size_t)(JT_CACHELINE - 1);
    void *p = NULL;
#if defined(_WIN32)
    p = _aligned_malloc(alloc, JT_CACHELINE);
    if (p == NULL) {
        errno = ENOMEM;
        return JT_ERR_NOMEM;
    }
#else
    p = aligned_alloc((size_t)JT_CACHELINE, alloc);
    if (p == NULL) {
        errno = ENOMEM;
        return JT_ERR_NOMEM;
    }
#endif
    memset(p, 0, alloc);
    *out_S = (float *)p;
    return JT_OK;
}

void jt_gdn2_state_free(float *S) {
    if (S == NULL) {
        return;
    }
#if defined(_WIN32)
    _aligned_free(S);
#else
    free(S);
#endif
}

int jt_gdn2_state_reset(float *restrict S, int dk, int dv) {
    if (S == NULL) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    size_t bytes = 0;
    int rc = jt_gdn2_state_bytes(dk, dv, &bytes);
    if (rc != JT_OK) {
        return rc;
    }
    // cuSeqlens 境界での state リセット (papers.md §2)。
    memset(S, 0, bytes);
    return JT_OK;
}

int jt_gdn2_scratch_floats(int dk, int dv, size_t *restrict out_n) {
    if (out_n == NULL) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    *out_n = 0;
    if (!jt_gdn2_valid_dims(dk, dv)) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    // 内訳: qn[dk] + kn[dk] + c[dv] + vw[dv]
    *out_n = (size_t)2 * (size_t)(unsigned int)dk +
             (size_t)2 * (size_t)(unsigned int)dv;
    return JT_OK;
}

int jt_gdn2_l2norm(const float *restrict x, float *restrict y, int n,
                   float eps) {
    if (x == NULL || y == NULL) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    if (n <= 0 || n > JT_GDN2_MAX_D) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    if (!(eps > 0.0f)) {  // 0・負・NaN を拒否
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    double acc = 0.0;
    for (int i = 0; i < n; i++) {
        double v = (double)x[i];
        acc += v * v;
    }
    double norm = sqrt(acc);
    if (norm < (double)eps) {
        for (int i = 0; i < n; i++) {
            y[i] = 0.0f;  // ゼロ割ガード: NaN を出さない
        }
        return JT_OK;
    }
    float inv = (float)(1.0 / norm);
    for (int i = 0; i < n; i++) {
        y[i] = x[i] * inv;
    }
    return JT_OK;
}

int jt_gdn2_decode_step(float *restrict S, float *restrict out_o,
                        const float *restrict q, const float *restrict k,
                        const float *restrict v, const float *restrict b,
                        const float *restrict w, const float *restrict alpha,
                        int dk, int dv, float *restrict scratch,
                        size_t scratch_n) {
    int rc = JT_ERR_INVAL;
    size_t need = 0;

    if (S == NULL || out_o == NULL || q == NULL || k == NULL || v == NULL ||
        b == NULL || w == NULL || alpha == NULL || scratch == NULL) {
        errno = EINVAL;
        goto cleanup;
    }
    if (!jt_gdn2_valid_dims(dk, dv)) {
        errno = EINVAL;
        goto cleanup;
    }
    if (((uintptr_t)(const void *)S % (uintptr_t)JT_CACHELINE) != 0) {
        errno = EINVAL;
        rc = JT_ERR_ALIGN;
        goto cleanup;
    }
    if (jt_gdn2_scratch_floats(dk, dv, &need) != JT_OK) {
        goto cleanup;  // errno は下位で設定済み
    }
    if (scratch_n < need) {
        errno = EINVAL;
        goto cleanup;
    }
    // MAJOR-2: 非有限入力は state 更新前に拒否 (fail-closed, tmac/routing整合)。
    // ホットパス検査コストは許容 (P1 正しさ優先)。
    if (!jt_gdn2_all_finite(q, dk) || !jt_gdn2_all_finite(k, dk) ||
        !jt_gdn2_all_finite(b, dk) || !jt_gdn2_all_finite(alpha, dk) ||
        !jt_gdn2_all_finite(v, dv) || !jt_gdn2_all_finite(w, dv)) {
        errno = EINVAL;
        goto cleanup;
    }
    for (int i = 0; i < dk; i++) {
        float ai = alpha[i];
        if (!(ai >= 0.0f && ai <= 1.0f)) {  // NaN もここで拒否される
            errno = EINVAL;
            goto cleanup;
        }
    }

    {
        float *qn = scratch;                  // [dk] 正規化 q
        float *kn = scratch + (size_t)dk;     // [dk] 正規化 k
        float *c = scratch + (size_t)2 * (size_t)dk;  // [dv] (b⊙k)^T S'
        float *vw = c + (size_t)dv;           // [dv] w⊙v
        const float eps = JT_GDN2_EPS_DEFAULT;

        // 0) q/k L2 正規化 (papers.md §2: 必須。呼び出し側は不要)。
        if (jt_gdn2_l2norm(q, qn, dk, eps) != JT_OK) {
            goto cleanup;
        }
        if (jt_gdn2_l2norm(k, kn, dk, eps) != JT_OK) {
            goto cleanup;
        }

        // 1) 大域忘却: S' = D S (alpha は fp32 累積済みを核外から受領)。
        for (int i = 0; i < dk; i++) {
            float ai = alpha[i];
            float *row = S + (size_t)i * (size_t)dv;
            int j = 0;
            // dv 4分割: 独立4チェーンで FMA レイテンシ隠蔽 (性能NOTE参照)。
            for (; j + 3 < dv; j += 4) {
                row[j] *= ai;
                row[j + 1] *= ai;
                row[j + 2] *= ai;
                row[j + 3] *= ai;
            }
            for (; j < dv; j++) {
                row[j] *= ai;
            }
        }

        // 2) 選択的消去の係数: c^T = (b⊙k)^T S'。
        for (int j = 0; j < dv; j++) {
            c[j] = 0.0f;
        }
        for (int i = 0; i < dk; i++) {
            float ke = b[i] * kn[i];  // b_t は fp32 維持 (精度優先)。
            const float *row = S + (size_t)i * (size_t)dv;
            int j = 0;
            for (; j + 3 < dv; j += 4) {
                c[j] += ke * row[j];
                c[j + 1] += ke * row[j + 1];
                c[j + 2] += ke * row[j + 2];
                c[j + 3] += ke * row[j + 3];
            }
            for (; j < dv; j++) {
                c[j] += ke * row[j];
            }
        }

        // 3) 消去: S'' = S' - k c^T。
        for (int i = 0; i < dk; i++) {
            float ki = kn[i];
            float *row = S + (size_t)i * (size_t)dv;
            int j = 0;
            for (; j + 3 < dv; j += 4) {
                row[j] -= ki * c[j];
                row[j + 1] -= ki * c[j + 1];
                row[j + 2] -= ki * c[j + 2];
                row[j + 3] -= ki * c[j + 3];
            }
            for (; j < dv; j++) {
                row[j] -= ki * c[j];
            }
        }

        // 4) 選択的書込みの値: vw = w⊙v。
        // 将来の削減はここから: (1) w スカラー化 (2) w 量子化 (b は維持)。
        for (int j = 0; j < dv; j++) {
            vw[j] = w[j] * v[j];
        }

        // 5) 書込み: S = S'' + k vw^T。
        for (int i = 0; i < dk; i++) {
            float ki = kn[i];
            float *row = S + (size_t)i * (size_t)dv;
            int j = 0;
            for (; j + 3 < dv; j += 4) {
                row[j] += ki * vw[j];
                row[j + 1] += ki * vw[j + 1];
                row[j + 2] += ki * vw[j + 2];
                row[j + 3] += ki * vw[j + 3];
            }
            for (; j < dv; j++) {
                row[j] += ki * vw[j];
            }
        }

        // 6) 読出し: o = S^T q (更新後 S を使用)。
        for (int j = 0; j < dv; j++) {
            out_o[j] = 0.0f;
        }
        for (int i = 0; i < dk; i++) {
            float qi = qn[i];
            const float *row = S + (size_t)i * (size_t)dv;
            int j = 0;
            for (; j + 3 < dv; j += 4) {
                out_o[j] += qi * row[j];
                out_o[j + 1] += qi * row[j + 1];
                out_o[j + 2] += qi * row[j + 2];
                out_o[j + 3] += qi * row[j + 3];
            }
            for (; j < dv; j++) {
                out_o[j] += qi * row[j];
            }
        }
    }

    rc = JT_OK;
cleanup:
    return rc;
}

// チャンク核スタブの共通処理: 入力検査のみ行い ENOSYS を返す。
// MINOR-1: 正常入力は JT_ERR_NOSUP+errno=ENOSYS、不正入力は
// JT_ERR_INVAL+errno=EINVAL (戻り値のみで区別可能)。
static int jt_gdn2_prefill_stub(const float *S, const float *Out,
                                const float *Q, const float *K,
                                const float *V, const float *B,
                                const float *W, const float *Alpha, int dk,
                                int dv, const float *scratch) {
    if (S == NULL || Out == NULL || Q == NULL || K == NULL || V == NULL ||
        B == NULL || W == NULL || Alpha == NULL || scratch == NULL) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    if (!jt_gdn2_valid_dims(dk, dv)) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    // P2 で WY 型チャンク核 (C 固定 solve + inter-chunk 漸化式) を実装。
    errno = ENOSYS;
    return JT_ERR_NOSUP;
}

int jt_gdn2_prefill_chunk16(float *restrict S, float *restrict Out,
                            const float *restrict Q, const float *restrict K,
                            const float *restrict V, const float *restrict B,
                            const float *restrict W,
                            const float *restrict Alpha, int dk, int dv,
                            float *restrict scratch, size_t scratch_n) {
    (void)scratch_n;
    return jt_gdn2_prefill_stub(S, Out, Q, K, V, B, W, Alpha, dk, dv, scratch);
}

int jt_gdn2_prefill_chunk32(float *restrict S, float *restrict Out,
                            const float *restrict Q, const float *restrict K,
                            const float *restrict V, const float *restrict B,
                            const float *restrict W,
                            const float *restrict Alpha, int dk, int dv,
                            float *restrict scratch, size_t scratch_n) {
    (void)scratch_n;
    return jt_gdn2_prefill_stub(S, Out, Q, K, V, B, W, Alpha, dk, dv, scratch);
}
