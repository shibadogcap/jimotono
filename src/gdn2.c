// jt_gdn2: Gated DeltaNet-2 デコード逐次核 (Phase 1) + プリフィル chunk 核
// (P2, WY 型、C=16/32)。C11 スカラー。
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

// ---- P2 プリフィル chunk 核 (WY 型、C 固定、スカラー) ----
//
// 更新式 (papers.md §2、gdn2.h 契約):
//   S_t = (I - k_t ke_t^T) D_t S_{t-1} + k_t vw_t^T,
//   o_t = S_t^T q_t,
//   ke_t = b_t⊙k_t (erase, fp32 維持)、vw_t = w_t⊙v_t (write)、D_t = Diag(alpha_t)。
//   k_t/q_t は核内で L2 正規化 (eps ガード)。alpha は核外累積済み [0,1]。
//
// チャンク分解 (decay 吸収で純粋非対称 delta 化):
//   cum(t,s)[i] = Π_{r=s+1..t} alpha_r[i]、cumfwd(t)[i] = Π_{r=0..t} alpha_r[i]。
//   L_{t,s} = ke_t^T diag(cum(t,s)) k_s (t>s のみ狭義下三角)。
//   B_t = vw_t - (ke_t⊙cumfwd(t))^T S_in ([dv])。
//   前進代入 (I+L) E = B で E ([C][dv]) を解く
//   ((I+L) は対角=1 のため除算なし)。
//   P_{t,s} = q_t^T diag(cum(t,s)) k_s (s≦t、下三角+対角)。
//   Out_t = (q_t⊙cumfwd(t))^T S_in + Σ_{s≦t} P_{t,s} E_s。
//   S_out = Diag(cumfwd(C-1)) S_in + Σ_s W_s E_s^T、
//   W_s[i] = k_s[i] cum(C-1,s)[i]。
// inter-chunk 漸化式はチャンク境界の S_in→S_out のみ。intra-chunk は
// C×C 下三角 solve + 密行列積。C 回 decode_step と fp32 丸めを除き等価。
//
// scratch 内訳 (gdn2.h 契約どおり 4*C*dk + 2*C*dv + C*C + dk):
//   Qn/Kn/E(ke)/Pdec(Wgt) + U(vw)/G(B→E) + LP(L→P時分割再利用) + tmp。
// 性能 NOTE: dv 方向 j 4 分割で独立 FMA チェーン×4 (DESIGN.MD §4.3)。
//   TODO(SIMD別タスク): l2norm の 1/sqrt を rsqrt 近似+Newton-Raphson 1 回にし、
//   j 4 レーンを AVX2(_mm256_fmadd_ps)/AVX-512/NEON(vfmaq_f32) に拡張すること。
//   除算は現状 l2norm 内の素朴な逆数乗算のみ (TODO: 近似化は SIMD 担当が実施)。

// プリフィル chunk 用 scratch の float 要素数 (4*C*dk + 2*C*dv + C*C + dk)。
// C は 16/32 のみ。戻り値: JT_OK / JT_ERR_INVAL (errno 併用)。
static int jt_gdn2_prefill_need(int C, int dk, int dv,
                                size_t *restrict out_n) {
    if (out_n == NULL) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    *out_n = 0;
    if (C != 16 && C != 32) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    if (!jt_gdn2_valid_dims(dk, dv)) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    *out_n = (size_t)4 * (size_t)C * (size_t)(unsigned int)dk +
             (size_t)2 * (size_t)C * (size_t)(unsigned int)dv +
             (size_t)C * (size_t)C + (size_t)(unsigned int)dk;
    return JT_OK;
}

// chunk 本体 (C 引数化の共通ヘルパー。chunk16/32 は C=16/32 で委譲)。
static int jt_gdn2_prefill_chunk_impl(float *restrict S, float *restrict Out,
                                      const float *restrict Q,
                                      const float *restrict K,
                                      const float *restrict V,
                                      const float *restrict B,
                                      const float *restrict W,
                                      const float *restrict Alpha, int dk,
                                      int dv, float *restrict scratch,
                                      size_t scratch_n, int C) {
    int rc = JT_ERR_INVAL;
    size_t need = 0;

    if (S == NULL || Out == NULL || Q == NULL || K == NULL || V == NULL ||
        B == NULL || W == NULL || Alpha == NULL || scratch == NULL) {
        errno = EINVAL;
        goto cleanup;
    }
    if (C != 16 && C != 32) {
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
    if (jt_gdn2_prefill_need(C, dk, dv, &need) != JT_OK) {
        goto cleanup;  // errno は下位で設定済み
    }
    if (scratch_n < need) {
        errno = EINVAL;
        goto cleanup;
    }
    // fail-closed: 非有限・alpha 範囲外は S/Out 更新前に拒否 (decode 核と同一方針)。
    {
        int Cdk = C * dk;
        int Cdv = C * dv;
        if (!jt_gdn2_all_finite(Q, Cdk) || !jt_gdn2_all_finite(K, Cdk) ||
            !jt_gdn2_all_finite(B, Cdk) ||
            !jt_gdn2_all_finite(Alpha, Cdk) ||
            !jt_gdn2_all_finite(V, Cdv) || !jt_gdn2_all_finite(W, Cdv)) {
            errno = EINVAL;
            goto cleanup;
        }
        for (int n = 0; n < Cdk; n++) {
            float a = Alpha[n];
            if (!(a >= 0.0f && a <= 1.0f)) {  // NaN もここで拒否される
                errno = EINVAL;
                goto cleanup;
            }
        }
    }

    {
        const float eps = JT_GDN2_EPS_DEFAULT;
        size_t Cdk = (size_t)C * (size_t)(unsigned int)dk;
        size_t Cdv = (size_t)C * (size_t)(unsigned int)dv;
        size_t C2 = (size_t)C * (size_t)C;
        size_t Uoff = (size_t)4 * Cdk;
        float *Qn = scratch;                // [C][dk] 正規化 Q
        float *Kn = scratch + Cdk;          // [C][dk] 正規化 K
        float *Eke = scratch + (size_t)2 * Cdk;  // [C][dk] ke = b⊙kn
        float *Pdec = scratch + (size_t)3 * Cdk;  // [C][dk] Wgt (後段で使用)
        float *U = scratch + Uoff;          // [C][dv] vw = w⊙v
        float *G = U + Cdv;                 // [C][dv] B→E (前進代入で上書き)
        float *LP = G + Cdv;                // [C][C] L→P (時分割再利用)
        float *tmp = LP + C2;               // [dk] cum/suf 作業域

        // 0) Q/K 行ごと L2 正規化 + ke/vw 生成。
        for (int t = 0; t < C; t++) {
            const float *Qr = Q + (size_t)t * (size_t)dk;
            const float *Kr = K + (size_t)t * (size_t)dk;
            float *Qnr = Qn + (size_t)t * (size_t)dk;
            float *Knr = Kn + (size_t)t * (size_t)dk;
            if (jt_gdn2_l2norm(Qr, Qnr, dk, eps) != JT_OK) {
                errno = EINVAL;
                goto cleanup;
            }
            if (jt_gdn2_l2norm(Kr, Knr, dk, eps) != JT_OK) {
                errno = EINVAL;
                goto cleanup;
            }
        }
        for (int t = 0; t < C; t++) {
            const float *Br = B + (size_t)t * (size_t)dk;
            const float *Knr = Kn + (size_t)t * (size_t)dk;
            float *Eker = Eke + (size_t)t * (size_t)dk;
            for (int i = 0; i < dk; i++) {
                Eker[i] = Br[i] * Knr[i];  // b は fp32 維持 (精度優先)
            }
        }
        for (int t = 0; t < C; t++) {
            const float *Vr = V + (size_t)t * (size_t)dv;
            const float *Wr = W + (size_t)t * (size_t)dv;
            float *Ur = U + (size_t)t * (size_t)dv;
            int j = 0;
            for (; j + 3 < dv; j += 4) {
                Ur[j] = Wr[j] * Vr[j];
                Ur[j + 1] = Wr[j + 1] * Vr[j + 1];
                Ur[j + 2] = Wr[j + 2] * Vr[j + 2];
                Ur[j + 3] = Wr[j + 3] * Vr[j + 3];
            }
            for (; j < dv; j++) {
                Ur[j] = Wr[j] * Vr[j];
            }
        }

        // 1) L (狭義下三角): L[t][s] = ke_t^T diag(cum(t,s)) k_s。
        // cum(t,s) は s 固定・t 昇順に tmp 上で前進累積 (除算なし)。
        for (size_t n = 0; n < C2; n++) {
            LP[n] = 0.0f;
        }
        for (int s = 0; s < C; s++) {
            const float *Ks = Kn + (size_t)s * (size_t)dk;
            for (int i = 0; i < dk; i++) {
                tmp[i] = 1.0f;
            }
            for (int t = s + 1; t < C; t++) {
                const float *At = Alpha + (size_t)t * (size_t)dk;
                const float *KEt = Eke + (size_t)t * (size_t)dk;
                float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
                int i = 0;
                for (i = 0; i < dk; i++) {
                    tmp[i] *= At[i];
                }
                // 4 アキュムレータで FMA レイテンシ隠蔽 (DESIGN.MD §4.3)。
                for (i = 0; i + 3 < dk; i += 4) {
                    s0 += KEt[i] * tmp[i] * Ks[i];
                    s1 += KEt[i + 1] * tmp[i + 1] * Ks[i + 1];
                    s2 += KEt[i + 2] * tmp[i + 2] * Ks[i + 2];
                    s3 += KEt[i + 3] * tmp[i + 3] * Ks[i + 3];
                }
                for (; i < dk; i++) {
                    s0 += KEt[i] * tmp[i] * Ks[i];
                }
                LP[(size_t)t * (size_t)C + (size_t)s] =
                    (s0 + s1) + (s2 + s3);
            }
        }

        // 2) B ([C][dv]): B_t = vw_t - (ke_t⊙cumfwd(t))^T S_in。
        // cumfwd(t) は tmp 上で前進累積。
        for (int i = 0; i < dk; i++) {
            tmp[i] = 1.0f;
        }
        for (int t = 0; t < C; t++) {
            const float *At = Alpha + (size_t)t * (size_t)dk;
            const float *KEt = Eke + (size_t)t * (size_t)dk;
            const float *Ut = U + (size_t)t * (size_t)dv;
            float *Gt = G + (size_t)t * (size_t)dv;
            for (int i = 0; i < dk; i++) {
                tmp[i] *= At[i];
            }
            for (int j = 0; j < dv; j++) {
                Gt[j] = Ut[j];
            }
            for (int i = 0; i < dk; i++) {
                float wi = KEt[i] * tmp[i];
                const float *Sr = S + (size_t)i * (size_t)dv;
                int j = 0;
                for (; j + 3 < dv; j += 4) {
                    Gt[j] -= wi * Sr[j];
                    Gt[j + 1] -= wi * Sr[j + 1];
                    Gt[j + 2] -= wi * Sr[j + 2];
                    Gt[j + 3] -= wi * Sr[j + 3];
                }
                for (; j < dv; j++) {
                    Gt[j] -= wi * Sr[j];
                }
            }
        }

        // 3) 前進代入 (I+L) E = B (対角=1 のため除算なし、G 上で上書き)。
        // 内側の ls==0 判定は分岐ミス回避のため入れない (DESIGN.MD §4.2)。
        for (int t = 0; t < C; t++) {
            float *Gt = G + (size_t)t * (size_t)dv;
            for (int s = 0; s < t; s++) {
                float ls = LP[(size_t)t * (size_t)C + (size_t)s];
                const float *Es = G + (size_t)s * (size_t)dv;
                int j = 0;
                for (; j + 3 < dv; j += 4) {
                    Gt[j] -= ls * Es[j];
                    Gt[j + 1] -= ls * Es[j + 1];
                    Gt[j + 2] -= ls * Es[j + 2];
                    Gt[j + 3] -= ls * Es[j + 3];
                }
                for (; j < dv; j++) {
                    Gt[j] -= ls * Es[j];
                }
            }
        }

        // 4) P ([C][C] 下三角+対角、L 領域を再利用):
        // P[t][s] = q_t^T diag(cum(t,s)) k_s。
        for (size_t n = 0; n < C2; n++) {
            LP[n] = 0.0f;
        }
        for (int s = 0; s < C; s++) {
            const float *Ks = Kn + (size_t)s * (size_t)dk;
            for (int i = 0; i < dk; i++) {
                tmp[i] = 1.0f;
            }
            for (int t = s; t < C; t++) {
                const float *QTt = Qn + (size_t)t * (size_t)dk;
                float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
                int i = 0;
                if (t != s) {
                    const float *At = Alpha + (size_t)t * (size_t)dk;
                    for (i = 0; i < dk; i++) {
                        tmp[i] *= At[i];
                    }
                }
                for (i = 0; i + 3 < dk; i += 4) {
                    s0 += QTt[i] * tmp[i] * Ks[i];
                    s1 += QTt[i + 1] * tmp[i + 1] * Ks[i + 1];
                    s2 += QTt[i + 2] * tmp[i + 2] * Ks[i + 2];
                    s3 += QTt[i + 3] * tmp[i + 3] * Ks[i + 3];
                }
                for (; i < dk; i++) {
                    s0 += QTt[i] * tmp[i] * Ks[i];
                }
                LP[(size_t)t * (size_t)C + (size_t)s] =
                    (s0 + s1) + (s2 + s3);
            }
        }

        // 5) Out_t = (q_t⊙cumfwd(t))^T S_in + Σ_{s≦t} P[t][s] E_s。
        for (int i = 0; i < dk; i++) {
            tmp[i] = 1.0f;
        }
        for (int t = 0; t < C; t++) {
            const float *At = Alpha + (size_t)t * (size_t)dk;
            const float *QTt = Qn + (size_t)t * (size_t)dk;
            float *Ot = Out + (size_t)t * (size_t)dv;
            for (int i = 0; i < dk; i++) {
                tmp[i] *= At[i];
            }
            for (int j = 0; j < dv; j++) {
                Ot[j] = 0.0f;
            }
            for (int i = 0; i < dk; i++) {
                float wi = QTt[i] * tmp[i];
                const float *Sr = S + (size_t)i * (size_t)dv;
                int j = 0;
                for (; j + 3 < dv; j += 4) {
                    Ot[j] += wi * Sr[j];
                    Ot[j + 1] += wi * Sr[j + 1];
                    Ot[j + 2] += wi * Sr[j + 2];
                    Ot[j + 3] += wi * Sr[j + 3];
                }
                for (; j < dv; j++) {
                    Ot[j] += wi * Sr[j];
                }
            }
            for (int s = 0; s <= t; s++) {
                float ps = LP[(size_t)t * (size_t)C + (size_t)s];
                const float *Es = G + (size_t)s * (size_t)dv;
                int j = 0;
                for (; j + 3 < dv; j += 4) {
                    Ot[j] += ps * Es[j];
                    Ot[j + 1] += ps * Es[j + 1];
                    Ot[j + 2] += ps * Es[j + 2];
                    Ot[j + 3] += ps * Es[j + 3];
                }
                for (; j < dv; j++) {
                    Ot[j] += ps * Es[j];
                }
            }
        }

        // 6) S_out (inter-chunk  carry は境界のみ):
        // Wgt[s][i] = k_s[i] cum(C-1,s)[i] を後退累積で Pdec に生成。
        for (int i = 0; i < dk; i++) {
            tmp[i] = 1.0f;
        }
        for (int s = C - 1; s >= 0; s--) {
            const float *Ks = Kn + (size_t)s * (size_t)dk;
            const float *As = Alpha + (size_t)s * (size_t)dk;
            float *Ws = Pdec + (size_t)s * (size_t)dk;
            for (int i = 0; i < dk; i++) {
                Ws[i] = Ks[i] * tmp[i];
                tmp[i] *= As[i];
            }
        }
        // cumfwd(C-1) を再生成し S_in をスケール。
        for (int i = 0; i < dk; i++) {
            tmp[i] = 1.0f;
        }
        for (int t = 0; t < C; t++) {
            const float *At = Alpha + (size_t)t * (size_t)dk;
            for (int i = 0; i < dk; i++) {
                tmp[i] *= At[i];
            }
        }
        for (int i = 0; i < dk; i++) {
            float cf = tmp[i];
            float *Sr = S + (size_t)i * (size_t)dv;
            int j = 0;
            for (; j + 3 < dv; j += 4) {
                Sr[j] *= cf;
                Sr[j + 1] *= cf;
                Sr[j + 2] *= cf;
                Sr[j + 3] *= cf;
            }
            for (; j < dv; j++) {
                Sr[j] *= cf;
            }
        }
        for (int s = 0; s < C; s++) {
            const float *Es = G + (size_t)s * (size_t)dv;
            const float *Ws = Pdec + (size_t)s * (size_t)dk;
            for (int i = 0; i < dk; i++) {
                float w = Ws[i];
                float *Sr = S + (size_t)i * (size_t)dv;
                int j = 0;
                for (; j + 3 < dv; j += 4) {
                    Sr[j] += w * Es[j];
                    Sr[j + 1] += w * Es[j + 1];
                    Sr[j + 2] += w * Es[j + 2];
                    Sr[j + 3] += w * Es[j + 3];
                }
                for (; j < dv; j++) {
                    Sr[j] += w * Es[j];
                }
            }
        }
    }

    rc = JT_OK;
cleanup:
    return rc;
}

int jt_gdn2_prefill_chunk16(float *restrict S, float *restrict Out,
                            const float *restrict Q, const float *restrict K,
                            const float *restrict V, const float *restrict B,
                            const float *restrict W,
                            const float *restrict Alpha, int dk, int dv,
                            float *restrict scratch, size_t scratch_n) {
    return jt_gdn2_prefill_chunk_impl(S, Out, Q, K, V, B, W, Alpha, dk, dv,
                                      scratch, scratch_n, 16);
}

int jt_gdn2_prefill_chunk32(float *restrict S, float *restrict Out,
                            const float *restrict Q, const float *restrict K,
                            const float *restrict V, const float *restrict B,
                            const float *restrict W,
                            const float *restrict Alpha, int dk, int dv,
                            float *restrict scratch, size_t scratch_n) {
    return jt_gdn2_prefill_chunk_impl(S, Out, Q, K, V, B, W, Alpha, dk, dv,
                                      scratch, scratch_n, 32);
}
