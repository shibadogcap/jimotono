// jt_lowbit: routed expert 低ビット副線 (Stage 3b-2)。
// C11, restrict積極使用, errnoベース + goto cleanup。
// 設計: analysis/p3b-tmac-design.md (fake-quant/真LUT分離、奇対称レベル、
//   per-columnスケール、decode GEMV禁止=M==1 fallbackは行わず GEMM体制専用
//   とするが、M==0 no-opとK%32 fallbackは許容する)。

#include "jimotono/lowbit.h"

#include <errno.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "jimotono/tmac.h"

static int g_lowbit_mode = 0;

int jt_lowbit_set_mode(int mode) {
    if (mode < 0 || mode > 2) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    g_lowbit_mode = mode;
    return JT_OK;
}

int jt_lowbit_get_mode(void) {
    return g_lowbit_mode;
}

int jt_lowbit_bits_for(int is_down, int *restrict out_bits) {
    if (out_bits == NULL) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    // gate/up=INT2, down=INT4 (AGENTS §2.2)。is_down!=0 → down。
    *out_bits = (is_down != 0) ? 4 : 2;
    return JT_OK;
}

// 奇対称レベルの正規化値 [-1,1]: t(li) = (2*li-(n-1))/(n-1)。
static double jt_lowbit_level(int li, int bits) {
    double n = (double)((1 << bits) - 1);
    return (2.0 * (double)li - n) / n;
}

int jt_lowbit_fakequant(const float *restrict w, float *restrict fq,
                        size_t n, int bits) {
    if (w == NULL || fq == NULL || n == 0) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    if (bits != 2 && bits != 4) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    int rc = JT_OK;
    double maxa = 0.0;
    for (size_t i = 0; i < n; i++) {
        double v = (double)w[i];
        if (!isfinite(v)) {
            errno = EINVAL;
            return JT_ERR_INVAL;
        }
        double a = fabs(v);
        if (a > maxa) {
            maxa = a;
        }
    }
    int nl = 1 << bits;
    if (maxa == 0.0) {
        for (size_t i = 0; i < n; i++) {
            fq[i] = 0.0f;
        }
        return JT_OK;
    }
    for (size_t i = 0; i < n; i++) {
        double t = (double)w[i] / maxa;  // [-1,1]
        double li = floor(t * (double)(nl - 1) / 2.0 + (double)(nl - 1) / 2.0 + 0.5);
        long l = (long)li;
        if (l < 0) {
            l = 0;
        }
        if (l > (long)(nl - 1)) {
            l = (long)(nl - 1);
        }
        fq[i] = (float)(jt_lowbit_level((int)l, bits) * maxa);
    }
    (void)rc;
    return JT_OK;
}

int jt_lowbit_col_to_idx(const float *restrict col, size_t k, int bits,
                         uint8_t *restrict idx, float *restrict w_scale) {
    if (col == NULL || idx == NULL || w_scale == NULL || k == 0) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    if ((bits != 2 && bits != 4) || (k % 4u) != 0u) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    double maxa = 0.0;
    for (size_t i = 0; i < k; i++) {
        double v = (double)col[i];
        if (!isfinite(v)) {
            errno = EINVAL;
            return JT_ERR_INVAL;
        }
        double a = fabs(v);
        if (a > maxa) {
            maxa = a;
        }
    }
    int nl = 1 << bits;
    size_t ng = k / 4u;
    // idxレイアウト: idx[b*ng+g]の下位4bit = グループgの4重みのbit b。
    for (int b = 0; b < bits; b++) {
        for (size_t g = 0; g < ng; g++) {
            unsigned p = 0u;
            for (unsigned j = 0u; j < 4u; j++) {
                double t = (maxa == 0.0) ? 0.0 : (double)col[g * 4u + j] / maxa;
                double li = floor(t * (double)(nl - 1) / 2.0 +
                                  (double)(nl - 1) / 2.0 + 0.5);
                long l = (long)li;
                if (l < 0) {
                    l = 0;
                }
                if (l > (long)(nl - 1)) {
                    l = (long)(nl - 1);
                }
                if (((unsigned)l >> (unsigned)b) & 1u) {
                    p |= (1u << j);
                }
            }
            idx[(size_t)b * ng + g] = (uint8_t)p;
        }
    }
    *w_scale = (float)maxa;
    return JT_OK;
}

// fp32参照GEMM (fallback用・テスト参照用)。Y[M][N] = A[M][K]·B[K][N]、B row-major。
static void jt_lowbit_ref_gemm(const float *restrict A, const float *restrict B,
                               float *restrict Y, int m, int n, int k) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            double acc = 0.0;
            for (int t = 0; t < k; t++) {
                acc += (double)A[(size_t)i * (size_t)k + (size_t)t] *
                       (double)B[(size_t)t * (size_t)n + (size_t)j];
            }
            Y[(size_t)i * (size_t)n + (size_t)j] = (float)acc;
        }
    }
}

int jt_lowbit_gemm_lut(const float *restrict X, const float *restrict W,
                       float *restrict Y, int me, int h, int k, int bits) {
    if (X == NULL || W == NULL || Y == NULL) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    if (me < 0 || h <= 0 || k <= 0) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    if (bits != 2 && bits != 4) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    if (me == 0) {
        return JT_OK;
    }
    // K%32!=0はQLUT ctor制約のためfp32 fallback (文書化された例外)。
    if (((size_t)k % 32u) != 0u) {
        jt_lowbit_ref_gemm(X, W, Y, me, h, k);
        return JT_OK;
    }
    int rc = JT_ERR_IO;
    size_t ng = (size_t)k / 4u;
    size_t nb = (size_t)k / 32u;
    int8_t *qlut = NULL;
    float *scales = NULL;
    float *biases = NULL;
    uint8_t *idx = NULL;
    float *col = NULL;
    size_t qn = 0, pn = 0;

    if (jt_tmac_qlut_bytes((size_t)me, (size_t)k, &qn) != JT_OK ||
        jt_tmac_param_bytes((size_t)me, (size_t)k, &pn) != JT_OK) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    qlut = (int8_t *)malloc(qn);
    scales = (float *)malloc(pn);
    biases = (float *)malloc(pn);
    idx = (uint8_t *)malloc((size_t)bits * ng);
    col = (float *)malloc((size_t)k * sizeof(float));
    if (qlut == NULL || scales == NULL || biases == NULL || idx == NULL ||
        col == NULL) {
        errno = ENOMEM;
        rc = JT_ERR_NOMEM;
        goto cleanup;
    }
    // 活性QLUTを全行一括構築 (GEMM体制のamortize)。
    if (jt_tmac_lut_ctor(X, (size_t)me, (size_t)k, qlut, scales,
                         biases) != JT_OK) {
        rc = JT_ERR_INVAL;  // errnoは下位由来 (非有限等)。Yは未更新。
        goto cleanup;
    }
    // 行和 (符号補正用。§注記参照)。
    double *rowsum = (double *)malloc((size_t)me * sizeof(double));
    if (rowsum == NULL) {
        errno = ENOMEM;
        rc = JT_ERR_NOMEM;
        goto cleanup;
    }
    for (int m = 0; m < me; m++) {
        double s = 0.0;
        for (int t = 0; t < k; t++) {
            s += (double)X[(size_t)m * (size_t)k + (size_t)t];
        }
        rowsum[m] = s;
    }
    {
        double n1 = (double)((1 << bits) - 1);  // n-1
        for (int j = 0; j < h; j++) {
            float ws = 0.0f;
            for (int t = 0; t < k; t++) {
                col[t] = W[(size_t)t * (size_t)h + (size_t)j];
            }
            if (jt_lowbit_col_to_idx(col, (size_t)k, bits, idx, &ws) !=
                JT_OK) {
                rc = JT_ERR_INVAL;
                goto cleanup;
            }
            for (int m = 0; m < me; m++) {
                float o = 0.0f;
                if (jt_tmac_lookup_accum(qlut + (size_t)m * ng * 8u, idx,
                                         scales + (size_t)m * nb,
                                         biases + (size_t)m * nb, ng, nb,
                                         bits, ws, &o) != JT_OK) {
                    rc = JT_ERR_INVAL;
                    goto cleanup;
                }
                // 符号補正: LUT核は符号なしレベル和 Σa·li を返すため、
                // 真値 Σa·(2li-(n-1))/(n-1)·ws = (2·o-(n-1)·ws·A)/(n-1) に戻す。
                // Aは行和 (線形性によりブロック合算後でも成立)。
                Y[(size_t)m * (size_t)h + (size_t)j] =
                    (float)((2.0 * (double)o - n1 * (double)ws * rowsum[m]) /
                            n1);
            }
        }
    }
    free(rowsum);
    rowsum = NULL;
    rc = JT_OK;

cleanup:
    free(qlut);
    free(scales);
    free(biases);
    free(idx);
    free(col);
    return rc;
}
