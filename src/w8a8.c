// jt_w8a8: Stage 3 W8A8 INT8量子化・逆量子化・int8 GEMM。
// P3b §3準拠 (W8A8本線。T-MAC/INT4/INT2は非スコープ)。
// C11・restrict・errno＋goto cleanup・クロスプラット (AGENTS.MD 7.1)。
// AVX-512不使用 (分岐なし)。スカラーのみ (SIMD化は将来のAVX2-VNNI/NEON-I8MM)。

#include "jimotono/w8a8.h"

#include <errno.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>

#include "jimotono/common.h"
#include "jimotono/train_bwd.h"

// 既定OFF (fp32)。プロセス内グローバル。
static int g_jt_w8a8_enabled = 0;

int jt_w8a8_set_enabled(int enabled) {
    if (enabled != 0 && enabled != 1) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    g_jt_w8a8_enabled = enabled;
    return JT_OK;
}

int jt_w8a8_get_enabled(int *restrict out_enabled) {
    if (out_enabled == NULL) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    *out_enabled = g_jt_w8a8_enabled;
    return JT_OK;
}

int jt_w8a8_is_enabled(void) { return g_jt_w8a8_enabled; }

static int jt_w8a8_valid_dims_q(size_t n) { return n > 0; }

int jt_w8a8_quant_per_tensor(const float *restrict src, size_t n,
                             int8_t *restrict dst,
                             float *restrict out_scale) {
    int rc = JT_ERR_INVAL;
    double maxabs = 0.0;
    double scale_d = 0.0;
    float scale_f = 1.0f;
    if (src == NULL || dst == NULL || out_scale == NULL) {
        errno = EINVAL;
        goto cleanup;
    }
    if (!jt_w8a8_valid_dims_q(n)) {
        errno = EINVAL;
        goto cleanup;
    }
    // fail-closed: 先に全要素の有限性を検査し、NaN/Inf混入時は書き込まない。
    for (size_t i = 0; i < n; i++) {
        if (!isfinite((double)src[i])) {
            errno = EINVAL;
            goto cleanup;
        }
    }
    for (size_t i = 0; i < n; i++) {
        double a = fabs((double)src[i]);
        if (a > maxabs) {
            maxabs = a;
        }
    }
    if (maxabs == 0.0) {
        *out_scale = 1.0f;
        for (size_t i = 0; i < n; i++) {
            dst[i] = 0;
        }
        rc = JT_OK;
        goto cleanup;
    }
    scale_d = maxabs / (double)JT_W8A8_QMAX;
    if (!isfinite(scale_d) || scale_d <= 0.0) {
        errno = ERANGE;
        goto cleanup;
    }
    scale_f = (float)scale_d;
    if (!isfinite((double)scale_f) || scale_f <= 0.0f) {
        errno = ERANGE;
        goto cleanup;
    }
    *out_scale = scale_f;
    for (size_t i = 0; i < n; i++) {
        double qd = round((double)src[i] / scale_d);
        if (qd > (double)JT_W8A8_QMAX) {
            qd = (double)JT_W8A8_QMAX;
        } else if (qd < (double)JT_W8A8_QMIN) {
            qd = (double)JT_W8A8_QMIN;
        }
        dst[i] = (int8_t)((int)qd);
    }
    rc = JT_OK;
cleanup:
    return rc;
}

int jt_w8a8_dequant_per_tensor(const int8_t *restrict src, float scale,
                               size_t n, float *restrict dst) {
    int rc = JT_ERR_INVAL;
    double sc = 0.0;
    if (src == NULL || dst == NULL) {
        errno = EINVAL;
        goto cleanup;
    }
    if (!jt_w8a8_valid_dims_q(n)) {
        errno = EINVAL;
        goto cleanup;
    }
    sc = (double)scale;
    if (!isfinite(sc) || sc <= 0.0) {
        errno = EINVAL;
        goto cleanup;
    }
    for (size_t i = 0; i < n; i++) {
        int qv = (int)src[i];
        if (qv < JT_W8A8_QMIN || qv > JT_W8A8_QMAX) {
            errno = EINVAL;
            goto cleanup;
        }
    }
    for (size_t i = 0; i < n; i++) {
        double v = (double)((int)src[i]) * sc;
        float f = (float)v;
        if (!isfinite((double)f)) {
            errno = ERANGE;
            goto cleanup;
        }
        dst[i] = f;
    }
    rc = JT_OK;
cleanup:
    return rc;
}

int jt_w8a8_quant_per_channel(const float *restrict src, size_t rows,
                              size_t cols, int8_t *restrict dst,
                              float *restrict scales) {
    int rc = JT_ERR_INVAL;
    if (src == NULL || dst == NULL || scales == NULL) {
        errno = EINVAL;
        goto cleanup;
    }
    if (rows == 0 || cols == 0) {
        errno = EINVAL;
        goto cleanup;
    }
    if (rows > SIZE_MAX / cols) {
        errno = EINVAL;
        goto cleanup;
    }
    {
        size_t n = rows * cols;
        for (size_t i = 0; i < n; i++) {
            if (!isfinite((double)src[i])) {
                errno = EINVAL;
                goto cleanup;
            }
        }
    }
    for (size_t r = 0; r < rows; r++) {
        const float *srow = src + r * cols;
        int8_t *drow = dst + r * cols;
        double maxabs = 0.0;
        double scale_d = 0.0;
        float scale_f = 1.0f;
        for (size_t c = 0; c < cols; c++) {
            double a = fabs((double)srow[c]);
            if (a > maxabs) {
                maxabs = a;
            }
        }
        if (maxabs == 0.0) {
            scales[r] = 1.0f;
            for (size_t c = 0; c < cols; c++) {
                drow[c] = 0;
            }
            continue;
        }
        scale_d = maxabs / (double)JT_W8A8_QMAX;
        if (!isfinite(scale_d) || scale_d <= 0.0) {
            errno = ERANGE;
            goto cleanup;
        }
        scale_f = (float)scale_d;
        if (!isfinite((double)scale_f) || scale_f <= 0.0f) {
            errno = ERANGE;
            goto cleanup;
        }
        scales[r] = scale_f;
        for (size_t c = 0; c < cols; c++) {
            double qd = round((double)srow[c] / scale_d);
            if (qd > (double)JT_W8A8_QMAX) {
                qd = (double)JT_W8A8_QMAX;
            } else if (qd < (double)JT_W8A8_QMIN) {
                qd = (double)JT_W8A8_QMIN;
            }
            drow[c] = (int8_t)((int)qd);
        }
    }
    rc = JT_OK;
cleanup:
    return rc;
}

int jt_w8a8_dequant_per_channel(const int8_t *restrict src,
                                const float *restrict scales, size_t rows,
                                size_t cols, float *restrict dst) {
    int rc = JT_ERR_INVAL;
    if (src == NULL || scales == NULL || dst == NULL) {
        errno = EINVAL;
        goto cleanup;
    }
    if (rows == 0 || cols == 0) {
        errno = EINVAL;
        goto cleanup;
    }
    if (rows > SIZE_MAX / cols) {
        errno = EINVAL;
        goto cleanup;
    }
    for (size_t r = 0; r < rows; r++) {
        double sc = (double)scales[r];
        if (!isfinite(sc) || sc <= 0.0) {
            errno = EINVAL;
            goto cleanup;
        }
    }
    {
        size_t n = rows * cols;
        for (size_t i = 0; i < n; i++) {
            int qv = (int)src[i];
            if (qv < JT_W8A8_QMIN || qv > JT_W8A8_QMAX) {
                errno = EINVAL;
                goto cleanup;
            }
        }
    }
    for (size_t r = 0; r < rows; r++) {
        double sc = (double)scales[r];
        const int8_t *srow = src + r * cols;
        float *drow = dst + r * cols;
        for (size_t c = 0; c < cols; c++) {
            double v = (double)((int)srow[c]) * sc;
            float f = (float)v;
            if (!isfinite((double)f)) {
                errno = ERANGE;
                goto cleanup;
            }
            drow[c] = f;
        }
    }
    rc = JT_OK;
cleanup:
    return rc;
}

// int8 GEMM共通検証。M==0は呼出し側で起動スキップ扱い。
static int jt_w8a8_gemm_validate(const int8_t *restrict Aq,
                                 const int8_t *restrict Bq, float sA,
                                 const float *restrict sB_col,
                                 const float *restrict C, int M, int N,
                                 int K, int need_col) {
    if (Aq == NULL || Bq == NULL || C == NULL) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    if (M < 0 || N <= 0 || K <= 0) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    if (M == 0) {
        return JT_OK;  // 起動スキップ (呼出し側でC不変)
    }
    if (M > JT_BWD_MAX_WIDE || N > JT_BWD_MAX_WIDE || K > JT_BWD_MAX_WIDE) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    if ((size_t)M > SIZE_MAX / (size_t)K ||
        (size_t)M > SIZE_MAX / (size_t)N ||
        (size_t)K > SIZE_MAX / (size_t)N) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    if (!isfinite((double)sA) || sA <= 0.0f) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    if (need_col) {
        if (sB_col == NULL) {
            errno = EINVAL;
            return JT_ERR_INVAL;
        }
        for (int n = 0; n < N; n++) {
            if (!isfinite((double)sB_col[n]) || sB_col[n] <= 0.0f) {
                errno = EINVAL;
                return JT_ERR_INVAL;
            }
        }
    }
    return JT_OK;
}

int jt_w8a8_gemm_per_tensor(const int8_t *restrict Aq,
                            const int8_t *restrict Bq, float sA, float sB,
                            float *restrict C, int M, int N, int K) {
    int rc = JT_ERR_INVAL;
    if (M == 0) {
        // jt_gemm_mat_f32と同一契約: M==0はC不変でJT_OK。ただしN/Kは正。
        if (Aq == NULL || Bq == NULL || C == NULL) {
            errno = EINVAL;
            goto cleanup;
        }
        if (N <= 0 || K <= 0) {
            errno = EINVAL;
            goto cleanup;
        }
        rc = JT_OK;
        goto cleanup;
    }
    if (jt_w8a8_gemm_validate(Aq, Bq, sA, NULL, C, M, N, K, 0) != JT_OK) {
        goto cleanup;  // errnoは下位で設定済み
    }
    if (!isfinite((double)sB) || sB <= 0.0f) {
        errno = EINVAL;
        goto cleanup;
    }
    // q値域検査 (破損qの黙殺防止。O(MK+KN))。
    for (int m = 0; m < M; m++) {
        const int8_t *arow = Aq + (size_t)m * (size_t)K;
        for (int k = 0; k < K; k++) {
            int qv = (int)arow[k];
            if (qv < JT_W8A8_QMIN || qv > JT_W8A8_QMAX) {
                errno = EINVAL;
                goto cleanup;
            }
        }
    }
    for (int k = 0; k < K; k++) {
        const int8_t *brow = Bq + (size_t)k * (size_t)N;
        for (int n = 0; n < N; n++) {
            int qv = (int)brow[n];
            if (qv < JT_W8A8_QMIN || qv > JT_W8A8_QMAX) {
                errno = EINVAL;
                goto cleanup;
            }
        }
    }
    {
        double ss = (double)sA * (double)sB;
        if (!isfinite(ss) || ss <= 0.0) {
            errno = ERANGE;
            goto cleanup;
        }
        for (int m = 0; m < M; m++) {
            const int8_t *arow = Aq + (size_t)m * (size_t)K;
            float *crow = C + (size_t)m * (size_t)N;
            for (int n = 0; n < N; n++) {
                int32_t acc = 0;
                double v = 0.0;
                float f = 0.0f;
                for (int k = 0; k < K; k++) {
                    acc += (int32_t)arow[k] *
                           (int32_t)Bq[(size_t)k * (size_t)N + (size_t)n];
                }
                v = (double)acc * ss;
                f = (float)v;
                if (!isfinite((double)f)) {
                    errno = ERANGE;
                    goto cleanup;
                }
                crow[n] = f;
            }
        }
    }
    rc = JT_OK;
cleanup:
    return rc;
}

int jt_w8a8_gemm_per_channel(const int8_t *restrict Aq,
                             const int8_t *restrict Bq, float sA,
                             const float *restrict sB_col,
                             float *restrict C, int M, int N, int K) {
    int rc = JT_ERR_INVAL;
    if (M == 0) {
        if (Aq == NULL || Bq == NULL || C == NULL) {
            errno = EINVAL;
            goto cleanup;
        }
        if (N <= 0 || K <= 0) {
            errno = EINVAL;
            goto cleanup;
        }
        rc = JT_OK;
        goto cleanup;
    }
    if (jt_w8a8_gemm_validate(Aq, Bq, sA, sB_col, C, M, N, K, 1) != JT_OK) {
        goto cleanup;
    }
    for (int m = 0; m < M; m++) {
        const int8_t *arow = Aq + (size_t)m * (size_t)K;
        for (int k = 0; k < K; k++) {
            int qv = (int)arow[k];
            if (qv < JT_W8A8_QMIN || qv > JT_W8A8_QMAX) {
                errno = EINVAL;
                goto cleanup;
            }
        }
    }
    for (int k = 0; k < K; k++) {
        const int8_t *brow = Bq + (size_t)k * (size_t)N;
        for (int n = 0; n < N; n++) {
            int qv = (int)brow[n];
            if (qv < JT_W8A8_QMIN || qv > JT_W8A8_QMAX) {
                errno = EINVAL;
                goto cleanup;
            }
        }
    }
    for (int m = 0; m < M; m++) {
        const int8_t *arow = Aq + (size_t)m * (size_t)K;
        float *crow = C + (size_t)m * (size_t)N;
        for (int n = 0; n < N; n++) {
            int32_t acc = 0;
            double ss = 0.0;
            double v = 0.0;
            float f = 0.0f;
            for (int k = 0; k < K; k++) {
                acc += (int32_t)arow[k] *
                       (int32_t)Bq[(size_t)k * (size_t)N + (size_t)n];
            }
            ss = (double)sA * (double)sB_col[n];
            if (!isfinite(ss) || ss <= 0.0) {
                errno = ERANGE;
                goto cleanup;
            }
            v = (double)acc * ss;
            f = (float)v;
            if (!isfinite((double)f)) {
                errno = ERANGE;
                goto cleanup;
            }
            crow[n] = f;
        }
    }
    rc = JT_OK;
cleanup:
    return rc;
}

int jt_w8a8_fakequant_per_tensor(const float *restrict src, size_t n,
                                 float *restrict dst) {
    int rc = JT_ERR_INVAL;
    double maxabs = 0.0;
    double scale_d = 0.0;
    if (src == NULL || dst == NULL) {
        errno = EINVAL;
        goto cleanup;
    }
    if (!jt_w8a8_valid_dims_q(n)) {
        errno = EINVAL;
        goto cleanup;
    }
    for (size_t i = 0; i < n; i++) {
        if (!isfinite((double)src[i])) {
            errno = EINVAL;
            goto cleanup;
        }
    }
    for (size_t i = 0; i < n; i++) {
        double a = fabs((double)src[i]);
        if (a > maxabs) {
            maxabs = a;
        }
    }
    if (maxabs == 0.0) {
        for (size_t i = 0; i < n; i++) {
            dst[i] = 0.0f;
        }
        rc = JT_OK;
        goto cleanup;
    }
    scale_d = maxabs / (double)JT_W8A8_QMAX;
    if (!isfinite(scale_d) || scale_d <= 0.0) {
        errno = ERANGE;
        goto cleanup;
    }
    for (size_t i = 0; i < n; i++) {
        double qd = round((double)src[i] / scale_d);
        double v = 0.0;
        float f = 0.0f;
        if (qd > (double)JT_W8A8_QMAX) {
            qd = (double)JT_W8A8_QMAX;
        } else if (qd < (double)JT_W8A8_QMIN) {
            qd = (double)JT_W8A8_QMIN;
        }
        v = qd * scale_d;
        f = (float)v;
        if (!isfinite((double)f)) {
            errno = ERANGE;
            goto cleanup;
        }
        dst[i] = f;
    }
    rc = JT_OK;
cleanup:
    return rc;
}

int jt_w8a8_fakequant_per_channel(const float *restrict src, size_t rows,
                                  size_t cols, float *restrict dst) {
    int rc = JT_ERR_INVAL;
    if (src == NULL || dst == NULL) {
        errno = EINVAL;
        goto cleanup;
    }
    if (rows == 0 || cols == 0) {
        errno = EINVAL;
        goto cleanup;
    }
    if (rows > SIZE_MAX / cols) {
        errno = EINVAL;
        goto cleanup;
    }
    {
        size_t n = rows * cols;
        for (size_t i = 0; i < n; i++) {
            if (!isfinite((double)src[i])) {
                errno = EINVAL;
                goto cleanup;
            }
        }
    }
    for (size_t r = 0; r < rows; r++) {
        const float *srow = src + r * cols;
        float *drow = dst + r * cols;
        double maxabs = 0.0;
        double scale_d = 0.0;
        for (size_t c = 0; c < cols; c++) {
            double a = fabs((double)srow[c]);
            if (a > maxabs) {
                maxabs = a;
            }
        }
        if (maxabs == 0.0) {
            for (size_t c = 0; c < cols; c++) {
                drow[c] = 0.0f;
            }
            continue;
        }
        scale_d = maxabs / (double)JT_W8A8_QMAX;
        if (!isfinite(scale_d) || scale_d <= 0.0) {
            errno = ERANGE;
            goto cleanup;
        }
        for (size_t c = 0; c < cols; c++) {
            double qd = round((double)srow[c] / scale_d);
            double v = 0.0;
            float f = 0.0f;
            if (qd > (double)JT_W8A8_QMAX) {
                qd = (double)JT_W8A8_QMAX;
            } else if (qd < (double)JT_W8A8_QMIN) {
                qd = (double)JT_W8A8_QMIN;
            }
            v = qd * scale_d;
            f = (float)v;
            if (!isfinite((double)f)) {
                errno = ERANGE;
                goto cleanup;
            }
            drow[c] = f;
        }
    }
    rc = JT_OK;
cleanup:
    return rc;
}
