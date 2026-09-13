// jt_lut_aware: HGQ-LUT方式の学習足場。
// forwardはfp32テンソル演算 + 量子化誤差記録、backwardはSTE素通し。
// exportはレイアウト記述子のみ (実バイナリはTODO/NOSUP)。
// C11, restrict, errno + goto cleanup。スカラー核。

#include "jimotono/lut_aware.h"

#include <errno.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "jimotono/common.h"
#include "jimotono/mixed_prec.h"

static uint64_t jt_align_up_u64(uint64_t v, uint64_t a) {
    // aは2の冪 (64/4096のみ使用)。オーバーフロー時はUINT64_MAX番兵。
    uint64_t m = a - 1u;
    if (v > UINT64_MAX - m) {
        return UINT64_MAX;
    }
    return (v + m) & ~m;
}

int jt_lut_dense_fwd(const float *restrict X, const float *restrict W,
                     const float *restrict b, float *restrict Y, size_t m,
                     size_t n, size_t k, int bits, size_t block,
                     jt_lut_err_t *restrict err) {
    int rc = JT_OK;
    int8_t *q = NULL;
    float *scales = NULL;
    float *wdq = NULL;
    size_t wsize = 0;
    size_t nblocks = 0;

    if (X == NULL || W == NULL || Y == NULL) {
        errno = EINVAL;
        rc = JT_ERR_INVAL;
        goto cleanup;
    }
    if (m == 0 || n == 0 || k == 0 || block == 0) {
        errno = EINVAL;
        rc = JT_ERR_INVAL;
        goto cleanup;
    }
    if (bits != 2 && bits != 4 && bits != 8) {
        errno = EINVAL;
        rc = JT_ERR_INVAL;
        goto cleanup;
    }
    if (k > SIZE_MAX / n) {
        errno = ENOMEM;
        rc = JT_ERR_NOMEM;
        goto cleanup;
    }
    wsize = k * n;
    if (m > SIZE_MAX / k || m * k > SIZE_MAX / n) {
        // 計算量番兵 (m*k*n積の桁あふれ防止の簡易検査)。
        errno = ENOMEM;
        rc = JT_ERR_NOMEM;
        goto cleanup;
    }
    if (jt_mp_nblocks(wsize, block, &nblocks) != JT_OK) {
        rc = (errno == ENOMEM) ? JT_ERR_NOMEM : JT_ERR_INVAL;
        goto cleanup;
    }

    // fail-closed: 先に全入力の有限性を検査 (Y/errは触らない)。
    for (size_t i = 0; i < m * k; i++) {
        if (!isfinite((double)X[i])) {
            errno = EINVAL;
            rc = JT_ERR_INVAL;
            goto cleanup;
        }
    }
    for (size_t i = 0; i < wsize; i++) {
        if (!isfinite((double)W[i])) {
            errno = EINVAL;
            rc = JT_ERR_INVAL;
            goto cleanup;
        }
    }
    if (b != NULL) {
        for (size_t j = 0; j < n; j++) {
            if (!isfinite((double)b[j])) {
                errno = EINVAL;
                rc = JT_ERR_INVAL;
                goto cleanup;
            }
        }
    }

    // fp32等価値の順伝播 (double累積)。
    for (size_t r = 0; r < m; r++) {
        for (size_t c = 0; c < n; c++) {
            double acc = (b != NULL) ? (double)b[c] : 0.0;
            for (size_t p = 0; p < k; p++) {
                acc += (double)X[r * k + p] * (double)W[p * n + c];
            }
            float f = (float)acc;
            if (!isfinite((double)f)) {
                errno = ERANGE;
                rc = JT_ERR_INVAL;
                goto cleanup;
            }
            Y[r * n + c] = f;
        }
    }

    if (err != NULL) {
        double se = 0.0;
        double mx = 0.0;
        double sa = 0.0;
        err->mse = 0.0f;
        err->max_abs_err = 0.0f;
        err->mean_abs_err = 0.0f;

        q = (int8_t *)malloc(wsize * sizeof(int8_t));
        scales = (float *)malloc(nblocks * sizeof(float));
        wdq = (float *)malloc(wsize * sizeof(float));
        if (q == NULL || scales == NULL || wdq == NULL) {
            errno = ENOMEM;
            rc = JT_ERR_NOMEM;
            goto cleanup;
        }
        // Wは有限検査済みのため量子化は成功するはず。失敗時はINVAL扱い。
        rc = jt_mp_quantize(W, wsize, bits, block, q, scales, nblocks);
        if (rc != JT_OK) {
            goto cleanup;
        }
        rc = jt_mp_dequantize(q, scales, wsize, bits, block, nblocks, wdq);
        if (rc != JT_OK) {
            goto cleanup;
        }
        for (size_t i = 0; i < wsize; i++) {
            double d = fabs((double)W[i] - (double)wdq[i]);
            se += d * d;
            sa += d;
            if (d > mx) {
                mx = d;
            }
        }
        {
            double mse = se / (double)wsize;
            double mae = sa / (double)wsize;
            if (!isfinite(mse) || !isfinite(mae) || !isfinite(mx)) {
                errno = ERANGE;
                rc = JT_ERR_INVAL;
                goto cleanup;
            }
            err->mse = (float)mse;
            err->mean_abs_err = (float)mae;
            err->max_abs_err = (float)mx;
        }
        rc = JT_OK;
    }

cleanup:
    free(q);
    free(scales);
    free(wdq);
    if (rc != JT_OK && err != NULL) {
        // 誤差統計は失敗時に信用させない (ゼロ埋め)。
        // Yは部分書き込み済みの可能性があるため呼び出し側で破棄すること。
        memset(err, 0, sizeof(*err));
    }
    return rc;
}

int jt_lut_ste_pass(const float *restrict upstream, float *restrict downstream,
                    size_t count) {
    if (upstream == NULL || downstream == NULL) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    if (count == 0) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    // 別名検査は呼び出し規約 (restrict)。有限ガード付き素通し。
    for (size_t i = 0; i < count; i++) {
        if (!isfinite((double)upstream[i])) {
            errno = EINVAL;
            return JT_ERR_INVAL;
        }
    }
    // TODO(実勾配): 融合SwiGLU/RMSNorm核と結合時は量子化点前後の
    // ヤコビアンを考慮する。現足場は恒等STE (dL/dw = dL/dq)。
    memcpy(downstream, upstream, count * sizeof(float));
    return JT_OK;
}

int jt_lut_export_desc(uint32_t rows, uint32_t cols, int bits,
                       uint32_t block, jt_lut_export_desc_t *restrict desc) {
    uint64_t elems = 0;
    uint64_t table = 0;
    uint64_t colblocks = 0;
    uint64_t scale_elems = 0;
    uint64_t scaleb = 0;
    uint64_t soff = 0;
    uint64_t total = 0;

    if (desc == NULL) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    memset(desc, 0, sizeof(*desc));
    if (rows == 0 || cols == 0 || block == 0) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    if (bits != 2 && bits != 4 && bits != 8) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    elems = (uint64_t)rows * (uint64_t)cols;
    // table_bytes = ceil(elems*bits/8)。
    if (elems > (UINT64_MAX - 7u) / (uint64_t)bits) {
        errno = ENOMEM;
        return JT_ERR_NOMEM;
    }
    table = (elems * (uint64_t)bits + 7u) / 8u;
    colblocks = ((uint64_t)cols + (uint64_t)block - 1u) / (uint64_t)block;
    if (colblocks > UINT64_MAX / (uint64_t)rows) {
        errno = ENOMEM;
        return JT_ERR_NOMEM;
    }
    scale_elems = (uint64_t)rows * colblocks;
    if (scale_elems > UINT64_MAX / 4u) {
        errno = ENOMEM;
        return JT_ERR_NOMEM;
    }
    scaleb = scale_elems * 4u;
    soff = jt_align_up_u64(table, (uint64_t)JT_LUT_ALIGN);
    if (soff == UINT64_MAX) {
        errno = ENOMEM;
        return JT_ERR_NOMEM;
    }
    if (soff > UINT64_MAX - scaleb) {
        errno = ENOMEM;
        return JT_ERR_NOMEM;
    }
    total = jt_align_up_u64(soff + scaleb, (uint64_t)JT_LUT_ALIGN);
    if (total == UINT64_MAX) {
        errno = ENOMEM;
        return JT_ERR_NOMEM;
    }

    desc->magic = JT_LUT_EXPORT_MAGIC;
    desc->bits = (int32_t)bits;
    desc->rows = rows;
    desc->cols = cols;
    desc->block_len = block;
    desc->table_bytes = table;
    desc->scale_bytes = scaleb;
    desc->table_offset = 0u;
    desc->scale_offset = soff;
    desc->page_bytes = total;
    desc->same_page = (total <= (uint64_t)JT_LUT_PAGE_SIZE) ? 1 : 0;
    return JT_OK;
}

int jt_lut_export_binary(const float *restrict W, uint32_t rows,
                         uint32_t cols, int bits, uint32_t block,
                         void *restrict out, size_t out_cap,
                         size_t *restrict out_written) {
    (void)W;
    (void)rows;
    (void)cols;
    (void)bits;
    (void)block;
    (void)out;
    (void)out_cap;
    if (out_written != NULL) {
        *out_written = 0;
    }
    // TODO: Wをper-block量子化→bit-planeパックし、記述子レイアウトで
    // table+scaleを同ページ配置で書き出す。現状は未実装。
    errno = ENOSYS;
    return JT_ERR_NOSUP;
}
