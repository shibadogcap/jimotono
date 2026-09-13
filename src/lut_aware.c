// jt_lut_aware: HGQ-LUT方式の学習足場。
// forwardはfp32テンソル演算 + 量子化誤差記録、backwardはSTE素通し。
// exportはレイアウト記述子 + 実バイナリ (jt_lut_export_binary、64B整列・
// 同ページ配置、行単位per-block量子化をパック書出し)。
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
    int rc = JT_ERR_INVAL;
    jt_lut_export_desc_t desc;
    uint64_t elems = 0;
    uint64_t colblocks = 0;
    uint64_t scale_elems = 0;
    size_t nelems = 0;
    size_t nscales = 0;
    int8_t *q = NULL;
    float *scales = NULL;
    unsigned char *dst = NULL;

    if (out_written != NULL) {
        *out_written = 0;
    }
    if (W == NULL || out == NULL) {
        errno = EINVAL;
        rc = JT_ERR_INVAL;
        goto cleanup;
    }
    // 64B整列要求 (DESIGN §5.1-5、T-MAC kAllocAlignment=64)。
    // 既存流儀ではなくタスク指定により不整列はINVAL (errno=EINVAL)。
    if (((uintptr_t)(const void *)out % (uintptr_t)JT_LUT_ALIGN) != 0u) {
        errno = EINVAL;
        rc = JT_ERR_INVAL;
        goto cleanup;
    }
    // レイアウトは記述子が唯一の真実 (table/scale/scale_off/page/same_page)。
    memset(&desc, 0, sizeof(desc));
    rc = jt_lut_export_desc(rows, cols, bits, block, &desc);
    if (rc != JT_OK) {
        goto cleanup;  // errnoは下位で設定済み
    }
    if (out_cap < (size_t)desc.page_bytes) {
        // page_bytesはalign_up済み64の倍数。out_cap不足はINVAL (既存流儀)。
        errno = EINVAL;
        rc = JT_ERR_INVAL;
        goto cleanup;
    }
    elems = (uint64_t)rows * (uint64_t)cols;
    colblocks = ((uint64_t)cols + (uint64_t)block - 1u) / (uint64_t)block;
    scale_elems = (uint64_t)rows * colblocks;
    if (elems > (uint64_t)SIZE_MAX || scale_elems > (uint64_t)SIZE_MAX) {
        errno = ENOMEM;
        rc = JT_ERR_NOMEM;
        goto cleanup;
    }
    nelems = (size_t)elems;
    nscales = (size_t)scale_elems;

    // fail-closed: 先にW全体の有限性を検査 (outには触らない)。
    for (size_t i = 0; i < nelems; i++) {
        if (!isfinite((double)W[i])) {
            errno = EINVAL;
            rc = JT_ERR_INVAL;
            goto cleanup;
        }
    }

    q = (int8_t *)malloc((nelems > 0 ? nelems : 1) * sizeof(int8_t));
    scales = (float *)malloc((nscales > 0 ? nscales : 1) * sizeof(float));
    if (q == NULL || scales == NULL) {
        errno = ENOMEM;
        rc = JT_ERR_NOMEM;
        goto cleanup;
    }

    // 行単位per-block量子化 (1行=cols要素をblock粒度で分割)。
    // scale配置: scales[r*colblocks + cb] (row-major、fp32 LE)。
    for (uint32_t r = 0; r < rows; r++) {
        const float *wrow = W + (size_t)r * (size_t)cols;
        int8_t *qrow = q + (size_t)r * (size_t)cols;
        float *srow = scales + (size_t)r * (size_t)colblocks;
        rc = jt_mp_quantize(wrow, (size_t)cols, bits, (size_t)block, qrow,
                            srow, (size_t)colblocks);
        if (rc != JT_OK) {
            goto cleanup;  // errnoは下位で設定済み
        }
    }

    // 記述子レイアウト通りに書き出す。パディングはゼロ埋め。
    // テーブル packing (LE前提、row-major線形順):
    // - bits=8: 生int8をそのまま1B/要素 (two's complement)。
    // - bits=4: 2要素/B、下位ニブル=偶数index、上位ニブル=奇数index、
    //   各ニブルは4bit two's complement (q & 0xF)。端数Bの上位は0。
    // - bits=2: 4要素/B、bits[1:0]=i%4==0、[3:2]==1、[5:4]==2、[7:6]==3、
    //   各2bitはtwo's complement (q & 0x3)。端数Bの未使用上位は0。
    dst = (unsigned char *)out;
    memset(dst, 0, (size_t)desc.page_bytes);
    if (bits == 8) {
        for (size_t i = 0; i < nelems; i++) {
            dst[i] = (unsigned char)q[i];
        }
    } else if (bits == 4) {
        for (size_t i = 0; i < nelems; i++) {
            unsigned nib = ((unsigned)q[i]) & 0xFu;
            size_t bi = i / 2u;
            if ((i % 2u) == 0u) {
                dst[bi] |= (unsigned char)nib;
            } else {
                dst[bi] |= (unsigned char)(nib << 4);
            }
        }
    } else {
        for (size_t i = 0; i < nelems; i++) {
            unsigned v = ((unsigned)q[i]) & 0x3u;
            size_t bi = i / 4u;
            unsigned sh = (unsigned)(i % 4u) * 2u;
            dst[bi] |= (unsigned char)(v << sh);
        }
    }
    memcpy(dst + (size_t)desc.scale_offset, scales,
           (size_t)desc.scale_bytes);
    // 末尾パディングは既にゼロ (align_up分)。

    if (out_written != NULL) {
        *out_written = (size_t)desc.page_bytes;
    }
    rc = JT_OK;

cleanup:
    free(q);
    free(scales);
    if (rc != JT_OK && out_written != NULL) {
        *out_written = 0;
    }
    return rc;
}
