// jt_mixed_prec: 層種別ビット幅ポリシー + per-block対称量子化 (素朴核)。
// AGENTS.MD §2.2 / FRI-MxMoE方針は mixed_prec.h の注記を参照。
// C11, restrict, errno + goto cleanup。スカラーのみ (SIMD化は将来)。

#include "jimotono/mixed_prec.h"

#include <errno.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>

#include "jimotono/common.h"

static int jt_mp_bits_valid(int bits) {
    return bits == 2 || bits == 4 || bits == 8;
}

int jt_mp_bits_for(jt_mp_layer_t layer, int *restrict out_bits) {
    if (out_bits == NULL) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    *out_bits = 0;
    switch (layer) {
    case JT_MP_SHARED:
        *out_bits = 8;
        return JT_OK;
    case JT_MP_ROUTING_DOWN:
        *out_bits = 4;
        return JT_OK;
    case JT_MP_ROUTING_GATE:
        *out_bits = 2;
        return JT_OK;
    case JT_MP_ROUTING_UP:
        *out_bits = 2;
        return JT_OK;
    case JT_MP_ATTN:
        *out_bits = 8;
        return JT_OK;
    case JT_MP_EMB:
        *out_bits = 8;
        return JT_OK;
    default:
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
}

int jt_mp_qmax_for(int bits, int *restrict out_qmax) {
    if (out_qmax == NULL) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    *out_qmax = 0;
    switch (bits) {
    case 2:
        *out_qmax = 1;
        return JT_OK;
    case 4:
        *out_qmax = 7;
        return JT_OK;
    case 8:
        *out_qmax = 127;
        return JT_OK;
    default:
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
}

int jt_mp_nblocks(size_t n, size_t block, size_t *restrict out) {
    if (out == NULL) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    *out = 0;
    if (n == 0 || block == 0) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    if (n > SIZE_MAX - (block - 1u)) {
        errno = ENOMEM;
        return JT_ERR_NOMEM;
    }
    *out = (n + block - 1u) / block;
    return JT_OK;
}

int jt_mp_quantize(const float *restrict src, size_t n, int bits,
                   size_t block, int8_t *restrict dst,
                   float *restrict scales, size_t nblocks) {
    int rc = JT_OK;
    int qmax = 0;
    int qmin = 0;
    size_t need = 0;

    if (src == NULL || dst == NULL || scales == NULL) {
        errno = EINVAL;
        rc = JT_ERR_INVAL;
        goto cleanup;
    }
    if (n == 0 || block == 0) {
        errno = EINVAL;
        rc = JT_ERR_INVAL;
        goto cleanup;
    }
    if (!jt_mp_bits_valid(bits)) {
        errno = EINVAL;
        rc = JT_ERR_INVAL;
        goto cleanup;
    }
    if (jt_mp_nblocks(n, block, &need) != JT_OK) {
        // jt_mp_nblocksがerrnoを設定済み。rcを写す。
        rc = (errno == ENOMEM) ? JT_ERR_NOMEM : JT_ERR_INVAL;
        goto cleanup;
    }
    if (need != nblocks) {
        errno = EINVAL;
        rc = JT_ERR_INVAL;
        goto cleanup;
    }
    if (jt_mp_qmax_for(bits, &qmax) != JT_OK) {
        rc = JT_ERR_INVAL;
        goto cleanup;
    }
    qmin = -(qmax + 1);

    // fail-closed: 先に全要素の有限性を検査し、NaN/Inf混入時は書き込まない。
    for (size_t i = 0; i < n; i++) {
        if (!isfinite((double)src[i])) {
            errno = EINVAL;
            rc = JT_ERR_INVAL;
            goto cleanup;
        }
    }

    for (size_t b = 0; b < nblocks; b++) {
        size_t base = b * block;
        size_t len = (n - base) < block ? (n - base) : block;
        double maxabs = 0.0;
        double scale_d = 0.0;
        float scale_f = 1.0f;
        for (size_t i = 0; i < len; i++) {
            double a = fabs((double)src[base + i]);
            if (a > maxabs) {
                maxabs = a;
            }
        }
        if (maxabs == 0.0) {
            scales[b] = 1.0f;
            for (size_t i = 0; i < len; i++) {
                dst[base + i] = 0;
            }
            continue;
        }
        scale_d = maxabs / (double)qmax;
        if (!isfinite(scale_d) || scale_d <= 0.0) {
            errno = ERANGE;
            rc = JT_ERR_INVAL;
            goto cleanup;
        }
        scale_f = (float)scale_d;
        if (!isfinite((double)scale_f) || scale_f <= 0.0f) {
            errno = ERANGE;
            rc = JT_ERR_INVAL;
            goto cleanup;
        }
        scales[b] = scale_f;
        for (size_t i = 0; i < len; i++) {
            double qd = round((double)src[base + i] / scale_d);
            if (qd > (double)qmax) {
                qd = (double)qmax;
            } else if (qd < (double)qmin) {
                qd = (double)qmin;
            }
            dst[base + i] = (int8_t)((int)qd);
        }
    }

cleanup:
    return rc;
}

int jt_mp_dequantize(const int8_t *restrict src, const float *restrict scales,
                     size_t n, int bits, size_t block, size_t nblocks,
                     float *restrict dst) {
    int rc = JT_OK;
    int qmax = 0;
    int qmin = 0;
    size_t need = 0;

    if (src == NULL || scales == NULL || dst == NULL) {
        errno = EINVAL;
        rc = JT_ERR_INVAL;
        goto cleanup;
    }
    if (n == 0 || block == 0) {
        errno = EINVAL;
        rc = JT_ERR_INVAL;
        goto cleanup;
    }
    if (!jt_mp_bits_valid(bits)) {
        errno = EINVAL;
        rc = JT_ERR_INVAL;
        goto cleanup;
    }
    if (jt_mp_nblocks(n, block, &need) != JT_OK) {
        rc = (errno == ENOMEM) ? JT_ERR_NOMEM : JT_ERR_INVAL;
        goto cleanup;
    }
    if (need != nblocks) {
        errno = EINVAL;
        rc = JT_ERR_INVAL;
        goto cleanup;
    }
    if (jt_mp_qmax_for(bits, &qmax) != JT_OK) {
        rc = JT_ERR_INVAL;
        goto cleanup;
    }
    qmin = -(qmax + 1);

    for (size_t b = 0; b < nblocks; b++) {
        double sc = (double)scales[b];
        if (!isfinite(sc) || sc <= 0.0) {
            errno = EINVAL;
            rc = JT_ERR_INVAL;
            goto cleanup;
        }
    }
    // q値域検査 (破損qの黙殺防止)。
    for (size_t i = 0; i < n; i++) {
        int qv = (int)src[i];
        if (qv < qmin || qv > qmax) {
            errno = EINVAL;
            rc = JT_ERR_INVAL;
            goto cleanup;
        }
    }
    for (size_t i = 0; i < n; i++) {
        double sc = (double)scales[i / block];
        double v = (double)((int)src[i]) * sc;
        float f = (float)v;
        if (!isfinite((double)f)) {
            errno = ERANGE;
            rc = JT_ERR_INVAL;
            goto cleanup;
        }
        dst[i] = f;
    }

cleanup:
    return rc;
}

int jt_mp_sensitivity_score(const jt_mp_sensitivity_t *restrict st,
                            float *restrict out_score) {
    if (st == NULL || out_score == NULL) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    *out_score = 0.0f;
    if (!isfinite((double)st->grad_var) || !isfinite((double)st->act_range) ||
        !isfinite((double)st->hess_trace)) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    if (st->grad_var < 0.0f || st->act_range < 0.0f || st->hess_trace < 0.0f) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    {
        double s = log1p((double)st->grad_var) + log1p((double)st->act_range) +
                   log1p((double)st->hess_trace);
        float f = (float)s;
        if (!isfinite((double)f)) {
            errno = ERANGE;
            return JT_ERR_INVAL;
        }
        *out_score = f;
        return JT_OK;
    }
}

int jt_mp_propose_bits(jt_mp_layer_t base_layer,
                       const jt_mp_sensitivity_t *restrict st,
                       int *restrict out_bits) {
    int base = 0;
    float score = 0.0f;
    int rc = JT_OK;

    if (out_bits == NULL || st == NULL) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    *out_bits = 0;
    if (jt_mp_bits_for(base_layer, &base) != JT_OK) {
        return JT_ERR_INVAL;
    }
    rc = jt_mp_sensitivity_score(st, &score);
    if (rc != JT_OK) {
        return rc;
    }
    // 暫定しきい値 (FRI本実装までの予約位置。分布仮定スタブ)。
    if (score > 2.0f) {
        if (base == 2) {
            base = 4;
        } else if (base == 4) {
            base = 8;
        }
    } else if (score < 0.5f) {
        if (base == 8) {
            base = 4;
        } else if (base == 4) {
            base = 2;
        }
    }
    *out_bits = base;
    return JT_OK;
}
