// jt_checkpoint: 勾配チェックポインティング足場 (O(√n))。
// C11, errnoベース + goto cleanup、クロスプラットフォーム。
// 実再計算はモデル依存のためスタブ (ENOSYS)。純粋関数は完全実装。

#include "jimotono/checkpoint.h"

#include <errno.h>
#include <math.h>
#include <stddef.h>

#ifndef ENOSYS
#define ENOSYS 38
#endif

int jt_ckpt_num_segments(int n_layers, int *restrict out_seg) {
    if (out_seg == NULL) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    *out_seg = 0;
    if (n_layers <= 0 || n_layers > JT_CKPT_MAX_LAYERS) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    {
        double s = sqrt((double)n_layers);
        int seg = (int)s;
        if ((double)seg < s) {
            seg += 1;  // ceil
        }
        if (seg < 1) {
            seg = 1;
        }
        if (seg > n_layers) {
            seg = n_layers;
        }
        *out_seg = seg;
    }
    return JT_OK;
}

int jt_ckpt_boundaries(int n_layers, int n_seg, int *restrict bounds,
                       size_t bounds_n) {
    if (bounds == NULL) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    if (n_layers <= 0 || n_layers > JT_CKPT_MAX_LAYERS) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    if (n_seg <= 0 || n_seg > n_layers) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    if (bounds_n < (size_t)(n_seg + 1)) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    {
        int base = n_layers / n_seg;
        int rem = n_layers % n_seg;
        int cur = 0;
        bounds[0] = 0;
        for (int i = 0; i < n_seg; i++) {
            int len = base + ((i < rem) ? 1 : 0);
            cur += len;
            bounds[i + 1] = cur;
        }
    }
    return JT_OK;
}

int jt_ckpt_mem_estimate(int n_layers, int n_seg, size_t per_layer,
                         size_t *restrict out_full,
                         size_t *restrict out_stored,
                         size_t *restrict out_peak) {
    if (out_full == NULL || out_stored == NULL || out_peak == NULL) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    *out_full = 0;
    *out_stored = 0;
    *out_peak = 0;
    if (n_layers <= 0 || n_layers > JT_CKPT_MAX_LAYERS) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    if (n_seg <= 0 || n_seg > n_layers) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    if (per_layer == 0) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    if (per_layer > SIZE_MAX / (size_t)(n_layers + 1)) {
        errno = ENOMEM;
        return JT_ERR_NOMEM;
    }
    {
        size_t full = (size_t)n_layers * per_layer;
        size_t stored = (size_t)(n_seg + 1) * per_layer;
        int seg_max = (n_layers + n_seg - 1) / n_seg;  // ceil
        size_t peak = 0;
        if ((size_t)seg_max > SIZE_MAX / per_layer) {
            errno = ENOMEM;
            return JT_ERR_NOMEM;
        }
        peak = stored + (size_t)seg_max * per_layer;
        *out_full = full;
        *out_stored = stored;
        *out_peak = peak;
    }
    return JT_OK;
}

int jt_ckpt_saving_ratio(size_t full, size_t stored,
                         double *restrict out_ratio) {
    if (out_ratio == NULL) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    *out_ratio = 0.0;
    if (full == 0 || stored > full) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    *out_ratio = 1.0 - (double)stored / (double)full;
    if (!isfinite(*out_ratio)) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    return JT_OK;
}

int jt_ckpt_plan_init(const jt_ckpt_plan_t *restrict plan) {
    if (plan == NULL || plan->bounds == NULL) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    if (plan->n_layers <= 0 || plan->n_layers > JT_CKPT_MAX_LAYERS) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    if (plan->n_seg <= 0 || plan->n_seg > plan->n_layers) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    if (plan->bounds[0] != 0) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    if (plan->bounds[plan->n_seg] != plan->n_layers) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    for (int i = 0; i < plan->n_seg; i++) {
        if (!(plan->bounds[i] < plan->bounds[i + 1])) {
            errno = EINVAL;
            return JT_ERR_INVAL;
        }
    }
    return JT_OK;
}

int jt_ckpt_find_segment(const jt_ckpt_plan_t *restrict plan, int layer,
                         int *restrict out_seg_idx) {
    int rc = JT_ERR_INVAL;
    if (out_seg_idx == NULL) {
        errno = EINVAL;
        goto cleanup;
    }
    *out_seg_idx = 0;
    if (jt_ckpt_plan_init(plan) != JT_OK) {
        goto cleanup;  // errnoは下位で設定済み
    }
    if (layer < 0 || layer >= plan->n_layers) {
        errno = EINVAL;
        goto cleanup;
    }
    for (int i = 0; i < plan->n_seg; i++) {
        if (layer >= plan->bounds[i] && layer < plan->bounds[i + 1]) {
            *out_seg_idx = i;
            rc = JT_OK;
            goto cleanup;
        }
    }
    errno = EINVAL;  // 到達不能のはずだがfail-closed
cleanup:
    return rc;
}

int jt_ckpt_recompute_range(const jt_ckpt_plan_t *restrict plan, int seg_idx,
                            jt_ckpt_fwd_fn fwd, void *ctx) {
    (void)ctx;
    if (plan == NULL || fwd == NULL) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    if (jt_ckpt_plan_init(plan) != JT_OK) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    if (seg_idx < 0 || seg_idx >= plan->n_seg) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    // P2足場: 実forwardコールバックの配線はモデル側。正常入力はENOSYS。
    errno = ENOSYS;
    return JT_ERR_NOSUP;
}

int jt_ckpt_bench_estimate(int n_layers, int n_seg, size_t per_layer,
                           jt_ckpt_bench_t *restrict out) {
    int rc = JT_ERR_INVAL;
    size_t full = 0;
    size_t stored = 0;
    size_t peak = 0;
    double ratio = 0.0;
    if (out == NULL) {
        errno = EINVAL;
        goto cleanup;
    }
    out->full = 0;
    out->stored = 0;
    out->peak = 0;
    out->ratio = 0.0;
    if (jt_ckpt_mem_estimate(n_layers, n_seg, per_layer, &full, &stored,
                             &peak) != JT_OK) {
        goto cleanup;
    }
    if (jt_ckpt_saving_ratio(full, stored, &ratio) != JT_OK) {
        goto cleanup;
    }
    out->full = full;
    out->stored = stored;
    out->peak = peak;
    out->ratio = ratio;
    rc = JT_OK;
cleanup:
    return rc;
}
