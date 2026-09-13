// jt_optim8: 8bit AdamW-lite足場 (bitsandbytes-CPU流儀)。
// C11, restrict, errno + goto cleanup (AGENTS.MD 7.1)。malloc/freeのみ。
// llama.cpp等リンクなし。

#include "jimotono/optim8.h"

#include <errno.h>
#include <math.h>
#include <stdlib.h>

void jt_optim8_cfg_default(jt_optim8_cfg_t *restrict cfg) {
    if (cfg == NULL) {
        return;
    }
    cfg->lr = 1e-3f;
    cfg->beta1 = 0.9f;
    cfg->beta2 = 0.999f;
    cfg->eps = 1e-8f;
    cfg->weight_decay = 0.0f;
}

size_t jt_optim8_nblocks(size_t n) {
    return (n + (size_t)JT_OPTIM8_BLOCK - (size_t)1) / (size_t)JT_OPTIM8_BLOCK;
}

static int jt_optim8_cfg_valid(const jt_optim8_cfg_t *restrict cfg) {
    if (!isfinite(cfg->lr) || !(cfg->lr > 0.0f)) {
        return 0;
    }
    if (!isfinite(cfg->beta1) || !(cfg->beta1 > 0.0f) || !(cfg->beta1 < 1.0f)) {
        return 0;
    }
    if (!isfinite(cfg->beta2) || !(cfg->beta2 > 0.0f) || !(cfg->beta2 < 1.0f)) {
        return 0;
    }
    if (!isfinite(cfg->eps) || !(cfg->eps > 0.0f)) {
        return 0;
    }
    if (!isfinite(cfg->weight_decay) || !(cfg->weight_decay >= 0.0f)) {
        return 0;
    }
    return 1;
}

int jt_optim8_init(jt_optim8_t *restrict opt, size_t n,
                   const jt_optim8_cfg_t *restrict cfg) {
    if (opt == NULL || n == 0) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    jt_optim8_cfg_t c;
    if (cfg == NULL) {
        jt_optim8_cfg_default(&c);
    } else {
        c = *cfg;
    }
    if (!jt_optim8_cfg_valid(&c)) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    size_t nb = jt_optim8_nblocks(n);
    if (nb == 0) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }

    int rc = JT_OK;
    int8_t *qm = NULL;
    int8_t *qv = NULL;
    float *sm = NULL;
    float *sv = NULL;

    qm = (int8_t *)calloc(n, sizeof(int8_t));
    if (qm == NULL) {
        errno = ENOMEM;
        rc = JT_ERR_NOMEM;
        goto cleanup;
    }
    qv = (int8_t *)calloc(n, sizeof(int8_t));
    if (qv == NULL) {
        errno = ENOMEM;
        rc = JT_ERR_NOMEM;
        goto cleanup;
    }
    sm = (float *)malloc(nb * sizeof(float));
    if (sm == NULL) {
        errno = ENOMEM;
        rc = JT_ERR_NOMEM;
        goto cleanup;
    }
    sv = (float *)malloc(nb * sizeof(float));
    if (sv == NULL) {
        errno = ENOMEM;
        rc = JT_ERR_NOMEM;
        goto cleanup;
    }
    for (size_t b = 0; b < nb; b++) {
        sm[b] = 1.0f;
        sv[b] = 1.0f;
    }

    opt->n = n;
    opt->nblocks = nb;
    opt->lr = c.lr;
    opt->beta1 = c.beta1;
    opt->beta2 = c.beta2;
    opt->eps = c.eps;
    opt->wd = c.weight_decay;
    opt->step = 0;
    opt->q_m = qm;
    opt->q_v = qv;
    opt->s_m = sm;
    opt->s_v = sv;
    qm = NULL;
    qv = NULL;
    sm = NULL;
    sv = NULL;

cleanup:
    free(sv);
    free(sm);
    free(qv);
    free(qm);
    return rc;
}

void jt_optim8_fini(jt_optim8_t *restrict opt) {
    if (opt == NULL) {
        return;
    }
    free(opt->q_m);
    free(opt->q_v);
    free(opt->s_m);
    free(opt->s_v);
    opt->q_m = NULL;
    opt->q_v = NULL;
    opt->s_m = NULL;
    opt->s_v = NULL;
    opt->n = 0;
    opt->nblocks = 0;
    opt->step = 0;
}

int jt_optim8_quantize(const float *restrict src, size_t n,
                       int8_t *restrict dst, float *restrict scales,
                       size_t nblocks) {
    if (src == NULL || dst == NULL || scales == NULL || n == 0) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    if (nblocks != jt_optim8_nblocks(n)) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    for (size_t b = 0; b < nblocks; b++) {
        size_t beg = b * (size_t)JT_OPTIM8_BLOCK;
        size_t end = beg + (size_t)JT_OPTIM8_BLOCK;
        if (end > n) {
            end = n;
        }
        float amax = 0.0f;
        for (size_t i = beg; i < end; i++) {
            float v = src[i];
            if (!isfinite(v)) {
                errno = EINVAL;
                return JT_ERR_INVAL;
            }
            float a = fabsf(v);
            amax = (a > amax) ? a : amax;
        }
        float s;
        if (amax == 0.0f) {
            s = 1.0f;
        } else {
            s = amax / 127.0f;
        }
        scales[b] = s;
        float inv = 1.0f / s;
        for (size_t i = beg; i < end; i++) {
            float q = roundf(src[i] * inv);
            q = (q < -127.0f) ? -127.0f : q;
            q = (q > 127.0f) ? 127.0f : q;
            dst[i] = (int8_t)q;
        }
    }
    return JT_OK;
}

int jt_optim8_dequantize(const int8_t *restrict src,
                         const float *restrict scales, size_t nblocks,
                         size_t n, float *restrict dst) {
    if (src == NULL || scales == NULL || dst == NULL || n == 0) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    if (nblocks != jt_optim8_nblocks(n)) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    for (size_t b = 0; b < nblocks; b++) {
        float s = scales[b];
        if (!isfinite(s) || !(s > 0.0f)) {
            errno = EINVAL;
            return JT_ERR_INVAL;
        }
    }
    for (size_t i = 0; i < n; i++) {
        dst[i] = (float)src[i] * scales[i / (size_t)JT_OPTIM8_BLOCK];
    }
    return JT_OK;
}

int jt_optim8_step(jt_optim8_t *restrict opt, float *restrict param,
                   const float *restrict grad, size_t n) {
    if (opt == NULL || param == NULL || grad == NULL || n == 0) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    if (n != opt->n || opt->q_m == NULL || opt->q_v == NULL ||
        opt->s_m == NULL || opt->s_v == NULL) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    // grad有限性はブロック処理前に全走査 (fail-closed: 非有限gradで状態を汚さない)。
    for (size_t i = 0; i < n; i++) {
        if (!isfinite(grad[i]) || !isfinite(param[i])) {
            errno = EINVAL;
            return JT_ERR_INVAL;
        }
    }

    opt->step += 1;
    double t = (double)opt->step;
    // bias補正係数 (doubleで計算しfloat化。b^tはt大で0に漸近し係数→1)。
    float bc1 = 1.0f - powf(opt->beta1, (float)t);
    float bc2 = 1.0f - powf(opt->beta2, (float)t);
    if (!(bc1 > 0.0f) || !(bc2 > 0.0f) || !isfinite(bc1) || !isfinite(bc2)) {
        // t=1でb∈(0,1)なら正のはず。ここに来るのは異常系のみ。
        errno = EINVAL;
        opt->step -= 1;
        return JT_ERR_INVAL;
    }
    float b1 = opt->beta1;
    float b2 = opt->beta2;
    float g1 = 1.0f - b1;
    float g2 = 1.0f - b2;
    float lr = opt->lr;
    float eps = opt->eps;
    float wd = opt->wd;

    // ブロック単位処理: m/v復元→更新→再量子化。作業域はスタック64要素。
    for (size_t b = 0; b < opt->nblocks; b++) {
        size_t beg = b * (size_t)JT_OPTIM8_BLOCK;
        size_t end = beg + (size_t)JT_OPTIM8_BLOCK;
        if (end > n) {
            end = n;
        }
        float sm = opt->s_m[b];
        float sv = opt->s_v[b];
        if (!isfinite(sm) || !(sm > 0.0f) || !isfinite(sv) || !(sv > 0.0f)) {
            errno = EINVAL;
            return JT_ERR_INVAL;
        }
        float mbuf[JT_OPTIM8_BLOCK];
        float vbuf[JT_OPTIM8_BLOCK];
        size_t blen = end - beg;
        for (size_t k = 0; k < blen; k++) {
            size_t i = beg + k;
            float m = (float)opt->q_m[i] * sm;
            float v = (float)opt->q_v[i] * sv;
            float g = grad[i];
            m = b1 * m + g1 * g;
            v = b2 * v + g2 * g * g;
            // vは非負を維持 (丸めで微小負が出てもclamp)。
            v = (v < 0.0f) ? 0.0f : v;
            mbuf[k] = m;
            vbuf[k] = v;
            float mh = m / bc1;
            float vh = v / bc2;
            float denom = sqrtf(vh) + eps;
            float upd = mh / denom;
            float p = param[i];
            p = p - lr * (upd + wd * p);
            param[i] = p;
        }
        // 再量子化 (ゼロブロックはscale=1)。
        float mmax = 0.0f;
        float vmax = 0.0f;
        for (size_t k = 0; k < blen; k++) {
            float a = fabsf(mbuf[k]);
            mmax = (a > mmax) ? a : mmax;
            float c = fabsf(vbuf[k]);
            vmax = (c > vmax) ? c : vmax;
        }
        float nsm = (mmax == 0.0f) ? 1.0f : mmax / 127.0f;
        float nsv = (vmax == 0.0f) ? 1.0f : vmax / 127.0f;
        opt->s_m[b] = nsm;
        opt->s_v[b] = nsv;
        float im = 1.0f / nsm;
        float iv = 1.0f / nsv;
        for (size_t k = 0; k < blen; k++) {
            size_t i = beg + k;
            float qm = roundf(mbuf[k] * im);
            qm = (qm < -127.0f) ? -127.0f : qm;
            qm = (qm > 127.0f) ? 127.0f : qm;
            opt->q_m[i] = (int8_t)qm;
            float qv = roundf(vbuf[k] * iv);
            qv = (qv < 0.0f) ? 0.0f : qv;
            qv = (qv > 127.0f) ? 127.0f : qv;
            opt->q_v[i] = (int8_t)qv;
        }
    }
    return JT_OK;
}
