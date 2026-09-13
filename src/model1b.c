// jt_model1b: 1B級MoE構成の会計・予算純粋関数。
// 計算式は model1b.h の注記を参照。C11, errnoベース, mallocなし。
// ビット幅は jt_mp_bits_for() (mixed_precポリシー) を参照し、
// 活性ピークは jt_ckpt_num_segments/jt_ckpt_mem_estimate() を再利用する。

#include "jimotono/model1b.h"

#include <errno.h>
#include <stddef.h>
#include <stdint.h>

#include "jimotono/checkpoint.h"
#include "jimotono/common.h"
#include "jimotono/mixed_prec.h"

// ---- u64 checked演算 (溢れでENOMEM) ----
static int jt_m1b_mul_u64(uint64_t a, uint64_t b, uint64_t *restrict out) {
    if (out == NULL) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    *out = 0;
    if (a != 0 && b > UINT64_MAX / a) {
        errno = ENOMEM;
        return JT_ERR_NOMEM;
    }
    *out = a * b;
    return JT_OK;
}

static int jt_m1b_add_u64(uint64_t a, uint64_t b, uint64_t *restrict out) {
    if (out == NULL) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    *out = 0;
    if (b > UINT64_MAX - a) {
        errno = ENOMEM;
        return JT_ERR_NOMEM;
    }
    *out = a + b;
    return JT_OK;
}

// params*bits/8。bitsは{2,4,8}のみ (policy由来)。
// 2bit時の端数(params%4≠0)は切上げ ((p*b+7)/8)。
static int jt_m1b_weight_bytes(uint64_t params, int bits, uint64_t *restrict out) {
    uint64_t t = 0;
    if (out == NULL) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    *out = 0;
    if (bits != 2 && bits != 4 && bits != 8) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    if (jt_m1b_mul_u64(params, (uint64_t)bits, &t) != JT_OK) {
        return JT_ERR_NOMEM;
    }
    if (jt_m1b_add_u64(t, 7u, &t) != JT_OK) {
        return JT_ERR_NOMEM;
    }
    *out = t / 8u;
    return JT_OK;
}

// ceil(params/32)*4 (per-block fp32スケール)。
static int jt_m1b_scale_bytes(uint64_t params, uint64_t *restrict out) {
    uint64_t nb = 0, t = 0;
    if (out == NULL) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    *out = 0;
    if (params == 0) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    nb = (params + (uint64_t)(JT_MODEL1B_QBLOCK - 1)) / (uint64_t)JT_MODEL1B_QBLOCK;
    if (jt_m1b_mul_u64(nb, (uint64_t)JT_MODEL1B_SCALE_BYTES, &t) != JT_OK) {
        return JT_ERR_NOMEM;
    }
    *out = t;
    return JT_OK;
}

int jt_model1b_default(jt_model1b_config_t *restrict out) {
    if (out == NULL) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    out->n_layers = JT_MODEL1B_LAYERS;
    out->d_model = JT_MODEL1B_D_MODEL;
    out->n_experts = JT_MODEL1B_EXPERTS;
    out->expert_hidden = JT_MODEL1B_EXPERT_HIDDEN;
    out->n_shared = JT_MODEL1B_SHARED;
    out->top_k = JT_MODEL1B_TOP_K;
    out->vocab = JT_MODEL1B_VOCAB;
    out->linear_every = JT_MODEL1B_LINEAR_EVERY;
    return JT_OK;
}

int jt_model1b_validate(const jt_model1b_config_t *restrict cfg) {
    if (cfg == NULL) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    if (cfg->n_layers <= 0 || cfg->n_layers > JT_CKPT_MAX_LAYERS) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    if (cfg->d_model <= 0 || cfg->n_experts <= 0 || cfg->expert_hidden <= 0) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    if (cfg->n_shared < 0 || cfg->n_shared > cfg->n_experts) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    if (cfg->top_k <= 0 || cfg->top_k > cfg->n_experts) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    if (cfg->vocab <= 0 || cfg->linear_every <= 0) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    return JT_OK;
}

int jt_model1b_counts(const jt_model1b_config_t *restrict cfg,
                      jt_model1b_counts_t *restrict out) {
    uint64_t d = 0, h = 0, L = 0, E = 0, S = 0, K = 0, V = 0;
    uint64_t expert = 0, t = 0, acc = 0;
    uint64_t attn_lin = 0, attn_full = 0, attn = 0;
    uint64_t routing = 0, shared = 0, emb = 0, router = 0, norms = 0;
    uint64_t active_topk = 0, active = 0, active_moe = 0;
    int32_t n_full = 0, n_lin = 0;

    if (cfg == NULL || out == NULL) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    if (jt_model1b_validate(cfg) != JT_OK) {
        return JT_ERR_INVAL;
    }
    d = (uint64_t)cfg->d_model;
    h = (uint64_t)cfg->expert_hidden;
    L = (uint64_t)cfg->n_layers;
    E = (uint64_t)cfg->n_experts;
    S = (uint64_t)cfg->n_shared;
    K = (uint64_t)cfg->top_k;
    V = (uint64_t)cfg->vocab;

    // expert = 3*d*h
    if (jt_m1b_mul_u64(d, h, &t) != JT_OK) {
        return JT_ERR_NOMEM;
    }
    if (jt_m1b_mul_u64(t, 3u, &expert) != JT_OK) {
        return JT_ERR_NOMEM;
    }
    // routing = L*E*expert
    if (jt_m1b_mul_u64(L, E, &t) != JT_OK) {
        return JT_ERR_NOMEM;
    }
    if (jt_m1b_mul_u64(t, expert, &routing) != JT_OK) {
        return JT_ERR_NOMEM;
    }
    // shared = L*S*expert
    if (jt_m1b_mul_u64(L, S, &t) != JT_OK) {
        return JT_ERR_NOMEM;
    }
    if (jt_m1b_mul_u64(t, expert, &shared) != JT_OK) {
        return JT_ERR_NOMEM;
    }
    // attn_lin = d*d + d、attn_full = 4*d*d + d
    if (jt_m1b_mul_u64(d, d, &t) != JT_OK) {
        return JT_ERR_NOMEM;
    }
    if (jt_m1b_add_u64(t, d, &attn_lin) != JT_OK) {
        return JT_ERR_NOMEM;
    }
    if (jt_m1b_mul_u64(t, 4u, &attn_full) != JT_OK) {
        return JT_ERR_NOMEM;
    }
    if (jt_m1b_add_u64(attn_full, d, &attn_full) != JT_OK) {
        return JT_ERR_NOMEM;
    }
    n_full = cfg->n_layers / cfg->linear_every; // 切り捨て (24/6=4)
    n_lin = cfg->n_layers - n_full;
    if (jt_m1b_mul_u64((uint64_t)n_lin, attn_lin, &t) != JT_OK) {
        return JT_ERR_NOMEM;
    }
    if (jt_m1b_mul_u64((uint64_t)n_full, attn_full, &attn) != JT_OK) {
        return JT_ERR_NOMEM;
    }
    if (jt_m1b_add_u64(t, attn, &attn) != JT_OK) {
        return JT_ERR_NOMEM;
    }
    // emb = vocab*d (tied: 1回計上)
    if (jt_m1b_mul_u64(V, d, &emb) != JT_OK) {
        return JT_ERR_NOMEM;
    }
    // router = L*d*E
    if (jt_m1b_mul_u64(L, d, &t) != JT_OK) {
        return JT_ERR_NOMEM;
    }
    if (jt_m1b_mul_u64(t, E, &router) != JT_OK) {
        return JT_ERR_NOMEM;
    }
    // norms = L*2*d
    if (jt_m1b_mul_u64(L, d, &t) != JT_OK) {
        return JT_ERR_NOMEM;
    }
    if (jt_m1b_mul_u64(t, 2u, &norms) != JT_OK) {
        return JT_ERR_NOMEM;
    }
    // total合算
    acc = 0;
    if (jt_m1b_add_u64(acc, routing, &acc) != JT_OK) {
        return JT_ERR_NOMEM;
    }
    if (jt_m1b_add_u64(acc, shared, &acc) != JT_OK) {
        return JT_ERR_NOMEM;
    }
    if (jt_m1b_add_u64(acc, attn, &acc) != JT_OK) {
        return JT_ERR_NOMEM;
    }
    if (jt_m1b_add_u64(acc, emb, &acc) != JT_OK) {
        return JT_ERR_NOMEM;
    }
    if (jt_m1b_add_u64(acc, router, &acc) != JT_OK) {
        return JT_ERR_NOMEM;
    }
    if (jt_m1b_add_u64(acc, norms, &acc) != JT_OK) {
        return JT_ERR_NOMEM;
    }
    // active_topk = L*K*expert、active_body = attn+shared+topk+router+norms
    if (jt_m1b_mul_u64(L, K, &t) != JT_OK) {
        return JT_ERR_NOMEM;
    }
    if (jt_m1b_mul_u64(t, expert, &active_topk) != JT_OK) {
        return JT_ERR_NOMEM;
    }
    active = 0;
    if (jt_m1b_add_u64(active, attn, &active) != JT_OK) {
        return JT_ERR_NOMEM;
    }
    if (jt_m1b_add_u64(active, shared, &active) != JT_OK) {
        return JT_ERR_NOMEM;
    }
    if (jt_m1b_add_u64(active, active_topk, &active) != JT_OK) {
        return JT_ERR_NOMEM;
    }
    if (jt_m1b_add_u64(active, router, &active) != JT_OK) {
        return JT_ERR_NOMEM;
    }
    if (jt_m1b_add_u64(active, norms, &active) != JT_OK) {
        return JT_ERR_NOMEM;
    }
    if (jt_m1b_add_u64(shared, active_topk, &active_moe) != JT_OK) {
        return JT_ERR_NOMEM;
    }

    out->expert_params = expert;
    out->routing_total = routing;
    out->shared_total = shared;
    out->attn_total = attn;
    out->emb_total = emb;
    out->router_total = router;
    out->norm_total = norms;
    out->total = acc;
    out->active_body = active;
    out->active_moe = active_moe;
    out->n_full = n_full;
    out->n_lin = n_lin;
    return JT_OK;
}

int jt_model1b_bytes(const jt_model1b_config_t *restrict cfg,
                     jt_model1b_bytes_t *restrict out) {
    jt_model1b_counts_t c;
    int b_shared = 0, b_down = 0, b_gate = 0, b_up = 0, b_attn = 0, b_emb = 0;
    uint64_t d = 0, h = 0, L = 0, E = 0, dh = 0, nmat = 0;
    uint64_t dense_p = 0; // dense量子化対象param数 (shared+attn+emb)
    uint64_t fp_p = 0, fp_b = 0; // router+norms (fp32常駐)
    uint64_t w_sh = 0, w_at = 0, w_em = 0, w8 = 0, s8 = 0;
    uint64_t w4 = 0, s4 = 0;
    uint64_t gate_p = 0, up_p = 0, down_p = 0;
    uint64_t wg = 0, wu = 0, wd = 0, sg = 0, su = 0, sd = 0;

    if (cfg == NULL || out == NULL) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    if (jt_model1b_counts(cfg, &c) != JT_OK) {
        return (errno == ENOMEM) ? JT_ERR_NOMEM : JT_ERR_INVAL;
    }
    // mixed_precポリシーからビット幅を取得 (AGENTS.MD §2.2)。
    if (jt_mp_bits_for(JT_MP_SHARED, &b_shared) != JT_OK ||
        jt_mp_bits_for(JT_MP_ROUTING_DOWN, &b_down) != JT_OK ||
        jt_mp_bits_for(JT_MP_ROUTING_GATE, &b_gate) != JT_OK ||
        jt_mp_bits_for(JT_MP_ROUTING_UP, &b_up) != JT_OK ||
        jt_mp_bits_for(JT_MP_ATTN, &b_attn) != JT_OK ||
        jt_mp_bits_for(JT_MP_EMB, &b_emb) != JT_OK) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    d = (uint64_t)cfg->d_model;
    h = (uint64_t)cfg->expert_hidden;
    L = (uint64_t)cfg->n_layers;
    E = (uint64_t)cfg->n_experts;
    if (jt_m1b_mul_u64(d, h, &dh) != JT_OK) {
        return JT_ERR_NOMEM;
    }
    if (jt_m1b_mul_u64(L, E, &nmat) != JT_OK) {
        return JT_ERR_NOMEM;
    }
    // dense量子化対象 = shared+attn+emb (ビット幅は層種別に個別適用)。
    dense_p = 0;
    if (jt_m1b_add_u64(dense_p, c.shared_total, &dense_p) != JT_OK) {
        return JT_ERR_NOMEM;
    }
    if (jt_m1b_add_u64(dense_p, c.attn_total, &dense_p) != JT_OK) {
        return JT_ERR_NOMEM;
    }
    if (jt_m1b_add_u64(dense_p, c.emb_total, &dense_p) != JT_OK) {
        return JT_ERR_NOMEM;
    }
    // router+normsはfp32常駐 (小規模のため量子化対象外と明示)。
    fp_p = 0;
    if (jt_m1b_add_u64(fp_p, c.router_total, &fp_p) != JT_OK) {
        return JT_ERR_NOMEM;
    }
    if (jt_m1b_add_u64(fp_p, c.norm_total, &fp_p) != JT_OK) {
        return JT_ERR_NOMEM;
    }
    if (jt_m1b_mul_u64(fp_p, 4u, &fp_b) != JT_OK) {
        return JT_ERR_NOMEM;
    }
    // INT8 dense: 層種別ビット幅で重み積算 + スケール。
    if (jt_m1b_weight_bytes(c.shared_total, b_shared, &w_sh) != JT_OK ||
        jt_m1b_weight_bytes(c.attn_total, b_attn, &w_at) != JT_OK ||
        jt_m1b_weight_bytes(c.emb_total, b_emb, &w_em) != JT_OK) {
        return (errno == ENOMEM) ? JT_ERR_NOMEM : JT_ERR_INVAL;
    }
    w8 = 0;
    if (jt_m1b_add_u64(w8, w_sh, &w8) != JT_OK ||
        jt_m1b_add_u64(w8, w_at, &w8) != JT_OK ||
        jt_m1b_add_u64(w8, w_em, &w8) != JT_OK ||
        jt_m1b_add_u64(w8, fp_b, &w8) != JT_OK) {
        return JT_ERR_NOMEM;
    }
    if (jt_m1b_scale_bytes(dense_p, &s8) != JT_OK) {
        return (errno == ENOMEM) ? JT_ERR_NOMEM : JT_ERR_INVAL;
    }
    // INT4 dense (成功基準2GB判定用): 全dense対象を一律4bit換算。
    if (jt_m1b_weight_bytes(dense_p, 4, &w4) != JT_OK) {
        return (errno == ENOMEM) ? JT_ERR_NOMEM : JT_ERR_INVAL;
    }
    if (jt_m1b_add_u64(w4, fp_b, &w4) != JT_OK) {
        return JT_ERR_NOMEM;
    }
    if (jt_m1b_scale_bytes(dense_p, &s4) != JT_OK) {
        return (errno == ENOMEM) ? JT_ERR_NOMEM : JT_ERR_INVAL;
    }
    // SSD routing: gate/up/down別ビット幅 (L*E*d*hずつ)。
    gate_p = 0;
    if (jt_m1b_mul_u64(nmat, dh, &gate_p) != JT_OK) {
        return JT_ERR_NOMEM;
    }
    up_p = gate_p;
    down_p = gate_p;
    if (jt_m1b_weight_bytes(gate_p, b_gate, &wg) != JT_OK ||
        jt_m1b_weight_bytes(up_p, b_up, &wu) != JT_OK ||
        jt_m1b_weight_bytes(down_p, b_down, &wd) != JT_OK ||
        jt_m1b_scale_bytes(gate_p, &sg) != JT_OK ||
        jt_m1b_scale_bytes(up_p, &su) != JT_OK ||
        jt_m1b_scale_bytes(down_p, &sd) != JT_OK) {
        return (errno == ENOMEM) ? JT_ERR_NOMEM : JT_ERR_INVAL;
    }

    out->dense_weight_int8 = 0;
    if (jt_m1b_add_u64(w_sh, w_at, &out->dense_weight_int8) != JT_OK ||
        jt_m1b_add_u64(out->dense_weight_int8, w_em, &out->dense_weight_int8) != JT_OK ||
        jt_m1b_add_u64(out->dense_weight_int8, fp_b, &out->dense_weight_int8) != JT_OK) {
        return JT_ERR_NOMEM;
    }
    out->dense_scale_int8 = s8;
    out->dense_total_int8 = 0;
    if (jt_m1b_add_u64(w8, s8, &out->dense_total_int8) != JT_OK) {
        return JT_ERR_NOMEM;
    }
    out->dense_weight_int4 = 0;
    if (jt_m1b_weight_bytes(dense_p, 4, &out->dense_weight_int4) != JT_OK) {
        return (errno == ENOMEM) ? JT_ERR_NOMEM : JT_ERR_INVAL;
    }
    if (jt_m1b_add_u64(out->dense_weight_int4, fp_b, &out->dense_weight_int4) != JT_OK) {
        return JT_ERR_NOMEM;
    }
    out->dense_scale_int4 = s4;
    out->dense_total_int4 = 0;
    if (jt_m1b_add_u64(w4, s4, &out->dense_total_int4) != JT_OK) {
        return JT_ERR_NOMEM;
    }
    out->ssd_weight = 0;
    if (jt_m1b_add_u64(wg, wu, &out->ssd_weight) != JT_OK ||
        jt_m1b_add_u64(out->ssd_weight, wd, &out->ssd_weight) != JT_OK) {
        return JT_ERR_NOMEM;
    }
    out->ssd_scale = 0;
    if (jt_m1b_add_u64(sg, su, &out->ssd_scale) != JT_OK ||
        jt_m1b_add_u64(out->ssd_scale, sd, &out->ssd_scale) != JT_OK) {
        return JT_ERR_NOMEM;
    }
    out->ssd_total = 0;
    if (jt_m1b_add_u64(out->ssd_weight, out->ssd_scale, &out->ssd_total) != JT_OK) {
        return JT_ERR_NOMEM;
    }
    return JT_OK;
}

void jt_model1b_train_default(jt_model1b_train_cfg_t *restrict out) {
    if (out == NULL) {
        return;
    }
    out->batch = 8;
    out->seq = 2048;
    out->elem_bytes = 4;
    out->offload_routing = 1;
    out->n_seg = 0; // auto
}

int jt_model1b_train_mem(const jt_model1b_config_t *restrict cfg,
                         const jt_model1b_train_cfg_t *restrict tcfg,
                         jt_model1b_train_mem_t *restrict out) {
    jt_model1b_counts_t c;
    jt_model1b_train_cfg_t t;
    uint64_t resident = 0, wram = 0, gram = 0, oram = 0, tmp = 0;
    uint64_t per_layer = 0, layer_expert_grad = 0, prefetch = 0;
    size_t full = 0, stored = 0, peak = 0;
    int n_seg = 0, rc = 0;

    if (cfg == NULL || tcfg == NULL || out == NULL) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    if (jt_model1b_counts(cfg, &c) != JT_OK) {
        return (errno == ENOMEM) ? JT_ERR_NOMEM : JT_ERR_INVAL;
    }
    t = *tcfg;
    if (t.batch <= 0 || t.seq <= 0 || t.elem_bytes <= 0) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    if (t.offload_routing != 0 && t.offload_routing != 1) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    if (t.n_seg < 0) {
        errno = EINVAL;
        return JT_ERR_INVAL;
    }
    // 常駐param: offload時はtotal-routing (dense+router+norms+emb)。
    if (t.offload_routing) {
        if (c.routing_total > c.total) {
            errno = EINVAL;
            return JT_ERR_INVAL;
        }
        resident = c.total - c.routing_total;
    } else {
        resident = c.total;
    }
    // weights fp32 master (常駐分のみ)。
    if (jt_m1b_mul_u64(resident, 4u, &wram) != JT_OK) {
        return JT_ERR_NOMEM;
    }
    // grads: 常駐分 + (offload時はin-flightの1層分expert gradsを加算)。
    if (jt_m1b_mul_u64(resident, 4u, &gram) != JT_OK) {
        return JT_ERR_NOMEM;
    }
    if (t.offload_routing) {
        // 1層分expert (E*expert) のfp32 gradsがbackward瞬間にRAM滞在。
        if (jt_m1b_mul_u64((uint64_t)cfg->n_experts, c.expert_params,
                           &layer_expert_grad) != JT_OK) {
            return JT_ERR_NOMEM;
        }
        if (jt_m1b_mul_u64(layer_expert_grad, 4u, &tmp) != JT_OK) {
            return JT_ERR_NOMEM;
        }
        if (jt_m1b_add_u64(gram, tmp, &gram) != JT_OK) {
            return JT_ERR_NOMEM;
        }
    }
    // optim8: 常駐分のみ resident*17/8 (offload experts分はSSD/SmartUpdate)。
    if (jt_m1b_mul_u64(resident, (uint64_t)JT_MODEL1B_OPTIM_NUM, &tmp) != JT_OK) {
        return JT_ERR_NOMEM;
    }
    oram = tmp / (uint64_t)JT_MODEL1B_OPTIM_DEN;
    // 活性: per_layer = B*S*d*elem → checkpoint見積りを再利用。
    if (jt_m1b_mul_u64((uint64_t)t.batch, (uint64_t)t.seq, &tmp) != JT_OK) {
        return JT_ERR_NOMEM;
    }
    if (jt_m1b_mul_u64(tmp, (uint64_t)cfg->d_model, &per_layer) != JT_OK) {
        return JT_ERR_NOMEM;
    }
    if (jt_m1b_mul_u64(per_layer, (uint64_t)t.elem_bytes, &per_layer) != JT_OK) {
        return JT_ERR_NOMEM;
    }
    if (t.n_seg == 0) {
        rc = jt_ckpt_num_segments(cfg->n_layers, &n_seg);
        if (rc != JT_OK) {
            return rc; // INVAL継承 (errno維持)
        }
    } else {
        n_seg = t.n_seg;
    }
    rc = jt_ckpt_mem_estimate(cfg->n_layers, n_seg, (size_t)per_layer, &full,
                              &stored, &peak);
    if (rc != JT_OK) {
        return rc; // INVAL/NOMEM継承
    }
    // prefetch: 投機的先読み和集合の上限として 2*K experts分 (fp32)。
    if (jt_m1b_mul_u64((uint64_t)(2 * cfg->top_k), c.expert_params, &tmp) != JT_OK) {
        return JT_ERR_NOMEM;
    }
    if (jt_m1b_mul_u64(tmp, 4u, &prefetch) != JT_OK) {
        return JT_ERR_NOMEM;
    }

    out->resident_params = resident;
    out->weights_ram = wram;
    out->grads_ram = gram;
    out->optim_ram = oram;
    out->act_full = (uint64_t)full;
    out->act_stored = (uint64_t)stored;
    out->act_peak = (uint64_t)peak;
    out->prefetch_buf = prefetch;
    out->misc_ram = JT_MODEL1B_MISC_BYTES;
    out->peak_total = 0;
    tmp = 0;
    if (jt_m1b_add_u64(tmp, wram, &tmp) != JT_OK ||
        jt_m1b_add_u64(tmp, gram, &tmp) != JT_OK ||
        jt_m1b_add_u64(tmp, oram, &tmp) != JT_OK ||
        jt_m1b_add_u64(tmp, (uint64_t)peak, &tmp) != JT_OK ||
        jt_m1b_add_u64(tmp, prefetch, &tmp) != JT_OK ||
        jt_m1b_add_u64(tmp, JT_MODEL1B_MISC_BYTES, &tmp) != JT_OK) {
        return JT_ERR_NOMEM;
    }
    out->peak_total = tmp;
    out->n_seg_used = n_seg;
    return JT_OK;
}
