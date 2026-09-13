// test_model1b: 1B級MoE構成の会計・予算検証 (P2 MAJOR-3対応)。
// - 既定構成の固定値検証 (計算式は model1b.h 注記。手計算検算済み)
// - 会計妥当性: 総数0.5B〜2B帯、active_body<100M、dense INT4換算<=2GB
// - 学習ピーク (offload=1, B=8/S=2048/fp32) が16GBに収まることをassert
// - N100実機11GBの可否はコメント判定 (printf) + 削減案表示
// - 不正入力拒否 + 決定性 (2回実行一致)
// 成功時 exit 0 + "model1b: OK"。
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "jimotono/common.h"
#include "jimotono/mixed_prec.h"
#include "jimotono/model1b.h"

static int g_fail = 0;

#define CHECK(cond, ...)                                                  \
    do {                                                                  \
        if (!(cond)) {                                                    \
            fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__);           \
            fprintf(stderr, __VA_ARGS__);                                 \
            fprintf(stderr, "\n");                                        \
            g_fail = 1;                                                   \
        }                                                                 \
    } while (0)

#define GIB ((uint64_t)1024 * 1024 * 1024)

static void test_defaults(void) {
    jt_model1b_config_t cfg;
    memset(&cfg, 0, sizeof(cfg));
    CHECK(jt_model1b_default(&cfg) == JT_OK, "default rc");
    CHECK(cfg.n_layers == 24, "layers=%d", cfg.n_layers);
    CHECK(cfg.d_model == 1024, "d=%d", cfg.d_model);
    CHECK(cfg.n_experts == 64, "E=%d", cfg.n_experts);
    CHECK(cfg.expert_hidden == 128, "h=%d", cfg.expert_hidden);
    CHECK(cfg.n_shared == 2, "S=%d", cfg.n_shared);
    CHECK(cfg.top_k == 4, "K=%d", cfg.top_k);
    CHECK(cfg.vocab == 48588, "vocab=%d", cfg.vocab);
    CHECK(cfg.linear_every == 6, "every=%d", cfg.linear_every);
    CHECK(jt_model1b_validate(&cfg) == JT_OK, "validate rc");
    CHECK(jt_model1b_default(NULL) == JT_ERR_INVAL, "default NULL");
    CHECK(jt_model1b_validate(NULL) == JT_ERR_INVAL, "validate NULL");
}

static void test_counts(void) {
    jt_model1b_config_t cfg;
    jt_model1b_counts_t c;
    jt_model1b_default(&cfg);
    memset(&c, 0, sizeof(c));
    CHECK(jt_model1b_counts(&cfg, &c) == JT_OK, "counts rc errno=%d", errno);
    // 手計算固定値 (model1b.h検算):
    // expert=3*1024*128=393,216 / routing=24*64*expert=603,979,776 /
    // shared=24*2*expert=18,874,368 /
    // attn=20*(1024^2+1024)+4*(4*1024^2+1024)=37,773,312 /
    // emb=48588*1024=49,754,112 / router=24*1024*64=1,572,864 /
    // norms=24*2*1024=49,152 / total=712,003,584 /
    // active=attn+shared+24*4*expert+router+norms=96,018,432.
    CHECK(c.expert_params == 393216u, "expert=%llu", (unsigned long long)c.expert_params);
    CHECK(c.routing_total == 603979776u, "routing=%llu",
          (unsigned long long)c.routing_total);
    CHECK(c.shared_total == 18874368u, "shared=%llu",
          (unsigned long long)c.shared_total);
    CHECK(c.attn_total == 37773312u, "attn=%llu", (unsigned long long)c.attn_total);
    CHECK(c.emb_total == 49754112u, "emb=%llu", (unsigned long long)c.emb_total);
    CHECK(c.router_total == 1572864u, "router=%llu",
          (unsigned long long)c.router_total);
    CHECK(c.norm_total == 49152u, "norms=%llu", (unsigned long long)c.norm_total);
    CHECK(c.total == 712003584u, "total=%llu", (unsigned long long)c.total);
    CHECK(c.active_body == 96018432u, "active=%llu",
          (unsigned long long)c.active_body);
    CHECK(c.active_moe == 56623104u, "active_moe=%llu",
          (unsigned long long)c.active_moe);
    CHECK(c.n_full == 4 && c.n_lin == 20, "split full=%d lin=%d", c.n_full, c.n_lin);
    // 妥当性バンド (タスク要件4): 総数0.5B〜2B、active<100M。
    CHECK(c.total >= 500000000u && c.total <= 2000000000u, "total band=%llu",
          (unsigned long long)c.total);
    CHECK(c.active_body < 100000000u, "active<100M got=%llu",
          (unsigned long long)c.active_body);
    printf("counts: total=%.3fB active=%.3fM (expert=%.3fM full=%d lin=%d)\n",
           (double)c.total / 1e9, (double)c.active_body / 1e6,
           (double)c.expert_params / 1e6, c.n_full, c.n_lin);
    // 不正系。
    CHECK(jt_model1b_counts(NULL, &c) == JT_ERR_INVAL, "counts NULL cfg");
    CHECK(jt_model1b_counts(&cfg, NULL) == JT_ERR_INVAL, "counts NULL out");
    {
        jt_model1b_config_t bad = cfg;
        bad.top_k = 65; // >E
        CHECK(jt_model1b_counts(&bad, &c) == JT_ERR_INVAL, "top_k>E");
        bad = cfg;
        bad.n_layers = 0;
        CHECK(jt_model1b_counts(&bad, &c) == JT_ERR_INVAL, "layers=0");
        bad = cfg;
        bad.linear_every = 0;
        CHECK(jt_model1b_counts(&bad, &c) == JT_ERR_INVAL, "every=0");
    }
    // 決定性。
    {
        jt_model1b_counts_t c2;
        memset(&c2, 0, sizeof(c2));
        CHECK(jt_model1b_counts(&cfg, &c2) == JT_OK, "counts2 rc");
        CHECK(memcmp(&c, &c2, sizeof(c)) == 0, "counts deterministic");
    }
}

static void test_bytes(void) {
    jt_model1b_config_t cfg;
    jt_model1b_bytes_t b;
    int bits = 0;
    jt_model1b_default(&cfg);
    memset(&b, 0, sizeof(b));
    // mixed_precポリシー参照の確認 (AGENTS.MD §2.2)。
    CHECK(jt_mp_bits_for(JT_MP_SHARED, &bits) == JT_OK && bits == 8, "shared=8");
    CHECK(jt_mp_bits_for(JT_MP_ROUTING_DOWN, &bits) == JT_OK && bits == 4, "down=4");
    CHECK(jt_mp_bits_for(JT_MP_ROUTING_GATE, &bits) == JT_OK && bits == 2, "gate=2");
    CHECK(jt_mp_bits_for(JT_MP_ROUTING_UP, &bits) == JT_OK && bits == 2, "up=2");
    CHECK(jt_mp_bits_for(JT_MP_ATTN, &bits) == JT_OK && bits == 8, "attn=8");
    CHECK(jt_mp_bits_for(JT_MP_EMB, &bits) == JT_OK && bits == 8, "emb=8");
    CHECK(jt_model1b_bytes(&cfg, &b) == JT_OK, "bytes rc errno=%d", errno);
    // 手計算固定値:
    // dense対象=shared+attn+emb=106,401,792 params。
    // INT8重み=同値B + fp(router+norms)=(1,572,864+49,152)*4=6,488,064 →
    //   dense_weight=112,889,856。scale=ceil(106401792/32)*4=13,300,224。
    //   dense_total_int8=126,190,080。
    // INT4重み=53,200,896+fp → 59,688,960。scale同値。
    //   dense_total_int4=72,989,184。
    // SSD: gate/up各201,326,592params@2bit=50,331,648B、
    //   down同数@4bit=100,663,296B → weight=201,326,592。
    //   scale=3*ceil(201326592/32)*4=75,497,472。ssd_total=276,824,064。
    CHECK(b.dense_weight_int8 == 112889856u, "dw8=%llu",
          (unsigned long long)b.dense_weight_int8);
    CHECK(b.dense_scale_int8 == 13300224u, "ds8=%llu",
          (unsigned long long)b.dense_scale_int8);
    CHECK(b.dense_total_int8 == 126190080u, "d8=%llu",
          (unsigned long long)b.dense_total_int8);
    CHECK(b.dense_weight_int4 == 59688960u, "dw4=%llu",
          (unsigned long long)b.dense_weight_int4);
    CHECK(b.dense_scale_int4 == 13300224u, "ds4=%llu",
          (unsigned long long)b.dense_scale_int4);
    CHECK(b.dense_total_int4 == 72989184u, "d4=%llu",
          (unsigned long long)b.dense_total_int4);
    CHECK(b.ssd_weight == 201326592u, "ssdw=%llu", (unsigned long long)b.ssd_weight);
    CHECK(b.ssd_scale == 75497472u, "ssds=%llu", (unsigned long long)b.ssd_scale);
    CHECK(b.ssd_total == 276824064u, "ssd=%llu", (unsigned long long)b.ssd_total);
    // スケール則 (タスク要件4): dense INT4換算で2GB以下。
    CHECK(b.dense_total_int4 <= 2u * 1024u * 1024u * 1024u, "dense int4<=2GB");
    CHECK(b.dense_total_int4 < b.dense_total_int8, "int4<int8");
    printf("bytes: dense_int8=%.1fMiB dense_int4=%.1fMiB ssd=%.1fMiB\n",
           (double)b.dense_total_int8 / 1048576.0,
           (double)b.dense_total_int4 / 1048576.0, (double)b.ssd_total / 1048576.0);
    CHECK(jt_model1b_bytes(NULL, &b) == JT_ERR_INVAL, "bytes NULL cfg");
    CHECK(jt_model1b_bytes(&cfg, NULL) == JT_ERR_INVAL, "bytes NULL out");
}

static void test_train_mem(void) {
    jt_model1b_config_t cfg;
    jt_model1b_train_cfg_t t;
    jt_model1b_train_mem_t m, m_nooff;
    const uint64_t cap16 = 16u * GIB;
    const uint64_t cap11 = 11u * GIB;
    jt_model1b_default(&cfg);
    jt_model1b_train_default(&t);
    CHECK(t.batch == 8 && t.seq == 2048 && t.elem_bytes == 4, "train default B/S");
    CHECK(t.offload_routing == 1 && t.n_seg == 0, "train default offload/auto");
    memset(&m, 0, sizeof(m));
    CHECK(jt_model1b_train_mem(&cfg, &t, &m) == JT_OK, "trainmem rc errno=%d", errno);
    CHECK(m.n_seg_used == 5, "seg=%d", m.n_seg_used); // ceil(sqrt(24))
    CHECK(m.resident_params == 108023808u, "resident=%llu",
          (unsigned long long)m.resident_params);
    CHECK(m.weights_ram == 432095232u, "wram=%llu", (unsigned long long)m.weights_ram);
    CHECK(m.grads_ram == 532758528u, "gram=%llu", (unsigned long long)m.grads_ram);
    CHECK(m.optim_ram == 229550592u, "oram=%llu", (unsigned long long)m.optim_ram);
    CHECK(m.act_full == 1610612736u, "full=%llu", (unsigned long long)m.act_full);
    CHECK(m.act_stored == 402653184u, "stored=%llu", (unsigned long long)m.act_stored);
    CHECK(m.act_peak == 738197504u, "peak=%llu", (unsigned long long)m.act_peak);
    CHECK(m.prefetch_buf == 12582912u, "pref=%llu", (unsigned long long)m.prefetch_buf);
    CHECK(m.peak_total == 2079402496u, "peak_total=%llu",
          (unsigned long long)m.peak_total);
    // 要件3: 16GBに収まることをassert。
    CHECK(m.peak_total <= cap16, "peak %.3fGiB <= 16GiB",
          (double)m.peak_total / (double)GIB);
    printf("train(offload): peak=%.3fGiB (w=%.2f g=%.2f o=%.2f act=%.2f) <=16GiB OK\n",
           (double)m.peak_total / (double)GIB, (double)m.weights_ram / (double)GIB,
           (double)m.grads_ram / (double)GIB, (double)m.optim_ram / (double)GIB,
           (double)m.act_peak / (double)GIB);
    // N100実機11GBのコメント判定 (assert対象外。削減案つき)。
    memset(&m_nooff, 0, sizeof(m_nooff));
    {
        jt_model1b_train_cfg_t t0 = t;
        t0.offload_routing = 0;
        CHECK(jt_model1b_train_mem(&cfg, &t0, &m_nooff) == JT_OK, "nooff rc");
    }
    printf("train(no-offload): peak=%.3fGiB\n", (double)m_nooff.peak_total / (double)GIB);
    if (m.peak_total <= cap11) {
        printf("N100(11GB)判定: 可 (offload時%.2fGiB)。余裕%.2fGiBをdataloader並列・OS分に充当。\n",
               (double)m.peak_total / (double)GIB,
               (double)(cap11 - m.peak_total) / (double)GIB);
    } else {
        printf("N100(11GB)判定: 不可。削減順: (1)offload率↑(expert master全量SSD) "
               "(2)B×S縮小(8×2048→4×2048で活性半減) (3)層24→20・E64→48。\n");
    }
    if (m_nooff.peak_total <= cap11) {
        printf("N100(11GB)判定(offloadなし): 可 (%.2fGiB) だがマージン薄のためoffload推奨。\n",
               (double)m_nooff.peak_total / (double)GIB);
    } else {
        printf("N100(11GB)判定(offloadなし): 不可 (%.2fGiB)。offload必須。\n",
               (double)m_nooff.peak_total / (double)GIB);
    }
    // 不正系。
    CHECK(jt_model1b_train_mem(NULL, &t, &m) == JT_ERR_INVAL, "NULL cfg");
    CHECK(jt_model1b_train_mem(&cfg, NULL, &m) == JT_ERR_INVAL, "NULL tcfg");
    CHECK(jt_model1b_train_mem(&cfg, &t, NULL) == JT_ERR_INVAL, "NULL out");
    {
        jt_model1b_train_cfg_t bad = t;
        bad.batch = 0;
        CHECK(jt_model1b_train_mem(&cfg, &bad, &m) == JT_ERR_INVAL, "batch=0");
        bad = t;
        bad.offload_routing = 2;
        CHECK(jt_model1b_train_mem(&cfg, &bad, &m) == JT_ERR_INVAL, "offload=2");
        bad = t;
        bad.n_seg = 99; // >L
        CHECK(jt_model1b_train_mem(&cfg, &bad, &m) == JT_ERR_INVAL, "seg>L");
    }
    jt_model1b_train_default(NULL); // NULL安全 (no-op)
}

int main(void) {
    test_defaults();
    test_counts();
    test_bytes();
    test_train_mem();
    if (g_fail != 0) {
        fprintf(stderr, "model1b: FAIL\n");
        return 1;
    }
    printf("model1b: OK (counts/bytes/train-mem)\n");
    return 0;
}
