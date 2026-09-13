// bench_bpe: 等価性・往復・pack連携・軽性能の実行ツール (ctest外)。
//
// 使い方: bench_bpe <vocab.jtvocab> <golden.txt> <corpus.bin> <out.jtdp>
//   golden.txt: scripts/bpe_make_golden.py の出力 (T:<hex>/I:id,... pair)。
//   corpus.bin: 混合テキスト (性能・pack用)。
//   out.jtdp:   corpusを行分割→encode→jt_data_pack書き出し (pack連携確認)。
//
// 性能は逐次1パス + ウォームアップ1回の軽計測 (--steps禁止の帯域規則に準拠)。
// 終了コード: golden全一致で0、不一致があれば1 (perf/packは実行継続)。
// 依存: jimotono_core + jimotono_bench_common (単調クロックのみ使用)。

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "bench_common.h"
#include "jimotono/bpe.h"
#include "jimotono/data_pack.h"

static int hexval(int c) {
    if (c >= '0' && c <= '9') {
        return c - '0';
    }
    if (c >= 'a' && c <= 'f') {
        return c - 'a' + 10;
    }
    if (c >= 'A' && c <= 'F') {
        return c - 'A' + 10;
    }
    return -1;
}

// 1行読み (改行除去)。戻り値は長さ。EOF時 SIZE_MAX。
static size_t read_line(FILE *restrict f, char *restrict buf, size_t cap) {
    size_t n = 0;
    int c = 0;
    for (;;) {
        c = fgetc(f);
        if (c == EOF) {
            if (n == 0) {
                return SIZE_MAX;
            }
            break;
        }
        if (c == '\n') {
            break;
        }
        if (n + 1 < cap) {
            buf[n] = (char)c;
        }
        n++;
    }
    if (n >= cap) {
        return SIZE_MAX - 1;  // 行長超過 (fail-closed)
    }
    buf[n] = '\0';
    return n;
}

static int run_golden(const jt_bpe_t *restrict b, const char *restrict path,
                      int *restrict out_pass, int *restrict out_total) {
    FILE *f = fopen(path, "r");
    static char line[1 << 20];
    static unsigned char text[1 << 20];
    static uint32_t got[1 << 16];
    static uint32_t want[1 << 16];
    static char back[1 << 20];
    int pass = 0, total = 0, shown = 0;
    if (f == NULL) {
        fprintf(stderr, "bench_bpe: cannot open golden '%s': %s\n", path,
                strerror(errno));
        return 1;
    }
    for (;;) {
        size_t tl = read_line(f, line, sizeof(line));
        size_t il = 0;
        size_t tn = 0, wn = 0;
        size_t n = 0;
        size_t k = 0;
        int rc = 0;
        if (tl == SIZE_MAX) {
            break;
        }
        if (tl < 2 || line[0] != 'T' || line[1] != ':') {
            fprintf(stderr, "bench_bpe: golden format error (T line)\n");
            fclose(f);
            return 1;
        }
        // hex decode
        if ((tl - 2) % 2 != 0 || (tl - 2) / 2 >= sizeof(text)) {
            fprintf(stderr, "bench_bpe: golden text too long\n");
            fclose(f);
            return 1;
        }
        tn = (tl - 2) / 2;
        for (k = 0; k < tn; k++) {
            int hi = hexval((unsigned char)line[2 + 2 * k]);
            int lo = hexval((unsigned char)line[2 + 2 * k + 1]);
            if (hi < 0 || lo < 0) {
                fprintf(stderr, "bench_bpe: golden hex error\n");
                fclose(f);
                return 1;
            }
            text[k] = (unsigned char)((hi << 4) | lo);
        }
        il = read_line(f, line, sizeof(line));
        if (il == SIZE_MAX || il < 2 || line[0] != 'I' || line[1] != ':') {
            fprintf(stderr, "bench_bpe: golden format error (I line)\n");
            fclose(f);
            return 1;
        }
        // id list parse
        wn = 0;
        if (il > 2) {
            const char *p = line + 2;
            char *end = NULL;
            for (;;) {
                unsigned long v = 0;
                errno = 0;
                v = strtoul(p, &end, 10);
                if (end == p || errno != 0 || v >= JT_BPE_MAX_ID) {
                    fprintf(stderr, "bench_bpe: golden id parse error\n");
                    fclose(f);
                    return 1;
                }
                if (wn >= sizeof(want) / sizeof(want[0])) {
                    fprintf(stderr, "bench_bpe: golden ids too many\n");
                    fclose(f);
                    return 1;
                }
                want[wn++] = (uint32_t)v;
                if (*end == '\0') {
                    break;
                }
                if (*end != ',') {
                    fprintf(stderr, "bench_bpe: golden id sep error\n");
                    fclose(f);
                    return 1;
                }
                p = end + 1;
            }
        }
        total++;
        n = sizeof(got) / sizeof(got[0]);
        {
            size_t gn = 0;
            rc = jt_bpe_encode(b, (const char *)text, tn, got,
                               sizeof(got) / sizeof(got[0]), &gn);
            if (rc != JT_OK) {
                printf("golden[%d]: ENCODE-FAIL rc=%d textlen=%zu\n", total,
                       rc, tn);
                continue;
            }
            n = gn;
        }
        if (n != wn || memcmp(got, want, n * sizeof(uint32_t)) != 0) {
            if (shown < 10) {
                size_t m = 0;
                printf("golden[%d]: MISMATCH textlen=%zu got_n=%zu want_n=%zu\n",
                       total, tn, n, wn);
                printf("  got :");
                for (m = 0; m < n && m < 12; m++) {
                    printf(" %u", got[m]);
                }
                printf("\n  want:");
                for (m = 0; m < wn && m < 12; m++) {
                    printf(" %u", want[m]);
                }
                printf("\n");
                shown++;
            }
            continue;
        }
        // 往復: decode(encode(x)) == x (バイト比較)。
        {
            size_t bn = 0;
            rc = jt_bpe_decode(b, got, n, back, sizeof(back), &bn);
            if (rc != JT_OK || bn != tn || memcmp(back, text, tn) != 0) {
                printf("golden[%d]: ROUNDTRIP-FAIL rc=%d bn=%zu tn=%zu\n",
                       total, rc, bn, tn);
                continue;
            }
        }
        pass++;
    }
    fclose(f);
    *out_pass = pass;
    *out_total = total;
    return 0;
}

static unsigned char *slurp(const char *restrict path, size_t *restrict n) {
    FILE *f = fopen(path, "rb");
    long sz = 0;
    unsigned char *buf = NULL;
    if (f == NULL) {
        return NULL;
    }
    if (fseek(f, 0, SEEK_END) != 0 || (sz = ftell(f)) < 0 ||
        fseek(f, 0, SEEK_SET) != 0) {
        fclose(f);
        return NULL;
    }
    buf = (unsigned char *)malloc((size_t)sz == 0 ? 1 : (size_t)sz);
    if (buf == NULL) {
        fclose(f);
        return NULL;
    }
    if (sz > 0 && fread(buf, 1, (size_t)sz, f) != (size_t)sz) {
        free(buf);
        fclose(f);
        return NULL;
    }
    fclose(f);
    *n = (size_t)sz;
    return buf;
}

int main(int argc, char **argv) {
    const char *vocab_path = NULL;
    const char *golden_path = NULL;
    const char *corpus_path = NULL;
    const char *pack_path = NULL;
    jt_bpe_t b = {0};
    int pass = 0, total = 0;
    int exit_code = 0;
    unsigned char *corpus = NULL;
    size_t clen = 0;
    uint32_t *ids = NULL;
    size_t nids = 0;

    if (argc != 5) {
        fprintf(stderr,
                "usage: %s <vocab.jtvocab> <golden.txt> <corpus.bin> "
                "<out.jtdp>\n",
                argv[0]);
        return 2;
    }
    vocab_path = argv[1];
    golden_path = argv[2];
    corpus_path = argv[3];
    pack_path = argv[4];

    if (jt_bpe_open(&b, vocab_path) != JT_OK) {
        fprintf(stderr, "bench_bpe: open vocab failed: %s\n",
                strerror(errno));
        return 2;
    }
    printf("bench_bpe: vocab_size=%u vocab_max=%u\n",
           jt_bpe_vocab_size(&b), jt_bpe_vocab_max(&b));

    // 1. 等価性 (参照実装 golden)。
    if (run_golden(&b, golden_path, &pass, &total) != 0) {
        jt_bpe_close(&b);
        return 2;
    }
    printf("golden: %d/%d match (encode + roundtrip)\n", pass, total);
    if (pass != total) {
        exit_code = 1;
    }

    // 2+3. 性能 + pack連携。
    corpus = slurp(corpus_path, &clen);
    if (corpus == NULL) {
        fprintf(stderr, "bench_bpe: cannot read corpus '%s': %s\n",
                corpus_path, strerror(errno));
        jt_bpe_close(&b);
        return 2;
    }
    ids = (uint32_t *)malloc((clen + 1) * sizeof(uint32_t));
    if (ids == NULL) {
        fprintf(stderr, "bench_bpe: oom\n");
        free(corpus);
        jt_bpe_close(&b);
        return 2;
    }
    // ウォームアップ (小入力1回。計測外)。
    {
        size_t dummy = 0;
        uint32_t wout[16];
        (void)jt_bpe_encode(&b, (const char *)corpus,
                            clen < 64 ? clen : 64, wout, 16, &dummy);
    }
    {
        uint64_t t0 = 0, t1 = 0;
        double sec = 0.0;
        int rc = 0;
        if (jt_bench_now_ns(&t0) != JT_OK) {
            fprintf(stderr, "bench_bpe: clock failed\n");
            free(ids);
            free(corpus);
            jt_bpe_close(&b);
            return 2;
        }
        rc = jt_bpe_encode(&b, (const char *)corpus, clen, ids, clen + 1,
                           &nids);
        if (jt_bench_now_ns(&t1) != JT_OK) {
            fprintf(stderr, "bench_bpe: clock failed\n");
            free(ids);
            free(corpus);
            jt_bpe_close(&b);
            return 2;
        }
        if (rc != JT_OK) {
            fprintf(stderr, "bench_bpe: corpus encode failed rc=%d: %s\n",
                    rc, strerror(errno));
            free(ids);
            free(corpus);
            jt_bpe_close(&b);
            return 1;
        }
        sec = (double)(t1 - t0) / 1e9;
        printf("perf: bytes=%zu tokens=%zu sec=%.4f MB/s=%.3f tok/s=%.0f "
               "(sequential 1-pass)\n",
               clen, nids, sec,
               sec > 0.0 ? (double)clen / sec / 1e6 : 0.0,
               sec > 0.0 ? (double)nids / sec : 0.0);
    }
    // pack連携: corpusを行分割→各行encode→.jtdp→reopen検証。
    {
        uint32_t *lids = NULL;
        size_t lcap = 0;
        size_t total_tok = 0;
        jt_dp_writer_t w;
        jt_dp_reader_t r;
        uint64_t rn = 0, rt = 0;
        size_t pos = 0;
        int rc = 0;
        memset(&w, 0, sizeof(w));
        memset(&r, 0, sizeof(r));
        // 空行も空seqとして保持。末尾改行直後の空は作らない。
        {
            size_t mx = 0;
            size_t s = 0;
            for (pos = 0; pos <= clen; pos++) {
                if (pos == clen || corpus[pos] == '\n') {
                    if (pos - s > mx) {
                        mx = pos - s;
                    }
                    s = pos + 1;
                }
            }
            lcap = mx + 1;
        }
        lids = (uint32_t *)malloc(lcap * sizeof(uint32_t));
        if (lids == NULL) {
            fprintf(stderr, "bench_bpe: oom\n");
            free(ids);
            free(corpus);
            jt_bpe_close(&b);
            return 2;
        }
        rc = jt_dp_writer_open(&w, pack_path);
        if (rc != JT_OK) {
            fprintf(stderr, "bench_bpe: pack open failed: %s\n",
                    strerror(errno));
            free(lids);
            free(ids);
            free(corpus);
            jt_bpe_close(&b);
            return 1;
        }
        {
            size_t s = 0;
            pos = 0;
            for (pos = 0; pos <= clen; pos++) {
                if (pos == clen || corpus[pos] == '\n') {
                    size_t ln = 0;
                    if (pos > s || pos == clen) {
                        // 空行も空seqとして保持。末尾改行直後の空は作らない。
                        if (pos == clen && pos == s && clen > 0 &&
                            corpus[clen - 1] == '\n') {
                            break;
                        }
                        rc = jt_bpe_encode(&b, (const char *)corpus + s,
                                           pos - s, lids, lcap, &ln);
                        if (rc != JT_OK) {
                            fprintf(stderr,
                                    "bench_bpe: line encode failed rc=%d\n",
                                    rc);
                            free(lids);
                            free(ids);
                            free(corpus);
                            jt_bpe_close(&b);
                            return 1;
                        }
                        rc = jt_dp_writer_add(&w, ln == 0 ? NULL : lids,
                                              ln);
                        if (rc != JT_OK) {
                            fprintf(stderr, "bench_bpe: pack add failed\n");
                            free(lids);
                            free(ids);
                            free(corpus);
                            jt_bpe_close(&b);
                            return 1;
                        }
                        total_tok += ln;
                    }
                    s = pos + 1;
                }
            }
        }
        free(lids);
        rc = jt_dp_writer_close(&w);
        if (rc != JT_OK) {
            fprintf(stderr, "bench_bpe: pack close failed: %s\n",
                    strerror(errno));
            free(ids);
            free(corpus);
            jt_bpe_close(&b);
            return 1;
        }
        rc = jt_dp_open(pack_path, &r);
        if (rc != JT_OK) {
            fprintf(stderr, "bench_bpe: pack reopen failed: %s\n",
                    strerror(errno));
            free(ids);
            free(corpus);
            jt_bpe_close(&b);
            return 1;
        }
        if (jt_dp_nseq(&r, &rn) != JT_OK) {
            fprintf(stderr, "bench_bpe: pack nseq failed\n");
            jt_dp_close(&r);
            free(ids);
            free(corpus);
            jt_bpe_close(&b);
            return 1;
        }
        for (pos = 0; pos < rn; pos++) {
            uint64_t L = 0;
            if (jt_dp_seq_len(&r, pos, &L) != JT_OK) {
                fprintf(stderr, "bench_bpe: pack seq_len failed\n");
                jt_dp_close(&r);
                free(ids);
                free(corpus);
                jt_bpe_close(&b);
                return 1;
            }
            rt += L;
        }
        printf("pack: nseq=%llu total=%llu (expect total=%zu) reopen=OK\n",
               (unsigned long long)rn, (unsigned long long)rt, total_tok);
        if (rt != total_tok) {
            fprintf(stderr, "bench_bpe: pack total mismatch\n");
            jt_dp_close(&r);
            free(ids);
            free(corpus);
            jt_bpe_close(&b);
            return 1;
        }
        // 先頭seqの一致確認。
        {
            const uint32_t *ptr = NULL;
            uint64_t L = 0;
            size_t eol = 0;
            size_t ln = 0;
            uint32_t *exp = NULL;
            while (eol < clen && corpus[eol] != '\n') {
                eol++;
            }
            exp = (uint32_t *)malloc((eol + 1) * sizeof(uint32_t));
            if (exp == NULL ||
                jt_bpe_encode(&b, (const char *)corpus, eol, exp, eol + 1,
                              &ln) != JT_OK ||
                jt_dp_seq_ptr(&r, 0, &ptr, &L) != JT_OK || L != ln ||
                (ln > 0 && memcmp(ptr, exp, ln * 4) != 0)) {
                fprintf(stderr, "bench_bpe: pack seq0 mismatch\n");
                free(exp);
                jt_dp_close(&r);
                free(ids);
                free(corpus);
                jt_bpe_close(&b);
                return 1;
            }
            free(exp);
        }
        jt_dp_close(&r);
    }

    free(ids);
    free(corpus);
    jt_bpe_close(&b);
    if (exit_code == 0) {
        printf("bench_bpe: OK\n");
    }
    return exit_code;
}
