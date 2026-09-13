// pack_jsonl: 空白区切り整数列テキスト → .jtdp 変換の最小実装 (実行バイナリ)。
//
// 暫定仕様 (トークナイザ確定前のつなぎ):
//   * 入力は「1行=1シーケンス」の素朴テキスト。各行は空白 (space/tab/CR) 区切り
//     の10進整数列。空行は空シーケンスとして保持 (行↔seq対応を崩さない)。
//   * JSONL本格パーサは禁止 (本ツールは暫定と明記)。JSONの文字列・
//     エスケープ・引用符は一切解釈しない。非数字・非空白文字があれば
//     即エラー終了 (fail-closed)。トークナイザ確定後に本格JSONL取込みへ
//     置換すること (TODO下記)。
//   * 全IDは JT_DP_VOCAB_SIZE (48588) 未満であること。範囲外はエラー。
//   * 出力は jt_data_pack v1 (mmap可)。JSONL直接読みはしない
//     (16ワーカーで破綻するため、本バイナリ経由が正規経路)。
//   * ctest外の実行バイナリ (重負荷ベンチ禁止の帯域規則に抵触しないよう、
//     テスト配線には入れない)。
//
// TODO(tokenizer確定後): llm-jp-tokenizer v2.2確定後に本格JSONLパーサ
// (テキスト抽出→encode→packの一貫パイプライン)へ置換し、本暫定仕様を廃止する。
//
// 使い方: pack_jsonl <in.txt> <out.jtdp>
// 成功時 stdout に "pack_jsonl: nseq=.. total=.." を出し exit 0。
// 失敗時 stderr + exit 1。出力部分ファイルは削除せず残す
// (readerがchecksumで拒否するため安全。必要なら呼び出し側で消す)。

#include <errno.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "jimotono/data_pack.h"

#define JT_PACK_CAP_INIT 256u

static void usage(const char *argv0) {
    fprintf(stderr,
            "usage: %s <in.txt> <out.jtdp>\n"
            "  interim spec (pre-tokenizer): whitespace-separated decimal ids, "
            "one seq per line.\n",
            argv0);
}

int main(int argc, char **argv) {
    if (argc != 3) {
        usage(argv[0]);
        return 1;
    }
    const char *in_path = argv[1];
    const char *out_path = argv[2];

    FILE *fin = fopen(in_path, "r");
    if (fin == NULL) {
        fprintf(stderr, "pack_jsonl: cannot open input '%s': %s\n", in_path,
                strerror(errno));
        return 1;
    }

    jt_dp_writer_t w = {0};
    int rc = jt_dp_writer_open(&w, out_path);
    if (rc != JT_OK) {
        fprintf(stderr, "pack_jsonl: cannot open output '%s': %s\n", out_path,
                strerror(errno));
        fclose(fin);
        return 1;
    }

    // 行バッファ (動的拡張)。fgetc駆動で任意長行・任意移植性 (C11のみ)。
    uint32_t *ids = NULL;
    size_t nids = 0, cap = 0;
    uint64_t nseq = 0, total = 0;
    int fails = 0;
    unsigned long cur = 0;  // 蓄積中の数値
    int in_num = 0;         // 数字蓄積中か
    int has_digit = 0;      // 現行に1つでも数字があったか (空行判定用)
    int c = 0;

    // idsへの1要素追加 (realloc拡張)。
    // ラベル不使用のgoto cleanup方針に合わせ、エラー時はfailsを立てて抜ける。
    for (;;) {
        c = fgetc(fin);
        if (c == ' ' || c == '\t' || c == '\r' || c == '\n' || c == EOF) {
            if (in_num) {
                if (cur >= JT_DP_VOCAB_SIZE) {
                    fprintf(stderr,
                            "pack_jsonl: token id %lu out of range (< %u) at "
                            "seq %llu\n",
                            cur, JT_DP_VOCAB_SIZE,
                            (unsigned long long)nseq);
                    fails = 1;
                    break;
                }
                if (nids == cap) {
                    size_t ncap = (cap == 0) ? JT_PACK_CAP_INIT : cap * 2;
                    if (ncap > SIZE_MAX / sizeof(uint32_t)) {
                        fprintf(stderr, "pack_jsonl: line too long\n");
                        fails = 1;
                        break;
                    }
                    uint32_t *np =
                        (uint32_t *)realloc(ids, ncap * sizeof(uint32_t));
                    if (np == NULL) {
                        fprintf(stderr, "pack_jsonl: out of memory\n");
                        fails = 1;
                        break;
                    }
                    ids = np;
                    cap = ncap;
                }
                ids[nids++] = (uint32_t)cur;
                cur = 0;
                in_num = 0;
            }
            if (c == '\n' || c == EOF) {
                // 行確定 (has_digit==0かつnids==0なら空シーケンス)。
                // ただしEOF直後かつ何も読んでいない場合 (空ファイル末尾) は
                // 余分な空seqを作らない。
                if (c == EOF && !has_digit && nids == 0 && nseq == 0 &&
                    total == 0) {
                    // 空ファイル: seqなしで確定 (下のferror分岐前に抜ける)。
                    break;
                }
                if (c == EOF && !has_digit && nids == 0) {
                    // 末尾改行済みファイルのEOF: 追加seqなし。
                    break;
                }
                rc = jt_dp_writer_add(&w, (nids == 0) ? NULL : ids, nids);
                if (rc != JT_OK) {
                    fprintf(stderr, "pack_jsonl: writer_add failed at seq %llu: %s\n",
                            (unsigned long long)nseq, strerror(errno));
                    fails = 1;
                    break;
                }
                nseq++;
                total += (uint64_t)nids;
                nids = 0;
                has_digit = 0;
                if (c == EOF) {
                    break;
                }
            }
            if (c == EOF) {
                break;
            }
            continue;
        }
        if (c >= '0' && c <= '9') {
            unsigned digit = (unsigned)(c - '0');
            // 10進オーバーフロー検査 (ULONG_MAX近傍で打ち切り)。
            if (cur > (ULONG_MAX - digit) / 10u) {
                fprintf(stderr, "pack_jsonl: integer overflow at seq %llu\n",
                        (unsigned long long)nseq);
                fails = 1;
                break;
            }
            cur = cur * 10u + digit;
            in_num = 1;
            has_digit = 1;
            continue;
        }
        fprintf(stderr, "pack_jsonl: invalid byte 0x%02x at seq %llu "
                        "(interim spec: whitespace-separated decimals only)\n",
                c, (unsigned long long)nseq);
        fails = 1;
        break;
    }

    if (!fails && ferror(fin)) {
        fprintf(stderr, "pack_jsonl: read error on '%s': %s\n", in_path,
                strerror(errno));
        fails = 1;
    }
    fclose(fin);

    if (fails) {
        // writerはcloseせず資源のみ解放…ではなく、API上close必須のため
        // closeする (部分ファイルはchecksum不一致でreaderに拒否される)。
        // エラー確定なので戻り値は捨てる。
        free(ids);
        return 1;
    }
    free(ids);
    ids = NULL;

    rc = jt_dp_writer_close(&w);
    if (rc != JT_OK) {
        fprintf(stderr, "pack_jsonl: writer_close failed: %s\n",
                strerror(errno));
        return 1;
    }
    printf("pack_jsonl: nseq=%llu total=%llu\n", (unsigned long long)nseq,
           (unsigned long long)total);
    return 0;
}
