#pragma once
#ifndef JIMOTONO_IO_DIRECT_H
#define JIMOTONO_IO_DIRECT_H
// jt_io_direct: O_DIRECT抽象化足場 (P2 GAP-3対応, DESIGN.MD §3)。
//
// ---- 使い分け方針 (DESIGN.MD §3.1, Colibri式) ----
//   HOT  (共有expert / attention射影 / 頻出routed expert):
//        LRU + pin + buffered I/O (ページキャッシュを使う)。O_DIRECT禁止。
//        本ヘッダのAPIを使わないこと (jt_io_pread_batchを使う)。
//   COLD (delta incoming行 / 低頻度routed expert):
//        O_DIRECT streaming (ページキャッシュをバイパス)。本APIの対象。
//        82〜85%持続の残り15〜18%のみを読む短い順次runの束を想定。
//
// ---- OS抽象化 (AGENTS.MD: io_uring/kqueue/IOCP。現段階は足場まで) ----
//   Linux:   open(path, O_RDONLY|O_DIRECT) + posix_memalign(512/4K) + pread。
//            O_DIRECT非対応FS (tmpfs等) では EINVAL/EOPNOTSUPP時に
//            buffered open (O_RDONLY) へfallbackする。
//   macOS:   O_DIRECTなし (F_NOCACHEが代替だが本足場では扱わない)。
//            全APIは JT_ERR_NOSUP + errno=ENOSYS のスタブ。
//   Windows: FILE_FLAG_NO_BUFFERINGが代替だが4096整列+セクタ倍数長が必須で
//            制約が強い。本足場ではスタブ (JT_ERR_NOSUP+ENOSYS)。
//
// ---- アライメント (fail-closed) ----
//   O_DIRECTのカーネル要件 (セクタ整列) を事前検査し、違反は
//   JT_ERR_INVAL + errno=EINVAL で拒否する (I/Oを発行しない)。
//   検査対象: dstアドレス・offset・length (512バイト倍数)。
//   推奨は4096 (ページ整列。将来のWindows NO_BUFFERINGと互換)。
//   allocは512/4096 (一般には512以上の2の冪) のみ受付け、
//   posix_memalignで確保する。
//
// ---- io_uringとの関係 ----
//   非同期化 (io_uring sliding-window) は jt_io_pread_batch_uring の仕事。
//   本APIは同期pread足場。将来のNeuroPrefetcher経路で組み合わせる。
//
// 規約: C11、restrict積極使用、errnoベース + goto cleanup (呼び出し側含む)。
// クロスプラットフォーム: 同一ソースがLinux/macOS/Windowsでビルド可能。
//   O_DIRECT実経路は #ifdef __linux__ でガードし、非Linuxはスタブ。

#include <stddef.h>
#include <stdint.h>

#include "jimotono/common.h"

// セクタ整列 (O_DIRECTの最小公倍数)。検査基準。
#define JT_IO_DIRECT_ALIGN_SECTOR 512
// ページ整列 (推奨。4096整列バッファはセクタ整列も満たす)。
#define JT_IO_DIRECT_ALIGN_PAGE 4096

// アライメント判定 (ポータブル、syscallなし)。
//   alignは512以上の2の冪であること (それ以外は0を返す)。
//   戻り値: 1=整列 (ptr/offset/lengthともalign倍数)、0=不整列または引数不正。
//   len==0はno-opとして1を返す (dstはNULL可)。
static inline int jt_io_direct_is_aligned(const void *ptr, uint64_t offset,
                                          size_t length, size_t align) {
    if (align < (size_t)JT_IO_DIRECT_ALIGN_SECTOR || (align & (align - 1)) != 0) {
        return 0;
    }
    if (length == 0) {
        return 1;
    }
    if (ptr == NULL) {
        return 0;
    }
    uintptr_t a = (uintptr_t)ptr;
    if ((a & (uintptr_t)(align - 1)) != 0) {
        return 0;
    }
    if ((offset & (uint64_t)(align - 1)) != 0) {
        return 0;
    }
    if ((length & (size_t)(align - 1)) != 0) {
        return 0;
    }
    return 1;
}

// O_DIRECTでopenする (Linux実経路)。非対応FSではbufferedにfallback。
//   path: 読み対象ファイル (>0文字)。out_fd: fd受取り先 (非NULL)。
//   戻り値: JT_OK (成功。*out_fd>=0。closeはjt_io_direct_close)
//            JT_ERR_INVAL (path/out_fd不正、errno=EINVAL)
//            JT_ERR_IO (open失敗。errnoはopen由来。ENOENT等)
//            JT_ERR_NOSUP + errno=ENOSYS (非Linuxスタブ)
//   NOTE: fallback時もfdは通常pread可能。呼び出し側は区別不要。
//   NOTE: 書き込みは対象外 (O_RDONLY)。SmartUpdateは別APIの仕事。
int jt_io_direct_open(const char *restrict path, int *restrict out_fd);

// O_DIRECT用アライン付きバッファ確保 (Linux実経路はposix_memalign)。
//   out: 受取り先 (非NULL、*outに設定)。size: バイト数 (>0)。
//   align: 512以上の2の冪 (512/4096を想定)。
//   戻り値: JT_OK / JT_ERR_INVAL (引数不正、errno=EINVAL)
//            JT_ERR_NOMEM (確保失敗、errno=ENOMEM)
//            JT_ERR_NOSUP + errno=ENOSYS (非Linuxスタブ)
//   NOTE: sizeのalign倍数化は呼び出し側責任 (preadがlength倍数を検査)。
//   解放は jt_io_direct_free。
int jt_io_direct_alloc(void **out, size_t size, size_t align);

// 確保バッファ解放 (NULL安全。free()相当)。
//   jt_io_direct_alloc/mallocいずれのポインタにも使える (Linux実経路)。
//   非Linuxでもfree()相当で安全 (戻り値なしのためNOSUPなし)。
void jt_io_direct_free(void *p);

// O_DIRECT fdからの同期pread (Linux実経路。short-readリトライ)。
//   fd: jt_io_direct_openの戻り (buffered fallback fdも可)。
//   dst: 512整列バッファ (len>0なら非NULL)。len: バイト数 (512倍数)。
//   offset: ファイル先頭からのバイト位置 (512倍数)。
//   len==0: 何もせず JT_OK (fd/dstの検証なし、dstはNULL可)。
//   戻り値: JT_OK / JT_ERR_INVAL (引数不正・アライメント違反、errno=EINVAL)
//            JT_ERR_IO (read失敗・EOF前打ち切り。errnoはpread由来、EOF時はEIO)
//            JT_ERR_NOSUP + errno=ENOSYS (非Linuxスタブ、len>0時)
//   NOTE: ファイルオフセットを変更しない (POSIX pread)。
//   NOTE: アライメント違反はfail-closed (I/O発行せずINVAL)。
//   NOTE: EINVAL/ENOSUP時のbuffered fallbackはopen側で行う。本関数は
//   フォールバック再試行しない (同一fdではmode変更不可のため)。
int jt_io_direct_pread(int fd, void *restrict dst, size_t len, uint64_t offset);

// fd close (Linux実経路はclose())。
//   戻り値: JT_OK / JT_ERR_INVAL (fd<0、errno=EINVAL)
//            JT_ERR_IO (close失敗。errnoはclose由来)
//            JT_ERR_NOSUP + errno=ENOSYS (非Linuxスタブ)
int jt_io_direct_close(int fd);

#endif  // JIMOTONO_IO_DIRECT_H
