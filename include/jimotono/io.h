#pragma once
#ifndef JIMOTONO_IO_H
#define JIMOTONO_IO_H
// jt_io: 共通I/O薄層 (submit/poll/wait) + OS別backend最小実装。
// analysis/p3-io-abstraction.md §1–§2 の実装。既存3経路は変更しない:
//
//   HOT  (buffered同期): jt_io_pread_batch (変更なし)
//   COLD-sync (Linux実経路/他OSスタブ): jt_io_direct_{open,alloc,pread,close} (変更なし)
//   COLD-async (Linux+liburing実経路/他NOSUP): jt_io_pread_batch_uring (変更なし)
//
// 本層は薄いディスパッチ＋状態保持に限定する。I/O意味論
// (short-readリトライ、EOF→EIO、len==0 no-op、n==0 no-op、未ソートINVAL等)は
// 既存のまま継承し、再定義しない。jt_io_runs_compressは本層の前段のまま
// (ソートは呼び出し側責任。内部ソート禁止)。
//
// Backend選択規則 (io_batch.h HOT/COLD方針の昇格):
//   HOT : 常にbuffered同期 (jt_io_pread_batch)。O_DIRECT禁止。
//         submit内で同期実行し、即時完了tokenとして返す。
//   COLD: OS能力で選択。
//     Linux+HAVE_LIBURING : jt_io_pread_batch_uring (buffered fd前提)。
//                           direct fd + uringの結合は意図的に行わない
//                           (分離維持。design §1.4)。O_DIRECT fdを使う場合は
//                           jt_io_direct_* を直接呼ぶこと。
//     Linux (liburing無効) : jt_io_direct_preadのper-reqループ
//                           (整列済みのためdirect/buffered両fdで動作。
//                           O_DIRECT足場への接続。JIMOTONO_USE_URING既定OFF維持)。
//     macOS/Windows       : jt_io_pread_batch同期fallback。新層はNOSUPを返さず
//                           同期実行して完了扱いにする (呼び出し側分岐を増やさない)。
//
// スレッド安全性: 同一fdへの並行submitは呼び出し側で直列化すること
// (io_batch_uring.c注記を継承)。本層は新規スレッドを作らない (1C1T前提)。
// token表は固定64枠のround-robin (古い完了は上書きで失われる。inflight上限256
// に対し十分小さい窓での使用を想定。unknown tokenはINVAL)。
//
// macOS F_NOCACHEの挙動差異 (RULE転記):
//   fcntl(fd, F_NOCACHE, 1)はLinux O_DIRECTほど強力でない。fd単位の
//   「キャッシュしない助言」に留まり、(i)他fd経由の同一ファイルキャッシュ残存を
//   防げない、(ii)メタデータ/先読みの影響が残る、(iii)整列要件がなく
//   fail-closed検査の対象にならない。したがってmacOSでは「COLD＝キャッシュ汚染
//   ゼロ」を保証できず、汚染低減に留まる。macOS数値はLinux数値と並べないこと。
//
// 規約: C11、restrict積極使用、errnoベース + goto cleanup (呼び出し側含む)。
// クロスプラットフォーム: 同一ソースがLinux/macOS/Windowsでビルド可能。

#include <stddef.h>
#include <stdint.h>

#include "jimotono/common.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum jt_io_kind {
    JT_IO_HOT = 0,   // buffered同期 (O_DIRECT禁止)
    JT_IO_COLD = 1,  // OS能力で選択 (整列必須・fail-closed)
} jt_io_kind_t;

typedef struct jt_io_req {
    uint64_t offset;  // jt_io_spec_tと同義 (ファイル先頭からのバイト位置)
    size_t length;    // 0はno-opとしてskip (dstはNULL可)
    void *dst;        // length>0なら非NULL
} jt_io_req_t;

typedef struct jt_io_token {
    uint64_t id;  // poll/wait用 (0は無効)
} jt_io_token_t;

// 非同期投入。戻り値は投入受付のみ表す (完了はpoll/waitで知る)。
//   fd: HOTはbuffered fd、COLDはjt_io_direct_openのfd (buffered fallback fdも可)。
//       本層はopenしない。
//   kind: JT_IO_HOT / JT_IO_COLD (他はINVAL)。
//   reqs[0..n): n==0は何もせずJT_OK (fd/reqs検証なし、reqsはNULL可)。
//   out_tok: 非NULL (完了token受取り)。
//   COLDバッファ要件: jt_io_direct_is_aligned(dst,off,len,512)を投入前に検査し
//       違反はJT_ERR_INVAL+errno=EINVAL (fail-closed、I/O発行なし)。
//   戻り値: JT_OK (投入受付。同期実行の成否はpoll/waitで返す)
//            JT_ERR_INVAL (引数不正・整列違反、errno=EINVAL)
//            JT_ERR_NOMEM (内部spec複写の確保失敗、errno=ENOMEM)
//            JT_ERR_IO (同期実行時のread失敗。errnoはpread由来。tokenは発行され
//                       poll/waitでも同値を返す)
//   NOTE: 現実装はsubmit内で同期実行する (窓進行の非同期化は将来。decode側は
//   submit→計算→waitの窓に入れてSSD待ちを隠すこと)。
int jt_io_submit(int fd, jt_io_kind_t kind,
                 const jt_io_req_t *restrict reqs, size_t n,
                 jt_io_token_t *restrict out_tok);

// 非ブロッキング回収: doneに完了件数 (同期実装のため0またはtotal)。
//   tok: jt_io_submitの戻りtoken。unknownはJT_ERR_INVAL+errno=EINVAL。
//   done: 非NULL。
//   戻り値: tokenに対応する実行結果 (JT_OK/IO)。未完了の概念はないため
//   常に確定値を返す (designの「未完了はJT_OK+*done==0」は将来の非同期化余地)。
int jt_io_poll(jt_io_token_t tok, size_t *restrict done);

// ブロッキング待機: 全件完了まで待つ (同期実装のため即時確定。
// EINTR内部リトライは既存pread/uring層で行う)。
//   戻り値: tokenに対応する実行結果 (JT_OK/IO)。unknownはJT_ERR_INVAL。
int jt_io_wait(jt_io_token_t tok);

// ---- macOS: F_NOCACHE助言 (best-effort) ----
//   fcntl(fd, F_NOCACHE, 1)を適用する。COLD fdにのみ適用しHOT fdには適用しない
//   こと (呼び出し側判断。本関数はfd種別を知らない)。
//   効果は汚染低減に留まる (上記RULE転記参照)。O_DIRECT相当の保証はない。
//   fcntl失敗時はJT_ERR_IOにせずJT_OKとして無視する (bufferedとして動作継続。
//   design §2.2推奨どおり)。fd<0はJT_ERR_INVAL。
//   非macOSではJT_ERR_NOSUP+errno=ENOSYS (適用対象なし)。
int jt_io_macos_set_nocache(int fd);

// ---- macOS: kqueue readiness通知 (最小実装) ----
//   EVFILT_READのreadinessを1回だけ待つ。io_uringのようなsubmission/completion
//   キューにはならない (完了通知ではない)。したがって非同期化の既定は同期pread
//   のまま窓進行を呼び出し側で行う (design §2.2 (a))。本関数はreadiness確認の
//   最小足場であり、I/O自体は発行しない。
//   out_ready: 非NULL (1=readable、0=timeout)。
//   戻り値: JT_OK / JT_ERR_INVAL (fd<0・NULL・timeout<0、errno=EINVAL)
//            JT_ERR_IO (kqueue/kevent失敗。errnoはOS由来)
//            JT_ERR_NOSUP+errno=ENOSYS (非macOS)。
int jt_io_macos_kqueue_ready(int fd, int timeout_ms, int *restrict out_ready);

// ---- Windows: FILE_FLAG_NO_BUFFERING + IOCP (将来用スタブ) ----
//   制約が強く (4096整列バッファ+セクタ倍数長必須、末端tailは切上げパディング読み
//   ＋呼び出し側切詰めが必要)、Phase 1正経路は_lseeki64+_read buffered fallback
//   (jt_io_pread_batch内) のまま。本スタブは将来の実経路のための予約であり、
//   常にJT_ERR_NOSUP+errno=ENOSYSを返す (Windows上でも同様。ビルド阻害しない)。
//   OVERLAPPED+GetQueuedCompletionStatus実経路は別関数に分離する (現行TODOどおり)。
int jt_io_win_direct_open(const char *restrict path, int *restrict out_fd);
int jt_io_win_iocp_submit(int fd, const jt_io_req_t *restrict reqs, size_t n,
                          jt_io_token_t *restrict out_tok);

#ifdef __cplusplus
}
#endif

#endif  // JIMOTONO_IO_H
