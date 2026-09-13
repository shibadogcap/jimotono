# Review: P0 build/code (subagent, 2026-09-13)

参照コミット: `cecc8bd` (branch: `develop`)

## 検証項目
- [x] ビルド: `cmake -S . -B build-dbg` configure成功、終了0
- [x] ビルド: `cmake --build build-dbg` 成功、終了0
- [x] `./build-dbg/test_arena` 出力 `arena: OK`、終了0。`ctest` 1/1 Passed
- [x] 警告なし (`-Wall -Wextra -Wpedantic -Wshadow` で warning|error 0件)
- [x] ARM64クロス: `CMAKE_OSX_ARCHITECTURES=arm64` configure成功
- [x] C11準拠: `aligned_alloc` のsize%alignment保証は通常径路で正しい、`_WIN32` 分岐あり
- [ ] エッジケース: `jt_align_up` wraparoundガードなし (重大) → 是正済み (下記)

## 指摘
### 重大
- `src/arena.c` `jt_align_up` にオーバーフローガードなし → `SIZE_MAX` 級入力で検査迂回・ヒープ破壊。是正: 2の冪検査 + `SIZE_MAX-mask` 加算前検査 + 0番兵で呼出側ENOMEM化。
- `aligned_alloc` のsize%alignment保証がwraparound時に崩れる → 同上是正で解消。

### 軽微 (是正済み)
- `cap-off` underflow防御 (`off>cap` ガード追加)、構造体メンバの無意味な `restrict` 除去、include guard併用、`<string.h>` 除去→`<stdint.h>`、`test_arena` に不正入力6件追加、`enable_testing()` トップレベル化、`.gitignore` に `*.dSYM/` 追加。

## 判定
条件付き承認 → 是正済みのため承認相当 (再ビルド+ctest通過確認済み)。
