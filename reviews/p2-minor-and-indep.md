# Review: MINOR-1〜4 + P2独立系 (develop @ f4d7644)

## 検証項目
- [x] MINOR-1: JT_ERR_NOSUP=5新設、既存値不変、prefill正常→NOSUP+ENOSYS/不正→INVAL+EINVAL
- [x] MINOR-2/3: コード無変更・コメントのみ、実装と整合
- [x] MINOR-4: ci.yml matrix 4件・steps正しさ・YAML構文、CMakeLists無変更
- [x] P2 io-uring: 既定OFF+fallback、非Linux安全、NOSUP追従TODO、test_uring配線
- [x] P2 bench: bench_common正しさ、bench_smoke配線、schemaとmachine.yaml/env.sh対応
- [x] ctest 7/7 pass（arena, tmac, gdn2, routing, io_batch, bench_smoke, uring）

## 指摘（全てfollow-up可）
- MINOR-a: chunk32の対称テスト追加推奨
- MINOR-b/c: routingコメント限定語・折返し
- MINOR-d: windows-msvcのcc明示・permissions
- P2-1: 非Linuxのdead symbol整理
- P2-2: io_uring_submit==0ガード
- P2-3: median_madのmalloc overflow検査
- P2-4: io-uring benchのCMakeLists隣接hunk → 手動解決済み

## 判定
承認。develop保持を承認（d9b5e0a/c832a38/b9d9ab7/6aa7c9b/f4d7644）。
