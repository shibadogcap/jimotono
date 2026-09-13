# Review: P1統合4件 (develop @ 953e470)

## 検証項目
- [x] ビルド（Debug, AppleClang 17 / x86_64）
- [x] クロス確認（arm64 configure+build）
- [x] ctest 5/5 pass（arena, tmac, gdn2, routing, io_batch）
- [x] papers.md §1〜§4との整合
- [x] エッジケース（空入力、K%4/K%32、NaN/Inf、bits範囲外、NULL、オーバーフロー）
- [x] AGENTS.MD 7.1遵守、禁止依存リンクなし

## 実測値
- ctest: 5/5 passed
- bench_tmac（K=512, Debug）: bits=1: 0.679 / bits=2: 1.199 / bits=4: 2.438 us/call（線形）
- bench_gdn2: 倍精度リファレンスとtol=2e-5で一致
- test_routing: top-k降順+tie-break+softmax合計1、L_cons/L_hard数値一致
- test_io: runs_compress + pread往復一致

## 指摘
- MAJOR-1: T-MAC bits=3拒否（papers.md §1の主利点と矛盾）→ 対応済み（fe1d944: bits=3許可、cherry-pick 794299b）
- MAJOR-2: GDN-2非有限入力の無検証通過・state汚染 → 対応済み（3ef5d54: isfinite+alpha範囲検査、cherry-pick 61acbb4）
- MINOR-1: GDN-2プリフィルスタブのエラー別名（JT_ERR_NOSUP検討、P2で可）
- MINOR-2: T-MAC K=4のサイズ/ctor不整合（文書化のみ）
- MINOR-3: routingの過剰拒否（fail-closedのため実害なし、仕様明記推奨）
- MINOR-4: 3OS実ビルドはmacOSのみ（CIでの3OS維持を推奨）

## 判定
条件付き承認 → 条件充足（MAJOR-1/2対応済み、ctest 5/5再確認）。develop保持を承認。
