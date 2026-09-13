# Review: P2学習エンジン中間 (develop @ 7820fb1)

レビュアー: subagent (wt-review, detached HEAD 7820fb1, 読み取り運用)
範囲: prefill chunk / T-MAC SIMD / backward+checkpoint / optim8+offload / mixed-precision+LUT-aware / E2Eスモーク

## 検証項目
- [x] ビルド健全性（11/11 pass: arena, tmac, gdn2, routing, io_batch, bench_smoke, uring[SKIP], train_bwd, optim_offload, mixed_lut, e2e_pipe）
- [x] 主張検証（E2E loss 0.161→0.054、mixed往復MAE、STE、checkpoint削減率0.859 — いずれも実測一致）
- [x] 参照整合（GDN-2 / StickyMoE / T-MAC / FRI-MxMoE / HGQ-LUT / ES-MoE — 大きな矛盾なし）
- [x] エッジケース fail-closed（NULL 92箇所、isfinite 45箇所、scratch不足・ALIGN・alpha範囲・NOSUP正常系 — 欠落なし）
- [x] P2完了条件ギャップ列挙（6点、下記）

## P2完了条件ギャップ（TODO追跡、中間のブロッカーとしない）
1. 1B級検証はスモーク代理のみ（n=8/h=8/dk=4/dv=8、2層、expert=4、gate勾配なし）
2. LUT-aware実バイナリ未実装（jt_lut_export_binary → NOSUP）
3. O_DIRECT未対応（buffered preadのみ）
4. io_uring既定OFF（Linux+liburing経路は未実行）
5. サイクル数未計測（DESIGN §6の20M/12M/6Mに対しメモリ見積のみ）
6. 16GB実証なし（64層×1MiB純粋関数見積のみ）

## 指摘
- MINOR-1（文書）: HGQ-LUT呼称が knowledge/papers.md §1 警告と衝突。AGENTS.MD §3.3修正か用語整理のこと
- MINOR-2（機能）: FRI感度が閾値スタブ。P2完了までに実推定量への差し替え計画を明示のこと
- MINOR-3（検証）: T-MAC SIMDが既定ビルドで未exercised（scalarパス）。-mavx2/-march=native別ビルドでのbit同一テスト実行記録のこと。NEON同様

## 判定
条件付き承認（MAJOR 0件）。MINOR-1〜3の解消を条件にdevelop保持を承認する。
