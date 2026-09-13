# Follow-up: P2最終レビュー指摘の解消状況

## MAJOR-1（forward核欠落）→ 解消
- jt_swiglu_fwd / jt_rmsnorm_fwd をライブラリ核として追加（手計算値tol=1e-5＋bwd疎通テスト）
- e2e・cycleベンチの手計算代替を新核呼び出しに置換
- 検証：macOS 13/13、N100(uring ON) 13/13、警告0

## MAJOR-2（ckpt実再計算なし）→ 解消
- jt_ckpt_recompute_range をコールバック方式で実実装（旧fwd_fn廃止、失敗伝播・INVAL検査付き）
- e2eを新API経由に置換（seg0境界検証＋seg1再実行一致<1e-5、両seg踏み分け）
- 検証：同上 13/13

## MAJOR-3（1B級未検証）→ 残存（要判断）
- スモーク代理のまま。P3へ繰越にはROADMAP改訂が必要

## MINOR a/b/d → 解消（staleコメント、DESIGN実測列、bench_schema運用済み確認）
## MINOR c（HGQ字面矛盾）→ 残存（AGENTS原文のまま、docs/運用のため未着手）
## MINOR e（NEON実機）→ 残存（ARM機なし）

## 追加修正
- fix(ssd) _GNU_SOURCE（N100ビルド破壊を修正）
- fix(build) gcc警告ゼロ（POSIX宣言をLinux全ターゲット化、未使用dS削除）
- bench: N150 machine label（JIMOTONO_MACHINE）、LUT export実実装＋roundtrip
- results/bench/20260913-n150.md（13/13×3ビルド、AVX2＋uring実経路）
