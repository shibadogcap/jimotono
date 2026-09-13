# Review: P2最終 (develop @ 6eb750a)

レビュアー: subagent (wt-review, detached HEAD 6eb750a, 読み取り運用)
範囲: ROADMAP §12完了条件の合否判定

## 検証項目
- [x] ビルド健全性（13/13 pass、macOS再実行）
- [x] 主張と実測の突合（e2e loss 0.161→0.054、cycle目標内、mixed往復MAE、scalar/AVX2 sink一致 — いずれも一致）
- [x] 中間MINOR-1〜3解消確認（注記/計画/AVX2実証 — 解消、NEON残存）
- [x] GAP-6点残存状況（LUT解消、O_DIRECT/uring/cycle部分解消、1B・16GB残存）
- [x] エッジケース・参照整合走査（新規欠落なし、虚偽表示なし）

## 判定
不合格。足場として5/5着手済みだが完了として2.5/5。MAJOR 3件がブロッカー。

- MAJOR-1: jt_swiglu_fwd / jt_rmsnorm_fwd がライブラリ核として存在しない（e2e・cycleは手計算代替）
- MAJOR-2: jt_ckpt_recompute_range がENOSYSスタブ（削減効果は純粋関数見積のみ）
- MAJOR-3: 1B級検証はスモーク代理のみ（n=8/h=8/dk=4/dv=8、2層、expert=4、gate勾配なし、top-k不使用）

## 指摘（MINOR）
- MINOR-a: LUT export実装後のstaleコメント（TODO/NOSUP残存）
- MINOR-b: DESIGN.MDへの実測反映なし（results/bench/20260913-n150.mdのみ）
- MINOR-c: HGQ呼称の三文書字面矛盾が残存（注記で緩和済み、AGENTS §3.3は原文のまま）
- MINOR-d: machine既定値フォールバック残存、運用明文化が必要
- MINOR-e: NEON実機記録なし

## スモーク代理の範囲
2層・expert=4・合成回帰60step・fp32スカラー。1B級外挿項目はTODO列挙。ループ疎通＋loss半減の実証に留まる。
