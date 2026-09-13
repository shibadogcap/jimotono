# Review: P0 papers (subagent, 2026-09-13)

## 検証項目
- [x] T-MAC: bit-serial分解・LUT 16→8圧縮・g=4固定・fast aggregation OFF・HGQ区別・性能数値、いずれも原典と一致
- [x] GDN-2: erase/write分離・更新式・C=64・デコード逐次核・性能数値、原典と一致。AGENTS選択と矛盾なし
- [x] StickyMoE: Soft/Hard数式・W=4・SR59%/CHR3.92倍・λ開始点、原典と一致。top-8拡張注意は妥当
- [x] NeuroPrefetcher: 82〜85%持続・103MiB・O_DIRECT使い分けは原典/DESIGNと一致。71.6%はColibri実測と明記あり
- [ ] io_uring batch=256等の詳細はコード未確認 → 「未確認」明記で是正済み
- [ ] Phase 1申し送りは測定定義不足 → 条件付記で是正済み

## 指摘
### 重大 (是正済み)
- M1: `103MiB/99%/59%/3.92x` の無条件目標化 → 各章に測定条件を付記し、申し送りに警告文追加。
- M2: AGENTS 3.3「HGQ-LUT方式」とpapers訂正の不整合放置 → papers側にAGENTS修正タスクを明記 (AGENTS本体修正はP1 docs作業で実施)。

### 軽微 (是正済み)
- m1: `act_group=32/64` に「JIMOTONO仮定」明記。m2: nibble案に「独自提案」明記。m3: W開始を `W=2〜4` に修正。m4: I/O詳細に「未確認」明記。m5: 71.6%に再測定注記。m6: タイル再検討を実験計画化。

## 判定
条件付き承認 → 是正済みのため承認相当。
