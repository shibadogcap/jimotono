# P3a factorized-head validation（P2 head-lrスキャン＋P3長期継続＋design scale検証計画）

- branch: `feat/p3/headlr-scan`（本wtのみ。コミットしない）
- 性質：実測＋文書。実装変更なし（`--head-lr-scale` 既存旗のみ使用。計算・API・tolに触れない）。
- 範囲：proxy d=64/k=16・TinyStories・逐次単一プロセス。k=512の実装・実行は禁止（contingencyに留める）。proxyでk=512を振らない。
- 基準：dense対照 2.322帯（5000 steps）とのval差≦1%。toks/sのcross-vocab比較は行わない（RULE.MD）。

## 1. P2 head-lrスキャン（各2000 steps・逐次・uptime確認）

条件（5点共通）：`--steps 2000 --patience 0 --val-every 100 --factorized-head --head-k 16 --head-init vm`
＋ `--head-lr-scale {0.15,0.20,0.25,0.30,0.35}`。d=64・layers=2・lr=3e-4・batch=64・V=258。
逐次ループ・単一プロセスで実行（開始 17:32／終了 17:38、uptimeで前後確認。各点約75秒）。

### 1.1 2000 steps時点のval（最良＝0.35）

| head-lr-scale | val@2000 | dense 2.322帯との差 | 判定（≦1%） |
|---|---|---|---|
| 0.15 | 2.6414 | +13.8% | 不合格 |
| 0.20 | 2.5445 | +9.6% | 不合格 |
| 0.25 | 2.5159 | +8.4% | 不合格 |
| 0.30 | 2.4975 | +7.6% | 不合格 |
| **0.35** | **2.4740** | **+6.5%** | 不合格（5点中最小） |

### 1.2 val曲線（500 steps毎抜粋。全点単調減少継続、未飽和）

| step | 0.15 | 0.20 | 0.25 | 0.30 | 0.35 |
|---|---|---|---|---|---|
| 500 | 3.3355 | 3.1255 | 2.9646 | 2.8675 | 2.7594 |
| 1000 | 2.7975 | 2.6467 | 2.5902 | 2.5482 | 2.5293 |
| 1500 | 2.6608 | 2.5574 | 2.5246 | 2.5061 | 2.4821 |
| 2000 | 2.6414 | 2.5445 | 2.5159 | 2.4975 | 2.4740 |

- 0.15→0.35へ単調改善（scale上昇ほどval低下）。範囲内では頭打ちなし。
- 0.35の再走（val-every 500、2000-horizon）は 500:2.7594／1000:2.5293／1500:2.4821／2000:2.4740 で完全再現（決定論的）。
- 全点で基準≦1%未達。2000 steps時点のゲート判定は不合格。ただし軌道は単調減少・未飽和のため「打切り」ではなく「最良設定で長期継続」（§2へ）。
- ログ：`build-hdlr/logs/scan-{0.15,0.20,0.25,0.30,0.35}.log`（wt内・git管理外）。

### 1.3 vm-init式のd依存（proxy最適≠design最適）

- `train_tinystories.c` のvm式 f=(d/k)^1/4・a=0.125×f は出力分散を d/k 倍だけ持ち上げる部分補償。
  dense比の絶対水準は d×V0 に依存し、proxy（d=64）では約0.33×（不足側）に対し、
  設計規模（d=1024）では約5.3×（過剰側）に反転する（`analysis/factorized-head-k-sweep.md` §3）。
- Adam bounded-step下では大初期値が相対進捗を遅らせる実測（完全一致形で200 step val＋18.6%悪化）がある。
- よって **proxy最適（0.35）≠design最適**。0.35は設計規模の既定値ではなく出発点であり、
  design scaleでは §3 のS1実走で再測定・再調整する。proxyでk=512を振って代替検証することはしない
  （proxyではk=256すらdense超過であり体制を再現しない。同文書§3）。

## 2. P3長期継続（最良0.35・5000 steps・逐次）

- P2完了後に逐次実行：`--steps 5000 --patience 0 --val-every 100 --factorized-head --head-k 16 --head-init vm --head-lr-scale 0.35`
 （17:38→17:42、約189秒、単一プロセス）。ログ：`build-hdlr/logs/long-0.35-5000.log`。
- 注意（スケジュール地平）：既定は warmup 100＋cosine decay（horizon＝`--steps`）のため、
  2000-horizon走と5000-horizon走は同stepでもbit一致しない（実測：0.35の2000時点が2.4740 vs 2.4257）。
  同一horizon内の相対順位（P2）は有効。飽和判定は5000-horizon走の3000→5000区間で読む。

| step | val | dense差 |
|---|---|---|
| 2000 | 2.4257 | +4.5% |
| 3000 | 2.3893 | +2.9% |
| 4000 | 2.3762 | +2.3% |
| 4500 | 2.3734 | +2.2% |
| 5000 | 2.3729 | +2.2% |

- 飽和点：4000→5000でΔ=0.0033（0.14%）、4500→5000でΔ=0.0005（0.02%）。
  4700:2.3730→4800:2.3729→4900:2.3729→5000:2.3729でflat。**飽和点は約4700〜4800 steps**。
- 5000時点でもdense差+2.2%で基準≦1%未達。**P3長期継続のゲート判定は不合格**（ただし改善は継続し発散なし）。
- 帰結：k=256＋head-lr調整だけではproxy 1%帯に届かない。design scale検証（§3）＋contingency（§4）へ進む。
  proxy側の追加sweep（0.35超・k=512）は行わない（0.35超は旗上限1.0内だが設計外挿の根拠にならず、k=512は禁止）。

## 3. design scale検証計画（S1 1B実実行・ゲート条件組込み）

- 目的：proxy最適を設計規模に外挿せず、S1（L=24・d=1024・V=48,588・k=256固定・untie・INT8、
  `analysis/p3-training-scale.md` §3 S1）実実行でfactorized headの妥当性を判定する。
- 実行主体：親管理（重負荷のため本wtでは提案に留め、実走しない）。
- 手順：
  1. S1構成の最初1000 stepsを現最良出発点（vm-init＋head-lr 0.35相当の設計換算）で実走する。
  2. 同一S1条件のdense-head対照（または受領済みdense帯）とのval差を測定する。比較は同一wall-clock・同一バイト数基準
     （CE_per_byte正規化。RULE.MD）。
  3. 基準：val差≦1%でS1継続、5〜15%で対策追加（`analysis/factorized-head-k-sweep.md` §4の区分を踏襲）。
- 未達時の再調整（優先度順）：
  - head-lr再調整（設計規模で0.15〜0.35周辺を再sweep。vm過剰側反転の立上がり遅延をlogで確認しながら）。
  - k=512への拡大（head 12.70M→25.40M。A100M予算内であることは同文書§2で確認済み。
    `TS_HEAD_MID_MAX` 256→512＋mid/dmバッファ拡大の実装改変が必須のため、設計改訂として親管理で実施）。
- S1ゲート条件への組込み（`analysis/p3-training-scale.md` §3 S1 exitに追加）：
  - 「S1最初1000 stepsでのfactorized/dense val差≦1%（未達時は上記再調整を経て再測定）」をexit条件の一項目とする。
  - 飽和＜0.3%・drop steady-state≦5%等の既存exit条件は変更しない。
- 禁止の再確認：本wtではk=512の実装・実行を行わない。proxyでのk=512 sweepは体制非再現のため行わない。

## 4. contingency（Stage 1行き詰まり時の脱出路）

- 本節は §3 未達・停滞時の対処であり、実装の着手ではない（P3本体判断・親管理）。
- C1：full head＋INT4（12.4MB/token）。
  factorizedの表現律速が疑われる場合、headのみfullに戻しINT4でfetchを抑える対案。
  交通費は指示値12.4MB/token（factorized k=256 INT8 12.7MBと同級。`analysis/p3-design-inputs.md` §(b)B1・
  `analysis/vocab-budget.md` §3との整合は実測で確定）。dense常駐・A100M会計への影響は設計改訂で再勘定する。
- C2：語彙プルーニング・head疎化（P4課題）。
  V=48,588削減・top-k logit等のhead疎化の実装はP4に記録のみ（`analysis/p3-design-inputs.md` §(b)遵守）。
  Stage 1では着手しない。
- C3：Stage 1行き詰まり時の脱出路。
  1. head-lr再調整→2. k=512拡大→3. C1 full head＋INT4の順で一段ずつ試し、各段でS1最初1000 stepsのval差を再測定する。
  2. いずれでもval差が5%を割らない場合は、S1 exit不通過としてS2/S3へ進まず設計会議へ申送る
     （RULE.MD §2.1：アーキテクチャ固定項目の変更は小規模アブレーション検証後に限定）。
  3. 学習中動的（ビット幅カリキュラム・ルーティング温度・損失重み）の場当たり調整でゲートを迂回しない
     （RULE.MD §2.4：変更ログと再現性を確保）。

## 5. 変更一覧・遵守記録

- 実装変更：なし（`git status` clean、`git diff` 空。`--head-lr-scale` は既存旗で範囲内使用。計算・API・tol不変）。
- 新規文書：本書1件（指示の「2件新規」は本書＋本書内contingency節§4として充足）。
- k=512：実装・実行なし（禁止遵守）。
- ctest：`build-hdlr` で18/18 pass（Total Test time 4.55s）。
- API fail-closed・tol：不変（既存テストがpass）。
- ビルド：同wt内 `build-hdlr/` のみ使用。他wt接触・checkout/switchなし。
- コミット：しない。

## 参照

- `analysis/factorized-head-k-sweep.md`（受領データ・k比較表・vm外挿注意・ゲート区分）
- `analysis/p3-training-scale.md` §3 S1（段階・exit条件）
- `analysis/p3-design-inputs.md` §(b)（head縮小算術・P4記録）
- `analysis/vocab-budget.md` §3（k=256の12.70M・語彙全体62.45M）
- `analysis/p3-architecture.md` §5（k=256固定既定）
- `bench/kernel/train_tinystories.c`（vm-init式・`TS_HEAD_MID_MAX`・`--head-lr-scale`）
- `bench/kernel/test_model1b.c:58-73`（active_bodyはhead/emb除外のbody会計）
- ログ（git管理外）：`build-hdlr/logs/scan-*.log`、`build-hdlr/logs/long-0.35-5000.log`
