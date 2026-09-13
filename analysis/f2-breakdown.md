# F2: train_proxy100m 1step内訳 (Phase F2 計測)

- branch: `perf/p2/f2-breakdown`（計測のみ。性能修正なし）
- 対象: `bench/kernel/train_proxy100m.c`（100M-proxy runnable、物理28,378,113 params、fp32、64 toks/step）
- 計測プローブのみ挿入（`--profile` フラグ化済み）。計算内容・順序・並列化は不変。
  parity確認: `--threads 6` profile有/無で elapsed 22.6s/22.6s、steps_sec 0.530/0.530、
  全stepのlossがbit一致。プローブ overhead は無視できる。
- ビルド: 当wt内 `build-f2/` のみ（`train_proxy100m` ターゲット、`-Wall -Wextra -Wpedantic -Wshadow` 警告0）。
- 方法: 1stepを10フェーズ（data/fwd/bwd/optim/ckpt_recompute/esmoe_io/sync/alloc/logging/other）に分割。
  wallは `clock_gettime(CLOCK_MONOTONIC)`、cyclesは x86 `rdtsc`（fenceなし参考値）。
  `other = step_wall − sum(他9相)` で残差吸収し、Σphase = Σstep_wall を厳密成立させる。
  全条件で `check diff_us=0.00` を確認。
- 実行条件: スレッド数 {1,2,4,6,8,12} × `--steps 12` を逐次・単一プロセスで測定。各run前に `uptime` 確認。
  マシン: i7-8700B（6C/12T）、メモリ常駐、バックグラウンド負荷あり（load avg 2.4〜6.8で変動）。

## 1. 10フェーズ内訳表（ms/step、% は step内比率）

logging内訳（`logging_detail`、per step平均）も併記。val以外（loss/sticky/misc）は全条件で <0.2ms。

| thr | step計 | data | fwd | bwd | optim | ckpt | esmoe_io | sync | alloc | logging (内val) | other |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 2836.7 | 0.03 (0.00%) | 510.2 (17.99%) | 1542.9 (54.39%) | 225.7 (7.96%) | 5.68 (0.20%) | 0.72 (0.03%) | 0.0 (0%) | 0.00 (0%) | 535.6 (18.88%, 内535.5) | 15.8 (0.56%) |
| 2 | 2079.1 | 0.03 (0.00%) | 259.7 (12.49%) | 989.7 (47.60%) | 221.2 (10.64%) | 5.66 (0.27%) | 0.71 (0.03%) | 45.8 (2.20%) | 0.00 (0%) | 540.4 (25.99%, 内540.3) | 15.9 (0.76%) |
| 4 | 1882.1 | 0.03 (0.00%) | 136.9 (7.27%) | 868.7 (46.16%) | 223.2 (11.86%) | 5.65 (0.30%) | 0.71 (0.04%) | 81.3 (4.32%) | 0.00 (0%) | 549.5 (29.20%, 内549.4) | 16.0 (0.85%) |
| 6 | 1885.8 | 0.03 (0.00%) | 113.3 (6.01%) | 872.4 (46.26%) | 222.0 (11.77%) | 5.65 (0.30%) | 0.79 (0.04%) | 117.1 (6.21%) | 0.00 (0%) | 538.7 (28.57%, 内538.6) | 15.8 (0.84%) |
| 8 | 1957.7 | 0.03 (0.00%) | 129.8 (6.63%) | 894.9 (45.71%) | 223.9 (11.43%) | 5.69 (0.29%) | 0.82 (0.04%) | 142.9 (7.30%) | 0.00 (0%) | 543.7 (27.77%, 内543.5) | 16.1 (0.82%) |
| 12 | 2042.5 | 0.03 (0.00%) | 126.0 (6.17%) | 898.6 (44.00%) | 222.3 (10.88%) | 5.71 (0.28%) | 0.92 (0.04%) | 230.6 (11.29%) | 0.00 (0%) | 542.2 (26.55%, 内542.1) | 16.0 (0.79%) |

- `check`: 全条件 `sum_phases == step_wall`（diff_us=0.00）。
- `other`（15.8ms、0.6〜0.8%）の主成分はステップ冒頭の重み一括 `isfinite` 走査（28M floats = 113MB read）＋残差。
- `logging` のほぼ全量が held-out `val`（single-thread CHECKED `jt_moe_fwd`）。val 1 event ≈ 2.15s。
  `--steps 12` では val が 3回（5, 10, 最終12）走るため償却 539ms/step。定常（1000-step級）では 2154/5 ≈ 431ms/step の固定費。
- `ckpt` 5.7ms/step（2 events/12 steps → 1 event ≈ 34ms。token0のみ再計算のため軽い）。
- `esmoe_io` 0.7〜0.9ms/step（2 events/12 steps → 1 event ≈ 4.7msで16MiB往復。page-cache resident）。
- `data`（acts←X memcpy 0.26MB）0.03ms、`alloc`（esmoe往復内malloc/freeのみ）0.005ms。無視できる。

## 2. I/O差分（getrusage代替＋esmoe tmpfile）

- `getrusage(RUSAGE_SELF)` の `ru_inblock/ru_oublock`: 全runで **0/0**（macOSでは維持されないことを確認）。
  Linux `/proc/self/io` の代替として以下を step前後差分で取得した。
- esmoe stats差分（12 steps共通）: `bytes_read=16,777,216`、`loads=32`、`write(register/pwrite計数)=16,777,216`。
  内訳: 1 eventあたり register 16 experts × 512KiB = 8MiB write、prefetch 2×512KiB + load 14 miss×512KiB ≈ 8MiB read。
  step換算 2.7MiB/step、帯域換算 ≈ 1.4MB/s。
- **28Mモデルでの要否判定**: esmoe/ckptともstepの0.3%以下で律速ではない。
  物理28Mは全常駐（RSS 319MiB@1T 〜 1890MiB@12T、2GB gate内）のため offload 自体が不要。
  tmpfile backing は page-cache resident で実SSD I/Oは発生していない（inblock/oublock=0 と整合）。

## 3. csw内訳（getrusage ru_nvcsw/ru_nivcsw）

- `csw_v`（自発的）は全条件・全フェーズで **0**。すべて非自発（`csw_iv`）のみ。
- stepあたり合計（profile合計＝stepログ平均、両者で一致を確認）:

| thr | csw_iv/step平均 | 最小〜最大 | 報告値(4000-6000)との関係 |
|---|---|---|---|
| 1 | 306 | 212〜470 | 単一スレッドでは1桁小さい |
| 2 | 608 | 432〜868 | — |
| 4 | 1848 | 1641〜2274 | — |
| 6 | 2783 | 2416〜4897 | 下限側と同オーダ |
| 8 | 3635 | 2095〜11106 | 範囲内（spikeあり） |
| 12 | 4329 | 2844〜13300 | 範囲内（spikeあり） |

- フェーズ別（6T例、/step）: bwd 2107 (76%)、fwd 272 (10%)、sync 294 (11%)、logging(val) 68 (2%)、optim 40 (1%)、他 ≈1。
  全条件で配分は同様（bwd 75〜80%）。
- 強度（csw_iv/ms）: MT相（fwd/bwd/sync）≈ 2.9/ms に対し、単一スレッド相（optim/val）≈ 0.1/ms。
  wall時間ではなく「並列実行」と相関 → スレッド間スケジューリング競合（7 runnable@6T＋系負荷）由来。
- スレッドあたり分解: macOSに `RUSAGE_THREAD` がなくプロセス粒度しか取れないため推定。
  6Tで 2783 ÷ 7（worker 6＋main）≈ **400/thread/step**。bwd wall按分は近似（脚注: 融合区間のcswをfwd/bwd/syncにwall比按分）。
- 結論: cswはスレッド数にほぼ比例して増大し、12Tでは4000超が常態。報告の4000-6000/stepは8〜12Tまたは高負荷時の値と整合。
  ただしcsw自体が律速の主因というより、並列区間の競合の symptom（bwd wallの75%が context-switch 下で経過）。

## 4. スレッドスイープ表（--steps 12、逐次・単一プロセス）

| thr | step (ms) | toks/s | speedup (対1T) | fwd倍率 | bwd倍率 | 備考 |
|---|---|---|---|---|---|---|
| 1 | 2836.7 | 22.6 | 1.00 | 1.00 | 1.00 | sync=0。ベース |
| 2 | 2079.1 | 30.8 | 1.36 | 1.96 | 1.56 | — |
| 4 | 1882.1 | 34.0 | 1.51 | 3.73 | 1.78 | 最速 |
| 6 | 1885.8 | 33.9 | 1.50 | 4.50 | 1.77 | ベースライン33.4と一致。4Tと同等 |
| 8 | 1957.7 | 32.7 | 1.45 | 3.93 | 1.72 | sync増で後退 |
| 12 | 2042.5 | 31.3 | 1.39 | 4.05 | 1.72 | sync 230ms (11%) で後退 |

- 最速は4〜6T（物理コア数）。12T（SMT超過）はsync増大で悪化。
- fwdは6Tで4.5xまで伸びるが、bwdは1.77xで頭打ち。以降どちらもDRAM壁で横ばい。
- optim（222〜226ms）は全条件で平坦＝serial固定費。
- run前uptime: いずれも `up 13 days`、load avg 2.4〜6.8（系負荷変動あり。8T/12Tのspikeは負荷競合と見られる）。

## 5. 律速候補の順位付け（6T、33.9 toks/s時点）

1. **bwd（872ms、46%）** — 学習ループ全体の支配項。6Tで1.77xしか伸びず頭打ち。
   推定DRAM traffic ≈ 4GB（重み+活性再読 2.3GB、Gpart zero/add/reduction 1.7GB）に対し実効 ≈ 4.6GB/s。
   同条件のfwd（2.3GB/113ms ≈ 21GB/s、実用peakの7〜8割）と比較し約1/5の効率。
   FLOPs換算でも bwd ≈ 2.75 GFLOP/s に対し fwd ≈ 10.6 GFLOP/s。バイトは動いているのに進まない＝bwdカーネル効率問題。
2. **val（償却539ms、29%。1 event ≈ 2.15s）** — single-thread CHECKED fwd。per-call重み有限スキャン
   （256 calls × 層7M floats ≈ 1.8G要素検査 ≈ 1.8s相当）が主因と推定。測定用overheadであり学習カーネルではないが、
   現行頻度（5 step毎＋最終）では第2の律速。定常でも431ms/stepの固定費。
3. **optim（222ms、12%）** — serial固定費（28Mのnorm二巡＋clip＋optim8更新）。スレッド数に無関係。
4. **sync（117ms@6T→231ms@12T、6→11%）** — 毎step fork/join＋順序付きreduction（n_total×nthr加算）。
   スレッド増で増大し、12Tでは第3位に浮上。スレッド数に対するAmdahl項。
5. **fwd（113ms、6%）** — 実用帯域の7〜8割で動作しており効率は良い。4.5x/6TでDRAM壁に到達。
6. **ckpt/esmoe/data/alloc/other（合計 <2%）** — 無視可能。28MモデルでI/O律速は存在しない。

## 6. 「帯域天井の1/192、AI 0.38-0.50では説明できない」への回答

- カーネル単体AI（dot 0.50 / swiglu_fwd 0.50 / swiglu_bwd 0.38、ridge=21 → memory-bound）は正しいが、
  ループ全体の律速はカーネルAIの問題ではない。内訳が示すのは3点の積み重ねである。
  - (a) **bwdの実効効率が帯域の約1/5**（§5-1）。AIが同じでもfwdの4〜5倍遅い。カーネル内部効率（H=128小行列の
    ベクトル化・依存連鎖・スパース加算パターン）に原因があると見られ、AI値だけでは見えない。
  - (b) **serial固定費がstepの約40%**（optim 222ms＋val償却 431〜539ms）。並列化の恩恵を受けないAmdahl項。
    valを除いた「真の学習」だけでも 1347ms/step（47.5 toks/s@6T）に留まる。
  - (c) **毎step fork/join＋reduction**（117〜231ms）。スレッドを増やすほど増加し、6T超の scaling を殺す。
- すなわち「帯域はある（fwdが実証）が、bwd効率×serial固定費×sync で 1/192 に潰れている」。
  性能修正は本任務の範囲外のため、対策の優先度のみ記す: bwd効率調査 > val頻度/checked-API見直し > optim並列化・reduction削減 > sync削減。

## 付録: 計測の実装と制約

- プローブ位置: `bench/kernel/train_proxy100m.c` のみ。`--profile` で [F2-profile] 行をstderrへ出力。
- fwd/bwd/sync分解: 融合ワーカー内でfwd/bwd wallを個別計時し、main側で `fwd=max(worker fwd)`、
  `bwd=max(worker bwd)`、`sync=融合wall−fwd−bwd`（reduction・fork/join overheadを含む）。
  単一スレッド時はfwd/bwdを直接計時しsync=0。計算内容・順序・分割は不変。
- csw按分: 融合区間のcsw差分はfwd/bwd/syncにwall比で按分（近似。getrusageはプロセス粒度のため正確な帰属は不可）。
- rdtsc: fenceなしのため参考値。tsc_GHz_est=3.192で全条件一致（ターボ変動なし）。
- 生ログ: `/tmp/f2-sweep/t{01,02,04,06,08,12}.log`、対照 `/tmp/f2-sweep/t06-noprof.log`（wt外・未コミット）。
- 未コミット（指示通りコミットしない）。`git status` では `bench/kernel/train_proxy100m.c` の変更と
  本ファイル、`build-f2/`（untracked）のみ。
