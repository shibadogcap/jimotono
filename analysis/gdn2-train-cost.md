# GDN-2学習カーネルのCエンジンコスト見積もり (Stage 0ブロッカー #2)

- branch: `feat/p3/stage0-blockers`、commit `96c10b1` (+ S0b-1の2点修正適用状態で計測。
  修正内容: `src/moe_layer.c` zero-fillのmemset化、`src/train_bwd.c`への`-mavx2`付与。
  gdn2-bwdはAVX2パス有効、gdn2-fwdは`src/gdn2.c`に`-mavx2`が付いていないため
  スカラー4-wayアンロール経路のままであることに注意)
- マシン: macmini-i7-8700B (i7-8700B 3.20GHz)、`freq=3.2GHz (sysctl:hw.cpufrequency)`
- 実測: `build-s0/test_cycle_bench` (dims: dk=32/dv=64、n=64/h=64、trials=11、
  repeat: gdn2-fwd 32 / gdn2-bwd 16)。wall→cycle換算 (perf不可のため見積り)
- 記法: 1 MAC = 2 FLOP (mul+add)。double累積も1 FLOPとして数える。
  検証スキャン (`all_finite` 等) はカーネルFLOPに含めず別記する

## 1. fwd (`jt_gdn2_decode_step`) のFLOP・バイト・AI (dk=32, dv=64, dd=2048)

| 区画 | FLOP (mul+add) | 備考 |
|---|---|---|
| 0) q/k L2正規化 ×2 | 2×(64+31)=190 | 各32mul+31add (double累積) +1 sqrt +1 div |
| 1) S'=D·S | 2048+0=2048 | 2048 mul |
| 2) ke + c=ke^T·S1 | 32+2048 / 2048 | ke 32mul + 2048 MAC |
| 3) S''=S'−k·c^T | 2048 / 2048 | 2048 MAC |
| 4) vw=w⊙v | 64 | 64 mul |
| 5) S=S''+k·vw^T | 2048 / 2048 | 2048 MAC |
| 6) o=S^T·q | 2048 / 2048 | 2048 MAC |
| **合計** | **mul 10,464 + add 8,254 = 18,718** | +2 sqrt +2 div (+検証スキャン) |

バイト (compulsory DRAM、f32):
- S行列: step1 R+W、step2 R、step3 R+W、step5 R+W、step6 R
  → read 5×2048 + write 3×2048 = 16,384 floats = **64 KiB**
- ベクトル (q,k,b,alpha 4×32、v,w 2×64、out_o 64、qn/kn/c/vwはL1常駐): **約2 KiB**
- 合計 **約66 KiB**。検証スキャンは入力ベクトルの再読 (L1ヒット) のみ
- **AI_fwd = 18,718 / 67,584 ≈ 0.28 FLOP/B**

## 2. bwd (`jt_gdn2_decode_bwd`) のFLOP・バイト・AI (dk=32, dv=64)

| 区画 | FLOP (mul+add) | 備考 |
|---|---|---|
| 0) nq/nk + qn/kn化 | 2×(32+31)+64=190 | +2 sqrt |
| 1) S1=D·Sprev (再計算) | 2048 | 2048 mul |
| 2) ke + c=ke^T·S1 (再計算) | 32+2048 / 2048 | 2048 MAC |
| 3) vw=w⊙v (再計算) | 64 | |
| 4) Sn再計算 (s1−ki·c+ki·vw) | 4096 / 4096 | 要素5 FLOP中の4 (2mul+2add) |
| 5) dqn=Sn·dO | 2048 / 2048 | 2048 MAC |
| 6) G=dS_next+qn·dO^T | 2048 / 2048 | affine 2048 MAC |
| 7) dc/dvw (同一積を±累積) | 2048 / 4096 | S0b-1所見: スカラー経路は積を2回計算する冗長あり (hot外のため不変) |
| 8) dkn_sn/dkn_s2 | 4096 / 4096 | 2×2048 MAC |
| 9) dke + dB/dkn合算 | 2048 / 2048 | +32 mul (dB scale) |
| 10) k正規化補正 (dot+3FLOP×dk) | 約128+96 | +dk除算 |
| 11) dS1/dSp/dAlpha (要素5 FLOP) | 3×2048 / 2×2048 | dS1 2 + dsp 1 + da 2 |
| 12) q正規化補正 | 約128+96 | |
| 13) dW/dV | 128 | |
| **合計** | **mul 22,784 + add 26,686 ≈ 49,470** | +2 sqrt +2dk除算 (+検証フルスキャン) |

バイト (compulsory DRAM、f32。dd=2048がL1/L2常駐のため再読は殆どキャッシュヒット):
- Sprev R×2、dS_next R、S1 W+R+R、Sn/G W+R×5 (dqn/dc/dkn×2/dS1)、dS_prev W
  → S系 26,624 floats ≈ **104 KiB** (実DRAMは初回のみ。dd=8 KiB常駐)
- ベクトル・勾配出力: **約3 KiB**。検証スキャン (S_prev/dS_next含む全入力) は計算と同領域の再読
- 合計 **約107 KiB** (compulsory。常駐後はL1/L2で賄われる)
- **AI_bwd = 49,470 / 109,568 ≈ 0.45 FLOP/B**

## 3. cycleベンチ実測との突合

| カーネル | median_ns/op | cycles/op | FLOP/op | GFLOP/s | FLOP/cycle | 実効帯域換算 |
|---|---|---|---|---|---|---|
| gdn2 fwd | 5,125 | 16,400 | 18,718 | **3.65** | 1.14 | 66KiB/5.1us ≈ 12.9 GB/s |
| gdn2 bwd | 10,750 | 34,400 | 49,470 | **4.60** | 1.44 | 107KiB/10.75us ≈ 10.0 GB/s (compulsory換算。常駐のため実DRAMは更に小さい) |
| (参考) swiglu fwd n64/h64 | 10,625 | 34,000 | — | — | — | — |
| (参考) swiglu bwd n64/h64 | 10,125 | 32,400 | — | — | — | — |
| (参考) rms fwd/bwd n64 | 174 / 285 | 556 / 913 | — | — | — | — |

- fwd 1.14 FLOP/cycle はスカラー4-wayアンロール (gdn2.cは`-mavx2`なし) として妥当
  (依存連鎖のないストリーミングMACが2ポートに乗る)。
  bwd 1.44 FLOP/cycle はS0b-1の`-mavx2`付与後 (jt_bwd_avx2_* 有効) の値。
  修正前の同条件はないが、swiglu単体で3.8倍 (0.566→0.148ms) の例から、
  bwdもスカラー時は約1/3程度だったと推定される
- roofline判定: STREAM Triad実測 (S0b-4 `bench/kernel/bw_stream.c`、単一スレッド、
  double N=4M=96MB): **triad 21.35 GB/s (再走 20.06) / copy 17.01 (再走19.34)**。
  公称 (dual DDR4-2666 = 42.6 GB/s) に対する **η ≈ 0.47〜0.50**。
  write-allocate (RFO) 補正後の真のDRAM転送は約1.33倍 (≈28 GB/s、η≈0.65) と推定される
  (triad/copyとも書きミス時にRFOリードが発生するため)。
  F2のfwd実測 (21 GB/s実用peak) と単一スレッドTriadが一致する。
  FP32-AVX2-noFMA peak 51.2 GFLOP/s (8-wide×(mul+add)/cycle×3.2GHz) に対する
  ridge = 51.2/21.3 ≈ 2.4。
  AI_fwd 0.28、AI_bwd 0.45 とも ridgeの1/5〜1/9 → **明確にmemory-bound**。
  実効 3.6〜4.6 GFLOP/s は roofline上限 (AI×21GB/s ≈ 5.9〜9.5 GFLOP/s) の6〜8割であり、
  小次元 (dd=8KiB) の固定費 (検証スキャン・関数呼出・repeatループ) を考慮すれば健全。
  すなわち GDN-2核自体にF2的な「1/5の謎」はない。F2のbwd問題はMoE側
  (zero-fillトラフィック + train_bwdスカラー律速。#1で対処) のものである
- N150既報値 (DESIGN.MD §6: fwd_sum 98,403 / bwd_sum 143,510 / step_est 2.24M cycles)
  に対し本計測は fwd_sum 50,956 / bwd_sum 67,713 / step_est 2.12M cycles。
  約1.9〜2.1倍速く、周波数 (N150 3.6GHz turbo vs 3.2GHz) 差では説明がつかない分は
  uarch・コンパイラ・S0b-1のAVX2有効化の寄与。オーダーは整合しており、
  DESIGN.MD §6のマイクロ目標 (fwd 6M / bwd 12M / step 20M) にはいずれも2桁余裕でOK

## 4. system-level注記 (外挿の限界)

- 上記は単一head・小次元の核コストであり、層数×head数×系列長の積み上げと
  MoE発火分・オプティマイザ・固定費 (val/optim/sync) が別途乗る。
  実学習1stepの支配項はF2内訳 (bwd 46%・val償却 29%・optim 12%) を参照。
  proxy実測 (S0b-1後、6T): fwd 58ms / bwd 847ms。
  DESIGN §6の20M/step目標は最終スパース・低ビットエンジン前提の値であり、
  fp32 scaffold現状との差は層数掛け+帯域モデル (GB/token) で別途見積もる (TODO継続)
- follow-up (本タスク範囲外): `src/gdn2.c` に`-mavx2`未付与 (内蔵AVX2ヘルパー死蔵)。
  S0b-1と同型の1行変更でfwdが改善する見込み。model1b/longrun側のhot度を確認して別途判断する

## 付録: 再現手順 (短条件)

```
cmake -S . -B build-s0 && cmake --build build-s0 -j 6 --target test_cycle_bench
./build-s0/test_cycle_bench   # 完走のみassert。OVERはfailにしない
```

- FLOP表の数え方は本ファイル§1-2の区画表による (1 MAC=2 FLOP、検証スキャン除外)。
  `bench/kernel/test_cycle_bench.c` 冒頭の旧コメント (16k/60〜80k) とは
  数え方 (MACの数え方・検証有無) の差であり、オーダーは一致する
