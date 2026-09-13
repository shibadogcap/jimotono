# Phase D (SIMD) + held-out results

date: Sep 14 2026
commit: heldout merge (post d1486f8)
rig: macmini-i7-8700B, sequential runs only (RULE相互排除)

## Phase D: AVX2 hot paths (gdn2/swiglu/rmsnorm/GEMM), FMA不使用・丸め順序保存
- 等価性：両ビルド15/15 pass。loss軌道一致 (step60: 0.000382、gnorm一致)
- 性能 (train_longrun --steps 60/12):
  - scalar12T: 451 toks/s / AVX2 12T: 474 (1.05x) — 帯域飽和で差が出ない
  - scalar1T: 435 / AVX2 1T: 852 (1.96x)
  - AVX2 2T: 1270 (peak) / 3T ~940 / 4T ~608 / 6T 498
- 結論：4x未達。律速はDRAM帯域（2スレッドで飽和）。ピーク1270はbaseline比2.8x
- 1B外挿：1270÷140 ≈ 9 toks/s相当（楽観側・帯域無視）。5 tok/s目標との関係はPhase Eで判定

## held-out val (synthetic, train[0,512) vs val[100000,100512), 同一生成式)
- --steps 200 --threads 2 (scalar): train 0.443→0.000003、val 0.470→0.343 (比0.73)
- valはstep50まで減少 (0.470→0.3435) 後 ~0.3426に横ばい。0.5x基準は未達
- 解釈：合成線形タスクの限界（記憶vs汎化の分離はできたが、val改善は27%で頭打ち）。「学習している」の主張には実データが必要
- 実データ候補：llm-jp-corpus-v4 (GitLab公開、33TB全体。ja_wiki 2.2G級の一部取得が現実的。100MB超DLは未実行)
