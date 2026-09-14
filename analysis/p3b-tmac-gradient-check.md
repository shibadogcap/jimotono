# T-MAC gradient check (fake-quant vs true LUT)

date: 2026-09-14

## 順伝播/逆伝播の経路確認
- fake-quant順伝播＋fp32逆伝播、真LUT順伝播＋fp32逆伝播。いずれも逆伝播は
  既存fp32 bwdをそのまま使う（STE）。量子化勾配の別経路は存在しない。
- よって勾配差は順伝播差の線形伝播に留まるはず。

## 単体検証（test_lowbit.c test_ste_grad、n=8/h=4）
- 順伝播相対差 frel に対する勾配相対差 grel が grel ≤ 5·frel + 1e-6 を満たす
  ことをassert。passを確認。
- 結論：fake-quantと真LUTの勾配は同一STE機構であり、学習ダイナミクスの差
  として記録する（設計修正なし）。

## G2/G4実測との整合
- 順伝播差（+1.0〜1.3%）がそのまま学習軌道差になる。別途の勾配起因乖離なし。
