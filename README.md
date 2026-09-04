# taichimps: High-Performance Taichi-Lang Granular DEM Engine

Taichi Lang（Python JIT / GPU Kernel Fusion）を用いた、土質・地盤工学特化の次世代離散要素法（DEM）シミュレータ。

---

## 1. 概要 (Overview)

`taichimps` は、LAMMPS `GRANULAR` パッケージの接触力学および境界サーボ制御を、**Taichi Lang（Python JIT）** の単一融合カーネル（Kernel Fusion）によって高速実装するプロジェクトです。

- **PyTorch版 (`torchmps`)** に対する利点:
  毎ステップ数十回の個別カーネル起動オーバーヘッドを解消し、近傍走査・接触力学・Verlet積分を融合JITコンパイル。
- **Mojo版 (`mojomps`)** に対する位置づけ:
  Python エコシステム内でネイティブC++並みの超高速DEMシミュレーションを実現。

詳細な設計計画・アーキテクチャ・モジュール構成・検証ロードマップについては、[PLAN.md](PLAN.md) をご覧ください。

---

## 2. 詳細実装計画 (Implementation Plan)

- [PLAN.md](PLAN.md): 包括的設計書・コンパイラ比較・モジュール設計・段階的ロードマップ
