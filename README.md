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

## 2. 詳細実装計画と機能一覧 (Implementation & Modules)

- [PLAN.md](PLAN.md): 包括的設計書・コンパイラ比較・モジュール設計・段階的ロードマップ

### 実装済みモジュール
1. **コア基盤**:
   - `Domain`: 周期境界 (PBC) および非周期境界、最小画像間距離計算。
   - `AtomSystem`: 位置 `x`、速度 `v`、角速度 `omega`、力 `f`、トルク `torque`、半径 `radius`、質量 `rmass`。
   - `NeighborList`: 空間グリッド分割（Cell linked-list）＋Skin変位監視（差分更新）。
   - `ContactHistory`: せん断・転がり変位履歴スロット管理。
   - `Simulation`: Verlet時間積分・オーケストレーション。
2. **GRANULAR 接触力学**（LAMMPS バイナリとの数値照合済み）:
   - 旧世代 `pair_style gran/*` 相当: `GranHooke` / `GranHookeHistory` / `GranHertz` / `GranHertzHistory`。
     線形Hooke・Hertz法線ばね、meff比例粘性減衰、クーロン摩擦、接線変位履歴。`limit_damping` 対応。
   - 新世代 `pair_style granular` 相当: `PairGranular`。normal (`hooke` / `hertz` / `hertz/material`)、
     damping (`velocity` / `mass_velocity` / `viscoelastic` / `tsuji` / `coeff_restitution`)、
     tangential (`linear_nohistory` / `linear_history` / `mindlin`) を組み合わせ指定。
     未実装のサブモデル（JKR / DMT / MDR / rolling / twisting / heat）は明示的にエラーになります。
   - サブモデル: `RollingResistance` (EPSD), `TwistingResistance`, `CohesionJKR`。
3. **EXTRA_FIX パッケージ**:
   - `FixNVESphere`: 速度Verlet並進・回転2段時間積分。
   - `FixGravity`, `FixWallGran`: 重力場加速度、平面壁面接触。
   - `FixDampingCundall`: Cundall局所非粘性減衰。
   - `FixDeformPressure`: LAMMPS `fix deform/pressure` 相当。軸ごとに
     `pressure` / `pressure/mean` / `erate` / `trate` / `vel` / `final` / `scale` / `delta` / `volume`
     を指定でき、ひずみ制御軸と圧力サーボ軸の混在（三軸試験）が可能。
     `couple` / `max/rate` / `normalize/pressure` / `remap` 対応。
   - `FixViscousSphere`, `FixDrag`: Stokes粘性抵抗、流体抗力。
   - `FixFreeze`: 固定拘束粒子の運動ゼロ化。
   - `FixPrint`: 変数評価ファイルの定期追記出力。
4. **熱力学量 (Computes)**:
   - `Computes`: 並進・回転運動エネルギー、圧力テンソル、配位数集計。
     Virial は接触ごとに `r_ij x f_ij` で積算されるため周期境界で並進不変です。
     `kinetic=False` で `compute pressure NULL pair`（pairのみ）相当になります。
   - `ComputeStressAtom`: 各粒子のVirial応力テンソル（6成分、pressure*volume単位）。
   - `ComputeFabric`: 接触異方性（2階ファブリックテンソル）。
   - `ComputeContactAtom`: 各粒子の配位数並列集計。
5. **入出力 (I/O) & パーサー**:
    - `read_data`: LAMMPS `read_data` (Sphereフォーマット) 高速パーサー。
    - `input.py` / `script_parser.py`: 変数式評価、動的参照（`${var}`, `c_1[1]`, `vol`, `lx` 等）、ループ制御（`variable loop`, `next`, `jump`）完全対応。
    - `DumpWriter`: LAMMPS custom dump形式およびVTK出力。
6. **リアルタイム可視化**:
   - `Visualizer3D` (`taichimps.vis`): GPU Zero-Copy による Taichi GGUI (Vulkan) リアルタイム3Dビューア。

---

## 3. 実行時の注意点 (Execution Guidelines)

### (1) バックエンドと実行精度 (Precision & Backend)
- **Vulkan バックエンド (推奨)**: AMD Radeon 890M / RDNA 3.5 APU や各種GPUで最速性能（100万粒子で 390 $\mu$s/step (F32), 200 $\mu$s/step (F16)）を発揮します。
- **精度指定 (f32t / f64t)**:
  - デフォルトでは `float_type=ti.f64` ですが、GPUベンチマークや大規模計算では `float_type=ti.f32` または `ti.f16` が使用可能です。
  - すべてのカーネル引数は `ti.template()` で抽象化されており、データフィールド型とスカラ定数型の不一致によるGPU暗黙キャストオーバーヘッドが発生しないよう設計されています。

### (2) LAMMPS スクリプトの実行方法
`taichimps <in.script>` または `-in <in.script>` により直接実行できます（デフォルトで Vulkan GPU 加速が有効）。
```bash
# Vulkan GPU 加速（デフォルト）
uv run taichimps /path/to/in.script

# 精度指定 (f32 または f64)
uv run taichimps -in /path/to/in.script --fp f32 --arch vulkan
```

### (3) GGUI リアルタイム可視化
ヘッドレス環境やCI実行時は `show_window=False` としてください。GUI表示を行う場合は Vulkan ディスプレイ環境（X11 / Wayland）が必要です。
```python
from taichimps.vis import Visualizer3D
vis = Visualizer3D(domain=sim.domain, max_particles=sim.atom.nlocal, show_window=True)
while vis.render_frame(sim.atom):
    sim.step()
```

---

## 4. 統合GPUベンチマーク実績 (Unified GPU Benchmark: 1,000,000 Particles)

AMD Ryzen AI 9 HX PRO 370 (12C/24T) + AMD Radeon 890M GPU (RDNA 3.5 UMA 75GB, Linux) における 1,000,000（100万）粒子DEM等方圧密系の統一ベンチマーク実測値です。

| 実装 / エンジン | 実行環境 / バックエンド | 計算精度 | 200ステップ時間 (ms) | 1ステップ時間 (μs) | スループット (Matoms/s) | vs LAMMPS CPU (基準) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **LAMMPS C++** | CPU (Ryzen 12C/24T, OpenMPI) | Float64 | 4,399.82 ms | 21,999.10 μs | 45.46 | **1.00x** *(Baseline)* |
| **torchmps** | GPU (Radeon 890M, PyTorch ROCm) | Float64 | 481.00 ms | 2,405.00 μs | 415.80 | **9.15x** |
| **mojomps** | GPU (Radeon 890M, Mojo MAX GPU) | Float64 | 153.96 ms | 769.80 μs | 1,299.04 | **28.58x** |
| **taichimps** | GPU (Radeon 890M, Taichi Vulkan) | Float64 | 151.74 ms | 758.70 μs | 1,318.04 | **28.99x** |
| **jammps** | GPU (Radeon 890M, JAX ROCm XLA) | Float64 | 143.62 ms | 718.08 μs | 1,392.60 | **30.64x** |
| **torchmps** | GPU (Radeon 890M, PyTorch ROCm) | Float32 | 280.69 ms | 1,403.45 μs | 712.53 | **15.68x** |
| **jammps** | GPU (Radeon 890M, JAX ROCm XLA) | Float32 | 94.65 ms | 473.23 μs | 2,113.14 | **46.49x** |
| **mojomps** | GPU (Radeon 890M, Mojo MAX GPU) | Float32 | 82.48 ms | 412.40 μs | 2,424.83 | **53.35x** |
| **taichimps** | GPU (Radeon 890M, Taichi Vulkan) | Float32 | **78.49 ms** | **392.45 μs** | **2,548.10** | **56.06x (最速)** |

> **今後の開発の柱：`taichimps`**
> - **Vulkan Compute による卓越した可搬性と超高速性能**: ROCm / HIP ドライバの不安定さや外部プロセス競合を受けず、AMD Radeon 890M 上で最も安定かつ最高速（Float32 で **56.06倍高速**・ステップあたり **392 μs**）を達成。
> - **JITカーネル融合**: 接触判定・せん断履歴更新・壁面反力・速度Verlet積分を単一GPUパスで実行し、GPUディスパッチオーバーヘッドを最小化。
> - **Pythonエコシステムとの完全親和性**: PyTorch/NumPy連携やGGUI 3Dリアルタイム可視化が同一プロセス内で完全シームレスに動作。
