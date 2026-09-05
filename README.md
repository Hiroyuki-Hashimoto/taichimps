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
2. **GRANULAR 接触力学**:
   - `GranHooke` / `GranHookeHistory`: 線形Hooke法線・接線ばね、粘性減衰、クーロン摩擦、変位履歴。
   - `GranHertz` / `GranHertzHistory`: Hertz接触理論、減衰補正クーロンスライド。
   - `GranularModular`: モジュラー型接触モデル。
   - サブモデル: `RollingResistance` (EPSD), `TwistingResistance`, `CohesionJKR`。
3. **EXTRA_FIX パッケージ**:
   - `FixNVESphere`: 速度Verlet並進・回転2段時間積分。
   - `FixGravity`, `FixWallGran`: 重力場加速度、平面壁面接触。
   - `FixDampingCundall`: Cundall局所非粘性減衰。
   - `FixDeformPressure`: 目標等方圧密サーボ制御（セルおよび粒子のアフィン変形）。
   - `FixViscousSphere`, `FixDrag`: Stokes粘性抵抗、流体抗力。
   - `FixFreeze`: 固定拘束粒子の運動ゼロ化。
   - `FixPrint`: 変数評価ファイルの定期追記出力。
4. **熱力学量 (Computes)**:
   - `Computes`: 並進・回転運動エネルギー、巨視的Virial応力テンソル、配位数集計。
   - `ComputeStressAtom`: 各粒子局所Virial応力テンソル（6成分）。
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
`uv run taichimps <in.script>` または `-in <in.script>` により直接実行できます。
```bash
uv run taichimps /path/to/in.script --arch vulkan --precision f32
```

### (3) GGUI リアルタイム可視化
ヘッドレス環境やCI実行時は `show_window=False` としてください。GUI表示を行う場合は Vulkan ディスプレイ環境（X11 / Wayland）が必要です。
```python
from taichimps.vis import Visualizer3D
vis = Visualizer3D(domain=sim.domain, max_particles=sim.atom.nlocal, show_window=True)
while vis.render_frame(sim.atom):
    sim.step()
```
