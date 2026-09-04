# Taichimps 実装計画書 (PLAN.md)

LAMMPS GRANULAR パッケージの Taichi Lang による超高速個別要素法（DEM）再実装プロジェクト。

---

## 1. プロジェクト概要

- **名称**: `Taichimps` (たいちんぷす)
- **目的**: LAMMPS C++ (GRANULAR package) の粒子接触力学・時間積分・境界サーボ制御を、**Taichi Lang（Python DSL JITコンパイラ）** を用いて超高速化し、単一GPUカーネル融合（Kernel Fusion）によりPyTorchのカーネル起動オーバーヘッドを打破する。
- **ターゲット環境**:
  - Linux (x86_64) / Windows (x86_64)
  - バックエンド: **Vulkan / CUDA / CPU (LLVM)**
  - AMD Radeon 890M (gfx1150) / NVIDIA GeForce / AMD Radeon
- **環境構築**: `uv` によるモダンPython環境管理（Python 3.12+）

---

## 2. 背景とアーキテクチャ比較

DEMシミュレーションにおける各フレームワークの特性比較：

| 項目 | LAMMPS C++ (Reference) | PyTorch (torchmps) | Mojo (mojommps) | **Taichi (Taichimps)** |
|---|---|---|---|---|
| **言語・処理系** | C++17 AOT | Python + LibTorch API | Mojo (MLIR/LLVM) | **Python DSL + Taichi LLVM JIT** |
| **GPUカーネル起動** | 1プロセス/MPI | 演算ごと個別起動 (30+回/step) | 単一カーネル融合 | **単一カーネル融合 (`@ti.kernel`)** |
| **データ構造** | ポインタ配列 / SoA | `torch.Tensor` | `UnsafePointer` / 構造体配列 | **`ti.field` (SoA / AoS 最適配置)** |
| **近傍・接触探索** | リンクセルリスト (C++) | セルハッシュ＆テンソルソート | グリッドハッシュポインタ | **空間ハッシュグリッド (`ti.field`)** |
| **接線変位履歴** | `firsthistory` ポインタ | `torch.searchsorted` | メモリスロット / ハッシュキー | **固定長スロット or コンパクトハッシュ** |
| **AMD GPU対応** | OpenMP / HIP | ROCm HIP (公式Wheel) | MAX Engine HIP (gfx1150) | **Vulkan / (Experimental HIP) / CPU** |
| **予想性能比** | 1.0x (CPU基準) | 0.5x〜15x (粒子数依存) | **30x〜50x** (CPU基準) | **10x〜30x (PyTorch比 3〜5倍高速)** |

---

## 3. ディレクトリ・モジュール構成（LAMMPS / mojommps準拠）

LAMMPSの `src/` および `mojommps` のフォルダ・ファイル階層と可能な限り1対1で一致させます。

```text
taichimps/
├── pyproject.toml
├── PLAN.md
├── src/
│   └── taichimps/
│       ├── __init__.py
│       ├── config.py                 # Taichi初期化 (ti.init, arch=vulkan/cuda/cpu, precision=f32/f64)
│       ├── atom.py                   # AtomFields (x, v, f, omega, torque, radius, mass 等の ti.field)
│       ├── domain.py                 # Domain (boxlo, boxhi, prd, 周期境界PBC, remap)
│       ├── neighbor.py               # SpatialGridNeighbor (空間グリッド探索・近傍リスト生成カーネル)
│       ├── fix_nve_sphere.py         # FixNVESphere (球の並進・回転Velocity-Verlet統合カーネル)
│       ├── fix_neigh_history.py      # FixNeighHistory (接線履歴変位の管理カーネル)
│       ├── compute_pressure.py       # ComputePressure (Virial応力テンソル集計カーネル)
│       ├── simulation.py             # Taichiシミュレーション統合エンジン
│       ├── input.py                  # LAMMPSスクリプト完全互換パーサー
│       ├── data_reader.py            # LAMMPS read_data パーサー
│       ├── dump.py                   # LAMMPS dump custom ライター
│       ├── GRANULAR/
│       │   ├── __init__.py
│       │   ├── pair_gran_hertz_history.py  # Hertz-Mindlin 接触力学カーネル
│       │   ├── pair_gran_hooke_history.py  # Hooke-Mindlin 接触力学カーネル
│       │   ├── pair_gran_hooke.py          # 線形Hooke接触カーネル
│       │   ├── pair_granular.py            # モジュラー接触ペアカーネル
│       │   ├── granular_model.py           # サブモデル結合エンジン
│       │   ├── gran_sub_mod_rolling.py     # 転がり摩擦サブモデル
│       │   ├── gran_sub_mod_twisting.py    # ねじり摩擦サブモデル
│       │   ├── gran_sub_mod_cohesion_jkr.py# JKR付着力サブモデル
│       │   ├── fix_damping_cundall.py      # Cundall局所非粘性減衰カーネル
│       │   ├── fix_gravity.py              # 重力加速度カーネル
│       │   ├── fix_wall_gran.py            # 6面境界壁接触カーネル
│       │   ├── fix_wall_gran_region.py     # 円筒・ボックス境界接触カーネル
│       │   ├── fix_freeze.py               # 粒子凍結カーネル
│       │   ├── fix_move.py                 # 規定運動カーネル
│       │   ├── fix_pour.py                 # 粒子生成・投入エンジン
│       │   ├── triaxial_controller.py      # 三軸圧縮サーボ制御
│       │   └── compute_contact_atom.py     # 配位数・接触数計算カーネル
│       ├── EXTRA_FIX/
│       │   ├── __init__.py
│       │   ├── fix_deform_pressure.py      # 等方圧密圧力フィードバックサーボカーネル
│       │   ├── fix_drag.py                 # 速度比例抗力カーネル
│       │   ├── fix_viscous_sphere.py       # Stokes粘性抵抗カーネル
│       │   ├── fix_controller.py           # PID制御カーネル
│       │   └── fix_addtorque_group.py      # グループトルク付与カーネル
│       └── computes/
│           ├── __init__.py
│           ├── compute_stress_atom.py      # 各粒子局所Virial応力カーネル
│           └── compute_fabric.py           # 2階ファブリックテンソル計算カーネル
└── tests/
    ├── test_atom_domain.py
    ├── test_hertz_analytical.py
    ├── test_cundall_damping.py
    ├── test_deform_pressure.py
    └── test_lammps_script_run.py
```

---

## 4. Taichiによる実装・最適化のキモ

### (1) 単一カーネル融合 (Kernel Fusion) によるメモリ帯域の極小化
PyTorch版では、接触ペアごとに力・トルク・相対速度・履歴更新・Virial集計を行う際に、30以上の小さなGPUカーネルが順次起動され、グローバルメモリ読み書きのオーバーヘッドが生じていました。
Taichi版では、これらを **1つの `@ti.kernel` 内に融合** します：
```python
@ti.kernel
def compute_hertz_contact():
    for p in range(num_active_pairs[None]):
        i = pair_i[p]
        j = pair_j[p]
        # 相対位置・法線ベクトル・重なり量計算
        # Hertz法線ばね＋法線ダッシュポット
        # Mindlin接線ばね＋接線履歴更新＋クーロン摩擦クリッピング
        # 力とトルクを atom.f[i], atom.f[j], atom.torque[i], atom.torque[j] にアトミック加算
        # Virial応力テンソルに足し込み
```
これにより、レジスタ内で力学計算が完結し、メモリアクセス回数が劇的に激減します。

### (2) 空間グリッド近傍探索 (Spatial Grid Hashing)
- 3次元グリッド `grid_head[gx, gy, gz]` と `particle_next[i]` によるリンクリストをTaichiフィールドで構築。
- または、粒子をセルIDでソートするカウントソートカーネルをTaichiで並列実装し、GPU上で完全自律動作。

### (3) 接線履歴変位（Shear History）の管理
- 粒子あたり最大接触数（例: 最大12〜16接触）を固定長スロットバッファ `shear_history[i, slot, 3]` および `contact_target[i, slot]` として保持。
- 毎ステップ接触ペアを走査し、同一ペアが存在すれば前ステップの $\boldsymbol{\xi}_t$ を更新、非接触となったスロットは即座にリセット。
- 動的メモリアロケーションを完全にゼロにし、GPU上で最高速で動作させます。

---

## 5. 開発ロードマップ

1. **フェーズ1: 基礎インフラとデータ構造設計**
   - `pyproject.toml` 設定（`taichi`, `numpy`, `pytest` 等）
   - `config.py`（アーキテクチャ切替: `ti.vulkan`, `ti.cuda`, `ti.cpu`、精度切替: `ti.f32`, `ti.f64`）
   - `atom.py`, `domain.py`
2. **フェーズ2: コア力学エンジンと接触モデル**
   - `neighbor.py`（空間ハッシュ並列近傍探索）
   - `pair_gran_hertz_history.py`, `fix_neigh_history.py`
   - `fix_nve_sphere.py`（球のVelocity-Verlet積分）
   - `fix_damping_cundall.py`
3. **フェーズ3: 境界制御と入出力互換性**
   - `fix_deform_pressure.py`（LAMMPS厳密一致の等方圧密サーボ）
   - `compute_pressure.py`（大域Virialテンソル計算）
   - `input.py`, `data_reader.py`, `dump.py`
4. **フェーズ4: 拡張パッケージの移植**
   - Hooke接触、Rolling/Twisting/Cohesion、三軸圧縮コントローラ、壁接触
5. **フェーズ5: 検証とベンチマーク比較**
   - 2球接触解析解照合（相対誤差 $10^{-6}$ 検証）
   - 3万粒子等方圧密ベンチマーク（`in.isotropic_test`）
   - 速度比較：**LAMMPS CPU vs PyTorch (torchmps) vs Mojo (mojommps) vs Taichi (Taichimps)**
