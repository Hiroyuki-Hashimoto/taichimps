# Taichimps 高速化・効率化および LAMMPS 機能移転ロードマップ (PLAN)

## 1. プロジェクト方針と目標
`taichimps` は Taichi Lang (Vulkan / CUDA / CPU) を基盤とした超高速 DEM (個別要素法) エンジンです。
AMD Radeon 890M (RDNA 3.5) をはじめとする GPU 環境において、Vulkan Compute による高い安定性と世界最速水準のスループット（Float32: 25.5 億粒子更新/s, 392.5 μs/step）を実証しました。
今後は `taichimps` を主軸エンジンとして位置づけ、**「さらなる高速化・GPU完全自律実行」** と **「地盤工学・土質力学に不可欠な LAMMPS 機能の完全移転」** を推進します。

---

## 2. 高速化・効率化計画 (Phase 1: Architecture & Performance)

### 2.1 Pythonループ・JITオーバーヘッドの完全排除 (`run_gpu`)
- **課題**: 現行の `run(steps)` は Python 側で `for _ in range(steps): self.step()` を回しており、1ステップあたり5〜8回のカーネルディスパッチ同期隙間が存在。
- **実装策**:
  - `run_gpu(nsteps)` による複数ステップの一括融合（Kernel Fusion）。
  - 近傍探索の判定（`delay` / `every` / `check`）を GPU カーネル内で評価し、近傍更新が不要な区間は完全自律で連続ステップ積分を実行。
  - **目標**: ホスト-デバイス通信オーバーヘッドを 0 にし、**100〜150 μs/step (現在の 2.5〜3 倍高速化)** を達成。

### 2.2 空間グリッド近傍探索のソート・メモリ局所化
- **課題**: リンクリスト方式（`grid_head`, `grid_next`）は粒子密集部でメモリアクセスが不連続になり、キャッシュ効率が低下。
- **実装策**:
  - 粒子座標から 1D セルハッシュ（Morton Code / Z-order 曲線）を計算。
  - Taichi 上で GPU カウントソート（Radix Sort）を実施し、同一セル・隣接セルの粒子データを SoA メモリ上で連続配置（Reordering）。
  - 接触判定を満たすペアのみをコンパクト配列（`active_pairs`）に収集し、力計算カーネルのスループットを倍増。

### 2.3 接触履歴変位（Shear History）のメモリフットプリント半減
- **課題**: `(max_atoms, max_neighbors, 3)` のフルアロケーションは 100 万粒子時に VRAM を圧迫。
- **実装策**:
  - 地盤土質粒子で実際に同時に生じる接触数は幾何学的に最大 12〜16 個程度。
  - 固定スロット長を 16 に最適化し、さらにビットパック・相対インデックス化。
  - メモリ帯域と VRAM 消費を 60% 削減。

### 2.4 単精度・半精度混合（Mixed Precision: FP32 / FP16）
- 位置座標・時間積分は FP32、局所接触幾何・弾性接触力の判定演算に FP16 を適用し、GPU のスループットを極限まで引き出す。

---

## 3. LAMMPS からの機能移転計画 (Phase 2 & 3: Geotechnical & DEM Models)

### 3.1 接触力学モデルの拡充 (`GRANULAR` パッケージ完全移転)
1. **転がり摩擦 (Rolling Friction)**:
   - 対象: `pair_style gran/hertz/history rolling ...`, `gran/hooke/history rolling ...`
   - モデル: EPSD (Elastic-Plastic Spring Dashpot: Iwashita & Oda 1998) および CDT (Constant Directional Torque: Ai et al. 2011)
   - 目的: 砂質土・不規則形状粒子の噛み合わせ効果（安息角、せん断強度）を再現。
2. **ねじり摩擦 (Twisting Friction)**:
   - 接触法線周りの相対回転角速度に対する履歴抵抗トルク。
3. **付着力・毛管張力モデル (Cohesion & Capillary Models)**:
   - JKR接触理論 (`gran_sub_mod_cohesion_jkr`): 微粒子・粘土・粉体の付着。
   - 液架橋力・毛管張力モデル (Capillary Bridge): 不飽和土の見かけの粘着力とサクション。

### 3.2 岩盤・固結土の破壊力学 (`BPM` パッケージ移転: Bonded Particle Model)
1. **平行結合モデル (Parallel Bonds / `bond_bpm_rotational`)**:
   - 粒子間に微小な円形断面の梁（Beam）を仮定（Potyondy & Cundall 2004）。
   - 垂直力・せん断力・曲げモーメント・ねじりモーメントを同時伝達。
2. **結合破壊基準 (Bond Breakage Criterion)**:
   - 最大引張応力基準（引張破断: Mode I）および Mohr-Coulomb 基準（せん断破断: Mode II）。
   - セメンテーション土（固結砂）、岩盤、コンクリートのき裂進展・破壊現象をシミュレーション。

### 3.3 室内土質試験サーボ制御 (`EXTRA_FIX` 境界条件)
1. **真三軸 / 円筒三軸圧縮サーボコントローラー (Triaxial Controller)**:
   - 軸荷重単調載荷（軸ひずみ制御: $\dot{\varepsilon}_1 = \text{const}$）。
   - 側圧一定制御（応力フィードバック: $\sigma_2 = \sigma_3 = \sigma_c$）。
   - 柔軟メンブレン境界モデル（Membrane Boundary）による拘束圧の均等印加。
2. **動的・繰返し単純せん断 (Simple Shear with PBC: SLLOD / Kraynik-Reinelt)**:
   - 地震波入力や液状化現象を解析するための周期境界せん断変形。

### 3.4 土質ミクロ構造・統計解析 (`COMPUTES` パッケージ)
1. **ファブリックテンソル (Fabric Tensor, 2階・4階)**:
   - 接触法線ベクトルの方向分布異方性（$F_{ij} = \frac{1}{N_c} \sum n_i n_j$）を GPU 上で並列集計。
2. **局所間隙比・配位数 (Coordination Number & Void Ratio)**:
   - 粒子ごとの配位数 $CN$ およびボロノイ分割に基づく局所間隙比 $e$ のリアルタイム算出。
3. **粒子局所 Virial 応力 (Per-atom Stress Tensor)**:
   - 各粒子における応力テンソル $\boldsymbol{\sigma}_{atom}$ の集計。

### 3.5 高速データ出力 & Web 可視化 (I/O & WebGPU)
1. **バイナリ VTK / Zarr エクスポーター**:
   - シミュレーションをブロッキングせず、GPU メモリから直接非同期出力。
2. **Web リアルタイムビューア (Bun + TypeScript + WebGPU / Three.js)**:
   - WebSocket 経由で点群データおよび力鎖（Force Chain）をブラウザへ直接ストリーミング。

---

## 4. 実装フェーズとスケジュール

```text
[Phase 1: 高速化コア基盤]
  ├── (1-1) run_gpu() カーネル内マルチステップ自律ループ実装
  ├── (1-2) 空間グリッドのメモリ再配置（Count Sort & コンパクト Pair List）
  └── (1-3) 接触履歴スロット（固定16長）の最適化

[Phase 2: 地盤力学接触モデル完全移植]
  ├── (2-1) Rolling Friction (転がり摩擦: EPSD / CDT)
  ├── (2-2) Cohesion (JKR & 液架橋毛管力)
  └── (2-3) BPM (Bonded Particle Model: セメンテーション・岩盤破壊)

[Phase 3: 室内試験サーボ制御 & ミクロ構造解析]
  ├── (3-1) 三軸圧縮コントローラー（側圧一定制御・メンブレン境界）
  ├── (3-2) ファブリックテンソル・配位数・局所間隙比の GPU Compute
  └── (3-3) WebGPU / VTK バイナリ高速ダンプ連携
```

---

## 5. 実行・検証コマンド体系
```bash
# 基本実行 (Vulkan GPU デフォルト)
python -m taichimps.main -in in.isotropic --arch vulkan --fp f32

# CPU フォールバック検証
python -m taichimps.main -in in.isotropic --arch cpu --fp f64

# テストスイート実行
uv run pytest
```
