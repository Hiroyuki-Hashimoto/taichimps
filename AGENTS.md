# taichimps: AI Agent Operational Instructions

本書は `taichimps` リポジトリを保守・拡張・検証するAIエージェント向けの厳格な運用指針です。

---

## 1. 精度・型整合性の絶対規約 (Precision Guidelines)
1. **暗黙キャストの禁止**:
   - `float_type` として `ti.f64`, `ti.f32`, `ti.f16` が注入されます。
   - カーネル内の定数・係数（`kn`, `kt`, `gamman`, `gammat`, `dt`, `volume` 等）の引数は **`ti.template()`** で受け取り、ハードコードされた `ti.f64` / `float` を使用してはなりません。
   - Taichiカーネル内で浮動小数点リテラル（例: `0.5`）を代入・乗算する際は、フィールド型に合わせたキャストが発生しないよう留意してください。
2. **Taichiデータフィールドのアライメント**:
   - 粒子属性は構造体（AoS）ではなく、SoA形式（`ti.Vector.field` または `ti.field`）で平坦に保持します。

---

## 2. ループ最適化規約 (Loop Optimization Guidelines)
1. **早期枝刈りとゼロ除算防止**:
   - 粒子間距離 $r^2$ のカットオフ超過判定（`rsq >= radsum * radsum`）は平方根計算 `ti.sqrt()` の前に必ず実行すること。
   - 粒子自己対話（`j == i`）や無効ペア（`rsq <= 0.0`）は即座に `continue` でスキップすること。
2. **差分近傍リスト更新 (Skin Rebuild)**:
   - 近傍探索（`NeighborList`）は毎ステップ再構築せず、粒子の最大累積変位が $0.5 \times \mathrm{skin}$ を超えた時のみ実行すること。
   - 変位監視カーネルはリダクションを極力最小化し、早期フラグ立てを行うこと。

---

## 3. LAMMPSスクリプトパーサー互換性
1. **LAMMPSディレクティブの純粋性**:
   - 独自の拡張構文を追加せず、LAMMPS本家の構文仕様（`variable`, `fix`, `compute`, `pair_style`, `jump`, `next` 等）に厳格に準拠すること。
2. **変数の動的評価**:
   - `eval_fn` での変数展開時は、正規表現・数式パーサーによる依存解決順序を破壊しないこと。

---

## 4. 自動品質検証
コード変更後は必ず以下を実行し、エラーゼロであることを確認すること。
```bash
uv run ruff check . --fix
uv run mypy src tests examples
uv run pytest
```
