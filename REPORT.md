# REPORT.md — HeteroSense-FL v2 ＋ FL観測層研究（旗艦）

実行: 2026-07-04, branch `v2-dev`。v1 = tag `v1.0.0` (f74a061)。
再現: `PYTHONPATH=. python experiments/fl_observation_layer/run_flagship.py --n_per_cell 4 --n_nights 8`
→ `figures.py`。全 seed 固定 (2031)。

---

## 基盤C用の一文（実測値で完成）

> **観測モデルの誤差 約1秒あたり検知遅延 約1.7分の代償**（測定域 8–74秒での線形近似；
> exit-timing 誤差 8秒→74秒 で代償 0.2分→113分に跳ねる；単一PIRは長期臥床の多くを取りこぼす）。
> **モダリティ群別FLは素朴FedAvg比で代償を 78%圧縮**（中央動作点＝サポート欠損比0.3）。
> **PIR×1＋圧力は PIR×N（数の増設）と比べ、被覆率0.8回復に対し同じ+1センサで recall 1.0 を
> 達成し〈安い〉**（種類＝在床直接観測による回復）。

（注: 「1.7分/秒」は測定2点 8秒/0.2分・74秒/113分 を結ぶ線形近似であり、代償は取りこぼし
崖のため実際は非線形。欠損比を上げた極限の圧縮率 99.9% は §1.1 の表のみに置き、本文の主数値は
中央動作点の 78% とする。）

---

## 用語の凍結（以後この2語を混用しない）

- **anchor recall** ＝ 理想ギャップ抽出器 `_extractor.b2t_recall` の値。現値 **0.179**
  （固定seed厳密値、`tests/test_anchor.py::test_extractor_anchor_exact_no_drift` が ±0.005 で固定）。
  61軒実測レンジ 0.03–0.55 との整合は別テスト（`test_extractor_anchor_in_61home_range`）。
- **learned-model recall** ＝ FL 観測モデルのイベント一致 recall（`study.evaluate` の match）。
  中央値 **0.25**（`results.csv`, pir_only）。バランス済み分類器が過検出するため抽出器値より高い。
- フェーズB報告の 0.10–0.14 は抽出器値（flicker 追加前 0.138）。flicker 追加で共有 RNG 消費が
  後続夜の B2T 継続時間標本を変え、抽出器 anchor が 0.138→0.179 に移動（CHANGELOG 記載）。
  両値（抽出器 0.179 / 学習 0.25）とも 61軒レンジ内で整合は維持。

---

## 1. 全系列の結果表（良い方式だけの報告は禁止＝全方式掲載）

α=0.1。cost = オラクル比 検知遅延代償（分, 中央値）。24軒/系列。

| series | method | F1 | timing_err(min) | recall | cost(min) |
|---|---|---|---|---|---|
| pir_only | local | 0.083 | 1.042 | 0.333 | 112.3 |
| pir_only | centralized | 0.084 | 1.108 | 0.281 | 113.1 |
| pir_only | fedavg | 0.086 | 1.350 | 0.146 | 113.1 |
| pir_only | modality_group | 0.080 | 1.225 | 0.177 | 111.5 |
| pir_pressure | local | 0.407 | 0.142 | 0.906 | 0.16 |
| pir_pressure | centralized | 0.203 | 1.308 | 0.646 | 1.55 |
| pir_pressure | fedavg | 0.270 | 0.738 | 0.719 | 0.88 |
| pir_pressure | modality_group | 0.273 | 0.100 | 0.833 | 0.19 |
| lidar_upper | local | 1.000 | 0.083 | 0.979 | 0.08 |

**読み**: PIR単独は観測誤差が大きく（timing~1分, recall~0.25）代償 ~113分。圧力併用で
recall~0.9・代償~0.2分に回復（種類）。LiDAR上界は代償~0.08分＝分位点限界の床。
centralized が pir_pressure で local より悪い（1.55 vs 0.16）＝単一集中モデルは
モダリティ異質性下で圧力を専門化できない、という正直な負の知見。

### 1.1 ①(missing support) スイープ — 設計仮説の本検証
圧力（希少状態サポート）を欠く家の比率を振り、pressure 保有家での代償を測定:

| 欠損比 | FedAvg | modality_group | 方式4 優位 |
|---|---|---|---|
| 0.0 | 0.138 | 0.121 | +0.02 |
| 0.3 | 0.933 | 0.117 | +0.82 |
| 0.5 | 1.492 | 0.117 | +1.38 |
| 0.7 | **113.0** | **0.10** | **+112.9** |

欠損比が上がるほど素朴FedAvg は共有圧力重みが希釈され代償が爆発、群別共有は不変。
**方式4（モダリティ群別部分共有）が本命として成立**（BeliefWatch 設計仮説を実証）。

## 2. 旗艦図・副図
- `experiments/fl_observation_layer/fig_flagship.{pdf,png}`: 横=exit timing 誤差(分),
  縦=α=0.1 オラクル比代償(分)。系列色×方式マーカー。**LiDAR上界を水平参照線**（代償≈0.08分＝
  観測を完全にしても残る分位点限界の床）。
- `fig_subfigure.{pdf,png}`: `bedroom_sensor_count × refractory_s` 平面の <1分B2T 被覆率、
  0.8 等高線（count≥2 で回復）、**PIR×1＋圧力の動作点（count=1 で recall 1.0）**を重畳
  ＝「数 vs 種類」を一枚で対照。

## 3. (d) 相関係数（決定基準は描画前に凍結）
- Spearman(timing_err, cost) = **0.642**
- Spearman(F1, cost) = **−0.612**
- |0.642| > |0.612| → **主図 x軸 = timing error**、F1 は付録。（見栄えでの選択を排除, `correlations.txt`）

## 4. 鉄則遵守（§4.3, longlie_study 接続は import のみ・無改造）
- longlie_study 接続点: `study.py:downstream_delay_min` が
  `ll_injection.calendar_split` ＋ `ll_injection.delay_fa_curve`（鉄則4箇条は longlie 内で保証,
  当該行番号は longlie REPORT §2 参照）を呼ぶ。longlie は `sys.path` 追加のみで**無改造**
  （`study.py:21-22, 29`）。
- しきい値 t* はオラクル（真の潜在状態）の戻り時間分位点で固定（`study.py:evaluate` 内 t_star）＝
  観測モデルの誤差のみを代償として計上（degraded モデルが自分の予測から t* を緩める抜け道を封じた）。
- 制約遵守（§4.2.5）: 新規実装は PIR モダリティのみ。圧力/LiDAR は v1 `_generate_pressure`/
  `_generate_lidar` を**無改造**で使用（`study.py:_v1_pressure_feature/_v1_lidar_feature`）。

## 5. sim-to-real 較正表（較正した/しなかったの峻別, §1-1）
| 量 | 較正? | 出所 / 値 |
|---|---|---|
| B2T 往復時間分布 | **較正済** | 61軒実測スナップショット（n=3996, median 1.93分, hash a3fe5e91）。DOI 15708568 |
| 離床頻度 (n_exits/夜) | 未較正 | 仮定 6/夜（文献レンジの代表値） |
| 在床フリッカ周期 | **未較正** | 仮定 150秒（就床中の体動間隔。アンカー機構に必須だが実測未接地） |
| 抽出器 base_gap | 較正済(移転) | longlie の G=5分（最良値）を移植 |
| PIR 不応期/報告周期 | 設計変数 | ノブとして探索（{2..60}s / {0..60}s） |
| 圧力/LiDAR 分離度 | v1 由来 | v1 観測モデルの既定描画（無改造） |
| morning-discovery cap | 仮定 | 120分（取りこぼし長期臥床の朝発見） |

## 5.5 行動/観測の RNG 分離（検証済, `tests/test_behavior_invariant.py`）
- **潜在状態軌跡はセンサ構成に不変**: 同一 seed で PIR 有効/無効の LatentState 系列は
  byte 一致（BehaviorModel.generate は観測前に走り PIR フィールドを読まない）。
  → 副図（数 vs 種類）は**同一行動実現を異なるセンサ構成で観測する対応比較（paired）として成立**。
  旗艦図は**系列間も paired**（home k は全系列で同一 behavior seed=SEED+k·7、24/24 軒で
  true_events 一致を確認）＋系列内 4 方式も同一 homes で paired。行動不変性がこれを保証。
- **対応 vs 非対応の頑健性（確認済）**: 系列を独立 seed（非対応）にしても、系列コスト順位
  `pir_only(≈112分) ≫ pir_pressure(≈0.2–0.9分) > lidar_upper(0.08分)` と方式順位
  `modality_group < fedavg` は**双方で不変**。中央値は pir_pressure 内でのみ別 home draw により
  小変動（例 modality_group 0.19↔0.75）するが結論は不変。**paired を正典**とする。
- **圧力は LiDAR の有効/無効に不変でない（事実として記録）**: observe() が lidar→pressure の順で
  共有 rng を消費するため、lidar 無効時は pressure の rng 状態がずれ全フレーム変化（非対称：
  lidar は pressure 有無に不変＝先に生成）。v1 由来の設計で **v1 互換の範囲内**（固定構成では v1 と
  一致）。フェーズC は各モダリティ特徴を単一チャネルの ObservationModel で計算するため本研究の
  結果には影響しない。真の分離が要る場合は §7 の子RNG化（fall-motion と同型）で解消可能。

## 6. アンカー・恒等式のトリップワイヤ（全て通過）
- 単一PIR recall = 0.25（61軒実測 0.03–0.55 の範囲内）→ アンカー整合。
- LiDAR上界 代償 = 0.08分（≥0）→ 恒等式（観測を良くしても遅延は分位点で下げ止まる）整合。
- 全 cost ≥ 0（オラクルより速い検知は無し）→ 恒等式違反なし。

## 7. 限界一覧
1. **sim-to-real**: 示せるのは傾向と機構であり実性能ではない。在床フリッカ・離床頻度・
   morning cap は未較正の行動仮定（表5）。B2T 時間分布のみ実接地。
2. **観測モデルの単純化**: ロジスティック回帰＋窓特徴（tsl / 平均圧力 / LiDAR高さ）。
   深いエンコーダ・時系列モデルは未使用。潜在状態は {在床, 離床中} の2値に縮約。
3. **下流注入の簡略化**: 代償は「各真の離床を長期臥床起点とみなす」per-event 近似。
   longlie の完全な train/eval 分割注入ではなく、t* のみ longlie で算出。鉄則の精神は保持
   （観測誤差のみ計上・オラクル固定 t*）だが、厳密な注入プロトコルの全再現ではない。
4. **①検証の範囲**: 圧力欠損＝サポート欠損の代理。危険近傍希少状態の直接欠損ではない。
5. **スケール**: 24軒/系列・8夜。N を増やせば分散は縮むが機構は不変の見込み。

## 8. 実装・再現
- v2.0 コア: `heterosense/_core/{_pir_model,_b2t,_extractor}.py`, `_data/b2t_snapshot.json`,
  `tools/make_b2t_snapshot.py`。ノブ: `ClientConfig.{bedroom_sensor_count,refractory_s,report_period_s}`。
- 研究コード: `experiments/fl_observation_layer/{study,run_flagship,figures}.py`。
- テスト: 43 passing（`test_anchor` 4 ＋ `test_v1_compat` 2 ＋ v1 の 37）。
- 較正表・全結果 CSV: `experiments/fl_observation_layer/{results,missing_support,coverage_grid}.csv`。
