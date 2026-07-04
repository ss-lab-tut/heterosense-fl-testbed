# EXTENSION_MAP.md — HeteroSense-FL v2 拡張設計（フェーズA成果物・PI承認待ち）

作成: 2026-07-04, branch `v2-dev`。v1 = tag `v1.0.0` (commit f74a061)。
本書は v1 の構造マップと、3ノブ（時間分解能・被覆・部屋形状）の挿入設計、
および**後方互換の差分方針**を示す。実装（フェーズB）は PI 承認後に着手する。

---

## 0. 最重要の設計判断（PI 承認事項）

**v1 のセンサパラダイムはミッションの前提と異なる。**

- v1 が模擬するのは **LiDAR 点群 + 圧力マット**（連続フレーム, 意味状態
  ABSENT/STATIONARY/WALKING/TRANSITION/ABNORMAL, 単一無差別ルーム）。
- ミッション §3.1 の 3ノブは **PIR の物理**（`refractory_s`=不応期,
  `report_period_s`=報告集約, `bedroom_sensor_count`=寝室 PIR 数）。
- 検証アンカー §1-2/§3.1 は **CASAS の PIR**（寝室単一 PIR で 1分未満 B2T が観測不能）。

v1 には **PIR モダリティ・トイレ・部屋間移動（B2T）・不応期の概念が存在しない**
（Explore 実測: `_behavior_model.py` は単一 5×5m ルームのランダムウォーク、
目的地指向の移動なし; `_observation_model.py` は毎フレーム LiDAR/圧力を描画、
イベントタイムスタンプ・不応期なし）。

→ **v2 はパラメータ追加では足りない。以下 2 つの新規・追加要素が必須:**
1. **PIR/モーション・モダリティの追加**（`refractory_s`/`report_period_s` が物理的に
   意味を持つ唯一の場。バイナリ発火列を占有から生成）。
2. **複数部屋ジオメトリ + B2T 行動**（寝室↔トイレ遷移。戻り時間分布を CASAS 実測に較正し、
   単一 PIR で 1分未満 B2T が観測不能、を再現＝アンカー検証の前提）。

いずれも **追加（additive）** であり、既定値で v1 挙動を完全再現する（後方互換）:
PIR モダリティ既定 OFF、`room_count` 既定 1、`refractory_s` 既定 0。
**この方針で進めてよいか（承認事項1）。** 代替案は §5 に記す。

---

## 1. v1 アーキテクチャ（3層）

```
Interface: ClientFactory · ConfigurationManager · DatasetBuilder · TemporalWindowSampler · run_validation
Latent:    BehaviorModel  -> LatentState(state,x,y,velocity,posture,bed_zone,abnormal_*)   [_behavior_model.py]
Observe:   ObservationModel -> ModalityBundle(lidar:(N,3), pressure:(16,16), labels)        [_observation_model.py]
Config:    SimConfig{delta_t,n_steps,room,sensors,perturbation,clients:[ClientConfig]}      [_config_schema.py]
Public API (heterosense/__init__): ClientFactory, ConfigurationManager, DatasetBuilder,
  TemporalWindowSampler, ModalityBundle, run_validation, BehaviorModel, LatentState,
  SemanticState, Posture, BedZone, AbnormalType, SimConfig, ClientConfig
Entry points: heterosense-demo, heterosense-benchmark; FL loop in _scripts/run_benchmark.py (FedAvg, TinyMLP)
Tests: 37 (ClientFactory, DatasetBuilder, TemporalWindowSampler, validation V1–V4)
```

**現状で「被覆/ジオメトリ」に相当するもの**: 単一ルーム(room_width×height),
円形 bed ゾーン(ON/EDGE/OFF), 単一 LiDAR 位置, `lidar_occlusion`(一様ランダム脱落のみ,
空間的 FOV でない)。多部屋・センサ密度・視野死角は無い。

**B2T タイミング**: 未モデル化（トイレ無し, 目的地指向移動無し）。

---

## 2. 3ノブの挿入設計

### 2.1 時間分解能ノブ（核）— 新規 PIR モダリティに実装
- **新規**: `heterosense/_core/_pir_model.py`（新ファイル）。占有(state≠ABSENT かつ
  部屋内)から PIR バイナリ発火を生成。パラメータ:
  - `refractory_s ∈ {2,5,10,30,60}`（発火後の不応期。同センサの再発火を抑止）
  - `report_period_s ∈ {0,10,30,60}`（発火を周期に量子化）
  - `bedroom_sensor_count ∈ {1,2,3}`（寝室 PIR 数; 1=CASAS 典型）
- **接続点**: `ObservationModel.observe()`（_observation_model.py:315–327）から
  PIR チャネルを条件付き呼び出し。`ModalityBundle` に `pir: Optional[dict[sensor_id, list[event_ts]]]`
  フィールドを追加（既定 None → v1 出力不変）。
- `ClientConfig` に上記3項目を追加（既定: refractory_s=0, report_period_s=0,
  bedroom_sensor_count=0 → PIR 無効 = v1 挙動）。

### 2.2 被覆ノブ — PIR 視野 + 既存 occlusion 拡張
- PIR: `pir_fov_coverage ∈ [0,1]`（部屋の被覆率）, `pir_blind_spot_rate`（空間死角）。
  `_pir_model.py` 内で占有位置が被覆域外なら発火せず。
- LiDAR/圧力: 既存 `lidar_occlusion` に加え `pressure_coverage_fraction`（bed 面の一部のみ描画）。
  `_generate_lidar/_generate_pressure`（_observation_model.py:159–310）に空間マスク追加。
- 既定値は全て「全被覆・死角0」→ v1 不変。

### 2.3 部屋形状ノブ — 多部屋 + B2T 行動
- **`_config_schema.py`**: `SimConfig.rooms: list[RoomConfig]`（既定 1 室）,
  `ClientConfig.room_count`(既定1), `room_topology`(隣接距離 m), `bed_toilet_distance_m`。
- **`_behavior_model.py`**（`_init_position`/`_update_position`, 189–212）:
  `LatentState` に `room_id` を追加（既定 0 → 単一室と同一）。`room_count≥2` のとき
  TRANSITION を寝室→トイレ→寝室の往復として生成し、**戻り時間 = 距離/歩行速度 + トイレ滞在**を
  CASAS 実測 B2T 分布（中央値<1分, longlie_study 由来）に較正。
- ObservationModel は `room_id` を見て当該部屋のセンサのみ描画。

### 2.4 挿入点サマリ
| ノブ | 主ファイル | 変更 | 既定=v1 |
|---|---|---|---|
| 時間分解能 | `_pir_model.py`(新), `_observation_model.py:observe` | PIR 発火+不応期+量子化 | PIR OFF |
| 被覆 | `_pir_model.py`, `_observation_model.py:_generate_*` | FOV/死角/圧力被覆マスク | 全被覆 |
| 部屋形状 | `_config_schema.py`, `_behavior_model.py:_update_position` | 多部屋+B2T 往復 | room_count=1 |

---

## 3. 後方互換の差分方針（回帰テストで保証）

1. **追加のみ・既存 API 不変**: 新モジュール（`_pir_model.py`）＋ `ClientConfig`/`SimConfig`
   の**新フィールド（全て既定値付き）**。既存クラス/関数のシグネチャ・戻り値型は不変。
   `ModalityBundle` は `pir` フィールドを追加するが既定 None（既存の lidar/pressure 消費側は無影響）。
2. **新ノブ既定 = v1 挙動**: refractory_s=0, report_period_s=0, bedroom_sensor_count=0,
   pir 無効, room_count=1, 全被覆。
3. **回帰テスト（フェーズB合格基準）**: `tests/test_v1_compat.py` を新設。
   v1.0.0 の既定 SimConfig とデフォルト ClientFactory 出力を **seed 固定で凍結**し、
   v2 コードが**バイト同一（または数値完全一致）**の `{client_id:[ModalityBundle]}` を生成することを検証。
   凍結参照は `git show v1.0.0:` から生成した黄金ファイル（`tests/golden/v1_default_*.npz`）。
4. **既存 37 テスト + V1–V4 検証を無改変で通す**。

---

## 4. フェーズC（FL観測層）への接続（設計メモ, 実装はC）

- 学習対象: `p(発火列窓 | 潜在状態; c_h)`。潜在状態最小集合 {在床, 離床中, 復帰}
  ← `LatentState.bed_zone`/`room_id` から導出。
- longlie_study への接続（**import のみ・無改変**, §1）: 学習済み観測モデルが推定した
  離床/復帰時刻列 → 疑似エピソード(start,return_min) → `longlie_study/src/injection.py`
  の `delay_fa_curve`（鉄則4箇条そのまま）へ。オラクル（真の潜在状態）との遅延差 = 観測誤差の代償。
- 生イベント列は家から出さない（BeliefWatch 設計継承）: 共有はモデル更新のみ、
  個人律動 prior（分位点/S(t)）はエッジ局所。

---

## 5. 代替案（承認判断の材料）

- **代替A（最小）**: PIR を追加せず、既存 LiDAR/圧力に report_period/refractory を適用。
  → 却下推奨: refractory は PIR 固有物理で LiDAR に意味薄く、**アンカー（CASAS=PIR）を
  再現できない**。ミッション §1-2 の検証アンカー要件を満たさない。
- **代替B（本設計・推奨）**: PIR モダリティ＋多部屋 B2T を追加。アンカー再現可能。追加のみで後方互換。
- **代替C（分割）**: v2.0=時間分解能ノブ(PIR)のみ先行リリース、被覆/形状は v2.1。
  → スコープを縮小したい場合の選択肢。

**承認事項**: (1) 代替B で進めてよいか。(2) スコープは 3ノブ一括(v2.0)か分割(C)か。
(3) B2T 較正に longlie_study の実測 B2T 分布を使う（import のみ）ことの可否。

---

## 6. 決定記録（2026-07-04, PI 承認済み）

- **スコープ = 代替C（分割）**: **v2.0 は時間分解能ノブ（PIR モダリティ）のみ**。被覆・部屋形状は
  v2.1 へ延期（理由: 論文の急所=アンカー再現＋副図は時間分解能ノブ単独で閉じる。9月拘束下で
  「リリース可能な v2.0 に最短到達」を優先。リリースはタグ+DOIでコストほぼ0のため分割にペナルティ無し）。
- **B2T 較正 = スナップショット凍結**（下記条件付き）:
  - `tools/make_b2t_snapshot.py` を同梱（longlie_study があれば1コマンド再生成）。
  - 来歴埋め込み: longlie_study commit `0baaf8c`, 元データ DOI 10.5281/zenodo.15708568,
    生成日 2026-07-04, CC-BY-4.0 帰属。→ `heterosense/_data/b2t_snapshot.json`。
  - アンカー検証テストは **data-only hash `a3fe5e91…` に固定**。更新は CHANGELOG 記載の明示操作のみ。
  - スナップショット実測: n=3996(60軒), median 1.93分, sub-minute 27%。

## 7. v2.1 拡張性コントラクト（分割リリースの唯一の事故=後から挿せない設計、を防ぐ）

被覆・部屋形状ノブは **実装しないが挿入点を確保**。以下を v2.0 実装で満たした:
- **PIR クラス（`_pir_model.py`）**: `PIRSensor` に `position/fov_coverage/blind_spot_rate/room_id`
  フィールドを既定=全被覆で保持。発火ループは各モーション標本を `_covers(rng)` に通すため、
  v2.1 被覆ノブは**これらフィールドを埋めるだけ**で有効化（ループ再設計不要）。
- **ジオメトリ seam**: `observe_sequence` は潜在状態の `room_id` を `getattr(默认0)` で読む設計。
  `LatentState.room_id`（既定0）を v2.0 で追加済み → v2.1 多部屋は room_id を分岐させるだけ。
- **B2T seam**: v2.0 は away-duration をスナップショットから直接標本化（`_b2t.py`）。v2.1 ジオメトリは
  同じ `generate_b2t_night` の duration 供給を「距離/歩行速度から導出」に差し替えるだけ。
- **抽出器（`_extractor.py`）**: `effective_gap_s` が `sensor_count>=2` で不在確定＝gap 縮小、を既に分岐。
  v2.1 被覆はこの分岐に FOV/死角を加えるだけ。副図（<1分B2T回復の最小センサ追加）はここで閉じる。
- **Config**: v2.1 フィールド（pir_fov_coverage 等）を `ClientConfig` コメントに予約済み（§ClientConfig）。

→ **後から挿せない設計になっていないことを確認済み**（PI 条件充足）。

## 8. フェーズB 実装済み（v2.0, tests 43 passing）

- `heterosense/_core/_pir_model.py`（PIRModel: refractory/report_period/count, 別 rng で v1 不撹乱）
- `heterosense/_core/_b2t.py`（B2TSnapshot 逆CDF標本化 + B2T 夜生成）
- `heterosense/_core/_extractor.py`（注釈非依存 B2T 抽出器・機構ベース）
- `heterosense/_data/b2t_snapshot.json`（凍結スナップショット）+ `tools/make_b2t_snapshot.py`
- `ClientConfig` に PIR ノブ3項目（既定=無効=v1）
- `tests/test_anchor.py`（4: hash固定・単一PIR recall 0.10–0.14 が 61軒範囲, report_period 単調, 2sensor回復）
- `tests/test_v1_compat.py`（2: v1.0.0 golden digest `30c0e25…` とバイト一致）
- **アンカー再現**: 単一 PIR recall≈0.10–0.14（61軒実測 0.03–0.55 の範囲内）。合わせ込みなし
  （base_gap=300s=longlie の G=5分）。

## 9. フェーズC 設計（§4.2.5 新設・§4.4 改訂 反映。実装はフェーズC）

### 9.1 モダリティ構成軸 3系列（学習方式1〜4と直交）
| 系列 | 実装 | 役割 | 新規実装? |
|---|---|---|---|
| **PIR単独**（主軸） | 新規 `_pir_model.py` ＋ 時間分解能ノブ | 主結果の変動源 | PIR のみ新規 |
| **PIR＋ベッド圧力**（対抗軸） | 新規 PIR ＋ **v1 既存 `_generate_pressure` を無改造で併用** | 「数(PIR×2) vs 種類(PIR×1＋圧力)」の被覆回復対照。**方式4 の本検証系列**（PIR群/圧力群の本物の異質性） | 圧力は改造禁止 |
| **LiDAR上界**（アンカー系列, 1本） | **v1 既存 `_generate_lidar` を無改造** | 観測品質上限での遅延＝分位点限界の可視化 | LiDAR は改造禁止 |

**制約遵守（§4.2.5）**: 新規実装は PIR モダリティのみ。`_observation_model.py` の
`_generate_lidar`/`_generate_pressure` は**無改造**（フェーズBで未変更を確認済み）。
圧力/LiDAR 系列は v1 の観測をそのまま観測層学習・下流評価に供給する。

### 9.2 学習方式 × 系列の格子（§4.2）
- 方式1 Local-only（下界）／2 Centralized（上界・制約違反参照）／3 FedAvg 素朴（全層平均）／
  4 モダリティ群別部分共有（観測層をモダリティ/構成群で共有・構成固有層はローカル）。
- **方式4 の本検証は「PIR＋圧力」系列**で行う（PIR群と圧力群という実異質性が前提）。
- **①(missing support) 検証**: 危険近傍の希少状態サポートを欠く家の混在比率を振り、方式3 vs 4 の
  バイアス差を測る（PIR＋圧力系列で、圧力欠損家 = サポート欠損の実体）。

### 9.3 評価二層（§4.3）と longlie_study 接続（import のみ・無改造, §1）
- 観測層直接: 潜在状態 {在床,離床中,復帰} 推定の F1・タイミング誤差 |Δt|（離床/復帰時刻）。
- 下流(旗艦): 学習済み観測モデルの推定離床/復帰列 → 疑似エピソード → `longlie_study/src/injection.py`
  の `delay_fa_curve`（鉄則4箇条そのまま）。オラクル（真の潜在状態直接観測）との遅延差＝観測誤差の代償(分)。
  α=0.1 主・{0.05,0.2} 副。longlie_study は **import のみ**（`sys.path` 追加）で無改造。

### 9.4 旗艦図（§4.4 改訂）
- 横軸: 観測モデルのタイミング誤差（と F1。両方描き情報量の多い方を主図）。
  変動源 = 3ノブ × 学習方式1〜4 × モダリティ3系列 の格子。
- 縦軸: α=0.1 での検知遅延の**オラクル比代償（分, 中央値＋IQR）**。
- **LiDAR上界系列を水平参照線**として重ね、「観測品質上限でも残る遅延＝分位点限界」を注記。

### 9.5 副図（§4.4 改訂）— 数 vs 種類
- 平面: `bedroom_sensor_count` × `refractory_s`。**1分未満B2T の被覆率 0.8 等高線**を描く。
- **PIR×1＋圧力の動作点を同平面に重ねる**（圧力による「種類」回復を、PIR 増設「数」回復と一枚で対照）。
- 機構は既実装の `_extractor.effective_gap_s` の `sensor_count>=2` 分岐を土台に、
  「PIR×1＋圧力」＝在床直接観測で不在確定 → gap=report 分解能、として動作点を置く
  （フェーズC で圧力確定経路を追加。PIR モダリティ側の改造は不要=既存 seam で閉じる）。

### 9.6 承認事項（この設計で進めてよいか）
(1) 9.1 の3系列実装方針（PIR のみ新規, 圧力/LiDAR は v1 無改造併用）。
(2) 方式4 を「PIR＋圧力」系列で本検証, ①検証を同系列の圧力欠損で。
(3) longlie_study を import のみ・無改造で下流評価に接続。
(4) 旗艦図の横軸主図は「タイミング誤差 or F1」を両描画後に決定（暫定=タイミング誤差）。

