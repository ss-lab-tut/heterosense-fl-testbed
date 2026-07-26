# OTANI_TODO — fall-motion を `main` に統合するための書き換え手順

対象: Git と Python の基礎がある人。むずかしい理論は不要。下の手順どおり直せば通ります。

## なぜ直すのか（1行だけ）
`0469ac3` が追加した乱数呼び出しが**共有の乱数列を消費**してしまい、fall と無関係な
既定出力まで変わりました（＝ v1 互換が壊れた）。**専用の子乱数**を使えば直ります。

## 直し方の型（これだけ覚える）
乱数を使う関数の**冒頭で子乱数を1つ作り**、`0469ac3` が足した呼び出しだけ子乱数に置き換える。
親（既存の `rng`）はそのまま。子は親を消費しないので既定出力は完全一致に戻る。

```python
child = rng.spawn(1)[0]     # メソッド内なら self.rng.spawn(1)[0]
```

---

## 1. 直す箇所（ファイル・行番号つき。ここだけ触る）

### A. `heterosense/_core/_behavior_model.py` — `BehaviorModel.generate()` の ABNORMAL 分岐
`0469ac3` が足した呼び出し（**これらを子に置換**）:
- L324 `mode = float(self.rng.random())`
- L326 `self.rng.integers(2, 4)` / L328 `self.rng.integers(5, 11)` / L330 `self.rng.integers(12, 25)`
- L331–337 `self.rng.choice([...], p=[...])`
- L338 `self.rng.uniform(0.0, 2.0 * np.pi)`

**触らない**（v1 から在る既存呼び出し）: L189, L194, L195, L205, L206, L245, L350。

### B. `heterosense/_core/_observation_model.py`
`0469ac3` が足した呼び出し（**これらを子に置換**）:
- `_sample_fall_variant(ls, rng)`: L174 `rng.choice(len(variants), p=probs)`
- `_generate_abnormal_impact(..., rng, ...)`: L232, L242, L249, L256, L265 `rng.uniform(...)`
- `_generate_lidar` の ABNORMAL レンダリング分岐: L408, L414, L453, L455, L458, L459,
  L465, L467, L469, L470, L476, L482, L483, L486 の `rng.random/choice/uniform`

**触らない**（v1 から在る既存呼び出し）: L136（motion blur）, L148（occlusion）,
L361・L419（background noise）。

---

## 2. 書き換え例（before / after）

### A. メソッド内（`self.rng`）— `_behavior_model.py` の ABNORMAL 分岐
```python
# --- before ---
if current_state != SemanticState.ABNORMAL and next_state == SemanticState.ABNORMAL:
    mode = float(self.rng.random())
    if mode < 0.15:
        abnormal_target = int(self.rng.integers(2, 4))
    ...
    abnormal_motion_pattern = str(self.rng.choice([...], p=[...]))
    abnormal_fall_direction = float(self.rng.uniform(0.0, 2.0 * np.pi))

# --- after ---
if current_state != SemanticState.ABNORMAL and next_state == SemanticState.ABNORMAL:
    child = self.rng.spawn(1)[0]              # ★ この分岐の先頭で1回だけ作る
    mode = float(child.random())              # self.rng -> child
    if mode < 0.15:
        abnormal_target = int(child.integers(2, 4))
    ...
    abnormal_motion_pattern = str(child.choice([...], p=[...]))
    abnormal_fall_direction = float(child.uniform(0.0, 2.0 * np.pi))
```

### B. 引数で `rng` を受け取る関数 — `_observation_model.py`
```python
# --- before ---
def _generate_abnormal_impact(ox, oy, variant, rng, noise_scale):
    ...
    pts[upper_mask, 2] *= float(rng.uniform(0.45, 0.65))
    ...

# --- after ---
def _generate_abnormal_impact(ox, oy, variant, rng, noise_scale):
    child = rng.spawn(1)[0]                   # ★ 関数の冒頭で1回だけ作る
    ...
    pts[upper_mask, 2] *= float(child.uniform(0.45, 0.65))   # rng -> child
    ...
```
`_sample_fall_variant` と `_generate_lidar` の ABNORMAL 分岐も同じ型：
分岐/関数の先頭で `child = rng.spawn(1)[0]` を作り、上の「直す箇所」の行だけ `child.` に変える。

> ポイント: `rng.spawn(1)[0]` は親を**消費しない**ので、fall 以外の出力は v1 と完全一致に戻る。
> fall の中身は子乱数で決まり、多様性はそのまま保たれる。

---

## 3. 確認 → PR（この3コマンド）
```bash
# 1) テスト実行（リポジトリ直下で）
PYTHONPATH=. python -m pytest tests/ -q
# 2) 全部緑になればOK。加えて v1 互換の確認（下記）も通すこと。
#    （赤なら「4. 詰まったら」を見る）
git add -A && git commit -m "fall-motion: use child RNG (rng.spawn) to preserve v1 output"
# 3) PR を作る
git push origin feature/fall-motion
#    表示される GitHub の URL を開き、base=main で Pull Request を作成。
```

v1 互換の確認: v1 既定 config の出力が、固定された v1.0.0 の golden digest
`30c0e2537a8c539edf7e218efa5a2489b93b1173a3042eb0bbf619fe94c567b8` と一致すること。
以前あった `tests/test_v1_compat.py` は現在リポジトリに無いので、この検査は
このブランチ上で作り直す必要がある（比較先は tag `v1.0.0` = `f74a061`。凍結物なので変更禁止）。

合格条件（README_REINTEGRATION.md と同じ）: 全テスト緑＋上の digest 一致＋
fall は LiDAR 系（即時検知）のみに影響。満たせば `main` に統合。

---

## 4. 詰まったら質問してよい箇所（遠慮なくどうぞ）
- **`child` を作る位置に迷う**とき（関数の冒頭か、if 分岐の中か）。原則は「その乱数呼び出しに
  最初に到達する直前」。ABNORMAL 専用の呼び出しは ABNORMAL 分岐の中で作る。
- **どの行が「新規」でどれが「既存」か自信がない**とき（触っていい行の判定）。
  `git diff 0469ac3^..0469ac3 -- <ファイル>` の `+` 行が新規です。
- **golden digest が一致しない**とき。親 `rng` を消費する呼び出しが1つ置換漏れしています。
  上の「触らない/触る」リストと突き合わせてください。
- **`spawn` が無いと言われた**とき（古い numpy）。`pip install -U numpy`（1.25以上が必要）。
