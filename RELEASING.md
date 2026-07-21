# RELEASING.md — HeteroSense-FL リリース工学（v1 記録・v2 手順）

## v1 の凍結（確認済み, 2026-07-04）

| 項目 | 値 |
|---|---|
| リリースタグ | `v1.0.0` → commit `f74a061cab51473958a7b452ec8005f7e2e60452` (2026-03-30) |
| GitHub | https://github.com/ss-lab-tut/heterosense-fl-testbed （tag `v1.0.0`） |
| **version DOI** | **10.5281/zenodo.19326703**（この版を引用） |
| **concept DOI** | **10.5281/zenodo.19326702**（全版・恒久。論文で「all versions」に使用） |
| Zenodo 記録 | version=v1.0.0, pub 2026-03-30, `isSupplementTo` github `tree/v1.0.0`, |
|  | file=`ss-lab-tut/heterosense-fl-testbed-v1.0.0.zip` |
| CITATION.cff | doi=10.5281/zenodo.19326703, version 1.0.0 |

**タグ↔DOI 対応**: Zenodo 19326703 の `related_identifiers` が github `tree/v1.0.0` を
`isSupplementTo` として指す → タグ v1.0.0 と version DOI が一対一で確認できた。
アーカイブ実体は GitHub Release の zip（下記の連携が v1 時点で有効だった証拠）。

## Zenodo–GitHub 連携（有効）

v1 のアーカイブが GitHub Release 由来の zip であることから、
**Zenodo–GitHub Webhook 連携は有効**と判断。→ v2.0.0 の GitHub Release を作成すると、
concept 10.5281/zenodo.19326702 の下に**新しい version DOI が自動発行**される。
人間の作業は「GitHub Release を publish する」ことのみ（Web UI, §v2 手順4）。

（万一連携が無効化されていた場合の手動手順: Zenodo にログイン →
GitHub 連携でリポジトリを有効化 → 新規 Release を publish → DOI 自動発行を確認。3手順。）

## v2 リリース手順（フェーズD, 実装完了後）

1. `v2-dev` で回帰テスト（`tests/test_v1_compat.py`）＋全テスト＋V1–V4 を green に。
2. `CHANGELOG.md` の `[2.0.0]` を確定（3ノブ・新設定・後方互換保証範囲・既知の限界）。
3. `README.md` を v2 対応に更新（新ノブ使用例, v1 再現= `git checkout v1.0.0` を明記）,
   `CITATION.cff` の version を 2.0.0 に更新。
4. `v2-dev` → `main` マージ、タグ `v2.0.0`、GitHub Release 作成
   （リリースノートは CHANGELOG から生成）。**← 人間が Web UI で publish（外向き操作）。**
5. Zenodo が新 version DOI を自動発行 → 本ファイルに追記。連携無効なら上記手動3手順を依頼。
6. 旗艦論文 Methods 用英文雛形 → 下記「Methods 用 英文雛形」を参照（v2.0 は
   temporal-resolution ノブのみ。coverage/geometry は v2.1）。

## v2 DOI（発行後に追記）
_(pending — フェーズD の GitHub Release publish 後)_

## v2.0.0 リリース — 残りの人間作業（外向き操作。自動ツールでは実行しない）

ローカル準備完了（v2-dev で全実装・全テスト green、CHANGELOG/README 確定、
下記の通りローカルで main へマージ＋タグ v2.0.0 済み・**未push**）。人間の Web UI 操作:

1. `git push origin main --tags`（公開リポジトリへ push。外向き・不可逆）。
2. GitHub で **Release v2.0.0 を publish**（リリースノートは CHANGELOG [2.0.0] から生成）。
3. Zenodo–GitHub 連携が新 **version DOI** を concept 10.5281/zenodo.19326702 の下に自動発行。
   発行後、その DOI を本ファイル「v2 DOI」欄と下記 Methods 雛形に記入。

## 旗艦論文 Methods 用 英文雛形（DOI はリリース後に確定）
> Simulations use HeteroSense-FL v2.0 (DOI: 10.5281/zenodo.XXXXXXXX), a backward-compatible
> extension of HeteroSense-FL v1 (Shao et al., 2026; DOI: 10.5281/zenodo.19326703) adding a
> temporal-resolution (PIR) control. The bed-to-toilet return-time distribution is calibrated
> to 61 community homes from the CASAS corpus (Zenodo 10.5281/zenodo.15708568, CC-BY-4.0).
> All-versions DOI: 10.5281/zenodo.19326702.
