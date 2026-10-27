# SoICT論文用の独立コード

[英語README](README.md)・[日本語手順](docs/reproduction.ja.md)・[全工程](docs/reproduction.md)

論文の2本SFT・全7条件・本文159項目・全2図4表・統計を生成するコードです。過去の実験結果・性能値・生成済み図表・source ID・モデル重みは配布しません。コードのみのレビュー用リポジトリです。結果・旧ZIP・内部監査ファイルはこのフォルダーの外へ退避しています。

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
PYTHONPATH=src python scripts/run_paper_experiments.py --condition all --plan-output outputs/plan.json
python scripts/make_release.py --check
```

CPU testsは合成データと接続を禁止したmockを使います。NumPyがなければ分析テストは明示的にskipします。runnerは既定でdry-run、実評価には手動serverと明示的な--executeが必要です。

利用者自身のsuiteとtree/training/corpus監査を渡して集計します。7条件、1658 source、5深さ、木・モデル設定、corpus整合、各10000例、固定159項目、必須成果物を検査します。partialと統計skipは不足を明記し、論文完了にはしません。過去値との一致は未比較で、historical_numeric_match=falseです。

生成する主図は3系列×6区分、表はImplementation・tree granularity・Public Top-k・6行ablationです。概念図は2パネルの説明図です。修正Top-kは形式失敗parentに順位を与えずsource分母を保持し、Mean/Medianは有効順位に条件付けます。Top-1はsource内の実際のOR正誤を使います。

Repro Childの過去値と旧CI手順は未確認です。各SFT 100例・各条件17 sourceの単一GPU実行はSDK 1とSGLangで確認しました。論文全量、新規環境、複数GPUの動作は未確認です。縮小実験のAPI再試行上限は計画の1ではなく10でした。正確なtree/checkpointの公開URLは未用意で、再構築は歴史的同一性を保証しません。概念図の配布条件も著者確認待ちです。コードライセンスを図へ自動適用しません。

```bash
python scripts/make_release.py --output /tmp/soict-code-v0.5.1-code-only.zip
```

ZIPを親研究リポジトリ外に展開し、その内容を新規repoの候補にします。このフォルダーにはソース・設定・文書・合成テストを残しています。親研究リポジトリのGit履歴は含めず、新規repoとして扱ってください。過去結果と内部根拠はこのリポジトリに含めません。

Synthetic report / 合成データのreport層確認（実験結果ではありません）:

```bash
PYTHONPATH=src python scripts/check_synthetic_report.py --output-dir outputs/synthetic_report
```

This uses a report-layer mock and 10 bootstrap draws. Without NumPy use `--skip-statistics`; that report remains partial. / NumPyなしでは`--skip-statistics`を使い、partialとして扱います。

主図のSVG/TeXは精度0–100%の全範囲と上端余白・独立した凡例に対応し、欠落値はNAのまま表示します。表示修正はCSV値・統計計算を変更しません。

