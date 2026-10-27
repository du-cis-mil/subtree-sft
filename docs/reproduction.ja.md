# 再現・集計手順

全7条件のコードを提供し、過去実験結果は同梱しません。[英語の全コマンド](reproduction.md)を参照してください。新しい環境と出力先を使います。

1. 元train/test・IR SFTデータとbackboneを用意し、学習木と評価木を別に作ります。再構築のembedding推論は利用者が意図的に実行する段階です。
2. Paper-SelectionとChild-Subtreeの各10000例を作成・監査します。両SFTは同じbackbone、seed42、1epoch、最終checkpoint-3334を使います。
3. configs/models.example.jsonを自身のcheckpoint/served名/endpointへ合わせます。全7条件のdry-run後、意図的な--executeで評価します。serverは手動起動し、逐次serveも可能です。
4. 保存した各suiteをmergeし、自身のtree/training/corpus監査を渡して集計します。

```bash
PYTHONPATH=src python scripts/make_paper_report.py --suite-dir outputs/paper_merged --output-dir outputs/report --tree-summary outputs/tree_granularity.json --training-audit outputs/training_source_audit.json --corpus-audit outputs/corpus_audit.json
```

2図4表、本文の全159項目、macro差、McNemar、bootstrap CIを自身の入力から生成します。原稿性能値との比較は未提供です。出力schemaは全ID・場所・計算先・型を固定hashで検査し、間引き・重複・改変を拒否します。

corpus件数の不整合、予算の不一致、null/bool/float予算はpartialでもエラーです。整合するmapping/深さ不足はpartialのmissingへ記録し、strictでは拒否します。統計skipもpartialです。全7条件・1658source・5深さ・妥当な監査・全計算・全必須ファイルがある場合のみpaper_complete=trueです。historical_numeric_matchは未比較のfalse、publication_readyもfalseを保持します。

形式失敗parentはunrankedとし、Top-k分母を保持、有効parentの最良順位を使います。Mean/Medianは有効順位sourceに条件付け、Top-1は実正誤のORです。bootstrapは5000回・seed42・PCG64・D0–D4順・ID辞書順・linear quantile。旧CIへ合わせる探索はしません。

historical木はconfigsのidentityを検査します。再構築木は明示的に--tree-profile rebuiltを使い、暗黙のfallbackはしません。toyは論文完了になりません。各SFT 100例・各条件17 sourceの単一GPU実行はSDK 1とSGLangで確認しました。論文全量、新規環境、複数GPU動作と正確なtree/checkpoint配布先は未確認、概念図の配布条件も著者確認待ちです。コードレビュー候補です。

主図のSVG/TeXは精度0–100%の全範囲と上端余白・独立した凡例に対応し、欠落値はNAのまま表示します。表示修正はCSV値・統計計算を変更しません。

