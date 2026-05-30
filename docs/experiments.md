# 跑實驗

## 環境設定

```bash
cd laboratory/detfuse
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 確認 Ollama 在跑（另一個 terminal）
ollama serve
ollama pull qwen2.5:0.5b
```

## Colab 實驗環境

notebook 位於 `experiments/detector_eval.ipynb`，托管在 GitHub，可直接從以下連結在 Colab 開啟：

```
https://colab.research.google.com/github/syoslyot/detfuse/blob/feature/parallel-fusion/experiments/detector_eval.ipynb
```

## 資料集

`data/training_data.json` — 583 筆，格式：
```json
[
  {"label": 1, "text": "研討會多的便當...", "hash": "abc123"},
  {"label": 0, "text": "免費課程報名...",   "hash": "def456"}
]
```

| | 筆數 |
|--|------|
| 正例（免費食物）| 350 |
| 負例（不是）| 233 |
| 合計 | 583 |

**注意**：這份資料是用關鍵字篩選爬出來的，正例比例（60%）遠高於真實社團（~1-3%）。
適合拿來**訓練和調整參數**，不適合估算真實精度。
真實精度要等 `eval.json`（無篩選的 feed 資料）建好才能測。

## Grid Search

```bash
python experiments/fusion.py
```

輸出範例：
```
 alpha  thresh      P      R     F1
  0.20    0.40  0.xxx  0.xxx  0.xxx
  0.20    0.50  ...
  ...
```

**解讀**：
- **Recall（R）優先**：漏掉真正的免費食物比誤報更嚴重
- F1 是 P 和 R 的調和平均，作為綜合參考

## 測試 regression

```bash
pytest tests/
```

確保改動 L1 regex 後沒有打壞原有的 test case。

## 把結果套回 freehiero

1. 在 detfuse 找到最佳 α、threshold、或改良的 regex pattern
2. 手動更新 `freehiero/detector.py`
3. 跑 `pytest` 確認 freehiero 的 test 還過
4. commit + PR

## 實驗紀錄

建議在這裡記下每次實驗的結果，方便比較：

| 日期 | 變更 | α | threshold | P | R | F1 | 備註 |
|------|------|---|-----------|---|---|----|------|
| — | baseline（現行 TwoLayerDetector）| — | — | — | — | — | 待測 |
