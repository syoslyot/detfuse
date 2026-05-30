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
https://colab.research.google.com/github/syoslyot/detfuse/blob/develop/experiments/detector_eval.ipynb?authuser=1
```

> `?authuser=1` 固定保留。平時指向 `develop`，release merge 到 main 後改成 `main`。

## 訓練與測試資料

標記資料位於 `data/categories/free_food/`，由 Claude Code 判斷標記：

| 檔案 | 筆數 | 正例 | 負例 | 用途 |
|------|------|------|------|------|
| `training_data.json` | 894 | 212 | 682 | 訓練 / 調參 |
| `test_data.json` | 234 | 58 | 176 | 評估（Colab 使用這個） |

欄位：`text`、`label`（1 = 免費食物，0 = 非）、`hash`、`query`、`source`

這兩個檔案已加入 git 追蹤（`data/` 其餘仍 gitignore）。
Colab notebook 透過 GitHub raw URL 直接載入，不需手動上傳或掛載 Drive。

> `data/eval.json` 標記不完整（254 筆中只有 4 筆正例），不用於評估。

## 資料集

`data/training_data.json` — 訓練集，格式：
```json
[
  {"text": "研討會多的便當...", "hash": "abc123", "query": "便當", "source": "bento", "label": 1},
  {"text": "免費課程報名...",   "hash": "def456", "query": "免費 活動", "source": "free_event", "label": 0}
]
```

`data/test_data.json` — 測試集，同格式。

| | 筆數 | 正例 | 負例 | 正負比 |
|--|------|------|------|------|
| `training_data.json` | 894 | 212 | 682 | 1:3.2 |
| `test_data.json` | 234 | 58 | 176 | 1:3.0 |

來源：11 個 query，stratified split by (label × source)，seed=42，80/20。

**標注方式**：由 Claude 依語意判斷（非 L1 regex 自標），詳見 [docs/labeling_criteria.md](labeling_criteria.md)。

**注意**：正例比 ~24%，遠高於真實社團（~1–3%）。適合訓練和調參，**不能直接估算真實 Precision**。

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
