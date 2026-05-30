# detfuse

## 訓練與測試資料

標記資料放在 `data/categories/free_food/`，由 Claude Code 判斷標記：

| 檔案 | 筆數 | 正例 | 負例 |
|------|------|------|------|
| `training_data.json` | 894 | 212 | 682 |
| `test_data.json` | 234 | 58 | 176 |

欄位：`text`、`label`（1 = 免費食物，0 = 非）、`hash`、`query`、`source`

這兩個檔案已加入 git 追蹤（其餘 `data/` 仍 gitignore）。
Colab notebook 直接從 GitHub raw URL 載入，不需手動上傳。

- `eval.json`（`data/eval.json`）標記不完整，**不使用**。
- SAMPLES 請用 `TEST_SAMPLES`，`TRAIN_SAMPLES` 保留供未來 fine-tune 使用。

## Colab URL 維護

`docs/experiments.md` 裡的 Colab 連結指向當前開發分支，格式：

```
https://colab.research.google.com/github/syoslyot/detfuse/blob/<branch>/experiments/detector_eval.ipynb?authuser=1
```

- `?authuser=1` 固定保留，確保開啟正確的 Google 帳號
- feature 分支 merge 到 main 後，URL 裡的分支名稱要從 `feature/*` 改成 `main`
