# detfuse

## 實驗報告

實驗紀錄統一放在 `report/experiment/`，檔名格式：`experiment_XX.md`。

### 觸發語：寫報告

說「寫報告」時：
1. 透過 GitHub API 讀取當前分支的 `experiments/detector_eval.ipynb` 輸出內容
2. 根據 outputs 寫入 `report/experiment/experiment_XX.md`（XX 自動遞增）

## HuggingFace 模型

Fine-tuned LoRA adapter 上傳至：`syoslyot/qwen-detfuse-finetuned`
https://huggingface.co/syoslyot/qwen-detfuse-finetuned

載入方式：
```python
from peft import PeftModel
model = PeftModel.from_pretrained(base_model, 'syoslyot/qwen-detfuse-finetuned')
```

## 訓練與測試資料

標記資料放在 `data/categories/free_food/`，由 Claude Code 判斷標記：

| 檔案 | 筆數 | 正例 | 負例 |
|------|------|------|------|
| `training_data.json` | 894 | 215 | 679 |
| `test_data.json` | 234 | 67 | 167 |

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
- Colab URL 平時指向 `develop`，release merge 到 main 後改成 `main`

## Notebook 修改工作流程

改完 `experiments/detector_eval.ipynb` 後，**立即 commit + push** 到當前分支。
使用者會重新整理 Colab 來驗證改動，必須先看到最新版本。

## Colab 存檔與結果保存

- 跑完實驗後，直接存回**當前 feature 分支**（不需切換到 develop）
- outputs 存在 `.ipynb` JSON 裡，PR merge 時會一起帶進 develop，無需額外處理
- notebook cell 6 的 `BRANCH` 變數必須與當前工作分支一致，否則資料載入 URL 會壞掉
- 開新 feature 分支時，記得同步更新 `BRANCH` 的值
