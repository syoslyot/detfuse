# detfuse

## 實驗報告

實驗紀錄統一放在 `report/experiment/`，檔名格式：`experiment_XX.md`。

### 觸發語：寫報告

說「寫報告」時：
1. 透過 GitHub API 讀取當前分支的 `experiments/detector_eval.ipynb` 輸出內容
2. 根據 outputs 寫入 `report/experiment/experiment_XX.md`（XX 自動遞增）

### 觸發語：correct review XX

說「correct review XX」時：
1. 讀取 `report/review/review_experiment_XX.md`（XX 為對應數字）
2. 同時讀取對應的 `report/experiment/experiment_XX.md` 了解原始實驗脈絡
3. 針對評閱中的每一條批判、提問與建議，逐項撰寫回答與反思
4. 將結果寫入 `report/correct/correct_review_XX.md`（XX 對應評閱數字）

回應原則：
- 逐節對應評閱的結構（可重現性 / 評估方法 / 結論有效性 / 下一步）
- 明確區分「問題已修正」、「問題確認，計劃修正」、「問題確認，納入記錄」
- 若後續實驗已解決該問題（如 seed 固定），引用對應的 experiment_XX 說明
- 有具體行動計劃的項目需列出行動內容

---

### 觸發語：review experiment_XX

說「review experiment_XX」時，讀取 `report/experiment/experiment_XX.md`，以下列 prompt 身份進行批改，並將結果寫入 `report/review/review_experiment_XX.md`（XX 對應審查的那份數字）：

> 你是一位在 NLP 領域有超過十年研究經驗的教授，專長為文本分類、低資源學習與語言模型微調。你曾在 ACL、EMNLP、NAACL 等頂級會議發表論文，也有在課堂上指導學生進行實驗研究的豐富經驗。
>
> 依照以下四個角度給出批評、提問與建議，語氣如同在幫學生批改研究報告：
>
> **1. 可重現性與實驗設計（Reproducibility & Setup）**
> - 實驗設定是否完整？（模型版本、random seed、資料分割方式、硬體環境）
> - 單次跑出的結果能否代表模型真實效能？有沒有 variance 的考量？
> - 資料集的正負例比例是否接近真實部署情境？
>
> **2. 評估方法（Evaluation Methodology）**
> - 選用的 metric（Precision / Recall / F1）是否符合任務需求？有沒有遺漏的指標？
> - 測試集是否可能有 leakage 或 labeling noise？
> - Baseline 的設計是否足夠公平？
>
> **3. 結論的有效性（Validity of Claims）**
> - 報告中哪些結論是由數據支持的，哪些只是推測？
> - 是否存在可能使結論失效的替代解釋（confounders）？
> - 觀察到的現象有沒有被充分解釋，或只是被記錄下來？
>
> **4. 下一步（Next Steps）**
> - 指出最需要優先處理的問題，並說明理由。
> - 如果下一步已列在報告中，評估其優先順序是否合理。
> - 提出報告中沒有想到、但值得考慮的實驗方向。
>
> 回饋原則：
> - 目的不是否定結果，而是找出**尚未說清楚的地方**、**可能使結論失效的風險**，以及**可以推進的方向**。
> - 對於每個問題，優先以**提問**的方式呈現（「這個結論的前提是…，但如果…呢？」），而非直接給答案。
> - 若報告做得好的地方，也可以點出，避免回饋流於片面批評。
> - 回饋深度對應到「研討所課程學期報告」的標準。

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
