# 實驗紀錄 01 — L1/L2 並行融合基準測試 + LoRA Fine-tune

日期：2026-05-30

---

## 實驗目標

比較以下四種方法在「免費食物貼文偵測」任務上的表現：
1. KeywordDetector（純關鍵字，L1）
2. Qwen2.5-0.5B-Instruct（零樣本 zero-shot，L2）
3. 並行融合（α × L1 + (1-α) × L2，α=0.35，threshold=0.50）
4. Qwen2.5-0.5B LoRA fine-tune 後重做 2 和 3

---

## 方法說明

### L1：KeywordDetector
以 regex 組合「免費詞」+「食物詞」做雙重比對，並帶有排除規則（如「免費課程」）。
輸出 bool，透過 `l1_score()` 轉為 [0, 1] 軟分數供 fusion 使用。

### L2：Qwen2.5-0.5B-Instruct
用 prompt 要求模型輸出 0–9 整數代表信心程度，除以 9.0 得 [0, 1] 分數。
Threshold > 0.5 視為正例。零樣本，未做任何特定訓練。

### 並行融合
```
score = α × L1_score + (1 - α) × L2_score
predict = score > threshold
```
α=0.35（L1 權重較低），threshold=0.50。

### LoRA Fine-tune
用 `training_data.json`（894 筆）對 Qwen2.5-0.5B-Instruct 做 instruction tuning。
- 格式：label=1 → 目標輸出 `'9'`，label=0 → `'0'`
- LoRA：r=16, lora_alpha=32, target_modules=[q_proj, v_proj]
- 訓練：3 epochs, batch_size=4, lr=2e-4, warmup_ratio=0.1
- 框架：純 PyTorch（AdamW + GradScaler）
- 環境：Colab T4 GPU

---

## 資料集

來源：`data/categories/free_food/`，由 Claude Code 標記。

| 資料集 | 總筆數 | 正例（免費食物）| 負例 | 正負比 |
|--------|--------|----------------|------|--------|
| training_data.json | 894 | 212 | 682 | 1 : 3.2 |
| test_data.json | 234 | 58 | 176 | 1 : 3.0 |

> 注意：正例比例（約 25%）高於真實社團 feed（估計 1–3%），
> 精度數字偏樂觀，不代表上線後的真實表現。

---

## 結果

| 模型 | Precision | Recall | F1 |
|------|-----------|--------|----|
| KeywordDetector (L1) | 78.05% | 55.17% | 64.65% |
| Qwen2.5-0.5B zero-shot (L2) | 14.29% | 22.41% | 17.45% |
| Fusion α=0.35 τ=0.50 | 19.79% | 32.76% | 24.68% |
| Qwen2.5-0.5B fine-tuned (L2) | 73.91% | 58.62% | 65.38% |
| Fusion ft α=0.35 τ=0.50 | 73.91% | 58.62% | 65.38% |

---

## 觀察

1. **Zero-shot Qwen 極差**（F1 17.45%）：誤報率極高（FP=78），模型對這個任務幾乎沒有判斷能力。融合後反而拖累 L1，F1 從 64.65% 掉到 24.68%。

2. **Fine-tune 效果顯著**：Qwen F1 從 17.45% 跳到 65.38%，Precision 從 14.29% 升至 73.91%。LoRA 3 epochs 就足以讓模型學會這個 domain 的語境。

3. **Fine-tune 後 Fusion 與 L2 單獨結果完全相同**（73.91% / 58.62% / 65.38%）：代表 fine-tuned L2 主導了所有決策，L1 沒有額外貢獻。可能原因：α=0.35 下 L1 權重不夠、或 L1 與 L2 在正確案例上高度重疊。

4. **Recall 仍是瓶頸**：fine-tuned 後 Recall 58.62%，仍有 41% 的免費食物貼文漏掉。優先改善方向。

---

## 下一步

- [ ] 調整 fusion α 值，看 fine-tuned L2 + L1 能否互補（目前 α=0.35 L1 沒有作用）
- [ ] 分析 FN 案例：哪類貼文 fine-tuned 模型仍漏掉
- [ ] 用更接近真實分佈的測試集驗證（正負比 1:30+）
