# 實驗紀錄 06 — 診斷分析節加入（Sections 12/13）

日期：2026-06-03

---

## 與前次實驗的關係

本次實驗在 `feature/correct-review-analysis` 分支執行，主要目的是回應 review_experiment_02 和 review_experiment_04 的評閱建議，在 notebook 中加入診斷分析節。

基礎訓練設定與 experiment_05 完全相同（SEED=42，完整 test_data.json）。

---

## 本次主要變動（notebook 新增內容）

### Section 12：零樣本模型診斷

| 新增內容 | 說明 |
|---------|------|
| `collect_scores()` | 收集連續分數，供 PR curve 使用 |
| L1 vs Zero-shot L2 分歧樣本分析 | 找出兩者決策不同的樣本，評估零樣本 Fusion 發揮空間 |
| Alpha sweep（α ∈ 0.0–1.0） | 系統性掃描，繪製 F1 vs α 曲線 |
| Precision-Recall Curve（AUC-PR） | 評估類別不均情境下更合適的指標 |

### Section 13：Fine-tuned 模型診斷

| 新增內容 | 說明 |
|---------|------|
| L1 vs Fine-tuned L2 分歧樣本分析 | 確認 Fusion 失效的根本原因：ft L2 輸出 0.0/1.0 極端值 |
| Alpha sweep（ft Fusion） | 確認 Fusion 在所有 α 下的行為 |
| Precision-Recall Curve（fine-tuned） | fine-tuned 模型的 AUC-PR |
| `sequential_detect()` | 實作 Sequential 架構（L1 高信心直接回傳，不確定才呼叫 ft L2），統計 L2 呼叫次數 |

---

## 訓練設定

- Random seed：`42`（固定）
- 模型：Qwen2.5-0.5B-Instruct
- LoRA：r=16, lora_alpha=32, target_modules=['q_proj', 'v_proj'], lora_dropout=0.05
- Trainable params：1,081,344 / 495,114,112（0.2184%）
- Epochs：3，batch_size=4，lr=2e-4
- 環境：Colab T4 GPU

---

## 訓練 Loss

| Epoch | Avg Loss |
|-------|----------|
| 1 | 0.1402 |
| 2 | 0.0515 |
| 3 | 0.0161 |

與 experiment_05 完全一致（SEED=42 固定，結果可重現）。

---

## 結果（評估於完整 test_data.json，234 筆）

| 模型 | Precision | Recall | F1 | TP | FP | TN | FN |
|------|-----------|--------|----|----|----|----|-----|
| KeywordDetector (L1) | 85.37% | 52.24% | 64.81% | 35 | 6 | 161 | 32 |
| Qwen2.5-0.5B zero-shot (L2) | 16.48% | 22.39% | 18.99% | 15 | 76 | 91 | 52 |
| Fusion α=0.35 τ=0.50 | 21.88% | 31.34% | 25.77% | 21 | 75 | 92 | 46 |
| **Qwen2.5-0.5B fine-tuned (L2)** | **98.33%** | **88.06%** | **92.91%** | 59 | 1 | 166 | 8 |
| Fusion ft α=0.35 τ=0.50 | 98.33% | 88.06% | 92.91% | 59 | 1 | 166 | 8 |

結果與 experiment_05 相同，SEED=42 確認可重現。

---

## 觀察

1. **結果與 experiment_05 完全一致**：SEED=42 固定後，模型行為可重現，Loss curve 和 test 結果均相同。

2. **Sections 12/13 尚未執行**：本次保存的 notebook outputs 僅包含基礎訓練評估（Cell 11–25），新增的診斷節（Cells 31–36）需要重新在 Colab 執行後才能取得輸出。

3. **BRANCH 變數需更新**：Cell 4 的 BRANCH 顯示為 `feature/experiment-05-seed-finetune`，已在本次 commit 更新為 `feature/correct-review-analysis`，但輸出尚未反映此變更（需重跑 Cell 4）。

4. **FN 模式持續**：8 筆 FN 主要為：
   - 「營隊剩下的吃的」系列（營隊、電機營、新生營）
   - 含 `#更` 的重複貼文（結構異常）
   - 「學生會辦免費領取」
   - 「颱風天警衛室領取便當」（語氣曖昧）

5. **Fusion ft = fine-tuned L2**：持續問題，待 Section 13 的 alpha sweep 和分歧分析確認根本原因。

6. **FP=1**：「社聯會 ig 追蹤換免費飲料」屬條件性兌換非直接提供，為模型語意邊界的合理難例。

---

## 下一步

- [ ] **重跑 Sections 12/13**：在 Colab 執行診斷分析節，取得 L1/L2 分歧樣本、alpha sweep 圖表、PR curve、Sequential detect 結果
- [ ] **更新 BRANCH 並重跑 Cell 4**：確認資料從正確分支載入
- [ ] **多 seed 實驗**：SEED=42 基礎上再跑 SEED=1 和 SEED=2，估算 variance
- [ ] **FN 資料增強**：針對「營隊」、「#更 重複貼文」類型加入更多 hard positive 訓練範例
