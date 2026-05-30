# 實驗紀錄 04 — Loss Masking 修正後第二次跑（HuggingFace 上傳成功）

日期：2026-05-30

---

## 與 experiment_03 的關係

與 experiment_03 使用相同的程式碼（loss masking 已修正），屬於第二次獨立執行。
本次 HuggingFace 上傳成功（experiment_03 上傳失敗）。

結果因訓練隨機性（batch shuffle、weight init）與 experiment_03 有差異，詳見比較節。

---

## 訓練 Loss

| Epoch | Avg Loss |
|-------|----------|
| 1 | 0.1433 |
| 2 | 0.0561 |
| 3 | 0.0228 |

---

## 結果

| 模型 | Precision | Recall | F1 |
|------|-----------|--------|----|
| KeywordDetector (L1) | 78.05% | 55.17% | 64.65% |
| Qwen2.5-0.5B zero-shot (L2) | 14.29% | 22.41% | 17.45% |
| Fusion α=0.35 τ=0.50 | 19.79% | 32.76% | 24.68% |
| **Qwen2.5-0.5B fine-tuned (L2)** | **85.29%** | **100.00%** | **92.06%** |
| Fusion ft α=0.35 τ=0.50 | 85.29% | 100.00% | 92.06% |

**Recall = 100%**：58/58 正例全部命中，FN=0。

---

## 觀察

1. **Recall 100%，FN=0**：模型本次傾向更寬鬆，所有真正的免費食物貼文全部偵測到，代價是 FP 從 3 增加到 10，Precision 從 94.64% 降至 85.29%。

2. **F1 與 experiment_03 相近**（92.06% vs 92.98%）：兩次訓練的整體品質相當，只是 Precision / Recall 的平衡點不同，符合隨機梯度下降的正常波動。

3. **FP 分析（10 筆）**：主要為「營隊/活動剩食」型貼文，如「電機營多的便當」、「新生營多的便當」等——這些實際上都是真正的免費食物，疑似**標注錯誤**（應為正例卻標成負例），非模型問題。需人工複查。

4. **Fusion ft = fine-tuned L2**：α=0.35 下 L1 仍無貢獻，問題持續。

5. **HuggingFace 上傳成功**：
   - Repo：`syoslyot/qwen-detfuse-finetuned`
   - Commit message：`experiment_03 (2026-05-30)`

---

## 與前次實驗比較

| | experiment_03 | experiment_04 |
|--|--|--|
| fine-tuned L2 Precision | 94.64% | 85.29% |
| fine-tuned L2 Recall | 91.38% | **100.00%** |
| fine-tuned L2 F1 | **92.98%** | 92.06% |
| FP | 3 | 10 |
| FN | 5 | **0** |
| HF 上傳 | 失敗 | ✓ 成功 |

---

## 下一步

- [ ] **人工複查 FP**：「電機營多的便當」、「新生營多的便當」等是否應重新標注為正例
- [ ] **多 seed 實驗**：experiment_03 和 04 結果有差異，需至少 3 個 seed 取平均才能給出可信的效能數字
- [ ] **調整 α**：測試 fine-tuned L2 + L1 能否在 Fusion 中產生互補效果
- [ ] **建立真實分佈測試集**（正負比 1:30+）
