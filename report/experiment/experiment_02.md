# 實驗紀錄 02 — LoRA Fine-tune 第二輪驗證

日期：2026-05-30

---

## 實驗目標

重新驗證第一輪（experiment_01）的結果，確認 fine-tune 方向可行。

**長期目標**：讓 fine-tuned L2 在 F1 上超越 L1（KeywordDetector），才有使用語言模型的意義。
目前 L1 F1 基準線：**64.65%**，L2 需要突破這個數字。

---

## 方法說明

與 experiment_01 相同：
- L1：KeywordDetector（regex）
- L2：Qwen2.5-0.5B-Instruct，zero-shot
- Fusion：α=0.35, threshold=0.50
- Fine-tune：LoRA r=16, lora_alpha=32, 3 epochs, batch_size=4, lr=2e-4
- 環境：Colab T4 GPU

---

## 資料集

| 資料集 | 總筆數 | 正例 | 負例 |
|--------|--------|------|------|
| training_data.json | 894 | 212 | 682 |
| test_data.json（評估用） | 234 | 58 | 176 |

---

## 結果

| 模型 | Precision | Recall | F1 |
|------|-----------|--------|----|
| KeywordDetector (L1) | 78.05% | 55.17% | 64.65% |
| Qwen2.5-0.5B zero-shot (L2) | 14.29% | 22.41% | 17.45% |
| Fusion α=0.35 τ=0.50 | 19.79% | 32.76% | 24.68% |
| **Qwen2.5-0.5B fine-tuned (L2)** | **70.91%** | **67.24%** | **69.03%** |
| Fusion ft α=0.35 τ=0.50 | 70.91% | 67.24% | 69.03% |

---

## 觀察

1. **Fine-tuned L2 F1 69.03% > L1 F1 64.65%**：首次達成長期目標——L2 在 F1 上超越 L1，使用語言模型有了實質意義。

2. **Recall 大幅改善**：相較 experiment_01（Recall 58.62%），這次 Recall 提升至 67.24%，代表漏報減少，更能捕捉到真正的免費食物貼文。Precision 則從 73.91% 略降至 70.91%，整體是更好的 trade-off。

3. **Fusion ft 與 fine-tuned L2 完全相同**：α=0.35 下 L1 對最終結果無額外貢獻，問題同 experiment_01。Fusion 的優勢尚未發揮。

4. **Zero-shot Qwen 仍舊極差**（F1 17.45%）：與 experiment_01 完全一致，確認 zero-shot 對這個任務無效。

---

## 與 experiment_01 比較

| | experiment_01 | experiment_02 | 變化 |
|---|---|---|---|
| fine-tuned L2 Precision | 73.91% | 70.91% | -3% |
| fine-tuned L2 Recall | 58.62% | 67.24% | **+8.62%** |
| fine-tuned L2 F1 | 65.38% | **69.03%** | **+3.65%** |
| 超越 L1 F1 基準線？ | 否（65.38% ≈ 64.65%） | **是（69.03%）** | ✓ |

---

## 下一步

- [ ] 調整 α 值，測試 fine-tuned L2 + L1 能否在 Fusion 中互補（目前 α=0.35 L1 貢獻為零）
- [ ] 分析仍漏掉的 FN：「研討會」、「營隊」類貼文是主要痛點
- [ ] 增加 training epochs 或資料量，看 Recall 能否繼續提升
- [ ] 用接近真實分佈的測試集驗證（正負比 1:30+）
