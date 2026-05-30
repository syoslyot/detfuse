# 實驗紀錄 03 — Loss Masking 修正後重跑

日期：2026-05-30

---

## 實驗目標

修正 experiment_02 中發現的 loss masking bug——`SFTDataset` 和 `collate_fn` 對整個對話序列（system prompt + user message + assistant reply）都計算 loss，導致訓練訊號被大量無關 token 稀釋。

修正後重跑，驗證效果差異。

---

## 修正內容

### Bug
`collate_fn` 將 `labels` 直接設為 `input_ids`，代表所有 token（包含 system prompt 和 user message）都計算 loss。

### Fix
`SFTDataset.__init__` 中：
1. 先 tokenize prompt-only（system + user）取得 `prompt_len`
2. labels 設為 `[-100] * prompt_len + input_ids[prompt_len:]`
3. 只有 assistant 回覆的 token（數字 `'0'` 或 `'9'` + EOS，共 3 個）計算 loss

**驗證輸出**：
```
Total tokens: 254, Label tokens (assistant only): 3
```
確認 masking 正確，3 個 label token 遠少於 254 個 total token。

---

## 方法說明

與 experiment_02 相同，差別只有 loss masking：
- L1：KeywordDetector（regex）
- L2：Qwen2.5-0.5B-Instruct，zero-shot
- Fusion：α=0.35, threshold=0.50
- Fine-tune：LoRA r=16, lora_alpha=32, 3 epochs, batch_size=4, lr=2e-4
- 環境：Colab T4 GPU

---

## 訓練 Loss

| Epoch | Avg Loss |
|-------|----------|
| 1 | 0.1363 |
| 2 | 0.0470 |
| 3 | 0.0163 |

相較 experiment_02（Epoch 1 avg loss: 2.44），loss 大幅降低。原因是現在只對 3 個 token 計算，訓練更聚焦，梯度訊號更純粹。

---

## 結果

| 模型 | Precision | Recall | F1 |
|------|-----------|--------|----|
| KeywordDetector (L1) | 78.05% | 55.17% | 64.65% |
| Qwen2.5-0.5B zero-shot (L2) | 14.29% | 22.41% | 17.45% |
| Fusion α=0.35 τ=0.50 | 19.79% | 32.76% | 24.68% |
| **Qwen2.5-0.5B fine-tuned (L2)** | **94.64%** | **91.38%** | **92.98%** |
| Fusion ft α=0.35 τ=0.50 | 94.64% | 91.38% | 92.98% |

---

## 與前兩次實驗比較

| | experiment_01 | experiment_02 | experiment_03 | 變化（vs 02） |
|--|--|--|--|--|
| fine-tuned L2 Precision | 73.91% | 70.91% | **94.64%** | +23.73% |
| fine-tuned L2 Recall | 58.62% | 67.24% | **91.38%** | +24.14% |
| fine-tuned L2 F1 | 65.38% | 69.03% | **92.98%** | +23.95% |

loss masking 修正是主要原因，並非超參數或資料量的改變。

---

## 觀察

1. **Loss masking 效果極為顯著**：F1 從 69.03% 跳至 92.98%，單純修正一個訓練 bug 帶來約 24% 的提升。這也說明 experiment_01 和 02 的結果其實是在「雜訊很大的訓練」下產生的，模型的真正潛力遠不止於此。

2. **FP 剩 3 筆，FN 剩 5 筆**：錯誤大幅減少。
   - FP 模式：「活動剩下的飲料（帶水瓶來裝）」、「新生營多的便當」——這些其實是真正的免費食物，可能是標注問題而非模型錯誤，值得人工複查
   - FN 模式：「研討會多的餐盒」、「營隊剩下的...」——仍有少數研討會/營隊類漏掉

3. **Fusion ft 與 fine-tuned L2 仍完全相同**：α=0.35 下 L1 依然沒有貢獻，問題與前兩次一致。Fine-tuned L2 已經非常強，L1 的預測完全被覆蓋。

4. **Loss 收斂快且低**：3 個 epoch 後 avg loss 降至 0.0163，訓練非常穩定。這是 label-only loss 的優勢——梯度訊號集中在真正需要學的地方。

---

## 下一步

- [ ] **人工複查 3 筆 FP**：「活動剩下的飲料」類型可能是標注錯誤（應為正例），需確認
- [ ] **分析 Fusion 無效的根本原因**：調整 α 值（如 α=0.6 或 α=0.8）讓 L1 有更多話語權，觀察是否能進一步提升
- [ ] **建立接近真實分佈的測試集**（正負比 1:30+），驗證高 F1 是否在真實環境成立
- [ ] **考慮 early stopping**：loss 已在 epoch 2 後趨近收斂，epoch 3 可能輕微 overfit
