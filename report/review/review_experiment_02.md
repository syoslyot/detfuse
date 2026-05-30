# 教授評閱：experiment_02

> 本文件以學術評閱者（指導教授）的角度，對 experiment_02 提出批判、提問與建議。
> 目的不是否定結果，而是找出尚未說清楚的事、可能使結論失效的風險、以及下一步的方向。

---

## 一、先肯定什麼

Fine-tuned L2 F1（69.03%）首次超越 L1 基準線（64.65%），這是一個有意義的節點。
訓練框架從 `trl` 改為純 PyTorch 解決了環境問題，流程也已可重現。

---

## 二、核心問題：這個評估可以信嗎？

### 2.1 測試集的正負比嚴重失真

**問題**：test_data.json 正例比例約 25%（58/234），但你在 report 裡自己也寫了「真實社團 feed 估計 1–3%」。

這意味著：在真實環境裡，模型面對的 99% 都是負例，而你的測試集只有 75%。這導致：
- **Precision 被嚴重高估**：真實環境下 FP 的基數遠大於測試集，Precision 可能從 70% 崩到 20% 以下
- **F1 失去意義**：F1 預設正負各有一定比例，在極度不均的分佈下，你應該看 **F-beta score**（β > 1，偏重 Recall）或 **Precision-Recall AUC**

**建議**：在做任何「超越基準線」的宣稱前，必須先建立一個接近真實分佈（正負比 1:30 以上）的測試集，哪怕只有 100 筆也好。

---

### 2.2 沒有驗證集，無法偵測 overfitting

訓練 3 epochs，只看 training loss 下降。你不知道模型在訓練過程中有沒有 overfit。

**問題**：loss 從 3.74 收斂到多少？最後幾個 epoch 的 loss 差距是多少？

fine-tuned L2 Recall 從 experiment_01 的 58.62% 提升到 67.24%，這是好事，但也可能是因為模型變得更「寬鬆」（傾向輸出 '9'），而不是真的學到更好的語意。你需要一個 validation set 在訓練過程中監控，而不是只看最終 test 結果。

**建議**：
- 把 training_data.json 切成 train / val（如 80/20），validation loss 降到最低點時停止（early stopping）
- 對比「epoch 1 / 2 / 3」分別在 test 上的表現，畫出 learning curve

---

### 2.3 單一隨機種子，沒有統計意義

你只跑了一次，報告的是單次結果。LoRA fine-tune 有隨機性（weight initialization、batch shuffle），不同 seed 可能有 ±3–5% 的 F1 變動。

「F1 69.03% > 64.65%」這個差距只有 4.38 個百分點，在單次實驗下，**不能排除是隨機波動**。

**建議**：至少跑 3 個不同 seed，報告平均值 ± 標準差。

---

## 三、Fusion 根本沒在運作

Fusion ft 與 fine-tuned L2 結果完全相同（Precision / Recall / F1 精確到小數點後兩位）。這不只是「L1 貢獻不夠」，而是一個需要深入調查的訊號。

**可能原因**：

1. **α=0.35 對 Fusion 結果的影響被 threshold 截斷**：fine-tuned L2 已經輸出接近 0 或 1 的極端值，L1 加進去後 final score 不跨越 threshold，所以不改變任何 prediction

2. **L1 和 fine-tuned L2 在所有樣本上的預測完全一致**：L1 正確的地方 L2 也正確，L1 錯誤的地方 L2 也錯誤，所以加權後 prediction 不變

3. **bug**：`fuse()` 呼叫的 `qwen_prob()` 仍指向 fine-tuned 的 model，這是對的；但有沒有確認 `l1_score()` 和 `qwen_prob()` 都有在 `fuse()` 裡被呼叫到？

**必須做的分析**：
```python
# 列出 L1 和 fine-tuned L2 預測不一致的樣本
for label, text in TEST_SAMPLES:
    s1 = l1_score(text)
    s2 = qwen_prob(text)
    pred_l1 = s1 > 0.5
    pred_l2 = s2 > 0.5
    if pred_l1 != pred_l2:
        print(f"label={label} L1={s1:.2f} L2={s2:.2f} | {text[:60]}")
```

如果這個列表是空的，代表 L1 和 fine-tuned L2 在每一筆樣本上都達成相同決策，Fusion 永遠不改變任何結果——Fusion 的研究目標就尚未開始。

---

## 四、模型行為沒被解釋

### 4.1 FP 案例讓人困惑

這些出現在 fine-tuned L2 的 FP（誤報）：

- 「大同冰箱出售」— 完全不相關，為什麼模型輸出高分？
- 「停電時住宿服務組外的木桌練習止血」— 完全不是食物
- 「億品鍋免費加肉卷 *40 張」— 這個確實有「免費」和「食物」，但性質上是「消費折扣」，不是「直接可拿的食物」

這些錯誤模式說明 fine-tune 可能讓模型學到「有免費字眼 → 輸出 9」的捷徑，而不是真正理解「可以現在就去拿」這個語意。

**建議**：對 FP 案例做質性分類（比如：折扣類、完全不相關類、過期貼文類），找出模型的系統性盲點。

### 4.2 FN 案例有規律

多數 FN 是「研討會 / 活動 / 營隊剩食」類型，如：
- 「研討會剩下的便當，地點國際會議廳」
- 「營隊剩下的熏雞吐司，在社科院」

這些在 training_data 裡應該有正例，但模型仍然漏掉。這代表：
1. 訓練資料裡這類正例不夠多，或格式變化太大
2. 「研討會」沒有出現在 L1 的 `_FREE_WORDS` / `_FOOD_WORDS` 組合中——L1 依靠 `_PATTERN_EVENT_FOOD` 這個單獨 pattern，但 L2 fine-tune 沒有把這個語意學進去

---

## 五、超參數沒有根據

| 超參數 | 使用值 | 問題 |
|--------|--------|------|
| LoRA r | 16 | 為什麼是 16？r=4 或 r=32 有沒有比較過？ |
| lora_alpha | 32 | 通常 alpha = 2×r，這是 default，有沒有試過其他值？ |
| target_modules | q_proj, v_proj | 加上 k_proj, o_proj 會不會更好？ |
| epochs | 3 | 是否太少？太多？有沒有看過 val loss curve？ |
| lr | 2e-4 | 對 0.5B 模型偏高，有沒有做 lr sensitivity test？ |
| loss function | cross entropy on full sequence | 是否應該只計算 assistant token（最後那個數字）的 loss？ |

最後一點特別值得注意：`collate_fn` 把整個對話序列的 `input_ids` 都設為 labels，代表模型在對 system prompt 和 user message 也做 language modeling loss。這會干擾訓練效率——標準做法是只對 assistant 回覆的 token 計算 loss，其餘設為 -100（你的 `collate_fn` 沒有做這個區分）。

---

## 六、你還沒回答的核心研究問題

experiment_01 和 02 建立了「fine-tune 有用」的基本事實，但 **detfuse 的核心研究問題尚未被正面回答**：

> 並行融合（Parallel Fusion）比串聯架構（Sequential，即 freehiero 現行做法）更好嗎？

目前的實驗只比較了不同模型的單獨表現，沒有直接和 production 架構（L1 先過，不確定才呼叫 L2）做對比。

**建議**：實作 `sequential_detect()`，在同一個 test set 上和 Fusion 比，用 throughput（速度）和 F1 同時評估。

---

## 七、應該學習的知識

### 立即需要的
- **Precision-Recall curve 與 AUC-PR**：比 F1 更適合類別不均的評估，sklearn 的 `precision_recall_curve` 10 行就能畫出來
- **Class imbalance 處理**：weighted loss（`pos_weight` in `BCEWithLogitsLoss`）、oversampling（`WeightedRandomSampler`）
- **Masked language model loss**：只對目標 token 計算 loss，讓訓練更聚焦

### 中期建議
- **LoRA 理論**：Hu et al. 2022 原論文，理解 r 和 alpha 的實際意義
- **Calibration**：你的模型輸出 0–9 整數，這不是機率，`temperature scaling` 可以把它校正成真正的信心分數
- **Error analysis methodology**：如何系統化分析 FP/FN，找出有意義的 error cluster

### 進階（如果要寫成論文）
- **Statistical significance testing**：McNemar's test 用於比較兩個分類器在同一測試集上的差異
- **Ensemble methods**：stacking、voting、blending 的設計原則
- **Active learning**：如何用最少的標注成本讓模型持續進步

---

## 八、給下一個實驗的具體建議

按優先順序：

1. **修 loss masking**：`collate_fn` 裡只對 assistant token 計算 loss，其餘設 -100，這是 instruction tuning 的標準做法，修完重跑一次看 F1 有沒有變化

2. **建 validation set**：從 training_data 切 20% 出來，跑完每個 epoch 後印出 val loss，確認沒有 overfit

3. **分析 L1/L2 分歧樣本**：找出兩者預測不同的樣本，這才是 Fusion 真正能發揮的空間

4. **建一個真實分佈的 mini test set**：哪怕只有 100 筆（正負比 1:30），用來給最終結果做壓力測試

5. **對比 sequential vs parallel**：在同一 test set 上跑，這才是這個研究的核心貢獻

---

*評閱人：Claude（代理教授角色）*
*評閱日期：2026-05-30*
*對應實驗：[experiment_02.md](../experiment/experiment_02.md)*
