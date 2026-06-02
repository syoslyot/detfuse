# 回應評閱 — experiment_02

針對 `review_experiment_02` 的批判逐項回答與反思。

---

## 2.1 測試集正負比失真

**確認問題成立。**

test_data.json 正負比約 1:3（58:176），而真實社團 feed 的正例率估計不超過 3%（約 1:33）。兩者相差一個數量級，導致以下系統性偏差：

- **Precision 被高估**：真實環境的 FP 基數遠大於測試集，如果模型的 FP 率在分佈外泛化不佳，Precision 可能從 70% 崩至 20–30%
- **F1 失去部署意義**：F1 隱含正負比對稱假設，在 1:30+ 的極度不均分佈下，AUC-PR 與 F2 score（β=2，偏重 Recall）是更合適的指標

目前所有 F1 數字的有效範圍只限於「正負比接近 1:3 的評估情境」，不能外推至真實部署。

**後續行動：**
1. 建立小型真實分佈測試集（目標 100 筆，正負比 1:30，從真實 feed 取樣）
2. evaluate() 加入 F2 score 輸出
3. 報告結論加入分佈限制聲明

---

## 2.2 沒有驗證集，無法偵測 overfitting

**確認問題成立。**

experiments 01–02 的訓練只觀察 training loss（而且是包含 system prompt 全序列的 loss，梯度訊號極雜）。在 experiment_03 修正 loss masking 後，epoch 3 的 avg loss 降至 0.0163——訓練目標本質上只需記住 '0' 或 '9' 這兩個 token 的對應，過擬合風險極高。

回顧 experiment_02：loss 從 epoch 1 的高值快速下降，代表模型在 3 個 epoch 內就學到了大量東西，但無法判斷是泛化還是 memorization，因為沒有 validation set 可以偵測分歧點。

**後續行動：**
1. 從 training_data 按 label 分層切 20% 為 validation set
2. 每個 epoch 後計算 val F1，保留最佳 checkpoint（early stopping）
3. 繪製 train loss vs val loss learning curve，確認 epoch 2 之後是否出現過擬合

---

## 2.3 單一隨機種子，無統計意義

**確認問題成立。**

69.03% 與 64.65% 的差距只有 4.38 個百分點，在 LoRA fine-tune 的隨機初始化下，這個差距完全可能是隨機波動。單次結果不能支持「F2 fine-tune 超越 L1」的宣稱。

experiment_05（2026-06-01）已首次固定所有隨機源（SEED=42），結果可重現。接下來需再跑 SEED=1、SEED=2，以平均值 ± 標準差給出有統計意義的效能數字。

---

## 3. Fusion 根本沒在運作

**確認問題成立，機制已分析清楚。**

Fusion ft 與 fine-tuned L2 結果完全相同的根本原因：

fine-tuned 模型的訓練目標是輸出離散整數 '0' 或 '9'，轉換為機率後只有 0.0 和 1.0 兩個極端值。在這個前提下：

- 若 `qwen_ft_prob = 0.0`，則 `fusion = 0.35 × L1 + 0.65 × 0.0 = 0.35 × L1 ≤ 0.35 < 0.5`，永遠為 False
- 若 `qwen_ft_prob = 1.0`，則 `fusion = 0.35 × L1 + 0.65 ≥ 0.65 > 0.5`，永遠為 True

**結論：α=0.35 下的 L1 擾動量，無法在任何樣本上跨越 threshold=0.5，Fusion 在架構設計上已失效。**

這不是 bug，是訓練目標與 Fusion 設計的根本矛盾——Fusion 預設 L2 輸出連續信心分數，但訓練的 L2 輸出二元整數。

已在 notebook 中加入 alpha sweep（α ∈ {0.0, 0.1, ..., 1.0}）和 L1/ft-L2 分歧樣本分析，以圖表形式確認此結論。

**後續設計選項：**
1. 改用 logit/softmax 機率作為 L2 的連續輸出（不依賴 '0'–'9' token 解析）
2. 讓 L2 輸出真正的連續分數（訓練目標改為 BCE regression 或 softmax over label set）
3. 將 Fusion 改為「L1 和 L2 投票制」：只有兩者決策相同才採用，否則交給人工處理

---

## 4. 模型行為分析

### 4.1 FP 案例分類（experiment_02 時期，loss masking 未修正）

| FP 類型 | 案例 | 推斷原因 |
|---------|------|---------|
| 完全無關型 | 「大同冰箱出售」 | 模型對「大同」品牌或「出售」字眼產生奇異反應，無關免費食物 |
| 完全無關型 | 「停電時住宿服務組外的木桌練習止血」 | 可能對「外」或其他字眼誤觸發 |
| 語意邊界型 | 「億品鍋免費加肉卷 *40 張」 | 有「免費」+「食物」，但需消費才能取得，屬條件性兌換 |

這些 FP 印證評閱所指的「學到捷徑」假說——未修正 loss masking 時，模型學到「有免費字眼 → 輸出 9」，而非理解「可以現在就去拿」的語意。experiment_03 修正 loss masking 後 FP 從約 10 筆降至 3 筆，直接驗證了此假說。

### 4.2 FN 案例規律分析

「研討會 / 活動 / 營隊剩食」是系統性 FN 痛點，原因有三：

1. **表達形式多樣**：「剩餘」「多的」「吃不完的」沒有明確「免費」字眼，模型訓練資料可能未涵蓋足夠變體
2. **L1 的 `_PATTERN_EVENT_FOOD` 能捕捉**，但 fine-tuned L2 沒有繼承這個語意，說明 fine-tune 訓練集中此類正例密度不足
3. **後續含 `#更` 的重複貼文**（experiment_05 發現）：文字結構異常，預處理應過濾這類更新貼

**資料增強建議：** 針對「研討會 / 活動 / 營隊」語境加入 hard positive 訓練範例。

---

## 5. 超參數反思

| 超參數 | 使用值 | 反思 |
|--------|--------|------|
| LoRA r | 16 | 任務只需預測 '0'/'9'，r=4 或 r=8 可能已足夠，且過擬合風險更低。未比較 |
| lora_alpha | 32 | alpha=2×r 是慣例 default，沒有任務依據 |
| target_modules | q_proj, v_proj | 加入 k_proj, o_proj 對這個簡單分類任務可能沒必要 |
| epochs | 3 | loss masking 修正後，epoch 2 已接近收斂（loss 0.047），epoch 3（0.016）有過擬合風險 |
| lr | 2e-4 | 對 0.5B 模型偏高，未做 sensitivity test |
| loss function | 全序列 CE | **已在 experiment_03 修正**：改為只對 assistant token（'0'/'9' + EOS）計算 loss |

loss masking 的修正是最關鍵的改進——單純修正這個 bug 就讓 F1 從 69% 跳至 93%，說明訓練目標的正確性對 LoRA fine-tune 有決定性影響。

---

## 6. 核心研究問題尚未回答

**確認：detfuse 的核心研究問題——Parallel Fusion 是否優於 Sequential 架構——至今仍未被正面比較。**

Sequential 架構定義（freehiero 現行做法近似）：
1. L1 先過（regex），高信心（score ≥ 0.7）直接回傳，低信心（score ≤ 0.3）直接拒絕
2. 不確定區間（0.3–0.7）才呼叫 L2

Sequential 的核心優勢是節省 L2 呼叫成本——在真實系統中，L2（LLM inference）遠比 L1（regex）昂貴。評估需同時記錄 F1 和 **L2 呼叫次數**，才能給出有意義的 efficiency vs accuracy trade-off 分析。

已在 notebook 中實作 `sequential_detect()`，並加入與 Parallel Fusion 的對比輸出。

---

## 7. 知識補充進度

| 主題 | 狀態 |
|------|------|
| Precision-Recall curve / AUC-PR | **已實作**（notebook Section 12/13） |
| F2 score（β=2） | 計劃加入 evaluate() |
| Alpha 系統性掃描 | **已實作**（notebook Section 12/13） |
| Sequential detection | **已實作**（notebook Section 13） |
| L1/L2 分歧分析 | **已實作**（notebook Section 12/13） |
| LoRA 理論（Hu et al. 2022） | 閱讀中 |
| Calibration / temperature scaling | 待研究 |
| McNemar's test | 待學習（多 seed 實驗後） |

---

*回應日期：2026-06-02*
*對應評閱：[review_experiment_02.md](../../report/review/review_experiment_02.md)*
