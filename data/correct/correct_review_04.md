# 回應評閱 — experiment_04

針對 `review_experiment_04` 的批判逐項回答與反思。

---

## 1. 可重現性與實驗設計

### 1.1 Seed 未固定

**已修正。**

experiment_05（2026-06-01）已首次固定所有隨機源：

```python
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
set_seed(SEED)
torch.backends.cudnn.deterministic = True
```

experiment_03 和 experiment_04 使用相同程式碼卻出現 Precision 94.64% vs 85.29%、FP 3 vs 10 的顯著差異，正是 seed 未固定造成的隨機快照。這兩次結果都無法代表模型的真實效能——它們只能說明「在這個特定初始化下的表現」。

### 1.2 Loss 系統性偏高（experiment_04 vs experiment_03）

評閱正確指出 experiment_04 每個 epoch 的 loss 都一致性高於 experiment_03（Epoch 3：0.0228 vs 0.0163），這不像純粹的隨機波動。

分析：這很可能反映不同初始化點進入了不同的 loss landscape，導致收斂到不同的局部最優。experiment_04 的初始化讓模型更傾向輸出高分（Recall=100%，FP=10），experiment_03 的初始化更保守（Recall=91.38%，FP=3）。

experiment_05 固定 SEED=42 後，loss 曲線（Epoch 3：0.0161）與 experiment_03 極為接近，間接說明 experiment_03 的隨機初始化恰好與 SEED=42 接近，是一個「比較幸運」的隨機種子。

### 1.3 測試集正例數不一致（58 vs 67）

experiment_03/04 報告 58 正例，experiment_05 報告 67 正例，差距 9 筆。

最可能的原因：experiment_03/04 的評估實際上使用了 notebook 內建的 inline SAMPLES（一個手動整理的子集），而非完整的 test_data.json。experiment_05 明確從 GitHub raw URL 載入完整 test_data.json（67 正例/167 負例），這才是正確的評估基準。

**前三次實驗的結果不能與 experiment_05 直接比較**，應在報告中標注「評估於不完整子集，正例數 58」。

---

## 2. 評估方法

### 2.1 Precision 85.29%、F1 92.06% 在真實環境的意義

目前測試集正負比 1:3，真實分佈估計 1:30。以 experiment_04 的數字估算：

- 測試集：TP=58，FP=10 → Precision=85.3%
- 若真實環境負例為測試集 10 倍，且 FP 率線性放大：FP 預計從 10 增至 100，Precision → 58/(58+100) = 36.7%

這是保守估算。實際情況可能更差，因為真實負例的語意分佈可能與訓練集不同（OOD 問題）。

**現有的 F1/Precision 數字只在「類似訓練分佈」的評估環境下有意義，不應直接作為部署決策依據。**

### 2.2 FP 與 FN 代價不對等

對使用者來說：
- **FN**（漏掉真正的免費食物）：直接損失，使用者錯過本可取得的資源
- **FP**（多推一則無關通知）：干擾，但代價較低

因此 β > 1 的 F-beta score 更符合任務需求。建議使用 **F2 score**（β=2，Recall 的權重為 Precision 的兩倍）作為主要指標，部署閾值也應低於 0.5（偏向高 Recall）。

### 2.3 Fusion α 掃描缺失

確認：報告只測試了 α=0.35，不足以判斷「Fusion 在所有 α 下都無效」。

已在 notebook 中加入 α ∈ {0.0, 0.1, ..., 1.0} 的系統性掃描，並繪製 F1 vs α 曲線。根據 correct_review_02 的機制分析，fine-tuned L2 輸出的 0.0/1.0 極端值使 Fusion 在任何 α 下都無法改變決策，alpha sweep 圖表將以視覺方式確認此結論。

---

## 3. 結論的有效性

### 3.1 「FP 是標注錯誤而非模型錯誤」的驗證標準

評閱正確指出這只是推測，缺乏明確的標注準則。

**現有（但未成文）的標注原則：**
- **正例**：貼文描述的食物**現在就可以去拿**，不需報名、不需付費、不需特定身份
- **負例**：需消費（折扣/買一送一）、需報名活動、限特定社群成員、已過期
- **邊界情況**：「電機營剩下的便當」——「電機營」暗示特定活動參與者，但若發在公開社群、實際上任何人可去拿，應標為正例

「電機營多的便當」、「新生營多的便當」這類 FP，高度懷疑是標注錯誤（負例被誤標）。需要：
1. 人工複查這批 FP，依標注原則重新判斷
2. 將標注原則正式寫入 `docs/labeling_criteria.md`，讓未來的標注可複用

### 3.2 Recall 100% 的機制

評閱提出兩種解釋：(A) 模型學到寬鬆特徵 vs (B) 真正泛化。

**evidence：**
- experiment_05（SEED=42）的 Recall=88.06%，FN=8，並非 Recall=100%
- experiment_04 的 Recall=100% 是特定初始化下的偶然，而非普遍現象
- experiment_03（不同隨機初始化）Recall=91.38%

結論：experiment_04 的 Recall=100% 更可能是解釋 (A)——該次初始化讓模型更「寬鬆」，傾向輸出高分，代價是 FP=10。這不是可靠的泛化能力，而是隨機性造成的行為偏移。

多 seed 實驗（計劃中的 SEED=1, 2）將確認 Recall 的穩定範圍。

### 3.3 FP 分佈差異（FP=3 vs FP=10）對系統行為的影響

F1 接近不等於整體品質相當。以下對比兩個模型的實際行為差異：

| | experiment_03 | experiment_04 |
|--|--|--|
| FP | 3 | 10 |
| FN | 5 | 0 |
| 部署影響 | 每 20 則通知有 3 則無關 | 每 13 則通知有 10 則無關 |

experiment_04 的模型在真實系統中推播雜訊更多，使用者體驗更差，即使 F1 只差 0.92%。「整體品質相當」的說法過於草率，應改為「在單次實驗下 F1 的隨機波動範圍，無法判斷哪個初始化更好」。

---

## 4. 下一步優先順序調整

**評閱的優先順序建議（修訂後）：**

| 優先順序 | 項目 | 調整理由 |
|---------|------|---------|
| 1 | 建立正式標注準則（`docs/labeling_criteria.md`） | 沒有準則，FP 複查結果無法累積為可複用標準 |
| 2 | 人工複查「營隊剩食」型 FP | 可能直接清除若干標注錯誤，提升資料品質 |
| 3 | 多 seed 實驗（SEED=1, 2） | 在 SEED=42 基礎上再跑兩次，得到有統計意義的效能估計 |
| 4 | α 系統性掃描 | 已在 notebook 實作，確認 Fusion 在所有 α 下的行為 |
| 5 | 真實分佈測試集（1:30） | 長期投資，確認真實部署效能 |

**評閱補充的方向（納入計劃）：**

1. **Calibration 分析**：fine-tuned L2 的輸出是 0/9 的二元整數，不是真正的機率。如果要讓 Fusion 有意義，需要將 L2 的 logit 轉換為連續信心分數（temperature scaling 或直接使用 softmax probability）。這也可能是解決「Fusion 永遠等於 ft L2」的根本方法。

2. **Hard negative mining**：訓練集是否包含困難負例（「限社員的便當」、「需購票的活動」、「條件性兌換」）？若訓練負例太容易，模型在邊界案例上 FP 率會更高。可以對現有 FP 案例做分類，找出是否有系統性的「hard negative 缺口」。

3. **Error consistency 分析**：experiment_03 的 FP=3 和 experiment_04 的 FP=10，有多少是重疊的？
   - 若大量重疊 → 這些是模型系統性盲點，值得針對性加強訓練
   - 若幾乎不重疊 → 隨機性影響大，需要更多 seed 才能穩定結論
   
   目前報告只記錄了部分錯誤文字，需要系統性保存逐筆預測結果（含正確/錯誤標記），才能做跨實驗的 error consistency 分析。建議後續實驗在 evaluate() 中加入結果匯出功能。

---

*回應日期：2026-06-02*
*對應評閱：[review_experiment_04.md](../../report/review/review_experiment_04.md)*
