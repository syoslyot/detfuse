# 教授評閱：correct_review_02

審查日期：2026-06-03

---

## 總評

這份回應比原始 experiment_02 報告成熟許多。你沒有防衛性地否認評閱，而是承認測試集分佈、validation、seed、Fusion 失效與 loss masking 等核心問題，並且能把後續 experiment_03/05 的改進串回來說明。這是好的研究反思。

但目前仍有一個主要缺口：許多回應停在「合理推論」或「已在 notebook 實作」的層次，還沒有把實際證據、數字或檔案位置整理成可審查的結果。因此這份 correction 的方向正確，但尚未完全達到「問題已被解決」的標準。

---

## 1. 可重現性與實驗設計（Reproducibility & Setup）

**做得好的地方**

- 明確承認 experiment_02 的單次結果不能支持「fine-tuned L2 超越 L1」的強宣稱。
- 能把 seed 未固定、validation 缺失、loss masking 錯誤三件事分開處理，沒有混成同一個泛泛的「實驗不穩定」問題。
- 引用 experiment_05 已固定 SEED=42，這是往可重現性推進的重要一步。

**提問**

- 你說 experiment_05 已固定所有隨機源，但 correction 只列出結論，沒有列出 experiment_05 的完整結果表、資料版本與 notebook commit。讀者要如何確認 experiment_05 與 experiment_02 的差異只來自修正流程，而不是測試集、資料載入方式或程式碼其他變動？

- 你提出「接下來需再跑 SEED=1、SEED=2」，這是合理的最低要求。但如果目前只有 SEED=42，一些段落是否應避免使用「已解決」的語氣？例如 seed 固定只能解決單次重跑的 reproducibility，還不能解決效能估計的 variance。

- validation set 的行動計劃是正確的，但你尚未說明切分後是否會改變 training_data 的正負比例，也沒有說明 test set 是否會完全保留到最後才使用。若 validation 也被反覆用來調 alpha 或 epoch，test set 仍可能被間接調參污染。

**建議**

把「已修正」和「計劃修正」再切得更嚴格：

| 問題 | 目前狀態 | 還缺什麼 |
|------|----------|----------|
| seed 固定 | 部分修正 | 至少 3 seeds 的 mean ± std |
| validation set | 計劃修正 | train/val/test 切分規則與 label stratification |
| loss masking | 已修正 | 修正前後同一資料切分下的對照表 |
| 真實分佈測試集 | 計劃修正 | 抽樣來源、標注準則、正負比例 |

---

## 2. 評估方法（Evaluation Methodology）

**做得好的地方**

你對正負比失真的影響解釋得很清楚。特別是指出 F1 只適用於「正負比接近 1:3 的評估情境」，不能外推到真實部署，這是正確且必要的限制聲明。

**提問**

- 你提出 F2 score 與 AUC-PR，但沒有明確說哪一個會成為主要指標。這個任務若更重視 Recall，F2 合理；但如果實際使用者對通知雜訊敏感，Precision@Recall 或 Recall@Precision 可能更貼近部署決策。你能否先定義一個產品層面的操作點，例如「Recall 至少 90%，Precision 盡量高」？

- 你對真實分佈的 Precision 崩落做了定性推論，但尚未實際用 FP rate 換算。若模型在 176 個負例中 FP=10，負例誤報率約 5.68%。在真實正負比 1:30 下，若 Recall=67.24%，Precision 大約會是多少？這類換算可以讓「Precision 被高估」更具體。

- 你說 alpha sweep 已在 notebook Section 12/13 實作，但 correction 沒有附上 sweep 的表格或關鍵結論。若讀者沒有打開 notebook，就無法判斷 Fusion 是否真的在所有 alpha 下失效。

**建議**

下一版 correction 應直接加入兩個最小表格：

1. `alpha` vs Precision / Recall / F1 / changed_predictions
2. 不同正負比假設下的 expected Precision，例如 1:3、1:10、1:30

這會讓評閱從「我相信你的機制推論」變成「我可以檢查你的數據」。

---

## 3. 結論的有效性（Validity of Claims）

**做得好的地方**

Fusion 失效的機制分析是本份 correction 最有價值的部分。你不是只說「結果一樣」，而是用 0/1 極端輸出與 α=0.35、threshold=0.5 的關係推導出 L1 不可能翻轉決策。這比原始報告清楚很多。

**需要更謹慎的地方**

- 「這不是 bug，是訓練目標與 Fusion 設計的根本矛盾」這句話目前略強。它很可能是正確的，但仍需先排除實作 bug：例如 `qwen_ft_prob` 是否真的只有 0/1、`l1_score` 是否在 fusion 中正常呼叫、alpha sweep 是否產生任何 changed prediction。若沒有這些檢查，直接排除 bug 仍然太快。

- 「experiment_03 修正 loss masking 後 FP 從約 10 筆降至 3 筆，直接驗證捷徑假說」這句也需要小心。FP 下降支持 loss masking 有幫助，但不一定直接驗證「有免費字眼 → 輸出 9」這個捷徑機制。要驗證捷徑，應該比較修正前後在含「免費但非直接可拿」hard negatives 上的錯誤率。

- FN 分析提到「研討會 / 活動 / 營隊剩食」是系統性痛點，這很合理；但 correction 沒有列出 FN 數量、比例與代表樣本來源。若要稱為系統性，應至少提供 cluster count。

**建議**

把「機制推論」標成 hypothesis，把「已有數據支持」標成 evidence。例如：

- Evidence：Fusion ft 與 L2 指標完全相同。
- Evidence：L2 parsed probability 只出現 0/1。
- Hypothesis：訓練目標與 Fusion 所需連續信心分數不相容。
- Test：改用 logit softmax probability 後，檢查 alpha sweep 是否出現 changed predictions。

這樣結論會更像研究報告，而不是事後解釋。

---

## 4. 下一步（Next Steps）

**優先順序評估**

你列出的後續行動大致合理，但還需要排序。以目前 detfuse 的研究問題來看，我建議優先順序如下：

| 優先順序 | 項目 | 理由 |
|---------|------|------|
| 1 | 固定資料切分與 seed，重跑 baseline / L2 / fusion | 沒有穩定基準，後續比較都不可靠 |
| 2 | 匯出逐筆 prediction 結果 | 才能做 FP/FN cluster、L1/L2 disagreement、error consistency |
| 3 | alpha sweep 加上 changed_predictions | 直接檢驗 Fusion 是否有作用 |
| 4 | 建 validation set 與 early stopping | 避免把 test set 當調參工具 |
| 5 | 真實分佈 mini test set | 決定部署外部效度 |

**報告未充分處理的方向**

1. **標注準則**：你在 FP/FN 分析中多次提到「可直接拿」、「需消費」、「活動剩食」，但尚未正式定義正負例。沒有 annotation guideline，後續 hard positive / hard negative 的判斷會持續漂移。

2. **逐筆結果保存**：如果每次實驗只保存 aggregate metrics，後續很難回答「錯的是同一批樣本嗎？」這類真正有研究價值的問題。

3. **Sequential vs Parallel 的成本指標**：你已提到 L2 呼叫次數，但還應加入 latency 或 estimated cost。否則 sequential 的優勢只停留在概念層次。

---

## 總結建議

這份 correction 已經成功把 experiment_02 從「單次結果報告」推進到「知道自己缺什麼的研究計劃」。下一步不要再增加太多新想法，而是把已承諾的幾件事做成可檢查的證據：固定 split、multi-seed、alpha sweep 表格、逐筆預測輸出、標注準則。只要這些補上，detfuse 的核心問題才有機會被正面回答。

---

*評閱人：Claude（代理教授角色）*  
*對應回應：[correct_review_02.md](../correct/correct_review_02.md)*  
*原始評閱：[review_experiment_02.md](review_experiment_02.md)*
