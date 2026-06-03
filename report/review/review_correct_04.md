# 教授評閱：correct_review_04

審查日期：2026-06-03

---

## 總評

這份回應清楚承認了 experiment_04 的主要問題：seed 未固定、測試集正例數不一致、真實分佈與測試分佈差距大、Fusion α 掃描不足，以及「FP 是標注錯誤」仍缺乏準則。整體來說，回應方向是正確的，也比原始報告更謹慎。

不過，這份 correction 有幾處把尚未完全驗證的解釋寫得太確定。特別是 loss landscape、experiment_03 與 SEED=42 的接近、inline SAMPLES 導致正例數差異、以及 experiment_04 Recall 100% 的機制判斷，目前都需要更多直接證據支撐。

---

## 1. 可重現性與實驗設計（Reproducibility & Setup）

**做得好的地方**

- 正確承認 experiment_03 和 experiment_04 都不能代表模型真實效能，只能視為未固定 seed 下的隨機快照。
- 指出 58 vs 67 正例數不一致，這是非常重要的資料管線問題。這個問題若不解決，後續所有跨實驗比較都可能失效。
- 把 seed 固定、資料載入來源、測試集版本三件事分開討論，這比單純說「下次固定 seed」更完整。

**提問**

- 你推測 experiment_03/04 使用 inline SAMPLES，而 experiment_05 使用完整 test_data.json。這個推測很合理，但目前 correction 沒有提供直接證據。能否列出 notebook 中當時實際使用的評估變數、樣本總數與 label count？否則「前三次實驗不能與 experiment_05 直接比較」雖然可能正確，但仍像事後推斷。

- 你說 experiment_05 的 loss 曲線與 experiment_03 接近，因而「間接說明 experiment_03 的隨機初始化恰好與 SEED=42 接近」。這個說法過度解釋。loss 接近不代表初始化接近，也可能是資料順序、batch 組成、訓練穩定性或不同測試集造成。這裡應改成較保守的表述：experiment_05 提供了一個可重現參考點，但不能反推 experiment_03 的 seed 性質。

- 你列出固定 seed 的程式碼，但沒有提到 CUDA/cuDNN、transformers、peft、torch、bitsandbytes 等版本。若實驗要能重現，軟體版本也是必要資訊。尤其 Colab 環境會變動，只固定 random seed 不足以完整重現。

**建議**

下一版報告應加一個「實驗指紋」表格：

| 欄位 | 內容 |
|------|------|
| git commit / notebook commit | 用來追溯程式碼 |
| data source | inline SAMPLES 或 GitHub raw test_data.json |
| test label count | 正例 / 負例 |
| seed | Python / NumPy / Torch / Transformers |
| package versions | torch、transformers、peft、bitsandbytes |
| hardware | Colab GPU 型號 |

這比在文字中解釋隨機性更有審查價值。

---

## 2. 評估方法（Evaluation Methodology）

**做得好的地方**

- 你把 experiment_04 的 Precision 放到真實分佈中重新估算，這是正確方向。原始報告的 85.29% Precision 在 1:3 測試集上看起來不錯，但在 1:30 分佈下可能變得不可接受。
- 你意識到 FP / FN 代價不對等，並提出 F2 score 作為候選指標，這比只報 F1 更符合任務需求。
- 你承認單一 α=0.35 不足以支持「Fusion 無效」，並補上 alpha sweep 的計劃或實作。

**提問**

- 你在 2.1 的真實分佈估算假設「FP 從 10 增至 100」，但沒有明確用 FP rate 來推導。若測試集中負例為 176、FP=10，FPR 約 5.68%。在正負比 1:30 且正例數 58 的情境下，負例約 1740，預期 FP 約 99，Precision 約 36.9%。這個估算可以成立，但應把公式寫出來，避免看起來像任意乘以 10。

- 你說 F2 score 更符合任務需求，但還沒有定義部署目標。是希望 Recall 至少 90%？還是允許每日最多幾則 FP？如果沒有產品約束，F2 只是另一個 metric，仍不能自動決定 threshold。

- correction 提到 alpha sweep 已實作，但沒有呈現曲線或表格。因此讀者仍無法判斷 Fusion 是否在 α=0.0 到 1.0 間有任何操作點優於 L2。

**建議**

對 experiment_04 的 correction，建議補上以下三個最小輸出：

1. `alpha`、Precision、Recall、F1、F2、changed_predictions。
2. `threshold` sweep 或 PR curve，避免只討論 τ=0.50。
3. 真實分佈估算表，至少列出 1:3、1:10、1:30 三種情境。

---

## 3. 結論的有效性（Validity of Claims）

**做得好的地方**

你對原始報告中幾個過強結論做了修正：

- 不再把 Recall 100% 視為可靠泛化能力。
- 不再把 FP=10 簡單歸因為標注錯誤。
- 不再把 F1 接近解讀為整體品質相當。

這些修正都很重要，表示你開始區分 aggregate metric 和實際錯誤分佈。

**需要更謹慎的地方**

- 「experiment_04 的 Recall=100% 更可能是模型更寬鬆」是合理假說，但目前需要逐筆 score 或 logit 分佈支持。若 L2 只輸出二元 0/9，至少應比較不同 run 的 positive prediction rate、FP/FN 類型、以及錯誤樣本重疊程度。

- 「電機營多的便當」可能是標注錯誤，但你的標注原則仍有模糊處：如果貼文發在公開社群，但文字暗示剩食屬於活動內部，是否算任何人可拿？如果沒有上下文，只靠文字要如何標？這類規則需要寫得能讓第二位標注者一致執行。

- Error consistency 分析被列為計劃，但 correction 沒有指出目前為什麼做不到。真正原因可能是沒有保存逐筆預測結果。這應被列為資料紀錄缺陷，而不是單純「後續值得做」。

**建議**

標注準則不應只寫正負例定義，還應包含邊界案例：

| 類型 | 建議處理 |
|------|----------|
| 需消費才能取得 | 負例 |
| 活動剩食但公開邀請領取 | 正例 |
| 活動剩食但限參與者或地點不明 | 負例或 uncertain |
| 已過期或已送完 | 負例 |
| 缺少是否可領取資訊 | uncertain，不納入訓練或交由人工 |

如果資料集只允許二元 label，至少要定義 uncertain 如何處理，否則邊界樣本會繼續污染訓練與測試。

---

## 4. 下一步（Next Steps）

**優先順序評估**

你提出的修訂優先順序整體合理，但我會略微調整：

| 優先順序 | 項目 | 理由 |
|---------|------|------|
| 1 | 確認資料管線與測試集版本 | 58 vs 67 正例不一致會使所有比較失效 |
| 2 | 建立標注準則與 uncertain 規則 | 沒有準則，FP 複查不可累積 |
| 3 | 匯出逐筆預測結果 | 支援 error consistency、FP/FN cluster、alpha sweep 驗證 |
| 4 | 固定 seed 後跑 multi-seed | 取得效能平均與 variance |
| 5 | alpha / threshold sweep 與 calibration | 判斷 Fusion 是否仍有研究價值 |
| 6 | 真實分佈測試集 | 最終檢驗部署外部效度 |

注意：我會把「確認資料管線」放在標注準則之前，因為如果你不知道 experiment_03/04/05 分別評估在哪個測試集上，就無法判斷哪些 FP 需要複查。

**報告未提到但應加入的紀錄規範**

每次 evaluate 應輸出一份 machine-readable 結果，例如：

```text
experiment_id, sample_hash, text, label, l1_score, l2_score, fusion_score, pred_l1, pred_l2, pred_fusion
```

有了這個檔案，後續才能穩定回答：

- 哪些錯誤跨 seed 重複出現？
- L1 與 L2 真正分歧在哪些樣本？
- Fusion 是否改變任何 prediction？
- FP 是否集中在特定語意類型？

---

## 總結建議

這份 correction 已經修正了 experiment_04 原始報告中最危險的幾個宣稱，但仍需要把「合理推測」轉成「可審查證據」。最優先的工作不是再訓練更多模型，而是先釐清每次實驗到底用了哪個資料集、保存逐筆預測、建立標注準則。否則 multi-seed、calibration 和 alpha sweep 都會建立在不穩定的資料基礎上。

---

*評閱人：Claude（代理教授角色）*  
*對應回應：[correct_review_04.md](../correct/correct_review_04.md)*  
*原始評閱：[review_experiment_04.md](review_experiment_04.md)*
