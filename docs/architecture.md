# 偵測架構

## 問題定義

**輸入**：一則 FB 貼文的純文字
**輸出**：`True`（是免費食物）或 `False`（不是）

---

## 原始設計（freehiero 現行）

```
貼文
 │
 ├─ L1 KeywordDetector（regex）
 │   ├─ 確定正  ──────────────────────► True
 │   ├─ 確定負  ──────────────────────► False
 │   └─ 不確定
 │        │
 │        └─ L2 OllamaDetector（Qwen）──► True / False
```

**問題**：L1 宣告「確定」後 L2 永遠不執行。
L1 的 false positive（把非食物當食物）和 false negative（漏掉沒有關鍵字的創意寫法）都無法被修正。

---

## 目標設計（detfuse 研究中）

兩者都給出一個**分數**，最後用公式決定：

```
final_score = α × l1_score + (1 − α) × l2_prob
prediction  = final_score > threshold
```

### L1 score（`experiments/fusion.py: l1_score()`）

把 regex pattern 的命中結果轉成浮點數：

| pattern 命中 | 加分 |
|-------------|------|
| `FREE_FOOD`（免費 + 食物同行）| +0.80 |
| `EVENT_FOOD`（研討會 + 餐點）| +0.60 |
| `HAS_CATERING`（有供餐）| +0.50 |
| 只有 free 關鍵字（無食物）| +0.20 |
| 只有 food 關鍵字（無免費）| +0.10 |
| `NOT_FOOD`（免費課程/票/活動）| → **強制回傳 0.0** |

上限 clamp 到 1.0。

### L2 prob（`experiments/fusion.py: l2_prob()`）

送給 Qwen，解析回應為 1.0（yes）或 0.0（no）。
未來可改 prompt 讓 Qwen 回傳信心分數（0.0–1.0）以獲得更細緻的結果。

### 融合公式

```
α = 0.35   → L1 佔 35%，L2 佔 65%
threshold  = 0.50
```

初始值，需用 `evaluate()` grid search 調整。

### 跳過 L2 的條件

當 `l1_score == 0.0`（完全沒有任何關鍵字信號），直接回傳 False，不呼叫 L2。
這是效能優化，不影響 recall（連食物相關詞都沒有的貼文不可能是正例）。

---

## 各情境行為

| L1 score | L2 prob | final (α=0.35) | 說明 |
|---------|---------|----------------|------|
| 0.9 | 0.9 | **0.90** → True | 雙方都確定，正確 |
| 0.9 | 0.1 | **0.38** → False | L1 誤判，L2 修正 ✓ |
| 0.1 | 0.9 | **0.62** → True | L1 弱，L2 救回 ✓ |
| 0.0 | — | **0.00** → False | 跳過 L2（無信號）|
| 0.4 | 0.5 | **0.47** → False | 都不確定，保守負 |

---

## 待研究方向

- [ ] 改 Qwen prompt，讓 L2 輸出信心分數而非 yes/no
- [ ] 用 training_data.json 做 grid search，找最佳 α 和 threshold
- [ ] 嘗試 fine-tune qwen2.5:0.5b，針對台灣學生社群語境
- [ ] 評估是否值得換更大的模型（1.5B / 3B）vs 速度代價
