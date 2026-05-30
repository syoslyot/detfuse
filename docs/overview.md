# 專案概覽：名詞定義與判斷流程

## 任務定義

給定一則社團貼文，判斷它是否在「提供可以現在就去拿的免費食物或飲料」。
輸出：`True`（是）/ `False`（否）。

---

## 名詞定義

### L1 — KeywordDetector（關鍵字層）

用正規表達式（regex）做快速比對，不需要任何 AI 模型。
核心邏輯：貼文中同時出現「免費詞」和「食物詞」→ 視為正例。
輸出兩種形式：
- `bool`：直接判斷（用於 production）
- `float [0, 1]`：軟分數（`l1_score()`，用於 fusion）

### L2 — OllamaDetector / Qwen2.5-0.5B（語言模型層）

用大型語言模型判斷貼文語意，彌補 L1 抓不到的情境（如諷刺語氣、隱晦表達）。
Prompt 要求模型輸出 0–9 整數（信心程度），除以 9.0 轉為 [0, 1] 分數。

**Production（本機）**：透過 Ollama 呼叫 `qwen2.5:0.5b`
**Colab 實驗**：直接用 HuggingFace `transformers` 載入 `Qwen/Qwen2.5-0.5B-Instruct`

### Fusion — 並行融合

L1 和 L2 各自獨立產生分數，加權合併後再做決策。

```
final_score = α × L1_score + (1 - α) × L2_score
prediction  = final_score > threshold
```

目前實驗參數：`α = 0.35`，`threshold = 0.50`（L2 權重較高）。

與 production 架構（L1 先跑、確定時直接返回、不確定才呼叫 L2）的差異：
- **production**：串聯（sequential），L2 是備援
- **detfuse 研究目標**：並聯（parallel），兩層都有發言權，L2 可以推翻 L1

### Fine-tune（微調）

用標記好的訓練資料，針對這個任務對 Qwen2.5-0.5B 做 LoRA instruction tuning，
讓模型學會台灣社團的語境（如「食安自負」、「研討會多的」）。

- 方法：LoRA（Low-Rank Adaptation），只訓練少量參數，不改動原始權重
- 訓練資料：`data/categories/free_food/training_data.json`（894 筆）
- 目標格式：label=1 → 模型輸出 `'9'`，label=0 → 輸出 `'0'`

---

## 使用的模型

| 模型 | 用途 | 執行環境 |
|------|------|----------|
| `qwen2.5:0.5b`（Ollama） | L2 production inference | 本機 |
| `Qwen/Qwen2.5-0.5B-Instruct`（HuggingFace） | L2 實驗 + fine-tune | Colab T4 GPU |

---

## Regex 詳細說明

### 免費詞（`_FREE_WORDS`）
```
免費｜free｜請拿｜拿走｜多餘｜多的｜多出｜多出來｜送人｜不要了
剩食｜剩菜｜剩下｜剩餘｜拿去｜有需要｜帶走｜送出｜分享｜食安自負
```

### 食物詞（`_FOOD_WORDS`）
```
食物｜食品｜飯｜麵｜便當｜零食｜餅乾｜水果｜蔬菜｜菜｜湯｜肉｜蛋｜麵包｜吐司
料理｜點心｜糕｜餅｜粽｜飲料｜奶茶｜咖啡｜茶｜寶特瓶
三明治｜沙拉｜漢堡｜披薩｜壽司｜飯糰｜泡麵｜湯圓
餐盒｜餐點｜供餐｜美食｜buffet｜豆花｜午餐｜晚餐｜早餐
```

### 四個比對規則

| Pattern | 邏輯 | 在 `l1_score` 的效果 |
|---------|------|---------------------|
| `_PATTERN_FREE_FOOD` | 同時出現免費詞 + 食物詞（lookahead） | +0.80 |
| `_PATTERN_EVENT_FOOD` | 研討會/活動/系上 + 10字內有食物詞 | +0.60 |
| `_PATTERN_HAS_CATERING` | 「有/免費/提供」+「供餐」 | +0.50 |
| `_PATTERN_NOT_FOOD` | 「免費」+「課程/講座/票/演講…」 | → 直接返回 0（硬排除） |

> 分數可疊加，上限為 1.0。

---

## 整體判斷流程

### Production（freehiero，串聯架構）

```
貼文輸入
   │
   ├─► L1 (KeywordDetector)
   │     ├─ 確定是 ──────────────────────────────► True
   │     ├─ 確定不是 ─────────────────────────────► False
   │     └─ 不確定 ──► L2 (OllamaDetector)
   │                     ├─ 信心分數 > 0.5 ────────► True
   │                     └─ 信心分數 ≤ 0.5 ────────► False
```

### detfuse 研究架構（並聯 Fusion）

```
貼文輸入
   │
   ├─► L1 → l1_score() ──────────────┐
   │                                  ├─► α×s1 + (1-α)×s2 > threshold → bool
   └─► L2 → qwen_prob() ─────────────┘
```

兩層同時執行，結果加權合併，沒有先後順序或 veto 機制。

---

## 專案與 freehiero 的關係

```
detfuse  ←──── 研究、實驗、調參
  │
  │  找到最佳公式後，手動套回
  ▼
freehiero ────► 上線產品，不做實驗
```

兩者**不自動連動**，這是刻意的設計：產品穩定性優先。

## 目錄結構

```
detfuse/
  detector.py          原始偵測器（從 freehiero 複製，可自由修改）
  config.py            Ollama 連線設定
  data/
    categories/
      free_food/
        training_data.json  894 筆訓練資料（正 212 / 負 682，Claude 標注）
        test_data.json      234 筆測試資料（正 58 / 負 176，Claude 標注）
  experiments/
    fusion.py          L1/L2 融合公式研究主檔
    detector_eval.ipynb  Colab 評估 notebook（含 fine-tune）
  tests/
    test_detector.py   基本 regression test
  docs/
    overview.md        本文件（名詞定義、流程圖）
    architecture.md    偵測架構與公式設計
    experiments.md     如何跑實驗、解讀結果
  report/
    experiment/
      experiment_01.md 第一次實驗紀錄
```
