# detfuse

**detection + fusion** — 研究如何把 regex（規則）和 LLM（Qwen）的判斷結合成一個更準確的偵測公式。

## 為什麼存在

這個專案從 [freehiero](../../side-projects/freehiero/) 拆出來。

freehiero 是一個監測 FB 社團、偵測免費食物貼文並發送通知的產品。
其中偵測邏輯（detector.py）原本是「L1 判斷完交給 L2」的串行設計，
問題是 L1 犯錯時 L2 完全沒有發言權。

detfuse 的目標是研究出更好的融合公式，驗證後再手動套回 freehiero。

## 兩個專案的關係

```
detfuse  ←──── 研究、實驗、調參
  │
  │  找到最佳公式後，手動更新
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
    training_data.json 583 筆標注資料（正 350 / 負 233）
  experiments/
    fusion.py          L1/L2 融合公式研究主檔
    colab_detector_test.ipynb  原有 Colab 測試 notebook
  tests/
    test_detector.py   基本 regression test
  docs/
    overview.md        本文件
    architecture.md    偵測架構與公式設計
    experiments.md     如何跑實驗、解讀結果
```
