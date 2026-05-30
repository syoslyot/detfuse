# detfuse

## Playwright 瀏覽器設定

這個專案的 `.claude/settings.json` 設定 Playwright MCP 透過 CDP 連線到現有 Chrome 視窗，而不是開新的瀏覽器（避免沒有登入 Google 帳號的問題）。

### 開啟

```bash
google-chrome --remote-debugging-port=9222 &
```

之後所有 `browser_navigate` 等操作都會在那個視窗裡開新分頁。

### 關閉

直接關掉 Chrome 視窗，或從終端機：

```bash
pkill -f "remote-debugging-port=9222"
```

### 其他專案不受影響

其他專案的 Playwright 沒有 `--cdp-endpoint` 設定，永遠開自己的新瀏覽器，不管 Chrome debug port 有沒有在跑都不受影響。

> `.claude/` 已加入 `.gitignore`，本機環境設定不會被 push 出去。
