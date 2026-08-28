# 🛡️ 介面零破壞與架構遵循原則 (Workspace Rules)

在執行本專案（色色 Studio / Minimax Apple Silicon）的任何編程或維護工作時，AI Agent 必須始終遵守以下規則：

1. **開發前必讀架構文檔**：
   在修改任何程式碼前，必須先使用 `view_file` 研讀專案根目錄的 `FRONTEND_LAYOUT.md` 與 `BACKEND_ARCHITECTURE.md`。

2. **零介面破壞**：
   嚴禁任意刪除、隱藏或修改現有的 UI 區塊（`HEADER`、`L1~L6`、`R1~R2`、`BOT`）。新增功能只能以最小侵入式方式嵌入對應區塊（例如在 `L2` 或 `L5`）。

3. **後端外插契約**：
   圖片生成永遠透過 `scripts/image_engine.py` 的標準 `generate_images(...)` 介面與 `IMAGE_MODELS` 註冊表擴充，絕不破壞前端解耦。嚴禁 Silent Fallback。

4. **文檔即時同步**：
   完成功能修改後，必須同步更新 `FRONTEND_LAYOUT.md` 與 `BACKEND_ARCHITECTURE.md`，並提交 Git。
