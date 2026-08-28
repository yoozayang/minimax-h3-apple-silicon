---
name: interface-preservation-and-arch-check
description: Mandatory workflow for the Minimax / 色色 Studio project. Enforces zero UI regression, strict adherence to FRONTEND_LAYOUT.md block definitions, mandatory pre-reading of FRONTEND_LAYOUT.md and BACKEND_ARCHITECTURE.md before any edits, and preserving pluggable backend engine contracts.
---

# 🛡️ 色色 Studio：介面零破壞與前後端架構規範遵循指南 (Skill)

本 Skill 定義「色色 Studio (Minimax Apple Silicon)」專案在進行任何程式碼修改、模型替換、功能增修時必須嚴格遵守的核心原則與標準作業程序 (SOP)。

---

## 🛑 第一條鐵律：開發前必讀規格文檔 (Mandatory Pre-Reading)

在進行任何前端 HTML/CSS/JS 或後端 Python 程式碼修改前，**必須先使用 `view_file` 閱讀專案根目錄下的兩份架構規格文檔**：

1. [`/Users/yoozayang/Minimax/FRONTEND_LAYOUT.md`](file:///Users/yoozayang/Minimax/FRONTEND_LAYOUT.md)
   * 掌握前端全域佈局架構與所有區塊代號：
     * **`HEADER`**：頂部狀態列（Logo、記憶體水位、系統狀態燈）
     * **`L1`**：模態切換（文生影 / 圖生影 I2V / 參考影 Ref2V）與提示詞輸入框
     * **`L2`**：圖片模型選單、品質拉桿（Draft / Balanced / High / Maximum）與核心動作按鈕
     * **`L3`**：發光即時進度卡片（階段說明、計時、進度條、🛑 中止生成）
     * **`L4`**：本機生成圖片庫（縮圖、🎬 設起始幀、🏷️ 設參考、📂 Finder）
     * **`L5`**：影片規格與產出儲存位置抽屜（自訂路徑、Finder 開啟、快速標籤、HF Token）
     * **`L6`**：排程隊列抽屜（自動連續生成、任務卡片、播放/重試/刪除）
     * **`R1`**：影片播放中心（HTML5 Player、循環、倍速、規格資訊）
     * **`R2`**：智能字幕壓制抽屜（台詞文字框、風格選單、位置選單、壓制按鈕）
     * **`BOT-HDR / BOT-GRID / BOT-LIST`**：歷史紀錄展示櫃（網格大縮圖、列表、懸浮預覽、隱藏/還原）
2. [`/Users/yoozayang/Minimax/BACKEND_ARCHITECTURE.md`](file:///Users/yoozayang/Minimax/BACKEND_ARCHITECTURE.md)
   * 掌握後端外插式標準協定：
     * **`IMAGE_MODELS` 註冊表**：動態提供模型清單，不寫死前端。
     * **`resolve_image_profile` 解析器**：模型專屬非線性品質設定檔。
     * **`generate_images(...)` 標準契約**：統一輸入參數與 `ImageResult` 回傳結構。
     * **`model_manager` 顯存管理**：`IMAGE` vs `VIDEO` 互斥鎖、Resident 模型快取與安全 Unload。

---

## 🔒 第二條鐵律：零介面破壞原則 (Zero UI Regression)

1. **除非使用者明確指示移除或調整，否則嚴禁擅自刪除、隱藏或破壞任何現有介面元件與版面。**
2. 新增功能時：
   * 必須將新功能歸類至適當的區塊代號（例如在 `L2` 或 `L5`）。
   * 採用最小侵入式設計，維持深色玻璃擬態 (Glassmorphism) 與俐落排版。
3. 程式碼修改完成後，必須進行自我檢查清單 (Checklist)：
   * [ ] `HEADER` 記憶體監控與狀態燈正常
   * [ ] `L1` 模態切換（文生影 / 圖生影 / 參考影）與上傳預覽正常
   * [ ] `L2` 動作按鈕、模型選單與品質分段拉桿完整存在
   * [ ] `L3` 生成進度卡片、計時與中止按鈕正常
   * [ ] `L4` 圖片庫能正常顯示縮圖，且具備 `🎬 設起始幀`、`🏷️ 設參考`、`📂 Finder`
   * [ ] `L5` 輸出目錄輸入框、Finder 按鈕、快速標籤與 HF Token 輸入框完整保留
   * [ ] `L6` 佇列卡片、自動連續生成正常
   * [ ] `R1` 播放器、倍速、循環正常
   * [ ] `R2` 字幕壓制抽屜正常
   * [ ] `BOT` 歷史展示櫃（網格/列表/隱藏還原）正常

---

## ⚙️ 第三條鐵律：後端外插式引擎規範 (Pluggable Backend Contract)

1. **替換或新增圖片/影片引擎時，絕不更改前端呼叫架構**：
   * 圖片引擎統一實作在 `scripts/image_engine.py` 的 `generate_images(...)`。
   * 透過 `IMAGE_MODELS` 註冊表擴充，前端透過 `GET /api/image/models` 自動感知新模型。
2. **顯存生命週期管理**：
   * 嚴格執行 `model_manager.switch_to_engine(...)`。
   * 模型切換時執行完整資源釋放：`del model` → `gc.collect()` → `mlx.core.metal.clear_cache()`。
   * 同模型連續生成時保持常駐（Resident），不重複 reload 權重。
3. **拒絕 Silent Fallback**：
   * 若模型遇到未授權（如 Hugging Face Gated Repo 401）、權重未下載或 OOM，介面必須呈現真實原因與操作指引，嚴禁偷偷切換其他模型冒充。

---

## 📝 第四條鐵律：文檔即時同步更新 (Documentation Sync)

任何功能增修、元件調整或模型註冊表更新完成後，必須立即更新：
1. `FRONTEND_LAYOUT.md`：記錄新的 DOM ID、元件位置與功能說明。
2. `BACKEND_ARCHITECTURE.md`：記錄新的 API 規格、模型註冊表與參數映射。
3. 透過 Git 提交更新紀錄。
