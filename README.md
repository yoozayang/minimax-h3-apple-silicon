# MiniMax-H3 on Apple Silicon (MLX Native)

MiniMax-H3 本地音訊與視訊端到端生成系統，針對 Apple Silicon (M1/M2/M3/M4 系列，含 32GB 統一記憶體機型) 進行深度的階段式記憶體管理（Phase-scoped Memory Residency）與 8-bit 量化調校。

---

## 🌟 特性亮點

- **純 MLX 原生推論**：無需 PyTorch / CUDA，直接利用 Apple Silicon Metal 統一記憶體架構（Unified Memory）。
- **階段式記憶體釋放**：將 50 層 Qwen3-VL 文字編碼器、33B DiT 擴散模型、Video VAE (FP16) 與 Audio VAE (FP32) 拆解為獨立執行階段，各階段結束後完全釋放記憶體，將 32GB RAM 機型的 Peak Memory 壓制在 **27.0 GB** 以內，達成 **0 次 OOM / 0 Crash** 的穩定生成。
- **一鍵啟動指令 `色色`**：終端機直接輸入 `色色` 或雙擊桌面 `色色.app` 捷徑，自動喚起現代深色玻璃風格 Web UI 與瀏覽器。
- **雙介面架構**：CLI 與 Web UI 共用同一套底層生成引擎 (`engine.py`)，保證行為與歷史紀錄完全一致。
- **彈性秒數與自訂儲存位置**：支援手動輸入任意秒數（最高支援架構上限 15.0 秒 / 362 幀），並支援將成品一鍵儲存至桌面、下載或自訂目錄。
- **完整音視訊同步**：生成 H.264 24fps 影像並同步輸出 32kHz AAC 雙聲道音訊，自動透過 FFmpeg 完成無損封裝。

---

## 🚀 快速啟動

### 1. 桌面捷徑啟動
在 macOS 桌面直接雙擊 **`色色.app`**，或使用 Spotlight (<kbd>⌘</kbd> + <kbd>Space</kbd>) 搜尋 `色色`。

### 2. 命令列啟動
在終端機任意目錄輸入：
```bash
色色
```
預設將會啟動本機 Web Studio 伺服器並自動於瀏覽器開啟 `http://127.0.0.1:7860`。

### 3. CLI 快速生成指令
```bash
# 進入互動式 CLI 提示詞輸入模式
色色 --cli

# 直接快速生成 (768x448, 10 steps, ~2s 影片)
色色 --fast "A corgi running through a vibrant grassy field, cinematic lighting"

# 生成 720p 影片 (1280x720, 20 steps, ~3s 影片)
色色 --720 "A majestic dragon flying over snowy mountains"

# 指定隨機種子
色色 --seed 42 "Cyberpunk city street at night, neon lights"
```

---

## 📊 實測基準數據 (Apple M1 Pro 32GB)

| 階段 | 組件與精度 | 磁碟大小 | 執行中活躍記憶體 (Active RAM) | 階段耗時 (768×448, 10 steps) | 階段結束釋放後 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Phase 1** | Qwen3-VL Text Encoder (8-bit) | 27.73 GB | **26.99 GB** | ~1.8 min | **0.00 GB** |
| **Phase 2** | MiniMax-H3 DiT (8-bit) | 34.75 GB | **21.20 GB** | ~19.2 min (~115s/step) | **0.00 GB** |
| **Phase 3** | Video VAE (FP16) | 4.85 GB | **4.60 GB** | ~0.4 min | **0.00 GB** |
| **Phase 4** | Audio VAE (FP32) | 0.56 GB | **0.30 GB** | < 0.1 min | **0.00 GB** |
| **Phase 5** | FFmpeg H.264/AAC Muxer | — | < 0.1 GB | < 0.1 min | **0.00 GB** |

* **Smoke Test 結果**：768×448, 16 幀, 10 steps, seed 42 耗時 **21.7 分鐘** 穩定產出。

---

## 📂 專案結構

```text
/Users/yoozayang/Minimax/
├── .venv/                   # Python 3.13 虛擬環境 (MLX 0.32.2)
├── models/                  # 8-bit MLX 權重庫 (~68 GB)
│   ├── tokenizer/
│   ├── mlx-8bit/
│   └── bf16/vae/
├── outputs/                 # 影片輸出目錄 (MP4)
├── logs/                    # 生成歷史紀錄 (history.jsonl)
├── scripts/                 # 核心腳本
│   ├── engine.py            # 共用生成引擎後端
│   ├── web_ui.py            # 本地 Web Studio 伺服器
│   └── download_models.py   # 模型自動下載校驗工具
├── repo/                    # mlx_h3 模組源碼
├── benchmark.md             # 基準測試完整分析報告
└── README.md
```

---

## 📜 授權與致謝
* 核心模型架構基於 MiniMax-H3 與 `appautomaton/mlx-h3`。
* 適用於 Apple Silicon 系列 Mac 本機影片實驗與創作。
