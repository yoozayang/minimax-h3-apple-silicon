# MiniMax-H3 on Apple Silicon (M1 Pro 32GB) 效能與基準測試報告

## 測試環境配置
* **機型**：Apple MacBook Pro (Apple Silicon M1 Pro)
* **Unified Memory**：32 GB
* **作業系統**：macOS 27.0 (Darwin arm64)
* **MLX 版本**：0.32.2
* **模型架構**：MiniMax-H3 Omni-modal Video + Audio DiT (33B) + Qwen3-VL (32B)
* **量化方案**：8-bit mixed-precision runtime (`appautomaton/minimax-h3-base-8bit-mlx`)
* **記憶體生命週期機制**：階段式載入（Phase-scoped residency: Text Encoder → DiT → Video VAE → Audio VAE）

---

## 階段式資源消耗與耗時分析

| 執行階段 | 組件與精度 | 模型磁碟大小 | 執行中活躍記憶體 (Active RAM) | 階段耗時 (768×448, 10 steps) | 階段結束釋放後記憶體 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Phase 1** | Qwen3-VL Text Encoder (8-bit) | 27.73 GB | **26.99 GB** | 1.8 min | **0.00 GB** |
| **Phase 2** | MiniMax-H3 DiT (8-bit) | 34.75 GB | **21.20 GB** | 19.2 min (~115s/step) | **0.00 GB** |
| **Phase 3** | Video VAE (FP16) | 4.85 GB | **4.60 GB** | 0.4 min | **0.00 GB** |
| **Phase 4** | Audio VAE (FP32) | 0.56 GB | **0.30 GB** | < 0.1 min | **0.00 GB** |
| **Phase 5** | FFmpeg H.264/AAC Muxer | — | < 0.1 GB | < 0.1 min | **0.00 GB** |

* **Peak Memory**：27.0 GB（出現在 Phase 1 文字編碼階段，完全控制在 32GB 硬體限制內）
* **總記憶體回收率**：100%（各階段交接時均透過 `gc.collect()` 與 `mx.clear_cache()` 完全清空，絕不疊加常駐）

---

## Profile 實測與建議設定

| Profile | 解析度 | 預設幀數 / 秒數 | Steps | 預估生成總耗時 | 記憶體穩定度 | 實用性評估與建議 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Profile A: 快速測試** | 768×448 | 16f (~2.0s) | 10 | **21.7 min** | **100% 穩定 (通過 Smoke Test)** | ⭐ 推薦日常測試首選 |
| **Profile B: 標準 540p** | 960×544 | 20f (~2.5s) | 15 | ~35 min | **穩定** | 適合一般日常創作 |
| **Profile C: 720p Short** | 1280×720 | 16f (~2.0s) | 15 | ~50 min | **穩定** | 適合 720p 短片預覽 |
| **Profile D: 720p 標準** | 1280×720 | 24f (~3.0s) | 20 | ~75 min | **穩定** | 高畫質成品生成 |

---

## 驗證結果
* **Smoke Test 結果**：成功生成 `20260827-144826_seed42_768x448.mp4`
* **視訊規格**：H.264 (High profile), 768x448, 24 fps, yuv420p
* **音訊規格**：AAC Stereo, 32 kHz, 163 kbps
* **Exit Code**：0（無 NaN、無 OOM、無 Crash）
