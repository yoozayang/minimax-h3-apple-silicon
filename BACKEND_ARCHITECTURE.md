# ⚙️ 色色 Studio 後端生成引擎架構與外插式規格說明 (Backend Architecture Specification)

本文件定義 **色色 Studio** 後端的架構設計、顯存管理機制、各生成引擎（圖片/影片/字幕）的標準介面協定（Interface Contract）。
**依據此架構規範，未來若需替換或新增「圖片生成引擎」或「影片生成引擎」，只需實現標準 Python 介面，完全不需要修改任何前端程式碼或 API 結構。**

---

## 🏛️ 後端分層架構總覽 (Architecture Diagram)

```text
+-----------------------------------------------------------------------------------------------+
| 前端 Web UI (Vanilla JS + HTML5 + CSS) [http://127.0.0.1:7860/]                               |
+-----------------------------------------------------------------------------------------------+
                                         │ JSON HTTP / Multipart / SSE Polling
                                         ▼
+-----------------------------------------------------------------------------------------------+
| FastAPI 路由層 (API Layer - scripts/web_ui.py)                                                |
|   • /api/image/generate  • /api/generate  • /api/job  • /api/queue  • /api/assets/upload ...   |
+-----------------------------------------------------------------------------------------------+
                                         │ 標準介面呼叫 (Standard Engine Contract)
                                         ▼
+-----------------------------------------------------------------------------------------------+
| 模型顯存生命週期管理層 (Model & Memory Manager - scripts/model_manager.py)                    |
|   • 32GB Unified Memory 互斥鎖 (IMAGE vs VIDEO 互斥常駐)                                       |
|   • 自動垃圾回收 (mlx.core.metal.clear_cache() + gc.collect())                                 |
+-----------------------------------------------------------------------------------------------+
          │                                      │                                      │
          ▼                                      ▼                                      ▼
+-----------------------+              +-----------------------+              +-----------------------+
| 🎨 圖片生成引擎       |              | 🎬 影片生成引擎       |              | 📝 字幕壓制引擎       |
| (scripts/image_engine)|              | (scripts/engine.py)   |              | (scripts/subtitles.py)|
| • FLUX.2-Klein-4B     |              | • MiniMax-Text-01 33B |              | • FFmpeg ASS / SRT    |
| • MFLUX MLX Native    |              | • I2V / Ref2V / Long  |              | • 自動字型解析縮放    |
+-----------------------+              +-----------------------+              +-----------------------+
          │                                      │                                      │
          └──────────────────────────────────────┴──────────────────────────────────────┘
                                                 ▼
+-----------------------------------------------------------------------------------------------+
| 本地儲存與持久化層 (Local Storage & State Persistence)                                        |
|   • 媒體檔案: outputs/ (或 Desktop/Downloads/Movies/自訂路徑)                                 |
|   • 資產登記: logs/assets.jsonl                                                               |
|   • 生成歷程: logs/image_history.jsonl & logs/history.jsonl                                   |
|   • 排程隊列: logs/queue.jsonl                                                                |
+-----------------------------------------------------------------------------------------------+
```

---

## 🔌 圖片生成引擎外插標準介面 (Pluggable Image Engine Contract)

為了實現「**替換圖片引擎完全不改動前端**」，圖片生成模組在 `scripts/image_engine.py` 中實現下列標準協定：

### 1. 模型註冊表 (Model Registry)：`IMAGE_MODELS`

後端統一維護模型註冊表，前端透過 `GET /api/image/models` 動態取得模型列表，不寫死前端邏輯：

```python
IMAGE_MODELS = {
    "krea-2": {
        "id": "krea-2",
        "display_name": "Krea 2 Turbo — Quality",
        "backend": "mlx_mflux",
        "supports_t2i": True,
        "supports_i2i": True,
        "supports_multi_reference": True,
        "supported_quantization": [4, 8],
        "recommended_profiles": ["draft", "balanced", "high", "maximum"],
        "default_profile": "high",
        "memory_requirement": "~8-12 GB",
        "is_default": True,
        "description": "高品質專用模型，細節細膩、光影層次豐富",
    },
    "flux2-klein-4b": {
        "id": "flux2-klein-4b",
        "display_name": "FLUX.2 Klein 4B — Fast",
        "backend": "mlx_mflux",
        "supports_t2i": True,
        "supports_i2i": True,
        "supports_multi_reference": False,
        "supported_quantization": [4, 8],
        "recommended_profiles": ["draft", "balanced", "high", "maximum"],
        "default_profile": "high",
        "memory_requirement": "~3-5 GB",
        "is_default": False,
        "description": "極速備用模型，4步快速構圖",
    },
}
```

### 2. 非線性品質設定檔解析器：`resolve_image_profile(...)`

提供 `Draft / Balanced / High / Maximum` 4 級品質設定，針對模型特性自適應最佳參數：

* **Krea 2 Turbo**：
  * `draft`：4 steps, 4-bit, 512×512, Guidance 1.0 (極速構圖)
  * `balanced`：6 steps, 4-bit, 768×768, Guidance 1.0 (效率平衡)
  * `high` (Default)：8 steps, 4-bit, 1024×1024, Guidance 1.0 (高品質正式產出)
  * `maximum`：12 steps, 8-bit, 1024×1024, Guidance 1.0 (極致畫質，精度優先)
* **FLUX.2 Klein 4B**：
  * `draft`：2 steps, 4-bit, 512×512
  * `balanced`：4 steps, 4-bit, 768×768
  * `high` (Default)：4 steps, 4-bit, 1024×1024
  * `maximum`：8 steps, 4-bit, 1024×1024

### 3. 核心生成函式：`generate_images(...)`

```python
def generate_images(
    prompt: str,
    width: int = 768,
    height: int = 768,
    steps: int = 4,
    seed: int = -1,
    model_name: str = "krea-2",
    quality_profile: str = "high",
    quantize: int = 4,
    count: int = 1,
    output_dir: str | Path | None = None,
    progress_callback: Callable[[float, str], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> list[ImageResult]:
    """
    【標準介面契約】
    1. 接收前端傳入之標準參數與 quality_profile。
    2. 自動確保顯存安全：呼叫 model_manager.switch_to_engine("IMAGE")。
    3. 執行圖片生成邏輯（本地模型、外部 API 或 ComfyUI）。
    4. 將圖片寫入指定之 output_dir（預設 outputs/images/）。
    5. 自動呼叫 register_asset(...) 登記至資產庫。
    6. 回傳標準 ImageResult 物件清單。
    """
```

#### 參數規格表：
| 參數名稱 | 類型 | 說明 | 預設值 |
| :--- | :--- | :--- | :--- |
| `prompt` | `str` | 提示詞描述（已去空白） | *(必填)* |
| `width` | `int` | 圖片寬度（自動對齊 16 的倍數） | `768` |
| `height` | `int` | 圖片高度（自動對齊 16 的倍數） | `768` |
| `steps` | `int` | 採樣步數 (Inference Steps，相容舊介面) | `4` |
| `seed` | `int` | 隨機種子（`-1` 代表隨機） | `-1` |
| `model_name` | `str` | 模型識別名稱 (`"krea-2"` / `"flux2-klein-4b"`) | `"krea-2"` |
| `quality_profile`| `str` | 品質檔位 (`"draft"` / `"balanced"` / `"high"` / `"maximum"`) | `"high"` |
| `quantize` | `int` | MLX 量化位元 (4 或 8) | `4` |
| `count` | `int` | 生成張數 (1 ~ 4，循序生成避免 OOM) | `1` |
| `output_dir` | `str \| None` | 儲存目標資料夾（若無則使用 outputs/images/） | `None` |
| `progress_callback` | `Callable` | 進度回報回呼：`callback(progress_float_0_to_1, stage_text)` | `None` |
| `cancel_check` | `Callable` | 中斷檢查函式：若回傳 `True` 則安全退出迴圈 | `None` |

---

### 2. 回傳資料結構：`ImageResult` Dataclass

無論底層替換成什麼引擎，回傳的清單元素必須符合 `ImageResult` 規範：

```python
@dataclass
class ImageResult:
    id: str                  # 唯一識別碼 (如 "a1b2c3d4")
    asset_id: str            # 資產庫 ID (如 "ast_e5f6g7h8")
    success: bool            # 是否成功 (True / False)
    output_path: str         # 輸出本機絕對路徑 (/Users/.../xxx.png)
    output_filename: str     # 輸出檔名 (xxx.png)
    prompt: str              # 提示詞
    seed: int                # 實際使用的 Seed
    width: int               # 寬度
    height: int              # 高度
    steps: int               # 步數
    model_name: str          # 模型名稱
    execution_time_sec: float# 耗時秒數
    created_at: str          # ISO 8601 時間戳記
    error_message: str | None = None # 錯誤訊息 (若失敗)
```

---

## 🛠️ 實戰教學：如何替換圖片生成引擎 (零前端修改)

### 情境 A：切換 MFLUX 支援的其他開源模型
若想更換為其他 MLX 模型（例如 `FLUX.2-Klein-Base-4B` 或其他量化版），只需在 `scripts/image_engine.py` 中的 `generate_images()` 調整模型名稱與加載器：
```python
from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein
from mflux.models.common.config.model_config import ModelConfig

# 僅需置換加載的模型設定檔：
cfg = ModelConfig.flux2_klein_base_4b()
_RESIDENT_FLUX_MODEL = Flux2Klein(quantize=4, model_config=cfg)
```

### 情境 B：替換為 SDXL-Lightning / Stable Diffusion
若想使用 `mlx-image` 或 `diffusers` 執行 SDXL，只需在 `generate_images()` 內部包裝：
```python
def generate_images(...):
    model_manager.switch_to_engine("IMAGE")
    
    # 1. 執行 SDXL 推理 (例如 4-step Lightning)
    image = sdxl_pipeline(prompt=prompt, width=width, height=height, num_inference_steps=steps)
    
    # 2. 儲存檔案並登記資產
    out_path = target_dir / filename
    image.save(out_path)
    asset = register_asset("IMAGE", "GENERATED", out_path, prompt=prompt)
    
    # 3. 回傳標準 ImageResult
    return [ImageResult(...)]
```

### 情境 C：橋接外部 API 或 ComfyUI 本地服務
若想透過 HTTP API 調用遠端 ComfyUI 或雲端 API 生成圖片：
```python
def generate_images(...):
    # 1. 向 ComfyUI / API 發送 POST
    response = requests.post("http://127.0.0.1:8188/prompt", json={...})
    img_bytes = download_output_image(response.json())
    
    # 2. 存入本地 target_dir
    with open(out_path, "wb") as f:
        f.write(img_bytes)
        
    # 3. 登記資產並回傳
    asset = register_asset("IMAGE", "GENERATED", out_path, prompt=prompt)
    return [ImageResult(...)]
```
> **前端完全不需要任何修改**，點擊「🖼️ 生成圖片」後，前端仍會正常顯示進度、將圖片加入「🖼️ 本機生成圖片庫」、並可一鍵填入「🎬 設起始幀」！

---

## 🔒 顯存生命週期管理 (`scripts/model_manager.py`)

為避免 Mac 在 32GB 統一記憶體下同時載入 33B 影片模型與 4B 圖片模型導致顯存溢出 (OOM) 崩潰，系統建立了**排他性常駐引擎鎖**：

1. **引擎註冊**：
   ```python
   # 各引擎在模組載入時註冊卸載回呼 (Unload Callback)
   model_manager.register_unload_callback("IMAGE", _unload_image_model)
   model_manager.register_unload_callback("VIDEO", _unload_video_model)
   ```
2. **自動切換與垃圾回收**：
   * 當用戶點擊「生成圖片」時，觸發 `switch_to_engine("IMAGE")`：
     - 若當前為 `VIDEO`，立即呼叫影片卸載回呼、釋放 Python 物件。
     - 執行 `mlx.core.metal.clear_cache()` 與 `gc.collect()` 立即向 macOS 歸還顯存。
     - 切換為 `IMAGE` 引擎執行生圖。
   * 當用戶點擊「生成影片」時，反向釋放圖片模型，安全載入影片去噪模型。

---

## 📡 後端 REST API 規格清單

| 端點 (Endpoint) | 方法 (Method) | 請求內容 (Request Payload) | 回應內容 (Response) | 說明 |
| :--- | :--- | :--- | :--- | :--- |
| `/api/status` | `GET` | 無 | `{ memory: {...}, default_output_dir: "...", ... }` | 查詢系統 RAM、當前常駐引擎與預設路徑 |
| `/api/job` | `GET` | 無 | `{ is_running: bool, stage: str, progress: float, ... }` | 輪詢當前正在執行的任務進度與結果 |
| `/api/image/generate` | `POST` | `{ prompt, width, height, steps, count, output_dir }` | `{ status: "ok", results: [...] }` | 觸發本地圖片生成 |
| `/api/image/history` | `GET` | 無 | `[ { id, path, prompt, created_at }, ... ]` | 取得最近生成的圖片歷史清單 |
| `/api/generate` | `POST` | `{ prompt, profile, width, height, duration_sec, steps, mode, start_image, output_dir }` | `{ status: "started" }` | 啟動影片生成工作 |
| `/api/generate/cancel` | `POST` | 無 | `{ status: "cancelling" }` | 安全中斷當前生成工作並釋放顯存 |
| `/api/queue` | `GET` | 無 | `{ auto_enabled: bool, items: [...] }` | 取得目前排程隊列清單 |
| `/api/queue/batch-add` | `POST` | `{ prompts_text, profile, width, height, duration_sec, steps, output_dir }` | `{ status: "ok", added_count: N }` | 批次加入提示詞至排程隊列 |
| `/api/queue/action` | `POST` | `{ item_id: str, action: "retry" \| "delete" \| "run_now" }` | `{ status: "ok", items: [...] }` | 操作排程中的特定工作 |
| `/api/history` | `GET` | 無 | `[ { prompt, output_path, width, height, duration_sec, ... }, ... ]` | 取得已完成影片的歷史紀錄清單 |
| `/api/assets/upload` | `POST` | `FormData (file: Binary)` | `{ status: "ok", path: "/Users/.../xxx.png", filename: "..." }` | 上傳圖生影之本機起始圖片 |
| `/api/subtitles/burn` | `POST` | `{ video_path, text, style, position }` | `{ status: "ok", output_path: "..." }` | 執行 FFmpeg 智能字幕壓制 |
| `/api/open-folder` | `POST` | `{ dir_path: str }` 或 `{ file_path: str }` | `{ status: "ok" }` | 呼叫 macOS 原生 Finder 開啟目標目錄 |
| `/api/video-stream` | `GET` | `?path=/Users/.../file.mp4` | `FileResponse (Binary Stream with MIME)` | 高效串流本機影片與圖片縮圖 |
