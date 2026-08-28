#!/usr/bin/env bash
# ============================================================
# 🎬 色色 Studio (Local Generative Studio) 啟動器
# ============================================================

set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PYTHON="$DIR/.venv/bin/python"

if [ ! -f "$VENV_PYTHON" ]; then
  echo "❌ 找不到 Python 虛擬環境: $VENV_PYTHON"
  exit 1
fi

echo "🚀 正在啟動 色色 Studio (MiniMax-H3 & FLUX Studio)..."
cd "$DIR"
exec "$VENV_PYTHON" "$DIR/scripts/web_ui.py" "$@"
