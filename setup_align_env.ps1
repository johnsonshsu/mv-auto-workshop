# 建立字幕對齊用的 Python 環境（打包版才需要）。在 MVAutoWorkshop.exe 所在的資料夾執行一次：
#     .\setup_align_env.ps1
#
# 會在這個資料夾建立 align-env，安裝 torch（CUDA 版）與 stable-ts，約 3 GB。
# 需要先安裝 Python 3.11 以上（https://www.python.org/downloads/）。
# 第一次執行字幕階段時，還會下載 Whisper large-v3 模型（約 3 GB）到 %USERPROFILE%\.cache\whisper。

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$launcher = Get-Command py -ErrorAction SilentlyContinue
if ($launcher) { & py -3 -m venv align-env } else { & python -m venv align-env }
if (-not (Test-Path "align-env\Scripts\python.exe")) { throw "建立虛擬環境失敗；請確認已安裝 Python 3.11 以上" }

$python = "align-env\Scripts\python.exe"
& $python -m pip install --quiet --upgrade pip
& $python -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu130
if ($LASTEXITCODE -ne 0) { throw "安裝 torch 失敗" }
& $python -m pip install stable-ts
if ($LASTEXITCODE -ne 0) { throw "安裝 stable-ts 失敗" }

& $python -c "import torch, stable_whisper; print('torch', torch.__version__, '| CUDA 可用：', torch.cuda.is_available())"
"完成。現在可以啟動 MVAutoWorkshop.exe。"
