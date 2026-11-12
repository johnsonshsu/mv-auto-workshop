# 把 MV 自動工坊打包成 Windows 執行檔。在專案資料夾執行：  .\build.ps1
#
# 產出 dist\MVAutoWorkshop\MVAutoWorkshop.exe，旁邊附上可以直接編輯的 config、workflows、prompts。
# torch 與 stable-ts（字幕對齊用，約 5 GB）沒有包進執行檔；到目標電腦後執行一次
# setup_align_env.ps1，就會在執行檔旁邊建立對齊用的 Python 環境。

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$python = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $python)) { throw "找不到 .venv，請先依 README 建立虛擬環境" }

# 執行檔開著的時候不能重新打包：Windows 會鎖住正在使用的檔案，刪到一半會留下壞掉的 dist
$running = Get-Process MVAutoWorkshop -ErrorAction SilentlyContinue |
    Where-Object { $_.Path -and $_.Path.StartsWith((Join-Path $PSScriptRoot "dist"), [StringComparison]::OrdinalIgnoreCase) }
if ($running) { throw "dist 裡的 MVAutoWorkshop.exe 正在執行，請先關閉它再重新打包" }

& $python -m pip install --quiet pyinstaller
if ($LASTEXITCODE -ne 0) { throw "安裝 PyInstaller 失敗" }

# 先打包到 build\stage，成功後再同步到 dist。直接清空 dist 的話，只要有檔案總管視窗開在那裡就會失敗，
# 而且失敗時會留下刪到一半的 dist。
Remove-Item -Recurse -Force build -ErrorAction SilentlyContinue

& $python -m PyInstaller --noconfirm --clean --windowed --name MVAutoWorkshop `
    --collect-data opencc `
    --hidden-import keyring.backends.Windows `
    --exclude-module torch --exclude-module torchaudio --exclude-module stable_whisper `
    --exclude-module whisper --exclude-module numba --exclude-module llvmlite `
    --distpath build\stage --exclude-module pytest --exclude-module tkinter `
    app\main.py
if ($LASTEXITCODE -ne 0) { throw "PyInstaller 打包失敗" }

$dist = "build\stage\MVAutoWorkshop"
# 設定、工作流範本、提示詞放在執行檔旁邊，使用者可以直接編輯
foreach ($folder in "config", "workflows", "prompts") {
    Copy-Item -Recurse -Force $folder (Join-Path $dist $folder)
}
New-Item -ItemType Directory -Force (Join-Path $dist "tools") | Out-Null
Copy-Item -Force app\services\align_worker.py (Join-Path $dist "tools\align_worker.py")
Copy-Item -Force setup_align_env.ps1, README.md $dist

# 同步到 dist（/MIR 會讓 dist 與剛打包的內容完全一致，但保留使用者另外建立的 align-env）
$final = "dist\MVAutoWorkshop"
New-Item -ItemType Directory -Force $final | Out-Null
robocopy $dist $final /MIR /XD align-env /NFL /NDL /NJH /NJS /NP | Out-Null
if ($LASTEXITCODE -ge 8) { throw "同步到 $final 失敗（robocopy 結束碼 $LASTEXITCODE）" }
$dist = $final
$size = (Get-ChildItem -Recurse $dist | Measure-Object Length -Sum).Sum / 1MB
"打包完成：$dist（{0:N0} MB）" -f $size
