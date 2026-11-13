# 把 MV 自動工坊打包成 Windows 執行檔。在專案資料夾執行：  .\build.ps1
#
# 產出 dist\MVAutoWorkshop\MVAutoWorkshop.exe，旁邊附上可以直接編輯的 config、workflows、prompts。
# torch 與 stable-ts（字幕對齊用，約 5 GB）沒有包進執行檔；到目標電腦後執行一次
# setup_align_env.ps1，就會在執行檔旁邊建立對齊用的 Python 環境。

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

# ---- 檢查 FFmpeg：沒有安裝或無法執行時用 winget 安裝，並加入全域 Path ----
function Test-FFmpeg {
    foreach ($tool in "ffmpeg", "ffprobe") {
        $command = Get-Command $tool -ErrorAction SilentlyContinue
        if (-not $command) { return $false }
        & $command.Source -version *> $null
        if ($LASTEXITCODE -ne 0) { return $false }
    }
    return $true
}

function Add-ToGlobalPath([string]$folder) {
    $isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()
        ).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
    # 優先寫入系統（Machine）Path；沒有系統管理員權限時改寫入使用者 Path
    $scope = if ($isAdmin) { "Machine" } else { "User" }
    $current = [Environment]::GetEnvironmentVariable("Path", $scope)
    $entries = @($current -split ";" | Where-Object { $_ })
    if ($entries -notcontains $folder) {
        [Environment]::SetEnvironmentVariable("Path", (($entries + $folder) -join ";"), $scope)
    }
    if ($scope -eq "User") {
        Write-Warning "目前不是系統管理員，FFmpeg 只加入『使用者』Path。要加入系統（全域）Path，請用系統管理員身分重新執行 build.ps1。"
    }
    # 讓目前這個視窗立刻生效；已開啟的其他視窗要重新開啟才會看到
    if (($env:Path -split ";") -notcontains $folder) { $env:Path = "$env:Path;$folder" }
}

if (-not (Test-FFmpeg)) {
    "找不到可以執行的 FFmpeg（ffmpeg、ffprobe），使用 winget 安裝中..."
    if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
        throw "找不到 winget，無法自動安裝 FFmpeg。請手動安裝 FFmpeg 並把它的 bin 資料夾加入 Path（https://www.gyan.dev/ffmpeg/builds/）"
    }
    winget install --id Gyan.FFmpeg -e --accept-source-agreements --accept-package-agreements
    if ($LASTEXITCODE -ne 0 -and -not (Test-FFmpeg)) { throw "用 winget 安裝 FFmpeg 失敗" }

    # winget 的安裝位置依套件版本而異：先重新讀取 Path，找不到再到常見位置搜尋
    $env:Path = ([Environment]::GetEnvironmentVariable("Path", "Machine"), [Environment]::GetEnvironmentVariable("Path", "User") -join ";")
    if (-not (Test-FFmpeg)) {
        $roots = @("$env:LOCALAPPDATA\Microsoft\WinGet\Packages", "$env:ProgramFiles\ffmpeg", "$env:ProgramFiles\WinGet\Packages") |
            Where-Object { Test-Path $_ }
        $found = $roots | ForEach-Object { Get-ChildItem $_ -Recurse -Filter ffmpeg.exe -ErrorAction SilentlyContinue } |
            Where-Object { Test-Path (Join-Path $_.DirectoryName "ffprobe.exe") } | Select-Object -First 1
        if (-not $found) { throw "FFmpeg 安裝完成，但找不到 ffmpeg.exe 與 ffprobe.exe；請手動把 FFmpeg 的 bin 資料夾加入 Path" }
        Add-ToGlobalPath $found.DirectoryName
    }
    if (-not (Test-FFmpeg)) { throw "FFmpeg 已安裝但仍無法執行，請重新開啟終端機後再試" }
}
$ffmpegVersion = ((& ffmpeg -version | Select-Object -First 1) -split " ")[2]
"FFmpeg 可用（版本 $ffmpegVersion）"

$python = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    # 第一次使用（例如剛 clone 專案）：自動建立虛擬環境並安裝執行所需的套件（不含 torch，字幕對齊另由 setup_align_env.ps1 處理）
    "找不到 .venv，自動建立中..."
    $launcher = Get-Command py -ErrorAction SilentlyContinue
    if ($launcher) { & py -3.11 -m venv .venv } else { & python -m venv .venv }
    if (-not (Test-Path $python)) { throw "建立 .venv 失敗；請確認已安裝 Python 3.11 以上" }
    & $python -m pip install PySide6 httpx websocket-client keyring Pillow numpy soundfile pytest opencc-python-reimplemented
    if ($LASTEXITCODE -ne 0) { throw "安裝套件失敗" }
}

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
