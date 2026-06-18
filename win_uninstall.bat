@echo off
REM ==========================================
REM  ASCEC UNINSTALLER (Windows)
REM ==========================================
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -Command ^
  "& { $bat = [System.IO.File]::ReadAllText('%~f0'); $m = [char]35 + [char]95 + [char]95 + [char]80 + [char]83 + [char]95 + [char]95; $i = $bat.IndexOf($m); if ($i -lt 0) { Write-Error "Marker not found."; exit 1 }; $ps = $bat.Substring($i + $m.Length); Invoke-Expression $ps }"
set "EC=%ERRORLEVEL%"
echo.
echo Press any key to close this window...
pause >nul
exit /b %EC%

REM ===== Lines below are NEVER read by cmd (exit /b above) =====
#__PS__
$ErrorActionPreference = "Stop"

# Reverses what win_install.bat set up, WITHOUT touching conda itself:
#   - Removes the 'py11' conda environment (leaves base conda alone)
#   - Removes the cloned source directory (%USERPROFILE%\software\ascec04)
#   - Removes the ascec.cmd / cosmic.cmd launchers from %USERPROFILE%\bin
#   - Removes %USERPROFILE%\bin from the user PATH (only if it was added)
#
# Conda is intentionally left installed. Pass -y to skip the confirmation.

$ENV_NAME      = "py11"
$TARGET_DIR    = Join-Path $env:USERPROFILE "software\ascec04"
$LAUNCHER_DIR  = Join-Path $env:USERPROFILE "bin"

$assumeYes = $args -contains "-y" -or $args -contains "--yes"

Write-Host "> ASCEC uninstaller (Windows)" -ForegroundColor Cyan
Write-Host ">"
Write-Host "> This will:"
Write-Host ">   - Remove the conda environment '$ENV_NAME' (conda itself is kept)"
Write-Host ">   - Remove the source directory $TARGET_DIR"
Write-Host ">   - Remove ascec.cmd / cosmic.cmd from $LAUNCHER_DIR"
Write-Host ">   - Remove $LAUNCHER_DIR from your user PATH (if it was added)"
Write-Host ">"

if (-not $assumeYes) {
    $reply = Read-Host "> Proceed? [y/N]"
    if ($reply -notmatch '^(y|Y|yes|YES)$') {
        Write-Host "> Aborted."
        exit 0
    }
}

# -----------------------------------
# 1. Remove the conda env (not conda)
# -----------------------------------
$MINICONDA_DIR = $null
$candidates = @(
    (Join-Path $env:USERPROFILE "Miniconda3"),
    (Join-Path $env:USERPROFILE "Anaconda3"),
    (Join-Path $env:USERPROFILE "miniforge3"),
    (Join-Path $env:USERPROFILE "mambaforge"),
    "C:\ProgramData\Miniconda3",
    "C:\ProgramData\Anaconda3"
)
foreach ($c in $candidates) {
    if (Test-Path (Join-Path $c "Scripts\conda.exe")) { $MINICONDA_DIR = $c; break }
}

if ($MINICONDA_DIR) {
    $CONDA_EXE = Join-Path $MINICONDA_DIR "Scripts\conda.exe"
    $envList = & $CONDA_EXE env list
    if ($envList -match "(?m)^\s*$ENV_NAME\s") {
        Write-Host "> Removing conda environment '$ENV_NAME'..."
        & $CONDA_EXE env remove -n $ENV_NAME -y
    } else {
        Write-Host "> Conda environment '$ENV_NAME' not found - nothing to remove."
    }
} else {
    Write-Host "  WARNING: conda not found in common locations." -ForegroundColor Yellow
    Write-Host "  Skipping env removal. If '$ENV_NAME' exists, remove it manually:"
    Write-Host "    conda env remove -n $ENV_NAME"
}

# -----------------------------------
# 2. Remove launcher scripts
# -----------------------------------
foreach ($cmd in @("ascec.cmd", "cosmic.cmd")) {
    $p = Join-Path $LAUNCHER_DIR $cmd
    if (Test-Path $p) {
        Write-Host "> Removing launcher $p..."
        Remove-Item $p -Force
    }
}

# -----------------------------------
# 3. Remove launcher dir from user PATH
# -----------------------------------
$userPath = [Environment]::GetEnvironmentVariable("Path", "User")
if ($userPath) {
    $parts = $userPath -split ";" | Where-Object { $_ -and ($_ -ine $LAUNCHER_DIR) }
    $newUserPath = ($parts -join ";")
    if ($newUserPath -ne $userPath) {
        [Environment]::SetEnvironmentVariable("Path", $newUserPath, "User")
        Write-Host "> Removed $LAUNCHER_DIR from your user PATH (takes effect in new shells)."
    }
}

# -----------------------------------
# 4. Remove the source directory
# -----------------------------------
if (Test-Path $TARGET_DIR) {
    Write-Host "> Removing source directory $TARGET_DIR..."
    Remove-Item $TARGET_DIR -Recurse -Force
} else {
    Write-Host "> Source directory $TARGET_DIR not found - nothing to remove."
}

Write-Host "-------------------------------------------------------" -ForegroundColor Green
Write-Host "> UNINSTALL COMPLETE!" -ForegroundColor Green
Write-Host "> Conda was left installed."
Write-Host "> Open a NEW terminal so the PATH change takes effect."
Write-Host "-------------------------------------------------------" -ForegroundColor Green
