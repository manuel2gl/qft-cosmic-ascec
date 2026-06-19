@echo off
REM ==========================================
REM  ASCEC ONE-CLICK INSTALLER (Windows)
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

$INSTALL_PY11          = $true
$TARGET_DIR            = Join-Path $env:USERPROFILE "software\ascec04"
$DEFAULT_MINICONDA_DIR = Join-Path $env:USERPROFILE "Miniconda3"
$REPO_URL              = "https://github.com/manuel2gl/qft-cosmic-ascec.git"

# Run a native command whose stderr is benign and return its combined output as
# plain text WITHOUT aborting the script.
#
# Why this exists: many of the tools we shell out to (xtb, conda, git) print
# progress -- and even success banners like "normal termination of xtb" -- to
# stderr. Under $ErrorActionPreference='Stop', merging that stderr with 2>&1
# turns the first stderr line into a *terminating* error that kills the whole
# script. That is the bug that aborted earlier installs at the "xtb --version"
# check, BEFORE the ascec/cosmic launchers and the PATH entry were ever written
# (hence "'ascec' is not recognized" and the aliases not working). Capturing
# combined output through here keeps Stop semantics for real failures while
# tolerating chatty-but-successful native commands.
function Invoke-Native {
    param([scriptblock]$Action)
    $prev = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try   { & $Action 2>&1 | Out-String }
    finally { $ErrorActionPreference = $prev }
}

Write-Host "> Starting ASCEC One-Click Installation (Windows)..." -ForegroundColor Cyan

Write-Host "> Setting up directories at $TARGET_DIR..."
New-Item -ItemType Directory -Path $TARGET_DIR -Force | Out-Null

# ---------------------------------------------------------------------------
# 1. Bootstrap conda FIRST. On Windows there is no system package manager, so
#    conda doubles as the tool that installs git for us (see step 2) -- the
#    Windows analogue of the Linux installer's apt/dnf/yum/... git install.
# ---------------------------------------------------------------------------
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

if (-not $MINICONDA_DIR) {
    Write-Host "> Conda not found. Installing Miniconda to $DEFAULT_MINICONDA_DIR..."
    $MINICONDA_DIR = $DEFAULT_MINICONDA_DIR
    $installer = Join-Path $env:TEMP "Miniconda3-latest-Windows-x86_64.exe"
    $url = "https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe"
    Write-Host "  Downloading $url ..."
    Invoke-WebRequest -Uri $url -OutFile $installer -UseBasicParsing
    Write-Host "  Running silent install..."
    Start-Process -FilePath $installer -ArgumentList "/S","/InstallationType=JustMe","/RegisterPython=0","/AddToPath=0","/D=$MINICONDA_DIR" -Wait
    Remove-Item $installer -Force
} else {
    Write-Host "> Conda installation found at $MINICONDA_DIR."
}

$CONDA_EXE = Join-Path $MINICONDA_DIR "Scripts\conda.exe"
if (-not (Test-Path $CONDA_EXE)) {
    Write-Host "  ERROR: conda.exe not found at $CONDA_EXE after install attempt." -ForegroundColor Red
    exit 1
}

# Put conda's own bin dirs on PATH for THIS process so conda-installed tools
# (git, etc.) become callable immediately, and load the conda PowerShell hook.
$env:Path = "$MINICONDA_DIR;$MINICONDA_DIR\Scripts;$MINICONDA_DIR\Library\bin;$env:Path"
$condaHook = Join-Path $MINICONDA_DIR "shell\condabin\conda-hook.ps1"
if (Test-Path $condaHook) { & $condaHook }

# ---------------------------------------------------------------------------
# 2. Ensure git is available. The Linux installer auto-installs git through the
#    system package manager; on Windows we install it from conda-forge so the
#    user does NOT have to install Git for Windows by hand.
# ---------------------------------------------------------------------------
function Ensure-Git {
    if (Get-Command git -ErrorAction SilentlyContinue) { return $true }
    Write-Host "> git not found -- installing it from conda-forge..."
    Invoke-Native { & $CONDA_EXE install -n base -c conda-forge git -y } | Write-Host
    # git.exe lands in Library\bin; make sure it is visible to this process.
    $env:Path = "$MINICONDA_DIR\Library\bin;$MINICONDA_DIR\Scripts;$env:Path"
    return [bool](Get-Command git -ErrorAction SilentlyContinue)
}

# ---------------------------------------------------------------------------
# 3. Get/refresh the source.
# ---------------------------------------------------------------------------
if (Test-Path (Join-Path $TARGET_DIR ".git")) {
    Write-Host "> Repo exists, pulling latest updates..."
    if (-not (Ensure-Git)) {
        Write-Host "  ERROR: git is required to pull updates and could not be installed." -ForegroundColor Red
        exit 1
    }
    Push-Location $TARGET_DIR
    git pull
    Pop-Location
} else {
    Write-Host "> Cloning repository..."
    if (-not (Ensure-Git)) {
        Write-Host "  ERROR: git is required to clone the repository and could not be"   -ForegroundColor Red
        Write-Host "  installed automatically. Install Git for Windows from"             -ForegroundColor Red
        Write-Host "  https://git-scm.com/download/win and re-run this installer."       -ForegroundColor Red
        exit 1
    }
    git clone $REPO_URL $TARGET_DIR
}

# ---------------------------------------------------------------------------
# 4. Conda Terms of Service + Python environment.
# ---------------------------------------------------------------------------
Write-Host "> Accepting conda Terms of Service..."
& $CONDA_EXE tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main  2>$null
& $CONDA_EXE tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r     2>$null
& $CONDA_EXE tos accept --override-channels --channel https://repo.anaconda.com/pkgs/msys2 2>$null

$ENV_NAME = if ($INSTALL_PY11) { "py11" } else { "base" }

if ($INSTALL_PY11) {
    Write-Host "> Creating py11 environment with Python 3.11..."
    $envList = & $CONDA_EXE env list
    if ($envList -match "(?m)^\s*py11\s") {
        Write-Host "> Environment py11 already exists. Skipping creation."
    } else {
        Write-Host "> Creating new environment py11..."
        & $CONDA_EXE create -n py11 python=3.11 -y
    }
    $ENV_PREFIX = Join-Path $MINICONDA_DIR "envs\py11"
} else {
    Write-Host "> Using base environment for installation..."
    $ENV_PREFIX = $MINICONDA_DIR
}

$ENV_PYTHON  = Join-Path $ENV_PREFIX "python.exe"
$ENV_SCRIPTS = Join-Path $ENV_PREFIX "Scripts"
$ENV_LIBBIN  = Join-Path $ENV_PREFIX "Library\bin"

Write-Host "> Installing core scientific deps..."
& $CONDA_EXE install -n $ENV_NAME numpy scipy matplotlib scikit-learn -y

Write-Host "> Installing chemistry deps from conda-forge..."
# xtb is pinned to >=6.7 for the same reason as the Linux installer: left
# unpinned, the conda-forge solve can backtrack to the ancient 6.5.0 (2022)
# build, which crashes mid-optimization with a libgfortran I/O error and never
# writes a "final structure:" block. cosmic then extracts zero geometries ->
# zero motifs -> "No motifs_/umotifs_ folder found" at the end of a green run.
# 6.7.x is statically linked and runs cleanly.
& $CONDA_EXE install -n $ENV_NAME -c conda-forge cclib openbabel "xtb>=6.7" -y

Write-Host "> Installing orca-pi parser via pip..."
$ENV_PIP = Join-Path $ENV_SCRIPTS "pip.exe"
& $CONDA_EXE install -n $ENV_NAME pip -y --override-channels -c conda-forge
& $ENV_PIP install orca-pi

$XTB_EXE = Join-Path $ENV_LIBBIN "xtb.exe"
if (Test-Path $XTB_EXE) {
    Write-Host "> xtb installed at $XTB_EXE"
    # Capture the version banner through Invoke-Native: xtb prints "normal
    # termination of xtb" to stderr, and merging that with 2>&1 under
    # ErrorActionPreference='Stop' would otherwise abort the whole installer
    # right here -- before the launchers below are written.
    $xtbOut = Invoke-Native { & $XTB_EXE --version }
    $xtbOut -split "`n" | Select-Object -First 3 | ForEach-Object { Write-Host "    $($_.TrimEnd())" }
    # Guard against a too-old build slipping through (e.g. if the pin above was
    # relaxed). 6.5.0 and older conda-forge builds crash with a libgfortran I/O
    # error and produce no usable geometry, which silently breaks cosmic.
    $m = [regex]::Match($xtbOut, 'version\s+(\d+)\.(\d+)')
    if ($m.Success) {
        $xtbMajor = [int]$m.Groups[1].Value
        $xtbMinor = [int]$m.Groups[2].Value
        if ($xtbMajor -lt 6 -or ($xtbMajor -eq 6 -and $xtbMinor -lt 6)) {
            Write-Host "  WARNING: xtb $xtbMajor.$xtbMinor is too old and is known to crash mid-optimization." -ForegroundColor Yellow
            Write-Host "  Reinstall a working build: conda install -n $ENV_NAME -c conda-forge 'xtb>=6.7'"
        }
    }
} else {
    Write-Host "  WARNING: xtb.exe not found at $XTB_EXE." -ForegroundColor Yellow
    Write-Host "  Try: conda install -n $ENV_NAME -c conda-forge 'xtb>=6.7'"
}

# ---------------------------------------------------------------------------
# 5. Launcher scripts (the Windows analogue of the bash aliases). Each one
#    PREPENDS the conda env's bin dirs to PATH so bare `xtb` / `obabel` resolve
#    to THIS env's build at runtime -- this is what guarantees the correct xtb
#    is used, regardless of any other xtb on the system PATH.
# ---------------------------------------------------------------------------
$LAUNCHER_DIR = Join-Path $env:USERPROFILE "bin"
New-Item -ItemType Directory -Path $LAUNCHER_DIR -Force | Out-Null

$ascecCmd  = Join-Path $LAUNCHER_DIR "ascec.cmd"
$cosmicCmd = Join-Path $LAUNCHER_DIR "cosmic.cmd"

$ascecBody = "@echo off`r`nsetlocal`r`nset `"PATH=$ENV_PREFIX;$ENV_SCRIPTS;$ENV_LIBBIN;%PATH%`"`r`n`"$ENV_PYTHON`" `"$TARGET_DIR\ascec-v04.py`" %*`r`n"
$cosmicBody = "@echo off`r`nsetlocal`r`nset `"PATH=$ENV_PREFIX;$ENV_SCRIPTS;$ENV_LIBBIN;%PATH%`"`r`n`"$ENV_PYTHON`" `"$TARGET_DIR\cosmic-v01.py`" %*`r`n"

Set-Content -Path $ascecCmd  -Value $ascecBody  -Encoding ASCII
Set-Content -Path $cosmicCmd -Value $cosmicBody -Encoding ASCII

Write-Host "> Launcher scripts written: $ascecCmd and $cosmicCmd"

$userPath = [Environment]::GetEnvironmentVariable("Path", "User")
if (-not ($userPath -split ";" | Where-Object { $_ -ieq $LAUNCHER_DIR })) {
    $newUserPath = if ($userPath) { "$userPath;$LAUNCHER_DIR" } else { $LAUNCHER_DIR }
    [Environment]::SetEnvironmentVariable("Path", $newUserPath, "User")
    Write-Host "> Added $LAUNCHER_DIR to your user PATH (takes effect in new shells)."
} else {
    Write-Host "> $LAUNCHER_DIR is already on your user PATH."
}

Write-Host "-------------------------------------------------------" -ForegroundColor Green
Write-Host "> INSTALLATION COMPLETE!" -ForegroundColor Green
Write-Host "> Open a NEW terminal window, then run:"
Write-Host "    ascec  your_input.asc"
Write-Host "    cosmic ..."
Write-Host "> Standalone xTB is the default annealing backend."
Write-Host "> ORCA is optional for DFT-level optimization."
Write-Host "-------------------------------------------------------" -ForegroundColor Green
