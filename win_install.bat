@echo off
setlocal enableextensions
title COSMIC-ASCEC Installer

REM ===========================================================================
REM  COSMIC-ASCEC ONE-CLICK INSTALLER (Windows)
REM ===========================================================================
REM  Double-click this file in File Explorer, or run it from cmd.
REM
REM  This script is deliberately plain, readable batch. It does NOT read its own
REM  source, assemble concealed markers from character codes, or evaluate a
REM  script fragment string with the policy check turned off. The previous
REM  version did all three, which is the signature antivirus heuristics look for
REM  in an obfuscated dropper -- it got this installer flagged even though it was
REM  doing nothing wrong. Everything below uses cmd built-ins plus curl.exe,
REM  certutil.exe and tar.exe, all shipped with Windows 10 1803 and later.
REM
REM  What it does:
REM    1. Installs Miniconda to %USERPROFILE%\Miniconda3 if no conda is found
REM    2. Creates the Python 3.11 conda environment "py11"
REM    3. Downloads the source into %USERPROFILE%\software\ascec04 (a plain zip
REM       download is used when git is absent; git is optional)
REM    4. Installs numpy, scipy, matplotlib, cclib, openbabel, xtb, orca-pi
REM    5. Writes ascec.cmd / cosmic.cmd into %USERPROFILE%\bin and adds it to PATH
REM
REM  Nothing here needs administrator rights.
REM ===========================================================================

REM --------------------------- CONFIGURATION ---------------------------------
set "TARGET_DIR=%USERPROFILE%\software\ascec04"
set "LAUNCHER_DIR=%USERPROFILE%\bin"
set "DEFAULT_MINICONDA_DIR=%USERPROFILE%\Miniconda3"
set "ENV_NAME=py11"
set "PY_VERSION=3.11"
set "REPO_URL=https://github.com/manuel2gl/qft-cosmic-ascec.git"
set "REPO_ZIP=https://github.com/manuel2gl/qft-cosmic-ascec/archive/refs/heads/main.zip"

REM Miniconda installer. To pin a reproducible version, replace MINICONDA_URL
REM with a versioned installer from https://repo.anaconda.com/miniconda/ and set
REM MINICONDA_SHA256 to the matching hash published at
REM https://docs.anaconda.com/miniconda/. Leaving MINICONDA_SHA256 empty skips
REM verification, and the script says so out loud rather than pretending.
set "MINICONDA_URL=https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe"
set "MINICONDA_SHA256="
REM ---------------------------------------------------------------------------

echo.
echo ===========================================================
echo   COSMIC-ASCEC installer for Windows
echo ===========================================================
echo.
echo   Source directory : %TARGET_DIR%
echo   Conda environment: %ENV_NAME% (Python %PY_VERSION%)
echo   Launchers        : %LAUNCHER_DIR%\ascec.cmd, cosmic.cmd
echo.
echo   No administrator rights are required.
echo.

REM ===========================================================================
REM  1. Find conda, or install Miniconda.
REM ===========================================================================
REM  Conda comes first because it doubles as a way to obtain git in step 3.
REM  Windows has no system package manager to fall back on.

echo [1/6] Looking for an existing conda installation...

set "MINICONDA_DIR="
for %%D in (
    "%USERPROFILE%\Miniconda3"
    "%USERPROFILE%\Anaconda3"
    "%USERPROFILE%\miniforge3"
    "%USERPROFILE%\mambaforge"
    "C:\ProgramData\Miniconda3"
    "C:\ProgramData\Anaconda3"
) do (
    if not defined MINICONDA_DIR if exist "%%~D\Scripts\conda.exe" set "MINICONDA_DIR=%%~D"
)

if defined MINICONDA_DIR (
    echo       Found: %MINICONDA_DIR%
    goto :conda_ready
)

echo       No conda found. Installing Miniconda to %DEFAULT_MINICONDA_DIR%
set "MINICONDA_DIR=%DEFAULT_MINICONDA_DIR%"
set "MC_EXE=%TEMP%\Miniconda3-installer.exe"

where curl.exe >nul 2>&1
if errorlevel 1 (
    echo.
    echo   ERROR: curl.exe was not found. It ships with Windows 10 1803 and later.
    echo   Download Miniconda manually from https://www.anaconda.com/download
    echo   install it, then re-run this script.
    goto :fail
)

echo       Downloading %MINICONDA_URL%
curl.exe -L --fail --progress-bar -o "%MC_EXE%" "%MINICONDA_URL%"
if errorlevel 1 (
    echo.
    echo   ERROR: download failed. Check your network connection or proxy.
    goto :fail
)

REM Verification lives in a subroutine on purpose. Inside a parenthesised block
REM cmd expands %VAR% when it PARSES the whole block, so a value assigned and
REM then read within the same block reads as empty -- which would have compared
REM two empty strings and "passed" every time. Subroutine lines are parsed one
REM at a time, so the reads see the assignments.
call :verify_sha256
if errorlevel 1 goto :fail

echo       Running the Miniconda installer (this takes a minute)...
start /wait "" "%MC_EXE%" /S /InstallationType=JustMe /RegisterPython=0 /AddToPath=0 /D=%MINICONDA_DIR%
del /q "%MC_EXE%" >nul 2>&1

:conda_ready
set "CONDA_EXE=%MINICONDA_DIR%\Scripts\conda.exe"
if not exist "%CONDA_EXE%" (
    echo.
    echo   ERROR: conda.exe not found at %CONDA_EXE% after the install attempt.
    goto :fail
)
REM Make conda's own tools visible to this process for the rest of the script.
set "PATH=%MINICONDA_DIR%;%MINICONDA_DIR%\Scripts;%MINICONDA_DIR%\Library\bin;%PATH%"

set "ENV_PREFIX=%MINICONDA_DIR%\envs\%ENV_NAME%"
set "ENV_PYTHON=%ENV_PREFIX%\python.exe"
set "ENV_SCRIPTS=%ENV_PREFIX%\Scripts"
set "ENV_LIBBIN=%ENV_PREFIX%\Library\bin"

REM ===========================================================================
REM  2. Create the environment.
REM ===========================================================================
REM  Done before fetching the source so that git can be installed into it.

echo.
echo [2/6] Setting up the "%ENV_NAME%" environment...

"%CONDA_EXE%" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main  >nul 2>&1
"%CONDA_EXE%" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r     >nul 2>&1
"%CONDA_EXE%" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/msys2 >nul 2>&1

if exist "%ENV_PYTHON%" (
    echo       Environment "%ENV_NAME%" already exists.
) else (
    echo       Creating "%ENV_NAME%" with Python %PY_VERSION%...
    "%CONDA_EXE%" create -n %ENV_NAME% --override-channels -c conda-forge -y python=%PY_VERSION%
    if errorlevel 1 (
        echo.
        echo   ERROR: could not create the conda environment.
        goto :fail
    )
)

REM ===========================================================================
REM  3. Obtain the source.
REM ===========================================================================
REM  git is NOT a prerequisite and this installer never installs it for you. A
REM  fresh install uses a plain zip download from GitHub, which needs no git at
REM  all. git is used only when it already exists on PATH: to clone a fresh copy
REM  (a real checkout you can update later) or to "git pull" an EXISTING one. If
REM  you want that, install Git for Windows yourself from
REM  https://git-scm.com/download/win and re-run this installer.

echo.
echo [3/6] Fetching the source...

set "HAVE_GIT="
where git.exe >nul 2>&1 && set "HAVE_GIT=1"

if exist "%TARGET_DIR%\.git" goto :src_update
if exist "%TARGET_DIR%\*" goto :src_check_empty
goto :src_fresh

:src_check_empty
REM Target exists. If it holds anything and is not a checkout, refuse rather
REM than letting the copy land on top of unrelated files.
dir /b /a "%TARGET_DIR%" 2>nul | findstr "." >nul
if errorlevel 1 goto :src_fresh
echo.
echo   ERROR: %TARGET_DIR%
echo   already exists, is not empty, and is not a git checkout.
echo.
echo   Move or delete it, then re-run this installer.
goto :fail

:src_update
echo       Existing checkout found.
if not defined HAVE_GIT (
    echo       git is not installed, so this checkout cannot be updated.
    echo       Install Git for Windows from https://git-scm.com/download/win
    echo       and re-run to pull updates. Using the existing code for now.
    goto :src_done
)
REM Never clobber local edits: someone who has been editing the code in place is
REM the normal case, and a plain "git pull" on a dirty tree fails outright.
set "GIT_DIRTY="
for /f "delims=" %%S in ('git -C "%TARGET_DIR%" status --porcelain 2^>nul') do set "GIT_DIRTY=1"
if defined GIT_DIRTY (
    echo       Local modifications detected - SKIPPING update to protect them:
    git -C "%TARGET_DIR%" status --porcelain
    echo       Commit or stash them and re-run if you want the update.
    goto :src_done
)
REM --ff-only: git 2.27+ refuses a plain pull on divergent branches, and an
REM installer has no business starting a merge on the user's behalf.
git -C "%TARGET_DIR%" pull --ff-only
if errorlevel 1 (
    echo       Update failed ^(diverged branch or no network^). Using existing code.
) else (
    echo       Updated to the latest revision.
)
goto :src_done

:src_fresh
REM Fresh install. If git already exists, a clone gives a real checkout that can
REM be updated later; otherwise fall straight to a zip download. We do NOT try to
REM INSTALL git here -- installing git just to clone is exactly what stalled a
REM fresh install before, and it buys nothing a zip download does not.
if defined HAVE_GIT (
    echo       Cloning %REPO_URL%
    git clone "%REPO_URL%" "%TARGET_DIR%"
    if not errorlevel 1 goto :src_done
    echo       Clone failed; falling back to a zip download.
)

echo       Downloading the source as a zip (no git required)...
set "SRC_ZIP=%TEMP%\cosmic-ascec-src.zip"
set "SRC_TMP=%TEMP%\cosmic-ascec-src"
rmdir /s /q "%SRC_TMP%" >nul 2>&1
mkdir "%SRC_TMP%" >nul 2>&1
curl.exe -L --fail --progress-bar -o "%SRC_ZIP%" "%REPO_ZIP%"
if errorlevel 1 (
    echo.
    echo   ERROR: could not download the source.
    echo   Check your network connection or proxy, then re-run this installer.
    echo   Or install Git for Windows from https://git-scm.com/download/win
    echo   and re-run.
    goto :fail
)
tar.exe -xf "%SRC_ZIP%" -C "%SRC_TMP%"
if errorlevel 1 (
    echo   ERROR: could not extract the source archive.
    goto :fail
)
if not exist "%TARGET_DIR%" mkdir "%TARGET_DIR%"
xcopy "%SRC_TMP%\qft-cosmic-ascec-main\*" "%TARGET_DIR%\" /E /I /Q /Y >nul
del /q "%SRC_ZIP%" >nul 2>&1
rmdir /s /q "%SRC_TMP%" >nul 2>&1
echo       NOTE: installed from a zip, so this is not a git checkout.
echo             Future updates will re-download rather than "git pull".

:src_done
if not exist "%TARGET_DIR%\ascec-v04.py" (
    echo.
    echo   ERROR: %TARGET_DIR%\ascec-v04.py is missing.
    echo   The download or copy did not complete correctly.
    goto :fail
)

REM ===========================================================================
REM  4. Install dependencies.
REM ===========================================================================
REM  Everything from conda-forge in one solve. The previous version took
REM  numpy/scipy/matplotlib/scikit-learn from "defaults" and only the chemistry
REM  packages from conda-forge; mixing channels risks ABI mismatches between
REM  packages that share libraries. --override-channels keeps defaults out.
REM
REM  scikit-learn is deliberately absent: it was ~50 MB (plus joblib and
REM  threadpoolctl) for a single StandardScaler z-score, which
REM  clustering/scaling.py now computes directly with numpy.
REM
REM  xtb is pinned to >=6.7. Left unpinned, the solve can backtrack to the
REM  ancient conda-forge 6.5.0 build, which crashes mid-optimization with a
REM  libgfortran I/O error and never writes a "final structure:" block. cosmic
REM  then extracts zero geometries, yields zero motifs, and reports "No motifs_/
REM  umotifs_ folder found" at the end of a run that otherwise looked fine.

echo.
echo [4/6] Installing dependencies (this is the slow part)...
"%CONDA_EXE%" install -n %ENV_NAME% --override-channels -c conda-forge -y ^
    numpy scipy matplotlib cclib openbabel "xtb>=6.7"
if errorlevel 1 (
    echo.
    echo   ERROR: dependency installation failed.
    goto :fail
)

echo       Installing orca-pi (ORCA 6.1+ output parser)...
"%ENV_SCRIPTS%\pip.exe" install orca-pi
if errorlevel 1 echo       NOTE: orca-pi failed to install; ORCA 6.1+ will fall back to a text scrape.

if exist "%ENV_LIBBIN%\xtb.exe" (
    echo       xtb installed at %ENV_LIBBIN%\xtb.exe
) else (
    echo       WARNING: xtb.exe not found. Annealing runs will fail.
    echo       Try: conda install -n %ENV_NAME% -c conda-forge "xtb>=6.7"
)

REM ===========================================================================
REM  5. Write the launchers.
REM ===========================================================================
REM  Each launcher PREPENDS the environment's directories to PATH so that a bare
REM  "xtb" or "obabel" resolves to THIS environment's build at runtime,
REM  regardless of anything else on the system PATH.
REM
REM  %%PATH%% and %%* below are written into the generated file as literal
REM  %PATH% and %*  -- they must expand when the launcher runs, not now.

echo.
echo [5/6] Writing launchers...
if not exist "%LAUNCHER_DIR%" mkdir "%LAUNCHER_DIR%"

set "ASCEC_CMD=%LAUNCHER_DIR%\ascec.cmd"
>  "%ASCEC_CMD%" echo @echo off
>> "%ASCEC_CMD%" echo setlocal
>> "%ASCEC_CMD%" echo set "PATH=%ENV_PREFIX%;%ENV_SCRIPTS%;%ENV_LIBBIN%;%%PATH%%"
>> "%ASCEC_CMD%" echo "%ENV_PYTHON%" "%TARGET_DIR%\ascec-v04.py" %%*

set "COSMIC_CMD=%LAUNCHER_DIR%\cosmic.cmd"
>  "%COSMIC_CMD%" echo @echo off
>> "%COSMIC_CMD%" echo setlocal
>> "%COSMIC_CMD%" echo set "PATH=%ENV_PREFIX%;%ENV_SCRIPTS%;%ENV_LIBBIN%;%%PATH%%"
>> "%COSMIC_CMD%" echo "%ENV_PYTHON%" "%TARGET_DIR%\cosmic-v01.py" %%*

echo       %ASCEC_CMD%
echo       %COSMIC_CMD%

REM Append the launcher directory to the USER PATH.
REM Deliberately NOT "setx": setx silently truncates PATH at 1024 characters,
REM which corrupts any PATH longer than that. This one PowerShell call reads and
REM writes the value properly. It runs a fixed, readable expression: the policy
REM check is left alone, nothing is evaluated from a string, nothing is
REM downloaded.
powershell -NoProfile -Command ^
  "$d = '%LAUNCHER_DIR%'; $p = [Environment]::GetEnvironmentVariable('Path','User'); if (($p -split ';') -notcontains $d) { $n = if ($p) { $p.TrimEnd(';') + ';' + $d } else { $d }; [Environment]::SetEnvironmentVariable('Path', $n, 'User'); Write-Host '      Added ' $d ' to your user PATH.' } else { Write-Host '      ' $d ' is already on your user PATH.' }"

REM ===========================================================================
REM  6. Verify.
REM ===========================================================================
REM  Fail here rather than at the user's first real run.

echo.
echo [6/6] Verifying the installation...
call "%ASCEC_CMD%" --version
if errorlevel 1 (
    echo.
    echo   ERROR: "ascec --version" failed.
    echo   The environment was created but the code does not run.
    goto :fail
)

echo.
echo ===========================================================
echo   INSTALLATION COMPLETE
echo ===========================================================
echo.
echo   Open a NEW terminal window (so the PATH change takes effect), then:
echo.
echo       ascec  your_input.asc
echo       cosmic --version
echo.
echo   Standalone xTB is the default annealing backend.
echo   ORCA is optional, for DFT-level optimization and refinement.
echo.
echo   Note: "ascec status", background detach and job queueing are
echo   Linux-only. Runs themselves work normally on Windows.
echo.
goto :done

REM ===========================================================================
REM  Subroutines
REM ===========================================================================

:verify_sha256
if not defined MINICONDA_SHA256 (
    echo       NOTE: MINICONDA_SHA256 is not set, so the download is not verified.
    echo             See the CONFIGURATION block at the top of this file to pin it.
    exit /b 0
)
echo       Verifying SHA256...
set "MC_HASH="
for /f "skip=1 tokens=* delims=" %%H in ('certutil -hashfile "%MC_EXE%" SHA256') do (
    if not defined MC_HASH set "MC_HASH=%%H"
)
REM certutil prints the digest space-separated on some Windows versions.
set "MC_HASH=%MC_HASH: =%"
if /i "%MC_HASH%"=="%MINICONDA_SHA256%" (
    echo       Checksum verified.
    exit /b 0
)
echo.
echo   ERROR: checksum mismatch for the Miniconda installer.
echo     expected: %MINICONDA_SHA256%
echo     actual:   %MC_HASH%
echo   Refusing to run an installer that does not match the pinned hash.
exit /b 1

:fail
echo.
echo -----------------------------------------------------------
echo   INSTALLATION FAILED - see the message above.
echo -----------------------------------------------------------
echo.
echo Press any key to close this window...
pause >nul
endlocal
exit /b 1

:done
echo Press any key to close this window...
pause >nul
endlocal
exit /b 0
