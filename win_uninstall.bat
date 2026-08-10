@echo off
setlocal enableextensions
title COSMIC-ASCEC Uninstaller

REM ===========================================================================
REM  COSMIC-ASCEC UNINSTALLER (Windows)
REM ===========================================================================
REM  Double-click this file in File Explorer, or run it from cmd.
REM  Pass -y (or --yes) to skip the confirmation prompt.
REM
REM  Like the installer, this is plain readable batch: no self-reading, no
REM  concealed PowerShell payload, nothing evaluated from a string. See
REM  win_install.bat for why that matters.
REM
REM  Reverses what win_install.bat set up, WITHOUT touching conda itself:
REM    - Removes the "py11" conda environment (base conda is left alone)
REM    - Removes ascec.cmd / cosmic.cmd from %USERPROFILE%\bin
REM    - Removes %USERPROFILE%\bin from the user PATH (only if it is there)
REM    - Removes the source directory %USERPROFILE%\software\ascec04
REM
REM  Conda is intentionally left installed, since it is usually shared with
REM  other work.
REM ===========================================================================

set "ENV_NAME=py11"
set "TARGET_DIR=%USERPROFILE%\software\ascec04"
set "LAUNCHER_DIR=%USERPROFILE%\bin"

set "ASSUME_YES="
if /i "%~1"=="-y"     set "ASSUME_YES=1"
if /i "%~1"=="--yes"  set "ASSUME_YES=1"
if /i "%~1"=="/y"     set "ASSUME_YES=1"

echo.
echo ===========================================================
echo   COSMIC-ASCEC uninstaller for Windows
echo ===========================================================
echo.
echo   This will:
echo     - Remove the conda environment "%ENV_NAME%" (conda itself is kept)
echo     - Remove ascec.cmd / cosmic.cmd from %LAUNCHER_DIR%
echo     - Remove %LAUNCHER_DIR% from your user PATH
echo     - Remove the source directory %TARGET_DIR%
echo.

if defined ASSUME_YES goto :confirmed
set "REPLY="
set /p "REPLY=  Proceed? [y/N] "
if /i "%REPLY%"=="y"   goto :confirmed
if /i "%REPLY%"=="yes" goto :confirmed
echo   Aborted.
goto :done

:confirmed

REM ===========================================================================
REM  1. Remove the conda environment (not conda itself).
REM ===========================================================================

echo.
echo [1/4] Removing the "%ENV_NAME%" conda environment...

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

if not defined MINICONDA_DIR (
    echo       WARNING: conda not found in the usual locations.
    echo       If "%ENV_NAME%" exists, remove it manually:
    echo           conda env remove -n %ENV_NAME%
    goto :launchers
)

set "CONDA_EXE=%MINICONDA_DIR%\Scripts\conda.exe"
if not exist "%MINICONDA_DIR%\envs\%ENV_NAME%" (
    echo       Environment "%ENV_NAME%" not found - nothing to remove.
    goto :launchers
)
"%CONDA_EXE%" env remove -n %ENV_NAME% -y
if errorlevel 1 (
    echo       WARNING: could not remove the environment. Remove it manually:
    echo           conda env remove -n %ENV_NAME%
) else (
    echo       Removed.
)

REM ===========================================================================
REM  2. Remove the launcher scripts.
REM ===========================================================================

:launchers
echo.
echo [2/4] Removing launchers...
if exist "%LAUNCHER_DIR%\ascec.cmd" (
    del /q "%LAUNCHER_DIR%\ascec.cmd"
    echo       Removed %LAUNCHER_DIR%\ascec.cmd
)
if exist "%LAUNCHER_DIR%\cosmic.cmd" (
    del /q "%LAUNCHER_DIR%\cosmic.cmd"
    echo       Removed %LAUNCHER_DIR%\cosmic.cmd
)

REM ===========================================================================
REM  3. Remove the launcher directory from the user PATH.
REM ===========================================================================
REM  Same reasoning as the installer: read/modify/write through PowerShell
REM  rather than setx, which truncates PATH at 1024 characters. Only an exact
REM  match on the entry is removed, so unrelated PATH entries are untouched.

echo.
echo [3/4] Updating your user PATH...
powershell -NoProfile -Command ^
  "$d = '%LAUNCHER_DIR%'; $p = [Environment]::GetEnvironmentVariable('Path','User'); if ($p) { $keep = $p -split ';' | Where-Object { $_ -and ($_ -ne $d) }; $n = $keep -join ';'; if ($n -ne $p) { [Environment]::SetEnvironmentVariable('Path', $n, 'User'); Write-Host '      Removed' $d 'from your user PATH.' } else { Write-Host '      ' $d 'was not on your user PATH.' } }"

REM ===========================================================================
REM  4. Remove the source directory.
REM ===========================================================================
REM  Removal fails with "in use" whenever something still holds a handle inside
REM  the directory. The two cases that bite real users are (a) a previous
REM  ascec/cosmic run that crashed or was Ctrl-C'd in another window and still
REM  has files open, and (b) a shell or File Explorer window sitting inside the
REM  tree. Handles are released lazily after a process exits, so retry a couple
REM  of times before giving up, then name the likely culprits.

echo.
echo [4/4] Removing %TARGET_DIR%...
if not exist "%TARGET_DIR%" (
    echo       Not found - nothing to remove.
    goto :finished
)

set "RETRY=0"

:rmdir_try
rmdir /s /q "%TARGET_DIR%" >nul 2>&1
if not exist "%TARGET_DIR%" goto :rmdir_ok
set /a RETRY+=1
if %RETRY% GEQ 3 goto :rmdir_fail
echo       Directory busy, retrying in 2s (attempt %RETRY% of 3)...
timeout /t 2 /nobreak >nul
goto :rmdir_try

:rmdir_ok
echo       Removed.
goto :finished

:rmdir_fail
echo.
echo   ERROR: could not remove %TARGET_DIR%
echo   A process still has files open inside it. Most common causes:
echo     - An "ascec" or "cosmic" run is still alive in another window.
echo     - Another cmd/PowerShell window has cd'd into a subfolder.
echo     - File Explorer is showing this folder in an open window.
echo.
echo   Python processes currently running:
REM  Listed, not killed: terminating these automatically could clobber an
REM  unrelated long-running job the user is in the middle of.
tasklist /FI "IMAGENAME eq python.exe" 2>nul | findstr /i "python.exe" || echo       (none)
tasklist /FI "IMAGENAME eq pythonw.exe" 2>nul | findstr /i "pythonw.exe"
echo.
echo   Close the offending window, then re-run this uninstaller.
echo.
echo Press any key to close this window...
pause >nul
endlocal
exit /b 1

:finished
echo.
echo ===========================================================
echo   UNINSTALL COMPLETE
echo ===========================================================
echo.
echo   Conda was left installed.
echo   Open a NEW terminal so the PATH change takes effect.
echo.

:done
echo Press any key to close this window...
pause >nul
endlocal
exit /b 0
