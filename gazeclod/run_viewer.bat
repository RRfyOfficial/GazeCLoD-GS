@echo off
rem GazeCLoD-GS viewer launcher for Windows.
rem Double-click this file, or run it from a terminal with extra options, e.g.
rem     gazeclod\run_viewer.bat --scene playroom
rem Inside the viewer press T to switch scenes, ESC to quit.
setlocal
set "ENV_NAME=clodgs"

rem Work from the repo root no matter where this was started from.
cd /d "%~dp0.."

rem Already inside the right conda env? Then just run.
if /i "%CONDA_DEFAULT_ENV%"=="%ENV_NAME%" goto run

rem Find conda's activate script: the conda on PATH first, then the usual install folders.
set "ACT="
if defined CONDA_EXE for %%I in ("%CONDA_EXE%") do if exist "%%~dpIactivate.bat" set "ACT=%%~dpIactivate.bat"
for %%D in ("%USERPROFILE%\miniconda3" "%USERPROFILE%\anaconda3" "%LOCALAPPDATA%\miniconda3" "%LOCALAPPDATA%\anaconda3" "%ProgramData%\miniconda3" "%ProgramData%\anaconda3") do if not defined ACT if exist "%%~D\Scripts\activate.bat" set "ACT=%%~D\Scripts\activate.bat"
if not defined ACT goto noconda

call "%ACT%" %ENV_NAME%
if errorlevel 1 goto noenv

:run
python gazeclod\viewer.py %*
if errorlevel 1 goto failed
exit /b 0

:noconda
echo Could not find Miniconda or Anaconda on this PC.
echo Open "Anaconda Prompt", then run:  conda activate %ENV_NAME%  and  python gazeclod\viewer.py
pause
exit /b 1

:noenv
echo Found conda but could not activate the "%ENV_NAME%" environment.
echo Check the name with:  conda env list   and edit ENV_NAME at the top of this file.
pause
exit /b 1

:failed
echo.
echo The viewer stopped with an error. The message above says why.
pause
exit /b 1
