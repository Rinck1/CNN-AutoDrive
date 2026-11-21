@echo off
chcp 65001 >nul
cd /d "%~dp0"
where py >nul 2>nul
if %errorlevel%==0 (
  py -3 deploy_web.py %*
) else (
  where python >nul 2>nul
  if errorlevel 1 (
    echo [部署] 未找到 Python，请先安装 Python 3.8+ 并勾选 Add to PATH。
    pause
    exit /b 1
  )
  python deploy_web.py %*
)
if errorlevel 1 pause
