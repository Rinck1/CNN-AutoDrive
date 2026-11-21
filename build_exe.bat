@echo off
chcp 65001 >nul
cd /d "%~dp0"

rem 重新打包 MiniFlowsStudio.exe（需要联网安装构建依赖）
if not exist .buildenv (
  python -m venv .buildenv
)
.buildenv\Scripts\python.exe -m pip install --upgrade pip
.buildenv\Scripts\python.exe -m pip install numpy==1.26.4 matplotlib==3.9.4 pyinstaller==6.12.0 backports.tarfile

.buildenv\Scripts\python.exe -m PyInstaller --noconfirm --clean --onefile --name MiniFlowsStudio ^
  --add-data "applications/web_studio_ui.html;applications" ^
  --add-data "applications/assets;applications/assets" ^
  --add-data "outputs;outputs" ^
  --add-data "data;data" ^
  --hidden-import matplotlib.backends.backend_agg ^
  --hidden-import applications.web_studio ^
  --hidden-import applications.donkeycar ^
  --hidden-import applications.donkeycar_eval ^
  --hidden-import applications.inspector ^
  --collect-data matplotlib ^
  --exclude-module torch --exclude-module torchvision ^
  --exclude-module sklearn --exclude-module scipy ^
  deploy_web.py

echo.
echo 打包完成：dist\MiniFlowsStudio.exe
pause
