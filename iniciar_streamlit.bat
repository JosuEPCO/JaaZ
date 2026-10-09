@echo off
setlocal
cd /d "%~dp0"
chcp 65001 >nul
title JaaZ (Streamlit)
echo ================================================
echo        JaaZ - version Streamlit
echo ================================================
where python >nul 2>nul || (echo ERROR: Python no esta instalado o no esta en PATH.&echo Descarguelo de https://www.python.org/downloads/ ^(3.12 o superior, marque "Add to PATH"^).&pause&exit /b 1)
if not exist .venv (
  echo Creando el entorno de Python por primera vez...
  python -m venv .venv || goto :error
)
call .venv\Scripts\activate.bat || goto :error
python -c "import streamlit, numpy, scipy, pandas, altair, openpyxl, xlrd" >nul 2>nul || (
  echo Instalando dependencias ^(solo la primera vez, requiere internet^)...
  python -m pip install --disable-pip-version-check -r requirements.txt || goto :error
)
echo.
echo Se abrira JaaZ en el navegador. Deje esta ventana abierta mientras lo usa.
echo.
python -m streamlit run app.py
goto :end
:error
echo ERROR: No se pudo iniciar JaaZ.
pause
:end
endlocal
