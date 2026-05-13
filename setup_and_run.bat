@echo off
REM Script para configurar venv e instalar dependencias

echo Creando/Activando virtual environment...
if not exist venv (
    python -m venv venv
)

call venv\Scripts\activate.bat

echo Instalando dependencias...
pip install --upgrade pip
pip install -r requirements.txt

echo.
echo Ejecutando preprocess_maestro.py...
python scripts/preprocess_maestro.py %*

pause
