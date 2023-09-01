@echo off
@REM pymarl does not appear to be a Python project: neither 'setup.py' nor 'pyproject.toml' found.
set external_repos=smac smacv2 pymarl pysc2
set start_dir=%cd%
cls
echo "Starting from %start_dir%"
cd %~dp0
echo %cd%

call venv\Scripts\activate
python.exe -m pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

(for /d %%D in (%external_repos%) do (
    echo %%~fD
    cd %start_dir%
    cd ..
    cd %%D
    pip install -e ".[dev]"
))

cd %start_dir%
venv\Scripts\deactivate
