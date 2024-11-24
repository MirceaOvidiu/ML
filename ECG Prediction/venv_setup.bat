@echo off

:: Create a virtual environment in the .venv directory
python -m venv .venv

:: Activate the virtual environment
call .venv\Scripts\activate

:: Upgrade pip
pip install --upgrade pip

:: Install the dependencies from requirements.txt
pip install -r requirements.txt

echo Setup complete. Virtual environment created and dependencies installed.