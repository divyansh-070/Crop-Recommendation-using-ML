# Setup project virtualenv and install requirements
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Write-Host "Virtual environment created and dependencies installed. Activate with .\.venv\Scripts\Activate.ps1"