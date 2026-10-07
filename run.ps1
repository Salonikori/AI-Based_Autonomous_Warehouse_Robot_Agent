$ErrorActionPreference = 'Stop'
if (-not (Test-Path '.venv')) { python -m venv .venv }
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
if (-not (Test-Path '.env')) { Copy-Item '.env.example' '.env' }
& .\.venv\Scripts\python.exe app.py
