# AskZac

## Prerequisites
- Python 3.10+ (`python --version` or `python3 --version`)
- Git

## 1) Get the code
```bash
git clone https://github.com/CodedbyZach/ask-zac.git
cd AskZac
```

## 2) Create & activate a virtual environment
**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

## 3) Install requirements
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 4) Create and edit `.env`
AskZac env needs:
- `OPENAI_API_KEY`
- `SERPAPI_API_KEY`

**Windows (PowerShell):**
```powershell
@"
OPENAI_API_KEY=your_openai_key_here
SERPAPI_API_KEY=your_serpapi_key_here
"@ | Out-File -Encoding ascii .env
```

**macOS / Linux:**
```bash
cat > .env << 'EOF'
OPENAI_API_KEY=your_openai_key_here
SERPAPI_API_KEY=your_serpapi_key_here
EOF
```

## 5) Run the app
```bash
python main.py
```

## Stop / clean up
```bash
deactivate
```
