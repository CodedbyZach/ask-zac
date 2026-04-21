# AskZac

## **Version:** 26.04.1
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

**Windows (PowerShell):**

Please open .env with your desired text editor and change the:
- Web Search API Key
- OpenAI API Key
- Zip Code
- And monitor number

**macOS / Linux:**
```bash
nano .env
```
Please change the:
- Web Search API Key
- OpenAI API Key
- Zip Code
- And monitor number

When you're done, press Ctrl + O, enter, to save, then ctrl X to exit

## 5) Run the app
```bash
chmod -x run.sh
./run.sh
```

## Stop / clean up
```bash
deactivate
```

**Note: Timezone variable in the .env is only used for the weather to be more accurate. Currently, it uses your OS time zone for the time and date displayed on the screen.**
