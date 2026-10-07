# Warehouse Robot AI Simulator — Working Build

A Flask-based interactive warehouse robot simulator with:

- Autonomous robot movement and A* path planning
- Dynamic obstacles and reactive replanning
- Battery management and automatic charging
- Random Forest ML order-selection policy
- Groq LLM operations assistant using a server-side API key
- Live seed/order/obstacle controls
- Rush-order injection
- Headless benchmark endpoint
- Thread-safe per-browser simulation sessions

## 1. Run on Windows

### PowerShell

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
.\run.ps1
```

### Command Prompt

```bat
run.bat
```

Or manually:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
copy .env.example .env
python app.py
```

Open **http://127.0.0.1:5000**.

## 2. Groq LLM

Create a Groq API key and put it in `.env`:

```env
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=openai/gpt-oss-120b
GROQ_BASE_URL=https://api.groq.com/openai/v1
```

The browser never receives the key. The Flask server calls Groq's OpenAI-compatible Chat Completions endpoint.

If no key is configured, the simulator and ML agent still work; only the LLM chat is disabled.

## 3. Controls

- **Play** — continuously advance the simulation
- **Step** — advance one tick
- **Finish run** — run until completion or the safety tick limit
- **Reset** — rebuild the scenario
- **Add rush order** — add a priority-3 order
- **Click the map** — add/remove a dynamic obstacle
- **Seed / Orders / Obstacles** — automatically rebuild the active scenario

## 4. Validation

The autonomous simulation is deterministic for a given seed and has been checked across multiple seeded runs. The browser JavaScript and Python modules are syntax-checked before packaging.

## 5. API endpoints

- `GET /` — simulator UI
- `POST /api/reset` — create/reset simulation
- `POST /api/step` — advance simulation
- `POST /api/order` — add rush order
- `POST /api/obstacle` — add/remove obstacle
- `GET /api/benchmark?runs=10` — benchmark runs
- `GET /api/ai-model` — ML model information
- `GET /api/llm-status` — Groq configuration status
- `POST /api/groq` — ask the LLM about the live simulation
- `GET /api/health` — server health check
