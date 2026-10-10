# Smart Semsar — UI Starter

A clean Streamlit-only UI prototype for an AI-powered real-estate assistant.

## Current scope
- UI and demo data only.
- No AI model, API, database, or external service is called.
- Page modules live in `views/`, not Streamlit's special `pages/` folder, so the app uses one custom sidebar navigation.
- `services/` is the integration boundary for future backend/API work.

## Setup (Windows PowerShell)

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

If PowerShell blocks activation, run:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## Project structure

```text
smart_semsar_ui/
├── app.py
├── components/
├── data/
├── models/
├── services/
├── utils/
├── views/
├── assets/
├── .streamlit/config.toml
├── .gitignore
├── requirements.txt
└── README.md
```

## Future AI integration design
Keep UI code inside `views/` and reusable widgets inside `components/`.
When ready to integrate AI:
1. Implement business logic in `services/`.
2. Validate payloads with models in `models/`.
3. Let views call service functions, never call model providers directly from the UI.
4. Store only transient UI state in `st.session_state`; use a database for persistent data later.
