# Virtual Try-On System

A Python-based virtual try-on application using computer vision.

## Setup

1. Create virtual environment: `python -m venv venv`
2. Activate: `.\venv\Scripts\Activate.ps1`
3. Install: `pip install -r requirements.txt`
4. Run tests: `pytest tests/ -v`

## Run

- Frontend (Streamlit): `streamlit run app.py`
- Backend (Flask API): `python -m backend.api`

## Team Structure

- `frontend/` - Streamlit UI (Vansh)
- `backend/` - Flask API layer (Megha)
- `ml_ai/core/` - ML/CV modules (Het)
- `database/` - Data, config, and model assets (Ronak)
- `tests/` - Test files

## Compatibility

- Existing imports like `from src...` still work via compatibility wrappers in `src/`.
- Root `app.py` still works and now forwards to `frontend/app.py`.
