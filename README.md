<div align="center">
<img width="1200" height="475" alt="GHBanner" src="https://github.com/user-attachments/assets/0aa67016-6eaf-458a-adb2-6e31a0763ed6" />
</div>

# Run and deploy your AI Studio app

This contains everything you need to run your app locally.

View your app in AI Studio: https://ai.studio/apps/drive/1qskcv1njMb3P0GtDfvFkYAkm54hLYmat

## Run Locally

**Prerequisites:** Node.js, Python 3.12+

### 1. Frontend

```bash
npm install
npm run dev
```

### 2. Backend (FastAPI)

```bash
cd server
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

Start the server:

```bash
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

This starts the FastAPI backend at `http://localhost:8000`. The `--reload` flag enables auto-restart on code changes during development.

### 3. Environment Variables

Create a `.env` file in the project root:

```
GEMINI_API_KEY=your_api_key_here
```

> **Note:** Never commit `.env` to version control. It is excluded via `.gitignore`.
