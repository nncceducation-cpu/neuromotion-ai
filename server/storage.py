"""
JSON file-based storage service for user auth.
Report storage has been consolidated into gemini_predictions.jsonl (see api.py).
"""

import json
import os
import uuid
from typing import List, Optional, Dict, Any
from filelock import FileLock  # pip install filelock

from models import User


DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
USERS_FILE = os.path.join(DATA_DIR, "users.json")
SESSIONS_FILE = os.path.join(DATA_DIR, "sessions.json")


def _ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)


def _read_json(filepath: str, default: Any = None) -> Any:
    """Read a JSON file, returning default if it doesn't exist."""
    if not os.path.exists(filepath):
        return default if default is not None else {}
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(filepath: str, data: Any):
    """Write data to a JSON file with locking."""
    _ensure_data_dir()
    lock = FileLock(filepath + ".lock")
    with lock:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


# --- Auth ---

def register(name: str, email: str, password: str) -> User:
    """Register a new user. Raises ValueError if email already exists."""
    users: List[Dict] = _read_json(USERS_FILE, [])

    for u in users:
        if u["email"] == email:
            raise ValueError("Email already registered")

    new_user = {
        "id": str(uuid.uuid4()),
        "name": name,
        "email": email,
        "password": password  # In production, hash this
    }
    users.append(new_user)
    _write_json(USERS_FILE, users)

    return User(id=new_user["id"], name=new_user["name"], email=new_user["email"])


def login(email: str, password: str) -> tuple[User, str]:
    """
    Login with email + password. Returns (User, session_token).
    Auto-seeds a demo user if the user list is empty.
    """
    users: List[Dict] = _read_json(USERS_FILE, [])

    # Auto-seed demo user if empty
    if len(users) == 0:
        default_user = {
            "id": "user-demo-123",
            "name": "Dr. Demo User",
            "email": "demo@neuromotion.ai",
            "password": "demo"
        }
        users = [default_user]
        _write_json(USERS_FILE, users)

    user = None
    for u in users:
        if u["email"] == email and u["password"] == password:
            user = u
            break

    if not user:
        raise ValueError("Invalid credentials")

    # Create session token
    token = str(uuid.uuid4())
    sessions: Dict[str, Dict] = _read_json(SESSIONS_FILE, {})
    sessions[token] = {"id": user["id"], "name": user["name"], "email": user["email"]}
    _write_json(SESSIONS_FILE, sessions)

    return User(id=user["id"], name=user["name"], email=user["email"]), token


def logout(token: str):
    """Invalidate a session token."""
    sessions: Dict[str, Dict] = _read_json(SESSIONS_FILE, {})
    sessions.pop(token, None)
    _write_json(SESSIONS_FILE, sessions)


def get_current_user(token: str) -> Optional[User]:
    """Get the user associated with a session token."""
    sessions: Dict[str, Dict] = _read_json(SESSIONS_FILE, {})
    user_data = sessions.get(token)
    if not user_data:
        return None
    return User(**user_data)
