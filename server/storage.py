"""
JSON file-based storage service for user auth and report persistence.
Converted from services/storage.ts (localStorage) to server-side JSON files.
"""

import json
import os
import uuid
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from filelock import FileLock  # pip install filelock

from models import User, UserWithPassword, SavedReport, ExpertCorrection


DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
USERS_FILE = os.path.join(DATA_DIR, "users.json")
REPORTS_FILE = os.path.join(DATA_DIR, "reports.json")
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


# --- Reports ---

def save_report(user_id: str, report: Dict[str, Any], video_name: str) -> Dict[str, Any]:
    """Save an analysis report for a user. Returns the saved report with id and date."""
    reports_map: Dict[str, List[Dict]] = _read_json(REPORTS_FILE, {})
    user_reports = reports_map.get(user_id, [])

    saved = {
        **report,
        "id": str(uuid.uuid4()),
        "date": datetime.now(timezone.utc).isoformat(),
        "videoName": video_name,
    }

    user_reports.insert(0, saved)  # Add to top (newest first)
    reports_map[user_id] = user_reports
    _write_json(REPORTS_FILE, reports_map)

    return saved


def get_reports(user_id: str) -> List[Dict[str, Any]]:
    """Get all reports for a user."""
    reports_map: Dict[str, List[Dict]] = _read_json(REPORTS_FILE, {})
    return reports_map.get(user_id, [])


def delete_report(user_id: str, report_id: str):
    """Delete a specific report."""
    reports_map: Dict[str, List[Dict]] = _read_json(REPORTS_FILE, {})
    user_reports = reports_map.get(user_id, [])
    reports_map[user_id] = [r for r in user_reports if r.get("id") != report_id]
    _write_json(REPORTS_FILE, reports_map)


def save_expert_correction(user_id: str, report_id: str, correction: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Attach an expert correction to a report."""
    reports_map: Dict[str, List[Dict]] = _read_json(REPORTS_FILE, {})
    user_reports: List[Dict] = reports_map.get(user_id, [])

    updated = None
    for r in user_reports:
        if r.get("id") == report_id:
            r["expertCorrection"] = correction
            updated = r
            break

    if updated:
        reports_map[user_id] = user_reports
        _write_json(REPORTS_FILE, reports_map)

    return updated


def get_training_examples() -> List[Dict[str, Any]]:
    """Get all expert-corrected reports across all users (last 10)."""
    reports_map: Dict[str, List[Dict]] = _read_json(REPORTS_FILE, {})
    examples = []

    for user_reports in reports_map.values():
        for r in user_reports:
            if r.get("expertCorrection"):
                examples.append({
                    "inputs": r.get("rawData"),
                    "groundTruth": r["expertCorrection"]
                })

    return examples[:10]


def get_learned_stats(user_id: str) -> Dict[str, Any]:
    """Get aggregated correction stats for dashboard."""
    reports_map: Dict[str, List[Dict]] = _read_json(REPORTS_FILE, {})
    user_reports = reports_map.get(user_id, [])

    corrections = [r for r in user_reports if r.get("expertCorrection")]
    by_category: Dict[str, int] = {}

    for r in corrections:
        cat = r["expertCorrection"].get("correctClassification", "Unknown")
        by_category[cat] = by_category.get(cat, 0) + 1

    return {
        "totalLearned": len(corrections),
        "breakdown": by_category
    }
