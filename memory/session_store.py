import json, os
from pathlib import Path
from datetime import datetime, timezone
from irt.scorer import StudentState

# ── Drive paths ───────────────────────────────────────
DRIVE_SESSIONS = "/content/drive/MyDrive/adaptiq/sessions"
LOCAL_SESSIONS = "data/sessions"

def get_sessions_dir() -> str:
    """Use Drive if mounted, else local"""
    if os.path.exists("/content/drive/MyDrive"):
        Path(DRIVE_SESSIONS).mkdir(parents=True, exist_ok=True)
        return DRIVE_SESSIONS
    Path(LOCAL_SESSIONS).mkdir(parents=True, exist_ok=True)
    return LOCAL_SESSIONS


def save_session(student_state, session_log: list = None):
    """
    Save student θ scores and session history to Drive

    What: Persists StudentState to JSON file
    Why:  Colab resets every session — without this
          student starts from θ=0 every time
    How:  Serialize StudentState → JSON → Drive file

    File: sessions/{student_id}.json
    """
    sessions_dir = get_sessions_dir()
    filepath     = f"{sessions_dir}/{student_state.student_id}.json"

    data = {
        "student_id":    student_state.student_id,
        "theta":         student_state.theta,
        "correct_count": student_state.correct_count,
        "total_count":   student_state.total_count,
        "answered":      student_state.answered,
        "session_log":   student_state.session_log,
        "saved_at":      datetime.now(timezone.utc).isoformat(),
        "sessions_completed": load_session_count(
            student_state.student_id
        ) + 1
    }

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)

    print(f"✅ Session saved: {filepath}")
    return filepath


def load_session(student_id: str):
    """
    Load student θ scores from previous session

    What: Reads saved StudentState from JSON
    Why:  Restores where student left off
    How:  Read JSON → rebuild StudentState object

    Returns: StudentState with previous θ scores
             OR fresh StudentState if no save found
    """
    #from irt.scorer import StudentState

    sessions_dir = get_sessions_dir()
    filepath     = f"{sessions_dir}/{student_id}.json"

    if not os.path.exists(filepath):
        print(f"📝 No previous session found for {student_id}")
        print(f"   Starting fresh session")
        return StudentState(student_id=student_id)

    with open(filepath) as f:
        data = json.load(f)

    # Rebuild StudentState
    state                = StudentState(student_id=student_id)
    state.theta          = data.get("theta", {})
    state.correct_count  = data.get("correct_count", {})
    state.total_count    = data.get("total_count", {})
    state.answered       = data.get("answered", [])
    state.session_log    = data.get("session_log", [])

    sessions_done = data.get("sessions_completed", 1)
    print(f"✅ Session loaded for {student_id}")
    print(f"   Previous sessions: {sessions_done}")
    print(f"   θ scores restored: {len(state.theta)} categories")
    print(f"   Questions answered before: {len(state.answered)}")

    return state


def load_session_count(student_id: str) -> int:
    """Get number of completed sessions for a student"""
    sessions_dir = get_sessions_dir()
    filepath     = f"{sessions_dir}/{student_id}.json"
    if not os.path.exists(filepath):
        return 0
    with open(filepath) as f:
        data = json.load(f)
    return data.get("sessions_completed", 0)


def get_session_history(student_id: str) -> dict:
    """
    Get full session history for analytics

    Returns dict with θ progression, accuracy per category,
    total questions answered, improvement over time
    """
    sessions_dir = get_sessions_dir()
    filepath     = f"{sessions_dir}/{student_id}.json"

    if not os.path.exists(filepath):
        return {}

    with open(filepath) as f:
        data = json.load(f)

    # Compute accuracy per category
    accuracy = {}
    for cat in data.get("total_count", {}):
        total   = data["total_count"][cat]
        correct = data.get("correct_count", {}).get(cat, 0)
        accuracy[cat] = round(correct / total * 100, 1) if total > 0 else 0

    return {
        "student_id":         student_id,
        "sessions_completed": data.get("sessions_completed", 0),
        "total_answered":     len(data.get("answered", [])),
        "theta":              data.get("theta", {}),
        "accuracy":           accuracy,
        "saved_at":           data.get("saved_at", "")
    }


print("✅ Session memory functions defined")