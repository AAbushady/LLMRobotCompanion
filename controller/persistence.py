"""Persistent storage for facts and session history.

Pure file I/O -- no LLM calls.  All writes are atomic (temp file + os.rename).
"""

import json
import logging
import os
import tempfile
import time

logger = logging.getLogger(__name__)


def ensure_dirs(base_dir):
    """Create base_dir and base_dir/sessions/ if they don't exist.

    Returns:
        True if directories exist (or were created), False on failure.
    """
    sessions_dir = os.path.join(base_dir, "sessions")
    try:
        if not os.path.isdir(base_dir):
            os.makedirs(base_dir, 0o700)
        if not os.path.isdir(sessions_dir):
            os.makedirs(sessions_dir, 0o700)
        return True
    except OSError:
        logger.warning("Failed to create persistence dirs: %s", base_dir,
                       exc_info=True)
        return False


def _atomic_write(path, data):
    """Write JSON data atomically via temp file + rename."""
    dir_name = os.path.dirname(path)
    fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
        os.rename(tmp_path, path)
    except Exception:
        # Clean up temp file on failure
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def save_facts(base_dir, facts):
    """Write facts to base_dir/facts.json atomically.

    Args:
        base_dir: persistence directory path.
        facts: list of fact strings.
    """
    path = os.path.join(base_dir, "facts.json")
    data = {
        "facts": list(facts),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    _atomic_write(path, data)
    logger.debug("Saved %d facts to %s", len(facts), path)


def load_facts(base_dir):
    """Read facts from base_dir/facts.json.

    Returns:
        list of fact strings, or [] on missing/corrupt file.
    """
    path = os.path.join(base_dir, "facts.json")
    try:
        with open(path, "r") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            logger.warning("facts.json: expected dict, got %s", type(data).__name__)
            return []
        facts = data.get("facts", [])
        if not isinstance(facts, list):
            logger.warning("facts.json: 'facts' is not a list")
            return []
        return [f for f in facts if isinstance(f, str)]
    except (IOError, OSError):
        return []  # file doesn't exist yet
    except (ValueError, KeyError):
        logger.warning("Corrupt facts.json at %s", path, exc_info=True)
        return []


def save_session(base_dir, session_data):
    """Write a session file to base_dir/sessions/{session_id}.json atomically.

    Args:
        base_dir: persistence directory path.
        session_data: dict with session_id, started_at, ended_at,
                      short_term_entries, long_term_entries.
    """
    session_id = session_data["session_id"]
    path = os.path.join(base_dir, "sessions", "{}.json".format(session_id))
    _atomic_write(path, session_data)
    logger.debug("Saved session %s", session_id)


def load_recent_sessions(base_dir, max_sessions):
    """Load the N most recent session files, newest first.

    Args:
        base_dir: persistence directory path.
        max_sessions: maximum number of sessions to load.

    Returns:
        list of session dicts, newest first.  Corrupt files are skipped.
    """
    sessions_dir = os.path.join(base_dir, "sessions")
    if not os.path.isdir(sessions_dir):
        return []

    # List JSON files, sort by name descending (timestamp-based names sort correctly)
    try:
        files = [f for f in os.listdir(sessions_dir) if f.endswith(".json")]
    except OSError:
        logger.warning("Failed to list sessions dir", exc_info=True)
        return []

    files.sort(reverse=True)  # newest first

    sessions = []
    for fname in files:
        if len(sessions) >= max_sessions:
            break
        path = os.path.join(sessions_dir, fname)
        try:
            with open(path, "r") as f:
                data = json.load(f)
            if not isinstance(data, dict) or "session_id" not in data:
                logger.warning("Skipping invalid session file: %s", fname)
                continue
            sessions.append(data)
        except (IOError, OSError, ValueError):
            logger.warning("Skipping corrupt session file: %s", fname,
                           exc_info=True)
            continue

    return sessions


def generate_session_id():
    """Generate a session ID from the current time.

    Returns:
        str like "2026-02-05T14-01-55".
    """
    return time.strftime("%Y-%m-%dT%H-%M-%S")


def format_session_label(started_at):
    """Format a session start timestamp as a human-readable label.

    Args:
        started_at: float (time.time() value).

    Returns:
        str like "[From session on Feb 5, 1:20 PM]".
    """
    t = time.localtime(started_at)
    month = time.strftime("%b", t)
    day = t.tm_mday
    hour = t.tm_hour % 12
    if hour == 0:
        hour = 12
    minute = time.strftime("%M", t)
    ampm = "AM" if t.tm_hour < 12 else "PM"
    return "[From session on {} {}, {}:{} {}]".format(month, day, hour, minute, ampm)
