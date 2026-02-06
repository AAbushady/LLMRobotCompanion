"""Unit tests for persistent memory (facts and session history).

Tests cover persistence I/O (real temp dirs, no mocks) and
context manager integration (mocked LLM, temp dirs).
Python 3.6 compatible -- no f-strings, dataclasses, or walrus.
"""

import json
import os
import shutil
import sys
import tempfile
import time
import unittest

try:
    from unittest import mock
except ImportError:
    import mock

# Ensure controller package is importable
sys.path.insert(0, ".")


# ---------------------------------------------------------------------------
# 1. Persistence I/O (real temp dirs)
# ---------------------------------------------------------------------------
class TestPersistenceIO(unittest.TestCase):
    """Test persistence.py functions with real filesystem operations."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp(prefix="companion_test_")

    def tearDown(self):
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    # -- ensure_dirs --

    def test_ensure_dirs_creates_structure(self):
        from controller.persistence import ensure_dirs
        base = os.path.join(self._tmpdir, "new_dir")
        result = ensure_dirs(base)
        self.assertTrue(result)
        self.assertTrue(os.path.isdir(base))
        self.assertTrue(os.path.isdir(os.path.join(base, "sessions")))

    def test_ensure_dirs_handles_existing(self):
        from controller.persistence import ensure_dirs
        # Create it twice -- should be fine
        ensure_dirs(self._tmpdir)
        result = ensure_dirs(self._tmpdir)
        self.assertTrue(result)

    def test_ensure_dirs_returns_false_on_failure(self):
        from controller.persistence import ensure_dirs
        # Use a path under a file (not a directory) to force failure
        fake_file = os.path.join(self._tmpdir, "afile")
        with open(fake_file, "w") as f:
            f.write("x")
        bad_path = os.path.join(fake_file, "subdir")
        result = ensure_dirs(bad_path)
        self.assertFalse(result)

    # -- save_facts / load_facts --

    def test_facts_round_trip(self):
        from controller.persistence import save_facts, load_facts
        facts = ["The user is named Alex", "They like coffee"]
        save_facts(self._tmpdir, facts)
        loaded = load_facts(self._tmpdir)
        self.assertEqual(loaded, facts)

    def test_load_facts_missing_file(self):
        from controller.persistence import load_facts
        loaded = load_facts(self._tmpdir)
        self.assertEqual(loaded, [])

    def test_load_facts_corrupt_json(self):
        from controller.persistence import load_facts
        path = os.path.join(self._tmpdir, "facts.json")
        with open(path, "w") as f:
            f.write("{not valid json")
        loaded = load_facts(self._tmpdir)
        self.assertEqual(loaded, [])

    def test_load_facts_wrong_structure(self):
        from controller.persistence import load_facts
        path = os.path.join(self._tmpdir, "facts.json")
        with open(path, "w") as f:
            json.dump(["just", "a", "list"], f)
        loaded = load_facts(self._tmpdir)
        self.assertEqual(loaded, [])

    def test_load_facts_facts_not_list(self):
        from controller.persistence import load_facts
        path = os.path.join(self._tmpdir, "facts.json")
        with open(path, "w") as f:
            json.dump({"facts": "not a list"}, f)
        loaded = load_facts(self._tmpdir)
        self.assertEqual(loaded, [])

    def test_load_facts_filters_non_strings(self):
        from controller.persistence import load_facts
        path = os.path.join(self._tmpdir, "facts.json")
        with open(path, "w") as f:
            json.dump({"facts": ["valid", 123, None, "also valid"]}, f)
        loaded = load_facts(self._tmpdir)
        self.assertEqual(loaded, ["valid", "also valid"])

    def test_save_facts_atomic_write(self):
        from controller.persistence import save_facts
        # First save
        save_facts(self._tmpdir, ["fact1"])
        # Verify no .tmp files remain
        files = os.listdir(self._tmpdir)
        tmp_files = [f for f in files if f.endswith(".tmp")]
        self.assertEqual(tmp_files, [])

    # -- save_session / load_recent_sessions --

    def test_session_round_trip(self):
        from controller.persistence import (
            save_session, load_recent_sessions, ensure_dirs
        )
        ensure_dirs(self._tmpdir)
        session = {
            "session_id": "2026-02-05T14-01-55",
            "started_at": 1738700000.0,
            "ended_at": 1738703600.0,
            "short_term_entries": [
                {"timestamp": 1738700100.0, "text": "Person entered."}
            ],
            "long_term_entries": [
                {"timestamp": 1738700200.0, "text": "Scene was busy."}
            ],
        }
        save_session(self._tmpdir, session)
        loaded = load_recent_sessions(self._tmpdir, 5)
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0]["session_id"], "2026-02-05T14-01-55")
        self.assertEqual(loaded[0]["short_term_entries"][0]["text"],
                         "Person entered.")

    def test_sessions_ordering_newest_first(self):
        from controller.persistence import (
            save_session, load_recent_sessions, ensure_dirs
        )
        ensure_dirs(self._tmpdir)
        for sid in ["2026-02-01T10-00-00", "2026-02-03T10-00-00",
                     "2026-02-02T10-00-00"]:
            save_session(self._tmpdir, {
                "session_id": sid,
                "started_at": 1000.0,
                "ended_at": 2000.0,
                "short_term_entries": [],
                "long_term_entries": [],
            })
        loaded = load_recent_sessions(self._tmpdir, 5)
        ids = [s["session_id"] for s in loaded]
        self.assertEqual(ids, [
            "2026-02-03T10-00-00",
            "2026-02-02T10-00-00",
            "2026-02-01T10-00-00",
        ])

    def test_sessions_max_limit(self):
        from controller.persistence import (
            save_session, load_recent_sessions, ensure_dirs
        )
        ensure_dirs(self._tmpdir)
        for i in range(10):
            save_session(self._tmpdir, {
                "session_id": "2026-02-{:02d}T10-00-00".format(i + 1),
                "started_at": 1000.0 + i,
                "ended_at": None,
                "short_term_entries": [],
                "long_term_entries": [],
            })
        loaded = load_recent_sessions(self._tmpdir, 3)
        self.assertEqual(len(loaded), 3)

    def test_sessions_skip_corrupt(self):
        from controller.persistence import (
            save_session, load_recent_sessions, ensure_dirs
        )
        ensure_dirs(self._tmpdir)
        # Write a valid session
        save_session(self._tmpdir, {
            "session_id": "2026-02-01T10-00-00",
            "started_at": 1000.0,
            "ended_at": None,
            "short_term_entries": [],
            "long_term_entries": [],
        })
        # Write a corrupt file
        corrupt_path = os.path.join(
            self._tmpdir, "sessions", "2026-02-02T10-00-00.json"
        )
        with open(corrupt_path, "w") as f:
            f.write("{corrupt")
        # Write a file missing session_id
        bad_path = os.path.join(
            self._tmpdir, "sessions", "2026-02-03T10-00-00.json"
        )
        with open(bad_path, "w") as f:
            json.dump({"no_session_id": True}, f)

        loaded = load_recent_sessions(self._tmpdir, 5)
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0]["session_id"], "2026-02-01T10-00-00")

    def test_sessions_empty_dir(self):
        from controller.persistence import load_recent_sessions, ensure_dirs
        ensure_dirs(self._tmpdir)
        loaded = load_recent_sessions(self._tmpdir, 5)
        self.assertEqual(loaded, [])

    def test_sessions_no_dir(self):
        from controller.persistence import load_recent_sessions
        loaded = load_recent_sessions(
            os.path.join(self._tmpdir, "nonexistent"), 5
        )
        self.assertEqual(loaded, [])

    # -- generate_session_id --

    def test_generate_session_id_format(self):
        from controller.persistence import generate_session_id
        sid = generate_session_id()
        # Should look like "2026-02-05T14-01-55"
        self.assertEqual(len(sid), 19)
        self.assertEqual(sid[4], "-")
        self.assertEqual(sid[7], "-")
        self.assertEqual(sid[10], "T")
        self.assertEqual(sid[13], "-")
        self.assertEqual(sid[16], "-")

    # -- format_session_label --

    def test_format_session_label(self):
        from controller.persistence import format_session_label
        # Feb 5, 2026 1:20 PM (approximate -- depends on timezone)
        t = 1738778400.0  # some timestamp
        label = format_session_label(t)
        self.assertTrue(label.startswith("[From session on "))
        self.assertTrue(label.endswith("]"))


# ---------------------------------------------------------------------------
# 2. Context Manager Persistence Integration
# ---------------------------------------------------------------------------
class TestContextManagerPersistence(unittest.TestCase):
    """Test ContextManager persistence with mocked LLM and temp dirs."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp(prefix="companion_ctx_test_")
        # Patch config values
        self._patches = []
        for attr, val in [
            ("PERSISTENCE_ENABLED", True),
            ("PERSISTENCE_DIR", self._tmpdir),
            ("MAX_SESSION_HISTORY", 5),
            ("SESSION_SAVE_INTERVAL", 60.0),
        ]:
            p = mock.patch("controller.config.{}".format(attr), val)
            p.start()
            self._patches.append(p)

    def tearDown(self):
        for p in self._patches:
            p.stop()
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def _make_mock_llm(self):
        llm = mock.MagicMock()
        llm.complete.return_value = "summary"
        return llm

    def _make_cm(self):
        from controller.context_manager import ContextManager
        cm = ContextManager(self._make_mock_llm())
        return cm

    # -- Facts persistence --

    def test_facts_saved_when_dirty(self):
        from controller.persistence import load_facts, ensure_dirs
        ensure_dirs(self._tmpdir)
        cm = self._make_cm()
        cm._session_id = "test-session"
        cm._session_started_at = time.time()

        cm.add_fact("test fact")
        cm._save_facts_if_dirty()

        loaded = load_facts(self._tmpdir)
        self.assertEqual(loaded, ["test fact"])

    def test_facts_not_saved_when_clean(self):
        from controller.persistence import ensure_dirs
        ensure_dirs(self._tmpdir)
        cm = self._make_cm()
        cm._session_id = "test-session"
        cm._session_started_at = time.time()

        # Don't add any facts, just try to save
        cm._save_facts_if_dirty()

        path = os.path.join(self._tmpdir, "facts.json")
        self.assertFalse(os.path.exists(path))

    def test_facts_loaded_on_start(self):
        from controller.persistence import save_facts, ensure_dirs
        ensure_dirs(self._tmpdir)
        save_facts(self._tmpdir, ["persisted fact"])

        cm = self._make_cm()
        # start() loads persisted state but also starts the thread --
        # we'll start and immediately stop
        cm.start()
        cm.stop()

        with cm._lock:
            self.assertIn("persisted fact", cm._facts)

    def test_facts_dedup_on_load(self):
        from controller.persistence import save_facts, ensure_dirs
        ensure_dirs(self._tmpdir)
        save_facts(self._tmpdir, ["Fact A", "Fact B"])

        cm = self._make_cm()
        # Pre-add a fact with same text (different case)
        cm.add_fact("fact a")

        cm.start()
        cm.stop()

        with cm._lock:
            # Should have 2 facts (fact a + Fact B), not 3
            lower_facts = [f.lower() for f in cm._facts]
            self.assertEqual(lower_facts.count("fact a"), 1)
            self.assertIn("fact b", lower_facts)

    # -- Session persistence --

    def test_session_snapshot_saved(self):
        from controller.persistence import ensure_dirs
        ensure_dirs(self._tmpdir)
        cm = self._make_cm()
        cm._session_id = "2026-02-05T14-00-00"
        cm._session_started_at = time.time()

        # Add some data to short-term
        with cm._lock:
            cm._short_term.append({
                "timestamp": time.time(),
                "covers_from": time.time(),
                "covers_to": time.time(),
                "text": "Person entered.",
                "source_count": 1,
            })

        cm._save_session_snapshot()

        path = os.path.join(
            self._tmpdir, "sessions", "2026-02-05T14-00-00.json"
        )
        self.assertTrue(os.path.exists(path))
        with open(path, "r") as f:
            data = json.load(f)
        self.assertEqual(data["session_id"], "2026-02-05T14-00-00")
        self.assertEqual(len(data["short_term_entries"]), 1)
        self.assertIsNone(data["ended_at"])

    def test_final_session_snapshot_has_ended_at(self):
        from controller.persistence import ensure_dirs
        ensure_dirs(self._tmpdir)
        cm = self._make_cm()
        cm._session_id = "2026-02-05T14-00-00"
        cm._session_started_at = time.time()

        cm._save_session_snapshot(final=True)

        path = os.path.join(
            self._tmpdir, "sessions", "2026-02-05T14-00-00.json"
        )
        with open(path, "r") as f:
            data = json.load(f)
        self.assertIsNotNone(data["ended_at"])

    def test_final_snapshot_includes_important_immediate_entries(self):
        from controller.persistence import ensure_dirs
        from controller.context_manager import IMPORTANCE_HIGH, IMPORTANCE_DEFAULT
        ensure_dirs(self._tmpdir)
        cm = self._make_cm()
        cm._session_id = "2026-02-05T14-00-00"
        cm._session_started_at = time.time()

        # Add entries to immediate tier with varying importance
        now = time.time()
        with cm._lock:
            cm._immediate.append({
                "timestamp": now,
                "source": "event",
                "text": "Person entered the scene.",
                "importance": IMPORTANCE_HIGH,
            })
            cm._immediate.append({
                "timestamp": now + 1,
                "source": "scene",
                "text": "Empty scene with no objects.",
                "importance": IMPORTANCE_DEFAULT,
            })
            cm._immediate.append({
                "timestamp": now + 2,
                "source": "user",
                "text": 'User said: "Hello"',
                "importance": 3,  # IMPORTANCE_USER
            })

        cm._save_session_snapshot(final=True)

        path = os.path.join(
            self._tmpdir, "sessions", "2026-02-05T14-00-00.json"
        )
        with open(path, "r") as f:
            data = json.load(f)
        texts = [e["text"] for e in data["short_term_entries"]]
        # High-importance entries should be included
        self.assertIn("Person entered the scene.", texts)
        self.assertIn('User said: "Hello"', texts)
        # Low-importance scene description should NOT be included
        self.assertNotIn("Empty scene with no objects.", texts)

    def test_periodic_snapshot_does_not_include_immediate(self):
        from controller.persistence import ensure_dirs
        from controller.context_manager import IMPORTANCE_HIGH
        ensure_dirs(self._tmpdir)
        cm = self._make_cm()
        cm._session_id = "2026-02-05T14-00-00"
        cm._session_started_at = time.time()

        with cm._lock:
            cm._immediate.append({
                "timestamp": time.time(),
                "source": "event",
                "text": "Person entered the scene.",
                "importance": IMPORTANCE_HIGH,
            })

        cm._save_session_snapshot(final=False)  # periodic, not final

        path = os.path.join(
            self._tmpdir, "sessions", "2026-02-05T14-00-00.json"
        )
        with open(path, "r") as f:
            data = json.load(f)
        self.assertEqual(data["short_term_entries"], [])

    def test_session_loaded_on_start(self):
        from controller.persistence import save_session, ensure_dirs
        ensure_dirs(self._tmpdir)
        save_session(self._tmpdir, {
            "session_id": "2026-02-04T10-00-00",
            "started_at": 1738600000.0,
            "ended_at": 1738603600.0,
            "short_term_entries": [
                {"timestamp": 1738600100.0, "text": "Person entered."}
            ],
            "long_term_entries": [
                {"timestamp": 1738600200.0, "text": "Scene was busy."}
            ],
        })

        cm = self._make_cm()
        cm.start()
        cm.stop()

        with cm._lock:
            self.assertEqual(len(cm._session_history), 1)
            entry = cm._session_history[0]
            self.assertTrue(entry["label"].startswith("[From session on "))
            self.assertIn("Person entered.", entry["text"])
            self.assertIn("Scene was busy.", entry["text"])

    def test_stop_saves_final_snapshot(self):
        from controller.persistence import ensure_dirs
        ensure_dirs(self._tmpdir)

        cm = self._make_cm()
        cm.start()
        # The session_id should have been generated
        self.assertIsNotNone(cm._session_id)

        # Add a fact so we can verify it's saved
        cm.add_fact("shutdown fact")
        cm.stop()

        # Check facts.json was written
        from controller.persistence import load_facts
        facts = load_facts(self._tmpdir)
        self.assertIn("shutdown fact", facts)

        # Check session file exists
        sessions_dir = os.path.join(self._tmpdir, "sessions")
        session_files = [f for f in os.listdir(sessions_dir)
                         if f.endswith(".json")]
        self.assertGreaterEqual(len(session_files), 1)

    # -- build_context includes session history --

    def test_build_context_includes_previous_sessions(self):
        cm = self._make_cm()
        with cm._lock:
            cm._session_history = [{
                "label": "[From session on Feb 4, 10:00 AM]",
                "text": "User asked about the weather.",
            }]

        context = cm.build_context()
        self.assertIn("[Previous sessions]", context)
        self.assertIn("Feb 4", context)
        self.assertIn("User asked about the weather.", context)

    def test_build_context_no_sessions_section_when_empty(self):
        cm = self._make_cm()
        context = cm.build_context()
        self.assertNotIn("[Previous sessions]", context)

    # -- Persistence disabled --

    def test_persistence_disabled_no_files(self):
        with mock.patch("controller.config.PERSISTENCE_ENABLED", False):
            cm = self._make_cm()
            cm.start()
            cm.add_fact("should not persist")
            cm.stop()

        # No facts.json should exist
        path = os.path.join(self._tmpdir, "facts.json")
        self.assertFalse(os.path.exists(path))

    # -- Bad directory graceful degradation --

    def test_bad_dir_no_crash(self):
        # Point to a path that can't be created
        fake_file = os.path.join(self._tmpdir, "afile")
        with open(fake_file, "w") as f:
            f.write("x")
        bad_path = os.path.join(fake_file, "subdir")

        with mock.patch("controller.config.PERSISTENCE_DIR", bad_path):
            cm = self._make_cm()
            cm.start()
            cm.add_fact("should not crash")
            cm.stop()

        # Should not raise, persistence_available should be False
        self.assertFalse(cm._persistence_available)

    # -- get_stats includes sessions --

    def test_get_stats_includes_sessions(self):
        cm = self._make_cm()
        with cm._lock:
            cm._session_history = [{"label": "x", "text": "y"}]
        stats = cm.get_stats()
        self.assertEqual(stats["sessions"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
