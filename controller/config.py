"""Phase 2 configuration: LLM settings, memory tiers, timing, prompts.

Priority: hardcoded defaults < .env file < environment variables.
"""

import os

# ---------------------------------------------------------------------------
# .env file loader
# ---------------------------------------------------------------------------

def _load_dotenv():
    """Parse KEY=VALUE pairs from .env in the project root.

    Skips blank lines, comments (#), and lines without '='.
    Strips optional quotes from values.
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dotenv_path = os.path.join(project_root, ".env")
    result = {}
    try:
        with open(dotenv_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()
                # Strip surrounding quotes
                if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
                    value = value[1:-1]
                result[key] = value
    except IOError:
        pass  # No .env file is fine
    return result


_dotenv = _load_dotenv()
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _get(key, default=""):
    """Get config value: env var > .env file > default."""
    env_val = os.environ.get(key)
    if env_val is not None:
        return env_val
    return _dotenv.get(key, default)


# ---------------------------------------------------------------------------
# LLM backend selection
# ---------------------------------------------------------------------------
LLM_BACKEND = _get("LLM_BACKEND", "claude")

# ---------------------------------------------------------------------------
# Claude backend
# ---------------------------------------------------------------------------
CLAUDE_API_KEY = _get("ANTHROPIC_API_KEY")
CLAUDE_MODEL = _get("CLAUDE_MODEL", "claude-sonnet-4-20250514")
CLAUDE_API_URL = "https://api.anthropic.com/v1/messages"
CLAUDE_API_VERSION = "2023-06-01"

# ---------------------------------------------------------------------------
# OpenAI-compatible backend (OpenRouter, Aphrodite, vLLM, etc.)
# ---------------------------------------------------------------------------
OPENAI_API_URL = _get("OPENAI_API_URL")
OPENAI_API_KEY = _get("OPENAI_API_KEY")
OPENAI_MODEL = _get("OPENAI_MODEL", "default")

# ---------------------------------------------------------------------------
# Summarizer overrides (falls back to main backend if not set)
# ---------------------------------------------------------------------------
SUMMARIZER_BACKEND = _get("SUMMARIZER_BACKEND")
SUMMARIZER_API_URL = _get("SUMMARIZER_API_URL")
SUMMARIZER_API_KEY = _get("SUMMARIZER_API_KEY")
SUMMARIZER_MODEL = _get("SUMMARIZER_MODEL")

# ---------------------------------------------------------------------------
# Common LLM parameters
# ---------------------------------------------------------------------------
LLM_MAX_TOKENS = 500
LLM_TEMPERATURE = 0.7
LLM_TIMEOUT = 30
LLM_RETRIES = 2
STREAMING_ENABLED = True  # stream LLM tokens to UI; disable for incompatible backends

# ---------------------------------------------------------------------------
# Context manager -- memory tiers
# ---------------------------------------------------------------------------

# Immediate tier: raw events, kept briefly
IMMEDIATE_MAX_ENTRIES = 60
IMMEDIATE_MAX_AGE = 30.0  # seconds

# Short-term tier: summarized batches
SHORT_TERM_MAX_ENTRIES = 20
SHORT_TERM_MAX_AGE = 300.0  # 5 minutes
SUMMARIZE_INTERVAL = 15.0  # seconds between summarization runs

# Long-term tier: compressed summaries
LONG_TERM_MAX_ENTRIES = 10
COMPRESS_INTERVAL = 60.0  # seconds between compression runs

# ---------------------------------------------------------------------------
# Token budget
# ---------------------------------------------------------------------------
TOKEN_BUDGET_TOTAL = 3000
TOKEN_BUDGET_FACTS_RATIO = 0.10
TOKEN_BUDGET_SESSION_HISTORY_RATIO = 0.05
TOKEN_BUDGET_IMMEDIATE_RATIO = 0.42
TOKEN_BUDGET_SHORT_TERM_RATIO = 0.25
TOKEN_BUDGET_LONG_TERM_RATIO = 0.18
CHARS_PER_TOKEN = 4  # rough estimate

# Facts tier
FACTS_MAX_ENTRIES = 20

# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
PERSISTENCE_ENABLED = _get("PERSISTENCE_ENABLED", "true").lower() in ("true", "1", "yes")
PERSISTENCE_DIR = _get("PERSISTENCE_DIR", os.path.join(_PROJECT_ROOT, ".companion"))
MAX_SESSION_HISTORY = int(_get("MAX_SESSION_HISTORY", "5"))
SESSION_SAVE_INTERVAL = 60.0

# ---------------------------------------------------------------------------
# Controller timing
# ---------------------------------------------------------------------------
SCENE_POLL_INTERVAL = 2.0  # seconds between scene description polls
PERIODIC_REASONING_INTERVAL = 30.0  # seconds between periodic reasoning
REACTIVE_COOLDOWN = 5.0  # minimum seconds between reactive reasoning calls

# Reasoning worker
REASONING_STALE_THRESHOLD = 10.0  # drop requests older than this (seconds)
REASONING_QUEUE_SIZE = 2           # max queued reasoning requests

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

REASONING_PROMPT = (
    "You are a robot companion observing the world through a camera. "
    "Based on the context below, provide a brief, natural observation or reaction "
    "to what you see. Be conversational and concise (1-3 sentences). "
    "If a person is present, acknowledge them. If the scene changed, note what changed. "
    "If nothing interesting is happening, say so briefly."
)

SUMMARIZE_PROMPT = (
    "Summarize the following sequence of vision events into one concise sentence. "
    "Preserve key facts: who/what appeared or left, significant movements, object counts. "
    "Drop redundant or trivial details. Output only the summary sentence."
)

COMPRESS_PROMPT = (
    "Compress the following summaries into one brief sentence capturing the overall "
    "situation and any important patterns. Drop timestamps and minor details. "
    "Output only the compressed sentence."
)

USER_REASONING_PROMPT = (
    "You are a robot companion observing the world through a camera. "
    "A person is talking to you. Answer using what you see and remember. "
    "Be conversational, friendly, and concise (1-3 sentences). "
    "Refer to things you can see in the scene when relevant."
)

FACT_EXTRACTION_PROMPT = (
    "Extract persistent facts worth remembering from this conversation exchange. "
    "Focus on: things the user explicitly asks you to remember, the user's name, "
    "preferences, interests, or important personal context. "
    "Do NOT extract facts about what the camera sees or the current scene. "
    "If the user asks you to remember something, always extract it. "
    "Output one fact per line. If nothing worth remembering, output 'none'."
)
