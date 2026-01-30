"""Phase 2 configuration: LLM settings, memory tiers, timing, prompts."""

import os

# ---------------------------------------------------------------------------
# LLM backend selection
# ---------------------------------------------------------------------------
LLM_BACKEND = os.environ.get("LLM_BACKEND", "claude")

# ---------------------------------------------------------------------------
# Claude backend
# ---------------------------------------------------------------------------
CLAUDE_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL = "claude-sonnet-4-20250514"
CLAUDE_API_URL = "https://api.anthropic.com/v1/messages"
CLAUDE_API_VERSION = "2023-06-01"

# ---------------------------------------------------------------------------
# Aphrodite backend (OpenAI-compatible)
# ---------------------------------------------------------------------------
APHRODITE_API_URL = os.environ.get("APHRODITE_API_URL", "")
APHRODITE_API_KEY = os.environ.get("APHRODITE_API_KEY", "")
APHRODITE_MODEL = os.environ.get("APHRODITE_MODEL", "default")

# ---------------------------------------------------------------------------
# Common LLM parameters
# ---------------------------------------------------------------------------
LLM_MAX_TOKENS = 500
LLM_TEMPERATURE = 0.7
LLM_TIMEOUT = 30
LLM_RETRIES = 2

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
TOKEN_BUDGET_IMMEDIATE_RATIO = 0.50
TOKEN_BUDGET_SHORT_TERM_RATIO = 0.30
TOKEN_BUDGET_LONG_TERM_RATIO = 0.20
CHARS_PER_TOKEN = 4  # rough estimate

# ---------------------------------------------------------------------------
# Controller timing
# ---------------------------------------------------------------------------
SCENE_POLL_INTERVAL = 2.0  # seconds between scene description polls
PERIODIC_REASONING_INTERVAL = 30.0  # seconds between periodic reasoning
REACTIVE_COOLDOWN = 5.0  # minimum seconds between reactive reasoning calls

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
