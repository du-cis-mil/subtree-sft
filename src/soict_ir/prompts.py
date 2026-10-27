# Adapted for the standalone SoICT release (2026): local imports and CLI paths revised.
# MOOSE-Star-derived code; see LICENSE and NOTICE.
import json
from pathlib import Path

PAPER_PROMPT = "inspiration_retrieval_with_reasoning_with_alphabetical_candidates"
BRANCH_PROMPT = "inspiration_branch_retrieval_with_reasoning_with_alphabetical_candidates"

def instruction_prompts(name):
    prompts = json.loads(Path(__file__).with_name("prompts.json").read_text(encoding="utf-8"))
    if name not in prompts:
        raise ValueError(f"Unsupported SoICT prompt: {name}")
    return prompts[name]
