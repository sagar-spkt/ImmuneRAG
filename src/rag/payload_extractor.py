"""
Payload extraction for `tool_output_misaligned` test examples.

Splits a user-turn string of the form
    "<benign query>\\n\\n[TOOL_OUTPUT_UNTRUSTED]<tool>...payload...</tool>"
into the benign query (used as the retrieval embedding query) and the
attack payload (used as the oracle document indexed in Chroma).
"""

import re
from typing import Tuple


class PayloadExtractionError(ValueError):
    """Raised when a user turn does not contain a recoverable attack payload."""


_PRIMARY = re.compile(
    r"^(?P<query>.*?)\s*\[TOOL_OUTPUT_UNTRUSTED\]\s*<tool>(?P<payload>.*?)</tool>\s*$",
    re.DOTALL,
)
_MARKER = "[TOOL_OUTPUT_UNTRUSTED]"
_OPEN_TAG = "<tool>"


def extract_payload(user_content: str) -> Tuple[str, str]:
    """
    Return (stripped_query, attack_payload).

    Falls back through three tiers:
      1. Anchored regex with both <tool>…</tool> tags present.
      2. Split on `[TOOL_OUTPUT_UNTRUSTED]<tool>` if the closing tag is missing.
      3. Split on the bare `[TOOL_OUTPUT_UNTRUSTED]` marker if no <tool> tag.

    Raises PayloadExtractionError if no marker is found or the payload is empty.
    """
    if not user_content:
        raise PayloadExtractionError("empty user_content")

    m = _PRIMARY.match(user_content)
    if m:
        query = m.group("query").strip()
        payload = m.group("payload").strip()
        if not payload:
            raise PayloadExtractionError("primary regex matched but payload empty")
        return query, payload

    marker_with_tag = f"{_MARKER}{_OPEN_TAG}"
    if marker_with_tag in user_content:
        head, _, tail = user_content.partition(marker_with_tag)
        payload = tail.split("</tool>", 1)[0].strip()
        if not payload:
            raise PayloadExtractionError("fallback <tool> split produced empty payload")
        return head.strip(), payload

    if _MARKER in user_content:
        head, _, tail = user_content.partition(_MARKER)
        payload = tail.strip()
        if not payload:
            raise PayloadExtractionError("fallback marker split produced empty payload")
        return head.strip(), payload

    raise PayloadExtractionError(
        f"no [TOOL_OUTPUT_UNTRUSTED] marker found in user_content (len={len(user_content)})"
    )
