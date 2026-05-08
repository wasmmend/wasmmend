"""
Response parser for LLM agent outputs.

Extracts the ACTION line, parameters, and action-specific fields
(code blocks, analysis, plan, etc.) from free-form LLM responses.

Response format expected from the agent:

    <free-form reasoning text>

    ACTION: <action_name>; param1=value1; param2=value2

    (for write_patch, additional code blocks follow)
    **Original Code:**
    ```cpp
    <code>
    ```
    **Patched Code:**
    ```cpp
    <code>
    ```

The parser is lenient: it extracts what it can and falls back gracefully.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, Optional


class ParseError(Exception):
    """Raised when the response cannot be parsed after all fallbacks."""
    pass


@dataclass
class ParsedResponse:
    """Structured representation of a parsed LLM response."""
    reasoning: str = ""             # everything before the ACTION line
    action: str = ""                # action name
    params: Dict[str, str] = field(default_factory=dict)

    # For propose_plan / transition_to_patch
    plan: Optional[str] = None

    # For write_patch
    original_code: Optional[str] = None
    fixed_code: Optional[str] = None

    # For analyze_patch
    analysis: Optional[str] = None
    root_cause_addressed: Optional[bool] = None
    same_functionality: Optional[bool] = None
    next_step: Optional[str] = None

    # For analyze_instrumentation
    findings: Optional[str] = None
    implication: Optional[str] = None

    # For give_up
    reason: Optional[str] = None


# ---------------------------------------------------------------------------
# Internal extraction helpers
# ---------------------------------------------------------------------------

def _extract_action_line(text: str):
    """Extract the ACTION line and split into (action_name, params_str, pre_action_text).

    Returns (action_name, params_dict, reasoning_text) or (None, {}, text).
    """
    # Primary pattern: ACTION: action_name; key=value; key=value
    pattern = r'^ACTION:\s*(\w+)\s*(?:;?\s*(.*))?$'
    for line in text.splitlines():
        stripped = line.strip()
        m = re.match(pattern, stripped, re.IGNORECASE)
        if m:
            action_name = m.group(1).strip()
            params_str = m.group(2).strip() if m.group(2) else ""
            # Everything before this line is reasoning
            idx = text.find(stripped)
            reasoning = text[:idx].strip() if idx > 0 else ""
            # Parse key=value pairs
            params = _parse_params(params_str)
            return action_name, params, reasoning

    return None, {}, text


def _parse_params(params_str: str) -> Dict[str, str]:
    """Parse semicolon-separated key=value pairs from the ACTION line."""
    params = {}
    if not params_str:
        return params
    # Split on semicolons, then parse key=value
    for part in re.split(r'\s*;\s*', params_str):
        part = part.strip()
        if not part:
            continue
        if '=' in part:
            key, _, value = part.partition('=')
            params[key.strip()] = value.strip()
    return params


def _extract_labeled_field(text: str, label: str) -> Optional[str]:
    """Extract a labeled field like 'PLAN: ...' or 'ANALYSIS: ...' from the text.

    Captures everything after the label until the next labeled field or end of text.
    """
    # Match the label at start of line (case-insensitive)
    pattern = rf'^\s*{re.escape(label)}:\s*(.*?)(?=^\s*[A-Z_]+:|\Z)'
    m = re.search(pattern, text, re.MULTILINE | re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()

    # Simpler fallback: just find the label line and take content
    for i, line in enumerate(text.splitlines()):
        stripped = line.strip()
        prefix = f"{label}:"
        if stripped.upper().startswith(prefix.upper()):
            content = stripped[len(prefix):].strip()
            # Also grab following lines until next label or blank gap
            following = []
            for next_line in text.splitlines()[i + 1:]:
                if re.match(r'^\s*[A-Z_]+:', next_line.strip()):
                    break
                following.append(next_line)
            if following:
                content += "\n" + "\n".join(following)
            return content.strip()

    return None


def _extract_code_blocks(text: str):
    """Extract **Original Code:** and **Fixed Code:** blocks.

    Returns (original_code, fixed_code) or (None, None).
    """
    original_code = None
    fixed_code = None

    # Pattern for labeled code blocks
    # Handles: **Original Code:** / **Original Code**: / Original Code: / ORIGINAL CODE:
    orig_pattern = r'(?:\*{0,2}Original\s+Code:?\*{0,2}:?|ORIGINAL\s+CODE:?)\s*```(?:cpp|c\+\+|c)?\s*\n(.*?)```'
    fixed_pattern = r'(?:\*{0,2}(?:Fixed|Patched)\s+Code:?\*{0,2}:?|(?:FIXED|PATCHED)\s+CODE:?)\s*```(?:cpp|c\+\+|c)?\s*\n(.*?)```'

    orig_match = re.search(orig_pattern, text, re.DOTALL | re.IGNORECASE)
    fixed_match = re.search(fixed_pattern, text, re.DOTALL | re.IGNORECASE)

    if orig_match and fixed_match:
        original_code = orig_match.group(1).strip()
        fixed_code = fixed_match.group(1).strip()

    return original_code, fixed_code


def _extract_any_code_block(text: str) -> Optional[str]:
    """Extract the first code block (```cpp or ```) from text."""
    for marker in ("```cpp", "```c++", "```"):
        start = text.find(marker)
        if start == -1:
            continue
        start += len(marker)
        nl = text.find('\n', start)
        if nl != -1:
            start = nl + 1
        end = text.find("```", start)
        if end == -1:
            continue
        code = text[start:end].strip()
        if code:
            return code
    return None


# ---------------------------------------------------------------------------
# Main parse function
# ---------------------------------------------------------------------------

def parse_response(response_text: str, current_state: str) -> ParsedResponse:
    """Parse an LLM response into a structured ParsedResponse.

    Attempts multiple strategies:
      1. Extract ACTION line with params.
      2. Extract action-specific fields based on the action name.
      3. Fallback: search for ACTION: anywhere in text.
      4. Fallback: infer action from content (code blocks → write_patch in PATCH).
      5. Raise ParseError if nothing works.

    Args:
        response_text: Raw LLM response text.
        current_state: "ANALYZE" or "PATCH" — used for fallback inference.

    Returns:
        ParsedResponse with extracted fields.

    Raises:
        ParseError: If the response cannot be parsed.
    """
    result = ParsedResponse()

    # Strategy 1: Extract ACTION line
    action, params, reasoning = _extract_action_line(response_text)

    if action:
        result.action = action
        result.params = params
        result.reasoning = reasoning
    else:
        # Fallback 1: Search for "ACTION:" anywhere with more lenient pattern
        fallback_pattern = r'ACTION:\s*(\w+)'
        fb_match = re.search(fallback_pattern, response_text, re.IGNORECASE)
        if fb_match:
            result.action = fb_match.group(1).strip()
            result.reasoning = response_text[:fb_match.start()].strip()
            # Try to extract params from the same line
            line_start = response_text.rfind('\n', 0, fb_match.start())
            line_end = response_text.find('\n', fb_match.end())
            if line_end == -1:
                line_end = len(response_text)
            action_line = response_text[line_start + 1:line_end] if line_start >= 0 else response_text[:line_end]
            # Parse params after the action name
            after_action = action_line[action_line.find(result.action) + len(result.action):]
            result.params = _parse_params(after_action)
        else:
            # Fallback 2: Infer from content
            orig, fixed = _extract_code_blocks(response_text)
            if orig and fixed and current_state == "PATCH":
                result.action = "write_patch"
                result.original_code = orig
                result.fixed_code = fixed
                result.reasoning = response_text
            else:
                raise ParseError(
                    "Could not parse ACTION from response. "
                    "Please format your response with an ACTION: line."
                )

    # Extract action-specific fields
    _extract_action_fields(result, response_text)

    return result


def _extract_action_fields(result: ParsedResponse, response_text: str):
    """Populate action-specific fields on the ParsedResponse."""

    action = result.action.lower() if result.action else ""

    if action == "write_patch":
        if not result.original_code:
            orig, fixed = _extract_code_blocks(response_text)
            result.original_code = orig
            result.fixed_code = fixed
        # Also try to get file_path from params or text
        if "file_path" not in result.params:
            fp = _extract_labeled_field(response_text, "FILE_PATH")
            if fp:
                result.params["file_path"] = fp.strip('`"\'')

    elif action == "propose_plan" or action == "transition_to_patch":
        result.plan = _extract_labeled_field(response_text, "PLAN")
        # Fallback: use everything after ACTION line as the plan
        if not result.plan:
            idx = response_text.lower().find("action:")
            if idx >= 0:
                after = response_text[idx:]
                nl = after.find('\n')
                if nl >= 0:
                    result.plan = after[nl:].strip()

    elif action == "analyze_patch":
        result.analysis = _extract_labeled_field(response_text, "ANALYSIS")
        rca = _extract_labeled_field(response_text, "ROOT_CAUSE_ADDRESSED")
        if rca:
            result.root_cause_addressed = rca.strip().lower() in ("yes", "true")
        else:
            result.root_cause_addressed = False
        sf = _extract_labeled_field(response_text, "SAME_FUNCTIONALITY")
        if sf:
            result.same_functionality = sf.strip().lower() in ("yes", "true")
        else:
            result.same_functionality = False
        result.next_step = _extract_labeled_field(response_text, "NEXT_STEP")

    elif action == "analyze_instrumentation":
        result.findings = _extract_labeled_field(response_text, "FINDINGS")
        result.implication = _extract_labeled_field(response_text, "IMPLICATION")

    elif action == "give_up":
        result.reason = _extract_labeled_field(response_text, "REASON")
        if not result.reason:
            result.reason = result.params.get("reason", result.reasoning)

    elif action == "instrument_function":
        # Extract Original Code block (function source to instrument).
        # Search for **Original Code:** block independently (no Fixed Code needed).
        orig_pattern = r'(?:\*{0,2}Original\s+Code:?\*{0,2}:?|ORIGINAL\s+CODE:?)\s*```(?:cpp|c\+\+|c)?\s*\n(.*?)```'
        orig_match = re.search(orig_pattern, response_text, re.DOTALL | re.IGNORECASE)
        if orig_match:
            result.original_code = orig_match.group(1).strip()
        else:
            # Fallback: try extracting any code block
            result.original_code = _extract_any_code_block(response_text)
        # instructions might be in params or as labeled field
        if "instructions" not in result.params:
            instr = _extract_labeled_field(response_text, "INSTRUCTIONS")
            if instr:
                result.params["instructions"] = instr

    elif action == "view_patch_history":
        # last_n might be in params
        if "last_n" not in result.params:
            result.params["last_n"] = "3"  # default
