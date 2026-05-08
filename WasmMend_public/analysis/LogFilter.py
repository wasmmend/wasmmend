"""
Log filtering utilities for compile and test output.

Extracts the most relevant information from potentially very long
build/test logs so they can be included in LLM prompts without
overwhelming the context.
"""

import re
from typing import Optional


def filter_compile_log(raw_log: str, max_lines: int = 50) -> str:
    """Extract the most relevant parts of a compilation log.

    Strategy:
      1. Extract lines containing "error:" with ±2 lines of context.
      2. Deduplicate repeated template instantiation notes.
      3. Prepend a one-line summary.
      4. Truncate to *max_lines*.
      5. If no error lines found, fall back to the last 30 lines.

    Args:
        raw_log:   Combined stdout+stderr from the compiler.
        max_lines: Maximum output lines (default 40).

    Returns:
        Filtered log string.
    """
    if not raw_log or not raw_log.strip():
        return "(empty compilation output)"

    lines = raw_log.splitlines()

    # Find lines containing "error:" or "fatal error:"
    error_indices = set()
    for i, line in enumerate(lines):
        if re.search(r'\berror:', line, re.IGNORECASE):
            # Add context: ±2 lines
            for j in range(max(0, i - 4), min(len(lines), i + 5)):
                error_indices.add(j)

    if not error_indices:
        # No explicit error lines — fall back to tail
        tail_n = min(30, max_lines)
        tail = lines[-tail_n:]
        return f"(no 'error:' lines found; showing last {len(tail)} lines)\n" + "\n".join(tail)

    # Collect error lines in order
    sorted_indices = sorted(error_indices)
    extracted = []
    prev_idx = -2
    for idx in sorted_indices:
        if idx > prev_idx + 1:
            extracted.append("...")  # gap indicator
        extracted.append(lines[idx])
        prev_idx = idx

    # Deduplicate "required from here" / "in instantiation of" spam
    deduped = []
    seen_notes = set()
    for line in extracted:
        if re.search(r'note:\s*(required from|in instantiation of)', line, re.IGNORECASE):
            # Keep only the first instance of each unique note
            key = re.sub(r'\d+', 'N', line.strip())
            if key in seen_notes:
                continue
            seen_notes.add(key)
        deduped.append(line)

    # Count actual errors
    error_count = sum(1 for line in deduped
                      if re.search(r'\berror:', line, re.IGNORECASE) and line != "...")
    warning_count = sum(1 for line in deduped
                        if re.search(r'\bwarning:', line, re.IGNORECASE) and line != "...")

    summary = f"({error_count} error(s), {warning_count} warning(s) extracted)"

    # Truncate
    if len(deduped) > max_lines:
        deduped = deduped[:max_lines]
        deduped.append(f"... (truncated, {len(extracted)} total lines)")

    return summary + "\n" + "\n".join(deduped)


def filter_test_log(stdout: str, stderr: str,
                    test_case_name: Optional[str] = None,
                    max_lines: int = 50) -> str:
    """Extract the most relevant parts of a test execution log.

    Strategy:
      1. If *test_case_name* is given, try to extract only output related
         to that test case.
      2. Extract assertion failure lines (FAIL, Assertion, Expected, Actual).
      3. Detect special conditions (segfault, abort, timeout).
      4. Keep the tail of the output (most recent = most relevant).
      5. Truncate to *max_lines*.

    Args:
        stdout:         Test stdout.
        stderr:         Test stderr.
        test_case_name: Optional test case name for targeted extraction.
        max_lines:      Maximum output lines (default 40).

    Returns:
        Filtered log string.
    """
    combined = ""
    if stdout and stdout.strip():
        combined += stdout.strip()
    if stderr and stderr.strip():
        if combined:
            combined += "\n--- stderr ---\n"
        combined += stderr.strip()

    if not combined:
        return "(empty test output)"

    lines = combined.splitlines()

    # Detect special conditions
    special = []
    for line in lines:
        if re.search(r'(Segmentation fault|SIGSEGV|segfault)', line, re.IGNORECASE):
            special.append("SEGFAULT detected")
        elif re.search(r'(Aborted|SIGABRT|abort\(\))', line, re.IGNORECASE):
            special.append("ABORT detected")
        elif re.search(r'(timed?\s*out|TIMEOUT)', line, re.IGNORECASE):
            special.append("TIMEOUT detected")
        elif re.search(r'(terminate called|uncaught exception|exception thrown|throw|std::\w+_error)', line, re.IGNORECASE):
            special.append(f"EXCEPTION detected: {line.strip()}")

    # If test_case_name is provided, try to extract relevant section
    relevant_lines = lines
    if test_case_name:
        # Look for a section that mentions the test case name
        in_section = False
        section_lines = []
        for line in lines:
            if test_case_name in line:
                in_section = True
            if in_section:
                section_lines.append(line)
                # End section on next test case header (GTest format)
                # Don't stop on blank lines — JS exceptions have blanks mid-trace
                if section_lines and re.match(r'^\[\s*(RUN|OK|FAILED)\s*\]', line):
                    if len(section_lines) > 5:  # Heuristic: section should be at least 5 lines to be relevant
                        break
        if section_lines:
            relevant_lines = section_lines

    # Extract assertion/failure lines with context
    failure_indices = set()
    failure_patterns = [
        r'\bFAIL\b', r'\bfailed\b', r'\bassertion\b',
        r'\bExpected\b', r'\bActual\b', r'\bexpected\b',
        r'\bError\b', r'!=', r'Aborted',
        r'terminate called', r'uncaught exception', r'exception thrown',
        r'TypeError', r'ReferenceError', r'RangeError',
        r'throw\b', r'at Object\.\w+', r'at __\w+',
    ]
    combined_pattern = '|'.join(failure_patterns)
    for i, line in enumerate(relevant_lines):
        if re.search(combined_pattern, line, re.IGNORECASE):
            for j in range(max(0, i - 1), min(len(relevant_lines), i + 9)):
                failure_indices.add(j)

    if failure_indices:
        sorted_indices = sorted(failure_indices)
        extracted = []
        prev_idx = -2
        for idx in sorted_indices:
            if idx > prev_idx + 1:
                extracted.append("...")
            extracted.append(relevant_lines[idx])
            prev_idx = idx
    else:
        # No assertion lines found — use tail
        tail_n = min(30, max_lines)
        extracted = relevant_lines[-tail_n:]

    # Combine special conditions + extracted lines
    parts = []
    if special:
        parts.extend(list(set(special)))
    parts.extend(extracted)

    # Strip test-framework separator noise (Catch2/GTest) before truncating.
    # These eat line budget without adding information.
    parts = [
        line for line in parts
        if not re.match(r'^[-=.]{20,}$', line.strip())
    ]

    # Truncate
    if len(parts) > max_lines:
        parts = parts[:max_lines]
        parts.append(f"... (truncated)")

    return "\n".join(parts)
