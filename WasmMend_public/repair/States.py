"""
State definitions and prompt builders for the function-level repairer.

Two states: ANALYZE and PATCH.
Each state has a prompt builder that constructs the LLM prompt based on
the current context, available actions, and response format requirements.
"""

from typing import Dict, List, Optional
from repair.Models import RepairInput, RepairHistory, FunctionInfo
from repair.WorkflowConfig import (
    WorkflowConfig, default_config,
    DEFAULT_ANALYZE_ACTIONS, DEFAULT_PATCH_ACTIONS,
)


# ---------------------------------------------------------------------------
# State constants
# ---------------------------------------------------------------------------

ANALYZE = "ANALYZE"
PATCH = "PATCH"

# Backward-compatible module-level action dicts (point to defaults).
# New code should use WorkflowConfig instead.
ANALYZE_ACTIONS = DEFAULT_ANALYZE_ACTIONS
PATCH_ACTIONS = DEFAULT_PATCH_ACTIONS


# ---------------------------------------------------------------------------
# Prompt format instructions
# ---------------------------------------------------------------------------

RESPONSE_FORMAT_BASE = """
**Response Format:**
Write your reasoning freely, then end with exactly ONE ACTION line.
Do NOT include more than one ACTION line — only one action is executed per turn.

    ACTION: <action_name>; param1=value1; param2=value2; ...
"""

RESPONSE_FORMAT_WRITE_FIX = """
For write_patch, include code blocks after the ACTION line:

    ACTION: write_patch; function_name=<function_name>; file_path=/path/to/file.h

    **Original Code:**
    ```c++
    <exact code to find and replace>
    ```

    **Patched Code:**
    ```c++
    <replacement code>
    ```
"""

RESPONSE_FORMAT_INSTRUMENT = """
For instrument_function, include an **Original Code:** block with the function source:

    ACTION: instrument_function; function_name=<name>; file_path=/path/to/file.h; instructions=<what to observe>

    **Original Code:**
    ```c++
    <the function source code to instrument>
    ```
"""

RESPONSE_FORMAT_LABELED_FIELDS_BASE = """
For actions with labeled fields, include them after the ACTION line.
Example for propose_plan (a different, separate response):

    ACTION: propose_plan
    PLAN: <your plan>
"""

RESPONSE_FORMAT_ANALYZE_PATCH = """
Example for analyze_patch (one complete response):

    ACTION: analyze_patch
    ANALYSIS: <your analysis>
    ROOT_CAUSE_ADDRESSED: Yes/No
    NEXT_STEP: write_patch
"""

# Backward compat
RESPONSE_FORMAT_LABELED_FIELDS = RESPONSE_FORMAT_LABELED_FIELDS_BASE + RESPONSE_FORMAT_ANALYZE_PATCH

# Backward-compatible combined constant (used by tests that import RESPONSE_FORMAT)
RESPONSE_FORMAT = RESPONSE_FORMAT_BASE + RESPONSE_FORMAT_WRITE_FIX + RESPONSE_FORMAT_INSTRUMENT + RESPONSE_FORMAT_LABELED_FIELDS


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _format_candidate_list(candidates: List[FunctionInfo]) -> str:
    """Format candidate functions as a concise list (names + locations)."""
    if not candidates:
        return "(no candidate functions provided)"
    lines = []
    for c in candidates:
        lines.append(f"  - {c.name}() at {c.location_str()}")
    return "\n".join(lines)


def _format_skeleton_graph(skeleton: Dict[str, List[str]]) -> str:
    """Format the type dependency skeleton graph."""
    if not skeleton:
        return "(no type dependencies)"
    lines = []
    for type_name, deps in skeleton.items():
        dep_str = ", ".join(deps) if deps else "(no dependencies)"
        lines.append(f"  {type_name} -> {dep_str}")
    return "\n".join(lines)


def _format_available_actions(actions: Dict[str, str]) -> str:
    """Format available actions for the prompt."""
    lines = []
    for name, desc in actions.items():
        lines.append(f"  - {name}: {desc}")
    return "\n".join(lines)


def _format_test_and_trace_sections(repair_input: 'RepairInput',
                                     config: 'WorkflowConfig') -> str:
    """Format the failing test and optional trace analysis sections.

    Always includes the test case location. Conditionally includes the
    trace analysis discrepancy comparison based on config.provide_trace_analysis.
    """
    parts = []

    # Test case location (always shown)
    parts.append(f"=== Failing Test ===")
    parts.append(f"Test file: {repair_input.test_info.test_path}")
    parts.append(f"Test case: {repair_input.test_info.test_case_name}")
    if repair_input.test_info.failing_statement:
        parts.append(f"Failing statement: {repair_input.test_info.failing_statement}")

    # Trace analysis comparison (optional)
    if config.provide_trace_analysis and repair_input.test_info.initial_failure_output:
        parts.append("")
        parts.append("=== Preprocessing: Discrepancy Analysis ===")
        parts.append("The preprocessing step detected the following wasm-vs-native discrepancy:")
        parts.append(repair_input.test_info.initial_failure_output[:1500])

    return "\n".join(parts)


def _format_function_list(functions: List[FunctionInfo]) -> str:
    """Format a list of functions as a numbered list (names + locations)."""
    if not functions:
        return "(no functions provided)"
    lines = []
    for i, f in enumerate(functions, 1):
        lines.append(f"  {i}. {f.name}() at {f.location_str()}")
    return "\n".join(lines)


def build_analyze_prompt(
    repair_input: RepairInput,
    history: RepairHistory,
    last_action: Optional[str] = None,
    last_action_result: Optional[Dict] = None,
    mandatory_next_action: Optional[str] = None,
    workflow_config: Optional[WorkflowConfig] = None,
) -> str:
    """Build the prompt for the ANALYZE state."""

    config = workflow_config or default_config()
    parts = []

    if config.provide_trace_analysis:
        # --- Original behavior: highlight root cause ---
        parts.append(f"""You are in ANALYZE state for project "{repair_input.project_name}", analyzing a C/C++ Wasm-vs-Native discrepancy.

The preprocessing step has identified a function where the Wasm-vs-Native discrepancy first surfaces.
The actual patch could be in this function, in a caller, or in the broader mechanism it depends on.
Your goal: understand why this discrepancy occurs, then adapt the implementation so that the
feature produces the same functional behavior under Wasm as it does natively. Transition to PATCH state when you have a clear plan for the adaption.

Do not disable or skip features. If you try your best but cannot adapt the feature, give up honestly.

=== Root Cause Function ===
Name: {repair_input.root_cause_function.name}()
Location: {repair_input.root_cause_function.location_str()}
Source:
```
{repair_input.root_cause_function.source_code}
```

{_format_test_and_trace_sections(repair_input, config)}""")

        if config.provide_candidates:
            parts.append(
                "\nThe root cause function is where the discrepancy first surfaces — "
                "investigate it first. The actual patch is often in this function itself, "
                "but may occasionally need to be in a caller that provides it with "
                "incompatible data or behavior."
            )
            # Show the nearest callers inline so the LLM has
            # immediate context without an extra tool call.
            _MAX_INLINE_CALLERS = 5
            if repair_input.caller_functions:
                # Take the LAST N callers — these are the ones nearest to the
                # root cause (the stack is ordered outermost -> root cause,
                # with the root cause already deduped from caller_functions).
                shown = repair_input.caller_functions[-_MAX_INLINE_CALLERS:]
                parts.append(
                    "\n=== Nearest Caller Functions ===\n"
                    + _format_candidate_list(shown)
                )
                if len(repair_input.caller_functions) > _MAX_INLINE_CALLERS:
                    parts.append(
                        f"\n({len(repair_input.caller_functions) - _MAX_INLINE_CALLERS} more caller(s) "
                        "available — use list_candidates for the full list.)"
                    )
            parts.append(
                "\nA caller may use Wasm-incompatible operations whose effects only "
                "manifest later in the root cause function. "
                "Use list_candidates to also retrieve potentially related functions "
                "if the root cause and callers do not fully explain the discrepancy."
            )

    else:
        # --- Ablation: no trace analysis ---
        # No function names, no discrepancy details — just test location.
        # The LLM must discover everything through its tools.
        parts.append(f"""You are in ANALYZE state for project "{repair_input.project_name}", analyzing a C/C++ Wasm-vs-Native discrepancy.

A test case is failing due to a behavioral difference between WebAssembly (Emscripten) and x86 Native (GCC).
Your goal: investigate the repository, understand why this discrepancy occurs, then adapt the implementation
so the feature produces the correct functional behavior under Wasm as it does natively. Transition to PATCH state
when you have a clear plan for the adaptation.

Do not disable or skip features. If you try your best but cannot adapt the feature, give up honestly.

{_format_test_and_trace_sections(repair_input, config)}""")

    # Previous action result
    if last_action and last_action_result:
        parts.append(f"\n=== Result of Previous Action ({last_action}) ===")
        result_str = str(last_action_result)
        if len(result_str) > 20000:
            result_str = result_str[:20000] + "\n... (truncated)"
        parts.append(result_str)

    # Mandatory follow-up
    if mandatory_next_action:
        parts.append(f"\n** Your next action MUST be: {mandatory_next_action} **")
        if mandatory_next_action == "analyze_instrumentation" and "analyze_instrumentation" in config.analyze_actions:
            parts.append(
                "You MUST format your response as:\n\n"
                "    ACTION: analyze_instrumentation\n"
                "    FINDINGS: <what you learned>\n"
                "    IMPLICATION: <what this means for the next steps in analysis or patching>\n\n"
            )

    # Available actions and format
    parts.append(f"\n=== Available Actions ===\n{_format_available_actions(config.analyze_actions)}")
    analyze_format = RESPONSE_FORMAT_BASE
    if "instrument_function" in config.analyze_actions:
        analyze_format += RESPONSE_FORMAT_INSTRUMENT
    analyze_format += RESPONSE_FORMAT_LABELED_FIELDS_BASE
    parts.append(analyze_format)

    return "\n".join(parts)


def build_patch_prompt(
    repair_input: RepairInput,
    history: RepairHistory,
    last_action: Optional[str] = None,
    last_action_result: Optional[Dict] = None,
    mandatory_next_action: Optional[str] = None,
    workflow_config: Optional[WorkflowConfig] = None,
) -> str:
    """Build the prompt for the PATCH state."""

    config = workflow_config or default_config()
    parts = []

    if config.provide_trace_analysis:
        # --- Original behavior ---
        parts.append(f"""You are in PATCH state for project "{repair_input.project_name}".
Your goal: adapt the implementation so the feature produces the correct functional behavior under Wasm, passing the test case.

=== Root Cause Function (reference) ===
Name: {repair_input.root_cause_function.name}()
Location: {repair_input.root_cause_function.location_str()}
Note: the actual patch may be in this function, a caller, or related code.

=== Current Plan ===
{history.current_plan if history.current_plan else "(no plan set — consider using propose_plan first)"}""")

    else:
        # --- Ablation: no trace analysis ---
        # No function names leaked — the LLM works from its own discoveries.
        parts.append(f"""You are in PATCH state for project "{repair_input.project_name}".
Your goal: adapt the implementation so the feature produces the same functional behavior under Wasm as natively, passing the test case.

=== Current Plan ===
{history.current_plan if history.current_plan else "(no plan set — consider using propose_plan first)"}""")

    # Previous action result (critical in PATCH — this carries fix results)
    if last_action and last_action_result:
        parts.append(f"\n=== Result of Previous Action ({last_action}) ===")
        result_str = str(last_action_result)
        if len(result_str) > 20000:
            result_str = result_str[:20000] + "\n... (truncated)"
        parts.append(result_str)

    # Mandatory follow-up
    if mandatory_next_action:
        parts.append(f"\n** Your next action MUST be: {mandatory_next_action} **")
        if mandatory_next_action == "analyze_patch" and "analyze_patch" in config.patch_actions:
            parts.append(
                "Analyze the patch results above. You MUST format your response as:\n\n"
                "    ACTION: analyze_patch\n"
                "    ANALYSIS: <what the patch does, whether it modifies test code or tries to bypass the test,\n"
                "              and why it addresses the root cause rather than bypassing the test>\n"
                "    ROOT_CAUSE_ADDRESSED: Yes/No\n"
                "    SAME_FUNCTIONALITY: Yes/No\n"
                "    NEXT_STEP: <write_patch|read_file|propose_plan|transition_to_analyze|give_up>\n\n"
                "=== Critical: Evaluate Whether the Patch Is a Test Bypass ===\n"
                "Even if tests pass, set ROOT_CAUSE_ADDRESSED: No if the patch bypasses "
                "the test rather than addressing the actual Wasm-vs-Native discrepancy root cause.\n\n"
                "SAME_FUNCTIONALITY: Trace the test's execution path WITH your patch applied.\n"
                "Does the test still exercise the same functionality and code paths as before?\n"
                "If your patch causes any test cases, assertions, or branches to be SKIPPED\n"
                "(e.g., by removing a capability flag that the test checks), answer No.\n"
                "A patch that makes the test pass by not reaching the failing code is a bypass.\n\n"
                "A patch is a TEST BYPASS (ROOT_CAUSE_ADDRESSED: No) if it:\n"
                "- Wraps test assertions in conditionals that skip them when they would fail\n"
                "- Changes expected values or test inputs to match the Wasm behavior\n"
                "- Returns early or produces hardcoded results to avoid the failing test path\n"
                "- Writes or fills expected data right before an assertion to force it to pass\n"
                "- Modify the configuration to disable or skip the test case entirely for WebAssembly\n"
                "- Redefines or sets existing compile-time configuration macros to disable features\n"
                "Bottom line: Bypassing or disabling tests is **CHEATING**\n\n"
                "A patch ADDRESSES the root cause (ROOT_CAUSE_ADDRESSED: Yes, SAME_FUNCTIONALITY: Yes) if it:\n"
                "- Adapts the underlying mechanism so it produces correct behavior in Wasm\n"
                "- Ensures the test passes because the underlying discrepancy is resolved\n"
                "- The test exercises the same functionality as before, just through an adapted path\n\n"
                "If ROOT_CAUSE_ADDRESSED: No or SAME_FUNCTIONALITY: No, the patch will be reverted.\n"
                "If after multiple attempts you have tried your best but find the only way to pass the test is to skip or\n"
                "disable the tested functionality, give up honestly rather than submitting a bypass patch."
            )

    # Important constraints
    parts.append("""
=== Important Constraints ===
- Focus on understanding and patching the Wasm-vs-Native functional discrepancy in the code.
  The root cause is in how the functionality behaves differently under Emscripten.
- Your patch should make the code work correctly under WebAssembly, not modify the test.
- Do NOT modify test conditions, test inputs, expected values, or test configurations.
- Do NOT redefine or set existing compile-time configuration macros to disable features.
- Patches must be compatible with Emscripten compilation.
- Prefer minimal, targeted changes that preserve original semantics.
- Your patch must be correct and generalizable — it should not be tailored to pass
  only this specific test case, as the patch will be validated against additional tests or audits.""")

    # Available actions and format
    parts.append(f"\n=== Available Actions ===\n{_format_available_actions(config.patch_actions)}")
    patch_format = RESPONSE_FORMAT_BASE + RESPONSE_FORMAT_WRITE_FIX
    if "instrument_function" in config.patch_actions:
        patch_format += RESPONSE_FORMAT_INSTRUMENT
    patch_format += RESPONSE_FORMAT_LABELED_FIELDS_BASE + RESPONSE_FORMAT_ANALYZE_PATCH
    parts.append(patch_format)

    return "\n".join(parts)


def build_prompt(
    state: str,
    repair_input: RepairInput,
    history: RepairHistory,
    last_action: Optional[str] = None,
    last_action_result: Optional[Dict] = None,
    mandatory_next_action: Optional[str] = None,
    workflow_config: Optional[WorkflowConfig] = None,
) -> str:
    """Dispatch to the appropriate prompt builder based on current state."""
    if state == ANALYZE:
        return build_analyze_prompt(
            repair_input, history, last_action, last_action_result,
            mandatory_next_action, workflow_config,
        )
    elif state == PATCH:
        return build_patch_prompt(
            repair_input, history, last_action, last_action_result,
            mandatory_next_action, workflow_config,
        )
    else:
        raise ValueError(f"Unknown state: {state}")
