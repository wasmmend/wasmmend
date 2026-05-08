"""
Function-level repairer: main loop and state machine.

Orchestrates the ANALYZE → PATCH repair loop, calling the LLM agent,
parsing responses, dispatching actions, and managing state transitions.
"""

import os
import sys
import logging
from typing import Optional

# Add src/ to path for shared utilities
from llm.LLMAgent import GeminiAgent, create_agent
from repair.Models import (
    RepairInput, RepairResult, RepairHistory,
    ToolCallRecord,
)
from repair.States import ANALYZE, PATCH, build_prompt
from repair.Toolkit import FunctionRepairToolkit
from repair.ResponseParser import parse_response, ParseError
from repair.WorkflowConfig import (
    WorkflowConfig, default_config,
    DEFAULT_MANDATORY_FOLLOWUPS,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert agent for analyzing and adapting C/C++ source code to achieve functional equivalence \
between WebAssembly (compiled with Emscripten) and x86 Native (compiled with GCC). \
Some features that work natively may not behave identically under Wasm due to platform \
differences. Your goal is to adapt the implementation so the feature produces the same \
functional behavior under Wasm as it does natively.

You operate in two states:
- ANALYZE: Investigate the discrepancy and gather information.
- PATCH: Apply adaptations and validate them.

You take exactly ONE action per turn. Follow the response format in the prompt.

Important rules:
- Do NOT modify or bypass test assertions.
- Adapt the source code, not the tests — the tests define the expected behavior.
- Adaptations must be compatible with Emscripten compilation.
- When transitioning states or updating plans, be explicit.
"""

# After these actions, the agent MUST call the corresponding analysis action.
# Kept as module-level constant for backward compatibility; the actual source
# of truth during repair() is workflow_config.mandatory_followups.
MANDATORY_FOLLOWUP = dict(DEFAULT_MANDATORY_FOLLOWUPS)

# Default valid action sets (derived from default config).
# Kept as module-level constant for backward compatibility / tests.
_default_cfg = default_config()
VALID_ACTIONS = {
    ANALYZE: _default_cfg.valid_analyze_actions,
    PATCH: _default_cfg.valid_patch_actions,
}


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FunctionRepairer")


# ---------------------------------------------------------------------------
# Main repair function
# ---------------------------------------------------------------------------

def repair(
    repair_input: RepairInput,
    workflow_config: Optional[WorkflowConfig] = None,
    output_dir: Optional[str] = None,
) -> RepairResult:
    """Run the function-level repair loop.

    Args:
        repair_input:    Complete repair specification including root cause function,
                         candidates, type dependencies, test info, and config.
        workflow_config: Optional workflow configuration controlling available
                         actions and provided information. If None, uses defaults
                         (full tool set, root cause + candidates provided).

    Returns:
        RepairResult with status, reason, fix details, history, and token usage.
    """
    config = workflow_config or default_config()

    # Derive valid actions and mandatory followups from config
    valid_actions = {
        ANALYZE: config.valid_analyze_actions,
        PATCH: config.valid_patch_actions,
    }
    mandatory_followup_map = config.mandatory_followups

    # Initialize agent
    agent = create_agent(
        model=repair_input.model,
        temperature=0,
        max_tokens=repair_input.max_tokens,
        system_prompt=SYSTEM_PROMPT,
        output_dir=output_dir,
    )

    # Initialize history and toolkit
    history = RepairHistory()
    tk_cls = FunctionRepairToolkit
    toolkit = tk_cls(repair_input, history, workflow_config=config, output_dir=output_dir)

    # State machine
    state = ANALYZE
    last_action = None
    last_action_result = None
    mandatory_next = None
    consecutive_parse_failures = 0
    max_parse_failures = 3  # Max consecutive parse failures before giving up

    logger.info(f"Starting repair for project '{repair_input.project_name}'")
    logger.info(f"Root cause: {repair_input.root_cause_function.name}() "
                f"at {repair_input.root_cause_function.location_str()}")
    logger.info(f"Candidates: {len(repair_input.candidate_functions)}")
    logger.info(f"Max iterations: {repair_input.max_iterations}")
    logger.info(f"Config: provide_trace_analysis={config.provide_trace_analysis}, "
                f"provide_candidates={config.provide_candidates}")
    logger.info(f"ANALYZE actions: {sorted(config.analyze_actions.keys())}")
    logger.info(f"PATCH actions: {sorted(config.patch_actions.keys())}")

    for iteration in range(repair_input.max_iterations):
        logger.info(f"=== Iteration {iteration + 1}/{repair_input.max_iterations} "
                     f"| State: {state} ===")

        # 1. Build prompt
        prompt = build_prompt(
            state=state,
            repair_input=repair_input,
            history=history,
            last_action=last_action,
            last_action_result=last_action_result,
            mandatory_next_action=mandatory_next,
            workflow_config=config,
        )

        # 2. Call LLM
        response_text = agent.get_response(prompt)

        logger.info(f"Received response ({len(response_text)} chars, "
                    f"call tokens: {agent.last_token_usage.get('total_tokens', 0)})")
        _save_response_log(iteration, state, response_text, output_dir=output_dir)

        # 3. Parse response
        try:
            parsed = parse_response(response_text, state)
            consecutive_parse_failures = 0
        except ParseError as e:
            consecutive_parse_failures += 1
            logger.warning(f"Parse error ({consecutive_parse_failures}/{max_parse_failures}): {e}")

            if consecutive_parse_failures >= max_parse_failures:
                tb = _get_token_breakdown(agent, toolkit)
                return RepairResult(
                    status="give_up",
                    reason=f"Failed to parse agent responses after {max_parse_failures} consecutive attempts.",
                    final_state=state,
                    fix=None,
                    history=history,
                    iterations_used=iteration + 1,
                    total_tokens=tb["total_tokens"],
                    repair_tokens=tb["repair_tokens"],
                    instrumentation_tokens=tb["instrumentation_tokens"],
                    repair_input_tokens=tb["repair_input_tokens"],
                    repair_output_tokens=tb["repair_output_tokens"],
                    instrumentation_input_tokens=tb["instrumentation_input_tokens"],
                    instrumentation_output_tokens=tb["instrumentation_output_tokens"],
                )

            # Ask agent to reformat
            last_action = None
            last_action_result = {
                "error": (
                    "Could not parse your response. Please use the required format:\n"
                    "  ACTION: <action_name>; param1=value1; param2=value2\n"
                    "Make sure the ACTION line is on its own line."
                )
            }
            continue

        action = parsed.action.lower()
        logger.info(f"Parsed action: {action} | Params: {parsed.params}")

        # 4. Validate: mandatory follow-up
        if mandatory_next and action != mandatory_next:
            logger.warning(f"Expected mandatory action '{mandatory_next}', got '{action}'")
            # Keep last_action and last_action_result intact so the LLM
            # still sees the original data it needs to analyze (e.g.
            # instrumentation output).  Only inject the error message.
            if isinstance(last_action_result, dict):
                last_action_result["mandatory_error"] = (
                    f"Action '{action}' was NOT EXECUTED. "
                    f"Your previous action requires you to call '{mandatory_next}' next. "
                    f"Please call '{mandatory_next}' now."
                )
            else:
                last_action_result = {
                    "mandatory_error": f"Action '{action}' was NOT EXECUTED. "
                                       f"Your previous action requires you to call '{mandatory_next}' next. "
                                       f"Please call '{mandatory_next}' now."
                }
            # Don't clear mandatory_next — keep enforcing it
            continue

        # 5. Validate: action allowed in current state
        if action not in valid_actions.get(state, set()):
            logger.warning(f"Action '{action}' not valid in state '{state}'")
            last_action = action
            last_action_result = {
                "error": f"Action '{action}' was NOT EXECUTED — it is not available in {state} state. "
                         f"You are still in {state} state. "
                         f"Available actions: {sorted(valid_actions[state])}"
            }
            continue

        # 6. Execute action
        result = toolkit.dispatch(action, parsed)
        logger.info(f"Action result keys: {list(result.keys()) if isinstance(result, dict) else 'N/A'}")

        # 7. Record in history
        result_summary = _summarize_result(result)
        history.tool_calls.append(ToolCallRecord(action, parsed.params, result_summary))

        # 8. Handle state transitions and termination
        tb = _get_token_breakdown(agent, toolkit)
        termination = _check_termination(action, parsed, result, state, history, iteration, tb)
        if termination:
            logger.info(f"Terminating: {termination.status} — {termination.reason}")
            return termination

        # Handle state transitions
        new_state = _handle_state_transition(action, parsed, state)
        if new_state != state:
            logger.info(f"State transition: {state} -> {new_state}")
            state = new_state

        # 9. Set mandatory follow-up if needed.
        # Skip if the action failed — nothing to follow up on.
        # Convention: results with "error" key indicate the action did not
        # execute successfully (e.g., compile failure, code matching failure).
        action_failed = isinstance(result, dict) and "error" in result
        if action_failed:
            mandatory_next = None
        else:
            mandatory_next = mandatory_followup_map.get(action, None)
        if mandatory_next:
            logger.info(f"Mandatory next action: {mandatory_next}")

        # 10. Prepare for next iteration
        last_action = action
        last_action_result = result

    # Max iterations reached
    logger.info("Max iterations reached.")
    tb = _get_token_breakdown(agent, toolkit)
    return RepairResult(
        status="max_iterations",
        reason=f"Reached maximum iteration limit ({repair_input.max_iterations}).",
        final_state=state,
        fix=None,
        history=history,
        iterations_used=repair_input.max_iterations,
        total_tokens=tb["total_tokens"],
        repair_tokens=tb["repair_tokens"],
        instrumentation_tokens=tb["instrumentation_tokens"],
        repair_input_tokens=tb["repair_input_tokens"],
        repair_output_tokens=tb["repair_output_tokens"],
        instrumentation_input_tokens=tb["instrumentation_input_tokens"],
        instrumentation_output_tokens=tb["instrumentation_output_tokens"],
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_total_tokens(agent: GeminiAgent, toolkit: FunctionRepairToolkit) -> int:
    """Get cumulative token count across all agents (repair + instrumentation).

    Both the main repair agent and the toolkit's instrumentation agent
    track their own cumulative usage in total_token_usage.
    """
    total = agent.total_token_usage.get("total_tokens", 0)
    instr_agent = getattr(toolkit, '_instr_agent', None)
    if instr_agent is not None:
        total += instr_agent.total_token_usage.get("total_tokens", 0)
    return total


def _get_token_breakdown(agent: GeminiAgent, toolkit: FunctionRepairToolkit) -> dict:
    """Get token counts broken down by agent type and direction.

    Returns dict with repair/instrumentation totals plus input/output splits.
    Input = prompt_tokens, Output = candidates_tokens + thoughts_tokens.
    """
    ru = agent.total_token_usage
    repair_tokens = ru.get("total_tokens", 0)
    repair_input = ru.get("prompt_tokens", 0)
    repair_output = ru.get("candidates_tokens", 0) + ru.get("thoughts_tokens", 0)

    instr_tokens = 0
    instr_input = 0
    instr_output = 0
    instr_agent = getattr(toolkit, '_instr_agent', None)
    if instr_agent is not None:
        iu = instr_agent.total_token_usage
        instr_tokens = iu.get("total_tokens", 0)
        instr_input = iu.get("prompt_tokens", 0)
        instr_output = iu.get("candidates_tokens", 0) + iu.get("thoughts_tokens", 0)

    return {
        "repair_tokens": repair_tokens,
        "repair_input_tokens": repair_input,
        "repair_output_tokens": repair_output,
        "instrumentation_tokens": instr_tokens,
        "instrumentation_input_tokens": instr_input,
        "instrumentation_output_tokens": instr_output,
        "total_tokens": repair_tokens + instr_tokens,
    }


def _summarize_result(result: dict, max_len: int = 200) -> str:
    """Create a compact summary of an action result for history logging."""
    if not isinstance(result, dict):
        return str(result)[:max_len]

    if "error" in result:
        return f"ERROR: {result['error'][:max_len]}"

    # Pick the most informative fields
    summary_parts = []
    for key in ["status", "compile_success", "test_passed", "function_name", "type_name"]:
        if key in result:
            summary_parts.append(f"{key}={result[key]}")

    if "filtered_output" in result:
        summary_parts.append(f"output={result['filtered_output'][:100]}")
    elif "source_code" in result:
        summary_parts.append(f"source=({len(result['source_code'])} chars)")
    elif "wasm_output" in result and "native_output" in result:
        summary_parts.append(f"wasm_output=({len(result['wasm_output'])} chars); native_output=({len(result['native_output'])} chars)")
    elif "definition" in result:
        summary_parts.append(f"definition=({len(result['definition'])} chars)")
    elif "history" in result:
        summary_parts.append(f"history=({len(result['history'])} chars)")

    return "; ".join(summary_parts) if summary_parts else str(result)[:max_len]


def _check_termination(action, parsed, result, state, history, iteration, token_breakdown):
    """Check if the repair should terminate.

    Returns a RepairResult if terminating, None otherwise.
    """
    tb = token_breakdown

    # Give up
    if action == "give_up":
        reason = parsed.reason or parsed.reasoning or "Agent chose to give up."
        return RepairResult(
            status="give_up",
            reason=reason,
            final_state=state,
            fix=None,
            history=history,
            iterations_used=iteration + 1,
            total_tokens=tb["total_tokens"],
            repair_tokens=tb["repair_tokens"],
            instrumentation_tokens=tb["instrumentation_tokens"],
            repair_input_tokens=tb["repair_input_tokens"],
            repair_output_tokens=tb["repair_output_tokens"],
            instrumentation_input_tokens=tb["instrumentation_input_tokens"],
            instrumentation_output_tokens=tb["instrumentation_output_tokens"],
        )

    # Fix succeeded: test passed AND root cause addressed AND same functionality.
    # All three conditions required: test_passed is ground truth for correctness,
    # root_cause_addressed + same_functionality are LLM judgments that the patch
    # is not a test bypass. File restoration for rejected bypasses is handled
    # in Toolkit.analyze_patch.
    if action == "analyze_patch":
        test_passed = result.get("test_passed", False)
        root_cause_addressed = result.get("root_cause_addressed", False)
        same_functionality = result.get("same_functionality", False)
        if test_passed and root_cause_addressed and same_functionality:
            successful_fix = history.fix_attempts[-1] if history.fix_attempts else None
            reason = parsed.analysis or "Patch verified: tests pass and root cause addressed."
            return RepairResult(
                status="patched",
                reason=reason,
                final_state=state,
                fix=successful_fix,
                history=history,
                iterations_used=iteration + 1,
                total_tokens=tb["total_tokens"],
                repair_tokens=tb["repair_tokens"],
                instrumentation_tokens=tb["instrumentation_tokens"],
                repair_input_tokens=tb["repair_input_tokens"],
                repair_output_tokens=tb["repair_output_tokens"],
                instrumentation_input_tokens=tb["instrumentation_input_tokens"],
                instrumentation_output_tokens=tb["instrumentation_output_tokens"],
            )

    return None


def _handle_state_transition(action, parsed, current_state):
    """Determine the new state after an action.

    Returns the new state (may be the same as current).
    """
    if action == "transition_to_patch" and current_state == ANALYZE:
        return PATCH
    elif action == "transition_to_analyze" and current_state == PATCH:
        return ANALYZE
    elif action == "analyze_patch" and parsed.next_step:
        next_step = parsed.next_step.strip().lower()
        if next_step == "transition_to_analyze":
            return ANALYZE
    return current_state


def _save_response_log(iteration: int, state: str, response_text: str,
                       output_dir: Optional[str] = None):
    """Save agent response to a log file for debugging."""
    log_file = os.path.join(output_dir, "repair_responses.log") if output_dir else "repair_responses.log"
    try:
        with open(log_file, "a") as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"Iteration {iteration + 1} | State: {state}\n")
            f.write(f"{'='*60}\n")
            f.write(response_text)
            f.write("\n")
    except OSError:
        pass  # Don't fail on log write errors
