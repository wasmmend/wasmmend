"""
Workflow configuration for the function-level repairer.

Defines which actions are available in each state and what contextual
information is provided to the LLM. Supports loading from JSON for
ablation studies on tool availability and provided information.

Usage:
    # Default config (current behavior):
    config = default_config()

    # Load from JSON for ablation:
    config = load_config("ablation_no_instrument.json")

    # Pass to repair():
    result = repair(repair_input, workflow_config=config)
"""

import json
from dataclasses import dataclass, field
from typing import Dict


# ---------------------------------------------------------------------------
# Default action definitions
# ---------------------------------------------------------------------------

DEFAULT_ANALYZE_ACTIONS: Dict[str, str] = {
    "read_file":
        "Read arbitrary file lines by path and line range. "
        "PARAMS: file_path=<path>; start_line=<N>; end_line=<N>",
    "list_directory":
        "List directory/file structure of the project. "
        "PARAMS: path=<subdir>; max_depth=2 (default)",
    "search_in_file":
        "Search a keyword/regex in a file, returning matching lines. "
        "PARAMS: file_path=<path>; keyword=<search term>; max_matches=50 (default)",
    "read_test_source":
        "Read the failing test function source code. No parameters needed.",
    "run_test":
        "Compile and run the failing test on wasm or native without any patching. "
        "Returns filtered compile errors on compilation failure, or raw test output "
        "on successful compile. PARAMS: target=wasm|native (default: wasm)",
    "list_candidates":
        "Retrieve the execution chain (caller functions leading to the root cause) "
        "and potentially related functions. Returns function names, locations, and roles. "
        "Use this to explore beyond the root cause function when needed.",
    "instrument_function":
        "Add debug prints to a function, compile, run both native and wasm, and return "
        "the original function annotated with wasm vs native values side-by-side. "
        "The source file is automatically restored after — this is a diagnostic probe. "
        "PARAMS: function_name=<name>; file_path=<path to file>; instructions=<what to observe>. "
        "Also provide the function source code in an **Original Code:** ```c++ block — "
        "this is matched against the file to locate the function.",
    "analyze_instrumentation":
        "MANDATORY after instrument_function. Provide structured analysis of instrumentation results. "
        "Required fields: FINDINGS: <what you learned>, IMPLICATION: <what this means for the plan and patch>",
    "query_function_type_deps":
        "Get the direct type dependencies of a function (metadata only: names and locations). "
        "Use get_type_definition to read the full source of a returned type. "
        "PARAMS: function_name=<function_name>",
    "query_type_deps":
        "Get the direct type dependencies of a type (metadata only). "
        "Use get_type_definition to read the full source. "
        "PARAMS: type_name=<qualified type name>",
    "get_type_definition":
        "Get the full source definition of a type. "
        "PARAMS: type_name=<qualified type name>",
    "view_patch_history":
        "View recent patch history (tool calls, patch attempts, instrumentation results). "
        "PARAMS: last_n=3 (default)",
    "propose_plan":
        "Record your current patch plan. The plan is saved and shown in future prompts "
        "so you can track your approach across turns. "
        "Required field: PLAN: <your plan including hypothesis, what to change, and why>",
    "transition_to_patch":
        "Move to PATCH state when you have enough understanding to attempt a patch. "
        "Required field: PLAN: <your patch plan including what to change and why>",
    "give_up":
        "Terminate if you believe the issue cannot be addressed with available information. "
        "Required field: REASON: <why you cannot proceed>",
}

DEFAULT_PATCH_ACTIONS: Dict[str, str] = {
    "read_file":
        "Read arbitrary file lines by path and line range. "
        "PARAMS: file_path=<path>; start_line=<N>; end_line=<N>",
    "list_directory":
        "List directory/file structure of the project. "
        "PARAMS: path=<subdir>; max_depth=2 (default)",
    "search_in_file":
        "Search a keyword/regex in a file, returning matching lines. "
        "PARAMS: file_path=<path>; keyword=<search term>; max_matches=50 (default)",
    "read_test_source":
        "Read the failing test function source code. No parameters needed.",
    "list_candidates":
        "Retrieve the execution chain (caller functions leading to the root cause) "
        "and potentially related functions. Returns function names, locations, and roles. "
        "Use this to explore beyond the root cause function when needed.",
    "query_type_deps":
        "Get the direct type dependencies of a type (metadata only). "
        "Use get_type_definition to read the full source. "
        "PARAMS: type_name=<qualified type name>",
    "get_type_definition":
        "Get the full source definition of a type. "
        "PARAMS: type_name=<qualified type name>",
    "view_patch_history":
        "View recent patch history. "
        "PARAMS: last_n=3 (default)",
    "propose_plan":
        "Revise your patch plan. The updated plan is saved and shown in future prompts. "
        "Required field: PLAN: <revised plan>",
    "write_patch":
        "Apply a search-and-replace patch write, then automatically compile and run tests. "
        "The Original Code snippet is string-matched in the file, so be accurate. "
        "The file is automatically restored if the patch fails to compile or pass tests. "
        "PARAMS: function_name=<function_name>; file_path=<path to file>. "
        "Also provide **Original Code:** and **Patched Code:** blocks (see format below).",
    "analyze_patch":
        "MANDATORY after write_patch results are shown. Analyze the patch results. "
        "Required fields: ANALYSIS: <what happened>, ROOT_CAUSE_ADDRESSED: Yes/No, "
        "NEXT_STEP: <write_patch|propose_plan|transition_to_analyze|give_up>",
    "transition_to_analyze":
        "Go back to ANALYZE state to gather more information. "
        "Provide reasoning for why more analysis is needed.",
    "give_up":
        "Terminate if you believe the issue cannot be addressed. "
        "Required field: REASON: <why you cannot proceed>",
}

DEFAULT_MANDATORY_FOLLOWUPS: Dict[str, str] = {
    "write_patch": "analyze_patch",
    "instrument_function": "analyze_instrumentation",
}


# ---------------------------------------------------------------------------
# WorkflowConfig
# ---------------------------------------------------------------------------

@dataclass
class WorkflowConfig:
    """Configuration for the repair workflow.

    Controls which actions are available in each state and what contextual
    information is provided to the LLM. Swap configs for ablation studies.

    Attributes:
        analyze_actions:         Actions available in ANALYZE state (name -> description).
        patch_actions:           Actions available in PATCH state (name -> description).
        mandatory_followups:     Actions that require a mandatory next action
                                 (e.g. write_patch -> analyze_patch).
        provide_trace_analysis:  If True, the root cause function is highlighted with
                                 its source code shown upfront, and the trace analysis
                                 discrepancy description (input/output comparison) is
                                 included. If False, all functions are presented equally
                                 and no discrepancy details are shown — only the test
                                 case location is provided.
        provide_candidates:      If True, candidate functions are available via
                                 list_candidates. If False, candidates are hidden and
                                 list_candidates is auto-removed from action sets.
        provide_instrumentation: If True, instrument_function and analyze_instrumentation
                                 are available. If False, they are auto-removed.
        provide_type_deps:       If True, type dependency tools are available.
                                 If False, query_function_type_deps, query_type_deps,
                                 and get_type_definition are auto-removed.
    """
    analyze_actions: Dict[str, str] = field(
        default_factory=lambda: dict(DEFAULT_ANALYZE_ACTIONS))
    patch_actions: Dict[str, str] = field(
        default_factory=lambda: dict(DEFAULT_PATCH_ACTIONS))
    mandatory_followups: Dict[str, str] = field(
        default_factory=lambda: dict(DEFAULT_MANDATORY_FOLLOWUPS))
    provide_trace_analysis: bool = True
    provide_candidates: bool = True
    provide_instrumentation: bool = True
    provide_type_deps: bool = True

    def __post_init__(self):
        """Auto-adjust actions based on flags."""
        if not self.provide_candidates or not self.provide_trace_analysis:
            self.analyze_actions.pop("list_candidates", None)
            self.patch_actions.pop("list_candidates", None)

        if not self.provide_instrumentation:
            self.analyze_actions.pop("instrument_function", None)
            self.analyze_actions.pop("analyze_instrumentation", None)
            self.mandatory_followups.pop("instrument_function", None)

        if not self.provide_type_deps or not self.provide_trace_analysis:
            self.analyze_actions.pop("query_function_type_deps", None)
            self.analyze_actions.pop("query_type_deps", None)
            self.analyze_actions.pop("get_type_definition", None)
            self.patch_actions.pop("query_type_deps", None)
            self.patch_actions.pop("get_type_definition", None)

    @property
    def valid_analyze_actions(self) -> set:
        """Set of valid action names in ANALYZE state."""
        return set(self.analyze_actions.keys())

    @property
    def valid_patch_actions(self) -> set:
        """Set of valid action names in PATCH state."""
        return set(self.patch_actions.keys())


def default_config() -> WorkflowConfig:
    """Return the default workflow configuration (matches current hardcoded behavior)."""
    return WorkflowConfig()


def load_config(path: str) -> WorkflowConfig:
    """Load a workflow configuration from a JSON file.

    The JSON may contain any subset of fields; missing fields use defaults.

    Expected JSON format::

        {
            "analyze_actions": {"action_name": "description", ...},
            "patch_actions": {"action_name": "description", ...},
            "mandatory_followups": {"action": "followup_action", ...},
            "provide_trace_analysis": true,
            "provide_candidates": true
        }

    To run an ablation that removes instrumentation, create a JSON where
    ``analyze_actions`` omits ``instrument_function`` and
    ``analyze_instrumentation``.
    """
    with open(path, 'r') as f:
        data = json.load(f)

    return WorkflowConfig(
        analyze_actions=data.get("analyze_actions", dict(DEFAULT_ANALYZE_ACTIONS)),
        patch_actions=data.get("patch_actions", dict(DEFAULT_PATCH_ACTIONS)),
        mandatory_followups=data.get("mandatory_followups", dict(DEFAULT_MANDATORY_FOLLOWUPS)),
        provide_trace_analysis=data.get("provide_trace_analysis", True),
        provide_candidates=data.get("provide_candidates", True),
        provide_instrumentation=data.get("provide_instrumentation", True),
        provide_type_deps=data.get("provide_type_deps", True),
    )
