"""
Data models for function-level repair.

Defines the input contract (RepairInput), internal state (RepairHistory),
and output contract (RepairResult) for the FunctionRepairer.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional


# ---------------------------------------------------------------------------
# Input models
# ---------------------------------------------------------------------------

@dataclass
class FunctionInfo:
    """A single function (root cause or candidate) with its source location."""
    name: str               # e.g. "strftime", "tm::tm_year"
    file_path: str          # absolute path
    start_line: int
    end_line: int
    source_code: str        # full definition text

    def location_str(self) -> str:
        return f"{self.file_path}:{self.start_line}-{self.end_line}"


@dataclass
class TypeMetadata:
    """Lightweight metadata for a user-defined type (no source body)."""
    qualified_name: str     # e.g. "tao::pegtl::file_input"
    file_path: str
    start_line: int
    end_line: int

    def location_str(self) -> str:
        return f"{self.file_path}:{self.start_line}-{self.end_line}"


@dataclass
class TypeDepInfo:
    """Type dependency information for the functions under repair.

    skeleton_graph:   type qualified_name -> list of direct dependency qualified_names.
    full_definitions: type qualified_name -> full source definition text.
    type_metadata:    type qualified_name -> TypeMetadata.

    All three dicts share the same key space (qualified type names).
    """
    skeleton_graph: Dict[str, List[str]]
    full_definitions: Dict[str, str]
    type_metadata: Dict[str, TypeMetadata]


@dataclass
class TestInfo:
    """Information about the failing test."""
    test_path: str              # path to the test source file
    test_case_name: str         # name of the specific failing test (for log filtering)
    test_start_line: int        # failing test function start line (1-based)
    test_end_line: int          # failing test function end line (1-based)
    initial_failure_output: str # pre-run stdout+stderr from the Wasm test failure
    failing_statement: str = "" # the assertion/statement that failed (e.g. 'EXPECT_EQ(result, "42L")')


@dataclass
class RepairInput:
    """Everything the FunctionRepairer needs to start repair.

    Args:
        project_name:         Human-readable project name (e.g. "fmt").
        project_path:         Absolute path to the project root directory.
        compile_script:       Path to the compile script (absolute or relative
                              to project_path). Defaults to "compile.sh".
        run_script:           Path to the test runner script (absolute or relative
                              to project_path). Defaults to "run.sh".
        test_info:            Information about the failing test case.
        root_cause_function:  The function identified by preprocessing as the
                              root cause of the Wasm/Native discrepancy.
        caller_functions:     Functions in the execution chain leading to the
                              root cause. The fix may belong here if a caller
                              uses Wasm-incompatible operations.
        suspicious_functions: Functions whose role is uncertain — preprocessing
                              could not fully determine their relationship to
                              the discrepancy. Worth investigating if the root
                              cause and callers don't explain the issue.
        type_deps:            Type dependency information for the root cause
                              and candidate functions (from TypeParser).
        max_iterations:       Maximum number of agent iterations before
                              terminating with "max_iterations" status.
        model:                LLM model identifier for the Gemini agent.
    """
    project_name: str
    project_path: str
    test_info: TestInfo

    root_cause_function: FunctionInfo
    caller_functions: List[FunctionInfo]
    suspicious_functions: List[FunctionInfo]

    type_deps: TypeDepInfo

    @property
    def candidate_functions(self) -> List['FunctionInfo']:
        """All non-root-cause functions (callers + suspicious) for backward compat."""
        return self.caller_functions + self.suspicious_functions

    max_iterations: int = 50
    max_tokens: int = 65536
    model: str = "gemini-2.5-flash"
    compile_script: str = "compile.sh"
    run_script: str = "run.sh"


# ---------------------------------------------------------------------------
# History / internal state models
# ---------------------------------------------------------------------------

@dataclass
class ToolCallRecord:
    """Record of a single tool invocation during the repair session."""
    tool_name: str
    params: Dict
    result_summary: str


@dataclass
class FixAttempt:
    """Record of a single fix attempt."""
    description: str        # from the agent's PLAN / REASONING
    file_path: str
    original_code: str
    fixed_code: str
    compile_success: bool
    test_passed: bool
    error_output: str = ""
    # Bypass detection fields (populated by analyze_patch after the fact)
    bypass_rejected: bool = False
    rejection_reason: str = ""  # "root_cause_not_addressed" | "same_functionality_no" | "both" | ""
    analysis: str = ""  # LLM's ANALYSIS text from analyze_patch


@dataclass
class RepairHistory:
    """Structured history of the repair session."""
    current_plan: str = ""
    tool_calls: List[ToolCallRecord] = field(default_factory=list)
    fix_attempts: List[FixAttempt] = field(default_factory=list)
    instrumentation_results: List[Dict] = field(default_factory=list)

    def summary_for_prompt(self, last_n_tools: int = 5) -> str:
        """Build a concise summary for inclusion in the LLM prompt.

        Includes:
          - Current plan (always)
          - Last ``last_n_tools`` tool call summaries
          - Most recent fix attempt result (if any)
        """
        parts = []

        if self.current_plan:
            parts.append(f"Current plan: {self.current_plan}")

        if self.tool_calls:
            recent = self.tool_calls[-last_n_tools:]
            parts.append("Recent tool calls:")
            for i, tc in enumerate(recent, 1):
                parts.append(f"  {i}. {tc.tool_name}({tc.params}) -> {tc.result_summary[:200]}")

        if self.fix_attempts:
            fa = self.fix_attempts[-1]
            status = ("PASSED" if fa.test_passed
                      else "compile error" if not fa.compile_success
                      else "test failed")
            parts.append(f"Last patch attempt: [{status}] {fa.description}")
            if fa.error_output:
                parts.append(f"  Error: {fa.error_output[:300]}")

        return "\n".join(parts) if parts else "(No history yet)"

    def full_history(self, last_n: int = 3) -> str:
        """Return a more detailed history view, used by the view_patch_history tool.

        Args:
            last_n: Number of recent tool calls to include (default 3).
        """
        parts = []

        if self.current_plan:
            parts.append(f"=== Current Plan ===\n{self.current_plan}")

        if self.tool_calls:
            recent = self.tool_calls[-last_n:]
            parts.append(f"=== Tool Calls (last {len(recent)}) ===")
            for i, tc in enumerate(recent, 1):
                parts.append(f"  {i}. [{tc.tool_name}] params={tc.params}")
                parts.append(f"     result: {tc.result_summary[:500]}")

        if self.fix_attempts:
            parts.append(f"=== Patch Attempts ({len(self.fix_attempts)} total) ===")
            for i, fa in enumerate(self.fix_attempts, 1):
                status = ("PASSED" if fa.test_passed
                          else "compile error" if not fa.compile_success
                          else "test failed")
                parts.append(f"  {i}. [{status}] {fa.description}")
                parts.append(f"     file: {fa.file_path}")
                if fa.error_output:
                    parts.append(f"     error: {fa.error_output[:500]}")

        if self.instrumentation_results:
            parts.append(f"=== Instrumentation Results ({len(self.instrumentation_results)} total) ===")
            for i, ir in enumerate(self.instrumentation_results, 1):
                parts.append(f"  {i}. function: {ir.get('function_name', '?')}")
                wasm = ir.get("wasm_output", "")
                native = ir.get("native_output", "")
                if wasm:
                    parts.append(f"     wasm:\n{wasm[:400]}")
                if native:
                    parts.append(f"     native:\n{native[:400]}")

        return "\n".join(parts) if parts else "(No history yet)"


# ---------------------------------------------------------------------------
# Output models
# ---------------------------------------------------------------------------

@dataclass
class RepairResult:
    """The final output of the repair process.

    Attributes:
        status:                "patched", "give_up", or "max_iterations".
        reason:                Human-readable explanation of why repair ended.
        final_state:           Which state the repairer was in when it terminated.
        fix:                   The successful FixAttempt (only if status == "patched").
        history:               Full repair session history.
        iterations_used:       How many loop iterations were consumed.
        total_tokens:          Total tokens used across all LLM calls.
        repair_tokens:         Tokens used by the main repair agent only.
        instrumentation_tokens: Tokens used by the instrumentation agent only.
        repair_input_tokens:   Input tokens used by the repair agent.
        repair_output_tokens:  Output tokens used by the repair agent.
        instrumentation_input_tokens:  Input tokens used by the instrumentation agent.
        instrumentation_output_tokens: Output tokens used by the instrumentation agent.
    """
    status: Literal["patched", "give_up", "max_iterations"]
    reason: str
    final_state: Literal["ANALYZE", "PATCH"]
    fix: Optional[FixAttempt]
    history: RepairHistory
    iterations_used: int
    total_tokens: int
    repair_tokens: int = 0
    instrumentation_tokens: int = 0
    repair_input_tokens: int = 0
    repair_output_tokens: int = 0
    instrumentation_input_tokens: int = 0
    instrumentation_output_tokens: int = 0
