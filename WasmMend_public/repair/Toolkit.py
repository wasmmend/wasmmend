"""
Function-level repair toolkit.

Implements all 14 actions available to the repair agent.
Handles file backup/restore, compilation, test execution,
instrumentation, and type dependency queries.
"""

import json
import os
import re
import sys
import difflib
import subprocess
import time
from typing import Dict, List, Optional, Tuple
from enum import Enum, auto

from llm.LLMAgent import create_agent
from repair.Models import (
    RepairInput, RepairHistory, FunctionInfo, TypeMetadata,
    ToolCallRecord, FixAttempt,
)
from analysis.LogFilter import filter_compile_log, filter_test_log
from repair.WorkflowConfig import WorkflowConfig, default_config


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INSTRUMENT_MARKER_START = "===INSTRUMENT_START==="
INSTRUMENT_MARKER_END = "===INSTRUMENT_END==="

# Preprocessing artifacts and instrumentation backups that must be hidden
# from the LLM (in ANALYZE, PATCH, and instrumentation states). Reading them
# would leak ablation-relevant info (e.g., trace_analysis.json) or confuse
# the agent with pre-modification source copies. Blocked paths are treated
# as if they do not exist (read/search/write all return "File not found",
# list_directory skips them entirely).
BLOCKED_FILENAMES = frozenset({
    # Preprocessing analysis outputs (JSON)
    "func_instrumentation_quality.json",
    "program_states_native.json",
    "program_states_wasm.json",
    "trace_analysis_combined.json",
    "trace_analysis.json",
    "function_types.json",
    "call_graph.json",
    "func_instr_status.json",
    "ostream_plan.json",
    "reference_exit_codes.json",
    "static_instrumentation_plan.json",
    "llm_metadata.json",
    "metadata.json",
    "instrumentation_recovery.json",
    # Preprocessing run-output / temp logs (would leak instrumented runtime state)
    "execution_output_wasm.log",
    "execution_output_native.log",
    "reference_output_native.log",
    "reference_output_wasm.log",
    "stack_trace.log",
    "temp_native_output.log",
    "temp_wasm_output.log",
    # Preprocessing tracking / researcher notes
    "modified_files.txt",
    "noexcept_discrepancy_walkthrough.md",
    "ostream_plan_new.json",
    "2025-10-30-fixing-location-testsfmttestutilcc.txt",
    "2025-10-30-regarding-the-workflow-when-i-run-python3-srcprep.txt",
    # Researcher scratch test variants (untracked, reveal which test was isolated)
    "color-test_print-only.cc",
    "chrono-test.cc_backup_strftime.cc",
    "chrono-test.cc_backup_year_month_day.cc",
    "chrono-test_instru_backup.cc",
})

BLOCKED_DIRS = frozenset({
    # Source-file copies (pre/post instrumentation)
    ".instrumentation_backups",
    ".pre_instrumentation_originals",
    "instrumented_files",
    # Preprocessing scratch/state dirs
    "metadata_related",
    "test_wasm_output_results",
    "test_line_refresh_results",
    # Saved Claude Code session state
    ".claude",
})


def _is_blocked_path(abs_path: str) -> bool:
    """True if the path should be hidden from the LLM.

    Returns True when the basename matches BLOCKED_FILENAMES or when any
    ancestor directory is in BLOCKED_DIRS. Blocked paths surface as
    'File not found' to the LLM, indistinguishable from missing files.
    """
    parts = os.path.normpath(abs_path).split(os.sep)
    if parts and parts[-1] in BLOCKED_FILENAMES:
        return True
    for part in parts:
        if part in BLOCKED_DIRS:
            return True
    return False

INSTRUMENTATION_SYSTEM_PROMPT = (
    "You are an expert in C/C++ programming and debugging. "
    "Your task is to instrument C/C++ code to gather runtime information."
)


def _safe_int(value: str, default: int = 0) -> int:
    """Parse an int from a possibly malformed LLM output (e.g., '150<ctrl46>')."""
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


# ---------------------------------------------------------------------------
# Test execution helpers
# ---------------------------------------------------------------------------

class TestResult(Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    ERROR = "ERROR"
    TIMEOUT = "TIMEOUT"


class TestExecutionResult:
    def __init__(self, result, stdout, stderr, execution_time, return_code, compilation_error=""):
        self.result = result
        self.stdout = stdout
        self.stderr = stderr
        self.execution_time = execution_time
        self.return_code = return_code
        self.compilation_error = compilation_error

    def to_dict(self):
        return {
            "result": self.result.value if hasattr(self.result, "value") else str(self.result),
            "stdout": self.stdout,
            "stderr": self.stderr,
            "execution_time": self.execution_time,
            "return_code": self.return_code,
            "compilation_error": self.compilation_error,
        }


# ---------------------------------------------------------------------------
# Fuzzy matching (adapted from Toolkits.py)
# ---------------------------------------------------------------------------

def _line_gate_passes(orig_line, file_line, gate_threshold):
    if orig_line in file_line:
        return True
    return difflib.SequenceMatcher(None, orig_line, file_line).ratio() >= gate_threshold


def fuzzy_find_code_block(file_content, original_code, threshold=0.6):
    """Find the best-matching code block in file_content for original_code.

    Uses dual-anchor search with first/last line fuzzy gates.
    Returns (start_char_idx, end_char_idx, similarity_ratio) or None.
    """
    import bisect

    file_lines = file_content.splitlines(keepends=True)
    orig = original_code.strip().replace('\r\n', '\n').replace('\r', '\n')
    orig_lines = orig.splitlines(keepends=True)
    orig_len = len(orig_lines)

    if orig_len == 0:
        return None

    first_stripped = orig_lines[0].strip()
    last_stripped = orig_lines[-1].strip()
    use_first_gate = len(first_stripped) > 0
    use_last_gate = len(last_stripped) > 0
    gate_threshold = min(0.6, threshold)

    if use_last_gate:
        end_anchors = sorted(
            i for i, line in enumerate(file_lines)
            if _line_gate_passes(last_stripped, line.strip(), gate_threshold)
        )
    else:
        end_anchors = None

    max_distance = max(2, orig_len // 3)

    cum_len = [0]
    for line in file_lines:
        cum_len.append(cum_len[-1] + len(line))

    best = None
    best_ratio = threshold

    for start in range(len(file_lines)):
        if use_first_gate:
            if not _line_gate_passes(first_stripped, file_lines[start].strip(), gate_threshold):
                continue

        expected_end = start + orig_len - 1
        search_min = max(start, expected_end - max_distance)
        search_max = min(len(file_lines) - 1, expected_end + max_distance)

        if end_anchors is not None:
            lo = bisect.bisect_left(end_anchors, search_min)
            hi = bisect.bisect_right(end_anchors, search_max)
            end_lines = end_anchors[lo:hi]
        else:
            end_lines = range(search_min, search_max + 1)

        for end_line in end_lines:
            win = end_line - start + 1
            candidate = ''.join(file_lines[start:start + win])
            ratio = difflib.SequenceMatcher(None, orig, candidate.rstrip()).ratio()
            if ratio >= best_ratio:
                best_ratio = ratio
                start_idx = cum_len[start]
                best = (start_idx, start_idx + len(candidate.rstrip()), ratio)

    return best


# ---------------------------------------------------------------------------
# Code extraction helpers
# ---------------------------------------------------------------------------

def _extract_code_block(response):
    """Extract code from ```cpp / ```c++ / ``` blocks in an LLM response."""
    for marker in ("```cpp", "```c++", "```"):
        start = response.find(marker)
        if start == -1:
            continue
        start += len(marker)
        nl = response.find('\n', start)
        if nl != -1:
            start = nl + 1
        end = response.find("```", start)
        if end == -1:
            continue
        code = response[start:end].strip()
        if code:
            return code
    return None


def _extract_between_markers(text, start_marker, end_marker):
    """Extract text between ALL marker pairs and concatenate.

    If the function is called multiple times, each call produces a
    START/END pair. This extracts all of them, separated by call numbers.
    """
    parts = []
    search_from = 0
    call_num = 0

    while True:
        start = text.find(start_marker, search_from)
        if start == -1:
            break
        start += len(start_marker)
        if start < len(text) and text[start] == '\n':
            start += 1
        end = text.find(end_marker, start)
        if end == -1:
            content = text[start:].rstrip()
        else:
            content = text[start:end].rstrip()

        if content:
            call_num += 1
            parts.append(f"--- Call {call_num} ---\n{content}")

        if end == -1:
            break
        search_from = end + len(end_marker)

    return "\n".join(parts)


def _parse_instrumentation_lines(instr_output):
    """Parse instrumentation output into {line_number: [values]} dict."""
    result = {}
    for line in instr_output.splitlines():
        line = line.strip()
        m = re.match(r'\[L(\d+)\]\s*(.*)', line)
        if m:
            lineno = int(m.group(1))
            value = m.group(2)
            result.setdefault(lineno, []).append(value)
    return result


def _annotate_function(original_function_content, wasm_output, native_output):
    """Annotate function with inline wasm/native value comments."""
    wasm_data = _parse_instrumentation_lines(wasm_output)
    native_data = _parse_instrumentation_lines(native_output)
    all_lines = set(wasm_data.keys()) | set(native_data.keys())

    lines = original_function_content.splitlines()
    annotated = []
    for i, line in enumerate(lines, 1):
        if i in all_lines:
            wasm_vals = "; ".join(wasm_data.get(i, ["(no output)"]))
            native_vals = "; ".join(native_data.get(i, ["(no output)"]))
            annotated.append(f"{line}  // [wasm] {wasm_vals} | [native] {native_vals}")
        else:
            annotated.append(line)
    return "\n".join(annotated)


# ---------------------------------------------------------------------------
# FunctionRepairToolkit
# ---------------------------------------------------------------------------

class FunctionRepairToolkit:
    """Implements all actions available to the function-level repair agent.

    Args:
        repair_input:    The RepairInput containing all necessary context.
        history:         Shared RepairHistory (mutated by some actions).
        workflow_config: Optional workflow configuration. Controls which
                         functions are accessible and how list_candidates works.
    """

    def __init__(self, repair_input: RepairInput, history: RepairHistory,
                 workflow_config: Optional[WorkflowConfig] = None,
                 output_dir: Optional[str] = None):
        self.input = repair_input
        self.history = history
        self.config = workflow_config or default_config()
        self.output_dir = output_dir
        self.file_backups: Dict[str, str] = {}  # file_path -> original content

        # Build function lookup: function name -> FunctionInfo
        self._function_map: Dict[str, FunctionInfo] = {
            repair_input.root_cause_function.name: repair_input.root_cause_function,
        }
        if self.config.provide_candidates:
            for c in repair_input.candidate_functions:
                self._function_map[c.name] = c

        # LLM agent for instrumentation
        self._instr_agent = create_agent(
            model=repair_input.model,
            temperature=0,
            max_tokens=8192,
            system_prompt=INSTRUMENTATION_SYSTEM_PROMPT,
            output_dir=self.output_dir,
            log_filename="llm_calls_instrumentation.jsonl",
        )

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def dispatch(self, action: str, parsed) -> Dict:
        """Dispatch an action by name, extracting params from the ParsedResponse.

        Args:
            action:  Action name string.
            parsed:  ParsedResponse from the response parser.

        Returns:
            Dict with action results.
        """
        action_lower = action.lower()

        if action_lower == "read_file":
            file_path = parsed.params.get("file_path", "")
            start_line = _safe_int(parsed.params.get("start_line", "1"), 1)
            end_line = _safe_int(parsed.params.get("end_line", "0"), 0)
            return self.read_file(file_path, start_line, end_line)

        elif action_lower == "list_directory":
            path = parsed.params.get("path", "")
            max_depth = _safe_int(parsed.params.get("max_depth", "2"), 2)
            return self.list_directory(path, max_depth)

        elif action_lower == "search_in_file":
            file_path = parsed.params.get("file_path", "")
            keyword = parsed.params.get("keyword", "")
            max_matches = _safe_int(parsed.params.get("max_matches", "50"), 50)
            return self.search_in_file(file_path, keyword, max_matches=max_matches)

        elif action_lower == "read_test_source":
            ctx = _safe_int(parsed.params.get("context_lines", "0"), 0)
            return self.read_test_source(context_lines=ctx)

        elif action_lower == "run_test":
            target = parsed.params.get("target", "wasm")
            return self.run_test(target=target)

        elif action_lower == "list_candidates":
            return self.list_candidates()

        elif action_lower == "instrument_function":
            func_name = parsed.params.get("function_name", "unknown")
            file_path = parsed.params.get("file_path", "")
            original_func_body = parsed.original_code or ""
            instructions = parsed.params.get("instructions", "Print all local variables and return values")
            return self.instrument_function(func_name, original_func_body, file_path, instructions)

        elif action_lower == "analyze_instrumentation":
            return self.analyze_instrumentation(
                findings=parsed.findings or "",
                implication=parsed.implication or "",
            )

        elif action_lower == "query_function_type_deps":
            fid = parsed.params.get("function_name", "")
            return self.query_function_type_deps(fid)

        elif action_lower == "query_type_deps":
            type_name = parsed.params.get("type_name", "")
            return self.query_type_deps(type_name)

        elif action_lower == "get_type_definition":
            type_name = parsed.params.get("type_name", "")
            return self.get_type_definition(type_name)

        elif action_lower == "view_patch_history":
            last_n = _safe_int(parsed.params.get("last_n", "3"), 3)
            return self.view_patch_history(last_n)

        elif action_lower == "propose_plan":
            return self.propose_plan(parsed.plan or "")

        elif action_lower == "write_patch":
            fid = parsed.params.get("function_name", "unknown")
            file_path = parsed.params.get("file_path", "")
            original_code = parsed.original_code or ""
            fixed_code = parsed.fixed_code or ""
            return self.write_patch(fid, file_path, original_code, fixed_code)

        elif action_lower == "analyze_patch":
            return self.analyze_patch(
                analysis=parsed.analysis or "",
                root_cause_addressed=parsed.root_cause_addressed or False,
                same_functionality=parsed.same_functionality or False,
                next_step=parsed.next_step or "",
            )

        elif action_lower == "transition_to_patch":
            return self.transition_to_patch(parsed.plan or "")

        elif action_lower == "transition_to_analyze":
            return self.transition_to_analyze(parsed.reasoning)

        elif action_lower == "give_up":
            return self.give_up(parsed.reason or parsed.reasoning)

        else:
            return {"error": f"Unknown action: {action}"}

    # ------------------------------------------------------------------
    # Function ID resolution
    # ------------------------------------------------------------------

    def _resolve_function_name(self, function_name: str) -> FunctionInfo:
        """Resolve a function name to a FunctionInfo object.

        Raises ValueError if the function name is not recognized.
        """
        name = function_name.strip()
        if name in self._function_map:
            return self._function_map[name]
        raise ValueError(
            f"Unknown function_name: '{function_name}'. "
            f"Available: {list(self._function_map.keys())}"
        )

    def _check_function_name(self, function_name: str) -> FunctionInfo:
        """Check that function name is in scope and return the FunctionInfo."""
        return self._resolve_function_name(function_name)

    # ------------------------------------------------------------------
    # Type ID resolution
    # ------------------------------------------------------------------

    def _resolve_type_name(self, type_name: str) -> str:
        """Resolve a type name to a canonical qualified name from the type deps.

        Tries exact match first, then suffix match, then fuzzy match.
        Raises ValueError with candidates if ambiguous or not found.
        """
        type_name = type_name.strip().strip('`"\'')
        all_types = self.input.type_deps.type_metadata

        # Exact match
        if type_name in all_types:
            return type_name

        # Suffix match: "file_input" matches "tao::pegtl::file_input"
        suffix_matches = [
            qn for qn in all_types
            if qn.endswith("::" + type_name) or qn.endswith(type_name)
        ]
        if len(suffix_matches) == 1:
            return suffix_matches[0]
        if len(suffix_matches) > 1:
            raise ValueError(
                f"Ambiguous type name '{type_name}'. Matches: {suffix_matches}. "
                f"Please use a more specific qualified name."
            )

        # Fuzzy match: find closest
        best_match = None
        best_ratio = 0.6
        for qn in all_types:
            ratio = difflib.SequenceMatcher(None, type_name.lower(), qn.lower()).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_match = qn

        if best_match:
            return best_match

        available = list(all_types.keys())[:20]
        raise ValueError(
            f"Type '{type_name}' not found in dependency graph. "
            f"Available types (first 20): {available}"
        )

    # ------------------------------------------------------------------
    # File backup / restore
    # ------------------------------------------------------------------

    def _backup_file(self, file_path: str):
        """Backup a file's content before modification.

        Saves to:
        1. In-memory dict (for runtime restore during the run)
        2. output_dir/original_files/ (permanent per-run backup)
        3. results/{proj}/{test_case}/latest_restore/ (consumed by next run's auto-restore)
        """
        abs_path = os.path.abspath(file_path)
        if abs_path not in self.file_backups:
            with open(abs_path, 'r') as f:
                content = f.read()
            self.file_backups[abs_path] = content

            if self.output_dir:
                # 1. Per-run permanent backup
                run_backup_dir = os.path.join(self.output_dir, "original_files")
                self._write_backup(run_backup_dir, abs_path, content)

                # 2. Project-level latest_restore (for auto-restore before next run)
                project_dir = os.path.dirname(self.output_dir)  # results/{proj}/
                latest_dir = os.path.join(project_dir, "latest_restore")
                self._write_backup(latest_dir, abs_path, content)

    @staticmethod
    def _make_safe_filename(abs_path: str) -> str:
        """Create a unique, safe filename from an absolute path.

        Replaces path separators with underscores.
        Example: /data/project/src/util.cc -> _data_project_src_util.cc
        """
        return abs_path.replace(os.sep, "_")

    @staticmethod
    def _write_backup(backup_dir: str, abs_path: str, content: str):
        """Write a file backup and update the manifest."""
        os.makedirs(backup_dir, exist_ok=True)
        safe_name = FunctionRepairToolkit._make_safe_filename(abs_path)
        with open(os.path.join(backup_dir, safe_name), 'w') as f:
            f.write(content)
        manifest_path = os.path.join(backup_dir, "manifest.json")
        manifest = {}
        if os.path.exists(manifest_path):
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
        manifest[safe_name] = abs_path
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)

    def _restore_file(self, file_path: str):
        """Restore a file to its backed-up state."""
        abs_path = os.path.abspath(file_path)
        if abs_path in self.file_backups:
            with open(abs_path, 'w') as f:
                f.write(self.file_backups[abs_path])
            del self.file_backups[abs_path]

    def _restore_all_files(self):
        """Restore all backed-up files."""
        for abs_path in list(self.file_backups.keys()):
            self._restore_file(abs_path)

    # ------------------------------------------------------------------
    # Compile / run helpers
    # ------------------------------------------------------------------

    def _compile_project(self, wasm: bool = True) -> Tuple[bool, str]:
        """Compile the project.

        Returns:
            (success, output_log)
        """
        script = self.input.compile_script
        compile_script = script if os.path.isabs(script) else os.path.join(self.input.project_path, script)
        if not os.path.exists(compile_script):
            return False, f"Compilation script not found: {compile_script}"

        try:
            compile_script = os.path.abspath(compile_script)
            result = subprocess.run(
                ["bash", compile_script],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=self.input.project_path,
                timeout=300,
            )
            stdout = result.stdout.decode("utf-8", errors="replace")
            stderr = result.stderr.decode("utf-8", errors="replace")
            return True, stdout + stderr
        except subprocess.CalledProcessError as e:
            stdout = (e.stdout or b"").decode("utf-8", errors="replace")
            stderr = (e.stderr or b"").decode("utf-8", errors="replace")
            return False, stdout + stderr
        except subprocess.TimeoutExpired:
            return False, "Compilation timed out (300s)"
        except Exception as e:
            return False, f"Compilation error: {str(e)}"

    def _run_tests(self, wasm: bool = True) -> TestExecutionResult:
        """Run the test suite.

        Returns:
            TestExecutionResult with the outcome.
        """
        script = self.input.run_script
        run_script = script if os.path.isabs(script) else os.path.join(self.input.project_path, script)
        if not os.path.exists(run_script):
            return TestExecutionResult(
                result=TestResult.ERROR,
                stdout="", stderr=f"Run script not found: {run_script}",
                execution_time=0.0, return_code=-1,
            )

        run_script = os.path.abspath(run_script)

        try:
            start_time = time.time()
            result = subprocess.run(
                ["bash", run_script, "wasm" if wasm else "native"],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                timeout=300,
                cwd=self.input.project_path,
            )
            execution_time = time.time() - start_time
            test_result = TestResult.PASS if result.returncode == 0 else TestResult.FAIL
            return TestExecutionResult(
                result=test_result,
                stdout=result.stdout.decode("utf-8", errors="replace"),
                stderr=result.stderr.decode("utf-8", errors="replace"),
                execution_time=execution_time,
                return_code=result.returncode,
            )
        except subprocess.TimeoutExpired as e:
            return TestExecutionResult(
                result=TestResult.TIMEOUT,
                stdout=(e.stdout or b"").decode("utf-8", errors="replace"),
                stderr=(e.stderr or b"").decode("utf-8", errors="replace"),
                execution_time=time.time() - start_time,
                return_code=-1,
            )
        except Exception as e:
            return TestExecutionResult(
                result=TestResult.ERROR,
                stdout="", stderr=f"Error: {str(e)}",
                execution_time=0.0, return_code=-1,
            )

    def _compile_and_run(self, wasm: bool = True) -> Dict:
        """Compile and run tests, returning a filtered result dict.

        Returns dict with keys: compile_success, test_passed, filtered_output,
        return_code.
        """
        compile_ok, compile_log = self._compile_project(wasm=wasm)
        if not compile_ok:
            filtered = filter_compile_log(compile_log)
            return {
                "compile_success": False,
                "test_passed": False,
                "filtered_output": filtered,
                "return_code": -1,
            }

        test_exec = self._run_tests(wasm=wasm)
        test_passed = test_exec.result == TestResult.PASS
        filtered = filter_test_log(
            test_exec.stdout, test_exec.stderr,
            test_case_name=self.input.test_info.test_case_name,
        )
        return {
            "compile_success": True,
            "test_passed": test_passed,
            "filtered_output": filtered,
            "return_code": test_exec.return_code,
        }

    # ------------------------------------------------------------------
    # External actions
    # ------------------------------------------------------------------

    def read_file(self, file_path: str, start_line: int = 1,
                  end_line: int = 0) -> Dict:
        """Read arbitrary file lines by path and line range."""
        file_path = file_path.strip().strip('`"\'')
        if not file_path:
            return {"error": "file_path is required. PARAMS: file_path=<path>; start_line=<N>; end_line=<N>"}

        if not os.path.isabs(file_path):
            file_path = os.path.join(self.input.project_path, file_path)
        file_path = os.path.abspath(file_path)

        project_root = os.path.abspath(self.input.project_path)
        if not file_path.startswith(project_root):
            return {"error": f"Access denied: path must be within project directory ({project_root})."}

        if not os.path.isfile(file_path) or _is_blocked_path(file_path):
            return {"error": f"File not found: {file_path}. "
                    f"The project root is: {os.path.abspath(self.input.project_path)}. "
                    f"You can use paths relative to the project root (e.g., src/foo.cpp) "
                    f"or use list_directory to verify the correct path."}

        try:
            with open(file_path, 'r', errors='replace') as f:
                all_lines = f.readlines()
        except OSError as e:
            return {"error": f"Cannot read file: {e}"}

        total_lines = len(all_lines)
        start_line = max(1, start_line)
        if end_line <= 0 or end_line > total_lines:
            end_line = total_lines

        selected = all_lines[start_line - 1:end_line]
        numbered = [f"{i:>6}\t{line.rstrip()}" for i, line in enumerate(selected, start_line)]
        content = "\n".join(numbered)

        return {
            "file_path": file_path,
            "start_line": start_line,
            "end_line": end_line,
            "total_lines": total_lines,
            "content": content,
        }

    def list_directory(self, path: str = "", max_depth: int = 2) -> Dict:
        """List directory/file structure of the project."""
        path = path.strip().strip('`"\'')

        if not path:
            base = self.input.project_path
        elif os.path.isabs(path):
            base = path
        else:
            base = os.path.join(self.input.project_path, path)
        base = os.path.abspath(base)

        project_root = os.path.abspath(self.input.project_path)
        if not base.startswith(project_root):
            return {"error": f"Access denied: path must be within project directory ({project_root})."}

        if not os.path.isdir(base):
            return {"error": f"Directory not found: {base}. "
                    f"The project root is: {project_root}. "
                    f"Try list_directory with path=. to see the top-level structure, "
                    f"or use a path relative to the project root."}

        max_depth = max(1, max_depth)
        max_entries = 500
        lines = []
        file_count = 0
        dir_count = 0
        entry_count = 0

        skip_dirs = {'.git', '__pycache__', '.pytest_cache', 'node_modules', '.cache', '.tox'}
        skip_suffixes = ('.pyc', '.o', '.obj')

        def _walk(dirpath, prefix, depth):
            nonlocal file_count, dir_count, entry_count
            if depth > max_depth or entry_count >= max_entries:
                return
            try:
                entries = sorted(os.listdir(dirpath))
            except OSError:
                return

            def _should_skip(name, full_path):
                if name.endswith(skip_suffixes):
                    return True
                # Hide preprocessing artifacts and instrumentation backups
                # from the LLM (basename match for files, dir name match
                # for directories — same logic as _is_blocked_path).
                if name in BLOCKED_FILENAMES or name in BLOCKED_DIRS:
                    return True
                if os.path.isdir(full_path):
                    if name in skip_dirs or name == 'build' or name.startswith('build_'):
                        return True
                return False

            entries = [e for e in entries if not _should_skip(e, os.path.join(dirpath, e))]
            for i, entry in enumerate(entries):
                if entry_count >= max_entries:
                    lines.append(f"{prefix}... (truncated at {max_entries} entries)")
                    return
                full = os.path.join(dirpath, entry)
                is_last = (i == len(entries) - 1)
                connector = "└── " if is_last else "├── "
                extension = "    " if is_last else "│   "
                if os.path.isdir(full):
                    lines.append(f"{prefix}{connector}{entry}/")
                    dir_count += 1
                    entry_count += 1
                    _walk(full, prefix + extension, depth + 1)
                else:
                    try:
                        with open(full, 'r', errors='replace') as fh:
                            line_count = sum(1 for _ in fh)
                        size_str = f"{line_count} lines"
                    except OSError:
                        size_str = "?"
                    lines.append(f"{prefix}{connector}{entry}  ({size_str})")
                    file_count += 1
                    entry_count += 1

        rel = os.path.relpath(base, project_root) if base != project_root else "."
        lines.append(f"{rel}/")
        _walk(base, "", 1)

        return {
            "path": base,
            "tree": "\n".join(lines),
            "file_count": file_count,
            "dir_count": dir_count,
        }

    def search_in_file(self, file_path: str, keyword: str,
                       max_matches: int = 50) -> Dict:
        """Search for a keyword or regex pattern in a file."""
        file_path = file_path.strip().strip('`"\'')
        keyword = keyword.strip().strip('`"\'')

        if not file_path:
            return {"error": "file_path is required. PARAMS: file_path=<path>; keyword=<search term>"}
        if not keyword:
            return {"error": "keyword is required. PARAMS: file_path=<path>; keyword=<search term>"}

        if not os.path.isabs(file_path):
            file_path = os.path.join(self.input.project_path, file_path)
        file_path = os.path.abspath(file_path)

        project_root = os.path.abspath(self.input.project_path)
        if not file_path.startswith(project_root):
            return {"error": f"Access denied: path must be within project directory ({project_root})."}

        if not os.path.isfile(file_path) or _is_blocked_path(file_path):
            return {"error": f"File not found: {file_path}. "
                    f"The project root is: {os.path.abspath(self.input.project_path)}. "
                    f"You can use paths relative to the project root (e.g., src/foo.cpp) "
                    f"or use list_directory to verify the correct path."}

        try:
            with open(file_path, 'r', errors='replace') as f:
                lines = f.readlines()
        except OSError as e:
            return {"error": f"Cannot read file: {e}"}

        try:
            pattern = re.compile(keyword)
        except re.error:
            pattern = re.compile(re.escape(keyword))

        matches = []
        max_matches = max(1, max_matches)
        for line_num, line in enumerate(lines, 1):
            for m in pattern.finditer(line):
                if len(matches) >= max_matches:
                    break
                matches.append({
                    "line": line_num,
                    "column": m.start() + 1,
                    "text": line.rstrip(),
                })
            if len(matches) >= max_matches:
                break

        return {
            "file_path": file_path,
            "keyword": keyword,
            "matches": matches,
            "match_count": len(matches),
            "truncated": len(matches) >= max_matches,
        }

    def read_test_source(self, context_lines: int = 0) -> Dict:
        """Read the failing test function source code."""
        test_info = self.input.test_info
        try:
            with open(test_info.test_path, 'r') as f:
                all_lines = f.readlines()
        except OSError as e:
            return {"error": f"Cannot read test file: {e}"}

        start = max(0, test_info.test_start_line - 1 - context_lines)
        end = min(len(all_lines), test_info.test_end_line + context_lines)
        source = "".join(all_lines[start:end])

        return {
            "test_path": test_info.test_path,
            "test_case_name": test_info.test_case_name,
            "lines": f"{start + 1}-{end}",
            "source_code": source,
        }

    def run_test(self, target: str = "wasm") -> Dict:
        """Run the failing test on wasm or native without any patching.

        Compiles first to ensure the binary matches the current source state
        (important when a previous failed write_patch left a stale binary).
        Returns raw (unfiltered) test output. Test outputs are typically
        small; a generous size cap is applied only as a safety net.
        """
        target = (target or "").strip().lower()
        if target not in ("wasm", "native"):
            return {"error": "target must be 'wasm' or 'native'. PARAMS: target=wasm|native"}

        wasm = (target == "wasm")

        # Compile first to ensure the binary reflects the current source.
        compile_ok, compile_log = self._compile_project(wasm=wasm)
        if not compile_ok:
            return {
                "target": target,
                "compile_success": False,
                "test_passed": False,
                "output": filter_compile_log(compile_log),
            }

        test_exec = self._run_tests(wasm=wasm)

        # Combine stdout + stderr with clear separator
        output_parts = []
        if test_exec.stdout and test_exec.stdout.strip():
            output_parts.append(test_exec.stdout.strip())
        if test_exec.stderr and test_exec.stderr.strip():
            if output_parts:
                output_parts.append("--- stderr ---")
            output_parts.append(test_exec.stderr.strip())
        combined = "\n".join(output_parts) if output_parts else "(empty test output)"

        # Safety cap: truncate if abnormally large (50KB)
        max_chars = 50_000
        if len(combined) > max_chars:
            combined = combined[:max_chars] + f"\n... (truncated: output exceeded {max_chars} chars)"

        return {
            "target": target,
            "compile_success": True,
            "test_passed": test_exec.result == TestResult.PASS,
            "return_code": test_exec.return_code,
            "output": combined,
        }

    def list_candidates(self) -> Dict:
        """List candidate functions grouped by role."""
        callers = [
            {"function_name": c.name, "location": c.location_str(), "role": "caller"}
            for c in self.input.caller_functions
        ]
        suspicious = [
            {"function_name": c.name, "location": c.location_str(), "role": "potentially_related"}
            for c in self.input.suspicious_functions
        ]
        return {
            "caller_functions": callers,
            "potentially_related_functions": suspicious,
            "count": len(callers) + len(suspicious),
        }

    def instrument_function(self, function_name: str, original_func_body: str,
                            file_path: str, instructions: str) -> Dict:
        """Instrument a function, compile, run both native and wasm, return annotated diff.

        The source file is always restored after instrumentation.

        Args:
            function_name:     Name of the function (used as label in results).
            original_func_body: Full source code of the function to instrument.
            file_path:         Path to the file containing the function.
            instructions:      Natural language describing what to observe.
        """
        # function_name is used as a label only; file_path and original_func_body are required.
        if not original_func_body or not original_func_body.strip():
            return {"error": "original_func_body is required. "
                    "Provide the function source code to instrument."}
        if not file_path or not file_path.strip():
            return {"error": "file_path is required. "
                    "Provide the path to the file containing the function."}

        file_path = file_path.strip().strip('`"\'')

        # Resolve relative paths against project_path (consistent with read_file).
        if not os.path.isabs(file_path):
            file_path = os.path.join(self.input.project_path, file_path)
        file_path = os.path.abspath(file_path)

        # Restrict access to within the project directory.
        project_root = os.path.abspath(self.input.project_path)
        if not file_path.startswith(project_root):
            return {"error": f"Access denied: path must be within project directory ({project_root})."}

        if not os.path.isfile(file_path) or _is_blocked_path(file_path):
            return {"error": f"File not found: {file_path}. "
                    f"The project root is: {os.path.abspath(self.input.project_path)}. "
                    f"You can use paths relative to the project root (e.g., src/foo.cpp) "
                    f"or use list_directory to verify the correct path."}

        original_function_content = original_func_body

        # Backup before instrumentation
        self._backup_file(file_path)

        annotated = original_function_content  # fallback
        wasm_instr = ""
        native_instr = ""
        wasm_result = None
        native_result = None

        try:
            # Step 1: LLM generates instrumented code
            numbered_lines = "\n".join(
                f"L{i}: {line}"
                for i, line in enumerate(original_function_content.splitlines(), 1)
            )

            prompt = f"""\
Here is the function with line numbers for your reference:
---
{numbered_lines}
---

Now instrument the following function (do NOT include the L<N>: prefixes
in your output — they are only shown above for reference):
```
{original_function_content}
```

Instrumentation instructions: {instructions}

Follow these requirements **exactly**:

1. As the FIRST statement inside the function body, add:
       fprintf(stderr, "{INSTRUMENT_MARKER_START}\\n");

2. Right before every return statement (and at the end of the function
   if it has no explicit return), add:
       fprintf(stderr, "{INSTRUMENT_MARKER_END}\\n");

3. For each variable or expression you instrument, print to stderr in
   this exact format (one per line):
       fprintf(stderr, "[L<N>] <variable_name> = <format>\\n", <value>);
   where <N> is the line number from the reference above (e.g. L2, L5).

4. Only ADD fprintf statements — do NOT modify the existing logic in any way.

5. Use fprintf(stderr, ...) for ALL instrumentation output so it does not
   interfere with the program's normal stdout.

Return ONLY the instrumented function inside a ```c++ code block.
"""
            # Reset the instrumentation agent conversation for each instrumentation
            self._instr_agent.reset_conversation()
            rsp = self._instr_agent.get_response(prompt)
            instrumented_code = _extract_code_block(rsp)

            if instrumented_code:
                instrumented_code = re.sub(
                    r'^L\d+:\s?', '', instrumented_code, flags=re.MULTILINE
                )
            if not instrumented_code:
                return {
                    "error": "Failed to extract instrumented code from LLM response",
                    "llm_response_preview": rsp[:500],
                }

            # Step 2: Fuzzy-find and write back
            with open(file_path, 'r') as f:
                file_content = f.read()

            # Strip existing instrumentation comments for matching
            instr_pattern = r'  // \[Line \d+\]\[COMPARE\].*$'
            content_for_matching = re.sub(instr_pattern, '', file_content, flags=re.MULTILINE)
            original_code_clean = re.sub(instr_pattern, '', original_function_content, flags=re.MULTILINE)

            match = fuzzy_find_code_block(content_for_matching, original_code_clean, threshold=0.6)
            if match is None:
                return {
                    "error": f"Could not locate function in {file_path} for instrumentation "
                             f"(fuzzy match failed at threshold 0.6).",
                }

            start_idx, end_idx, ratio = match

            new_content = (
                content_for_matching[:start_idx]
                + instrumented_code.strip('\r\n')
                + content_for_matching[end_idx:]
            )

            # Ensure <stdio.h> is included (needed for fprintf).
            # <stdio.h> works in both C and C++ (global-namespace symbols),
            # whereas <cstdio> is C++-only and fails to compile in C files.
            if '#include <cstdio>' not in new_content and '#include <stdio.h>' not in new_content:
                new_content = '#include <stdio.h>\n' + new_content

            with open(file_path, 'w') as f:
                f.write(new_content)

            # Step 3: Compile and run both wasm and native
            compile_ok, compile_log = self._compile_project(wasm=True)
            if not compile_ok:
                filtered = filter_compile_log(compile_log)
                return {
                    "error": "Instrumented code failed to compile",
                    "compile_log": filtered,
                    "instrumented_code_preview": instrumented_code[:500],
                }

            wasm_exec = self._run_tests(wasm=True)
            native_exec = self._run_tests(wasm=False)

            # Extract instrumentation output from stderr (fallback to stdout)
            wasm_instr = _extract_between_markers(
                wasm_exec.stderr, INSTRUMENT_MARKER_START, INSTRUMENT_MARKER_END
            )
            native_instr = _extract_between_markers(
                native_exec.stderr, INSTRUMENT_MARKER_START, INSTRUMENT_MARKER_END
            )
            if not wasm_instr:
                wasm_instr = _extract_between_markers(
                    wasm_exec.stdout, INSTRUMENT_MARKER_START, INSTRUMENT_MARKER_END
                )
            if not native_instr:
                native_instr = _extract_between_markers(
                    native_exec.stdout, INSTRUMENT_MARKER_START, INSTRUMENT_MARKER_END
                )

        finally:
            # Step 4: Always restore
            self._restore_file(file_path)

        result = {
            "function_name": function_name,
            "wasm_output": wasm_instr if wasm_instr else "(no instrumentation output captured)",
            "native_output": native_instr if native_instr else "(no instrumentation output captured)",
        }

        # Store in history
        self.history.instrumentation_results.append(result)

        return result

    def query_function_type_deps(self, function_name: str) -> Dict:
        """Get the direct type dependencies of a function (metadata only)."""
        try:
            func = self._check_function_name(function_name)
        except ValueError as e:
            return {"error": str(e)}

        # Look up types in the skeleton graph that are associated with this function.
        # We check all types whose definition file matches the function's file,
        # or that appear in the skeleton graph.
        # For now, return all types in the skeleton graph (they were computed
        # specifically for the provided functions during preprocessing).
        type_deps = self.input.type_deps
        result_types = []
        for type_name, metadata in type_deps.type_metadata.items():
            # All types in the dependency info are relevant to the provided functions
            result_types.append({
                "qualified_name": metadata.qualified_name,
                "file_path": metadata.file_path,
                "location": metadata.location_str(),
            })

        return {
            "function_name": func.name,
            "direct_type_dependencies": result_types,
        }

    def query_type_deps(self, type_name: str) -> Dict:
        """Get the direct dependencies of a type (metadata only)."""
        try:
            canonical = self._resolve_type_name(type_name)
        except ValueError as e:
            return {"error": str(e)}

        deps = self.input.type_deps.skeleton_graph.get(canonical, [])
        dep_metadata = []
        for dep_name in deps:
            meta = self.input.type_deps.type_metadata.get(dep_name)
            if meta:
                dep_metadata.append({
                    "qualified_name": meta.qualified_name,
                    "file_path": meta.file_path,
                    "location": meta.location_str(),
                })
            else:
                dep_metadata.append({
                    "qualified_name": dep_name,
                    "file_path": "(unknown)",
                    "location": "(unknown)",
                })

        return {
            "type_name": canonical,
            "direct_dependencies": dep_metadata,
        }

    def get_type_definition(self, type_name: str) -> Dict:
        """Get the full source definition of a type."""
        try:
            canonical = self._resolve_type_name(type_name)
        except ValueError as e:
            return {"error": str(e)}

        definition = self.input.type_deps.full_definitions.get(canonical)
        if not definition:
            return {
                "error": f"No source definition available for type '{canonical}'.",
                "type_name": canonical,
            }

        meta = self.input.type_deps.type_metadata.get(canonical)
        return {
            "type_name": canonical,
            "location": meta.location_str() if meta else "(unknown)",
            "definition": definition,
        }

    def view_patch_history(self, last_n: int = 3) -> Dict:
        """View recent repair history."""
        return {
            "history": self.history.full_history(last_n=last_n),
        }

    def write_patch(self, function_name: str, file_path: str,
                  original_code: str, fixed_code: str) -> Dict:
        """Apply a search-and-replace patch, compile, and run tests.

        The file is automatically restored if compilation fails.
        On test failure, the file is also restored (agent re-proposes each time).

        Returns dict with compile_success, test_passed, filtered_output.
        """
        # function_name is used as a label only; file_path is required.
        if not file_path or not file_path.strip():
            return {"error": "file_path is required. "
                    "Provide the path to the file to modify."}
        file_path = file_path.strip().strip('`"\'')

        # Resolve relative paths against project_path (consistent with read_file).
        if not os.path.isabs(file_path):
            file_path = os.path.join(self.input.project_path, file_path)
        file_path = os.path.abspath(file_path)

        # Restrict access to within the project directory.
        project_root = os.path.abspath(self.input.project_path)
        if not file_path.startswith(project_root):
            return {"error": f"Access denied: path must be within project directory ({project_root})."}

        if not os.path.isfile(file_path) or _is_blocked_path(file_path):
            return {"error": f"File not found: {file_path}. "
                    f"The project root is: {os.path.abspath(self.input.project_path)}. "
                    f"You can use paths relative to the project root (e.g., src/foo.cpp) "
                    f"or use list_directory to verify the correct path."}

        if not original_code or not original_code.strip() or not fixed_code or not fixed_code.strip():
            return {"error": "Both original_code and patched_code must be provided."}

        # Backup
        self._backup_file(file_path)

        try:
            # Read file
            with open(file_path, 'r') as f:
                content = f.read()

            # Strip instrumentation comments for matching
            instr_pattern = r'  // \[Line \d+\]\[COMPARE\].*$'
            content_stripped = re.sub(instr_pattern, '', content, flags=re.MULTILINE)

            original_normalized = original_code.strip()
            fixed_normalized = fixed_code.strip()

            # Try exact match
            if original_normalized in content_stripped:
                new_content = content_stripped.replace(original_normalized, fixed_normalized, 1)
            else:
                # Try whitespace-flexible match
                ws_pattern = re.escape(original_normalized)
                ws_pattern = re.sub(r'\\ ', r'\\s+', ws_pattern)
                ws_pattern = re.sub(r'\\n', r'\\s*\\n\\s*', ws_pattern)
                matches = list(re.finditer(ws_pattern, content_stripped))
                if matches:
                    match = matches[0]
                    new_content = content_stripped[:match.start()] + fixed_normalized + content_stripped[match.end():]
                else:
                    # Try fuzzy match
                    block_match = fuzzy_find_code_block(content_stripped, original_normalized, threshold=0.6)
                    if block_match:
                        start_idx, end_idx, ratio = block_match
                        new_content = content_stripped[:start_idx] + fixed_normalized + content_stripped[end_idx:]
                    else:
                        self._restore_file(file_path)
                        return {
                            "error": f"Original code not found in {file_path} "
                                     f"(tried exact, whitespace-flexible, and fuzzy matching). "
                                     f"The file may have been restored to its original state after a "
                                     f"previous failed patch. You MUST use read_file to get the CURRENT "
                                     f"file content before retrying — do NOT reuse code from a previous "
                                     f"attempt.",
                            "compile_success": False,
                            "test_passed": False,
                            "filtered_output": "Code matching failed before compilation.",
                        }

            # Write the fix
            with open(file_path, 'w') as f:
                f.write(new_content)

            # Compile and run wasm
            result = self._compile_and_run(wasm=True)

            # Always run native tests when compilation succeeds, not
            # just when wasm passes.  This gives the LLM agent full
            # visibility into both sides so it can distinguish
            # "wasm-specific issue" (native passes) from "fundamental
            # breakage" (both fail).
            if result["compile_success"]:
                native_result = self._run_tests(wasm=False)
                native_passed = native_result.result == TestResult.PASS
                native_output = filter_test_log(
                    native_result.stdout, native_result.stderr,
                    test_case_name=self.input.test_info.test_case_name,
                )
                wasm_passed = result["test_passed"]
                wasm_output = result["filtered_output"]

                # If either side failed, show both outputs so the LLM
                # can compare wasm vs native behavior side by side.
                if not wasm_passed or not native_passed:
                    result["test_passed"] = False
                    result["filtered_output"] = (
                        f"=== WASM test: {'PASS' if wasm_passed else 'FAIL'} ===\n"
                        f"{wasm_output}\n\n"
                        f"=== Native test: {'PASS' if native_passed else 'FAIL'} ===\n"
                        f"{native_output}"
                    )
                    if not native_passed:
                        result["filtered_output"] += (
                            "\n\n[WARNING: Native test FAILED — "
                            "the patch may have broken existing functionality.]"
                        )
                    result["return_code"] = (
                        result.get("return_code", -1) if not wasm_passed
                        else native_result.return_code
                    )

            # Record the fix attempt
            fix_attempt = FixAttempt(
                description=self.history.current_plan or "(no plan)",
                file_path=file_path,
                original_code=original_code,
                fixed_code=fixed_code,
                compile_success=result["compile_success"],
                test_passed=result["test_passed"],
                error_output=result["filtered_output"] if not result["test_passed"] else "",
            )
            self.history.fix_attempts.append(fix_attempt)

            # Restore on failure so the next write_patch starts from original source.
            # On success, keep the patched file on disk.
            if not result["test_passed"]:
                self._restore_file(file_path)
                result["file_restored"] = (
                    f"The file has been restored to its ORIGINAL state. "
                    f"Your next write_patch must use Original Code from the ORIGINAL file, "
                    f"not from any previous patch. Use read_file to get the current content."
                )

            return result

        except Exception as e:
            self._restore_file(file_path)
            return {
                "error": f"Exception during patch application: {str(e)}",
                "compile_success": False,
                "test_passed": False,
                "filtered_output": str(e),
            }

    # ------------------------------------------------------------------
    # Internal actions
    # ------------------------------------------------------------------

    def analyze_instrumentation(self, findings: str, implication: str) -> Dict:
        """Record structured analysis of instrumentation results."""
        return {
            "status": "instrumentation analysis recorded",
            "findings": findings,
            "implication": implication,
        }

    def analyze_patch(self, analysis: str, root_cause_addressed: bool,
                      same_functionality: bool, next_step: str) -> Dict:
        """Record structured analysis of patch results.

        The repairer main loop checks root_cause_addressed + same_functionality
        + test_passed to determine if repair is complete.

        When test_passed=True but the patch is judged as a bypass
        (root_cause_addressed=False OR same_functionality=False),
        the patched file is restored to its original state so the next write_patch
        starts from clean source code.
        """
        # Get the most recent fix attempt's test result
        test_passed = False
        last_fix = None
        if self.history.fix_attempts:
            last_fix = self.history.fix_attempts[-1]
            test_passed = last_fix.test_passed
            # Always record the LLM's analysis text on the FixAttempt
            last_fix.analysis = analysis

        result = {
            "status": "patch analysis recorded",
            "analysis": analysis,
            "root_cause_addressed": root_cause_addressed,
            "same_functionality": same_functionality,
            "test_passed": test_passed,
            "next_step": next_step,
        }

        # Bypass rejection: tests pass but LLM judges patch is a bypass.
        # Either root cause not addressed OR test no longer exercises the same functionality.
        # Restore the file so the next write_patch starts from original source.
        is_bypass = not root_cause_addressed or not same_functionality
        if test_passed and is_bypass and last_fix:
            self._restore_file(last_fix.file_path)
            last_fix.bypass_rejected = True
            if not root_cause_addressed and not same_functionality:
                last_fix.rejection_reason = "both"
            elif not same_functionality:
                last_fix.rejection_reason = "same_functionality_no"
            else:
                last_fix.rejection_reason = "root_cause_not_addressed"
            result["bypass_rejected"] = True
            if not same_functionality:
                result["status"] = (
                    "patch REJECTED — the test no longer exercises the same functionality. "
                    "Your patch causes test cases or assertions to be skipped. File restored."
                )
            else:
                result["status"] = "patch REJECTED as test bypass — file restored"

        return result

    def propose_plan(self, plan: str) -> Dict:
        """Update the repair plan."""
        self.history.current_plan = plan
        return {
            "status": "plan updated",
            "plan": plan,
        }

    def transition_to_patch(self, plan: str) -> Dict:
        """Transition to PATCH state with a plan."""
        self.history.current_plan = plan
        return {
            "status": "transitioning to PATCH",
            "plan": plan,
        }

    def transition_to_analyze(self, reason: str) -> Dict:
        """Transition back to ANALYZE state."""
        return {
            "status": "transitioning to ANALYZE",
            "reason": reason,
        }

    def give_up(self, reason: str) -> Dict:
        """Terminate the repair process."""
        return {
            "status": "giving up",
            "reason": reason,
        }
