#!/usr/bin/env python3
"""
diff_trace_analysis.py - Preprocessing for Dynamic Trace Analysis Workflow

Pipeline:
  Phase 0: Build Call Graph
  Phase 1: Instrumentation (Preprocess.pre_analysze + LLMInstrumentor)
  Phase 2: Execution (Preprocess._collect_program_state_no_write)
  Phase 3: Trace Analysis (DynamicTraceAnalysis.RootCauseAnalyzer)
  Phase 4: Annotate Source Files

Usage:
    python diff_trace_analysis.py <project_path> <compile_commands_path>
"""

import os
import sys
import re
import json
import shutil
import hashlib
import threading
import queue
import tempfile
import subprocess as _subprocess
from typing import Dict, Tuple, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import pdb

# Add src/ to path for shared utilities (Preprocess, DynamicTraceAnalysis, CallGraphBuilder)
# sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from preprocess.Preprocess import Preprocess
from analysis.AST_builder import ProjectASTBuilder
from analysis.DynamicTraceAnalysis import RootCauseAnalyzer, Discrepancy
from analysis.CallGraphBuilder import CallGraphBuilder

from llm.LLMInstrumentor import LLMInstrumentor
from analysis.TypeParser import TypeParser
from llm.LLMAgent import GLOBAL_TOKEN_USAGE



class DiffTraceAnalysis:
    """
    Unified preprocessor for dynamic trace analysis workflow.
    Wraps Preprocess.py for Phase 1 & 2, adds Phase 3 (trace analysis)
    and Phase 4 (source annotation).
    """

    def __init__(self, project_path: str, compile_commands_path: str,
                 backend: str = "gemini", full_instr: bool = False,
                 fixed_time: bool = False, num_workers: int = 8):
        self.project_path = project_path
        self.compile_commands_path = compile_commands_path
        self.backend = backend
        self.full_instr = full_instr
        self.fixed_time = fixed_time
        self.num_workers = num_workers

        # Use existing Preprocess for Phase 1 & 2
        self.preprocessor = Preprocess(project_path, compile_commands_path)
        self.preprocessor.fixed_time = fixed_time

        # Output file paths
        self.call_graph_path = os.path.join(project_path, "call_graph.json")
        self.native_log = os.path.join(project_path, "execution_output_native.log")
        self.wasm_log = os.path.join(project_path, "execution_output_wasm.log")
        self.report_path = os.path.join(
            os.path.dirname(compile_commands_path),
            "instrumentation_report.json"
        )
        self.trace_analysis_path = os.path.join(project_path, "trace_analysis.json")
        self.llm_metadata_path = os.path.join(project_path, "llm_metadata.json")

        # Semaphore to limit concurrent LLM API calls across all worker
        # threads.  Without this, N workers hitting the API simultaneously
        # causes 429 RESOURCE_EXHAUSTED errors (the per-minute token quota
        # is shared across all requests).  Default: 3 concurrent calls.
        self._llm_semaphore = threading.Semaphore(3)

        # Check if instrumentation should use stderr (for projects that
        # compare stdout output, e.g. via EXPECT_WRITE).
        meta = self.preprocessor.metadata.get("Test Case Failure Info", {})
        self.output_to_stderr = (
            str(meta.get("related_to_stdout", "")).strip().lower() == "true")

        # Project language: "C" or "C++" (default). Controls whether
        # instrumentation uses std::cout/operator<< (C++) or fprintf/
        # print_<type>() helpers (C). Threaded into LLMInstrumentor and
        # gates the C++-specific code paths below. See
        # src/CPrintInstrumentor.py and src/c_instrumentation_prompts.py.
        _lang = str(meta.get("project_language", "C++")).strip().upper()
        self.is_c_project = (_lang == "C")
        if self.is_c_project:
            print(f"  Project language: C (fprintf-based instrumentation)")
        else:
            print(f"  Project language: C++ (iostream-based instrumentation)")

        # Load white_list: files/dirs to SKIP during instrumentation.
        # metadata.json can contain:
        #   "white_list": "path/to/whitelist.json"
        # The JSON file is a list of file paths or directory paths
        # (relative to project_path or absolute).  Any instrumented
        # file whose path starts with a listed directory or matches a
        # listed file will be skipped.
        self._whitelist_paths = []   # list of absolute paths
        wl_path = meta.get("white_list", "")
        if wl_path:
            if not os.path.isabs(wl_path):
                wl_path = os.path.join(project_path, wl_path)
            if os.path.exists(wl_path):
                try:
                    with open(wl_path) as f:
                        wl_entries = json.load(f)
                    for entry in wl_entries:
                        entry = entry.strip()
                        if not os.path.isabs(entry):
                            entry = os.path.join(project_path, entry)
                        self._whitelist_paths.append(
                            os.path.abspath(entry))
                    if self._whitelist_paths:
                        print(f"  White list loaded: {len(self._whitelist_paths)} "
                              f"path(s) from {wl_path}")
                except Exception as e:
                    print(f"  WARNING: Failed to load white_list "
                          f"from {wl_path}: {e}")
            else:
                print(f"  WARNING: white_list file not found: {wl_path}")

        # State
        self.states_native = None
        self.states_wasm = None
        self.root_causes = None
        self.symptoms = None
        self.root_cause_info = None        # Dict from _find_root_cause_function
        self.function_comparisons = None   # List from _compare_function_io
        self.originals_dir = None          # Pre-instrumentation originals for Step 4a

    def _is_whitelisted(self, file_path: str) -> bool:
        """Return True if *file_path* should be skipped (white-listed).

        A file is white-listed if:
        - It is inside ``build_native/`` or ``build_wasm/`` (always
          skipped — these are build artifacts / fetched dependencies,
          not project source code).
        - Its absolute path equals a listed file entry, or starts
          with a listed directory entry from the user's whitelist.
        """
        abs_path = os.path.abspath(file_path)

        # Always skip files under build directories and internal
        # backup/artifact directories.
        project_abs = os.path.abspath(self.project_path)
        for skip_dir in ('build_native', 'build_wasm', 'build',
                         '.pre_instrumentation_originals',
                         '.instrumentation_backups',
                         'instrumented_files'):
            sd = os.path.join(project_abs, skip_dir) + os.sep
            if abs_path.startswith(sd):
                return True

        if not self._whitelist_paths:
            return False
        for wl in self._whitelist_paths:
            if abs_path == wl:
                return True
            # Directory match: wl is a prefix of abs_path.
            wl_dir = wl if wl.endswith(os.sep) else wl + os.sep
            if abs_path.startswith(wl_dir):
                return True
        return False

    # ------------------------------------------------------------------
    # operator<< generation for user-defined types
    # ------------------------------------------------------------------

    def _generate_ostream_operators(self, function_types, tp, func_list):
        """
        For every user-defined type discovered by TypeParser, generate an
        ``operator<<`` overload using ``OstreamInstrumentor.generate_instrumentation_plan()``
        and insert it into the header.

        Types are processed in topological (dependency) order from the plan.
        Each type is processed one at a time: insert → compile.
        If compilation fails the LLM gets up to MAX_REPAIR_ITERATIONS repair
        attempts; if those all fail a trivial name-only fallback is tried.

        Args:
            function_types: The dict returned by
                ``TypeParser.extract_function_types`` (keyed by source file →
                function name → types).
            tp: A :class:`TypeParser` instance that has already been run.
            func_list: List of dicts with ``name``, ``file``, ``line`` keys
                for the instrumented functions.
        """
        # Route to the C backend if this is a C project. The C path
        # generates ``print_<type>()`` helpers via fprintf instead of
        # ``operator<<`` overloads. See _generate_print_helpers below.
        if self.is_c_project:
            return self._generate_print_helpers(
                function_types, tp, func_list)

        import subprocess
        from instrumentation.OstreamInstrumentor import OstreamInstrumentor

        # --- 1. Generate instrumentation plan (topological order) ---------
        plan_output_path = os.path.join(self.project_path, "ostream_plan.json")
        instrumentor = OstreamInstrumentor(tp)
        plan = instrumentor.generate_instrumentation_plan(
            func_list, plan_output_path, function_types=function_types)

        

        # Save plan for debugging
        with open(plan_output_path, 'w') as f:
            json.dump(plan, f, indent=2)
        print(f"  [ostream] Instrumentation plan saved to {plan_output_path}")

        plan_types = plan.get("types", [])
        auxiliary = plan.get("auxiliary", {})

        if not plan_types:
            print("  [ostream] No types in instrumentation plan")
            return

        # --- Print summary from plan data ---------------------------------
        files_set = set(t.get("file", "") for t in plan_types if t.get("file"))
        print(f"\n  ========== operator<< Generation Summary ==========")
        print(f"  Total types : {len(plan_types)}")
        print(f"  Total files : {len(files_set)}")
        print(f"  ───────────────────────────────────────────────────")
        for t in plan_types:
            print(f"    [{t.get('order', '?')}] {t['qualified_name']}  "
                  f"[{t.get('kind', '?')}]  "
                  f"(lines {t.get('start_line', '?')}–{t.get('end_line', '?')})  "
                  f"injection={t.get('injection_type', '?')}  "
                  f"file={os.path.basename(t.get('file', ''))}")
        print(f"  ===================================================\n")

        # --- 2. Lazy LLM agent (only created on first repair) -------------
        agent = [None]  # mutable container for nonlocal access

        def _get_agent():
            if agent[0] is None:
                from llm.LLMAgent import GeminiAgent, DeepSeekAgent
                ostream_system_prompt = (
                    "You are an expert in C/C++ programming. Your task is to write "
                    "friend operator<< overloads inside class/struct definitions so "
                    "that user-defined types can be printed with std::cout. The "
                    "function must always be declared as 'friend' inside the type. "
                    "Return ONLY the requested C++ code with NO explanation, NO "
                    "markdown formatting, NO additional text."
                )
                if self.backend == "gemini":
                    agent[0] = GeminiAgent(
                        model="gemini-3-flash-preview",
                        temperature=0,
                        max_tokens=65536,
                        system_prompt=ostream_system_prompt,
                    )
                else:
                    agent[0] = DeepSeekAgent(
                        model="deepseek-reasoner",
                        temperature=0,
                        max_tokens=8192,
                        system_prompt=ostream_system_prompt,
                    )
            return agent[0]

        # --- 3. Process each type one-by-one: insert, compile, repair -----
        # Set up prompt logging directory
        prompt_log_dir = os.path.join(
            self.project_path, ".instrumentation_backups", "prompt_logs")
        os.makedirs(prompt_log_dir, exist_ok=True)
        log_counter = [0]  # mutable counter for closures

        def _log_prompt_begin(tag, prompt_text):
            """Write prompt to disk before the API call. Returns log path."""
            log_counter[0] += 1
            log_path = os.path.join(
                prompt_log_dir,
                f"ostream_{log_counter[0]:03d}_{tag}.txt")
            with open(log_path, 'w') as lf:
                lf.write(f"=== PROMPT ===\n{prompt_text}\n\n"
                         f"=== RESPONSE ===\n[AWAITING RESPONSE...]\n")
            return log_path

        def _log_prompt_finish(log_path, response_text):
            """Append the LLM response to an existing log file."""
            if not log_path or not os.path.exists(log_path):
                return
            with open(log_path, 'a') as lf:
                lf.write(f"\n{response_text}\n")

        project_abs = os.path.abspath(self.project_path)
        compile_script = os.path.join(project_abs, "compile.sh")
        has_compile = os.path.exists(compile_script)

        if not has_compile:
            raise FileNotFoundError(
                f"compile.sh not found at {compile_script} — "
                f"cannot verify ostream operator<< compilation")

        # --- Add #include <iostream> to files listed in the plan ----------
        iostream_added = set()
        for file_path in auxiliary.get("files_needing_include", []):
            if not os.path.exists(file_path) or file_path in iostream_added:
                continue
            if self._is_whitelisted(file_path):
                print(f"  [ostream] Skipping #include <iostream> for "
                      f"{os.path.basename(file_path)} (white-listed)")
                continue
            with open(file_path, 'r') as f:
                content = f.read()
            if '#include <iostream>' not in content:
                content = '#include <iostream>\n' + content
                with open(file_path, 'w') as f:
                    f.write(content)
                iostream_added.add(file_path)
                print(f"  [ostream] Added #include <iostream> to "
                      f"{os.path.basename(file_path)}")

        # Build mutable list from plan types in topological order.
        # Entry format: [file_path, qname, type_entry_dict, end_line]
        all_types = []
        for t in plan_types:
            fpath = t.get("file")
            el = t.get("end_line")
            if fpath and el is not None:
                all_types.append([fpath, t["qualified_name"], t, el])

        # Drop types whose file is white-listed. The iostream-include
        # step above already skips these files, so injecting
        # `operator<<` here would land in a file without <iostream>
        # declared and break compilation. Types can reach plan_types
        # via transitive dependency from non-whitelisted types (e.g.
        # a user class holding a rapidjson::GenericValue member), so
        # we re-check the whitelist at the instrumentation step.
        _wl_types_dropped = 0
        _filtered_all_types = []
        for entry in all_types:
            if self._is_whitelisted(entry[0]):
                _wl_types_dropped += 1
            else:
                _filtered_all_types.append(entry)
        if _wl_types_dropped:
            print(f"  [ostream] Skipping {_wl_types_dropped} type(s) "
                  f"in white-listed file(s)")
        all_types = _filtered_all_types

        # Shift end_lines +1 for files that got #include <iostream> at top
        for entry in all_types:
            if entry[0] in iostream_added:
                entry[3] += 1

        # Static try_print helpers inserted inside class body for types
        # whose generated code uses _instrumentation::try_print.
        TRY_PRINT_HELPERS = (
            "    template <typename U>\n"
            "    static auto try_print(std::ostream& os, const char* label, const U& val, int)\n"
            "        -> decltype(os << val, void()) {\n"
            "        os << label << val;\n"
            "    }\n"
            "    template <typename U>\n"
            "    static void try_print(std::ostream& os, const char* label, const U&, ...) {\n"
            "        os << label << \"(non-printable)\";\n"
            "    }"
        )

        # NOTE: Do NOT re-sort all_types — process in plan order (topological).
        from tqdm import tqdm
        ostream_stats = {'ok': 0, 'repaired': 0, 'fallback': 0, 'skipped': 0}
        ostream_pbar = tqdm(all_types, desc="  [ostream]", unit="type",
                            leave=True)
        for i, entry in enumerate(ostream_pbar):
            header_path, qname = entry[0], entry[1]
            type_entry = entry[2]  # plan dict for this type
            end_line = entry[3]    # current (possibly updated) end_line
            short_name = qname.rsplit('::', 1)[-1] if '::' in qname else qname

            if not os.path.exists(header_path):
                ostream_stats['skipped'] += 1
                continue

            if self._is_whitelisted(header_path):
                ostream_stats['skipped'] += 1
                ostream_pbar.set_postfix_str(
                    f"{short_name}: skip (white-listed)")
                continue

            # --- Get code from plan and handle try_print ------------------
            code = type_entry["generated_code"]
            injection_type = type_entry.get("injection_type", "friend_operator")
            type_body = type_entry.get("original_body", "")

            # For types using _instrumentation::try_print, replace with
            # static member try_print and prepend helper functions.
            if '_instrumentation::try_print' in code:
                code = code.replace('_instrumentation::try_print', 'try_print')
                code = TRY_PRINT_HELPERS + "\n" + code

            marker = f"// [ostream] Auto-generated operator<< for {qname}"

            # --- Insert operator<< into the file -------------------------
            with open(header_path, 'r') as f:
                lines = f.readlines()

            if injection_type == "friend_operator":
                closing_brace_idx = None
                for line_idx in range(min(end_line - 1, len(lines) - 1),
                                      max(end_line - 10, -1), -1):
                    if '}' in lines[line_idx]:
                        closing_brace_idx = line_idx
                        break
                if closing_brace_idx is None:
                    closing_brace_idx = min(end_line - 1, len(lines) - 1)

                brace_line = lines[closing_brace_idx]
                brace_pos = brace_line.rfind('}')
                before_brace = brace_line[:brace_pos]
                after_brace = brace_line[brace_pos:]

                new_content = before_brace + f"\n{marker}\n{code}\n" + after_brace

                old_line_count = brace_line.count('\n')
                new_line_count = new_content.count('\n')
                num_new_lines = new_line_count - old_line_count

                lines[closing_brace_idx] = new_content
                with open(header_path, 'w') as f:
                    f.writelines(lines)

                tracked_block = f"{marker}\n{code}"

                for other in all_types:
                    if other[0] == header_path \
                            and other[3] > closing_brace_idx:
                        other[3] += num_new_lines
                end_line = entry[3]

                ostream_pbar.set_postfix_str(f"{short_name}: inserted")

            elif injection_type == "free_operator":
                ostream_stats['skipped'] += 1
                ostream_pbar.set_postfix_str(f"{short_name}: skip (enum)")
                continue
            else:
                ostream_stats['skipped'] += 1
                ostream_pbar.set_postfix_str(f"{short_name}: skip (unknown)")
                continue

            # --- Compile --------------------------------------------------
            MAX_REPAIR_ITERATIONS = 2

            success, stderr, stdout = self._compile_project(
                compile_script, project_abs)
            if success:
                ostream_stats['ok'] += 1
                ostream_pbar.set_postfix_str(f"{short_name}: OK")
                continue

            # --- Repair loop ----------------------------------------------
            for repair_iter in range(1, MAX_REPAIR_ITERATIONS + 1):
                ostream_pbar.set_postfix_str(
                    f"{short_name}: LLM_repair_{repair_iter}")

                error_snippet = self._filter_errors_for_file(stderr, header_path)

                with open(header_path, 'r') as f:
                    file_content = f.read()

                block_pos = file_content.find(tracked_block)
                if block_pos == -1:
                    tqdm.write(f"  [ostream] ERROR: lost tracked block for {short_name}")
                    exit(1)
                    break

                old_code = tracked_block[len(marker) + 1:]

                repair_prompt = (
                    f"Fix the compilation errors in this friend operator<< function "
                    f"for type `{qname.split(':')[-1]}` in {os.path.basename(header_path)}.\n\n"
                    f"COMPILATION ERRORS (relevant to this file):\n"
                    f"```\n{error_snippet}\n```\n\n"
                    f"TYPE DEFINITION:\n```cpp\n{type_body}\n```\n\n"
                    f"YOUR PREVIOUS ATTEMPT (has compilation error):\n"
                    f"```cpp\n{old_code}\n```\n\n"
                    f"IMPORTANT NOTES:\n"
                    f"- The function MUST be declared as a `friend` function "
                    f"inside the class/struct definition.\n"
                    f"- Feel free to access all private members of the type.\n"
                    f"- If any variable you need to print is a template, you can use the _instrumentation::try_print function. For example:\n"
                    f"`_instrumentation::try_print(os, \" val=\", obj.val, 0);`"
                    f"where os is the output stream.\n"
                    f"- If a member is not printable with <<, print "
                    f"\"[unprintable]\" for it.\n\n"
                    f"CRITICAL: The code MUST compile. Read the error messages "
                    f"carefully and fix exactly what they say.\n\n"
                    f"Return ONLY the corrected friend operator<< function definition "
                    f"with NO explanation, NO markdown formatting, NO additional "
                    f"text."
                )

                repair_tag = f"repair{repair_iter}_{qname.replace('::', '_')}"
                repair_lp = _log_prompt_begin(repair_tag, repair_prompt)
                try:
                    response = _get_agent().get_response(repair_prompt)
                    _log_prompt_finish(repair_lp, response)
                except Exception as e:
                    _log_prompt_finish(repair_lp, f"[ERROR: {e}]")
                    tqdm.write(f"  [ostream] LLM error for {short_name}: {e}")
                    break

                fixed_code = self._parse_llm_code(response)
                if not fixed_code:
                    tqdm.write(f"  [ostream] Empty response for {short_name}")
                    break

                new_tracked = f"{marker}\n{fixed_code}"
                old_line_count = tracked_block.count('\n')
                new_line_count = new_tracked.count('\n')
                line_delta = new_line_count - old_line_count

                file_content = file_content.replace(tracked_block, new_tracked, 1)
                with open(header_path, 'w') as f:
                    f.write(file_content)
                tracked_block = new_tracked

                if line_delta != 0:
                    block_line = file_content[:file_content.find(new_tracked)].count('\n') + 1
                    for other in all_types:
                        if other[0] == header_path \
                                and other[3] > block_line:
                            other[3] += line_delta
                    end_line = entry[3]

                success, stderr, stdout = self._compile_project(
                    compile_script, project_abs)
                if success:
                    ostream_stats['repaired'] += 1
                    ostream_pbar.set_postfix_str(
                        f"{short_name}: repaired (attempt {repair_iter})")
                    break
            else:
                ostream_pbar.set_postfix_str(f"{short_name}: fallback")

                fallback_prompt = (
                    f"The previous operator<< for type `{qname}` failed to "
                    f"compile after multiple attempts. Write a trivial "
                    f"friend operator<< that simply prints the type name "
                    f"\"{qname}\" as a string literal.\n\n"
                    f"TYPE DEFINITION:\n```cpp\n{type_body}\n```\n\n"
                    f"The function MUST be declared as `friend` inside the "
                    f"class/struct definition. For example:\n"
                    f"```cpp\nfriend std::ostream& operator<<(std::ostream& os, "
                    f"const TypeName& obj) {{\n"
                    f"    os << \"{qname}{{}}\";\n"
                    f"    return os;\n}}\n```\n\n"
                    f"Return ONLY the friend operator<< function definition "
                    f"with NO explanation, NO markdown formatting, NO "
                    f"additional text."
                )

                fallback_tag = f"fallback_{qname.replace('::', '_')}"
                fallback_lp = _log_prompt_begin(fallback_tag, fallback_prompt)
                try:
                    response = _get_agent().get_response(fallback_prompt)
                    _log_prompt_finish(fallback_lp, response)
                except Exception as e:
                    _log_prompt_finish(fallback_lp, f"[ERROR: {e}]")
                    tqdm.write(f"  [ostream] Fallback LLM error for {short_name}: {e}")
                    # Hard rollback: remove the tracked block.
                    self._rollback_ostream_block(
                        header_path, tracked_block, marker,
                        all_types)
                    ostream_stats['skipped'] += 1
                    ostream_pbar.set_postfix_str(
                        f"{short_name}: rollback")
                    continue

                fallback_code = self._parse_llm_code(response)
                if not fallback_code:
                    self._rollback_ostream_block(
                        header_path, tracked_block, marker,
                        all_types)
                    ostream_stats['skipped'] += 1
                    continue

                with open(header_path, 'r') as f:
                    file_content = f.read()

                if file_content.find(tracked_block) == -1:
                    tqdm.write(f"  [ostream] ERROR: lost tracked block for {short_name}")
                    ostream_stats['skipped'] += 1
                    continue

                new_tracked = f"{marker}\n{fallback_code}"
                old_line_count = tracked_block.count('\n')
                new_line_count = new_tracked.count('\n')
                line_delta = new_line_count - old_line_count

                file_content = file_content.replace(tracked_block, new_tracked, 1)
                with open(header_path, 'w') as f:
                    f.write(file_content)
                tracked_block = new_tracked

                if line_delta != 0:
                    block_line = file_content[:file_content.find(new_tracked)].count('\n') + 1
                    for other in all_types:
                        if other[0] == header_path \
                                and other[3] > block_line:
                            other[3] += line_delta
                    end_line = entry[3]

                success, stderr, stdout = self._compile_project(
                    compile_script, project_abs)
                if success:
                    ostream_stats['fallback'] += 1
                    ostream_pbar.set_postfix_str(
                        f"{short_name}: fallback OK")
                else:
                    ostream_stats['skipped'] += 1
                    ostream_pbar.set_postfix_str(
                        f"{short_name}: FAILED — rolling back")
                    tqdm.write(
                        f"  [ostream] {short_name} failed all attempts"
                        f" — rolling back")
                    # pdb.set_trace()
                    self._rollback_ostream_block(
                        header_path, tracked_block, marker,
                        all_types)

        ostream_pbar.close()
        print(f"  [ostream] Summary: {ostream_stats['ok']} OK, "
              f"{ostream_stats['repaired']} repaired, "
              f"{ostream_stats['fallback']} fallback, "
              f"{ostream_stats['skipped']} skipped")

        # # --- Done — rebuild call graph ------------------------------------
        self._rebuild_call_graph()

    # ------------------------------------------------------------------
    # C backend: print_<type>() helper generation
    # ------------------------------------------------------------------

    def _generate_print_helpers(self, function_types, tp, func_list):
        """C equivalent of :meth:`_generate_ostream_operators`.

        Generates ``static inline void print_<kind>_<name>(FILE*, ...)``
        helper functions for every user-defined struct/union/enum
        discovered by TypeParser, and injects them **after** the type
        definition (as free functions, because C has no ``friend``).

        Pipeline mirrors the C++ path: plan -> per-type insert -> compile
        -> LLM repair (up to 2 iterations) -> trivial fallback.
        """
        from instrumentation.CPrintInstrumentor import CPrintInstrumentor
        from instrumentation.c_instrumentation_prompts import (
            C_PRINT_HELPER_SYSTEM_PROMPT,
            build_c_print_helper_repair_prompt,
            build_c_print_helper_fallback_prompt,
        )

        # --- 1. Generate plan (topological order) ------------------------
        plan_output_path = os.path.join(
            self.project_path, "ostream_plan.json")
        instrumentor = CPrintInstrumentor(tp)
        plan = instrumentor.generate_instrumentation_plan(
            func_list, plan_output_path, function_types=function_types)

        with open(plan_output_path, 'w') as f:
            json.dump(plan, f, indent=2)
        print(f"  [cprint] Instrumentation plan saved to "
              f"{plan_output_path}")

        plan_types = plan.get("types", [])
        auxiliary = plan.get("auxiliary", {})

        if not plan_types:
            print("  [cprint] No types in instrumentation plan")
            return

        # --- 2. Summary --------------------------------------------------
        files_set = set(
            t.get("file", "") for t in plan_types if t.get("file"))
        print(f"\n  ========== print_T Generation Summary ==========")
        print(f"  Total types : {len(plan_types)}")
        print(f"  Total files : {len(files_set)}")
        print(f"  ───────────────────────────────────────────────────")
        for t in plan_types:
            print(f"    [{t.get('order', '?')}] {t['qualified_name']}  "
                  f"[{t.get('kind', '?')}]  "
                  f"(lines {t.get('start_line', '?')}–"
                  f"{t.get('end_line', '?')})  "
                  f"helper={t.get('helper_name', '?')}  "
                  f"file={os.path.basename(t.get('file', ''))}")
        print(f"  ===================================================\n")

        # --- 3. Lazy LLM agent -------------------------------------------
        agent = [None]

        def _get_agent():
            if agent[0] is None:
                from llm.LLMAgent import GeminiAgent, DeepSeekAgent
                if self.backend == "gemini":
                    agent[0] = GeminiAgent(
                        model="gemini-3-flash-preview",
                        temperature=0,
                        max_tokens=65536,
                        system_prompt=C_PRINT_HELPER_SYSTEM_PROMPT,
                    )
                else:
                    agent[0] = DeepSeekAgent(
                        model="deepseek-reasoner",
                        temperature=0,
                        max_tokens=8192,
                        system_prompt=C_PRINT_HELPER_SYSTEM_PROMPT,
                    )
            return agent[0]

        # --- 4. Prompt logging -------------------------------------------
        prompt_log_dir = os.path.join(
            self.project_path, ".instrumentation_backups", "prompt_logs")
        os.makedirs(prompt_log_dir, exist_ok=True)
        log_counter = [0]

        def _log_prompt_begin(tag, prompt_text):
            log_counter[0] += 1
            log_path = os.path.join(
                prompt_log_dir,
                f"cprint_{log_counter[0]:03d}_{tag}.txt")
            with open(log_path, 'w') as lf:
                lf.write(f"=== PROMPT ===\n{prompt_text}\n\n"
                         f"=== RESPONSE ===\n[AWAITING RESPONSE...]\n")
            return log_path

        def _log_prompt_finish(log_path, response_text):
            if not log_path or not os.path.exists(log_path):
                return
            with open(log_path, 'a') as lf:
                lf.write(f"\n{response_text}\n")

        project_abs = os.path.abspath(self.project_path)
        compile_script = os.path.join(project_abs, "compile.sh")
        if not os.path.exists(compile_script):
            raise FileNotFoundError(
                f"compile.sh not found at {compile_script} — "
                f"cannot verify print helper compilation")

        # --- 5. Add #include <stdio.h> to files listed in the plan ------
        stdio_added = set()
        for file_path in auxiliary.get("files_needing_include", []):
            if not os.path.exists(file_path) or file_path in stdio_added:
                continue
            if self._is_whitelisted(file_path):
                print(f"  [cprint] Skipping #include <stdio.h> for "
                      f"{os.path.basename(file_path)} (white-listed)")
                continue
            with open(file_path, 'r') as f:
                content = f.read()
            if '#include <stdio.h>' not in content:
                content = '#include <stdio.h>\n' + content
                with open(file_path, 'w') as f:
                    f.write(content)
                stdio_added.add(file_path)
                print(f"  [cprint] Added #include <stdio.h> to "
                      f"{os.path.basename(file_path)}")

        # --- 6. Build mutable type list, track end_line deltas ----------
        # Entry format: [file_path, qname, type_entry_dict, end_line]
        all_types = []
        for t in plan_types:
            fpath = t.get("file")
            el = t.get("end_line")
            if fpath and el is not None:
                all_types.append([fpath, t["qualified_name"], t, el])

        # Drop types whose file is white-listed. Same reasoning as the
        # C++ path: transitive dependencies can pull whitelisted types
        # into the plan, but we never want to instrument framework or
        # vendored code.
        _wl_types_dropped = 0
        _filtered_all_types = []
        for entry in all_types:
            if self._is_whitelisted(entry[0]):
                _wl_types_dropped += 1
            else:
                _filtered_all_types.append(entry)
        if _wl_types_dropped:
            print(f"  [cprint] Skipping {_wl_types_dropped} type(s) "
                  f"in white-listed file(s)")
        all_types = _filtered_all_types

        # Shift end_lines +1 for files that got #include <stdio.h> at top.
        for entry in all_types:
            if entry[0] in stdio_added:
                entry[3] += 1

        # --- 7. Per-type insert + compile + repair loop -----------------
        from tqdm import tqdm
        cprint_stats = {'ok': 0, 'repaired': 0,
                        'fallback': 0, 'skipped': 0}
        cprint_pbar = tqdm(all_types, desc="  [cprint]", unit="type",
                           leave=True)

        for entry in cprint_pbar:
            header_path, qname = entry[0], entry[1]
            type_entry = entry[2]
            end_line = entry[3]
            short_name = type_entry.get(
                'helper_name', qname.rsplit('::', 1)[-1])

            if not os.path.exists(header_path):
                cprint_stats['skipped'] += 1
                continue

            if self._is_whitelisted(header_path):
                cprint_stats['skipped'] += 1
                cprint_pbar.set_postfix_str(
                    f"{short_name}: skip (white-listed)")
                continue

            code = type_entry["generated_code"]
            type_body = type_entry.get("original_body", "")
            kind = type_entry.get("kind", "struct")

            marker = (f"/* [cprint] Auto-generated "
                      f"{type_entry.get('helper_name', '')} for {qname} */")

            # --- Insert after the type's closing ``};`` -----------------
            with open(header_path, 'r') as f:
                lines = f.readlines()

            # Find the closing brace of the type definition. Scan from
            # end_line backwards to find a ``};`` (or ``}``). end_line
            # from TypeParser is 1-based and usually points at ``};``.
            insert_after_idx = None  # 0-based line index to insert AFTER
            for line_idx in range(min(end_line - 1, len(lines) - 1),
                                  max(end_line - 10, -1), -1):
                if '};' in lines[line_idx] or '}' in lines[line_idx]:
                    insert_after_idx = line_idx
                    break
            if insert_after_idx is None:
                insert_after_idx = min(end_line - 1, len(lines) - 1)

            # Splice: insert a newline + marker + code + newline after
            # the line containing ``};``.
            insertion = f"\n{marker}\n{code}\n"
            brace_line = lines[insert_after_idx]
            if not brace_line.endswith('\n'):
                brace_line += '\n'
                lines[insert_after_idx] = brace_line
            lines[insert_after_idx] = brace_line + insertion

            with open(header_path, 'w') as f:
                f.writelines(lines)

            tracked_block = f"{marker}\n{code}"
            num_new_lines = insertion.count('\n')

            # Shift end_line for later types in the same file that sit
            # below the insertion point.
            insert_line_1based = insert_after_idx + 1
            for other in all_types:
                if (other[0] == header_path
                        and other[3] > insert_line_1based):
                    other[3] += num_new_lines

            cprint_pbar.set_postfix_str(f"{short_name}: inserted")

            # --- Compile ---------------------------------------------
            MAX_REPAIR_ITERATIONS = 2
            success, stderr, stdout = self._compile_project(
                compile_script, project_abs)
            if success:
                cprint_stats['ok'] += 1
                cprint_pbar.set_postfix_str(f"{short_name}: OK")
                continue

            # --- Repair loop -----------------------------------------
            repaired = False
            for repair_iter in range(1, MAX_REPAIR_ITERATIONS + 1):
                cprint_pbar.set_postfix_str(
                    f"{short_name}: LLM_repair_{repair_iter}")

                error_snippet = self._filter_errors_for_file(
                    stderr, header_path)

                with open(header_path, 'r') as f:
                    file_content = f.read()

                if tracked_block not in file_content:
                    tqdm.write(
                        f"  [cprint] ERROR: lost tracked block for "
                        f"{short_name}")
                    break

                old_code = tracked_block[len(marker) + 1:]

                repair_prompt = build_c_print_helper_repair_prompt(
                    qname=qname,
                    file_basename=os.path.basename(header_path),
                    error_snippet=error_snippet,
                    type_body=type_body,
                    old_code=old_code,
                )

                repair_tag = \
                    f"repair{repair_iter}_{qname.replace('::', '_')}"
                repair_lp = _log_prompt_begin(repair_tag, repair_prompt)
                try:
                    response = _get_agent().get_response(repair_prompt)
                    _log_prompt_finish(repair_lp, response)
                except Exception as e:
                    _log_prompt_finish(repair_lp, f"[ERROR: {e}]")
                    tqdm.write(
                        f"  [cprint] LLM error for {short_name}: {e}")
                    break

                fixed_code = self._parse_llm_code(response)
                if not fixed_code:
                    tqdm.write(
                        f"  [cprint] Empty response for {short_name}")
                    break

                new_tracked = f"{marker}\n{fixed_code}"
                old_line_count = tracked_block.count('\n')
                new_line_count = new_tracked.count('\n')
                line_delta = new_line_count - old_line_count

                file_content = file_content.replace(
                    tracked_block, new_tracked, 1)
                with open(header_path, 'w') as f:
                    f.write(file_content)
                tracked_block = new_tracked

                if line_delta != 0:
                    block_line = (
                        file_content[:file_content.find(new_tracked)]
                        .count('\n') + 1
                    )
                    for other in all_types:
                        if (other[0] == header_path
                                and other[3] > block_line):
                            other[3] += line_delta

                success, stderr, stdout = self._compile_project(
                    compile_script, project_abs)
                if success:
                    cprint_stats['repaired'] += 1
                    cprint_pbar.set_postfix_str(
                        f"{short_name}: repaired "
                        f"(attempt {repair_iter})")
                    repaired = True
                    break

            if repaired:
                continue

            # --- Trivial fallback ------------------------------------
            cprint_pbar.set_postfix_str(f"{short_name}: fallback")

            fallback_prompt = build_c_print_helper_fallback_prompt(
                qname=qname, type_body=type_body, kind=kind)

            fallback_tag = f"fallback_{qname.replace('::', '_')}"
            fallback_lp = _log_prompt_begin(fallback_tag, fallback_prompt)
            try:
                response = _get_agent().get_response(fallback_prompt)
                _log_prompt_finish(fallback_lp, response)
            except Exception as e:
                _log_prompt_finish(fallback_lp, f"[ERROR: {e}]")
                tqdm.write(
                    f"  [cprint] Fallback LLM error for "
                    f"{short_name}: {e}")
                self._rollback_ostream_block(
                    header_path, tracked_block, marker, all_types)
                cprint_stats['skipped'] += 1
                cprint_pbar.set_postfix_str(f"{short_name}: rollback")
                continue

            fallback_code = self._parse_llm_code(response)
            if not fallback_code:
                self._rollback_ostream_block(
                    header_path, tracked_block, marker, all_types)
                cprint_stats['skipped'] += 1
                continue

            with open(header_path, 'r') as f:
                file_content = f.read()
            if tracked_block not in file_content:
                tqdm.write(
                    f"  [cprint] ERROR: lost tracked block for "
                    f"{short_name}")
                cprint_stats['skipped'] += 1
                continue

            new_tracked = f"{marker}\n{fallback_code}"
            old_line_count = tracked_block.count('\n')
            new_line_count = new_tracked.count('\n')
            line_delta = new_line_count - old_line_count

            file_content = file_content.replace(
                tracked_block, new_tracked, 1)
            with open(header_path, 'w') as f:
                f.write(file_content)
            tracked_block = new_tracked

            if line_delta != 0:
                block_line = (
                    file_content[:file_content.find(new_tracked)]
                    .count('\n') + 1
                )
                for other in all_types:
                    if (other[0] == header_path
                            and other[3] > block_line):
                        other[3] += line_delta

            success, stderr, stdout = self._compile_project(
                compile_script, project_abs)
            if success:
                cprint_stats['fallback'] += 1
                cprint_pbar.set_postfix_str(f"{short_name}: fallback OK")
            else:
                cprint_stats['skipped'] += 1
                cprint_pbar.set_postfix_str(
                    f"{short_name}: FAILED — rolling back")
                tqdm.write(
                    f"  [cprint] {short_name} failed all attempts "
                    f"— rolling back")
                self._rollback_ostream_block(
                    header_path, tracked_block, marker, all_types)

        cprint_pbar.close()
        print(f"  [cprint] Summary: {cprint_stats['ok']} OK, "
              f"{cprint_stats['repaired']} repaired, "
              f"{cprint_stats['fallback']} fallback, "
              f"{cprint_stats['skipped']} skipped")

        # --- Done — rebuild call graph ----------------------------------
        self._rebuild_call_graph()

    @staticmethod
    def _rollback_ostream_block(header_path, tracked_block, marker,
                                all_types):
        """Remove a tracked operator<< block from a header file.

        Restores the file to its state before the operator<< was
        inserted, and adjusts ``end_line`` entries in *all_types*
        for other types in the same file.
        """
        with open(header_path, 'r') as f:
            content = f.read()

        # The tracked_block is "marker\ncode".  It was inserted as
        # "\nmarker\ncode\n" before a closing '}'.  Remove the full
        # insertion including the surrounding newlines.
        search = f"\n{tracked_block}\n"
        if search in content:
            line_delta = -(search.count('\n') - 1)
            block_pos = content.find(search)
            block_line = content[:block_pos].count('\n') + 1
            content = content.replace(search, "\n", 1)
            with open(header_path, 'w') as f:
                f.write(content)
            # Adjust end_lines for other types in the same file.
            for other in all_types:
                if other[0] == header_path \
                        and other[3] > block_line:
                    other[3] += line_delta

    def _rebuild_call_graph(self):
        """Regenerate the call graph after source files have been modified."""
        print("  [ostream] Rebuilding call graph (line numbers changed)...")
        try:
            if os.path.exists(self.call_graph_path):
                os.remove(self.call_graph_path)
            # Use expanded compile_commands if available (more
            # complete), but never modify the original.
            _cc = getattr(self, '_expanded_compile_commands_path',
                          self.compile_commands_path)
            builder = CallGraphBuilder(_cc, parallel=True, max_workers=128)
            builder.build_call_graph()
            builder.export_to_json(self.call_graph_path)
            # Re-apply template-dependent-call discovery on rebuild so
            # the edges survive the round-trip (Phase 0 added them, but
            # this fresh build would otherwise drop them).
            _discover_tmpl = False
            try:
                _meta_path = os.path.join(
                    self.project_path, "metadata.json")
                if os.path.exists(_meta_path):
                    with open(_meta_path) as _mf:
                        _mjson = json.load(_mf)
                    _minfo = _mjson.get(
                        "Test Case Failure Info", _mjson)
                    _discover_tmpl = (
                        str(_minfo.get("discover_template", ""))
                        .strip().lower() == "true"
                        or str(_mjson.get("discover_template", ""))
                        .strip().lower() == "true"
                    )
            except Exception:
                pass
            if _discover_tmpl:
                builder.discover_template_dependent_calls()
                builder.export_to_json(self.call_graph_path)
            print(f"  [ostream] Call graph rebuilt: {self.call_graph_path}")
        except Exception as e:
            print(f"  [ostream] WARNING: Failed to rebuild call graph: {e}")

    @staticmethod
    def _compile(compile_script, cwd, build_type="native"):
        """Run ``compile.sh <build_type>`` and return (success, stderr, stdout)."""
        import subprocess
        try:
            result = subprocess.run(
                [compile_script, build_type],
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=300,
            )
            return (result.returncode == 0, result.stderr, result.stdout)
        except subprocess.TimeoutExpired:
            return (False, "Compilation timed out", "")
        except Exception as e:
            return (False, f"Compilation error: {e}", "")

    @staticmethod
    def _compile_project(compile_script, cwd):
        """Run compile.sh for both native and wasm targets.

        Returns ``(success, stderr, stdout)`` where *success* is True
        only when both builds pass.  When native fails, wasm is skipped
        and native's error is returned.  When native passes but wasm
        fails, the returned stderr is the wasm error, prefixed with a
        tag so LLM repair can tell which stage broke.
        """
        ok_n, err_n, out_n = DiffTraceAnalysis._compile(
            compile_script, cwd, "native")
        if not ok_n:
            return (False, err_n, out_n)
        ok_w, err_w, out_w = DiffTraceAnalysis._compile(
            compile_script, cwd, "wasm")
        if not ok_w:
            prefixed_err = (
                "[wasm compile failed — native compile passed]\n"
                + (err_w or "")
            )
            return (False, prefixed_err, out_w)
        return (True, "", (out_n or "") + (out_w or ""))

    @staticmethod
    def _filter_errors_for_file(stderr, file_path, max_lines=30):
        """Extract the last *max_lines* stderr lines that reference *file_path*.

        Matching is done against both the full path and the basename so that
        compiler messages using either form are captured.  If no lines match,
        fall back to the last *max_lines* lines of stderr.
        """
        if not stderr or not stderr.strip():
            return "(no output)"

        all_lines = stderr.strip().splitlines()
        basename = os.path.basename(file_path)

        matched = [
            line for line in all_lines
            if file_path in line or basename in line
        ]

        source = matched if matched else all_lines

        # Collect up to max_lines unique lines (skip duplicates
        # from repeated compiler notes for different overload
        # candidates).
        seen = set()
        deduped = []
        for line in source:
            if line not in seen:
                seen.add(line)
                deduped.append(line)
                if len(deduped) >= max_lines:
                    break

        return '\n'.join(deduped) if deduped else "(no output)"

    @staticmethod
    def _parse_llm_code(response):
        """Extract code from an LLM response, stripping markdown fences."""
        response = response.strip()
        matches = re.findall(r'```(?:c\+\+|cpp|c)?\s*(.*?)```', response,
                             re.DOTALL | re.IGNORECASE)
        if matches:
            return matches[0].strip()
        # Opening fence but no closing fence
        open_fence = re.match(r'^```(?:c\+\+|cpp|c)?\s*\n', response, re.IGNORECASE)
        if open_fence:
            response = response[open_fence.end():]
            if response.rstrip().endswith('```'):
                response = response.rstrip()[:-3]
            return response.strip()
        return response

    def _pre_analysze(self, calls_only: bool = True):
        """
        Custom pre_analysze that supports calls_only filtering.

        The work is split into four substeps:

        1a-i   **Analyse** – Run ``ProjectASTBuilder.process_project`` to
               discover which functions are involved and collect the
               ``instrumented_functions`` list.
        1a-ii  **Extract types** – Use ``TypeParser`` to find user-defined
               types related to the discovered functions.
        1a-iii **Add operator<<** – For each type, generate and insert an
               ``operator<<`` overload (with compile-then-fix loop).
        1a-iv  **Generate and apply function instrumentation** – Rebuild
               the call graph (line numbers may have shifted), then use
               ``FunctionInstrumentor`` to statically generate entry/exit
               ``std::cout`` code and apply it to the source files.

        Args:
            calls_only: If True (default), only instrument statements
                        involving function calls. If False, instrument all.
        """
        try:
            p = self.preprocessor
            print("=" * 60)
            print("Starting instrumentation preprocessing...")
            print(f"  Root file: {p.root_file}")
            print(f"  Root line: {p.root_line}")
            print("=" * 60)

            # Load call graph for callee-side instrumentation (if available)
            call_graph = None
            if calls_only and os.path.exists(self.call_graph_path):
                with open(self.call_graph_path, 'r') as f:
                    call_graph = json.load(f)
                print(f"  Loaded call graph ({len(call_graph.get('functions', {}))} functions)")
                print("  Mode: callee-side instrumentation (inside called function bodies)")
            elif calls_only:
                print("  WARNING: No call graph found, falling back to caller-side instrumentation")

            # ==============================================================
            # Step 1a-i: Analyse – discover involved functions
            # ==============================================================
            print("\n--- Step 1a-i: Analyse (discover involved functions) ---")

            # Compute the extra USRs for functions in the test case scope
            # BEFORE the first process_project call so they can be passed in
            # as additional instrumentation seeds.  The same set is reused for
            # Phase 1a-iv's second pass.
            extra_usrs: set = set()
            if call_graph:
                _meta_extra = p.metadata.get('Test Case Failure Info', {})
                _tc_start_extra = int(_meta_extra.get('test case start line', 0))
                _tc_end_extra = int(_meta_extra.get('test case end line', 0))
                if _tc_start_extra > 0 and _tc_end_extra >= _tc_start_extra:
                    _cg_funcs_e = call_graph.get('functions', {})
                    _cg_edges_e = call_graph.get('call_edges', {})
                    _root_abs_e = os.path.abspath(p.root_file)

                    # Functions defined in the root file AND inside the
                    # test case line range — these are the roots of the
                    # transitive reach we want to instrument.
                    _seed = []
                    for _u, _fi in _cg_funcs_e.items():
                        _fi_file = _fi.get('file', '')
                        if not _fi_file:
                            continue
                        if os.path.abspath(_fi_file) != _root_abs_e:
                            continue
                        _fl = _fi.get('line', 0)
                        if _tc_start_extra <= _fl <= _tc_end_extra:
                            _seed.append(_u)

                    # Transitive closure through call edges.
                    _wl = list(_seed)
                    while _wl:
                        _cur = _wl.pop()
                        if _cur in extra_usrs:
                            continue
                        extra_usrs.add(_cur)
                        for _callee in _cg_edges_e.get(_cur, []):
                            if _callee not in extra_usrs:
                                _wl.append(_callee)

                    # Unconditionally include every function defined in
                    # the root file.  Signal handlers registered via
                    # std::signal and callbacks passed as function
                    # pointers are invisible to static call edges, but
                    # they live in the same TU as the test body.
                    for _u, _fi in _cg_funcs_e.items():
                        _fi_file = _fi.get('file', '')
                        if _fi_file and \
                                os.path.abspath(_fi_file) == _root_abs_e \
                                and _fi.get('is_definition', False):
                            extra_usrs.add(_u)

                    # Drop anything that lives in a white-listed dir.
                    if extra_usrs:
                        _filtered = set()
                        for _u in extra_usrs:
                            _fi = _cg_funcs_e.get(_u, {})
                            _f = _fi.get('file', '')
                            if _f and not self._is_whitelisted(_f):
                                _filtered.add(_u)
                        extra_usrs = _filtered

            project_builder = ProjectASTBuilder(
                compile_commands_path=self.compile_commands_path,
                root_file=p.root_file,
                root_line=p.root_line,
                root_func_name=""
            )
            project_builder.process_project(
                calls_only=calls_only,
                call_graph=call_graph,
                output_to_stderr=self.output_to_stderr,
                additional_function_usrs=extra_usrs,
                is_c_project=self.is_c_project,
            )
            func_list = getattr(project_builder, 'instrumented_functions', [])
            print(f"  Discovered {len(func_list)} function(s) from root line")

            # Also discover functions called from OTHER lines in the
            # test case (not just the failed line).  This captures
            # setup functions, signal handlers, helper calls, etc.
            # that are in the test body but have no data dependency
            # on the failed line.
            if call_graph:
                _meta = p.metadata.get('Test Case Failure Info', {})
                _tc_start = int(_meta.get('test case start line', 0))
                _tc_end = int(_meta.get('test case end line', 0))
                if _tc_start > 0 and _tc_end >= _tc_start:
                    _existing = {
                        (f.get('file'), f.get('line'))
                        for f in func_list}
                    _cg_funcs = call_graph.get('functions', {})
                    _cg_edges = call_graph.get('call_edges', {})

                    # Find the test case function in the call graph.
                    _root_abs = os.path.abspath(p.root_file)
                    _test_func_usrs = []
                    for _u, _fi in _cg_funcs.items():
                        _fi_file = _fi.get('file', '')
                        if not _fi_file:
                            continue
                        if os.path.abspath(_fi_file) == _root_abs:
                            _fl = _fi.get('line', 0)
                            if _tc_start <= _fl <= _tc_end:
                                _test_func_usrs.append(_u)

                    # Collect ALL functions called transitively from
                    # the test case function(s).
                    _reachable = set()
                    _wl = list(_test_func_usrs)
                    while _wl:
                        _cur = _wl.pop()
                        if _cur in _reachable:
                            continue
                        _reachable.add(_cur)
                        for _callee in _cg_edges.get(_cur, []):
                            if _callee not in _reachable:
                                _wl.append(_callee)

                    # Also add ALL functions defined in the root
                    # file — even if not reachable via call edges.
                    # This catches signal handlers, callbacks, and
                    # helper functions passed as function pointers.
                    for _u, _fi in _cg_funcs.items():
                        _fi_file = _fi.get('file', '')
                        if _fi_file and \
                                os.path.abspath(_fi_file) == _root_abs \
                                and _fi.get('is_definition', False):
                            _reachable.add(_u)

                    # Add reachable functions not already in func_list.
                    _added = 0
                    for _u in _reachable:
                        _fi = _cg_funcs.get(_u, {})
                        _f = _fi.get('file', '')
                        _l = _fi.get('line')
                        if not _f or not _l:
                            continue
                        if (_f, _l) in _existing:
                            continue
                        if not _fi.get('is_definition', False):
                            continue
                        if self._is_whitelisted(_f):
                            continue
                        func_list.append({
                            'name': _fi.get('qualified_name',
                                            _fi.get('name', '')),
                            'file': _f,
                            'line': _l,
                        })
                        _existing.add((_f, _l))
                        _added += 1
                    if _added:
                        print(f"  Added {_added} function(s) from "
                              f"test case scope (lines "
                              f"{_tc_start}-{_tc_end})")

            # Filter out functions in white-listed files so that
            # TypeParser does not discover or expand types reachable
            # only through those files.
            if self._whitelist_paths and func_list:
                before = len(func_list)
                func_list = [
                    f for f in func_list
                    if not self._is_whitelisted(f.get('file', ''))
                ]
                if len(func_list) < before:
                    print(f"  White list: filtered {before - len(func_list)} "
                          f"function(s), {len(func_list)} remaining")

            # ==============================================================
            # Step 1a-ii: Extract types related to discovered functions
            # ==============================================================
            print("\n--- Step 1a-ii: Extract types ---")
            function_types = {}
            if func_list:
                # TypeParser needs the full project compile_commands.json (not
                # the build_native subset which may only contain the test file).
                meta_ti = p.metadata.get("Test Case Failure Info", {})
                full_cc = meta_ti.get("compile_command", "")
                if not full_cc or not os.path.exists(full_cc):
                    full_cc = p.compile_command_json_path  # fallback
                print(f"  Using compile commands: {full_cc}")

                tp = TypeParser(full_cc)
                tp.run()
                print("Finished tp.run()")
                function_types = tp.extract_function_types(func_list)

                # Remove types defined in white-listed files so we
                # don't generate operator<< for them or follow their
                # transitive dependencies.
                if self._whitelist_paths and function_types:
                    _wl_types_removed = 0
                    for src_file in list(function_types.keys()):
                        for func_name in list(
                                function_types[src_file].keys()):
                            types_dict = function_types[src_file][
                                func_name].get('types', {})
                            for type_file in list(types_dict.keys()):
                                if self._is_whitelisted(type_file):
                                    _wl_types_removed += len(
                                        types_dict[type_file])
                                    del types_dict[type_file]
                    if _wl_types_removed:
                        print(f"  White list: removed "
                              f"{_wl_types_removed} type(s) in "
                              f"white-listed files")

                type_output_path = os.path.join(
                    self.project_path, "function_types.json")
                with open(type_output_path, 'w') as f:
                    json.dump(function_types, f, indent=2)

                print(f"  Type extraction complete -> {type_output_path}")
            else:
                print("  No instrumented functions — skipping type extraction")

            # ==============================================================
            # Step 1a-ii.5 (optional): Auto-expand compact one-line function
            # bodies so instrumentation has room between `{` and `}`.
            # Runs BEFORE any source-modifying step (ostream / TODO).
            # Controlled by metadata.json: set "auto_expansion": "True"
            # (top-level or under "Test Case Failure Info") to opt in.
            # ==============================================================
            _auto_exp = False
            try:
                _meta_path = getattr(self, '_metadata_path', None) or \
                    os.path.join(self.project_path, "metadata.json")
                if os.path.exists(_meta_path):
                    with open(_meta_path) as _mf:
                        _mjson = json.load(_mf)
                    _info = _mjson.get("Test Case Failure Info", {})
                    _auto_exp = (
                        str(_info.get("auto_expansion", ""))
                        .strip().lower() == "true"
                        or str(_mjson.get("auto_expansion", ""))
                        .strip().lower() == "true"
                    )
            except Exception:
                _auto_exp = False

            if _auto_exp:
                print("\n--- Step 1a-ii.5: Auto-expand compact "
                      "one-line function bodies ---")
                from instrumentation.expand_oneliners import expand_files as _expand_files
                _proj_abs = os.path.abspath(self.project_path)
                _targets = set()
                # Collect every file referenced by the current call
                # graph's discovered functions.
                try:
                    with open(self.call_graph_path) as _cgf:
                        _cg_ae = json.load(_cgf)
                    for _finfo in _cg_ae.get('functions', {}).values():
                        _fp = _finfo.get('file', '')
                        if not _fp:
                            continue
                        _abs = os.path.abspath(_fp)
                        if _abs.startswith(_proj_abs + os.sep):
                            _targets.add(_abs)
                except Exception as _e:
                    print(f"  [auto-expand] warning: could not read "
                          f"call graph: {_e}")
                # Also include files from extracted type metadata.
                if function_types:
                    for _fp in function_types.keys():
                        _abs = os.path.abspath(_fp)
                        if _abs.startswith(_proj_abs + os.sep):
                            _targets.add(_abs)
                _filtered = [
                    fp for fp in sorted(_targets)
                    if os.path.exists(fp)
                    and not self._is_whitelisted(fp)
                ]
                if _filtered:
                    print(f"  [auto-expand] running on "
                          f"{len(_filtered)} file(s)...")
                    _n_changes = _expand_files(_filtered)
                    print(f"  [auto-expand] total {_n_changes} "
                          f"compact body line(s) expanded")
                    if _n_changes != 0:
                        # Expansion shifted line numbers; rebuild the
                        # call graph so downstream steps read accurate
                        # function start/end lines.
                        self._rebuild_call_graph()
                        # Also re-extract type metadata. `function_types`
                        # and `tp` were built in Step 1a-ii against the
                        # pre-expansion source; their end_line values
                        # now point inside unrelated functions. Without
                        # this refresh, `_generate_ostream_operators`
                        # inserts operator<< at stale lines, landing
                        # mid-function and breaking compilation.
                        if func_list:
                            # Resolve compile_commands exactly like
                            # Step 1a-ii does so we reuse the same DB.
                            _meta_ti = p.metadata.get(
                                "Test Case Failure Info", {})
                            _full_cc = _meta_ti.get("compile_command", "")
                            if not _full_cc or not os.path.exists(_full_cc):
                                _full_cc = p.compile_command_json_path
                            print("  [auto-expand] re-extracting type "
                                  "metadata for post-expansion source...")
                            tp = TypeParser(_full_cc)
                            tp.run()
                            function_types = tp.extract_function_types(
                                func_list)
                            if self._whitelist_paths and function_types:
                                _wl_removed = 0
                                for src_file in list(
                                        function_types.keys()):
                                    if self._is_whitelisted(src_file):
                                        _wl_removed += len(
                                            function_types[src_file])
                                        del function_types[src_file]
                                if _wl_removed:
                                    print(f"  [auto-expand] White "
                                          f"list: removed "
                                          f"{_wl_removed} type(s) "
                                          f"from whitelisted files")

                            # Persist both versions of function_types:
                            #   function_types_pristine.json = pristine
                            #     (pre-expansion) copy, kept so the
                            #     end-of-run trace_analysis_combined.json
                            #     can be built against line numbers
                            #     matching the Phase-4a-restored source.
                            #   function_types.json = post-expansion,
                            #     used by mid-run consumers (Phase 3,
                            #     llm_metadata, etc.) while the tree
                            #     is still in its expanded state.
                            _ft_live = os.path.join(
                                self.project_path,
                                "function_types.json")
                            _ft_pristine = os.path.join(
                                self.project_path,
                                "function_types_pristine.json")
                            try:
                                if os.path.exists(_ft_live):
                                    import shutil as _sh
                                    _sh.copy2(_ft_live, _ft_pristine)
                                with open(_ft_live, 'w') as _wf:
                                    json.dump(function_types, _wf,
                                              indent=2)
                                print("  [auto-expand] saved pristine "
                                      "function_types -> "
                                      "function_types_pristine.json "
                                      "and updated function_types.json "
                                      "to post-expansion lines")
                            except Exception as _e:
                                print(f"  [auto-expand] WARNING: "
                                      f"could not persist refreshed "
                                      f"function_types: {_e}")
                else:
                    print("  [auto-expand] no candidate files; skipping")

            # ==============================================================
            # Step 1a-iii: Add operator<< for discovered types
            # ==============================================================
            print("\n--- Step 1a-iii: Add operator<< ---")
            if function_types:
                self._generate_ostream_operators(function_types, tp, func_list)
            else:
                print("  No types — skipping operator<< generation")

            # ==============================================================
            # Step 1a-iv: Add instrumentation TODO markers
            # ==============================================================
            print("\n--- Step 1a-iv: Add instrumentation TODO markers ---")

            # Reload call graph (already rebuilt by _generate_ostream_operators
            # after ostream insertion shifted line numbers).
            with open(self.call_graph_path, 'r') as f:
                call_graph = json.load(f)

            # Step 1a-iii may have inserted code into the root file
            # (e.g. #include <iostream>, operator<< for types defined
            # there), shifting line numbers.  Re-locate the root line
            # by searching for its original content.
            _orig_dir = getattr(self, 'originals_dir', None)
            if _orig_dir and os.path.exists(p.root_file):
                _orig_root = os.path.join(
                    _orig_dir,
                    os.path.relpath(p.root_file, self.project_path))
                if os.path.exists(_orig_root):
                    with open(_orig_root) as _f:
                        _orig_lines = _f.readlines()
                    _root_ln = int(p.root_line)
                    if 0 < _root_ln <= len(_orig_lines):
                        _target = _orig_lines[_root_ln - 1].strip()
                        if _target:
                            with open(p.root_file) as _f:
                                _cur_lines = _f.readlines()
                            for _li, _ln in enumerate(_cur_lines):
                                if _ln.strip() == _target:
                                    if _li + 1 != _root_ln:
                                        print(
                                            f"  Root line shifted: "
                                            f"{_root_ln} -> "
                                            f"{_li + 1}")
                                        p.root_line = _li + 1
                                    break

            # Re-run analysis to insert TODO markers at function entry/exit.
            # Use the same compile_commands as step 1a-i (which may be
            # a filtered/optimized version provided by the user).
            project_builder2 = ProjectASTBuilder(
                compile_commands_path=self.compile_commands_path,
                root_file=p.root_file,
                root_line=p.root_line,
                root_func_name=""
            )
            modified_files = project_builder2.process_project(
                calls_only=calls_only,
                call_graph=call_graph,
                output_to_stderr=self.output_to_stderr,
                additional_function_usrs=extra_usrs,
                is_c_project=self.is_c_project,
            )
            func_metadata = getattr(
                project_builder2, 'function_metadata', {})
            print(f"  Collected structured metadata for "
                  f"{len(func_metadata)} function(s)")

            # The AST builder may have inserted TODO blocks into files
            # we don't want to instrument (e.g. build_native/_deps/).
            # Restore those files from pre-instrumentation originals
            # and remove them from modified_files / func_metadata.
            _orig_dir = getattr(self, 'originals_dir', None)
            if modified_files:
                _cleaned = []
                for fp, code in modified_files:
                    if self._is_whitelisted(fp):
                        # Restore original file.
                        if _orig_dir:
                            _orig = os.path.join(
                                _orig_dir,
                                os.path.relpath(fp, self.project_path))
                            if os.path.exists(_orig):
                                shutil.copy2(_orig, fp)
                    else:
                        _cleaned.append((fp, code))
                _wl = len(modified_files) - len(_cleaned)
                if _wl:
                    print(f"  Restored {_wl} whitelisted file(s)")
                modified_files = _cleaned

            if func_metadata:
                _fids_to_remove = [
                    fid for fid, meta in func_metadata.items()
                    if self._is_whitelisted(meta.get('file_path', ''))
                ]
                for fid in _fids_to_remove:
                    del func_metadata[fid]
                if _fids_to_remove:
                    print(f"  Filtered {len(_fids_to_remove)} "
                          f"whitelisted function(s)")

            if not modified_files:
                print("  No files need instrumentation")
                return [], {}

            print(f"  {len(modified_files)} file(s) with TODO markers:")
            for file_path, _ in modified_files:
                print(f"    - {file_path}")

            for file_path, instrumented_code in modified_files:
                if instrumented_code:
                    p.write_back_entire_file(file_path, instrumented_code)

            p.save_backup_manifest()

            # Save static instrumentation plan
            if func_metadata:
                static_plan_path = os.path.join(
                    self.project_path, "static_instrumentation_plan.json")
                with open(static_plan_path, 'w') as f:
                    json.dump(func_metadata, f, indent=2)
                print(f"  Static instrumentation plan saved to "
                      f"{static_plan_path}")

            print("\n" + "=" * 60)
            print("Step 1a complete — TODO markers inserted")
            print(f"  Backups saved to: {p.backup_dir}")
            print("=" * 60)

            return modified_files, func_metadata

        except Exception as e:
            print(f"\n{'=' * 60}")
            print(f"ERROR during preprocessing: {e}")
            print("Attempting to restore from backups...")
            print("=" * 60)

            try:
                self.preprocessor.restore_backups()
                print("Rollback completed successfully")
            except Exception as restore_error:
                print(f"ERROR during rollback: {restore_error}")
                print("Manual restoration may be required!")

            raise

    def run_full_preprocessing(self, skip_comment_instrumentation: bool = False, skip_instrumentation: bool = False) -> bool:
        """
        Run the complete preprocessing pipeline.

        Args:
            skip_comment_instrumentation: Skip Step 1a (TODO markers), go directly to LLM.
            skip_instrumentation: Skip Phase 1 entirely (files already instrumented).
        """
        print("\n" + "="*80)
        print("DYNAMIC TRACE ANALYSIS - FULL PREPROCESSING PIPELINE")
        print("="*80)

        # Phase 0: Build Call Graph
        print("\n" + "="*80)
        print("PHASE 0: BUILD CALL GRAPH")
        print("="*80)
        if not self._build_call_graph():
            return False

        # Save pre-instrumentation copies for Step 4a recovery
        self._save_pre_instrumentation_copies()

        # Capture reference (pristine) execution output + exit codes
        # for native and wasm BEFORE Phase 1 modifies the source tree.
        # Phase 2's verification then just compares against these
        # cached reference values -- no need to swap files and
        # recompile every verify.
        self._capture_reference_execution()

        # Phase 1: Instrumentation
        if not self.phase1_instrumentation(skip_comment_instrumentation=skip_comment_instrumentation):
            return False

        print("+++++Phase 1 finished. All code for instrumentation are added to the files.+++++")
        
        # Save instrumented files immediately so they survive even if
        # Phase 2 or later phases fail or get re-run.
        self._save_instrumented_copies()

        # Phase 2: Execution
        if not self.phase2_execution():
            return False

        

        # Phase 3: Trace Analysis
        if not self.phase3_trace_analysis():
            return False

        # Step 4a: Recover files with instrumentation placeholders
        if not self.phase4a_recover_with_placeholders():
            return False

        # Re-resolve root cause line numbers against the post-Phase-4a
        # source files (placeholders instead of full instrumentation) and
        # re-save trace_analysis.json so the line numbers are accurate.
        self._refresh_root_cause_line_numbers()

        # Phase 4 is done -- safe to remove worker directories kept
        # around for post-mortem debugging of Phase 2/3 failures.
        _worker_roots = getattr(self, '_worker_tmp_roots', [])
        for _root in _worker_roots:
            if os.path.exists(_root):
                shutil.rmtree(_root, ignore_errors=True)
        if _worker_roots:
            print(f"  Cleaned up {len(_worker_roots)} worker "
                  f"directory tree(s).")
            self._worker_tmp_roots = []

        # Remove build_native / build_wasm now that Phase 4 has completed.
        for _build_dir_name in ('build_native', 'build_wasm'):
            _build_dir = os.path.join(self.project_path, _build_dir_name)
            if os.path.isdir(_build_dir):
                shutil.rmtree(_build_dir, ignore_errors=True)
                print(f"  Removed {_build_dir_name}/")

        print("\n" + "="*80)
        print("PREPROCESSING COMPLETE")
        print("="*80)
        print(f"  Instrumentation report: {self.report_path}")
        print(f"  Native execution log:   {self.native_log}")
        print(f"  Wasm execution log:     {self.wasm_log}")
        print(f"  Trace analysis:         {self.trace_analysis_path}")
        print(f"  LLM metadata:           {self.llm_metadata_path}")
        if self.root_cause_info:
            print(f"  Root cause: {self.root_cause_info['function_name']} "
                  f"(stack-based detection)")
        else:
            print(f"  Root cause: none identified")

        print(f"\n" + "="*80)
        print("LLM TOKEN USAGE SUMMARY")
        print("="*80)
        print(f"  Total input tokens:  {GLOBAL_TOKEN_USAGE['total_input_tokens']}")
        print(f"  Total output tokens: {GLOBAL_TOKEN_USAGE['total_output_tokens']}")
        print(f"  Total tokens:        {GLOBAL_TOKEN_USAGE['total_input_tokens'] + GLOBAL_TOKEN_USAGE['total_output_tokens']}")

        return True

    # ------------------------------------------------------------------
    # Phase 0: Build Call Graph
    # ------------------------------------------------------------------

    def _build_call_graph(self) -> bool:
        """Build call graph, expanding compile_commands if needed.

        After the initial build, checks for functions that have a
        declaration but no definition in the call graph.  For each,
        searches sibling ``.cc``/``.cpp`` files for the definition and
        rebuilds the call graph with an expanded compile_commands so
        that callee functions (like ``do_get_locale``) are visible.
        """
        try:
            builder = CallGraphBuilder(self.compile_commands_path, parallel=True, max_workers=128)
            builder.build_call_graph()
            builder.export_to_json(self.call_graph_path)
            print(f"  Call graph saved: {self.call_graph_path}")

            # Check for declaration-only functions whose .cc files
            # are not in compile_commands — expand and rebuild if needed.
            expanded = self._expand_compile_commands_for_deps(
                self.call_graph_path)
            if expanded:
                print(f"  Rebuilding call graph with {expanded} "
                      f"additional source file(s)...")
                _exp_cc = getattr(self,
                    '_expanded_compile_commands_path',
                    self.compile_commands_path)
                builder2 = CallGraphBuilder(
                    _exp_cc,
                    parallel=True, max_workers=128)
                builder2.build_call_graph()
                builder2.export_to_json(self.call_graph_path)
                print(f"  Expanded call graph saved: {self.call_graph_path}")
                builder = builder2  # use expanded builder below

            # When discover_template is enabled, scan template bodies
            # for dependent member calls that clang omits from normal
            # call edges.
            _discover_tmpl = False
            try:
                _meta_path = os.path.join(
                    self.project_path, "metadata.json")
                if os.path.exists(_meta_path):
                    with open(_meta_path) as _mf:
                        _mjson = json.load(_mf)
                    _minfo = _mjson.get(
                        "Test Case Failure Info", _mjson)
                    _discover_tmpl = (
                        str(_minfo.get("discover_template", ""))
                        .strip().lower() == "true"
                        or str(_mjson.get("discover_template", ""))
                        .strip().lower() == "true"
                    )
            except Exception:
                pass
            if _discover_tmpl:
                builder.discover_template_dependent_calls()
                builder.export_to_json(self.call_graph_path)
                print(f"  Call graph updated with template edges: "
                      f"{self.call_graph_path}")

            return True
        except Exception as e:
            print(f"  ERROR building call graph: {e}")
            import traceback; traceback.print_exc()
            return False

    def _expand_compile_commands_for_deps(self, call_graph_path: str) -> int:
        """Find .cc/.cpp files that define declaration-only functions.

        When ``compile_commands.json`` only lists the test file (e.g.
        ``chrono-test.cc``), helper files like ``util.cc`` are missing.
        Functions declared in headers but defined in those missing files
        appear in the call graph with ``is_definition: false`` and empty
        call edges.  This method:

        1. Loads the call graph and collects declaration-only function
           names grouped by their header file.
        2. For each header, derives candidate ``.cc``/``.cpp`` files
           (e.g. ``util.h`` → ``util.cc``).
        3. Reads each candidate and checks whether it actually contains
           any of the declaration-only function names from the
           corresponding header.  Only files with matches are added.
        4. Appends new entries to ``compile_commands.json`` using the
           compiler flags from the first existing entry.

        Returns the number of files added (0 if none).
        """
        with open(call_graph_path) as f:
            cg = json.load(f)

        functions = cg.get('functions', {})

        # Collect declaration-only function names grouped by header.
        # header_path -> set of simple function names
        header_func_names: Dict[str, set] = {}
        for _usr, info in functions.items():
            if not info.get('is_definition', True):
                fpath = info.get('file', '')
                fname = info.get('name', '')
                if fpath and fname:
                    key = os.path.abspath(fpath)
                    if key not in header_func_names:
                        header_func_names[key] = set()
                    # Use the bare name (strip template/params for matching)
                    bare = fname.split('(')[0].split('<')[0].strip()
                    if bare:
                        header_func_names[key].add(bare)

        if not header_func_names:
            return 0

        # Load current compile_commands
        with open(self.compile_commands_path) as f:
            cc = json.load(f)

        existing_files = set()
        for entry in cc:
            fp = entry.get('file', '')
            if not os.path.isabs(fp):
                fp = os.path.join(entry.get('directory', '.'), fp)
            existing_files.add(os.path.abspath(fp))

        if not cc:
            return 0
        template = cc[0]

        source_exts = ('.cc', '.cpp', '.cxx', '.c')

        # Build candidates: header → list of matching source files
        added = 0
        for hdr, func_names in header_func_names.items():
            if self._is_whitelisted(hdr):
                continue
            hdr_dir = os.path.dirname(hdr)
            hdr_base = os.path.splitext(os.path.basename(hdr))[0]

            for ext in source_exts:
                candidate = os.path.join(hdr_dir, hdr_base + ext)
                abs_cand = os.path.abspath(candidate)
                if not os.path.exists(candidate):
                    continue
                if self._is_whitelisted(abs_cand):
                    continue
                if abs_cand in existing_files:
                    continue
                if any(d in abs_cand for d in ('/build_', '/.', '/CMakeFiles/')):
                    continue

                # Read the source file and check if any of the
                # declaration-only function names appear in it.
                try:
                    with open(candidate, 'r') as f:
                        src_content = f.read()
                except Exception:
                    continue

                matched = [n for n in func_names if n in src_content]
                if not matched:
                    print(f"    Skipping {os.path.relpath(candidate, self.project_path)}: "
                          f"no matching function names from "
                          f"{os.path.basename(hdr)}")
                    continue

                new_entry = dict(template)
                new_entry['file'] = abs_cand
                old_file = template.get('file', '')
                cmd = new_entry.get('command', '')
                if old_file and cmd:
                    new_entry['command'] = cmd.replace(old_file, abs_cand)

                cc.append(new_entry)
                added += 1
                print(f"    Added to compile_commands: "
                      f"{os.path.relpath(abs_cand, self.project_path)} "
                      f"(defines: {', '.join(sorted(matched))})")

        if added:
            # Write the expanded compile_commands to a SEPARATE file
            # so the original is never modified.
            expanded_path = os.path.join(
                os.path.dirname(self.compile_commands_path),
                'compile_commands_expanded.json')
            with open(expanded_path, 'w') as f:
                json.dump(cc, f, indent=2)
            self._expanded_compile_commands_path = expanded_path

        return added

    # ------------------------------------------------------------------
    # Step 1b helpers: TODO-block parsing and static code generation
    # ------------------------------------------------------------------

    @staticmethod
    def _find_todo_blocks(lines):
        """Find all ``/*[Instrumented] ... */`` TODO blocks in *lines*.

        Returns a list of ``(block_start, block_end)`` tuples (0-indexed,
        inclusive).
        """
        blocks = []
        i = 0
        while i < len(lines):
            if '/*[Instrumented]' in lines[i]:
                block_start = i
                j = i + 1
                while j < len(lines) and '*/' not in lines[j]:
                    j += 1
                blocks.append((block_start, j))
                i = j + 1
            else:
                i += 1
        return blocks

    @staticmethod
    def _parse_todo_block(block_lines):
        """Parse a TODO instrumentation block and extract its metadata.

        Args:
            block_lines: list of strings (the lines of the TODO block,
                including the opening ``/*`` and closing ``*/`` lines).

        Returns:
            A dict with keys ``type``, ``marker``, ``func_id``,
            ``func_name``, ``params``, and (for exit blocks)
            ``return_type``, ``return_expr``.  Returns ``None`` for
            unrecognised blocks (e.g. ASSIGNMENT blocks).
        """
        text = ''.join(block_lines)

        # --- block type ---
        if 'Function ENTRY:' in text:
            block_type = 'entry'
        elif 'Function EXIT (void):' in text or 'EXIT (void)' in text:
            block_type = 'void_exit'
        elif 'Function EXIT:' in text:
            block_type = 'exit'
        else:
            return None  # ASSIGNMENT or unknown

        # --- marker (from @@INST_START_{marker}@@) ---
        m = re.search(r'@@INST_START_([0-9a-f]{8})@@', text)
        marker = m.group(1) if m else None

        # --- func_id (from @@FUNC_ID_{name}_{func_id}@@) ---
        m = re.search(r'@@FUNC_ID_(.+?)_([0-9a-f]{8})@@', text)
        func_id = m.group(2) if m else None
        func_name_from_id = m.group(1) if m else None

        # --- func_name ---
        m = re.search(
            r'Function (?:ENTRY|EXIT(?: \(void\))?): (\S+)', text)
        func_name = m.group(1) if m else (func_name_from_id or '')

        # --- parameters ---
        params = []
        m = re.search(r'Parameters(?:\s*\([^)]*\))?\s*:\s*(.+?);', text)
        if m:
            params_str = m.group(1).strip()
            if params_str and params_str != '(none)':
                params = DiffTraceAnalysis._parse_params_desc(params_str)

        result = {
            'type': block_type,
            'marker': marker,
            'func_id': func_id,
            'func_name': func_name,
            'params': params,
        }

        # --- exit-specific fields ---
        if block_type == 'exit':
            m = re.search(r'Return type:\s*(.+?);\s*Return expression:', text)
            result['return_type'] = m.group(1).strip() if m else 'void'
            m = re.search(
                r'Return expression:\s*(.+?);\s*$', text, re.MULTILINE)
            result['return_expr'] = m.group(1).strip() if m else ''

        return result

    @staticmethod
    def _parse_params_desc(params_str):
        """Parse ``'name1: type1, name2: type2'`` into a list of dicts.

        Handles template types that contain commas (e.g.
        ``std::pair<int, int>``) by tracking ``<>`` depth.
        """
        params = []
        depth = 0
        current = ''
        for ch in params_str:
            if ch == '<':
                depth += 1
                current += ch
            elif ch == '>':
                depth -= 1
                current += ch
            elif ch == ',' and depth == 0:
                parts = current.strip().split(': ', 1)
                if len(parts) == 2:
                    params.append({
                        'name': parts[0].strip(),
                        'type': parts[1].strip(),
                    })
                current = ''
            else:
                current += ch
        if current.strip():
            parts = current.strip().split(': ', 1)
            if len(parts) == 2:
                params.append({
                    'name': parts[0].strip(),
                    'type': parts[1].strip(),
                })
        return params

    @staticmethod
    def _find_return_after_todo(lines, block_end):
        """Find the ``return`` statement following a non-void EXIT TODO block.

        Searches forward from *block_end* (0-indexed, inclusive) for the
        first ``return`` statement.

        Returns ``(ret_start, ret_end)`` as 0-indexed inclusive line
        indices, or ``None`` if no return is found within 5 lines.
        """
        i = block_end + 1
        limit = min(len(lines), block_end + 6)
        while i < limit:
            stripped = lines[i].strip()
            if not stripped or stripped.startswith('//'):
                i += 1
                continue
            if stripped.startswith('return'):
                ret_start = i
                # Multi-line return statement: scan until ';'
                while i < len(lines):
                    if ';' in lines[i]:
                        return (ret_start, i)
                    i += 1
                return (ret_start, i - 1)
            break  # non-return, non-blank line => no return found
        return None

    @staticmethod
    def _detect_indent(lines, block_end):
        """Detect indentation from lines near *block_end*.

        Returns the whitespace prefix to use for generated code.
        """
        # Try the line after the block
        if block_end + 1 < len(lines):
            nxt = lines[block_end + 1]
            indent = nxt[:len(nxt) - len(nxt.lstrip())]
            if indent:
                return indent
        # Fallback: use the block's own line
        if block_end < len(lines):
            line = lines[block_end]
            indent = line[:len(line) - len(line.lstrip())]
            if indent:
                return indent
        return "    "

    # ------------------------------------------------------------------
    # Step 1b: Parallel worker helpers
    # ------------------------------------------------------------------

    def _create_worker_dir(self, tmp_root, worker_id):
        """Create an isolated copy of the project for one worker thread.

        Copies the full project tree, rewrites hardcoded absolute paths
        in **all** text files (build scripts, CMake caches, dependency
        files, Makefiles, JSON, etc.), and clears compilation caches
        (``.o``, ``.a``, ``.d``, ``.so``, binaries) so ``make`` performs
        a clean rebuild.

        Returns the worker directory path.
        """
        original_abs = os.path.abspath(self.project_path)
        worker_dir = os.path.join(tmp_root, f'w{worker_id}')
        # Skip large directories that the worker doesn't need:
        # build_wasm/ (310MB+), build/ (cmake source build),
        # .pre_instrumentation_originals/, instrumented_files/.
        _skip_dirs = {
            'build_wasm', 'build',
            '.pre_instrumentation_originals', 'instrumented_files',
            '.instrumentation_backups',
            # IDE / editor caches — not needed for compilation and
            # can contain transient files that vanish mid-copy.
            'my_custom_cache', '.cache', '.vs', '.vscode',
            '.idea', '.ccls-cache', '.clangd',
        }
        # Skip compilation cache files (.o, .a, .so, .d) and binaries
        # during copy — they're stale and would be deleted anyway.
        _cache_exts = ('.o', '.a', '.so', '.d')

        # Pipeline-generated .log files that workers never need (these
        # can be huge — e.g. 200MB+ execution traces — and would
        # otherwise time out the path-rewrite sed pass).  We match by
        # prefix so both the live copy (e.g. ``stack_trace.log``) and
        # user-kept archives (e.g. ``stack_trace_arch.log``) are
        # skipped, but any .log belonging to the project itself (e.g.
        # a build-system log) is preserved.
        _pipeline_log_prefixes = (
            'execution_output_native',
            'execution_output_wasm',
            'reference_output_native',
            'reference_output_wasm',
            'stack_trace',
            'node_trace',
        )

        def _is_pipeline_log(item: str) -> bool:
            if not item.endswith('.log'):
                return False
            return any(item.startswith(pfx)
                       for pfx in _pipeline_log_prefixes)

        def _ignore_fn(directory, contents):
            """Skip entire dirs + cache files in build_native/ +
            pipeline-generated execution / trace logs (which can be
            hundreds of MB and don't belong in worker dirs)."""
            ignored = set()
            for item in contents:
                if item in _skip_dirs:
                    ignored.add(item)
                    continue
                if _is_pipeline_log(item):
                    ignored.add(item)
                    continue
                # Skip cache files inside build_native/
                if 'build_native' in directory:
                    full = os.path.join(directory, item)
                    if os.path.isfile(full):
                        if any(item.endswith(ext) for ext in _cache_exts):
                            ignored.add(item)
                        # Skip binaries in bin/
                        elif os.path.basename(directory) == 'bin' \
                                and not item.endswith(
                                    ('.cmake', '.txt', '.json')):
                            ignored.add(item)
            return ignored

        shutil.copytree(
            original_abs, worker_dir, symlinks=True,
            ignore=_ignore_fn,
            ignore_dangling_symlinks=True)

        worker_abs = os.path.abspath(worker_dir)

        # --- Rewrite absolute paths in ALL text files -----------------
        # Use -I to skip binary files (avoids corrupting .o / .a / bins
        # with sed).  This covers: .cmake, .make, .d, .txt, .sh, .json,
        # .yaml, .log, .pc, Makefile, Makefile2, CMakeCache.txt,
        # link.txt, flags.make, DependInfo.cmake, etc.
        #
        # We replace both with and without trailing slash to handle
        # compile.sh  ROOT_DIR="/path/fmt/"  vs CMake's  /path/fmt
        sed_failed = []
        try:
            result = _subprocess.run(
                ['grep', '-rlI',           # -I = skip binary files
                 original_abs, worker_dir],
                capture_output=True, text=True, timeout=60)
            files_to_rewrite = [
                p for p in result.stdout.strip().split('\n') if p]
        except Exception as e:
            print(f"    [worker {worker_id}] grep failed: {e}")
            files_to_rewrite = []

        # Rewrite each file independently — a timeout on one huge
        # file (e.g. a leftover multi-hundred-MB log) must not abort
        # the remaining files, especially compile.sh.
        for fpath in files_to_rewrite:
            try:
                r = _subprocess.run(
                    ['sed', '-i', f's|{original_abs}|{worker_abs}|g',
                     fpath],
                    capture_output=True, timeout=30)
                if r.returncode != 0:
                    sed_failed.append(fpath)
            except Exception as e:
                sed_failed.append(fpath)
                print(f"    [worker {worker_id}] sed failed on "
                      f"{os.path.basename(fpath)}: {e}")

        if sed_failed:
            print(f"    [worker {worker_id}] WARNING: sed failed on "
                  f"{len(sed_failed)} file(s): "
                  f"{sed_failed[:3]}")

        # --- Verify critical file: compile.sh -------------------------
        # compile.sh is the single most important file — if its
        # ROOT_DIR still points to the original, the worker compiles
        # the ORIGINAL project (fake success).
        worker_compile_sh = os.path.join(worker_dir, 'compile.sh')
        if os.path.exists(worker_compile_sh):
            with open(worker_compile_sh, 'r') as f:
                cs_content = f.read()
            if original_abs in cs_content:
                print(f"    [worker {worker_id}] CRITICAL: compile.sh "
                      f"still contains original path — forcing "
                      f"replacement")
                cs_content = cs_content.replace(
                    original_abs, worker_abs)
                with open(worker_compile_sh, 'w') as f:
                    f.write(cs_content)

        # --- Verify no remaining references to original path ----------
        try:
            check = _subprocess.run(
                ['grep', '-rlI', original_abs, worker_dir],
                capture_output=True, text=True, timeout=30)
            remaining = [
                p for p in check.stdout.strip().split('\n') if p]
            if remaining:
                print(f"    [worker {worker_id}] WARNING: "
                      f"{len(remaining)} file(s) still contain "
                      f"original path after rewrite")
        except Exception:
            pass

        # Cache files (.o, .a, .so, .d) and binaries are already
        # skipped during copytree via _ignore_fn above — no need to
        # delete them post-copy.

        return worker_dir

    # ------------------------------------------------------------------

    def _worker_instrument_one_function(
        self, worker_dir, file_path, fid, meta, owner_type,
        func_metadata, backup_dir,
        class_names=None, class_member_vars=None,
    ):
        """Instrument a single function inside an isolated worker copy.

        Performs static code generation, compilation, LLM repair (if
        needed), and fallback — all in ``worker_dir``.

        Returns a dict with keys:
            ``fid``, ``func_name``, ``file_path`` (original),
            ``status`` ('ok'|'repaired'|'fallback'|'rollback'),
            ``original_func_text``, ``instrumented_func_text``,
            ``hoisted_includes``.
        """
        from instrumentation.FunctionInstrumentor import (
            generate_entry_code,
            generate_exit_code,
            generate_void_exit_code,
        )

        original_abs = os.path.abspath(self.project_path)
        worker_abs = os.path.abspath(worker_dir)
        worker_compile_script = os.path.join(worker_abs, "compile.sh")

        # -- Map original path → worker path ----------------------------
        worker_file = os.path.abspath(file_path).replace(
            original_abs, worker_abs)

        # -- Sync target file from original (gets latest merged state) --
        shutil.copy2(file_path, worker_file)

        result = {
            'fid': fid,
            'func_name': meta['func_name'] if meta else fid,
            'file_path': file_path,  # original path (for merge)
            'status': 'rollback',
            'original_func_text': None,
            'instrumented_func_text': None,
            'hoisted_includes': [],
        }

        # -- Read worker file and find TODO blocks for this fid ---------
        with open(worker_file, 'r') as f:
            snapshot = f.read()
        lines = snapshot.splitlines(keepends=True)

        current_blocks = self._find_todo_blocks(lines)

        if meta:
            func_name = meta['func_name']
            fid_marker = f"@@FUNC_ID_{func_name}_{fid}@@"
            marker_re = re.compile(r'@@INST_START_([0-9a-f]{8})@@')
            marker_to_mb = {
                mb['marker']: mb for mb in meta['blocks']}
            blocks = []
            for bs, be in current_blocks:
                block_text = ''.join(lines[bs:be + 1])
                if fid_marker not in block_text:
                    continue
                m = marker_re.search(block_text)
                if not m:
                    continue
                marker_val = m.group(1)
                mb = marker_to_mb.get(marker_val)
                if mb:
                    blocks.append((bs, be, mb))
                else:
                    parsed = self._parse_todo_block(lines[bs:be + 1])
                    if parsed:
                        blocks.append((bs, be, parsed))
        else:
            blocks = []
            for bs, be in current_blocks:
                parsed = self._parse_todo_block(lines[bs:be + 1])
                if parsed is None:
                    continue
                if parsed.get('func_id') == fid:
                    blocks.append((bs, be, parsed))
            func_name = (blocks[0][2]['func_name']
                         if blocks else fid)

        result['func_name'] = func_name

        if not blocks:
            return result  # nothing to do — status stays 'rollback'

        # -- Find function bounds ----------------------------------------
        # Create a lightweight LLMInstrumentor just to access the
        # _find_function_from_todo_block helper.  (Only creates an LLM
        # agent on first repair call, so this is cheap.)
        _helper_instrumentor = LLMInstrumentor(
            [(worker_file, None)],
            backend=self.backend,
            compile_db_path=self.compile_commands_path,
            size_threshold=200,
            use_function_level=True,
            fixed_time=self.fixed_time,
            is_c_project=self.is_c_project,
        )

        entry_block = next(
            ((bs, be, mb) for bs, be, mb in blocks
             if mb.get('block_type', mb.get('type')) == 'entry'),
            None)
        _ref_block = entry_block or blocks[0]
        _fb_result = _helper_instrumentor._find_function_from_todo_block(
            lines, _ref_block[0], _ref_block[1])
        if _fb_result:
            func_start, func_end = _fb_result[0], _fb_result[1]
            original_func_text = "".join(
                lines[func_start:func_end + 1])
        else:
            func_start = func_end = None
            original_func_text = None

        result['original_func_text'] = original_func_text

        params = meta['params'] if meta else blocks[0][2].get('params', [])

        # -- Detect constructors/destructors and look up member vars ----
        # A constructor's bare name (before '(') matches a class name.
        # A destructor's bare name starts with '~' followed by a class
        # name.  Both benefit from printing this->member values —
        # constructors at exit (fully-constructed state) and destructors
        # at exit (about-to-be-invalidated state).
        _ctor_member_vars = None
        bare_func = func_name.split('(')[0].strip()
        # Strip leading '~' so destructors also match their class name.
        _lookup_name = bare_func.lstrip('~')
        if class_names and _lookup_name in class_names:
            if class_member_vars:
                _ctor_member_vars = (
                    class_member_vars.get(_lookup_name)
                    or class_member_vars.get(bare_func)
                )

        # -- Replace TODO blocks (bottom-up) with generated code --------
        blocks_sorted = sorted(blocks, key=lambda b: b[0], reverse=True)
        line_delta = 0

        for bs, be, mb in blocks_sorted:
            indent = self._detect_indent(lines, be)
            block_type = mb.get('block_type', mb.get('type'))
            marker = mb['marker']
            _owner = owner_type

            if block_type == 'entry':
                code = generate_entry_code(
                    func_name, params,
                    marker=marker, func_id=fid,
                    indent=indent, owner_type=_owner,
                    output_to_stderr=self.output_to_stderr,
                    is_c_project=self.is_c_project,
                )
                mb['generated_code'] = code
                code_lines = [cl + '\n' for cl in code.split('\n')]
                old_len = be - bs + 1
                lines[bs:be + 1] = code_lines
                line_delta += len(code_lines) - old_len

            elif block_type == 'exit':
                return_expr = mb.get('return_expr', '')
                return_type = mb.get('return_type', 'void')
                code = generate_exit_code(
                    func_name, params,
                    return_expr=return_expr,
                    return_type=return_type,
                    marker=marker, func_id=fid,
                    indent=indent, owner_type=_owner,
                    output_to_stderr=self.output_to_stderr,
                    is_c_project=self.is_c_project,
                )
                mb['generated_code'] = code
                code_lines = [cl + '\n' for cl in code.split('\n')]
                ret_range = self._find_return_after_todo(lines, be)
                if ret_range:
                    old_len = ret_range[1] - bs + 1
                    lines[bs:ret_range[1] + 1] = code_lines
                else:
                    old_len = be - bs + 1
                    lines[bs:be + 1] = code_lines
                line_delta += len(code_lines) - old_len

            elif block_type == 'void_exit':
                code = generate_void_exit_code(
                    func_name, params,
                    marker=marker, func_id=fid,
                    indent=indent, owner_type=_owner,
                    output_to_stderr=self.output_to_stderr,
                    member_vars=_ctor_member_vars,
                    is_c_project=self.is_c_project,
                )
                mb['generated_code'] = code
                code_lines = [cl + '\n' for cl in code.split('\n')]
                old_len = be - bs + 1
                lines[bs:be + 1] = code_lines
                line_delta += len(code_lines) - old_len

        # Write modified worker file
        with open(worker_file, 'w') as f:
            f.writelines(lines)

        # -- Compute instrumented function text --------------------------
        snapshot_func_end = func_end
        if func_end is not None:
            func_end += line_delta
            static_gen_range = (func_start, func_end)
            static_gen_func_text = "".join(
                lines[func_start:func_end + 1])
        else:
            static_gen_range = None
            static_gen_func_text = "".join(lines)

        # -- Compile native in worker dir ----------------------------------
        compile_ok, stderr, stdout = self._compile_project(
            worker_compile_script, worker_abs)

        if compile_ok:
            result['status'] = 'ok'
            result['instrumented_func_text'] = self._read_func_text(
                worker_file, func_start, func_end)
            # Fall through to WASM check below.

        # -- Static gen failed → LLM repair (native) ---------------------
        if not compile_ok:
            print(f"    [FAIL] {func_name} native — repairing "
                  f"with LLM...")

            error_output = LLMInstrumentor._format_compile_errors(
                stderr, stdout,
                file_path=worker_file, func_name=func_name,
                func_range=static_gen_range)

            attempt_succeeded = False
            max_repair = 2
            cur_range = static_gen_range
            _helper_instrumentor.agent.reset_conversation()

            for _attempt in range(max_repair):
                with open(worker_file, 'r') as f:
                    current_code = f.read()

                if cur_range is not None:
                    cur_lines = current_code.splitlines(keepends=True)
                    static_gen_func_text = "".join(
                        cur_lines[cur_range[0]:cur_range[1] + 1])

                with self._llm_semaphore:
                    fixed_code, new_range = \
                        _helper_instrumentor._repair_instrumented_functions(
                            func_name=func_name,
                            original_func_text=original_func_text,
                            previous_attempt=static_gen_func_text,
                            file_path=worker_file,
                            current_code=current_code,
                            compilation_error=error_output,
                            backup_dir=backup_dir,
                            func_range=cur_range,
                        )
                cur_range = new_range

                if fixed_code and fixed_code != current_code:
                    with open(worker_file, 'w') as f:
                        f.write(fixed_code)

                ok2, stderr2, stdout2 = self._compile_project(
                    worker_compile_script, worker_abs)
                if ok2:
                    attempt_succeeded = True
                    result['status'] = 'repaired'
                    if cur_range is not None:
                        result['instrumented_func_text'] = \
                            self._read_func_text(
                                worker_file,
                                cur_range[0], cur_range[1])
                    else:
                        with open(worker_file, 'r') as f:
                            result['instrumented_func_text'] = \
                                f.read()
                    break
                else:
                    error_output = \
                        LLMInstrumentor._format_compile_errors(
                            stderr2, stdout2,
                            file_path=worker_file,
                            func_name=func_name,
                            func_range=cur_range)

            # -- Fallback: markers-only instrumentation ------------------
            if not attempt_succeeded:
                print(f"    WARNING: Native repair failed for "
                      f"'{func_name}', trying fallback")

                if func_start is None or snapshot_func_end is None \
                        or original_func_text is None:
                    print(f"      Cannot determine function bounds"
                          f" — rolling back")
                    with open(worker_file, 'w') as f:
                        f.write(snapshot)
                    return result  # status='rollback'

                # Rollback worker to snapshot (with TODO blocks)
                with open(worker_file, 'w') as f:
                    f.write(snapshot)

                if self.is_c_project:
                    from instrumentation.c_instrumentation_prompts import (
                        build_c_fallback_instrumentation_prompt)
                    _fb_stream = ("stderr"
                                  if self.output_to_stderr else "stdout")
                    fallback_prompt = \
                        build_c_fallback_instrumentation_prompt(
                            original_func_text, _fb_stream)
                else:
                    _fb_os = ("std::cerr"
                              if self.output_to_stderr else "std::cout")
                    fallback_prompt = f"""The instrumentation of the following C++ function failed to compile after multiple attempts.
As a fallback, produce a MINIMAL version that ONLY prints entry and exit markers at the very beginning and end of the function body.

ORIGINAL FUNCTION (with TODO comments containing the marker strings):
```cpp
{original_func_text}
```

REQUIREMENTS:
1. Extract the @@INST_START_...@@ and @@INST_END_...@@ marker strings from the TODO comments.
2. At the BEGINNING of the function body (right after the opening '{{'), add an entry block for each TODO ENTRY marker:
     {_fb_os} << "@@INST_START_<marker>@@" << std::endl;
     {_fb_os} << "@@FUNC_ID_<funcname>_<hash>@@" << std::endl;
     {_fb_os} << "+++Below are Input+++" << std::endl;
     {_fb_os} << "Fail to print vars" << std::endl;
     {_fb_os} << "@@INST_END_<marker>@@" << std::endl;
   Copy the FUNC_ID line exactly from the TODO comment.
3. At the END of the function body (right before the closing '}}'), add an exit block for each TODO EXIT marker:
     {_fb_os} << "@@INST_START_<marker>@@" << std::endl;
     {_fb_os} << "@@FUNC_ID_<funcname>_<hash>@@" << std::endl;
     {_fb_os} << "---Below are Outputs---" << std::endl;
     {_fb_os} << "Fail to print vars" << std::endl;
     {_fb_os} << "@@INST_END_<marker>@@" << std::endl;
   IMPORTANT: @@FUNC_ID_...@@ MUST be INSIDE the @@INST_START_...@@/@@INST_END_...@@ block.
4. REMOVE all the /*[Instrumented]...TODO...*/ comment blocks.
5. Do NOT print any variables — just the marker pairs.
6. Use {_fb_os} for ALL printing statements.
7. Do NOT modify any other code in the function.
8. The code MUST compile with BOTH g++ (native) and emcc (wasm).
9. Return ONLY the modified function with NO explanation, NO markdown formatting, NO additional text."""

                fb_log = _helper_instrumentor._begin_prompt_log(
                    backup_dir=backup_dir,
                    source_file=file_path,
                    call_type='fallback_function',
                    prompt=fallback_prompt,
                    func_name=func_name,
                    attempt=max_repair + 1,
                )
                try:
                    _helper_instrumentor.agent.reset_conversation()
                    with self._llm_semaphore:
                        response = \
                            _helper_instrumentor.agent.get_response(
                                fallback_prompt)
                    fallback_func = \
                        _helper_instrumentor._parse_response(
                            response)
                    _helper_instrumentor._finish_prompt_log(
                        fb_log, response=response)
                except Exception:
                    _helper_instrumentor._finish_prompt_log(fb_log)
                    return result  # status='rollback'

                # Splice fallback into the snapshot file.
                with open(worker_file, 'r') as f:
                    fb_lines = f.readlines()
                fallback_func, hoisted = \
                    LLMInstrumentor._strip_includes(fallback_func)
                new_lines = fallback_func.splitlines(keepends=True)
                if new_lines and not new_lines[-1].endswith('\n'):
                    new_lines[-1] += '\n'
                fb_lines[func_start:snapshot_func_end + 1] = \
                    new_lines
                LLMInstrumentor._insert_includes_at_top(
                    fb_lines, hoisted)
                with open(worker_file, 'w') as f:
                    f.writelines(fb_lines)

                ok3, _, _ = self._compile_project(
                    worker_compile_script, worker_abs)
                if ok3:
                    result['status'] = 'fallback'
                    result['instrumented_func_text'] = fallback_func
                    result['hoisted_includes'] = hoisted
                else:
                    with open(worker_file, 'w') as f:
                        f.write(snapshot)
                    return result  # status='rollback'

        # ================================================================
        # WASM compilation check — runs after native succeeded.
        # emcc (clang) may report errors in the worker's file OR in a
        # completely different file (e.g. a header that includes our
        # modified code).  We detect which file has the error and
        # repair that file.
        # ================================================================
        if result['status'] == 'rollback':
            return result  # native never succeeded

        wasm_ok, wasm_stderr, wasm_stdout = self._compile(
            worker_compile_script, worker_abs, "wasm")

        if wasm_ok:
            return result  # both native and wasm pass

        print(f"    [WASM FAIL] {func_name} — native OK but wasm "
              f"failed, repairing...")

        # -- Collect and analyse WASM errors ----------------------------
        wasm_error_all = (wasm_stderr or '') + '\n' + (wasm_stdout or '')

        # Extract the file:line where the first real error is.
        _err_file_re = re.compile(
            r'^(/\S+?):(\d+):\d+: (?:fatal )?error:',
            re.MULTILINE)
        _err_match = _err_file_re.search(wasm_error_all)
        wasm_error_file = _err_match.group(1) if _err_match else None
        wasm_error_line = (int(_err_match.group(2))
                           if _err_match else None)

        # Determine whether the error is in the worker's own file or
        # a different file in the worker copy.
        error_in_worker_file = (
            wasm_error_file is not None
            and os.path.abspath(wasm_error_file)
            == os.path.abspath(worker_file))

        # Build filtered error output for the LLM.
        if wasm_error_file:
            wasm_error_filtered = self._filter_errors_for_file(
                wasm_error_all, wasm_error_file, max_lines=30)
        else:
            # No parseable error file — use stderr tail.
            err_lines = wasm_error_all.strip().splitlines()
            wasm_error_filtered = '\n'.join(err_lines[-40:])

        # Decide which file to send to the LLM for repair.
        if error_in_worker_file:
            repair_file = worker_file
        elif wasm_error_file and os.path.exists(wasm_error_file):
            repair_file = wasm_error_file
        else:
            # Cannot determine error file — skip WASM repair.
            print(f"    [WASM] cannot determine error file — "
                  f"skipping WASM repair")
            return result

        print(f"    [WASM] error in: {repair_file}"
              f"{'' if error_in_worker_file else ' (DIFFERENT file)'}"
              f" line {wasm_error_line or '?'}")

        max_wasm_repair = 2
        _helper_instrumentor.agent.reset_conversation()

        for _wasm_attempt in range(max_wasm_repair):
            # Read the file that actually has the error.
            with open(repair_file, 'r') as f:
                repair_content = f.read()

            # Extract a context window around the error line.
            if wasm_error_line is not None:
                rp_lines = repair_content.splitlines(keepends=True)
                ctx_start = max(0, wasm_error_line - 20)
                ctx_end = min(len(rp_lines),
                              wasm_error_line + 20)
                error_context = "".join(
                    rp_lines[ctx_start:ctx_end])
            else:
                error_context = repair_content[:6000]

            wasm_repair_prompt = f"""The following C++ code compiles successfully with g++ (native) but FAILS with emcc (Emscripten/clang, for WebAssembly).

FILE: {repair_file}

CODE AROUND THE ERROR (lines {wasm_error_line - 20 if wasm_error_line else '?'}-{wasm_error_line + 20 if wasm_error_line else '?'}):
```cpp
{error_context}
```

EMCC (WASM) COMPILATION ERRORS:
```
{wasm_error_filtered}
```

Common issues when porting from g++ to emcc:
- `auto` deduces value types, not references. For functions returning `T&`, use `decltype(auto)` instead of `auto` to preserve the reference.
- emcc/clang is stricter about implicit conversions and non-const lvalue references binding to temporaries.
- Template instantiation may resolve differently.

Fix the code so it compiles with BOTH g++ and emcc.
Return ONLY the complete fixed file content with NO explanation, NO markdown formatting.
The fix MUST preserve all @@INST_START_...@@, @@INST_END_...@@, @@FUNC_ID_...@@ markers."""

            fb_log = _helper_instrumentor._begin_prompt_log(
                backup_dir=backup_dir,
                source_file=repair_file,
                call_type='wasm_repair',
                prompt=wasm_repair_prompt,
                func_name=func_name,
                attempt=_wasm_attempt + 1,
            )
            try:
                with self._llm_semaphore:
                    response = \
                        _helper_instrumentor.agent.get_response(
                            wasm_repair_prompt)
                fixed_code = _helper_instrumentor._parse_response(
                    response)
                _helper_instrumentor._finish_prompt_log(
                    fb_log, response=response)
            except Exception as e:
                _helper_instrumentor._finish_prompt_log(fb_log)
                print(f"    [WASM] LLM error: {e}")
                break

            if not fixed_code:
                break

            # Apply the LLM fix to the error file.
            with open(repair_file, 'w') as f:
                f.write(fixed_code)

            # Verify BOTH native and wasm still compile.
            n_ok, _, _ = self._compile_project(
                worker_compile_script, worker_abs)
            if not n_ok:
                print(f"    [WASM] repair broke native — reverting")
                with open(repair_file, 'w') as f:
                    f.write(repair_content)
                continue

            w_ok, w_err2, w_out2 = self._compile(
                worker_compile_script, worker_abs, "wasm")
            if w_ok:
                print(f"    [WASM] repair succeeded for "
                      f"'{os.path.basename(repair_file)}'")
                if error_in_worker_file:
                    # Update the instrumented function text.
                    result['instrumented_func_text'] = \
                        self._read_func_text(
                            worker_file, func_start, func_end)
                else:
                    # The fix is in a DIFFERENT file — record it
                    # so the merge step can apply it to the
                    # original project.
                    original_repair_file = repair_file.replace(
                        worker_abs, original_abs)
                    result.setdefault(
                        'extra_file_patches', []).append({
                            'file_path': original_repair_file,
                            'content': fixed_code,
                        })
                return result
            else:
                # Update error for next attempt.
                wasm_error_all = (w_err2 or '') + '\n' + \
                    (w_out2 or '')
                _err_match2 = _err_file_re.search(wasm_error_all)
                if _err_match2:
                    wasm_error_file = _err_match2.group(1)
                    wasm_error_line = int(_err_match2.group(2))
                wasm_error_filtered = \
                    self._filter_errors_for_file(
                        wasm_error_all,
                        wasm_error_file or repair_file,
                        max_lines=30)
                # Revert so next attempt starts clean.
                with open(repair_file, 'w') as f:
                    f.write(repair_content)

        # WASM repair exhausted — rollback the instrumentation entirely
        # so native and WASM stay consistent (both uninstrumented for
        # this function).
        print(f"    [WASM] repair failed for '{func_name}' — "
              f"rolling back instrumentation")
        # Restore worker file to the pre-instrumentation snapshot.
        with open(worker_file, 'w') as f:
            f.write(snapshot)
        result['status'] = 'rollback'
        result['instrumented_func_text'] = None
        return result

    @staticmethod
    def _read_func_text(file_path, func_start, func_end):
        """Read a function's text from *file_path* by line range."""
        if func_start is None or func_end is None:
            return None
        with open(file_path, 'r') as f:
            lines = f.readlines()
        if func_end >= len(lines):
            func_end = len(lines) - 1
        return "".join(lines[func_start:func_end + 1])

    # ------------------------------------------------------------------

    @staticmethod
    def _braces_balanced(text):
        """Check that ``{`` and ``}`` are balanced outside comments/strings.

        Returns True if balanced.  A quick heuristic that skips ``//``
        line comments, ``/* ... */`` block comments, and string/char
        literals to avoid false positives from braces in strings.
        """
        depth = 0
        i = 0
        n = len(text)
        while i < n:
            c = text[i]
            if c == '/' and i + 1 < n:
                if text[i + 1] == '/':
                    # Skip line comment
                    i = text.find('\n', i + 2)
                    if i == -1:
                        break
                    i += 1
                    continue
                elif text[i + 1] == '*':
                    # Skip block comment
                    end = text.find('*/', i + 2)
                    i = end + 2 if end != -1 else n
                    continue
            elif c in ('"', "'"):
                # Skip string/char literal
                quote = c
                i += 1
                while i < n and text[i] != quote:
                    if text[i] == '\\':
                        i += 1  # skip escaped char
                    i += 1
                i += 1  # skip closing quote
                continue
            elif c == '{':
                depth += 1
            elif c == '}':
                depth -= 1
                if depth < 0:
                    return False
            i += 1
        return depth == 0

    @staticmethod
    def _merge_instrumentation_result(result, merge_lock):
        """Apply a successful worker result back to the original file.

        Uses ``content.find()`` on the original function text (which
        contains unique ``@@FUNC_ID_...@@`` markers) to locate and
        replace the function.

        Before accepting the merge, validates that the instrumented
        function text has balanced braces.  Partially-instrumented
        functions (e.g. entry replaced but exit TODO left as a comment
        inside a branch) can corrupt the brace structure, leading to
        compilation errors when merged.
        """
        if result['status'] == 'rollback' \
                or not result.get('instrumented_func_text') \
                or not result.get('original_func_text'):
            return

        with merge_lock:
            with open(result['file_path'], 'r') as f:
                content = f.read()

            idx = content.find(result['original_func_text'])
            if idx == -1:
                print(f"    WARNING: merge failed for "
                      f"{result['func_name']} — original text "
                      f"not found in {result['file_path']}")
                return

            content = (
                content[:idx]
                + result['instrumented_func_text']
                + content[idx + len(result['original_func_text']):]
            )

            # Validate the merged instrumented text.
            instr_text = result['instrumented_func_text']
            merge_ok = True

            # Check 1: brace balance.
            if not DiffTraceAnalysis._braces_balanced(instr_text):
                print(f"    WARNING: merge for {result['func_name']} "
                      f"has unbalanced braces — "
                      f"partial instrumentation?")
                merge_ok = False

            # Check 2: unreplaced TODO blocks.
            if '/*[Instrumented]' in instr_text and 'TODO:' in instr_text:
                print(f"    WARNING: merge for {result['func_name']} "
                      f"has unreplaced TODO block")
                merge_ok = False

            if not merge_ok:
                # Collect for post-run pdb inspection (can't break
                # into pdb from a worker thread).
                _failed_merges = getattr(
                    DiffTraceAnalysis, '_failed_merges', None)
                if _failed_merges is not None:
                    _failed_merges.append({
                        'func_name': result['func_name'],
                        'file_path': result['file_path'],
                        'fid': result.get('fid', ''),
                        'instrumented_func_text': result['instrumented_func_text'],
                        'original_func_text': result['original_func_text'],
                    })
                return

            if result.get('hoisted_includes'):
                lines = content.splitlines(keepends=True)
                LLMInstrumentor._insert_includes_at_top(
                    lines, result['hoisted_includes'])
                content = ''.join(lines)

            with open(result['file_path'], 'w') as f:
                f.write(content)

            # Apply any extra file patches from WASM repair (fixes
            # to files other than the one being instrumented).
            for patch in result.get('extra_file_patches', []):
                patch_path = patch['file_path']
                if os.path.exists(patch_path):
                    with open(patch_path, 'w') as f:
                        f.write(patch['content'])
                    print(f"    [WASM merge] patched "
                          f"{os.path.basename(patch_path)}")

    # ------------------------------------------------------------------
    # Step 1b: Static code generation + compile-and-repair (parallel)
    # ------------------------------------------------------------------

    def _instrument_with_static_gen(self, modified_files,
                                    func_metadata=None):
        """Replace TODO markers with statically-generated instrumentation
        code using a fixed-size thread pool.

        Each worker thread operates on its own copy of the project
        directory.  On successful compilation the result is merged back
        to the original source tree under a global lock.

        Args:
            modified_files: List of ``(file_path, source_code)`` tuples
                from ``_pre_analysze``.
            func_metadata: Optional dict ``{func_id: {...}}`` with
                per-function metadata from the AST builder.

        Returns ``True`` on success.
        """
        if func_metadata is None:
            func_metadata = {}
        from instrumentation.FunctionInstrumentor import (
            generate_entry_code,
            generate_exit_code,
            generate_void_exit_code,
        )

        # Build func_id → owner_type mapping from the call graph.
        owner_type_by_fid: Dict[str, str] = {}
        if os.path.exists(self.call_graph_path):
            with open(self.call_graph_path, 'r') as f:
                _cg = json.load(f)
            _cg_funcs = _cg.get('functions', {})
            for fid, meta in func_metadata.items():
                plan_name = meta.get('func_name', '')
                plan_file = meta.get('file_path', '')
                plan_base = os.path.basename(plan_file) if plan_file else ''
                for usr, fi in _cg_funcs.items():
                    cg_base = os.path.basename(fi.get('file', ''))
                    if (fi.get('qualified_name', '') == plan_name or
                            fi.get('name', '') == plan_name) \
                            and plan_base and cg_base == plan_base:
                        classes = re.findall(r'@S@([^@]+)', usr)
                        if classes:
                            owner_type_by_fid[fid] = classes[-1]
                        break

        # Build class_name → [member_vars] mapping from ostream plan.
        # Used to print this->member at constructor exit.
        class_member_vars: Dict[str, list] = {}
        ostream_plan_path = os.path.join(
            self.project_path, "ostream_plan.json")
        if os.path.exists(ostream_plan_path):
            with open(ostream_plan_path) as f:
                _oplan = json.load(f)
            for t in _oplan.get('types', []):
                qname = t.get('qualified_name', '')
                members = t.get('member_variables', [])
                if qname and members:
                    # Store by both qualified and bare name.
                    class_member_vars[qname] = members
                    bare = qname.rsplit('::', 1)[-1]
                    if bare not in class_member_vars:
                        class_member_vars[bare] = members

        # Build set of class names for constructor detection.
        class_names = set()
        if os.path.exists(self.call_graph_path):
            with open(self.call_graph_path, 'r') as f:
                _cg2 = json.load(f)
            for _usr, name in _cg2.get('classes', {}).items():
                class_names.add(name)

        project_abs = os.path.abspath(self.project_path)
        compile_script = os.path.join(project_abs, "compile.sh")

        if not os.path.exists(compile_script):
            print("    ERROR: compile.sh not found — cannot verify "
                  "compilation")
            return False

        # ---- prompt logging setup --------------------------------------
        backup_dir = os.path.join(
            self.project_path, ".instrumentation_backups")
        os.makedirs(os.path.join(backup_dir, "prompt_logs"), exist_ok=True)

        # ---- Collect all (file, fid, meta) tasks -----------------------
        all_tasks = []
        _wl_skipped = 0
        for file_path, _ in modified_files:
            if not os.path.exists(file_path):
                continue
            if self._is_whitelisted(file_path):
                _wl_skipped += 1
                continue
            file_abs = os.path.abspath(file_path)
            if func_metadata:
                for fid, meta in func_metadata.items():
                    if os.path.abspath(meta['file_path']) == file_abs:
                        # Also skip if the meta's file_path itself
                        # is whitelisted (can differ from file_path
                        # in modified_files).
                        if self._is_whitelisted(meta['file_path']):
                            continue
                        all_tasks.append((file_path, fid, meta))
            else:
                # Fallback: discover func_ids from TODO blocks.
                with open(file_path, 'r') as f:
                    init_lines = f.readlines()
                todo_blocks = self._find_todo_blocks(init_lines)
                seen_fids = {}
                for bs, be in todo_blocks:
                    parsed = self._parse_todo_block(
                        init_lines[bs:be + 1])
                    if parsed is None:
                        continue
                    _fid = parsed.get('func_id')
                    if _fid and _fid not in seen_fids:
                        seen_fids[_fid] = True
                        all_tasks.append(
                            (file_path, _fid, None))

        if _wl_skipped:
            print(f"    Skipped {_wl_skipped} white-listed file(s)")
        if not all_tasks:
            print("    No functions to instrument")
            return True

        # ---- Create worker directories (in parallel) -------------------
        # Each copy is independent (separate destination dir, no shared
        # state), and the work is I/O bound (copytree + grep + sed).
        # Running serially with 32 workers took ~5 min at ~10s/copy.
        # Parallelizing lets the OS schedule disk I/O and finishes in
        # roughly one copy-time regardless of worker count.
        NUM_WORKERS = min(self.num_workers, len(all_tasks))
        # Use a configurable temp base for worker copies. Override via the
        # WASM_PREPROCESS_TMP env var; defaults to the system temp dir.
        _temp_base = os.environ.get('WASM_PREPROCESS_TMP', tempfile.gettempdir())
        os.makedirs(_temp_base, exist_ok=True)
        tmp_root = tempfile.mkdtemp(prefix='instr_workers_',
                                    dir=_temp_base)
        from tqdm import tqdm as _tqdm
        worker_queue = queue.Queue()
        _copy_pbar = _tqdm(total=NUM_WORKERS,
                           desc="    Creating worker copies",
                           unit="copy", leave=False)
        _copy_lock = threading.Lock()

        def _make_one(worker_id):
            wdir = self._create_worker_dir(tmp_root, worker_id)
            with _copy_lock:
                worker_queue.put(wdir)
                _copy_pbar.update(1)
            return wdir

        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as _copy_pool:
            list(_copy_pool.map(_make_one, range(NUM_WORKERS)))
        _copy_pbar.close()
        print(f"    {NUM_WORKERS} worker copies ready in {tmp_root}")

        # ---- Shared state for parallel dispatch ------------------------
        merge_lock = threading.Lock()
        stats = {'ok': 0, 'repaired': 0, 'fallback': 0, 'rollback': 0}
        func_instr_status = {}
        stats_lock = threading.Lock()

        # Per-source-file lock: at most one worker at a time runs the
        # full read → modify → compile → merge cycle for a given file.
        # Workers on DIFFERENT files still run in parallel.  This
        # eliminates the race where worker A's merge-back to
        # TestSuite.c shifts content out from under worker B's earlier
        # copy of the same file, causing B to either find its TODO
        # blocks missing (silent rollback) or corrupt the merge.
        from collections import defaultdict
        _file_locks = {}
        _file_locks_mutex = threading.Lock()

        def _get_file_lock(path):
            """Return the lock guarding *path*. Thread-safe."""
            key = os.path.abspath(path)
            with _file_locks_mutex:
                if key not in _file_locks:
                    _file_locks[key] = threading.Lock()
                return _file_locks[key]

        from tqdm import tqdm
        # Main progress bar: completed functions.
        step1b_pbar = tqdm(
            total=len(all_tasks), desc="  [step1b]",
            unit="func", leave=True,
            bar_format=(
                "{l_bar}{bar}| {n_fmt}/{total_fmt} "
                "[{elapsed}<{remaining}, {rate_fmt}] {postfix}"
            ),
        )
        # Track active workers for the postfix display.
        active_workers = {'count': 0}
        pbar_lock = threading.Lock()

        def _update_pbar(func_short, status, entering=False):
            """Thread-safe progress bar update."""
            with pbar_lock:
                if entering:
                    active_workers['count'] += 1
                    step1b_pbar.set_postfix_str(
                        f"active={active_workers['count']}/"
                        f"{NUM_WORKERS}  {func_short}: compiling")
                else:
                    active_workers['count'] = max(
                        0, active_workers['count'] - 1)
                    step1b_pbar.update(1)
                    step1b_pbar.set_postfix_str(
                        f"active={active_workers['count']}/"
                        f"{NUM_WORKERS}  {func_short}: {status}")

        def _dispatch(task):
            file_path, fid, meta = task
            _owner = owner_type_by_fid.get(fid)
            _fn = (meta['func_name'] if meta else fid)
            _short = _fn.split('(')[0].rsplit('::', 1)[-1]

            worker_dir = worker_queue.get()  # blocks until free
            _update_pbar(_short, '', entering=True)
            try:
                # Serialize all workers that modify the same source
                # file.  The lock covers the entire read → modify →
                # compile → merge cycle so that no other worker can
                # read a half-merged version of the file.  Workers
                # on different files still run fully in parallel.
                # file_lock = _get_file_lock(file_path)
                # with file_lock:
                res = self._worker_instrument_one_function(
                    worker_dir=worker_dir,
                    file_path=file_path,
                    fid=fid,
                    meta=meta,
                    owner_type=_owner,
                    func_metadata=func_metadata,
                    backup_dir=backup_dir,
                    class_names=class_names,
                    class_member_vars=class_member_vars,
                )

                # Merge successful result to original source tree.
                self._merge_instrumentation_result(res, merge_lock)

                # Update shared stats.
                with stats_lock:
                    stats[res['status']] += 1
                    func_instr_status[fid] = res['status']

                _update_pbar(_short, res['status'], entering=False)
                return res
            except Exception:
                _update_pbar(_short, 'error', entering=False)
                raise
            finally:
                worker_queue.put(worker_dir)  # return to pool

        # ---- Execute in parallel ---------------------------------------
        DiffTraceAnalysis._failed_merges = []
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            futures = [executor.submit(_dispatch, t)
                       for t in all_tasks]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"    Worker error: {e}")
                    import traceback
                    traceback.print_exc()

        step1b_pbar.close()

        # ---- Inspect failed merges in main thread ----------------------
        _failed = DiffTraceAnalysis._failed_merges
        if _failed:
            print(f"\n    WARNING: {len(_failed)} merge(s) rejected "
                  f"(unbalanced braces / unreplaced TODO):")
            for fm in _failed:
                print(f"      - {fm['func_name']} in "
                      f"{os.path.basename(fm['file_path'])}")
        DiffTraceAnalysis._failed_merges = None

        # ---- Ensure #include <iostream> in every instrumented file -----
        # Static-gen inserts std::cout statements into function bodies
        # but does not touch includes. Header files (e.g. *-inl.h) that
        # never included <iostream> will fail to compile. Prepend the
        # include to any file that now contains '@@INST_START_' markers.
        # For each touched file, record the 1-indexed line AT which the
        # include was inserted so we can bump downstream function line
        # numbers in func_metadata (keeping the plan consistent with the
        # live post-1b file).
        _iostream_touched = {}  # {abs_path: insert_line_1indexed}
        _files_with_instr = {
            os.path.abspath(t[0]) for t in all_tasks
        }
        # C projects use <stdio.h>; C++ projects use <iostream>.
        _needed_inc = ('#include <stdio.h>'
                       if self.is_c_project
                       else '#include <iostream>')
        for _fpath in _files_with_instr:
            if not os.path.exists(_fpath):
                continue
            try:
                with open(_fpath, 'r') as _rf:
                    _content = _rf.read()
            except Exception:
                continue
            if '@@INST_START_' not in _content:
                continue
            if _needed_inc in _content:
                continue
            # Insert after the last existing #include, or at top.
            _lines = _content.splitlines(keepends=True)
            _last_inc = -1
            for _i, _ln in enumerate(_lines):
                if _ln.lstrip().startswith('#include'):
                    _last_inc = _i
            _insert_at = (_last_inc + 1) if _last_inc >= 0 else 0
            _lines.insert(_insert_at, _needed_inc + '\n')
            with open(_fpath, 'w') as _wf:
                _wf.writelines(_lines)
            # Record as 1-indexed. Functions at or below this line shift
            # down by +1 in the live file.
            _iostream_touched[_fpath] = _insert_at + 1
        if _iostream_touched:
            print(f"    Added {_needed_inc} to "
                  f"{len(_iostream_touched)} file(s).")

        # ---- Bookkeeping: bump plan line numbers for shifted funcs -----
        # Mirrors the ostream phase pattern (lines 314-317): when a file
        # gets a new include, every function below the insertion point
        # shifts down by 1. Update start_line/end_line so later code
        # that consults the plan against the live post-1b file stays
        # consistent. original_start_line/original_end_line are NOT
        # touched -- they track the pristine pre-1a source (restored
        # by Phase 4a), which never receives the iostream include.
        if _iostream_touched and func_metadata:
            _bumped = 0
            for _meta in func_metadata.values():
                _meta_abs = os.path.abspath(_meta.get('file_path', ''))
                _insert_line = _iostream_touched.get(_meta_abs)
                if _insert_line is None:
                    continue
                if _meta.get('start_line') is not None \
                        and _meta['start_line'] >= _insert_line:
                    _meta['start_line'] += 1
                    _bumped += 1
                if _meta.get('end_line') is not None \
                        and _meta['end_line'] >= _insert_line:
                    _meta['end_line'] += 1
            # Persist updated plan so downstream phases (and any re-run
            # that skips Phase 1a) see consistent line numbers.
            _plan_path = os.path.join(
                self.project_path, "static_instrumentation_plan.json")
            try:
                with open(_plan_path, 'w') as _pf:
                    json.dump(func_metadata, _pf, indent=2)
                if _bumped:
                    print(f"    Bumped start_line for {_bumped} "
                          f"function(s) in the plan.")
            except Exception as _e:
                print(f"    WARNING: failed to re-save plan: {_e}")

        # ---- Final compile to verify all together ----------------------
        print(f"\n    Verifying combined instrumentation...")
        final_ok, final_err, _ = self._compile_project(
            compile_script, project_abs)
        if final_ok:
            print(f"    Final verification compile: OK")
        else:
            print(f"    WARNING: Final verification compile FAILED")
            print(f"    stderr (first 500 chars): "
                  f"{final_err[:500] if final_err else '(empty)'}")
            # pdb.set_trace()

        # ---- Defer worker directory cleanup until after Phase 4 --------
        # Keeping the worker copies around through Phase 2 (execution)
        # and Phase 3 (trace analysis) means that if either phase throws,
        # the user can still inspect the per-worker instrumented trees
        # to debug a specific function's output. Cleanup happens in
        # run_full_preprocessing() after Phase 4a succeeds.
        if not hasattr(self, '_worker_tmp_roots'):
            self._worker_tmp_roots = []
        self._worker_tmp_roots.append(tmp_root)

        # ---- Summary ---------------------------------------------------
        print(f"\n    Step 1b summary: "
              f"{stats['ok']} OK (static), "
              f"{stats['repaired']} repaired (LLM), "
              f"{stats['fallback']} fallback (markers only), "
              f"{stats['rollback']} rolled back")

        status_path = os.path.join(self.project_path,
                                   "func_instr_status.json")
        with open(status_path, 'w') as f:
            json.dump(func_instr_status, f, indent=2)
        print(f"    Per-function status saved to {status_path}")

        # pdb.set_trace()

        return True

    # ------------------------------------------------------------------
    # Step 1c: Constructor logger injection
    # ------------------------------------------------------------------

    def _inject_constructor_loggers(self, func_metadata: Dict,
                                    call_graph: Dict) -> None:
        """Inject a ``ConstructorLogger`` base class into classes whose
        constructors were instrumented in step 1b.

        Uses ``clang.cindex`` AST to reliably locate class definitions,
        base-specifier lists, and constructor initializer lists.

        For each instrumented constructor that belongs to a class with
        existing base classes:

        1.  Define ``logger_<ClassName>::ConstructorLogger`` inline in the
            target file (after ``#include`` directives).
        2.  Add it as the **first** base in the class inheritance list.
        3.  Prepend a ``ConstructorLogger("<ClassName>")`` call as the
            first entry in the constructor's initializer list (adding one
            if none exists).
        4.  Compile with ``compile.sh native`` and drop into ``pdb`` on
            failure.
        """
        # Gate: only run this step if metadata explicitly enables it.
        # metadata.json should set "constructor_inheritance": "True"
        # (top-level or under "Test Case Failure Info") to opt in.
        _enable = False
        _increased_traversal = False
        try:
            _meta_path = os.path.join(
                self.project_path, "metadata.json")
            if os.path.exists(_meta_path):
                with open(_meta_path) as _mf:
                    _mjson = json.load(_mf)
                _info = _mjson.get(
                    "Test Case Failure Info", _mjson)
                _enable = (
                    str(_info.get("constructor_inheritance", ""))
                    .strip().lower() == "true"
                    or str(_mjson.get("constructor_inheritance", ""))
                    .strip().lower() == "true"
                )
                _increased_traversal = (
                    str(_info.get("increased_traversal_depth", ""))
                    .strip().lower() == "true"
                    or str(_mjson.get("increased_traversal_depth", ""))
                    .strip().lower() == "true"
                )
        except Exception:
            _enable = False
        if not _enable:
            print("  Step 1c: skipped "
                  "(metadata.constructor_inheritance != True)")
            return

        import subprocess
        from clang import cindex
        import repair.config as _cfg  # noqa: F401 — triggers setup_libclang()

        classes = call_graph.get('classes', {})
        functions = call_graph.get('functions', {})

        project_abs = os.path.abspath(self.project_path)
        compile_script = os.path.join(project_abs, "compile.sh")

        # --- build compile-arg lookup from compile_commands.json --------
        compile_args_map: Dict[str, List[str]] = {}
        try:
            with open(self.compile_commands_path) as f:
                compile_db = json.load(f)
            for entry in compile_db:
                src = entry.get('file', '')
                cmd = entry.get('command', '')
                if not cmd:
                    continue
                # Extract compiler flags (skip compiler name, -c, -o, and
                # the source file itself).
                parts = cmd.split()
                args = []
                skip_next = False
                for p in parts[1:]:
                    if skip_next:
                        skip_next = False
                        continue
                    if p in ('-o', '-c'):
                        skip_next = True
                        continue
                    if p == src or p.endswith('.cpp') or p.endswith('.c'):
                        continue
                    args.append(p)
                compile_args_map[os.path.abspath(src)] = args
        except Exception:
            pass  # best-effort; will fall back to basic args

        def _get_compile_args(file_path: str) -> List[str]:
            """Return clang args for *file_path*, falling back to the
            first entry's flags or basic ``-std=c++17``."""
            abs_fp = os.path.abspath(file_path)
            if abs_fp in compile_args_map:
                return compile_args_map[abs_fp]
            # For header files, use any entry's flags.
            if compile_args_map:
                return next(iter(compile_args_map.values()))
            return ['-std=c++17']

        # --- build reverse map: class_name -> [class_usr, ...] ---------
        class_name_to_usrs: Dict[str, List[str]] = {}
        for usr, name in classes.items():
            class_name_to_usrs.setdefault(name, []).append(usr)

        class_bases = call_graph.get('class_bases', {})

        # --- collect qualifying classes (deduplicate by class_usr) ------
        # A constructor qualifies if:
        #   1. Its bare name (before '(' and '<') matches a class name
        #   2. At least one class_usr for that name is in the same file
        #   3. The class already has base classes in class_bases
        #      (classes without bases don't need the logger — the body
        #       instrumentation is sufficient for them)
        qualifying: Dict[str, Dict] = {}

        for fid, meta in func_metadata.items():
            func_name = meta['func_name']
            file_path = meta['file_path']
            file_abs = os.path.abspath(file_path)

            if self._is_whitelisted(file_path):
                continue

            # Extract bare constructor name (strip params and templates).
            bare_func = func_name.split('(')[0].strip()
            bare_func = bare_func.split('<')[0].strip()

            if bare_func not in class_name_to_usrs:
                continue

            # Find the class_usr that lives in the same file.
            class_usr = None
            for candidate_usr in class_name_to_usrs[bare_func]:
                # The constructor USR embeds the class USR, e.g.:
                #   c:@N@...@S@ClassName@F@ClassName#...
                # and the class USR is:
                #   c:@N@...@S@ClassName
                # So check if the func USR starts with the class USR.
                func_usr = None
                for usr, finfo in functions.items():
                    f_file = finfo.get('file', '')
                    if not f_file:
                        continue
                    if (os.path.abspath(f_file) == file_abs
                            and finfo.get('line')
                            == meta.get('start_line')):
                        func_usr = usr
                        break
                if func_usr and func_usr.startswith(candidate_usr + '@'):
                    class_usr = candidate_usr
                    break

            if class_usr is None:
                continue

            # Note: we no longer filter out classes without bases.
            # Classes with no existing bases also need the logger —
            # we'll insert `: public logger_<name>::ConstructorLogger`
            # on the class definition line.

            if class_usr not in qualifying:
                qualifying[class_usr] = {
                    'class_name': bare_func,
                    'file_path': file_path,
                    'func_usr': func_usr,
                    'start_line': meta.get('start_line'),
                }

        # --- Second pass: when increased_traversal_depth is enabled, also
        # qualify every class/struct that already received an operator<<
        # insertion via the ostream plan, even if its constructor never
        # made it into func_metadata (e.g. inherited ctors that clang
        # reports with 0 callees, so BFS never reaches the real body).
        # The class name match in the downstream AST walk is how injection
        # is actually located; func_usr is not consulted there.
        if _increased_traversal:
            ostream_plan_path = os.path.join(
                self.project_path, "ostream_plan.json")
            if os.path.exists(ostream_plan_path):
                try:
                    with open(ostream_plan_path) as _pf:
                        _oplan = json.load(_pf)
                except Exception:
                    _oplan = {}
                _extra_added = 0
                for t in _oplan.get('types', []):
                    if t.get('kind') not in (
                            'class', 'struct', 'class_template'):
                        continue
                    t_usr = t.get('usr', '')
                    t_spelling = t.get('spelling', '')
                    t_file = t.get('file', '')
                    if not (t_usr and t_spelling and t_file):
                        continue
                    if t_usr in qualifying:
                        continue
                    if self._is_whitelisted(t_file):
                        continue
                    qualifying[t_usr] = {
                        'class_name': t_spelling,
                        'file_path': t_file,
                        'func_usr': None,
                        'start_line': t.get('start_line'),
                    }
                    _extra_added += 1
                if _extra_added:
                    print(f"  Step 1c: increased_traversal_depth added "
                          f"{_extra_added} class(es) from ostream plan")

        if not qualifying:
            print("  Step 1c: No qualifying constructors found — skipping")
            return

        print(f"  Step 1c: Injecting ConstructorLogger into "
              f"{len(qualifying)} class(es)...")

        for class_usr, info in qualifying.items():
            class_name = info['class_name']
            file_path = info['file_path']
            logger_base = f'logger_{class_name}::ConstructorLogger'

            # ---- 1. Parse the file with clang to collect AST locations -
            index = cindex.Index.create()
            tu = index.parse(
                file_path,
                args=_get_compile_args(file_path),
                options=cindex.TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD,
            )

            file_abs = os.path.abspath(file_path)

            # Structures to collect from AST walk:
            #   first_base_loc: (line, col) of the first CXX_BASE_SPECIFIER
            #                   None when the class has no bases yet.
            #   class_decl_extent_end: (line, col) of the class name token
            #                          end, used to insert `: base` when
            #                          no bases exist.
            #   ctor_info: list of dicts with keys:
            #     'line'          – constructor decl line
            #     'has_body'      – has COMPOUND_STMT child
            #     'body_line/col' – start of COMPOUND_STMT
            #     'first_init_line/col' – start of first MEMBER_REF or
            #                             TYPE_REF child (init-list item)
            first_base_loc = None  # (line, col) of first base specifier
            class_open_brace = None  # (line, col) of '{' when class has no bases
            ctor_infos: List[Dict] = []

            def _walk(cursor):
                nonlocal first_base_loc, class_open_brace
                if (cursor.location.file and
                        os.path.abspath(str(cursor.location.file))
                        != file_abs):
                    return

                kind = cursor.kind

                if kind in (cindex.CursorKind.CLASS_DECL,
                            cindex.CursorKind.STRUCT_DECL,
                            cindex.CursorKind.CLASS_TEMPLATE):
                    if cursor.spelling == class_name:
                        # Collect first CXX_BASE_SPECIFIER.
                        for child in cursor.get_children():
                            if child.kind == cindex.CursorKind.CXX_BASE_SPECIFIER:
                                loc = child.extent.start
                                if (first_base_loc is None or
                                        (loc.line, loc.column)
                                        < first_base_loc):
                                    first_base_loc = (loc.line, loc.column)
                                break  # only need the first
                        # If class has no bases, record the opening brace
                        # location so we can insert `: public Logger` there.
                        if first_base_loc is None:
                            # Find the '{' that opens the class body.
                            for token in cursor.get_tokens():
                                if token.spelling == '{':
                                    class_open_brace = (
                                        token.extent.start.line,
                                        token.extent.start.column)
                                    break

                if kind == cindex.CursorKind.CONSTRUCTOR:
                    # Template classes report spelling as e.g.
                    # "mmap_input<P, Eol>" instead of "mmap_input".
                    ctor_spelling = cursor.spelling
                    if (ctor_spelling == class_name
                            or ctor_spelling.startswith(class_name + '<')):
                        ci: Dict = {
                            'line': cursor.location.line,
                            'has_body': False,
                            'body_line': None,
                            'body_col': None,
                            'body_end_line': None,
                            'body_end_col': None,
                            'first_init_line': None,
                            'first_init_col': None,
                            'is_deleted': False,
                            'is_defaulted': False,
                            'is_delegating': False,
                        }
                        for child in cursor.get_children():
                            ck = child.kind
                            if ck == cindex.CursorKind.COMPOUND_STMT:
                                ci['has_body'] = True
                                ci['body_line'] = child.extent.start.line
                                ci['body_col'] = child.extent.start.column
                                ci['body_end_line'] = child.extent.end.line
                                ci['body_end_col'] = child.extent.end.column
                            elif ck == cindex.CursorKind.TYPE_REF:
                                # A TYPE_REF that references the same
                                # class means this is a delegating
                                # constructor — C++ forbids mixing
                                # delegation with other initializers.
                                # Use referenced.spelling (unqualified
                                # name) to avoid false matches on base
                                # classes that share a suffix.
                                ref = child.referenced
                                ref_name = (ref.spelling
                                            if ref else None)
                                if ref_name == class_name:
                                    ci['is_delegating'] = True
                                # Also record as first init-list entry.
                                if (child.location.file and
                                        os.path.abspath(
                                            str(child.location.file))
                                        == file_abs):
                                    if ci['first_init_line'] is None:
                                        ci['first_init_line'] = (
                                            child.extent.start.line)
                                        ci['first_init_col'] = (
                                            child.extent.start.column)
                            elif ck == cindex.CursorKind.MEMBER_REF:
                                # Initializer-list entry — keep first.
                                if (child.location.file and
                                        os.path.abspath(
                                            str(child.location.file))
                                        == file_abs):
                                    if ci['first_init_line'] is None:
                                        ci['first_init_line'] = (
                                            child.extent.start.line)
                                        ci['first_init_col'] = (
                                            child.extent.start.column)
                        # Detect = delete / = default by checking extent
                        ext_text_lines = []
                        try:
                            with open(file_path) as _f:
                                all_src = _f.readlines()
                            s = cursor.extent.start.line - 1
                            e = cursor.extent.end.line
                            ext_text_lines = all_src[s:e]
                        except Exception:
                            pass
                        ext_text = ''.join(ext_text_lines)
                        if '= delete' in ext_text:
                            ci['is_deleted'] = True
                        if '= default' in ext_text:
                            ci['is_defaulted'] = True
                        ctor_infos.append(ci)
                        return  # no need to recurse deeper

                for child in cursor.get_children():
                    _walk(child)

            _walk(tu.cursor)

            has_existing_bases = first_base_loc is not None
            if not has_existing_bases and class_open_brace is None:
                print(f"    WARNING: AST found no class definition for "
                      f"{class_name} in {file_path} — skipping")
                continue

            # ---- 2. Read the file and apply text edits -----------------
            with open(file_path, 'r') as f:
                lines = f.readlines()

            # Track cumulative line offset from insertions so AST-derived
            # positions (captured before edits) stay valid.
            line_offset = 0

            # 2a. Ensure <iostream> and <string> are present.
            has_iostream = any('#include <iostream>' in l for l in lines)
            has_string = any('#include <string>' in l for l in lines)
            last_include_idx = -1
            for i, ln in enumerate(lines):
                if ln.strip().startswith('#include'):
                    last_include_idx = i
            insert_at = (last_include_idx + 1) if last_include_idx >= 0 else 0
            extra_includes: List[str] = []
            if not has_string:
                extra_includes.append('#include <string>\n')
            if not has_iostream:
                extra_includes.append('#include <iostream>\n')
            for inc in reversed(extra_includes):
                lines.insert(insert_at, inc)
                line_offset += 1

            # 2b. Insert ConstructorLogger struct inline after includes.
            last_include_idx = -1
            for i, ln in enumerate(lines):
                if ln.strip().startswith('#include'):
                    last_include_idx = i
            logger_block_lines = [
                '\n',
                f'namespace logger_{class_name} {{\n',
                'struct ConstructorLogger\n',
                '{\n',
                '    explicit ConstructorLogger( std::string type_name )\n',
                '    {\n',
                '        {os} << "=================The constructor for " '
                '<< type_name << " begins=================\\n";\n'
                .format(os="std::cerr" if self.output_to_stderr else "std::cout"),
                '    }\n',
                '};\n',
                '}\n',
            ]
            for idx, bl in enumerate(logger_block_lines):
                lines.insert(last_include_idx + 1 + idx, bl)
            line_offset += len(logger_block_lines)

            # 2c. Insert logger base into class definition.
            if has_existing_bases:
                # Insert before the first existing base specifier.
                base_line_idx = first_base_loc[0] - 1 + line_offset
                base_col_idx = first_base_loc[1] - 1  # 0-based
                ln = lines[base_line_idx]
                lines[base_line_idx] = (
                    ln[:base_col_idx] + f'{logger_base}, '
                    + ln[base_col_idx:]
                )
            else:
                # Class has no bases — insert `: public Logger` before '{'.
                brace_line_idx = class_open_brace[0] - 1 + line_offset
                brace_col_idx = class_open_brace[1] - 1
                ln = lines[brace_line_idx]
                lines[brace_line_idx] = (
                    ln[:brace_col_idx]
                    + f': public {logger_base} '
                    + ln[brace_col_idx:]
                )

            # 2d. Modify each non-deleted/defaulted constructor.
            init_text_prefix = f'{logger_base}("{class_name}"), '
            init_text_only = f'{logger_base}("{class_name}") '

            # Process constructors bottom-up so earlier edits don't
            # shift positions of later ones.
            active_ctors = [
                ci for ci in ctor_infos
                if ci['has_body']
                and not ci['is_deleted']
                and not ci['is_defaulted']
                and not ci['is_delegating']
            ]
            active_ctors.sort(key=lambda c: c['line'], reverse=True)

            _os = "std::cerr" if self.output_to_stderr else "std::cout"
            end_msg = (f'{_os} << "=================The constructor'
                       f' for {class_name}'
                       f' ends=================\\n";\n')

            for ci in active_ctors:
                if ci['first_init_line'] is not None:
                    # Has existing initializer list — insert before the
                    # first initializer.
                    il = ci['first_init_line'] - 1 + line_offset
                    ic = ci['first_init_col'] - 1
                    ln = lines[il]
                    # The AST TYPE_REF may point to e.g. "mmap_file"
                    # inside "internal::mmap_file".  Scan backwards
                    # past any chain of  identifier<...>::  or
                    # identifier::  qualifiers (with optional spaces)
                    # so the insertion goes before the whole name.
                    prefix = ln[:ic]
                    m = re.search(
                        r'((?:[A-Za-z_]\w*(?:<[^>]*>)?\s*::\s*)+)$',
                        prefix)
                    if m:
                        ic = m.start(1)
                    lines[il] = (
                        ln[:ic] + init_text_prefix + ln[ic:]
                    )
                else:
                    # No initializer list — insert `: Logger(...) `
                    # before the opening '{' of the body.
                    bl = ci['body_line'] - 1 + line_offset
                    bc = ci['body_col'] - 1
                    ln = lines[bl]
                    lines[bl] = (
                        ln[:bc] + ': ' + init_text_only + ln[bc:]
                    )

                # Insert "ends" message just before the closing '}'
                # of the body.  Done AFTER the init-list edit (which
                # is in-place) so line_offset stays valid.
                if ci['body_end_line'] is not None:
                    el = ci['body_end_line'] - 1 + line_offset
                    ln = lines[el]
                    brace_idx = ln.rfind('}')
                    if brace_idx >= 0:
                        lines[el] = (ln[:brace_idx] + '\n'
                                     + end_msg + ln[brace_idx:])
                        line_offset += 1

            # ---- 3. Snapshot + Write back --------------------------------
            # Keep the pre-Step-1c content so we can roll back if all
            # compile+repair attempts fail, leaving the tree compilable.
            with open(file_path, 'r') as f:
                _pre_step1c_snapshot = f.read()
            with open(file_path, 'w') as f:
                f.writelines(lines)
            print(f"    Modified {file_path}")

            # ---- 4. Compile and check; LLM repair on failure --------
            if os.path.exists(compile_script):
                max_repair_attempts = 3
                repair_agent = None  # lazy init, reused across attempts

                for repair_attempt in range(max_repair_attempts + 1):
                    result = subprocess.run(
                        [compile_script, "native"],
                        cwd=project_abs,
                        capture_output=True, text=True, timeout=300,
                    )
                    if result.returncode == 0:
                        print(f"    Compilation OK for {class_name}")
                        break

                    # Format errors like LLMInstrumentor does
                    formatted_errors = LLMInstrumentor._format_compile_errors(
                        result.stderr, result.stdout,
                        file_path=file_path)

                    if repair_attempt >= max_repair_attempts:
                        print(f"    Compilation FAILED for {class_name} "
                              f"after {max_repair_attempts} repair "
                              f"attempts — rolling back")
                        print(f"    errors:\n{formatted_errors[:500]}")
                        with open(file_path, 'w') as _rf:
                            _rf.write(_pre_step1c_snapshot)
                        print(f"    Rolled back "
                              f"{os.path.basename(file_path)} to "
                              f"pre-step1c state.")
                        break

                    # --- LLM repair attempt ---
                    print(f"    Compilation FAILED for {class_name} "
                          f"(repair attempt {repair_attempt + 1}/"
                          f"{max_repair_attempts})")

                    with open(file_path, 'r') as f:
                        current_code = f.read()

                    if repair_agent is None:
                        from llm.LLMAgent import GeminiAgent, DeepSeekAgent
                        if self.backend == "gemini":
                            repair_agent = GeminiAgent(
                                model="gemini-3-flash-preview",
                                temperature=0,
                                max_tokens=65536,
                                system_prompt="You are an expert C++ programmer.",
                            )
                        else:
                            repair_agent = DeepSeekAgent(
                                model="deepseek-reasoner",
                                temperature=0,
                                max_tokens=8192,
                                system_prompt="You are an expert C++ programmer.",
                            )

                    repair_prompt = f"""Fix the compilation errors in this C++ file.

COMPILATION ERRORS:
{formatted_errors}

FULL FILE ({file_path}):
```cpp
{current_code}
```

IMPORTANT INSTRUCTIONS:
1. The file was modified to add constructor begin/end logging via:
   - A `logger_{class_name}::ConstructorLogger` base class (prints "begins")
   - A `std::cout` line at the end of each constructor body (prints "ends")
2. ONLY fix the constructor instrumentation code (the ConstructorLogger
   inheritance, the ConstructorLogger initializer in constructor init-lists,
   and the "ends" std::cout line). Do NOT modify any other instrumentation
   (@@INST_START_@@, @@FUNC_ID_@@, etc.) or original program logic.
3. If a constructor body is empty (e.g. `{{}}`), make sure the "ends" std::cout
   goes INSIDE the braces, not outside.
4. The ConstructorLogger struct must NOT appear in the constructor's member
   initializer list for delegating constructors (constructors that call
   another constructor of the same class).
5. Return ONLY the complete fixed file. No explanation, no markdown fences,
   no additional text.
"""
                    try:
                        response = repair_agent.get_response(repair_prompt)
                        # Strip markdown fences if present
                        fixed = response.strip()
                        if fixed.startswith('```'):
                            first_nl = fixed.index('\n')
                            fixed = fixed[first_nl + 1:]
                        if fixed.endswith('```'):
                            fixed = fixed[:fixed.rfind('```')]
                        fixed = fixed.strip() + '\n'

                        with open(file_path, 'w') as f:
                            f.write(fixed)
                        print(f"    LLM repair applied to {file_path}")
                    except Exception as e:
                        print(f"    LLM repair failed for "
                              f"{class_name}: {e} — rolling back")
                        with open(file_path, 'w') as _rf:
                            _rf.write(_pre_step1c_snapshot)
                        print(f"    Rolled back "
                              f"{os.path.basename(file_path)} to "
                              f"pre-step1c state.")
                        break

    # ------------------------------------------------------------------
    # Phase 1: Instrumentation (Preprocess + FunctionInstrumentor + LLM)
    # ------------------------------------------------------------------

    def phase1_instrumentation(self, skip_comment_instrumentation: bool = False) -> bool:
        """
        Phase 1: Instrumentation.

        Step 1a: AST analysis + TODO marker insertion (ProjectASTBuilder)
        Step 1b: Replace TODO markers with static code (FunctionInstrumentor),
                 compile after each function, LLM repair on failure.
        """
        print("\n" + "="*80)
        print("PHASE 1: INSTRUMENTATION")
        print("="*80)

        try:

            calls_only = not self.full_instr
            mode_str = "all assignments" if self.full_instr else "function calls only"
            print(f"  Step 1a: Creating TODO instrumentation markers ({mode_str})...")
            modified_files, func_metadata = self._pre_analysze(
                calls_only=calls_only)
            if not modified_files:
                print("Warning: No files were instrumented")
                return False
            print(f"  Step 1a complete: {len(modified_files)} file(s) with TODO markers")

            # ---- Post-Step-1a sanity compile ----
            # TODO comments are just ``/* ... */`` blocks, so they
            # should never break the build.  But when a TODO lands
            # inside a ``#define ... \``-continued macro body, the
            # comment's non-``\``-terminated lines abort macro
            # continuation and the expansion becomes malformed.
            # Detecting this now, before the expensive Step 1b worker
            # compiles, surfaces the problem with an unambiguous
            # error instead of appearing as a generic Step 1b
            # "rollback" on dozens of functions.
            project_abs = os.path.abspath(self.project_path)
            _post1a_script = os.path.join(project_abs, "compile.sh")
            if os.path.exists(_post1a_script):
                print(f"  Step 1a sanity compile: running "
                      f"compile.sh native on TODO-marked sources...")
                _ok1a, _err1a, _ = self._compile(
                    _post1a_script, project_abs, "native")
                if not _ok1a:
                    print(f"  ERROR: Step 1a sanity compile FAILED.")
                    print(f"  The TODO markers broke the build (most "
                          f"likely a TODO landed inside a "
                          f"``#define ... \\`` macro).  Compiler "
                          f"stderr (last 2 KB):")
                    _tail = (_err1a or '')[-2000:]
                    print(_tail)
                    # Drop into pdb so the caller can inspect the
                    # TODO-marked sources and compiler output
                    # before Phase 1 returns.
                    import pdb; pdb.set_trace()
                    return False
                print(f"  Step 1a sanity compile: OK")

            # pdb.set_trace()
            # Step 1b: Replace TODO markers with statically-generated code,
            # compile after each function, fall back to LLM on failure.
            print(f"\n  Step 1b: Generating instrumentation code "
                  f"(static gen, LLM repair on failure)...")
            success = self._instrument_with_static_gen(
                modified_files, func_metadata)

            # Step 1c: Inject ConstructorLogger into classes whose
            # constructors were instrumented. C has no constructors, so
            # this phase is skipped entirely for C projects.
            if func_metadata and not self.is_c_project:
                with open(self.call_graph_path, 'r') as f:
                    call_graph = json.load(f)
                self._inject_constructor_loggers(func_metadata, call_graph)

            # Re-save plan with generated code included
            if func_metadata:
                static_plan_path = os.path.join(
                    self.project_path,
                    "static_instrumentation_plan.json")
                with open(static_plan_path, 'w') as f:
                    json.dump(func_metadata, f, indent=2)
                print(f"  Updated static instrumentation plan with "
                      f"generated code -> {static_plan_path}")

            print(f"\n  Phase 1 complete: {len(modified_files)} file(s) instrumented")
            return success

        except Exception as e:
            print(f"ERROR in Phase 1: {e}")
            import traceback; traceback.print_exc()
            return False

    # ------------------------------------------------------------------
    # Phase 2: Execution
    # ------------------------------------------------------------------

    def _capture_reference_execution(self) -> None:
        """
        Compile + run the pristine (pre-instrumentation) code ONCE for
        both native and wasm, capturing output and exit codes.

        Called from ``run_full_preprocessing`` after Phase 0 and the
        pre-instrumentation snapshot, BEFORE any source file is
        mutated. Results are cached on ``self`` and persisted to
        ``reference_output_<build>.log`` + ``reference_exit_codes.json``
        so Phase 2's verification can compare against them without
        re-running anything.
        """
        import subprocess

        project_abs = os.path.abspath(self.project_path)
        compile_script = os.path.join(project_abs, "compile.sh")
        run_script = os.path.join(project_abs, "run.sh")

        if not os.path.exists(compile_script):
            print(f"  [reference] compile.sh not found; "
                  f"skipping reference capture")
            return
        if not os.path.exists(run_script):
            print(f"  [reference] run.sh not found; "
                  f"skipping reference capture")
            return

        # Resolve fixed-time env_script once.
        env_script = None
        if self.fixed_time:
            env_script = os.path.join(
                os.environ.get('CONDA_PREFIX', ''),
                'etc', 'conda', 'activate.d', 'env_vars.sh')

        print("\n" + "="*80)
        print("REFERENCE EXECUTION CAPTURE (pristine source)")
        print("="*80)

        exit_codes = {}
        for build_type in ("native", "wasm"):
            print(f"\n  [reference] Compiling + running {build_type}...")
            try:
                # Compile
                if self.fixed_time and env_script:
                    cmd = ["bash", "-c",
                           f"source {env_script} 2>/dev/null; "
                           f"{compile_script} {build_type}"]
                else:
                    cmd = [compile_script, build_type]
                res = subprocess.run(
                    cmd, cwd=project_abs,
                    capture_output=True, text=True, timeout=300)
                if res.returncode != 0:
                    print(f"  [reference] {build_type} compile FAILED "
                          f"(exit {res.returncode}); stderr tail:\n"
                          f"{res.stderr[-300:]}")
                    setattr(self, f'_ref_output_{build_type}', None)
                    setattr(self, f'_ref_exit_{build_type}', None)
                    exit_codes[build_type] = None
                    continue

                # Run
                if self.fixed_time and env_script:
                    cmd = ["bash", "-c",
                           f"source {env_script} 2>/dev/null; "
                           f"with_faketime {run_script} {build_type}"]
                else:
                    cmd = [run_script, build_type]
                res = subprocess.run(
                    cmd, cwd=project_abs,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True, errors='surrogateescape',
                    timeout=300)

                out = res.stdout or ""
                setattr(self, f'_ref_output_{build_type}', out)
                setattr(self, f'_ref_exit_{build_type}', res.returncode)
                exit_codes[build_type] = res.returncode

                # Persist output for post-hoc inspection.
                out_path = os.path.join(
                    self.project_path,
                    f"reference_output_{build_type}.log")
                with open(out_path, 'w',
                          errors='surrogateescape') as f:
                    f.write(out)
                print(f"  [reference] {build_type}: exit="
                      f"{res.returncode}, output -> "
                      f"{os.path.basename(out_path)} "
                      f"({len(out)} chars)")
            except subprocess.TimeoutExpired:
                print(f"  [reference] {build_type} TIMEOUT")
                setattr(self, f'_ref_output_{build_type}', None)
                setattr(self, f'_ref_exit_{build_type}', None)
                exit_codes[build_type] = None
            except Exception as _e:
                print(f"  [reference] {build_type} ERROR: {_e}")
                setattr(self, f'_ref_output_{build_type}', None)
                setattr(self, f'_ref_exit_{build_type}', None)
                exit_codes[build_type] = None

        # Persist exit codes as structured metadata.
        try:
            codes_path = os.path.join(
                self.project_path, "reference_exit_codes.json")
            with open(codes_path, 'w') as f:
                json.dump(exit_codes, f, indent=2)
        except Exception as _e:
            print(f"  [reference] warning: could not save "
                  f"reference_exit_codes.json: {_e}")

    @staticmethod
    def _is_gtest_output(text: str) -> bool:
        """Return True if *text* looks like Google Test output."""
        # Google Test prints "[==========]" header and "[ OK ]" / "[ FAILED ]"
        # per test.  Check for the header and at least one result line.
        return ('[==========]' in text
                and ('[ ' in text)
                and ('OK' in text or 'FAILED' in text))

    @staticmethod
    def _extract_test_results(text: str) -> Dict[str, str]:
        """
        Extract per-test results from Google Test output.

        Returns a dict mapping test name -> 'OK' or 'FAILED', e.g.:
            {'chrono_test.format_tm': 'OK', 'chrono_test.weekday': 'FAILED'}

        Returns an empty dict if the output is not in Google Test format.
        """
        if not DiffTraceAnalysis._is_gtest_output(text):
            return {}
        results = {}
        for m in re.finditer(r'^\[\s+(OK|FAILED)\s+\]\s+(\S+)',
                             text, re.MULTILINE):
            status, name = m.group(1), m.group(2)
            results[name] = status
        return results

    def _verify_execution(self, build_type: str) -> bool:
        """
        Verify that the instrumented execution passes the same tests as the
        reference (pristine) code. Uses the cached reference output/exit
        code captured by :meth:`_capture_reference_execution` at the very
        start of ``run_full_preprocessing`` -- no recompilation here.

        For Google Test output: compares per-test OK/FAILED results.
        For other formats: compares exit codes.
        """
        log_path = os.path.join(
            self.project_path,
            f"execution_output_{build_type}.log")
        if not os.path.exists(log_path):
            return True  # nothing to verify

        print(f"\n  Verifying {build_type} test results against "
              f"reference (cached)...")

        ref_output = getattr(self, f'_ref_output_{build_type}', None)
        ref_exit_code = getattr(self, f'_ref_exit_{build_type}', None)

        if ref_output is None:
            # Fall back to on-disk reference log if the in-memory
            # cache is empty (e.g. this object was constructed mid-run).
            _ref_path = os.path.join(
                self.project_path,
                f"reference_output_{build_type}.log")
            if os.path.exists(_ref_path):
                with open(_ref_path, 'r',
                          errors='surrogateescape') as _rf:
                    ref_output = _rf.read()
            _codes_path = os.path.join(
                self.project_path, "reference_exit_codes.json")
            if ref_exit_code is None and os.path.exists(_codes_path):
                try:
                    with open(_codes_path) as _cf:
                        _codes = json.load(_cf)
                    ref_exit_code = _codes.get(build_type)
                except Exception:
                    pass

        if ref_output is None:
            print(f"    WARNING: no reference output captured; "
                  f"skipping verification")
            return True

        with open(log_path, 'r', errors='surrogateescape') as f:
            instr_output = f.read()
        instr_exit_code = getattr(
            self, f'_exec_returncode_{build_type}', None)

        if self._is_gtest_output(ref_output):
            ref_results = self._extract_test_results(ref_output)
            instr_results = self._extract_test_results(instr_output)
            if ref_results == instr_results:
                passed = sum(
                    1 for v in ref_results.values() if v == 'OK')
                failed = sum(
                    1 for v in ref_results.values()
                    if v == 'FAILED')
                print(f"    [gtest] Test results match reference "
                      f"({passed} passed, {failed} failed) -- OK")
                return True
            all_tests = sorted(
                set(ref_results) | set(instr_results))
            print(f"    WARNING: {build_type} test results DIFFER "
                  f"from reference!")
            for test in all_tests:
                ref_s = ref_results.get(test, '(missing)')
                instr_s = instr_results.get(test, '(missing)')
                if ref_s != instr_s:
                    print(f"      {test}: reference={ref_s}, "
                          f"instrumented={instr_s}")
            return False

        # Non-gtest: compare exit codes.
        print(f"    [exit-code] Non-gtest output detected, "
              f"comparing exit codes")
        print(f"    Reference exit code: {ref_exit_code}")
        print(f"    Instrumented exit code: {instr_exit_code}")
        if ref_exit_code is not None and instr_exit_code is not None:
            if ref_exit_code == instr_exit_code:
                print(f"    Exit codes match ({ref_exit_code}) -- OK")
                return True
            print(f"    WARNING: Exit codes differ! "
                  f"ref={ref_exit_code}, "
                  f"instr={instr_exit_code}")
            return False
        print(f"    WARNING: Could not compare exit codes")
        return True

    def phase2_execution(self) -> bool:
        """
        Phase 2: Compile and run native + wasm, collect program states.
        Delegates to :meth:`Preprocess._collect_program_state_no_write`,
        which runs the instrumented binary once, stashes the returncode
        on ``self.preprocessor`` (as ``_exec_returncode_<build>``), and
        writes ``execution_output_<build>.log``.

        Verification compares the captured instrumented output against
        the reference output captured ONCE at run startup (see
        :meth:`_capture_reference_execution`).
        """
        print("\n" + "="*80)
        print("PHASE 2: EXECUTION")
        print("="*80)

        try:
            print("  Running both native and wasm executions...")
            self.states_native = self.preprocessor._collect_program_state_no_write("native")
            self.states_wasm = self.preprocessor._collect_program_state_no_write("wasm")

            if not os.path.exists(self.native_log):
                print(f"  ERROR: Native log not created: {self.native_log}")
                return False
            if not os.path.exists(self.wasm_log):
                print(f"  ERROR: Wasm log not created: {self.wasm_log}")
                return False

            # Mirror the instrumented exit codes from the preprocessor
            # onto ``self`` so ``_verify_execution`` (which reads
            # ``self._exec_returncode_<build>``) sees them.
            for bt in ('native', 'wasm'):
                _rc = getattr(self.preprocessor,
                              f'_exec_returncode_{bt}', None)
                if _rc is not None:
                    setattr(self, f'_exec_returncode_{bt}', _rc)

            # Compare instrumented output against the cached reference.
            self._verify_execution("native")
            self._verify_execution("wasm")

            print(f"\n  Phase 2 complete")
            return True

        except Exception as e:
            print(f"ERROR in Phase 2: {e}")
            import traceback; traceback.print_exc()
            return False

    # ------------------------------------------------------------------
    # Phase 3: Trace Analysis
    # ------------------------------------------------------------------

    def phase3_trace_analysis(self) -> bool:
        """
        Phase 3: Function boundary analysis.

        Finds the root cause function: the function closest to the root problem
        (in call-graph distance) that receives consistent inputs but produces
        inconsistent outputs between native and WASM.
        """
        print("\n" + "="*80)
        print("PHASE 3: FUNCTION BOUNDARY ANALYSIS")
        print("="*80)

        try:
            # 3.1 Load call graph
            print("\n  [3.1] Loading call graph...")
            with open(self.call_graph_path, "r") as f:
                call_graph = json.load(f)
            functions = call_graph.get('functions', {})
            print(f"    Functions: {len(functions)}")
            print(f"    Call edges: {len(call_graph.get('call_edges', {}))}")

            # 3.2 Build marker-to-function mapping
            print("\n  [3.2] Building marker-to-function mapping...")
            marker_func_map = self._build_marker_function_map(call_graph)
            print(f"    Mapped {len(marker_func_map)} markers to functions")
            entry_count = sum(1 for m in marker_func_map.values() if m['type'] == 'entry')
            exit_count = sum(1 for m in marker_func_map.values() if m['type'] == 'exit')
            print(f"    Entry markers: {entry_count}, Exit markers: {exit_count}")

            # 3.2b Build fd metadata for non-deterministic value normalization
            print("\n  [3.2b] Building fd metadata...")
            self._build_fd_metadata(call_graph)

            # 3.3 Parse execution traces (ordered events for stack-based analysis)
            print("\n  [3.3] Parsing execution traces (ordered events)...")
            print(f"    Native log: {self.native_log}")
            native_events = self._parse_execution_events(self.native_log, marker_func_map)
            print(f"    WASM log: {self.wasm_log}")
            wasm_events = self._parse_execution_events(self.wasm_log, marker_func_map)
            print(f"    Native: {len(native_events)} events")
            print(f"    WASM: {len(wasm_events)} events")

            # 3.4 Classify instrumentation quality per function
            print("\n  [3.4] Classifying instrumentation quality...")
            func_quality = self._classify_instrumentation_quality(
                native_events, wasm_events)
            self.func_instrumentation_quality = func_quality

            # Print summary
            counts = {}
            for q in func_quality.values():
                label = q['quality']
                counts[label] = counts.get(label, 0) + 1
            print(f"    Classification results:")
            for label in ('normal', 'soft_rollback', 'hard_rollback'):
                print(f"      {label}: {counts.get(label, 0)}")

            # Save to file
            quality_path = os.path.join(
                self.project_path, "func_instrumentation_quality.json")
            with open(quality_path, 'w') as f:
                json.dump(func_quality, f, indent=2)
            print(f"    Saved to {quality_path}")

            # 3.5 Stack-based root cause detection
            print("\n  [3.5] Stack-based root cause detection...")
            print(f"    Native: {len(native_events)} events, "
                  f"WASM: {len(wasm_events)} events")
            self.root_cause_info = self._find_root_cause_by_stack(
                native_events, wasm_events, call_graph,
                func_quality=func_quality,
            )

            if self.root_cause_info:
                rc = self.root_cause_info
                print(f"\n  {'='*60}")
                print(f"  ROOT CAUSE FUNCTION: {rc['function_name']}")
                print(f"  File: {rc['file_path']}:{rc['line']}-{rc['end_line']}")
                print(f"  Detection method: stack-based execution log tracking")
                if 'stack_depth' in rc:
                    print(f"  Stack depth at detection: {rc['stack_depth']}")
                print(f"  {'='*60}")
                comp = rc['comparison']
                def _safe(s):
                    """Encode surrogates for safe printing."""
                    if isinstance(s, str):
                        return s.encode('utf-8', errors='replace').decode('utf-8')
                    return str(s)
                print(f"  Input (consistent):")
                for i, inp in enumerate(comp['native_inputs']):
                    vals = ", ".join(f"{k}={_safe(v)}" for k, v in inp.items())
                    print(f"    Invocation {i+1}: {vals}")
                print(f"  Output (INCONSISTENT):")
                max_outs = max(len(comp['native_outputs']), len(comp['wasm_outputs']))
                for i in range(max_outs):
                    print(f"    Invocation {i+1}:")
                    if i < len(comp['native_outputs']):
                        vals = ", ".join(f"{k}={_safe(repr(v))}" for k, v in comp['native_outputs'][i].items())
                        print(f"      Native: {vals}")
                    if i < len(comp['wasm_outputs']):
                        vals = ", ".join(f"{k}={_safe(repr(v))}" for k, v in comp['wasm_outputs'][i].items())
                        print(f"      WASM:   {vals}")
            else:
                print("\n  WARNING: No root cause function identified")

            # 3.6 Save results
            print("\n  [3.6] Saving results...")
            self._refresh_root_cause_line_numbers()
            self._save_trace_analysis()
            self._generate_llm_metadata()

            print(f"\n  Phase 3 complete")
            return True

        except Exception as e:
            print(f"ERROR in Phase 3: {e}")
            import traceback; traceback.print_exc()
            return False

    # ------------------------------------------------------------------
    # Pre-instrumentation copy (for Step 4a)
    # ------------------------------------------------------------------

    def _save_pre_instrumentation_copies(self):
        """Save copies of all source files before any instrumentation.

        These copies serve as the baseline for Step 4a, which diffs
        instrumented files against originals to identify and replace
        instrumentation with comment placeholders.

        If ``.pre_instrumentation_originals/`` already exists (from a
        previous run), restore files from there first.  This prevents
        contamination: if the previous run crashed before Phase 4a
        cleanup, the working tree may still contain ``#include
        <iostream>``, ostream operators, or TODO markers from that run.
        Restoring first puts the tree back in a known-clean state.
        """
        self.originals_dir = os.path.join(
            self.project_path, ".pre_instrumentation_originals")

        # Restore from existing originals before overwriting them.
        if os.path.exists(self.originals_dir):
            _restored = 0
            for root, dirs, files in os.walk(self.originals_dir):
                dirs[:] = [d for d in dirs
                           if not d.startswith('.')
                           and d not in ('build', 'build_native',
                                         'build_wasm')]
                for filename in files:
                    orig_path = os.path.join(root, filename)
                    rel_path = os.path.relpath(
                        orig_path, self.originals_dir)
                    target = os.path.join(self.project_path, rel_path)
                    if os.path.exists(target):
                        shutil.copy2(orig_path, target)
                        _restored += 1
            if _restored:
                print(f"  Restored {_restored} file(s) from previous "
                      f".pre_instrumentation_originals "
                      f"(cleanup from prior run)")
            shutil.rmtree(self.originals_dir)

        os.makedirs(self.originals_dir, exist_ok=True)

        source_extensions = {
            '.c', '.cc', '.cpp', '.cxx', '.h', '.hpp', '.hxx'}
        saved_count = 0

        for root, dirs, files in os.walk(self.project_path):
            # Skip hidden dirs, build dirs
            dirs[:] = [d for d in dirs
                       if not d.startswith('.')
                       and d not in ('build', 'build_native', 'build_wasm')]

            for filename in files:
                _, ext = os.path.splitext(filename)
                if ext not in source_extensions:
                    continue

                src_path = os.path.join(root, filename)
                rel_path = os.path.relpath(src_path, self.project_path)
                dst_path = os.path.join(self.originals_dir, rel_path)
                os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                shutil.copy2(src_path, dst_path)
                saved_count += 1

        print(f"  Saved {saved_count} original source file(s) to "
              f"{os.path.relpath(self.originals_dir, self.project_path)}")

    def _save_instrumented_copies(self):
        """Save copies of source files that were actually modified
        during Phase 1 instrumentation.

        Compares each source file against its pre-instrumentation
        original in ``.pre_instrumentation_originals/``.  Only files
        that differ (i.e. have instrumentation added) are copied to
        ``instrumented_files/``.
        """
        instrumented_dir = os.path.join(
            self.project_path, "instrumented_files")
        if os.path.exists(instrumented_dir):
            shutil.rmtree(instrumented_dir)
        os.makedirs(instrumented_dir, exist_ok=True)

        originals_dir = getattr(self, 'originals_dir', None)
        source_extensions = {
            '.c', '.cc', '.cpp', '.cxx', '.h', '.hpp', '.hxx'}
        saved_count = 0

        for root, dirs, files in os.walk(self.project_path):
            dirs[:] = [d for d in dirs
                       if not d.startswith('.')
                       and d not in ('build', 'build_native',
                                     'build_wasm', 'instrumented_files')]

            for filename in files:
                _, ext = os.path.splitext(filename)
                if ext not in source_extensions:
                    continue

                src_path = os.path.join(root, filename)
                rel_path = os.path.relpath(src_path, self.project_path)

                # Only save files that differ from the original.
                if originals_dir:
                    orig_path = os.path.join(originals_dir, rel_path)
                    if os.path.exists(orig_path):
                        with open(src_path, 'rb') as f1, \
                             open(orig_path, 'rb') as f2:
                            if f1.read() == f2.read():
                                continue  # unchanged — skip

                dst_path = os.path.join(instrumented_dir, rel_path)
                os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                shutil.copy2(src_path, dst_path)
                saved_count += 1

        print(f"  Saved {saved_count} instrumented file(s) to "
              f"{os.path.relpath(instrumented_dir, self.project_path)}/")

    # ------------------------------------------------------------------
    # Step 4a: Recover files with instrumentation placeholders
    # ------------------------------------------------------------------

    def phase4a_recover_with_placeholders(self) -> bool:
        """Revert source files to their pre-instrumentation state.

        1. Restores the original, pre-instrumentation version from
           ``.pre_instrumentation_originals/``.
           (Instrumented copies were already saved to
           ``instrumented_files/`` right after Phase 1 by
           ``_save_instrumented_copies``.)
        2. Re-resolves root-cause line numbers against the now-clean files
           and re-saves ``trace_analysis.json``.
        """
        originals_dir = getattr(self, 'originals_dir', None)
        if not originals_dir or not os.path.exists(originals_dir):
            print("  ERROR: Pre-instrumentation originals directory not found")
            return False

        print("\n" + "=" * 80)
        print("STEP 4a: REVERT FILES TO PRE-INSTRUMENTATION STATE")
        print("=" * 80)

        # Collect file pairs: (current_instrumented, original_copy)
        file_pairs = []
        for root, dirs, files in os.walk(originals_dir):
            dirs[:] = [d for d in dirs
                       if not d.startswith('.')
                       and d != 'instrumented_files']
            for filename in files:
                orig_path = os.path.join(root, filename)
                rel_path = os.path.relpath(orig_path, originals_dir)
                instr_path = os.path.join(self.project_path, rel_path)
                if os.path.exists(instr_path):
                    file_pairs.append((instr_path, orig_path, rel_path))

        print(f"  Processing {len(file_pairs)} file(s)...")

        reverted_count = 0
        for instr_path, orig_path, rel_path in file_pairs:
            shutil.copy2(orig_path, instr_path)
            reverted_count += 1

        print(f"  Reverted {reverted_count} file(s) to "
              f"pre-instrumentation state")
        print(f"  Instrumented copies are in: instrumented_files/")

        # 3. Rebuild call graph on the CLEAN source files so that
        #    function file/line info is accurate (definitions in .cc
        #    files, correct line numbers without instrumentation).
        print(f"\n  Rebuilding call graph on clean source files...")
        try:
            if os.path.exists(self.call_graph_path):
                os.remove(self.call_graph_path)
            _cc = getattr(self, '_expanded_compile_commands_path',
                          self.compile_commands_path)
            builder = CallGraphBuilder(
                _cc, parallel=True, max_workers=128)
            builder.build_call_graph()
            builder.export_to_json(self.call_graph_path)
            # Re-apply template discovery so the call graph used for
            # the line-refresh step keeps the edges Phase 0 added.
            _discover_tmpl = False
            try:
                _meta_path = os.path.join(
                    self.project_path, "metadata.json")
                if os.path.exists(_meta_path):
                    with open(_meta_path) as _mf:
                        _mjson = json.load(_mf)
                    _minfo = _mjson.get(
                        "Test Case Failure Info", _mjson)
                    _discover_tmpl = (
                        str(_minfo.get("discover_template", ""))
                        .strip().lower() == "true"
                        or str(_mjson.get("discover_template", ""))
                        .strip().lower() == "true"
                    )
            except Exception:
                pass
            if _discover_tmpl:
                builder.discover_template_dependent_calls()
                builder.export_to_json(self.call_graph_path)
            print(f"  Call graph rebuilt: {self.call_graph_path}")
        except Exception as e:
            print(f"  WARNING: Failed to rebuild call graph: {e}")

        # 4. Re-resolve root cause, suspicious list, and remaining
        #    stack entry line numbers using the fresh call graph.
        if self.root_cause_info:
            self._refresh_root_cause_line_numbers()

        return True

    # ------------------------------------------------------------------
    # Phase 4: Annotate Source Files
    # ------------------------------------------------------------------

   
    # Helper methods
    # ------------------------------------------------------------------

    def _load_program_states(self) -> bool:
        """Load program states from saved JSON files."""
        native_file = os.path.join(self.project_path, "program_states_native.json")
        wasm_file = os.path.join(self.project_path, "program_states_wasm.json")

        if not os.path.exists(native_file) or not os.path.exists(wasm_file):
            return False

        try:
            with open(native_file, 'r') as f:
                self.states_native = self._convert_line_keys(json.load(f))
            with open(wasm_file, 'r') as f:
                self.states_wasm = self._convert_line_keys(json.load(f))
            return True
        except Exception as e:
            print(f"    Error loading states: {e}")
            return False

    def _convert_line_keys(self, states: dict) -> dict:
        """Convert string line keys to integers."""
        converted = {}
        for fpath, line_states in states.items():
            converted[fpath] = {}
            for line_str, info in line_states.items():
                converted[fpath][int(line_str)] = info
        return converted

    # ------------------------------------------------------------------
    # Function I/O analysis helpers (Phase 3 rewrite)
    # ------------------------------------------------------------------

    def _find_function_for_line(self, file_path: str, line: int, functions: Dict) -> Optional[str]:
        """Find the function USR that contains the given file:line (tightest range)."""
        file_path_abs = os.path.abspath(file_path)
        best_usr = None
        best_range = float('inf')

        for usr, func_info in functions.items():
            func_file = func_info.get('file', '')
            if not func_file:
                continue
            if os.path.abspath(func_file) != file_path_abs:
                continue
            start = func_info.get('line', 0)
            end = func_info.get('end_line', 0)
            if start <= line <= end:
                func_range = end - start
                if func_range < best_range:
                    best_range = func_range
                    best_usr = usr

        return best_usr

    def _build_marker_function_map(self, call_graph: Dict) -> Dict:
        """
        Build mapping: marker -> {type, function_usr, function_name, file, line}.

        Classifies each instrumentation marker as ENTRY or EXIT and associates
        it with a function from the call graph.
        """
        with open(self.report_path) as f:
            report = json.load(f)

        functions = call_graph.get('functions', {})
        marker_map = {}

        for point in report.get('instrumentation_points', []):
            file_path = point['file']
            line = point['line']
            instr_text = point.get('instrumentation', '')

            # Determine type from instrumentation text
            if 'Function ENTRY' in instr_text:
                marker_type = 'entry'
            elif 'Function EXIT' in instr_text:
                marker_type = 'exit'
            else:
                continue

            # Extract marker from text
            marker_match = re.search(r'Marker: INST_([a-f0-9]+)', instr_text)
            if not marker_match:
                continue
            marker = marker_match.group(1)

            # Find which function this belongs to
            func_usr = self._find_function_for_line(file_path, line, functions)

            # Fallback: the call graph may record only the *declaration*
            # location (e.g. a header) while the instrumentation was placed
            # in the *definition* (e.g. a .cc file).  Try matching by the
            # function name+signature extracted from the instrumentation text.
            if not func_usr:
                func_usr = self._find_function_by_name_in_report(
                    instr_text, functions)

            if not func_usr:
                continue

            func_info = functions[func_usr]
            marker_map[marker] = {
                'type': marker_type,
                'function_usr': func_usr,
                'function_name': func_info.get('name', 'unknown'),
                'file': file_path,
                'line': line
            }

        return marker_map

    def _find_function_by_name_in_report(self, instr_text: str,
                                          functions: Dict) -> Optional[str]:
        """
        Fallback: resolve a function USR by matching the name+signature in
        the instrumentation text against the call graph.

        This handles the case where the call graph records the *declaration*
        location (e.g. a header file) while the instrumentation was placed
        in the *definition* (e.g. a .cc source file).
        """
        # Extract "FunctionName(params)" from the instrumentation text.
        # Pattern: "Function ENTRY: <signature> at ..." or
        #          "Function EXIT: <signature> at ..."
        sig_match = re.search(
            r'Function (?:ENTRY|EXIT): (.+?) at ', instr_text)
        if not sig_match:
            return None
        target_sig = sig_match.group(1).strip()
        # target_sig looks like "get_locale(const char *, const char *)"
        target_name = target_sig.split('(')[0].strip()

        best_usr = None
        for usr, info in functions.items():
            qname = info.get('qualified_name', '')
            name = info.get('name', '')
            # Exact qualified_name match
            if qname == target_sig:
                return usr
            # Name match (may have multiple overloads — pick first)
            if name == target_name and best_usr is None:
                best_usr = usr
        return best_usr

    def _find_definition_from_report(self, func_name: str) -> Optional[Dict]:
        """
        Find the definition file and **current** line range for a function.

        1. Uses the instrumentation report to identify which file contains the
           definition (handles the header-declaration vs source-definition
           mismatch).
        2. Scans the *current* version of that file to find the function
           signature and traces matching braces to the closing ``}``, so the
           returned line numbers are always consistent with the file on disk
           (even after instrumentation / placeholder modifications that may
           have shifted lines).

        Returns: {'file': str, 'line': int, 'end_line': int} or None.
        """
        if not os.path.exists(self.report_path):
            return None

        with open(self.report_path) as f:
            report = json.load(f)

        # Step 1: Find which file this function was instrumented in.
        file_path = None
        for point in report.get('instrumentation_points', []):
            instr_text = point.get('instrumentation', '')
            sig_match = re.search(
                r'Function (?:ENTRY|EXIT): (.+?) at ', instr_text)
            if not sig_match:
                continue
            if sig_match.group(1).strip() == func_name:
                file_path = point['file']
                break

        if not file_path or not os.path.exists(file_path):
            return None

        # Step 2: Scan the current file for the full function extent.
        return self._find_function_extent_in_file(file_path, func_name)

    def _find_function_extent_in_file(self, file_path: str,
                                       func_name: str,
                                       near_line: int = 0) -> Optional[Dict]:
        """
        Scan the *current* version of *file_path* for a function whose
        name matches *func_name* and return its full extent.

        The search looks for the bare function name (before ``(``) in a
        line that also contains ``{`` (or is followed by one shortly after).
        It then counts braces to find the matching ``}``.

        Args:
            near_line: If > 0, prefer matches closest to this 1-based
                line number.  When multiple functions share the same
                bare name (e.g. ``get()`` in different classes), this
                disambiguates by proximity to the expected location.

        Returns: {'file': str, 'line': int, 'end_line': int} or None.
        """
        bare_name = func_name.split('(')[0].strip()
        # Also strip template args: "basic_format_string<>" -> "basic_format_string"
        bare_name_no_tpl = bare_name.split('<')[0].strip()

        with open(file_path, 'r') as f:
            lines = f.readlines()

        # Pass 1: find ALL candidate function signature lines.
        pattern = r'(?<!\w)(?:\w+::)*' + re.escape(bare_name_no_tpl) + r'\s*(?:<[^>]*>\s*)?\('
        candidates = []
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith('//') or stripped.startswith('#'):
                continue
            if not re.search(pattern, stripped):
                continue
            found_brace = False
            for j in range(i, min(i + 15, len(lines))):
                if '{' in lines[j]:
                    text_so_far = ''.join(lines[i:j+1])
                    semi_pos = text_so_far.find(';')
                    brace_pos = text_so_far.find('{')
                    if semi_pos == -1 or brace_pos < semi_pos:
                        found_brace = True
                        break
            if found_brace:
                candidates.append(i)

        if not candidates:
            return None

        # Pick the best candidate.
        if near_line > 0 and len(candidates) > 1:
            # Prefer the candidate closest to near_line.
            sig_line_idx = min(candidates,
                               key=lambda c: abs(c - (near_line - 1)))
        else:
            sig_line_idx = candidates[0]

        if sig_line_idx is None:
            return None

        # Pass 2: find the matching closing brace.
        brace_depth = 0
        end_line_idx = sig_line_idx
        started = False
        for i in range(sig_line_idx, len(lines)):
            for ch in lines[i]:
                if ch == '{':
                    brace_depth += 1
                    started = True
                elif ch == '}':
                    brace_depth -= 1
            if started and brace_depth <= 0:
                end_line_idx = i
                break

        return {
            'file': file_path,
            'line': sig_line_idx + 1,      # 1-indexed
            'end_line': end_line_idx + 1,   # 1-indexed
        }

    def _parse_execution_events(self, log_path: str, marker_func_map: Dict) -> List[Dict]:
        """
        Parse execution log into a chronologically-ordered list of events.

        Each event dict has:
            marker:     8-char hex marker
            func_id:    function identifier from @@FUNC_ID_...@@ line (or None)
            func_name:  function name extracted from FUNC_ID (or from marker_func_map)
            func_usr:   function USR from marker_func_map (or None)
            event_type: 'entry' or 'exit'
            values:     dict of key=value pairs (excluding header lines)

        Returns: ordered list of event dicts
        """
        with open(log_path, errors='surrogateescape') as f:
            raw_content = f.read()

        # Filter to test block if present
        content = raw_content
        content_offset = 0  # character offset of content within raw_content
        test_start = raw_content.find("@@TEST BLOCK START@@")
        test_end = raw_content.find("@@TEST BLOCK END@@")
        if test_start != -1 and test_end != -1 and test_end > test_start:
            content_offset = test_start + len("@@TEST BLOCK START@@")
            content = raw_content[content_offset:test_end]

        # Pre-compute line number lookup from the raw file content
        # (cumulative newline count → 1-indexed line number).
        _newline_offsets = [0]
        for i, ch in enumerate(raw_content):
            if ch == '\n':
                _newline_offsets.append(i + 1)

        def _offset_to_line(offset):
            """Convert a character offset in *raw_content* to a 1-indexed line number."""
            import bisect
            return bisect.bisect_right(_newline_offsets, offset)

        pattern = r'@@INST_START_([a-f0-9]{8})@@(.*?)@@INST_END_\1@@'
        events = []

        for match in re.finditer(pattern, content, re.DOTALL):
            marker = match.group(1)
            block_content = match.group(2).strip()
            # Line number of the @@INST_START_...@@ line in the log file
            log_line = _offset_to_line(content_offset + match.start())

            lines = block_content.split('\n')

            # Extract FUNC_ID if present: @@FUNC_ID_{name}_{hash}@@
            func_id = None
            func_id_name = None
            event_type = None
            values = {}

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                # Parse @@FUNC_ID_{name}_{hash}@@
                fid_match = re.match(r'^@@FUNC_ID_(.+)_([a-f0-9]{8})@@$', line)
                if fid_match:
                    func_id_name = fid_match.group(1)
                    func_id = f"{func_id_name}_{fid_match.group(2)}"
                    continue

                # Determine entry/exit
                if line == '+++Below are Input+++':
                    event_type = 'entry'
                    continue
                if line == '---Below are Outputs---':
                    event_type = 'exit'
                    continue

                # Parse key=value
                if '=' in line:
                    key, _, val = line.partition('=')
                    values[key.strip()] = val.strip()

            # Clean up fragment keys from broken struct values.
            # Binary data in struct members can contain newlines that
            # split the struct across parser lines, creating phantom
            # keys with non-printable chars or leading commas.
            _clean_values = {}
            _last_struct_key = None
            for k in list(values.keys()):
                # Detect fragment: key has control chars, surrogates,
                # or starts with comma (continuation of a struct).
                _is_fragment = (
                    any(ord(c) < 0x20 or ord(c) == 0x7F
                        for c in k)
                    or k.lstrip().startswith(',')
                )
                if _is_fragment and _last_struct_key:
                    # Append fragment back to the last struct value.
                    _clean_values[_last_struct_key] += (
                        ', ' + k.strip().lstrip(',').strip()
                        + '=' + values[k])
                elif _is_fragment:
                    # No preceding struct to attach to — drop it.
                    pass
                else:
                    _clean_values[k] = values[k]
                    # Track if this value looks like a struct.
                    if re.match(r'^\w+\{', values[k]):
                        _last_struct_key = k
                    else:
                        _last_struct_key = None
            values = _clean_values

            # Fallback: use marker_func_map if FUNC_ID not in log
            func_usr = None
            func_name = func_id_name or 'unknown'
            if marker in marker_func_map:
                info = marker_func_map[marker]
                func_usr = info['function_usr']
                if not func_id_name:
                    func_name = info['function_name']
                if event_type is None:
                    event_type = info['type']

            if event_type is None:
                continue  # skip unrecognised blocks

            events.append({
                'marker': marker,
                'func_id': func_id,
                'func_name': func_name,
                'func_usr': func_usr,
                'event_type': event_type,
                'values': values,
                'log_line': log_line,
                'content_offset': content_offset + match.start(),
            })

        # Also detect constructor logger messages (not inside INST blocks).
        ctor_pattern = re.compile(
            r'=================The constructor for (\S+) '
            r'(begins|ends)=================')
        for m in ctor_pattern.finditer(content):
            ctor_name = m.group(1)
            ctor_type = 'ctor_entry' if m.group(2) == 'begins' else 'ctor_exit'
            offset = content_offset + m.start()
            events.append({
                'marker': None,
                'func_id': None,
                'func_name': f'{ctor_name}::<constructor>',
                'func_usr': None,
                'event_type': ctor_type,
                'values': {},
                'log_line': _offset_to_line(offset),
                'content_offset': offset,
            })

        # Sort all events by their position in the log file.
        events.sort(key=lambda e: e['content_offset'])
        for e in events:
            del e['content_offset']

        return events

    def _event_match_key(self, evt):
        """
        Build a hash key for matching events across native/WASM logs.

        Key = (func_name, event_type, normalized_values) where:
          - Memory addresses (0x...) are replaced with a sentinel '<ptr>'
            so that any address matches any other address.
          - [unprintable] values are replaced with '<unprintable>' so they
            still occupy the key slot (preventing mismatches when one side
            has more keys than the other).
          - File descriptor return values from fd-producing functions are
            replaced with '<fd>' (only non-negative values; negative
            values like -1 are error sentinels and compared literally).
          - Embedded fd member values in struct ostream representations
            (e.g. ``m_fd=3``) are replaced with ``m_fd=<fd>``.
          - All other values are kept as-is for exact matching.
        """
        func_name = evt.get('func_name', 'unknown')
        evt_type = evt['event_type']

        # Determine whether this function returns a file descriptor.
        is_fd_func = False
        fd_ids = getattr(self, '_fd_returning_func_ids', None)
        if fd_ids:
            func_id_raw = evt.get('func_id')
            if func_id_raw:
                fid_hash = func_id_raw.rsplit('_', 1)[-1]
                is_fd_func = fid_hash in fd_ids

        normalized = []
        for k, v in sorted(evt['values'].items()):
            # Normalize null bytes: native prints \u0000 (null char)
            # while WASM prints empty string.  Strip all null bytes
            # so they compare as equivalent.
            v = v.replace('\x00', '')

            # Treat build_native/ and build_wasm/ path fragments as
            # equivalent. These only differ because the two sides run
            # in separate build directories; the program logic that
            # produced the path is identical. Without this the first
            # call that records a path (e.g. argv[0] of a test binary)
            # would spuriously mark itself as the root cause.
            v = re.sub(r'build_(?:native|wasm)', 'build_<side>', v)

            # Check pointer/object/struct BEFORE surrogates.  A struct
            # representation like "scan_buffer{ ptr_=\x01|\udcae..."
            # may contain binary data that triggers the surrogate check,
            # but it's still an object — both sides should normalize to
            # <ptr>.  Also handle PARTIAL struct reprs (no closing })
            # which happen when binary data in a member truncates the
            # output or confuses the key=value parser.
            if self._is_pointer_or_object(v):
                normalized.append((k, '<ptr>'))
                continue
            # Partial struct: starts with "TypeName{" but no closing }.
            if re.match(r'^\w+\{', v.strip()):
                normalized.append((k, '<ptr>'))
                continue

            # Values containing surrogates (from native raw bytes) or
            # Unicode replacement characters U+FFFD (from WASM/Emscripten)
            # came from raw bytes that are not valid UTF-8 (e.g. uint8_t
            # printed as a char).  Native and WASM encode these
            # differently, so skip comparison.
            _has_surrogates = False
            try:
                v.encode('utf-8')
            except UnicodeEncodeError:
                _has_surrogates = True
            if _has_surrogates or '\ufffd' in v:
                normalized.append((k, '<non-utf8>'))
                continue

            if v == '[unprintable]':
                normalized.append((k, '<unprintable>'))
            elif is_fd_func and k == 'ret' and evt_type == 'exit':
                # fd return value — normalize only non-negative values.
                # Negative values (e.g. -1) are error sentinels that
                # should be compared literally.
                try:
                    if int(v) >= 0:
                        normalized.append((k, '<fd>'))
                    else:
                        normalized.append((k, v))
                except ValueError:
                    normalized.append((k, v))
            elif self._FD_NAME_RE.match(k):
                # Parameter / struct-field name looks like a file
                # descriptor (fd, fd_, m_fd, file_descriptor, socket_fd,
                # …).  Native vs WASM assign unrelated fd values, so
                # compare only the sign: non-negative → <fd>,
                # negative → literal (it's an error sentinel).
                try:
                    if int(v) >= 0:
                        normalized.append((k, '<fd>'))
                    else:
                        normalized.append((k, v))
                except ValueError:
                    normalized.append((k, v))
            else:
                # Replace embedded hex addresses and embedded fd members.
                nv = self._normalize_addresses(v)
                nv = self._normalize_fd_in_value(nv)
                normalized.append((k, nv))
        return (func_name, evt_type, tuple(normalized))

    def _find_root_cause_by_stack(self, native_events: List[Dict],
                                   wasm_events: List[Dict],
                                   call_graph: Dict,
                                   func_quality: Optional[Dict] = None,
                                   ) -> Optional[Dict]:
        """
        Hash-based matching + stack root cause detection.

        Algorithm:
          1. Build a hash index of all WASM events by (func_name, event_type,
             comparable_values).  Track which WASM events have been "used".
          2. Walk native_events in order:
             - ENTRY: find a matching unused WASM entry (same func_name, same
               comparable input values).  If found → mark used, push onto
               stack.  If NOT found → top of stack is root cause (or unit
               test if stack empty).
             - EXIT: find a matching unused WASM exit (same func_name, same
               comparable output values).  If found → mark used, pop from
               stack (warn if func_name mismatches stack top).  If NOT
               found → top of stack is root cause.

        Also maintains a *suspicious_func_list* tracking functions whose
        instrumentation may be unreliable:
          - hard_rollback functions are always suspicious.
          - soft_rollback functions are added when pushed; removed when their
            guardian (the stack entry directly below at push time) is
            successfully popped.

        Returns a root_cause_info dict or None.
        """
        functions = call_graph.get('functions', {})
        from collections import defaultdict

        # Pre-compute the "no root cause" prefix once. Used by the two
        # stack-empty branches below; refers to the test case via the
        # file_path + line range recorded in metadata.json so consumers
        # know exactly where to start debugging.
        _tc_meta = self.preprocessor.metadata.get(
            "Test Case Failure Info", {})
        _tc_file = _tc_meta.get("file_path", "(unknown)")
        _tc_start = _tc_meta.get("test case start line", "?")
        _tc_end = _tc_meta.get("test case end line", "?")
        _no_root_prefix = (
            f"No root cause found, the discrepancy may be caused by "
            f"the test case at {_tc_file} from line {_tc_start} "
            f"to line {_tc_end}. Report Reason: ")

        # ── Build hash index of WASM events ───────────────────────────────
        # key → list of wasm event indices (in order), so we always pick the
        # earliest unused match.
        wasm_index = defaultdict(list)
        wasm_used = [False] * len(wasm_events)

        for idx, evt in enumerate(wasm_events):
            key = self._event_match_key(evt)
            wasm_index[key].append(idx)

        print(f"    [matching] Built WASM hash index: {len(wasm_events)} events, "
              f"{len(wasm_index)} unique keys")

        # ── Name-only index for diagnosing value-level discrepancies ──────
        # (func_name, event_type) → list of wasm event indices.  Used only
        # when the hash-key match fails, to distinguish "function absent
        # from WASM" from "function present but with different values".
        wasm_name_index = defaultdict(list)
        for idx, evt in enumerate(wasm_events):
            _name_key = (evt.get('func_name', 'unknown'),
                         evt.get('event_type'))
            wasm_name_index[_name_key].append(idx)

        def _find_name_only_wasm_match(n_evt):
            """Return the index of the earliest wasm event with the same
            func_name and event_type as *n_evt*, regardless of values.
            Used for diagnostics only — does not mark the wasm event as
            consumed.  Returns None if no name match exists at all.
            """
            key = (n_evt.get('func_name', 'unknown'),
                   n_evt['event_type'])
            candidates = wasm_name_index.get(key, [])
            for wi in candidates:
                if not wasm_used[wi]:
                    return wi
            # All matches are already consumed: still return the first
            # one — it's informative even if it's been used.
            return candidates[0] if candidates else None

        def _build_discrepancy_reason(n_evt, native_values):
            """Build the discrepancy_reason string for a native event
            that has no hash-key match in WASM.  Distinguishes the two
            cases the user wants to see separately.
            """
            func_name = n_evt.get('func_name', 'unknown')
            evt_type = n_evt['event_type']
            evt_label = 'ENTRY' if evt_type in ('entry', 'ctor_entry') else 'EXIT'

            wi = _find_name_only_wasm_match(n_evt)
            if wi is None:
                return (f"'{func_name}': NO MATCHING WASM {evt_label} "
                        f"(function not observed in WASM, likely due to WASM crash).")

            w_values = wasm_events[wi].get('values', {}) or {}
            n_values = native_values or {}
            diffs = []
            all_keys = sorted(set(n_values) | set(w_values))
            for k in all_keys:
                nv = n_values.get(k, '<missing>')
                wv = w_values.get(k, '<missing>')
                if nv != wv:
                    diffs.append(f"{k}: native={nv!r} vs wasm={wv!r}")

            if diffs:
                diff_str = "; ".join(diffs[:5])
                if len(diffs) > 5:
                    diff_str += f"; ... (+{len(diffs) - 5} more)"
                return (f"'{func_name}': WASM {evt_label} FOUND BUT "
                        f"VALUES DIFFER ({diff_str})")
            # No per-key diffs found (rare: normalization stripped the
            # difference but the hash key still differed).
            return (f"'{func_name}': WASM {evt_label} FOUND BUT "
                    f"NORMALIZED KEY DIFFERS")

        # ── Open stack trace log ─────────────────────────────────────────
        stack_trace_path = os.path.join(
            self.project_path, "stack_trace.log")
        stack_trace_f = open(stack_trace_path, 'w', errors='surrogateescape')

        def _stack_repr(stk):
            """Readable representation of the current stack."""
            if not stk:
                return "(empty)"
            if len(stk) == 1:
                return f"[0] {stk[0]['func_name']}"
            parts = []
            for idx, s in enumerate(stk):
                parts.append(f"[{idx}] {s['func_name']}")
            return " , ".join(parts)

        def _log_stack(msg):
            stack_trace_f.write(msg + '\n')

        _log_stack(f"# Stack trace log — generated by phase3_trace_analysis")
        _log_stack(f"# native log: {self.native_log}")
        _log_stack(f"# wasm log:   {self.wasm_log}")
        _log_stack("")

        # ── Suspicious function list ──────────────────────────────────────
        # Maps func_id → {func_name, quality, reason} for suspicious funcs.
        # Hard rollbacks are always suspicious; soft rollbacks are added
        # when pushed and removed when their guardian is successfully popped.
        if func_quality is None:
            func_quality = {}

        # Build quality lookup by func_name for matching against events
        quality_by_func_name = {}
        for fid, q in func_quality.items():
            quality_by_func_name[q['func_name']] = q

        suspicious_func_list = {}  # func_id → info dict
        for fid, q in func_quality.items():
            if q['quality'] == 'hard_rollback':
                suspicious_func_list[fid] = {
                    'func_name': q['func_name'],
                    'quality': 'hard_rollback',
                    'reason': 'instrumentation failed (rolled back)',
                }

        # guardian_id → [(fid, func_name)] : soft_rollback entries to remove
        # when the guardian stack entry is popped.
        guardian_removals = defaultdict(list)
        stack_entry_counter = 0

        _log_stack("# ── Suspicious function list (initial) ──")
        _log_stack(f"# {len(suspicious_func_list)} hard_rollback function(s):")
        for fid, info in suspicious_func_list.items():
            _log_stack(f"#   [{fid}] {info['func_name']}")
        _log_stack("")

        def _log_suspicious():
            if suspicious_func_list:
                _log_stack(f"  suspicious_func_list ({len(suspicious_func_list)}): "
                           + ", ".join(f"'{v['func_name']}' ({v['quality']})"
                                      for v in suspicious_func_list.values()))
            else:
                _log_stack(f"  suspicious_func_list: (empty)")

        def _push_stack_entry(func_name, func_usr, native_input,
                              wasm_input, func_id=None):
            """Push a new entry onto the stack and handle soft_rollback tracking."""
            nonlocal stack_entry_counter
            stack_entry_counter += 1
            eid = stack_entry_counter
            guardian_id = stack[-1]['entry_id'] if stack else None
            stack.append({
                'func_name': func_name,
                'func_usr': func_usr,
                'func_id': func_id,
                'native_input': native_input,
                'wasm_input': wasm_input,
                'entry_id': eid,
            })
            # Check if this function is soft_rollback
            q = quality_by_func_name.get(func_name)
            if q and q['quality'] == 'soft_rollback':
                # Find the func_id for this function
                fid_match = None
                for fid, qi in func_quality.items():
                    if qi['func_name'] == func_name:
                        fid_match = fid
                        break
                if fid_match and fid_match not in suspicious_func_list:
                    suspicious_func_list[fid_match] = {
                        'func_name': func_name,
                        'quality': 'soft_rollback',
                        'reason': 'missing variables: '
                                  + ', '.join(q.get('missing_vars', [])),
                    }
                    _log_stack(f"  +suspicious: added '{func_name}' "
                               f"(soft_rollback, guardian_id={guardian_id})")
                    if guardian_id is not None:
                        guardian_removals[guardian_id].append(
                            (fid_match, func_name))
                    else:
                        _log_stack(f"    (no guardian — will never be removed)")
                    _log_suspicious()

        def _pop_stack_entry():
            """Pop the top stack entry and handle guardian removal."""
            if not stack:
                return None
            popped = stack.pop()
            popped_id = popped['entry_id']
            # Check if any soft_rollback entries had this as guardian
            if popped_id in guardian_removals:
                for fid, fname in guardian_removals[popped_id]:
                    if fid in suspicious_func_list:
                        del suspicious_func_list[fid]
                        _log_stack(f"  -suspicious: removed '{fname}' "
                                   f"(guardian '{popped['func_name']}' "
                                   f"popped successfully)")
                del guardian_removals[popped_id]
                _log_suspicious()
            return popped

        # ── Walk native events, match against WASM ────────────────────────
        from tqdm import tqdm
        stack = []  # list of {func_name, func_usr, native_input, wasm_input,
                     #          entry_id}
        root_cause_comparison = None
        last_matched_wasm_idx = None  # wasm event index of most recent match

        def _find_wasm_output_from(rc_func_name):
            """Scan wasm_events forward from last_matched_wasm_idx for
            the first exit event of *rc_func_name*.

            Returns the event's values dict, or {} if not found.
            """
            if last_matched_wasm_idx is None:
                _log_stack("  [wasm output search] no previous matched "
                           "wasm event — cannot find wasm output")
                return {}
            start = last_matched_wasm_idx + 1
            w_log_ln = wasm_events[last_matched_wasm_idx].get(
                'log_line', '?')
            _log_stack(f"  [wasm output search] scanning wasm events "
                       f"from index {start} (after wasm log L{w_log_ln}) "
                       f"for exit of '{rc_func_name}'")
            for j in range(start, len(wasm_events)):
                w = wasm_events[j]
                if (w['event_type'] == 'exit'
                        and w.get('func_name') == rc_func_name):
                    found_ln = w.get('log_line', '?')
                    _log_stack(f"  [wasm output search] FOUND at wasm "
                               f"event index {j}, wasm log L{found_ln}")
                    _log_stack(f"  [wasm output search] values: "
                               f"{w['values']}")
                    return w['values']
            _log_stack(f"  [wasm output search] NOT FOUND "
                       f"(scanned {len(wasm_events) - start} events)")
            return {}

        def _find_native_output_from(rc_func_name, current_idx):
            """Scan native_events forward from *current_idx* for the
            first exit event of *rc_func_name* (the root-cause function).

            When a callee's event triggers root-cause detection, the
            mismatched event belongs to the callee, not the root cause.
            This helper finds the root-cause function's **own** exit
            event so its output values can be reported correctly.

            Returns the event's values dict, or {} if not found.
            """
            start = current_idx + 1
            _log_stack(f"  [native output search] scanning native events "
                       f"from index {start} for exit of '{rc_func_name}'")
            for j in range(start, len(native_events)):
                n = native_events[j]
                if (n['event_type'] == 'exit'
                        and n.get('func_name') == rc_func_name):
                    found_ln = n.get('log_line', '?')
                    _log_stack(f"  [native output search] FOUND at native "
                               f"event index {j}, native log L{found_ln}")
                    _log_stack(f"  [native output search] values: "
                               f"{n['values']}")
                    return n['values']
            _log_stack(f"  [native output search] NOT FOUND "
                       f"(scanned {len(native_events) - start} events)")
            return {}

        for i, n_evt in enumerate(tqdm(native_events,
                                       desc="    [stack] Analyzing",
                                       unit="evt", leave=True)):
            func_name = n_evt.get('func_name', 'unknown')
            func_usr = n_evt.get('func_usr')
            evt_type = n_evt['event_type']

            # Find first unused WASM event with the same hash key
            key = self._event_match_key(n_evt)
            match_idx = None
            if key in wasm_index:
                for wasm_idx in wasm_index[key]:
                    if not wasm_used[wasm_idx]:
                        match_idx = wasm_idx
                        wasm_used[wasm_idx] = True
                        break

            _log_ln = n_evt.get('log_line', '?')

            # ── Constructor logger events ──────────────────────────
            if evt_type == 'ctor_entry':
                if match_idx is not None:
                    last_matched_wasm_idx = match_idx
                    _wasm_ln = wasm_events[match_idx].get('log_line', '?')
                    _push_stack_entry(func_name, func_usr, {}, {})
                    _log_stack(
                        f"[native log L{_log_ln}] PUSH  "
                        f"'{func_name}' (ctor begin matched "
                        f"@ wasm log L{_wasm_ln})")
                    _log_stack(f"  stack: {_stack_repr(stack)}")
                else:
                    _log_stack(
                        f"[native log L{_log_ln}] PUSH  "
                        f"'{func_name}' (ctor begin, no WASM match)")
                    _push_stack_entry(func_name, func_usr, {}, {})
                    _log_stack(f"  stack: {_stack_repr(stack)}")
                continue

            if evt_type == 'ctor_exit':
                if match_idx is not None:
                    # Matching WASM ctor end found → normal pop.
                    last_matched_wasm_idx = match_idx
                    _wasm_ln = wasm_events[match_idx].get('log_line', '?')
                    if stack and stack[-1]['func_name'] == func_name:
                        _pop_stack_entry()
                        _log_stack(
                            f"[native log L{_log_ln}] POP   "
                            f"'{func_name}' (ctor end matched "
                            f"@ wasm log L{_wasm_ln})")
                    else:
                        _log_stack(
                            f"[native log L{_log_ln}] POP   "
                            f"'{func_name}' (ctor end matched "
                            f"@ wasm log L{_wasm_ln}, top was "
                            f"'{stack[-1]['func_name'] if stack else '(empty)'}')")
                        if stack:
                            _pop_stack_entry()
                    _log_stack(f"  stack: {_stack_repr(stack)}")
                    continue
                else:
                    # No matching WASM ctor end → top of stack is root cause.
                    _log_stack(
                        f"[native log L{_log_ln}] *** CTOR EXIT "
                        f"'{func_name}': NO MATCHING WASM CTOR EXIT")
                    _log_stack(f"  stack: {_stack_repr(stack)}")
                    print(f"    [stack] CTOR EXIT '{func_name}' "
                          f"(native event {i}, log line {_log_ln}): "
                          f"no matching WASM ctor end found")
                    _reason = (
                        f"'{func_name}': native emitted the constructor "
                        f"end marker but WASM did not (WASM likely crashed "
                        f"or terminated inside this constructor)")
                    _log_stack(f"  discrepancy_reason: {_reason}")
                    if stack:
                        top = stack[-1]
                        _log_stack(
                            f"  => ROOT CAUSE: '{top['func_name']}' "
                            f"(top of stack)")
                        print(f"    [stack] Root cause: top of stack "
                              f"'{top['func_name']}'")
                        native_output = _find_native_output_from(
                            top['func_name'], i)
                        wasm_output = _find_wasm_output_from(
                            top['func_name'])
                        root_cause_comparison = {
                            'func_name': top['func_name'],
                            'func_usr': top['func_usr'],
                            'func_id': top.get('func_id'),
                            'native_input': top['native_input'],
                            'wasm_input': top['wasm_input'],
                            'native_output': native_output,
                            'wasm_output': wasm_output,
                            'discrepancy_reason': _reason,
                        }
                    else:
                        _prefixed = f"{_no_root_prefix}\n{_reason}"
                        root_cause_comparison = {
                            'func_name': '',
                            'func_usr': '',
                            'func_id': '',
                            'native_input': {},
                            'wasm_input': {},
                            'native_output': {},
                            'wasm_output': {},
                            'discrepancy_reason': _prefixed,
                        }
                        _log_stack(
                            f"  => Stack empty — problem likely in "
                            f"the unit test script")
                        _log_stack(
                            f"  discrepancy_reason (prefixed): "
                            f"{_prefixed}")
                        print(f"    [stack] Stack is empty — problem likely "
                              f"in the unit test script")
                    break

            if evt_type == 'entry':
                if match_idx is not None:
                    # Matching WASM entry found → push
                    last_matched_wasm_idx = match_idx
                    w_evt = wasm_events[match_idx]
                    _wasm_ln = w_evt.get('log_line', '?')
                    _push_stack_entry(
                        func_name,
                        func_usr or w_evt.get('func_usr'),
                        n_evt['values'],
                        w_evt['values'],
                        func_id=n_evt.get('func_id'),
                    )
                    _log_stack(
                        f"[native log L{_log_ln}] PUSH  "
                        f"'{func_name}' (entry matched "
                        f"@ wasm log L{_wasm_ln})")
                    _log_stack(f"  stack: {_stack_repr(stack)}")
                else:
                    # No matching WASM entry → root cause
                    _log_stack(
                        f"[native log L{_log_ln}] *** ENTRY "
                        f"'{func_name}': NO MATCHING WASM ENTRY")
                    _log_stack(f"  native values: {n_evt['values']}")
                    _log_stack(f"  stack: {_stack_repr(stack)}")
                    print(f"    [stack] ENTRY '{func_name}' (native event {i}, "
                          f"log line {_log_ln}): no matching WASM entry found")
                    print(f"             native values: {n_evt['values']}")
                    _reason = _build_discrepancy_reason(n_evt, n_evt['values'])
                    _log_stack(f"  discrepancy_reason: {_reason}")
                    if stack:
                        top = stack[-1]
                        _log_stack(
                            f"  => ROOT CAUSE: '{top['func_name']}' "
                            f"(top of stack)")
                        print(f"    [stack] Root cause: top of stack "
                              f"'{top['func_name']}'")
                        # Find the root-cause function's OWN output
                        # (not the callee's values that triggered the
                        # mismatch).
                        native_output = _find_native_output_from(
                            top['func_name'], i)
                        wasm_output = _find_wasm_output_from(
                            top['func_name'])
                        root_cause_comparison = {
                            'func_name': top['func_name'],
                            'func_usr': top['func_usr'],
                            'func_id': top.get('func_id'),
                            'native_input': top['native_input'],
                            'wasm_input': top['wasm_input'],
                            'native_output': native_output,
                            'wasm_output': wasm_output,
                            'discrepancy_reason': _reason,
                        }
                    else:
                        # Stack empty: no caller function to pin the
                        # discrepancy on. Leave every root-cause field
                        # empty; only discrepancy_reason carries the
                        # diagnosis (prefixed with an explicit "no
                        # root cause" note so consumers know this
                        # mismatch lives outside the instrumented
                        # call chain, usually the test scaffold).
                        _prefixed = f"{_no_root_prefix}\n{_reason}"
                        root_cause_comparison = {
                            'func_name': '',
                            'func_usr': '',
                            'func_id': '',
                            'native_input': {},
                            'wasm_input': {},
                            'native_output': {},
                            'wasm_output': {},
                            'discrepancy_reason': _prefixed,
                        }
                        _log_stack(
                            f"  => Stack empty — problem likely in "
                            f"the unit test script")
                        _log_stack(
                            f"  discrepancy_reason (prefixed): "
                            f"{_prefixed}")
                        print(f"    [stack] Stack is empty — problem likely "
                              f"in the unit test script")
                    break

            elif evt_type == 'exit':
                if match_idx is not None:
                    # Matching WASM exit found → pop
                    last_matched_wasm_idx = match_idx
                    _wasm_ln = wasm_events[match_idx].get('log_line', '?')
                    if stack:
                        top_name = stack[-1]['func_name']
                        if top_name != func_name:
                            _log_stack(
                                f"[native log L{_log_ln}] POP   "
                                f"'{func_name}' (exit matched "
                                f"@ wasm log L{_wasm_ln}) "
                                f"WARNING: top was '{top_name}'")
                            print(f"    [stack] WARNING: popping EXIT "
                                  f"'{func_name}' but top of stack is "
                                  f"'{top_name}'")
                        else:
                            _log_stack(
                                f"[native log L{_log_ln}] POP   "
                                f"'{func_name}' (exit matched "
                                f"@ wasm log L{_wasm_ln})")
                        _pop_stack_entry()
                    else:
                        _log_stack(
                            f"[native log L{_log_ln}] POP   "
                            f"'{func_name}' (exit matched "
                            f"@ wasm log L{_wasm_ln}, "
                            f"stack already empty)")
                    _log_stack(f"  stack: {_stack_repr(stack)}")
                else:
                    # No matching WASM exit → top of stack is root cause
                    _log_stack(
                        f"[native log L{_log_ln}] *** EXIT "
                        f"'{func_name}': NO MATCHING WASM EXIT")
                    _log_stack(f"  native output: {n_evt['values']}")
                    _log_stack(f"  stack: {_stack_repr(stack)}")
                    print(f"    [stack] EXIT '{func_name}' (native event {i}, "
                          f"log line {_log_ln}): no matching WASM exit found")
                    print(f"             native output: {n_evt['values']}")
                    _reason = _build_discrepancy_reason(n_evt, n_evt['values'])
                    _log_stack(f"  discrepancy_reason: {_reason}")
                    if stack:
                        top = stack[-1]
                        _log_stack(
                            f"  => ROOT CAUSE: '{top['func_name']}' "
                            f"(top of stack)")
                        print(f"    [stack] Root cause: top of stack "
                              f"'{top['func_name']}'")
                        # Find the root-cause function's OWN output.
                        # If the mismatched event IS the root-cause
                        # function's own exit, use it directly;
                        # otherwise scan forward.
                        if func_name == top['func_name']:
                            native_output = n_evt['values']
                        else:
                            native_output = _find_native_output_from(
                                top['func_name'], i)
                        wasm_output = _find_wasm_output_from(
                            top['func_name'])
                        root_cause_comparison = {
                            'func_name': top['func_name'],
                            'func_usr': top['func_usr'],
                            'func_id': top.get('func_id'),
                            'native_input': top['native_input'],
                            'wasm_input': top['wasm_input'],
                            'native_output': native_output,
                            'wasm_output': wasm_output,
                            'discrepancy_reason': _reason,
                        }
                    else:
                        # Stack empty: no caller to attribute to.
                        # Leave every root-cause field empty; only
                        # discrepancy_reason carries the prefixed
                        # diagnosis so downstream consumers know this
                        # wasn't pinned to a specific function in the
                        # instrumented call chain.
                        _prefixed = f"{_no_root_prefix}\n{_reason}"
                        root_cause_comparison = {
                            'func_name': '',
                            'func_usr': '',
                            'func_id': '',
                            'native_input': {},
                            'wasm_input': {},
                            'native_output': {},
                            'wasm_output': {},
                            'discrepancy_reason': _prefixed,
                        }
                        _log_stack(
                            f"  => Stack empty — no root cause; "
                            f"attributing discrepancy to unit test scaffold")
                        _log_stack(
                            f"  discrepancy_reason (prefixed): "
                            f"{_prefixed}")
                    break

        # ── Close stack trace log ─────────────────────────────────────────
        _log_stack("")
        if root_cause_comparison:
            _log_stack(f"ROOT CAUSE: {root_cause_comparison['func_name']}")
        else:
            _log_stack("No discrepancy detected — all events matched.")

        _log_stack("")
        _log_stack("# ── Final suspicious function list ──")
        if suspicious_func_list:
            _log_stack(f"# {len(suspicious_func_list)} suspicious function(s):")
            for fid, info in suspicious_func_list.items():
                _log_stack(f"#   [{fid}] {info['func_name']}  "
                           f"quality={info['quality']}  "
                           f"reason={info['reason']}")
        else:
            _log_stack("# (empty — all soft_rollback functions were cleared)")
        stack_trace_f.close()
        print(f"    Stack trace log: {stack_trace_path}")
        print(f"    Suspicious functions remaining: "
              f"{len(suspicious_func_list)}")

        if not root_cause_comparison:
            # All native events matched successfully — no discrepancy found
            print(f"    [stack] All {len(native_events)} native events matched "
                  f"in WASM — no discrepancy detected")
            print(f"    [stack] Problem likely in the unit test script")
            return None

        # ── Helper: resolve a function name to call-graph info ─────────────
        def _resolve_func(fname, fusr):
            """Return (usr, func_info_dict) for *fname*/*fusr*."""
            fi = functions.get(fusr, {}) if fusr else {}
            resolved_usr = fusr
            if not fi:
                for u, inf in functions.items():
                    if inf.get('qualified_name', '') == fname:
                        return u, inf
                bare = fname.split('(')[0].strip()
                for u, inf in functions.items():
                    if inf.get('name', '') == bare:
                        return u, inf
                # Constructor: "ClassName::<constructor>"
                if fname.endswith('::<constructor>'):
                    cn = fname[:-len('::<constructor>')]
                    best_span, best_backup = -1, True
                    best_u, best_inf = None, {}
                    for u, inf in functions.items():
                        if inf.get('name', '') != cn:
                            continue
                        if not inf.get('is_definition', False):
                            continue
                        fp = inf.get('file', '')
                        bk = ('.pre_instrumentation' in fp
                              or '.instrumentation_backup' in fp
                              or '/Modification_copy/' in fp)
                        sp = inf.get('end_line', 0) - inf.get('line', 0)
                        if (best_backup and not bk) or \
                           (bk == best_backup and sp > best_span):
                            best_span, best_backup = sp, bk
                            best_u, best_inf = u, inf
                    if best_inf:
                        return best_u, best_inf
            return resolved_usr, fi

        def _resolve_file_lines(fname, fusr):
            """Return (file_path, line, end_line) for *fname*."""
            u, fi = _resolve_func(fname, fusr)
            f_path = fi.get('file', '')
            f_line = fi.get('line', 0)
            f_end = fi.get('end_line', 0)
            d_file = f_path
            if not fi.get('is_definition', True) or not f_path:
                dl = self._find_definition_from_report(fname)
                if dl:
                    d_file = dl['file']
            if d_file and os.path.exists(d_file):
                ext = self._find_function_extent_in_file(d_file, fname)
                if ext:
                    return ext['file'], ext['line'], ext['end_line']
            return f_path, f_line, f_end

        # ── Stack-empty special case: no root cause function ─────────────
        # When the mismatch fires with an empty stack, every root-cause
        # field was deliberately left empty by the two stack-empty
        # branches above. Skip call-graph enrichment (no point resolving
        # an empty name) and return a result whose ONLY populated field
        # is discrepancy_reason. Everything else stays empty/None so
        # consumers can detect the "no root cause" case cleanly.
        if (not root_cause_comparison.get('func_name')
                and not root_cause_comparison.get('func_usr')):
            return {
                'function_usr': '',
                'func_id': '',
                'function_name': '',
                'qualified_name': '',
                'file_path': '',
                'line': None,
                'end_line': None,
                'distance': 'N/A (no root cause — stack empty)',
                'discrepancy_reason': root_cause_comparison.get(
                    'discrepancy_reason', ''),
                'comparison': {
                    'function_usr': '',
                    'function_name': '',
                    'inputs_consistent': True,
                    'outputs_consistent': False,
                    'native_inputs': [],
                    'wasm_inputs': [],
                    'native_outputs': [],
                    'wasm_outputs': [],
                },
                'stack_depth': 0,
                'remaining_stack': [],
                'suspicious_func_list': suspicious_func_list,
            }

        # ── Resolve root cause ────────────────────────────────────────────
        rc_usr, func_info = _resolve_func(
            root_cause_comparison['func_name'],
            root_cause_comparison['func_usr'])
        rc_file, rc_line, rc_end_line = _resolve_file_lines(
            root_cause_comparison['func_name'],
            root_cause_comparison['func_usr'])

        # ── Resolve remaining stack entries ────────────────────────────────
        remaining_stack = []
        for entry in stack:
            s_name = entry['func_name']
            s_usr = entry.get('func_usr')
            s_file, s_line, s_end = _resolve_file_lines(s_name, s_usr)
            remaining_stack.append({
                'function_name': s_name,
                'function_usr': s_usr,
                'func_id': entry.get('func_id'),
                'file_path': s_file,
                'line': s_line,
                'end_line': s_end,
                'native_input': entry.get('native_input', {}),
                'wasm_input': entry.get('wasm_input', {}),
            })

        return {
            'function_usr': rc_usr or '',
            'func_id': root_cause_comparison.get('func_id'),
            'function_name': root_cause_comparison['func_name'],
            'qualified_name': func_info.get('qualified_name', ''),
            'file_path': rc_file,
            'line': rc_line,
            'end_line': rc_end_line,
            'distance': 'N/A (stack-based)',
            'discrepancy_reason': root_cause_comparison.get(
                'discrepancy_reason', ''),
            'comparison': {
                'function_usr': rc_usr or '',
                'function_name': root_cause_comparison['func_name'],
                'inputs_consistent': True,
                'outputs_consistent': False,
                'native_inputs': [root_cause_comparison['native_input']],
                'wasm_inputs': [root_cause_comparison['wasm_input']],
                'native_outputs': [root_cause_comparison['native_output']],
                'wasm_outputs': [root_cause_comparison['wasm_output']],
            },
            'stack_depth': len(stack),
            'remaining_stack': remaining_stack,
            'suspicious_func_list': suspicious_func_list,
        }

    # ------------------------------------------------------------------
    # Instrumentation quality classification
    # ------------------------------------------------------------------

    def _classify_instrumentation_quality(
        self,
        native_events: List[Dict],
        wasm_events: List[Dict],
    ) -> Dict:
        """
        Classify each instrumented function as:
          - 'normal':        all expected variables are printed
          - 'soft_rollback': one or more expected variables are missing
          - 'hard_rollback': function not instrumented at all

        Uses:
          - func_instr_status.json  (Step 1b outcome per function)
          - static_instrumentation_plan.json (expected params from AST)
          - Execution log events (actual printed variables)
          - Clang AST on instrumented source (fallback for never-called
            repaired functions)

        Returns: {func_id: {quality, step1b_status, func_name,
                            expected_vars, actual_vars, missing_vars}}
        """
        # Load Step 1b status
        status_path = os.path.join(
            self.project_path, "func_instr_status.json")
        if os.path.exists(status_path):
            with open(status_path) as f:
                func_instr_status = json.load(f)
        else:
            print("    WARNING: func_instr_status.json not found, "
                  "treating all as unknown")
            func_instr_status = {}

        # Load static instrumentation plan (expected variables from AST)
        plan_path = os.path.join(
            self.project_path, "static_instrumentation_plan.json")
        if os.path.exists(plan_path):
            with open(plan_path) as f:
                func_plan = json.load(f)
        else:
            print("    WARNING: static_instrumentation_plan.json not found")
            func_plan = {}

        # Build expected variable names per func_id from the plan.
        # Expected vars = parameter names + "ret" for non-void exits.
        # For constructors: also include member variable names.
        # These come from clang AST analysis in Step 1a.

        # Load class member vars for constructor detection.
        _class_member_vars = {}
        _class_names_set = set()
        ostream_plan_path = os.path.join(
            self.project_path, "ostream_plan.json")
        if os.path.exists(ostream_plan_path):
            with open(ostream_plan_path) as f:
                _oplan = json.load(f)
            for t in _oplan.get('types', []):
                qname = t.get('qualified_name', '')
                members = t.get('member_variables', [])
                if qname and members:
                    bare = qname.rsplit('::', 1)[-1]
                    _class_member_vars[bare] = members
        if os.path.exists(self.call_graph_path):
            with open(self.call_graph_path) as f:
                _cg_q = json.load(f)
            for _u, name in _cg_q.get('classes', {}).items():
                _class_names_set.add(name)

        expected_vars_by_fid = {}
        markers_by_fid = {}
        for fid, meta in func_plan.items():
            param_names = {p['name'] for p in meta.get('params', [])}
            if meta.get('return_type', 'void') != 'void':
                param_names.add('ret')

            # For constructors: add member variable names to expected.
            bare_fn = meta.get('func_name', '').split('(')[0].strip()
            if bare_fn in _class_names_set:
                members = _class_member_vars.get(bare_fn, [])
                for m in members:
                    param_names.add(m['name'])

            expected_vars_by_fid[fid] = param_names
            markers_by_fid[fid] = [
                b['marker'] for b in meta.get('blocks', [])]

        # Build actual printed variable names per func_id from the
        # native execution log.
        actual_vars_by_fid = {}
        for evt in native_events:
            fid_raw = evt.get('func_id')
            if not fid_raw:
                continue
            # func_id in events is "funcname_hash"; extract the hash
            parts = fid_raw.rsplit('_', 1)
            fid_hash = parts[-1] if len(parts) >= 2 else fid_raw
            if fid_hash not in actual_vars_by_fid:
                actual_vars_by_fid[fid_hash] = set()
            actual_vars_by_fid[fid_hash].update(evt['values'].keys())

        # For repaired functions NOT in the execution log, use clang AST
        # to scan the instrumented source and find which variables are
        # actually being printed.
        self._fill_actual_vars_from_ast(
            func_plan, func_instr_status, actual_vars_by_fid)

        # Classify each function
        result = {}
        for fid, meta in func_plan.items():
            func_name = meta.get('func_name', 'unknown')
            status = func_instr_status.get(fid, 'unknown')
            expected = expected_vars_by_fid.get(fid, set())
            actual = actual_vars_by_fid.get(fid, set())

            if status == 'rollback':
                quality = 'hard_rollback'
                missing = expected
            elif status == 'fallback':
                quality = 'soft_rollback'
                missing = expected
            elif status == 'ok':
                # Static gen always prints all expected vars
                quality = 'normal'
                missing = set()
            elif status == 'repaired':
                # Check if LLM repair preserved all variables
                missing = expected - actual
                quality = 'normal' if not missing else 'soft_rollback'
            else:
                # Unknown status — try to determine from actual vars
                missing = expected - actual if expected else set()
                quality = 'normal' if not missing else 'soft_rollback'

            result[fid] = {
                'quality': quality,
                'step1b_status': status,
                'func_name': func_name,
                'expected_vars': sorted(expected),
                'actual_vars': sorted(actual),
                'missing_vars': sorted(missing),
            }

        return result

    def _fill_actual_vars_from_ast(
        self,
        func_plan: Dict,
        func_instr_status: Dict,
        actual_vars_by_fid: Dict,
    ) -> None:
        """
        For repaired functions that were never called (not in execution log),
        use clang AST to parse the instrumented source and find which
        variables are actually being printed within instrumentation blocks.

        Walks the AST looking for STRING_LITERAL nodes within the function's
        line range that match the 'varname = ' pattern generated by
        _gen_print_var / the LLM.
        """
        from clang import cindex
        try:
            cindex.Config.set_library_file(
                "/usr/lib/llvm-15/lib/libclang-15.so.1")
        except Exception:
            pass

        # Group functions by file for efficient per-file parsing
        files_to_scan = {}  # file_path -> [(fid, start, end)]
        for fid, meta in func_plan.items():
            if fid in actual_vars_by_fid:
                continue  # already have runtime data
            status = func_instr_status.get(fid, 'unknown')
            if status not in ('repaired', 'unknown'):
                continue  # only need source scan for these
            file_path = meta.get('file_path', '')
            start_line = meta.get('start_line')
            end_line = meta.get('end_line')
            if not file_path or not os.path.exists(file_path):
                continue
            if start_line is None or end_line is None:
                continue
            files_to_scan.setdefault(file_path, []).append(
                (fid, start_line, end_line))

        if not files_to_scan:
            return

        print(f"    AST scanning {len(files_to_scan)} file(s) for "
              f"never-called repaired functions...")

        # Build compile args from compile_commands.json
        compile_args_map = {}
        if os.path.exists(self.compile_commands_path):
            import shlex
            with open(self.compile_commands_path) as f:
                cc = json.load(f)
            for entry in cc:
                fp = entry.get('file', '')
                if not os.path.isabs(fp):
                    fp = os.path.join(
                        entry.get('directory', '.'), fp)
                fp = os.path.abspath(fp)
                raw = shlex.split(entry.get('command', ''))[1:]
                if '-o' in raw:
                    idx = raw.index('-o')
                    del raw[idx:idx + 2]
                compile_args_map[fp] = raw

        index = cindex.Index.create()
        var_label_re = re.compile(r'^"(.+?)\s*=\s*"$')

        for file_path, funcs in files_to_scan.items():
            abs_path = os.path.abspath(file_path)
            args = compile_args_map.get(
                abs_path, ['-std=c++17', '-xc++'])
            if abs_path.endswith(('.h', '.hpp', '.hxx')):
                if '-xc++' not in args:
                    args.append('-xc++')

            try:
                tu = index.parse(abs_path, args=args,
                                 options=0x01)  # PARSE_DETAILED
            except Exception as e:
                print(f"      WARNING: clang parse failed for "
                      f"{os.path.basename(file_path)}: {e}")
                continue

            # Walk AST once, collect string literals by line
            string_lits_by_line = {}

            def _collect(cursor):
                loc = cursor.location
                if loc.file and os.path.abspath(
                        loc.file.name) == abs_path:
                    if cursor.kind == cindex.CursorKind.STRING_LITERAL:
                        raw = cursor.spelling or ''
                        m = var_label_re.match(raw)
                        if m:
                            line = loc.line
                            string_lits_by_line.setdefault(
                                line, set()).add(m.group(1).strip())
                for child in cursor.get_children():
                    _collect(child)

            _collect(tu.cursor)

            # Match collected literals to function ranges
            for fid, start_line, end_line in funcs:
                found_vars = set()
                for line_no, var_names in string_lits_by_line.items():
                    if start_line <= line_no <= end_line:
                        found_vars.update(var_names)
                if found_vars:
                    actual_vars_by_fid[fid] = found_vars

    _HEX_ADDR_RE = re.compile(r'0x[0-9a-fA-F]+')

    # Matches auto-generated operator<< output like "file{ fd_=3}" or
    # "pipe{ read_end=file{ fd_=5} write_end=file{ fd_=6}}".
    # These struct representations are equivalent to printing an object's
    # address — they represent the object itself, not a comparable scalar.
    _STRUCT_REPR_RE = re.compile(r'^\w+\{.*\}$')

    # ------------------------------------------------------------------
    # File descriptor detection & normalization
    # ------------------------------------------------------------------

    # POSIX syscalls that produce file descriptors (clang USRs).
    _FD_PRODUCING_USRS = frozenset({
        # file I/O
        'c:@F@open', 'c:@F@open64', 'c:@F@openat', 'c:@F@openat64',
        'c:@F@creat', 'c:@F@creat64',
        # sockets
        'c:@F@socket', 'c:@F@accept', 'c:@F@accept4',
        # duplicating
        'c:@F@dup', 'c:@F@dup2', 'c:@F@dup3', 'c:@F@fcntl',
        # pipes / FIFOs
        'c:@F@pipe', 'c:@F@pipe2', 'c:@F@mkfifo',
        # epoll
        'c:@F@epoll_create', 'c:@F@epoll_create1',
        # event / signal / timer fds
        'c:@F@eventfd', 'c:@F@signalfd', 'c:@F@timerfd_create',
        # inotify
        'c:@F@inotify_init', 'c:@F@inotify_init1',
        # memory fds / shared memory
        'c:@F@memfd_create', 'c:@F@shm_open',
        # misc
        'c:@F@fileno', 'c:@F@dirfd',
        'c:@F@kqueue',                          # BSD
        'c:@F@posix_openpt', 'c:@F@getpt',      # pseudo-terminal
        'c:@F@socketpair',
        'c:@F@pidfd_open',                       # Linux 5.3+
        'c:@F@userfaultfd',                      # Linux 4.3+
        'c:@F@fanotify_init',                    # Linux 2.6.37+
        'c:@F@perf_event_open',                  # Linux perf
        'c:@F@io_uring_setup',                   # io_uring
    })

    # POSIX syscalls that consume file descriptors (clang USRs).
    # Used to detect fd-managing classes via destructor → ::close.
    _FD_CONSUMING_USRS = frozenset({
        # closing
        'c:@F@close', 'c:@F@fclose', 'c:@F@shutdown',
        # reading
        'c:@F@read', 'c:@F@pread', 'c:@F@pread64',
        'c:@F@readv', 'c:@F@preadv', 'c:@F@preadv2',
        'c:@F@recv', 'c:@F@recvfrom', 'c:@F@recvmsg', 'c:@F@recvmmsg',
        # writing
        'c:@F@write', 'c:@F@pwrite', 'c:@F@pwrite64',
        'c:@F@writev', 'c:@F@pwritev', 'c:@F@pwritev2',
        'c:@F@send', 'c:@F@sendto', 'c:@F@sendmsg', 'c:@F@sendmmsg',
        # memory mapping
        'c:@F@mmap', 'c:@F@mmap64',
        # stat / metadata
        'c:@F@fstat', 'c:@F@fstat64', 'c:@F@fstatat', 'c:@F@fstatvfs',
        'c:@F@fstatfs', 'c:@F@fstatat64',
        'c:@F@fgetxattr', 'c:@F@fsetxattr', 'c:@F@flistxattr',
        'c:@F@fremovexattr',
        # truncate / allocate
        'c:@F@ftruncate', 'c:@F@ftruncate64',
        'c:@F@fallocate', 'c:@F@posix_fallocate',
        # seek
        'c:@F@lseek', 'c:@F@lseek64',
        # control
        'c:@F@ioctl', 'c:@F@fcntl',
        # permissions
        'c:@F@fchmod', 'c:@F@fchown', 'c:@F@fchownat',
        'c:@F@fchmodat',
        # sync
        'c:@F@fdatasync', 'c:@F@fsync', 'c:@F@syncfs',
        # lock
        'c:@F@flock', 'c:@F@lockf',
        # sendfile / splice
        'c:@F@sendfile', 'c:@F@sendfile64',
        'c:@F@splice', 'c:@F@tee', 'c:@F@vmsplice',
        'c:@F@copy_file_range',
        # directory
        'c:@F@fdopendir', 'c:@F@fchdir',
        'c:@F@openat', 'c:@F@mkdirat', 'c:@F@mknodat',
        'c:@F@unlinkat', 'c:@F@renameat', 'c:@F@renameat2',
        'c:@F@linkat', 'c:@F@symlinkat', 'c:@F@readlinkat',
        # socket specific
        'c:@F@bind', 'c:@F@listen', 'c:@F@connect',
        'c:@F@getsockopt', 'c:@F@setsockopt',
        'c:@F@getsockname', 'c:@F@getpeername',
        # epoll operations
        'c:@F@epoll_ctl', 'c:@F@epoll_wait', 'c:@F@epoll_pwait',
        'c:@F@epoll_pwait2',
        # poll / select (take fd sets)
        'c:@F@poll', 'c:@F@ppoll',
        # misc
        'c:@F@fexecve',
        'c:@F@fdopen',
        'c:@F@dprintf', 'c:@F@vdprintf',
        'c:@F@timerfd_settime', 'c:@F@timerfd_gettime',
        'c:@F@signalfd',
        'c:@F@inotify_add_watch', 'c:@F@inotify_rm_watch',
        'c:@F@fanotify_mark',
        'c:@F@io_uring_register', 'c:@F@io_uring_enter',
    })

    # Name heuristic used both (a) as tiebreaker when a class has
    # multiple int members, and (b) to normalize fd-typed function
    # parameters at event-matching time (see _event_match_key).
    # Covers common C++/POSIX spellings:
    #   fd, fd_, m_fd                       — generic
    #   filedesc, file_descriptor           — verbose
    #   sock, sockfd, socket_fd             — socket variants
    #   fildes, fildes2                     — POSIX (dup/dup2/close/read)
    #   oldfd, newfd                        — dup2 params
    _FD_NAME_RE = re.compile(
        r'^(?:m_)?'
        r'(?:fd_?'
        r'|file_?desc(?:riptor)?_?'
        r'|sock(?:et)?_?fd_?'
        r'|fildes2?'
        r'|(?:old|new|source|target)_?fd_?'
        r')$',
        re.IGNORECASE,
    )

    def _build_fd_metadata(self, call_graph: Dict) -> None:
        """Detect fd-returning functions and fd-holding struct members.

        Uses two structural signals from the call graph:

        Signal 1 — fd-producing functions:
            User functions that directly call a POSIX fd-producing syscall
            (``::open``, ``::socket``, …) **and** return ``int``.

        Signal 2 — fd-holding struct members:
            Classes whose **destructor** calls ``::close``.  Among the
            class's ``int``-typed members, pick the fd:
              - 1 int member  → unambiguous
              - >1 int members → name-heuristic tiebreaker

        Populates
            ``self._fd_returning_func_ids``  (set of plan func_id hashes)
            ``self._fd_member_names``        (set of member variable names)
            ``self._fd_member_re``           (compiled regex or None)
        """
        self._fd_returning_func_ids: set = set()
        self._fd_member_names: set = set()
        self._fd_member_re = None

        functions = call_graph.get('functions', {})
        call_edges = call_graph.get('call_edges', {})

        # ── Signal 1: fd-producing user functions ────────────────────
        fd_producing_usrs: set = set()
        for usr, callees in call_edges.items():
            info = functions.get(usr, {})
            if info.get('return_type') != 'int':
                continue
            if any(c in self._FD_PRODUCING_USRS for c in callees):
                fd_producing_usrs.add(usr)

        # Map fd-producing USRs → plan func_ids.
        # The plan uses 8-char MD5 hashes; the call graph uses clang USRs.
        # Match via (file basename, function name).
        plan_path = os.path.join(
            self.project_path, "static_instrumentation_plan.json")
        if os.path.exists(plan_path) and fd_producing_usrs:
            with open(plan_path) as f:
                plan = json.load(f)
            for fid, meta in plan.items():
                plan_name = meta.get('func_name', '')
                plan_file = meta.get('file_path', '')
                plan_base = os.path.basename(plan_file) if plan_file else ''
                for usr in fd_producing_usrs:
                    fi = functions.get(usr, {})
                    cg_name = fi.get('name', '')
                    cg_qname = fi.get('qualified_name', '')
                    cg_file = fi.get('file', '')
                    cg_base = os.path.basename(cg_file) if cg_file else ''
                    name_match = (plan_name == cg_name
                                  or plan_name == cg_qname)
                    if name_match and plan_base and cg_base == plan_base:
                        self._fd_returning_func_ids.add(fid)
                        break

        # ── Signal 2: fd-holding struct members ──────────────────────
        # Find classes whose destructor calls ::close.
        fd_managing_classes: set = set()  # bare class names
        for usr, callees in call_edges.items():
            if '@F@~' not in usr:
                continue
            if not any(c in self._FD_CONSUMING_USRS for c in callees):
                continue
            # Extract class names from the @S@ClassName components.
            for m in re.findall(r'@S@([^@]+)', usr):
                fd_managing_classes.add(m)

        # Look up those classes in function_types.json and find int members.
        types_path = os.path.join(
            self.project_path, "function_types.json")
        if os.path.exists(types_path) and fd_managing_classes:
            with open(types_path) as f:
                function_types = json.load(f)
            for _src, funcs_in_file in function_types.items():
                for _fn, fdata in funcs_in_file.items():
                    for _tf, types_map in fdata.get('types', {}).items():
                        for type_name, tinfo in types_map.items():
                            # Check if this type matches an fd-managing class
                            # e.g. type_name = "tao::pegtl::internal::mmap_file_open"
                            #      class_name = "mmap_file_open"
                            if not any(type_name == cn
                                       or type_name.endswith('::' + cn)
                                       for cn in fd_managing_classes):
                                continue
                            members = tinfo.get('member_variables', [])
                            int_members = [
                                mem for mem in members
                                if mem.get('type_kind') == 'INT'
                                or 'int' in mem.get('type_spelling', '').lower()
                            ]
                            if len(int_members) == 1:
                                self._fd_member_names.add(
                                    int_members[0]['name'])
                            elif len(int_members) > 1:
                                # Tiebreaker: name heuristic
                                for mem in int_members:
                                    if self._FD_NAME_RE.match(mem['name']):
                                        self._fd_member_names.add(
                                            mem['name'])

        # ── Fallback: scan ALL types for int members with fd-like names.
        # Signal 2 misses classes whose destructor calls ::close via a
        # macro (e.g. FMT_POSIX_CALL(close(fd_))) because the call graph
        # doesn't expand macros into call edges.  When this fallback
        # fires, also register the owning class as fd-managing so
        # Signal 1b (below) can pick up its int-returning getters.
        if os.path.exists(types_path):
            with open(types_path) as f:
                function_types = json.load(f)
            for _src, funcs_in_file in function_types.items():
                for _fn, fdata in funcs_in_file.items():
                    for _tf, types_map in fdata.get('types', {}).items():
                        for type_name, tinfo in types_map.items():
                            for member in tinfo.get('member_variables', []):
                                kind = member.get('type_kind', '')
                                spell = member.get('type_spelling', '')
                                is_int = (kind == 'INT'
                                          or 'int' in spell.lower())
                                if is_int and \
                                        self._FD_NAME_RE.match(member['name']):
                                    self._fd_member_names.add(member['name'])
                                    # Record bare class name
                                    bare = (type_name.rsplit('::', 1)[-1]
                                            if '::' in type_name
                                            else type_name)
                                    fd_managing_classes.add(bare)

        # ── Signal 1b: fd-returning getters on fd-managing classes ───
        # Runs AFTER the fallback so fd_managing_classes is fully
        # populated even when the destructor calls ::close via a macro
        # (FMT_POSIX_CALL etc.) that the call graph didn't capture.
        # A member of an fd-managing class that returns ``int`` is
        # almost always a getter for the fd (e.g.
        # ``fmt::file::descriptor() const noexcept { return fd_; }``)
        # whose concrete value is runtime-assigned and must not be
        # compared literally between native and wasm.
        if fd_managing_classes:
            getter_fd_usrs: set = set()
            for usr, info in functions.items():
                if info.get('return_type') != 'int':
                    continue
                for m in re.findall(r'@S@([^@]+)', usr):
                    if m in fd_managing_classes:
                        getter_fd_usrs.add(usr)
                        break

            if os.path.exists(plan_path) and getter_fd_usrs:
                with open(plan_path) as f:
                    plan = json.load(f)
                for fid, meta in plan.items():
                    plan_name = meta.get('func_name', '')
                    plan_file = meta.get('file_path', '')
                    plan_base = os.path.basename(plan_file) if plan_file else ''
                    for usr in getter_fd_usrs:
                        fi = functions.get(usr, {})
                        cg_name = fi.get('name', '')
                        cg_qname = fi.get('qualified_name', '')
                        cg_file = fi.get('file', '')
                        cg_base = os.path.basename(cg_file) if cg_file else ''
                        name_match = (plan_name == cg_name
                                      or plan_name == cg_qname)
                        if name_match and plan_base and cg_base == plan_base:
                            self._fd_returning_func_ids.add(fid)
                            break

        # ── Build regex for embedded fd members in ostream output ────
        if self._fd_member_names:
            escaped = [re.escape(n) for n in sorted(self._fd_member_names)]
            # Only match non-negative integers (valid fds).
            # Negative values like -1 are error sentinels and should
            # be compared literally.
            self._fd_member_re = re.compile(
                r'(' + '|'.join(escaped) + r')\s*=\s*(\d+)')
        else:
            self._fd_member_re = None

        # ── Log results ──────────────────────────────────────────────
        if self._fd_returning_func_ids:
            print(f"    fd-returning func_ids: "
                  f"{sorted(self._fd_returning_func_ids)}")
        if self._fd_member_names:
            print(f"    fd struct members: "
                  f"{sorted(self._fd_member_names)}")
        if not self._fd_returning_func_ids and not self._fd_member_names:
            print(f"    (no fd metadata detected)")

    def _normalize_fd_in_value(self, value: str) -> str:
        """Replace non-negative fd member values in struct representations.

        ``mmap_file_open{ m_fd=3}`` → ``mmap_file_open{ m_fd=<fd>}``

        Negative values (e.g. ``m_fd=-1``) are left as-is because they
        represent error sentinels, not actual file descriptors.
        """
        if not self._fd_member_re:
            return value
        return self._fd_member_re.sub(r'\1=<fd>', value)

    # ------------------------------------------------------------------

    def _is_struct_repr(self, value: str) -> bool:
        """Check if a value looks like an auto-generated struct print.

        Matches output from generated ``operator<<`` overloads such as
        ``file{ fd_=3}`` or ``pipe{ read_end=file{ fd_=5} ...}``.

        These represent printing the object itself (equivalent to
        printing its address) and should not be compared as scalars.
        """
        return bool(self._STRUCT_REPR_RE.match(value.strip()))

    def _is_pointer_or_object(self, value: str) -> bool:
        """Check if a value is a pointer address OR a struct representation.

        Both represent "the object itself" — one as an address, the
        other as member values from an auto-generated ``operator<<``.
        They should be treated equivalently for comparison purposes
        (since native and WASM compilers may print the same object
        differently).
        """
        return self._is_pointer_value(value) or self._is_struct_repr(value)

    def _is_pointer_value(self, value: str) -> bool:
        """Check if a value looks like a memory address.

        Matches:
          - Pure hex addresses:    '0x7ffc033ca4b0'
          - Compound with deref:   '0x7ffc033ca4b0 (*ptr = 42)'
          - Null pointer variants: '(nil)', 'nullptr', '0x0', '0'
          - Plain '0' is included because WASM represents null
            pointers and varargs as plain 0.
        """
        v = value.strip()
        if v in ('(nil)', 'nullptr', 'NULL', '0'):
            return True
        return bool(re.match(r'^0x[0-9a-fA-F]+', v))

    def _contains_address(self, value: str) -> bool:
        """Check if a value contains any hex address anywhere.

        Matches pure pointer values (``0x4ab08``) as well as values
        with an embedded address (``<unprintable type at address 0x4ab08>``).
        """
        v = value.strip()
        if v in ('(nil)', 'nullptr', 'NULL'):
            return True
        return bool(self._HEX_ADDR_RE.search(v))

    def _normalize_addresses(self, value: str) -> str:
        """Replace all hex addresses in *value* with ``<ptr>``.

        ``<unprintable type at address 0x4ab08>``
        → ``<unprintable type at address <ptr>>``

        Pure addresses like ``0x7ffc033ca4b0`` become ``<ptr>``.
        """
        v = value.strip()
        if v in ('(nil)', 'nullptr', 'NULL'):
            return '<ptr>'
        return self._HEX_ADDR_RE.sub('<ptr>', v)

    def _values_consistent(self, native_values: List[Dict], wasm_values: List[Dict]) -> bool:
        """
        Compare lists of value-dicts, ignoring pointer/address differences.

        Returns True if all non-pointer values match across invocations.
        Returns False if invocation counts differ.
        """
        if len(native_values) != len(wasm_values):
            return False

        for n_vals, w_vals in zip(native_values, wasm_values):
            all_keys = set(n_vals.keys()) | set(w_vals.keys())
            for key in all_keys:
                n_val = n_vals.get(key, '').replace('\x00', '')
                w_val = w_vals.get(key, '').replace('\x00', '')
                # Skip when EITHER side is a pointer/object reference.
                # Pointers, null variants, and struct representations
                # can match anything because native and WASM represent
                # these values completely differently (addresses,
                # varargs, object prints, etc.).
                if self._is_pointer_or_object(n_val) \
                        or self._is_pointer_or_object(w_val):
                    continue
                # Normalize embedded addresses before comparing so e.g.
                # '<unprintable type at address 0x4ab08>' matches
                # '<unprintable type at address 0x7ffc79cfd630>'
                if self._contains_address(n_val) or self._contains_address(w_val):
                    if self._normalize_addresses(n_val) != self._normalize_addresses(w_val):
                        return False
                    continue
                if n_val != w_val:
                    return False

        return True

    def _compare_function_io(self, native_funcs: Dict, wasm_funcs: Dict,
                             functions: Dict) -> List[Dict]:
        """
        Compare native vs WASM per function.

        Returns list of comparison dicts with inputs_consistent/outputs_consistent flags.
        """
        all_func_usrs = set(native_funcs.keys()) | set(wasm_funcs.keys())
        results = []

        for func_usr in all_func_usrs:
            native = native_funcs.get(func_usr, {'entries': [], 'exits': []})
            wasm = wasm_funcs.get(func_usr, {'entries': [], 'exits': []})

            inputs_consistent = self._values_consistent(
                native['entries'], wasm['entries']
            )
            outputs_consistent = self._values_consistent(
                native['exits'], wasm['exits']
            )

            results.append({
                'function_usr': func_usr,
                'function_name': functions.get(func_usr, {}).get('name', 'unknown'),
                'inputs_consistent': inputs_consistent,
                'outputs_consistent': outputs_consistent,
                'native_inputs': native['entries'],
                'wasm_inputs': wasm['entries'],
                'native_outputs': native['exits'],
                'wasm_outputs': wasm['exits'],
            })

        return results

    def _compute_call_distances(self, call_graph: Dict, root_func_usr: Optional[str]) -> Dict[str, int]:
        """BFS from root function through call_edges to compute distances."""
        if not root_func_usr:
            return {}

        call_edges = call_graph.get('call_edges', {})
        distances = {root_func_usr: 0}
        queue = [root_func_usr]

        while queue:
            current = queue.pop(0)
            current_dist = distances[current]
            for callee in call_edges.get(current, []):
                if callee not in distances:
                    distances[callee] = current_dist + 1
                    queue.append(callee)

        return distances

    def _find_root_cause_function(self, comparisons: List[Dict],
                                  distances: Dict[str, int],
                                  call_graph: Dict) -> Optional[Dict]:
        """
        Find the function with consistent inputs, inconsistent outputs,
        and smallest call-graph distance from root.
        """
        functions = call_graph.get('functions', {})
        candidates = []

        for comp in comparisons:
            if comp['inputs_consistent'] and not comp['outputs_consistent']:
                func_usr = comp['function_usr']
                dist = distances.get(func_usr, float('inf'))
                candidates.append((dist, func_usr, comp))

        if not candidates:
            # Fallback: any function with output inconsistency
            for comp in comparisons:
                if not comp['outputs_consistent']:
                    func_usr = comp['function_usr']
                    dist = distances.get(func_usr, float('inf'))
                    candidates.append((dist, func_usr, comp))

        if not candidates:
            return None

        # Sort by distance (smallest first)
        candidates.sort(key=lambda x: x[0])

        best_dist, best_usr, best_comp = candidates[0]
        func_info = functions.get(best_usr, {})

        return {
            'function_usr': best_usr,
            'function_name': func_info.get('name', 'unknown'),
            'qualified_name': func_info.get('qualified_name', ''),
            'file_path': func_info.get('file', ''),
            'line': func_info.get('line', 0),
            'end_line': func_info.get('end_line', 0),
            'distance': best_dist,
            'comparison': best_comp,
        }

    def _format_io_annotation(self, rc: Dict) -> str:
        """Build the comment block for annotating the root cause function."""
        comp = rc['comparison']
        lines = []
        lines.append("/* " + "=" * 70)
        lines.append(f" * ROOT CAUSE: {rc['function_name']}")
        lines.append(f" * File: {rc['file_path']}:{rc['line']}-{rc['end_line']}")
        lines.append(f" * Detection: stack-based execution log tracking")
        lines.append(f" *")
        lines.append(f" * This function receives CONSISTENT inputs but produces")
        lines.append(f" * INCONSISTENT outputs between native and WASM.")
        lines.append(f" *")

        # Input section
        lines.append(f" * INPUTS (consistent across native & WASM):")
        for i, inp in enumerate(comp['native_inputs']):
            vals = ", ".join(f"{k}={v}" for k, v in inp.items())
            lines.append(f" *   Invocation {i+1}: {vals}")
        if not comp['native_inputs']:
            lines.append(f" *   (no invocations recorded)")

        lines.append(f" *")

        # Output section
        lines.append(f" * OUTPUTS (INCONSISTENT):")
        n_outs = comp['native_outputs']
        w_outs = comp['wasm_outputs']
        max_outs = max(len(n_outs), len(w_outs))
        for i in range(max_outs):
            lines.append(f" *   Invocation {i+1}:")
            if i < len(n_outs):
                vals = ", ".join(f"{k}={v}" for k, v in n_outs[i].items())
                lines.append(f" *     Native: {vals}")
            else:
                lines.append(f" *     Native: (no execution)")
            if i < len(w_outs):
                vals = ", ".join(f"{k}={v}" for k, v in w_outs[i].items())
                lines.append(f" *     WASM:   {vals}")
            else:
                lines.append(f" *     WASM:   (no execution)")
        if max_outs == 0:
            lines.append(f" *   (no invocations recorded)")

        lines.append(" * " + "=" * 70 + " */")
        return "\n".join(lines)

    def _save_trace_analysis(self):
        """Save trace analysis results to JSON.

        The output includes:
          - root cause function name and location file
          - suspicious_func_list from stack-based analysis
          - custom-defined types needed by all relevant functions
            (root cause + suspicious)
        """
        rc = self.root_cause_info

        # --- Collect suspicious function list (with file/line) ---
        # Load instrumentation plan to resolve func_id → file.
        instr_plan = {}
        plan_path = os.path.join(
            self.project_path, "static_instrumentation_plan.json")
        if os.path.exists(plan_path):
            with open(plan_path) as f:
                instr_plan = json.load(f)

        # Load call graph for resolving suspicious function locations.
        cg_functions = {}
        if os.path.exists(self.call_graph_path):
            with open(self.call_graph_path) as f:
                _cg = json.load(f)
            cg_functions = _cg.get('functions', {})

        suspicious_list = []
        if rc:
            for fid, info in rc.get('suspicious_func_list', {}).items():
                func_name = info['func_name']
                plan_entry = instr_plan.get(fid, {})
                s_file = plan_entry.get('file_path', '')
                s_line = None
                s_end_line = None

                # Strategy 1: call graph by qualified_name (exact
                # signature match — avoids picking the wrong overload).
                for _u, inf in cg_functions.items():
                    if inf.get('qualified_name', '') == func_name:
                        cg_file = inf.get('file', '')
                        cg_line = inf.get('line')
                        if cg_file and cg_line:
                            s_file = cg_file
                            s_line = cg_line
                            s_end_line = inf.get('end_line', cg_line)
                            break

                # Strategy 2: scan file for function extent (fallback).
                if s_line is None and s_file and os.path.exists(s_file):
                    ext = self._find_function_extent_in_file(
                        s_file, func_name)
                    if ext:
                        s_file = ext['file']
                        s_line = ext['line']
                        s_end_line = ext['end_line']

                suspicious_list.append({
                    "func_id": fid,
                    "func_name": func_name,
                    "file": s_file or None,
                    "line": s_line,
                    "end_line": s_end_line,
                    "quality": info['quality'],
                    "reason": info['reason'],
                })

        # --- Collect custom-defined types for relevant functions ---
        custom_types = self._collect_custom_types_for_functions(rc, suspicious_list)

        # --- Include test case metadata if available ---
        test_case_info = None
        metadata = getattr(self.preprocessor, 'metadata', None)
        if metadata:
            test_case_info = metadata.get("Test Case Failure Info")

        # --- Collect remaining call stack (callers of the root cause) ---
        remaining_stack = []
        if rc:
            for entry in rc.get('remaining_stack', []):
                remaining_stack.append({
                    "function_name": entry['function_name'],
                    "file_path": entry['file_path'],
                    "line": entry['line'],
                    "end_line": entry['end_line'],
                    "native_input": entry.get('native_input', {}),
                    "wasm_input": entry.get('wasm_input', {}),
                })

        # --- Collect suspicious_types from function_types.json ----------
        # Gather all function names from root cause + stack + suspicious
        # list, look up their types, and resolve line numbers against
        # the clean source files.
        suspicious_types = {}
        ft_path = os.path.join(
            self.project_path, "function_types.json")
        if rc and os.path.exists(ft_path):
            with open(ft_path) as f:
                func_types_data = json.load(f)

            # Collect bare function names to look up.
            _target_funcs = set()
            _target_funcs.add(
                rc['function_name'].split('(')[0].strip())
            for entry in rc.get('remaining_stack', []):
                _target_funcs.add(
                    entry['function_name'].split('(')[0].strip())
            for _fid, info in rc.get(
                    'suspicious_func_list', {}).items():
                _target_funcs.add(
                    info['func_name'].split('(')[0].strip())

            # Look up types for each function.
            for _src_file, funcs in func_types_data.items():
                for fn_name, fn_entry in funcs.items():
                    if fn_name not in _target_funcs:
                        continue
                    for type_file, types_dict in fn_entry.get(
                            'types', {}).items():
                        for tname, tinfo in types_dict.items():
                            if tname in suspicious_types:
                                continue
                            loc = tinfo.get('location', {})
                            s_line = loc.get('start_line')
                            e_line = loc.get('end_line')
                            # Resolve against clean file.
                            if type_file and os.path.exists(
                                    type_file) and s_line:
                                ext = self._find_function_extent_in_file(
                                    type_file, tname.rsplit(
                                        '::', 1)[-1])
                                if ext:
                                    type_file = ext['file']
                                    s_line = ext['line']
                                    e_line = ext['end_line']
                            suspicious_types[tname] = {
                                "name": tname,
                                "kind": tinfo.get('kind'),
                                "file": type_file,
                                "line": s_line,
                                "end_line": e_line,
                                "depends_on": tinfo.get(
                                    'depends_on', []),
                            }

        data = {
            "analysis_type": "function_boundary_analysis",
            "test_case": test_case_info,
            "root_cause": {
                "function_usr": rc['function_usr'],
                "func_id": rc.get('func_id'),
                "function_name": rc['function_name'],
                "qualified_name": rc['qualified_name'],
                "file": rc['file_path'],
                "line": rc['line'],
                "end_line": rc['end_line'],
                "detection_method": "stack-based",
                "discrepancy_reason": rc.get('discrepancy_reason', ''),
                "stack_depth": rc.get('stack_depth'),
                "comparison": rc.get('comparison'),
            } if rc else None,
            "remaining_stack": remaining_stack,
            "suspicious_func_list": suspicious_list,
            "suspicious_types": list(suspicious_types.values()),
            "custom_defined_types": custom_types,
        }
        with open(self.trace_analysis_path, "w") as f:
            json.dump(data, f, indent=2)

    def _refresh_root_cause_line_numbers(self):
        """Re-resolve root cause function and test case line numbers
        from the current source files and re-save ``trace_analysis.json``.

        Uses the freshly rebuilt call graph (from phase4a) to find the
        correct definition file for each function, then scans the clean
        source to get accurate line numbers.
        """
        rc = self.root_cause_info
        if not rc:
            return
        # Stack-empty special case: no root cause function was pinned,
        # every identifier field was deliberately left empty. Don't
        # try to resolve line numbers against the call graph -- those
        # would only fill the file/line fields with arbitrary
        # matches. Re-save trace_analysis.json as-is (keeping the
        # existing test-case line numbers untouched) and bail out.
        if not rc.get('function_name') and not rc.get('function_usr'):
            print(f"  [line refresh] Root cause is empty (no pinned "
                  f"function); skipping function line refresh")
            try:
                with open(self.trace_analysis_path) as _rf:
                    _td = json.load(_rf)
                _td['root_cause'] = rc
                with open(self.trace_analysis_path, 'w') as _wf:
                    json.dump(_td, _wf, indent=2)
                print(f"  [line refresh] Re-saved {self.trace_analysis_path}")
            except Exception as _e:
                print(f"  [line refresh] WARNING: could not re-save "
                      f"trace_analysis.json: {_e}")
            return

        # Ensure remaining_stack and suspicious_func_list are in
        # root_cause_info.  They live at the JSON top level (not
        # inside "root_cause"), so they may be missing if
        # root_cause_info was loaded from the JSON's root_cause key.
        if 'remaining_stack' not in rc or 'suspicious_func_list' not in rc:
            if os.path.exists(self.trace_analysis_path):
                with open(self.trace_analysis_path) as f:
                    _saved = json.load(f)
                if 'remaining_stack' not in rc:
                    rc['remaining_stack'] = _saved.get(
                        'remaining_stack', [])
                if 'suspicious_func_list' not in rc:
                    rc['suspicious_func_list'] = {
                        s['func_id']: s
                        for s in _saved.get(
                            'suspicious_func_list', [])
                    }

        # Load the static instrumentation plan — maps func_id hashes
        # to the exact file that was instrumented.
        instr_plan = {}
        plan_path = os.path.join(
            self.project_path, "static_instrumentation_plan.json")
        if os.path.exists(plan_path):
            with open(plan_path) as f:
                instr_plan = json.load(f)

        # Load the fresh call graph (rebuilt on clean files in phase4a).
        cg_functions = {}
        if os.path.exists(self.call_graph_path):
            with open(self.call_graph_path) as f:
                cg = json.load(f)
            cg_functions = cg.get('functions', {})

        def _resolve_file_and_lines(func_name, func_usr, hint_file,
                                    func_id=None):
            """Find the definition file and line range for a function.

            Strategy (most reliable first):
            0. Look up USR in call graph — the USR is unique and gives
               an exact match even for common names like ``get()``.
            1. Use func_id to look up static_instrumentation_plan.json
               — this gives the exact file that was instrumented.
            2. Try the instrumentation report (matches by name).
            3. Search call graph by name (project files preferred).
            4. Scan resolved file for function extent.
            """
            # --- 0. Static instrumentation plan by func_id hash ---
            # The runtime log carries the func_id hash (@@FUNC_ID_<hash>@@)
            # which maps to exactly one plan entry per call site.  This
            # is the ONLY way to disambiguate template overloads that
            # share the same bare name (e.g. 5 configure_new_service
            # overloads in kangaru).  The plan gives us the exact file
            # and an anchor line in that file; _find_function_extent_in_file
            # with near_line=anchor then picks the closest matching
            # function definition — which is the correct overload.
            if func_id:
                fid_hash = func_id.rsplit('_', 1)[-1]
                plan_entry = instr_plan.get(fid_hash)
                if plan_entry:
                    plan_file = plan_entry.get('file_path', '')
                    # Prefer the pristine line (pre-instrumentation) when
                    # present; fall back to the current line otherwise.
                    anchor = (plan_entry.get('original_start_line')
                              or plan_entry.get('start_line') or 0)
                    if plan_file and os.path.exists(plan_file):
                        ext = self._find_function_extent_in_file(
                            plan_file,
                            func_name.split('(')[0].strip(),
                            near_line=anchor)
                        if ext:
                            return ext['file'], ext['line'], \
                                ext['end_line']

            # --- 1. Call graph by USR ---
            if func_usr:
                fi_usr = cg_functions.get(func_usr, {})
                if fi_usr.get('is_definition', False):
                    usr_file = fi_usr.get('file', '')
                    usr_line = fi_usr.get('line')
                    usr_end = fi_usr.get('end_line')
                    if usr_file and usr_line:
                        # Verify the name matches — the USR may have
                        # been misassigned upstream (func_id collision).
                        usr_name = fi_usr.get('name', '')
                        bare = func_name.split('(')[0].strip()
                        if usr_name == bare:
                            return usr_file, usr_line, usr_end

            # --- 2. Static instrumentation plan by func_id hash ---
            if func_id:
                fid_hash = func_id.rsplit('_', 1)[-1]
                plan_entry = instr_plan.get(fid_hash)
                if plan_entry:
                    plan_file = plan_entry.get('file_path', '')
                    if plan_file and os.path.exists(plan_file):
                        ext = self._find_function_extent_in_file(
                            plan_file, func_name)
                        if ext:
                            return ext['file'], ext['line'], \
                                ext['end_line']

            # --- 2. Instrumentation report ---
            dl = self._find_definition_from_report(func_name)
            if dl:
                return dl['file'], dl['line'], dl['end_line']

            # --- 3. Call graph by USR fallback (already tried in 0) ---
            fi = cg_functions.get(func_usr, {}) if func_usr else {}
            resolved_file = fi.get('file', '')
            is_def = fi.get('is_definition', False)

            # --- 4. Call graph by name (project files preferred) ---
            if not is_def or not resolved_file:
                bare = func_name.split('(')[0].strip()
                project_abs = os.path.abspath(self.project_path)
                best_match = None
                for _u, inf in cg_functions.items():
                    if not inf.get('is_definition', False):
                        continue
                    qn = inf.get('qualified_name', '')
                    nm = inf.get('name', '')
                    if qn != func_name and nm != bare:
                        continue
                    f = inf.get('file', '')
                    in_project = f.startswith(project_abs)
                    if qn == func_name and in_project:
                        best_match = inf
                        break
                    if best_match is None or (
                            in_project and not best_match.get(
                                'file', '').startswith(
                                    project_abs)):
                        best_match = inf
                if best_match:
                    resolved_file = best_match.get('file', '')
                    is_def = True

            # --- 5. Scan the resolved file ---
            target_file = resolved_file or hint_file
            if target_file and os.path.exists(target_file):
                ext = self._find_function_extent_in_file(
                    target_file, func_name)
                if ext:
                    return ext['file'], ext['line'], ext['end_line']

            if hint_file and hint_file != target_file \
                    and os.path.exists(hint_file):
                ext = self._find_function_extent_in_file(
                    hint_file, func_name)
                if ext:
                    return ext['file'], ext['line'], ext['end_line']

            return hint_file, rc.get('line', 0), rc.get('end_line', 0)

        # ── Refresh root cause function lines ────────────────────────
        func_name = rc['function_name']
        old_file = rc.get('file') or rc.get('file_path', '')
        old_line = rc.get('line', 0)
        old_end = rc.get('end_line', 0)

        new_file, new_line, new_end = _resolve_file_and_lines(
            func_name, rc.get('function_usr'), old_file,
            func_id=rc.get('func_id'))

        rc['file'] = new_file
        rc['file_path'] = new_file
        rc['line'] = new_line
        rc['end_line'] = new_end

        if old_file != new_file or old_line != new_line \
                or old_end != new_end:
            print(f"  [line refresh] Updated root cause: "
                  f"{os.path.basename(old_file)}:{old_line}-{old_end}"
                  f" -> {os.path.basename(new_file)}:"
                  f"{new_line}-{new_end}")
        else:
            print(f"  [line refresh] Root cause unchanged: "
                  f"{os.path.basename(new_file)}:"
                  f"{new_line}-{new_end}")

        # ── Refresh remaining stack entry lines ─────────────────────
        for entry in rc.get('remaining_stack', []):
            s_name = entry.get('function_name', '')
            s_file = entry.get('file_path', '')
            s_usr = entry.get('function_usr')
            s_fid = entry.get('func_id')
            s_new_file, s_new_line, s_new_end = \
                _resolve_file_and_lines(
                    s_name, s_usr, s_file, func_id=s_fid)
            entry['file_path'] = s_new_file
            entry['line'] = s_new_line
            entry['end_line'] = s_new_end

        # ── Refresh test case lines in preprocessor metadata ─────────
        metadata = getattr(self.preprocessor, 'metadata', None)
        if metadata:
            test_info = metadata.get("Test Case Failure Info")
            if test_info:
                self._refresh_test_case_lines(test_info)

        # Re-save trace_analysis.json with updated line numbers
        self._save_trace_analysis()
        print(f"  [line refresh] Re-saved {self.trace_analysis_path}")

        # Re-save llm_metadata.json with updated line numbers
        self._generate_llm_metadata()
        print(f"  [line refresh] Re-saved {self.llm_metadata_path}")

    def _refresh_test_case_lines(self, test_info: Dict):
        """Re-resolve test case start/end/failed line numbers from the
        current source file.

        Finds the ``TEST(`` or ``TEST_F(`` macro nearest to the original
        start line, brace-matches to find the end, and locates the
        failed line by counting non-placeholder source lines from the
        test start (instrumentation placeholders inside the test body
        shift lines non-uniformly).
        """
        tc_file = test_info.get('file_path', '')
        if not tc_file or not os.path.exists(tc_file):
            return

        orig_start = int(test_info.get('test case start line', 0))
        orig_end = int(test_info.get('test case end line', 0))
        orig_failed = int(test_info.get('failed line', 0))
        if orig_start <= 0:
            return

        with open(tc_file, 'r') as f:
            lines = f.readlines()

        # Find the TEST( or TEST_F( line closest to the original start.
        test_macro_re = re.compile(r'^\s*TEST(?:_F)?\s*\(')
        best_idx = None
        best_dist = float('inf')
        for i, line in enumerate(lines):
            if test_macro_re.match(line):
                dist = abs(i + 1 - orig_start)
                if dist < best_dist:
                    best_dist = dist
                    best_idx = i

        if best_idx is None:
            print(f"  [line refresh] WARNING: no TEST( macro found "
                  f"in {tc_file}")
            return

        new_start = best_idx + 1  # 1-indexed

        # Brace-match to find the end of the test case.
        brace_depth = 0
        started = False
        new_end = new_start
        for i in range(best_idx, len(lines)):
            for ch in lines[i]:
                if ch == '{':
                    brace_depth += 1
                    started = True
                elif ch == '}':
                    brace_depth -= 1
            if started and brace_depth <= 0:
                new_end = i + 1  # 1-indexed
                break

        # Locate the failed line.  The original failed line was at a
        # known offset (in source lines) from the original test start.
        # Instrumentation placeholders ("// instrumentation position_")
        # may have been inserted inside the test body, so a simple delta
        # shift is wrong.  Instead, walk forward from the new test start
        # counting only non-placeholder lines until we reach the same
        # source-line offset as the original.
        placeholder_re = re.compile(
            r'^\s*//\s*instrumentation\s+position_\d+')
        new_failed = 0
        if orig_failed > 0:
            target_offset = orig_failed - orig_start
            source_count = 0
            for i in range(best_idx, len(lines)):
                if not placeholder_re.match(lines[i]):
                    if source_count == target_offset:
                        new_failed = i + 1  # 1-indexed
                        break
                    source_count += 1
            # Fallback: if we ran out of lines, use delta shift
            if new_failed == 0:
                new_failed = orig_failed + (new_start - orig_start)

        if (new_start != orig_start or new_end != orig_end
                or new_failed != orig_failed):
            print(f"  [line refresh] Updated test case line numbers: "
                  f"start {orig_start}->{new_start}, "
                  f"end {orig_end}->{new_end}, "
                  f"failed {orig_failed}->{new_failed}")
        else:
            print(f"  [line refresh] Test case line numbers unchanged")

        test_info['test case start line'] = str(new_start)
        test_info['test case end line'] = str(new_end)
        if orig_failed > 0:
            test_info['failed line'] = str(new_failed)


    def _collect_custom_types_for_functions(
        self,
        root_cause_info: Optional[Dict],
        suspicious_list: List[Dict],
    ) -> Dict:
        """Collect custom-defined types needed by the root cause function
        and all suspicious functions.

        Loads ``function_types.json`` (produced by TypeParser in Phase 1)
        and extracts types used by each relevant function.

        Returns a dict keyed by qualified type name, each value containing
        the type's definition file, kind, body, location, dependencies,
        and which functions use it.
        """
        function_types_path = os.path.join(
            self.project_path, "function_types.json")
        if not os.path.exists(function_types_path):
            print("    WARNING: function_types.json not found, "
                  "cannot collect custom-defined types")
            return {}

        with open(function_types_path, 'r') as f:
            function_types = json.load(f)

        # Build set of relevant function names (and optional file paths)
        # Format: [(func_name, file_path_or_None), ...]
        relevant_funcs = []
        if root_cause_info:
            relevant_funcs.append(
                (root_cause_info['function_name'],
                 root_cause_info.get('file_path')))
        for entry in suspicious_list:
            relevant_funcs.append((entry['func_name'], None))

        # Walk function_types.json and collect types for matching functions
        # function_types structure:
        #   { source_file: { func_name: { types: { type_file: { TypeName: {...} } } } } }
        custom_types = {}  # type_name -> type_info_with_used_by

        for source_file, funcs_in_file in function_types.items():
            for func_name, func_data in funcs_in_file.items():
                # Check if this function is one we care about
                matched = False
                for rel_name, rel_file in relevant_funcs:
                    if func_name == rel_name:
                        if rel_file and os.path.abspath(source_file) != os.path.abspath(rel_file):
                            continue
                        matched = True
                        break
                if not matched:
                    continue

                types_by_file = func_data.get('types', {})
                for type_file, types_in_file in types_by_file.items():
                    for type_name, type_info in types_in_file.items():
                        if type_name not in custom_types:
                            custom_types[type_name] = {
                                "defined_in_file": type_file,
                                "kind": type_info.get('kind', ''),
                                "body": type_info.get('body', ''),
                                "location": type_info.get('location', {}),
                                "depends_on": type_info.get('depends_on', []),
                                "member_variables": type_info.get('member_variables', []),
                                "template_parameters": type_info.get('template_parameters', []),
                                "used_by_functions": [func_name],
                            }
                        else:
                            if func_name not in custom_types[type_name]['used_by_functions']:
                                custom_types[type_name]['used_by_functions'].append(func_name)

        return custom_types

    def _generate_llm_metadata(self):
        """Generate llm_metadata.json for Repairer.py."""
        rc = self.root_cause_info

        discrepant_functions = []
        if rc:
            comp = rc['comparison']
            # Build native/wasm summaries from I/O data
            native_parts = []
            for i, out in enumerate(comp['native_outputs']):
                vals = ", ".join(f"{k}={v}" for k, v in out.items())
                native_parts.append(f"#{i+1}: {vals}")
            wasm_parts = []
            for i, out in enumerate(comp['wasm_outputs']):
                vals = ", ".join(f"{k}={v}" for k, v in out.items())
                wasm_parts.append(f"#{i+1}: {vals}")

            discrepant_functions.append({
                "file": rc['file_path'],
                "line": rc['line'],
                "end_line": rc['end_line'],
                "function_name": rc['function_name'],
                "qualified_name": rc['qualified_name'],
                "discrepancy_type": "consistent_input_inconsistent_output",
                "detection_method": "stack-based",
                "native_summary": " | ".join(native_parts) if native_parts else "(no output)",
                "wasm_summary": " | ".join(wasm_parts) if wasm_parts else "(no output)",
                "annotation": (
                    f"[ROOT CAUSE] Consistent inputs but inconsistent outputs. "
                    f"Detected by stack-based execution log tracking."
                )
            })

        metadata = {
            "analysis_method": "function_boundary_analysis",
            "summary": {
                "note": "Root cause identified by stack-based execution log tracking. "
                        "Native events are followed in order; matching WASM events "
                        "are found by hash lookup. Unmatched exit = root cause.",
                "total_discrepant_functions": len(discrepant_functions),
            },
            "discrepant_functions": discrepant_functions,
            "preprocessing": {
                "analysis_file": self.trace_analysis_path,
                "native_log": self.native_log,
                "wasm_log": self.wasm_log,
                "call_graph": self.call_graph_path
            }
        }
        with open(self.llm_metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

    def _restore_non_root_cause_functions(self, root_cause):
        """
        Restore all non-root-cause functions to clean state from backup.

        Args:
            root_cause: Either a Discrepancy object or a dict with
                        'file_path' and 'function_usr' keys.
        """
        if isinstance(root_cause, dict):
            root_cause_file = os.path.abspath(root_cause['file_path'])
            root_cause_usr = root_cause['function_usr']
        else:
            root_cause_file = os.path.abspath(root_cause.file_path)
            root_cause_usr = root_cause.function_usr

        backup_dir = self.preprocessor.backup_dir
        if not os.path.exists(backup_dir):
            print("  Warning: Backup directory not found")
            return

        with open(self.call_graph_path, 'r') as f:
            call_graph = json.load(f)

        restored_count = 0

        for root, _, files in os.walk(backup_dir):
            for filename in files:
                if not any(filename.endswith(ext) for ext in ('.cc', '.cpp', '.c', '.cxx', '.h', '.hpp')):
                    continue

                backup_path = os.path.join(root, filename)
                rel_path = os.path.relpath(backup_path, backup_dir)
                current_file = os.path.join(self.project_path, rel_path)
                current_file_abs = os.path.abspath(current_file)

                if not os.path.exists(current_file):
                    continue

                if current_file_abs == root_cause_file:
                    # Skip: phase4 already wrote this file from backup + annotation.
                    # Restoring slices here would corrupt the annotation because the
                    # inserted annotation lines shift all subsequent line indices.
                    continue
                else:
                    # Not root cause file — restore entirely
                    with open(backup_path, 'r') as f:
                        backup_content = f.read()
                    with open(current_file, 'w') as f:
                        f.write(backup_content)
                    restored_count += 1

        print(f"    Restored {restored_count} non-root-cause items from backup")

    def restore_backups(self):
        """Restore source files from backup (delegates to Preprocess)."""
        self.preprocessor.restore_backups()



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Preprocessing for Dynamic Trace Analysis Workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  # Default: instrument only function call inputs/outputs
  python diff_trace_analysis.py ../tests/fmt ../tests/fmt/build_native/compile_commands.json

  # Instrument ALL assignments (original behavior):
  python diff_trace_analysis.py ../tests/fmt ../tests/fmt/build_native/compile_commands.json --full-instr

  # Use DeepSeek instead of Gemini:
  python diff_trace_analysis.py ../tests/fmt ../tests/fmt/build_native/compile_commands.json --backend deepseek
        """
    )
    parser.add_argument("project_path", help="Path to project root")
    parser.add_argument("compile_commands_path", help="Path to compile_commands.json")
    parser.add_argument(
        "--backend", "-b",
        choices=["gemini", "deepseek"],
        default="gemini",
        help="LLM backend to use for instrumentation (default: gemini)"
    )
    parser.add_argument(
        "--full-instr",
        action="store_true",
        help="Instrument ALL assignments (original behavior). "
             "Default: only instrument function call inputs/outputs."
    )
    parser.add_argument(
        "--skip-comment-instrumentation", "-s",
        action="store_true",
        help="Skip Step 1a (TODO markers) and go directly to LLM instrumentation."
    )
    parser.add_argument(
        "--skip-instrumentation",
        action="store_true",
        help="Skip Phase 1 entirely (files already instrumented with std::cout)."
    )
    parser.add_argument(
        "--fixed-time",
        action="store_true",
        help="Prefix compile and run commands with 'with_faketime' to fix the system time "
             "(avoids SSL certificate errors from time mismatch without affecting API calls)."
    )
    parser.add_argument(
        "--num-workers", "-j",
        type=int,
        default=8,
        help="Number of parallel worker threads for step 1b instrumentation "
             "(default: 8, capped to the number of functions)."
    )
    args = parser.parse_args()

    preprocessor = DiffTraceAnalysis(
        args.project_path, args.compile_commands_path,
        backend=args.backend, full_instr=args.full_instr,
        fixed_time=args.fixed_time,
        num_workers=args.num_workers,
    )

    success = preprocessor.run_full_preprocessing(
        skip_comment_instrumentation=args.skip_comment_instrumentation,
        skip_instrumentation=args.skip_instrumentation
    )

    # Always print token usage, even on failure
    print(f"\n" + "="*80)
    print("LLM TOKEN USAGE SUMMARY")
    print("="*80)
    print(f"  Total input tokens:  {GLOBAL_TOKEN_USAGE['total_input_tokens']}")
    print(f"  Total output tokens: {GLOBAL_TOKEN_USAGE['total_output_tokens']}")
    print(f"  Total tokens:        {GLOBAL_TOKEN_USAGE['total_input_tokens'] + GLOBAL_TOKEN_USAGE['total_output_tokens']}")

    # Merge trace_analysis.json and function_types.json into trace_analysis_combined.json
    ta_path = os.path.join(args.project_path, "trace_analysis.json")
    ft_path = os.path.join(args.project_path, "function_types.json")
    combined_path = os.path.join(args.project_path, "trace_analysis_combined.json")

    # If auto-expansion ran, function_types.json currently holds
    # POST-expansion line numbers (used mid-run). Phase 4a restored
    # the source tree to PRISTINE state, so the final combined file
    # must reference pristine line numbers to stay consistent. The
    # pristine snapshot was saved to function_types_pristine.json by
    # Step 1a-ii.5 -- restore it here before the merge.
    ft_pristine_path = os.path.join(
        args.project_path, "function_types_pristine.json")
    if os.path.isfile(ft_pristine_path):
        try:
            import shutil as _sh_post
            _sh_post.copy2(ft_pristine_path, ft_path)
            os.remove(ft_pristine_path)
            print(f"Restored pristine function_types.json "
                  f"(removed function_types_pristine.json)")
        except Exception as _e:
            print(f"WARNING: could not restore pristine "
                  f"function_types.json: {_e}")

    if os.path.isfile(ta_path) and os.path.isfile(ft_path):
        with open(ta_path, "r") as f:
            ta_data = json.load(f)
        with open(ft_path, "r") as f:
            ft_data = json.load(f)
        ta_data["type_info"] = ft_data
        with open(combined_path, "w") as f:
            json.dump(ta_data, f, indent=2)
        print(f"\nMerged trace_analysis.json + function_types.json -> {combined_path}")
    else:
        missing = [p for p in (ta_path, ft_path) if not os.path.isfile(p)]
        print(f"\nSkipping merge: missing {', '.join(missing)}")

    sys.exit(0 if success else 1)
