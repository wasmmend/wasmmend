"""
LLMInstrumentor.py

Supports both Gemini and DeepSeek backends via the `backend` parameter.
Mirrors the structure of src/LLMInstrumentor.py (including function-level instrumentation).
"""

import os
import sys
import re
import subprocess
import pdb
import bdb

# Add src/ for shared utilities (LLMAgent, clang)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from llm.LLMAgent import GeminiAgent, DeepSeekAgent

from clang import cindex

# Configure clang for function extraction
try:
    cindex.Config.set_library_file("/usr/lib/llvm-15/lib/libclang-15.so.1")
except:
    pass  # Clang might already be configured

SYSTEM_PROMPT = """
You are an expert in C/C++ programming and debugging. Your task is to help instrument C/C++ code to gather more information during execution.
"""

# Constructor instrumentation rules (shared between file-level and function-level)
CONSTRUCTOR_INSTRUMENTATION_RULES = """CRITICAL C++ CONSTRUCTOR INSTRUMENTATION RULES:

**CASE 1: Instrumenting INSIDE initialization list (preferred for catching initialization bugs)**
If the TODO comment appears on a member initializer line (e.g., : m_member(expr)),
wrap the expression in an immediately-invoked lambda to instrument it:

WRONG (syntax error - can't put print statements directly in initializer):
    Constructor(int x) : m_member(x) {{ std::cout << "..." << std::endl; }}

CORRECT (lambda wrapper allows instrumentation during initialization):
    Constructor(int x) : m_member([&]() {{
        std::cout << "@@INST_START_xxx@@" << std::endl;
        std::cout << "x = " << x << std::endl;
        std::cout << "@@INST_END_xxx@@" << std::endl;
        return x;
    }}()) {{}}

The [&]() {{ ... }}() creates a lambda that captures by reference and executes immediately.
This allows instrumentation to run DURING member initialization, catching bugs that occur there.

**CASE 2: Instrumenting constructor body (when TODO is inside {{ }})**
If the TODO is in the constructor body, add print statements normally:

Original:  Constructor(int x) : member(x) {{}}
Correct:   Constructor(int x) : member(x) {{ std::cout << "..." << std::endl; }}

**Important**:
- Don't print addresses of variables -- it is not useful for debugging value discrepancies.
- DO print ALL variables listed in "Variables to print".
- Place instrumentation AFTER variable definitions so the value is available to print.

Examples:
- Lambda wrapper for initialization list:
  Original:  Foo() : m_x(compute()) {{}}
  Correct:   Foo() : m_x([&]() {{
                 std::cout << "@@INST_START_abc@@" << std::endl;
                 std::cout << "@@INST_END_abc@@" << std::endl;
                 return compute();
             }}()) {{}}

- Multiple initializers (wrap each separately if TODO appears on each):
  Original:  Foo(int a, int b) : m_x(a), m_y(b) {{}}
  With TODO on m_x line:
             Foo(int a, int b) : m_x([&]() {{
                 std::cout << "@@INST_START_xxx@@" << std::endl;
                 std::cout << "a = " << a << std::endl;
                 std::cout << "@@INST_END_xxx@@" << std::endl;
                 return a;
             }}()), m_y(b) {{}}

- Regular function (no change needed):
  Original:  void function(int x) {{}}
  Correct:   void function(int x) {{ std::cout << "..." << std::endl; }}

- Variable assignment with TODO immediately after (IMPORTANT PATTERN):
  Original:  auto result = compute_value(input);
             /*TODO: print {'input', 'result', 'compute_value'} */
  Correct:   auto result = compute_value(input);
             std::cout << "@@INST_START_marker@@" << std::endl;
             std::cout << "input = " << input << std::endl;
             std::cout << "result = " << result << std::endl;
             std::cout << "@@INST_END_marker@@" << std::endl;
  NOTE: 'compute_value' is a function name (not a variable), so skip it. But DO print both 'input' and 'result'.
"""

# Default models for each backend
DEFAULT_MODELS = {
    "gemini": "gemini-3-flash-preview",
    "deepseek": "deepseek-reasoner",
}

# Max tokens for each backend
DEFAULT_MAX_TOKENS = {
    "gemini": 65536,   # gemini-2.5-flash supports up to 65536 output tokens
    "deepseek": 8192,  # deepseek max valid is 65536, but 8192 is reasonable
}


class FunctionExtractor:
    """
    Extract functions from source files using clang AST.

    Fixes for three known issues:
    1. .h/.hpp files need -xc++ flag (libclang treats .h as C otherwise,
       causing TranslationUnitLoadError).
    2. File path comparison uses os.path.abspath() on both sides to handle
       paths containing '../' segments.
    3. Does NOT modify the file (no iostream insertion). The caller
       (_instrument_file_by_todo_chunks Phase 0) handles iostream before
       Phase 1 scans for TODO blocks, so inserting again here would shift
       line numbers and break Phase 1's block positions.
    """

    def __init__(self, file_path, compile_args=None):
        # Normalise the file path so all comparisons use the same form
        self.file_path = os.path.abspath(file_path)
        self.compile_args = list(compile_args or ['-std=c++17'])
        self._prepare_compile_args()

    def _prepare_compile_args(self):
        """
        Adjust compile_args for libclang compatibility.

        - Adds -xc++ for .h/.hpp files (libclang treats .h as C by default,
          causing clang_parseTranslationUnit() to return NULL when the file
          contains C++ constructs like templates or namespaces).
        - Ensures a -std= flag is present (defaults to -std=c++17).
        """
        ext = os.path.splitext(self.file_path)[1].lower()
        if ext in ('.h', '.hpp', '.hxx', '.hh'):
            # Must come BEFORE -std= in the args list
            if '-xc++' not in self.compile_args and '-xc++-header' not in self.compile_args:
                self.compile_args.insert(0, '-xc++')

        # Ensure a C++ standard flag is present
        has_std = any(a.startswith('-std=') for a in self.compile_args)
        if not has_std:
            self.compile_args.append('-std=c++17')

    def extract_function_containing_line(self, target_line):
        """
        Extract the function containing target_line (1-indexed).
        Returns: (start_line, end_line, function_text, function_name) or None
        All line numbers are 1-indexed.
        """
        try:
            index = cindex.Index.create()
            tu = index.parse(self.file_path, args=self.compile_args)

            file_path_abs = self.file_path  # already normalised in __init__

            def find_function(cursor):
                if cursor.kind in (cindex.CursorKind.FUNCTION_DECL,
                                  cindex.CursorKind.CXX_METHOD,
                                  cindex.CursorKind.CONSTRUCTOR,
                                  cindex.CursorKind.DESTRUCTOR,
                                  cindex.CursorKind.CONVERSION_FUNCTION,
                                  cindex.CursorKind.FUNCTION_TEMPLATE):
                    if cursor.location.file:
                        cursor_file = os.path.abspath(str(cursor.location.file))
                        if cursor_file == file_path_abs:
                            if cursor.extent.start.line <= target_line <= cursor.extent.end.line:
                                return cursor

                for child in cursor.get_children():
                    result = find_function(child)
                    if result:
                        return result
                return None

            func_cursor = find_function(tu.cursor)

            if not func_cursor:
                return None

            start_line = func_cursor.extent.start.line
            end_line = func_cursor.extent.end.line

            with open(self.file_path, 'r') as f:
                lines = f.readlines()
                function_text = ''.join(lines[start_line - 1:end_line])

            return (start_line, end_line, function_text, func_cursor.spelling)

        except Exception as e:
            print(f"      Warning: AST extraction failed: {e}")
            return None


class LLMInstrumentor:
    def __init__(self,
                 instrument_files,
                 backend="gemini",
                 model=None,
                 compile_db_path=None,
                 size_threshold=200,
                 use_function_level=True,
                 fixed_time=False,
                 is_c_project=False,
                 ):
        self.backend = backend.lower()
        if self.backend not in ("gemini", "deepseek"):
            raise ValueError(f"Unknown backend: {backend}. Use 'gemini' or 'deepseek'.")
        # When True the instrumentation uses fprintf/<stdio.h> instead of
        # std::cout/<iostream>. Threaded in by the caller based on the
        # project's source language.
        self.is_c_project = is_c_project

        resolved_model = model or DEFAULT_MODELS[self.backend]
        max_tokens = DEFAULT_MAX_TOKENS[self.backend]

        if self.backend == "gemini":
            self.agent = GeminiAgent(
                model=resolved_model,
                temperature=0,
                max_tokens=max_tokens,
                system_prompt=SYSTEM_PROMPT,
            )
        else:
            self.agent = DeepSeekAgent(
                model=resolved_model,
                temperature=0,
                max_tokens=max_tokens,
                system_prompt=SYSTEM_PROMPT,
            )

        # print(f"  LLM backend: {self.backend} (model={resolved_model}, max_tokens={max_tokens})")

        self.instrument_files = instrument_files
        self.compile_db_path = compile_db_path
        self.size_threshold = size_threshold
        self.use_function_level = use_function_level
        self.fixed_time = fixed_time

#     def instrument_file(self, file_path, workdir=None):
#         """Send a single file to the LLM for instrumentation."""
#         with open(file_path, 'r') as f:
#             content = f.read()

#         prompt = f"""The following is a C/C++ source code file. Please add instrumentation code following the TODO marks to help print the corresponding program states during the execution.

# {CONSTRUCTOR_INSTRUMENTATION_RULES}

# GENERAL REQUIREMENTS:
# - Consider the context of the file to ensure the instrumentation correctly handles variable types and scopes
# - Print ALL variables listed in the TODO comment "Variables to print" set, even if they appear to be just defined
# - For variables defined on the same line (e.g., "var = func()"), print them AFTER the assignment completes
# - Include required header files (like <iostream>) at the top if needed
# - Do NOT modify any other parts of the code
# - The modified file MUST be compilable
# - Preserve all formatting, comments, and structure except where instrumentation is added
# - IMPORTANT: Do not skip variables - print every variable in the "Variables to print" set unless it would cause compilation errors or they are not variables. Also, please avoid print the addresses of variables -- that usually isn't helpful.
# - CUSTOM TYPES: operator<< overloads have been pre-generated for user-defined types (structs, classes, enums) used in the code. Assume any such type is printable with << and print it directly.
# - CONSTEXPR FUNCTIONS: If a function is declared constexpr or consteval, you should STILL add std::cout instrumentation inside it as normal. Do NOT skip instrumentation just because the function is constexpr. The build system will automatically remove the constexpr/consteval specifier from instrumented functions so that std::cout calls are valid. Treat constexpr functions the same as regular functions for instrumentation purposes.
# - POINTER DEREFERENCES (CRITICAL): The null-check + length-cap pattern applies ONLY when the value being printed is a POINTER. For non-pointer types (int, double, std::string, enums, user-defined types with operator<<), just print the value directly: `std::cout << "x = " << x << std::endl;`. Do NOT cast non-pointers to `(void*)` and do NOT apply `*value` — that will fail to compile. When the value IS a pointer, apply the guard:
#     * char*/const char*: `if (p) { std::string __s(p); if (__s.size() > 200) __s = __s.substr(0, 200) + "..."; std::cout << "p = " << __s << std::endl; } else { std::cout << "p = (nullptr)" << std::endl; }`
#     * char**/const char**: `std::cout << "p = " << (void*)p; if (p && *p) { std::string __s(*p); if (__s.size() > 200) __s = __s.substr(0, 200) + "..."; std::cout << " (*p = " << __s << ")"; } std::cout << std::endl;`
#     * Other pointers (int*, MyType*): print address only — `std::cout << "p = " << (void*)p << std::endl;` — do NOT dereference.
#   Printing (const char*)nullptr is undefined behavior; printing a non-null const char* with no \0 terminator dumps arbitrary memory and swallows downstream instrumentation markers. For pointer-to-pointer, the outer pointer can be non-null while *ptr is null — always guard both.

# Some useful examples of handling specific data types:
# - std::locale. We can print its name using locale.name():
#     std::locale loc = std::locale::classic();
#     std::cout << "loc.name() = " << loc.name() << std::endl

# Please return ONLY the full modified file content with NO explanation, NO markdown formatting, NO additional text.

# CRITICAL ALERT: Please make sure that the instrumented code compiles and runs correctly. You are an expert C++ programmer, so don't write instrumentation code that may lead to compilation errors! The analysis will fail if the code does not compile. Please make sure your instrumentation is safe and correct.

# Here is the code to instrument:
# ```
# {content}
# ```
# """

#         log_path = None
#         if workdir:
#             backup_dir = os.path.join(workdir, ".instrumentation_backups")
#             log_path = self._begin_prompt_log(
#                 backup_dir=backup_dir,
#                 source_file=file_path,
#                 call_type='instrument_file',
#                 prompt=prompt,
#                 func_name='<whole_file>',
#             )

#         response = self.agent.get_response(prompt)

#         if log_path:
#             self._finish_prompt_log(log_path, response=response)

#         return self._parse_response(response)

    def _ensure_iostream_include(self, lines):
        """
        Always add an instrumentation header include at the top of the file.

        For C++ projects this is ``#include <iostream>``; for C projects
        (``self.is_c_project == True``) it is ``#include <stdio.h>``.
        Modifies ``lines`` in place.
        """
        include_line = ('#include <stdio.h>\n' if self.is_c_project
                        else '#include <iostream>\n')

        # Find insertion point: after the last existing #include, or at top.
        last_include_idx = -1
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith('#include'):
                last_include_idx = i

        insert_at = last_include_idx + 1 if last_include_idx >= 0 else 0
        lines.insert(insert_at, include_line)

    @staticmethod
    def _strip_includes(code_text):
        """
        Extract any #include lines from LLM-generated function code.

        Returns:
            (cleaned_code, includes_list) where cleaned_code has the #include
            lines removed and includes_list contains the extracted includes.
        """
        out_lines = []
        includes = []
        for line in code_text.splitlines(keepends=True):
            if line.strip().startswith('#include'):
                includes.append(line.strip() + '\n')
            else:
                out_lines.append(line)
        return (''.join(out_lines), includes)

    @staticmethod
    def _insert_includes_at_top(file_lines, includes):
        """
        Insert #include lines at the top of the file (after the last existing
        #include). Skips duplicates. Modifies file_lines in place.

        IMPORTANT: Call this AFTER splicing the function into file_lines,
        so that splice indices are not shifted.

        Returns:
            The number of lines actually inserted.
        """
        if not includes:
            return 0
        existing = {fl.strip() for fl in file_lines if fl.strip().startswith('#include')}
        to_add = [inc for inc in includes if inc.strip() not in existing]
        if not to_add:
            return 0
        last_include_idx = -1
        for i, fl in enumerate(file_lines):
            if fl.strip().startswith('#include'):
                last_include_idx = i
        insert_at = last_include_idx + 1 if last_include_idx >= 0 else 0
        for j, inc in enumerate(to_add):
            file_lines.insert(insert_at + j, inc)
        return len(to_add)

    def _compile_project(self, compile_script):
        """
        Centralized compilation: runs 'bash compile_script'.
        Returns (success: bool, stderr: str, stdout: str).

        Compiler errors go to stderr; build-system progress goes to stdout.
        """
        try:
            if self.fixed_time:
                # source it explicitly so it's available in the subprocess.
                env_script = os.path.join(
                    os.environ.get('CONDA_PREFIX', ''),
                    'etc', 'conda', 'activate.d', 'env_vars.sh')
                cmd = ["bash", "-c",
                       f"source {env_script} 2>/dev/null; bash {compile_script}"]
            else:
                cmd = ["bash", compile_script]
            result = subprocess.run(
                cmd,
                cwd=".",
                capture_output=True,
                text=True,
                timeout=300
            )
            return (result.returncode == 0, result.stderr, result.stdout)
        except subprocess.TimeoutExpired:
            return (False, "Compilation timed out", "")
        except Exception as e:
            return (False, f"Compilation error: {e}", "")

    @staticmethod
    def _format_compile_errors(stderr, stdout, stderr_lines=30,
                               stdout_lines=0, file_path=None,
                               func_name=None, func_range=None):
        """Format compilation errors for inclusion in repair prompts.

        Filtering tiers (most → least specific):

        1. **Line-range** — *file_path* + *func_range* ``(start, end)``
           (0-indexed).  Only errors whose primary line falls within
           ``[start+1-margin, end+1+margin]`` (1-indexed) are kept.
           Consequential errors outside this range are discarded.
        2. **File + func name** — errors in *file_path* that mention
           *func_name* in the diagnostic text.
        3. **File only** — any error in *file_path*.
        4. **Raw tail** — last *stderr_lines* of stderr.

        Within the chosen tier, groups are sorted by line number
        (earliest first = root cause) and deduplicated by primary line.
        Output is capped at *stderr_lines* from the **beginning** so
        that root-cause errors are never truncated.

        Parameters:
            func_range: Optional ``(start, end)`` 0-indexed inclusive
                line range of the modified function in the current file.
        """
        _diag_re = re.compile(
            r'^(.+?):(\d+):\d+:\s*(error|warning|note|fatal error):')

        def _matches_file(err_file, target_path):
            if not target_path:
                return False
            if err_file == target_path:
                return True
            if os.path.basename(err_file) == os.path.basename(
                    target_path):
                return True
            try:
                if os.path.abspath(err_file) == os.path.abspath(
                        target_path):
                    return True
            except (OSError, ValueError):
                pass
            return False

        MARGIN = 5  # lines of tolerance around the function range

        def _in_range(line_1idx, rng):
            """True if 1-indexed *line_1idx* is within the 0-indexed
            *rng* (start, end) ± MARGIN."""
            if rng is None:
                return False
            lo = rng[0] + 1 - MARGIN
            hi = rng[1] + 1 + MARGIN
            return lo <= line_1idx <= hi

        def _try_filter(raw_lines, target_path, target_func,
                        target_range, limit):
            if not target_path:
                return None

            # -- Parse stderr into diagnostic groups. --
            # Each group: one error/warning + following notes/snippets.
            groups = []
            cur_group = None
            cur_file = False
            cur_func = False
            cur_in_range = False
            cur_line_no = 0

            for line in raw_lines:
                m = _diag_re.match(line)
                if m and m.group(3) in ('error', 'warning',
                                        'fatal error'):
                    if cur_group is not None:
                        groups.append((cur_file, cur_func,
                                       cur_in_range, cur_line_no,
                                       cur_group))
                    cur_group = [line]
                    cur_line_no = int(m.group(2))
                    cur_file = _matches_file(m.group(1), target_path)
                    cur_in_range = (cur_file and
                                    _in_range(cur_line_no,
                                              target_range))
                    cur_func = bool(target_func and
                                    target_func in line)
                elif cur_group is not None:
                    cur_group.append(line)
                    if m:
                        if _matches_file(m.group(1), target_path):
                            cur_file = True
                            if _in_range(int(m.group(2)),
                                         target_range):
                                cur_in_range = True
                        if target_func and target_func in line:
                            cur_func = True

            if cur_group is not None:
                groups.append((cur_file, cur_func, cur_in_range,
                               cur_line_no, cur_group))
            if not groups:
                return None

            # -- Pick the most specific matching tier. --
            tier1 = [(ln, g) for fm, _, rm, ln, g in groups
                     if fm and rm]
            tier2 = [(ln, g) for fm, fnm, _, ln, g in groups
                     if fm and fnm]
            tier3 = [(ln, g) for fm, _, _, ln, g in groups if fm]

            if tier1:
                chosen, tag = tier1, "line-range"
            elif tier2:
                chosen, tag = tier2, "file+func"
            elif tier3:
                chosen, tag = tier3, "file"
            else:
                return None

            # Sort earliest first (root cause before consequences).
            chosen.sort(key=lambda x: x[0])

            # Deduplicate by primary line number.
            seen = set()
            deduped = []
            for ln, g in chosen:
                if ln not in seen:
                    seen.add(ln)
                    deduped.append(g)

            # Flatten.  Cap from the BEGINNING so root causes survive.
            flat = []
            for g in deduped:
                flat.extend(g)
            total = len(flat)
            capped = flat[:limit] if total > limit else flat

            rng_info = ""
            if target_range is not None and tag == "line-range":
                rng_info = (f", func lines "
                            f"{target_range[0]+1}-{target_range[1]+1}")
            header = (f"[{len(deduped)} diagnostic group(s) matched "
                      f"by {tag}{rng_info} — showing first "
                      f"{len(capped)} of {total} lines]")
            return header + '\n' + '\n'.join(capped)

        # ---- main logic ----
        parts = []
        if stderr and stderr.strip():
            raw = stderr.strip().split('\n')
            filtered = _try_filter(raw, file_path, func_name,
                                   func_range, stderr_lines)
            if filtered:
                parts.append(filtered)
            else:
                tail = raw[-stderr_lines:] if len(raw) > stderr_lines \
                    else raw
                parts.append(
                    "[stderr — last %d of %d lines]\n%s"
                    % (len(tail), len(raw), '\n'.join(tail)))
        if stdout and stdout.strip() and stdout_lines > 0:
            raw = stdout.strip().split('\n')
            tail = raw[-stdout_lines:] if len(raw) > stdout_lines \
                else raw
            parts.append(
                "[stdout — last %d of %d lines]\n%s"
                % (len(tail), len(raw), '\n'.join(tail)))
        return '\n\n'.join(parts) if parts else "(no output)"

    def _begin_prompt_log(self, backup_dir, source_file, call_type, prompt,
                          func_name=None, attempt=None):
        """
        Write the prompt portion of a log file **before** the LLM API call.
        Returns the log file path so the caller can append the response later
        via :meth:`_finish_prompt_log`.
        """
        from datetime import datetime

        log_dir = os.path.join(backup_dir, "prompt_logs")
        os.makedirs(log_dir, exist_ok=True)

        basename = os.path.basename(source_file).replace('.', '_')
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        safe_func = re.sub(r'[^a-zA-Z0-9_]', '_', func_name or 'unknown')[:60]

        if attempt is not None:
            filename = f"{ts}__{basename}__{call_type}__{safe_func}__attempt{attempt}.txt"
        else:
            filename = f"{ts}__{basename}__{call_type}__{safe_func}.txt"

        log_path = os.path.join(log_dir, filename)

        with open(log_path, 'w') as f:
            f.write(f"=== LLM PROMPT LOG ===\n")
            f.write(f"Timestamp:   {datetime.now().isoformat()}\n")
            f.write(f"Source file: {source_file}\n")
            f.write(f"Call type:   {call_type}\n")
            f.write(f"Function:    {func_name or 'N/A'}\n")
            f.write(f"Attempt:     {attempt or 'N/A'}\n")
            f.write(f"Backend:     {self.backend}\n")
            f.write(f"\n{'='*80}\n")
            f.write(f"PROMPT ({len(prompt)} chars):\n")
            f.write(f"{'='*80}\n")
            f.write(prompt)
            f.write(f"\n\n{'='*80}\n")
            f.write(f"[AWAITING RESPONSE...]\n")

        return log_path

    def _finish_prompt_log(self, log_path, response=None,
                           compile_ok=None, compile_errors=None):
        """
        Append the LLM response (and optional compilation result) to an
        existing log file created by :meth:`_begin_prompt_log`.
        """
        if not log_path or not os.path.exists(log_path):
            return

        with open(log_path, 'a') as f:
            if response is not None:
                f.write(f"RAW RESPONSE ({len(response)} chars):\n")
                f.write(f"{'='*80}\n")
                f.write(response)
                f.write(f"\n\n{'='*80}\n")
            else:
                f.write(f"[NO RESPONSE — call may have failed]\n")
                f.write(f"{'='*80}\n")
            if compile_ok is not None:
                f.write(f"COMPILATION RESULT: {'SUCCESS' if compile_ok else 'FAILED'}\n")
                f.write(f"{'='*80}\n")
                if compile_errors and not compile_ok:
                    f.write(f"COMPILATION ERRORS:\n")
                    f.write(compile_errors[:5000])
                    f.write("\n")

    def _find_enclosing_function(self, lines, todo_start_idx):
        """
        Find the enclosing function around a TODO block.

        Searches backward from the TODO for the function's opening brace,
        then forward for the matching closing brace.

        Returns (func_start_idx, func_end_idx) as 0-indexed line indices,
        or None if not found.
        """
        # Search backward for the opening brace of the enclosing function
        brace_depth = 0
        func_body_start = None

        for i in range(todo_start_idx, -1, -1):
            line = lines[i]
            for ch in reversed(line):
                if ch == '}':
                    brace_depth += 1
                elif ch == '{':
                    if brace_depth == 0:
                        func_body_start = i
                        break
                    brace_depth -= 1
            if func_body_start is not None:
                break

        if func_body_start is None:
            return None

        # Search backward from the opening brace to find the function signature
        func_start = func_body_start
        for i in range(func_body_start - 1, max(-1, func_body_start - 15), -1):
            if i < 0:
                break
            stripped = lines[i].strip()
            if not stripped or stripped.startswith('//') or stripped.startswith('*'):
                break
            func_start = i

        # Search forward from the opening brace for the matching closing brace
        brace_depth = 0
        func_end = None
        for i in range(func_body_start, len(lines)):
            for ch in lines[i]:
                if ch == '{':
                    brace_depth += 1
                elif ch == '}':
                    brace_depth -= 1
                    if brace_depth == 0:
                        func_end = i
                        break
            if func_end is not None:
                break

        if func_end is None:
            return None

        return (func_start, func_end)

    def _find_function_from_todo_block(self, lines, block_start, block_end):
        """
        Use the TODO block's type (ENTRY / EXIT / ASSIGNMENT) to locate the
        enclosing function more reliably than generic brace-matching.

        - ENTRY block ("Function ENTRY:"): the opening '{' is just above the
          block.  Search backward for '{', then backward for the signature,
          then forward for the matching '}'.
        - EXIT block ("Function EXIT:"): the closing '}' is shortly after the
          return statement that follows the block.  Search forward past the
          'return ...;' for '}', then backward for the matching '{', then
          backward for the signature.
        - ASSIGNMENT block ("Variables to print:"): fall back to the generic
          _find_enclosing_function().

        Returns (func_start_idx, func_end_idx) as 0-indexed, or None.
        Also returns func_name extracted from the TODO text when available.
        Return type: (func_start, func_end, func_name) or None.
        """
        # Collect the TODO block text
        block_text = ''.join(lines[block_start:block_end + 1])

        # Extract function name from TODO text if present
        func_name = None
        name_match = re.search(r'Function (?:ENTRY|EXIT)(?:\s*\(void\))?:\s*(\S+)\s*\(', block_text)
        if name_match:
            func_name = name_match.group(1)

        # ── ENTRY block: '{' is right above the TODO ────────────────────────
        if 'Function ENTRY:' in block_text:
            # Search backward from block_start for the opening '{'
            func_body_start = None
            for i in range(block_start - 1, max(-1, block_start - 5), -1):
                if i < 0:
                    break
                if '{' in lines[i]:
                    func_body_start = i
                    break

            if func_body_start is None:
                # Wider search
                for i in range(block_start, -1, -1):
                    if '{' in lines[i]:
                        func_body_start = i
                        break

            if func_body_start is None:
                return None

            # Search backward from '{' for the function signature
            func_start = func_body_start
            for i in range(func_body_start - 1, max(-1, func_body_start - 20), -1):
                if i < 0:
                    break
                stripped = lines[i].strip()
                if not stripped or stripped.startswith('//') or stripped.startswith('/*') or stripped.startswith('*'):
                    break
                # Stop at lines that look like end of previous statement/function
                if stripped.endswith(';') or stripped.endswith('}'):
                    break
                func_start = i

            # Search forward from '{' for the matching '}'
            brace_depth = 0
            func_end = None
            for i in range(func_body_start, len(lines)):
                for ch in lines[i]:
                    if ch == '{':
                        brace_depth += 1
                    elif ch == '}':
                        brace_depth -= 1
                        if brace_depth == 0:
                            func_end = i
                            break
                if func_end is not None:
                    break

            if func_end is None:
                return None

            return (func_start, func_end, func_name)

        # ── EXIT block: '}' is after the return statement below the TODO ────
        if 'Function EXIT:' in block_text:
            # Search forward from block_end for the return statement,
            # then for the closing '}'
            func_body_end = None

            # First, find the return statement (usually within a few lines)
            return_line = None
            for i in range(block_end, min(len(lines), block_end + 10)):
                stripped = lines[i].strip()
                if stripped.startswith('return ') or stripped == 'return;':
                    return_line = i
                    break

            # Now search forward from return (or block_end) for the closing '}'
            search_from = return_line if return_line is not None else block_end
            brace_depth = 0
            for i in range(search_from, len(lines)):
                for ch in lines[i]:
                    if ch == '{':
                        brace_depth += 1
                    elif ch == '}':
                        if brace_depth == 0:
                            func_body_end = i
                            break
                        brace_depth -= 1
                if func_body_end is not None:
                    break

            if func_body_end is None:
                return None

            # Now find the matching '{' by searching backward
            brace_depth = 0
            func_body_start = None
            for i in range(func_body_end, -1, -1):
                for ch in reversed(lines[i]):
                    if ch == '}':
                        brace_depth += 1
                    elif ch == '{':
                        if brace_depth == 0:
                            func_body_start = i
                            break
                        brace_depth -= 1
                if func_body_start is not None:
                    break

            if func_body_start is None:
                return None

            # Search backward from '{' for the function signature
            func_start = func_body_start
            for i in range(func_body_start - 1, max(-1, func_body_start - 20), -1):
                if i < 0:
                    break
                stripped = lines[i].strip()
                if not stripped or stripped.startswith('//') or stripped.startswith('/*') or stripped.startswith('*'):
                    break
                if stripped.endswith(';') or stripped.endswith('}'):
                    break
                func_start = i

            return (func_start, func_body_end, func_name)

        # ── ASSIGNMENT block: generic brace-matching fallback ───────────────
        result = self._find_enclosing_function(lines, block_start)
        if result:
            return (result[0], result[1], func_name)
        return None

    def instrument_files_in_project(self, workdir=None):
        """
        Instrument all files in the project.
        Uses hybrid approach: chunk-level for large files, file-level for small files.
        """
        if not self.instrument_files:
            print("No files to instrument")
            return []

        modified_files = []
        function_level_files = []

        for i in self.instrument_files:
            file_path = i[0]
            # Skip gtest files — no need to instrument test framework internals
            basename = os.path.basename(file_path)
            # if basename.startswith('gtest') or '/gtest/' in file_path:
            #     print(f"Skipping gtest file: {file_path}")
            #     continue

            if "chrono.h" in file_path:
                print(f"Skipping chrono.h (known to cause issues with instrumentation): {file_path}")
                print("Remember to remove!!!!!")
                continue

            print(f"Instrumenting file: {file_path}")

            # Check file size and choose instrumentation strategy
            file_size = self.count_lines(file_path)

            if self.use_function_level:
                # Large file: instrument by extracting TODO chunks
                print(f" Using function-level instrumentation")

                success = self._instrument_file_by_todo_chunks(file_path, workdir)
                if success:
                    modified_files.append(file_path)
                    function_level_files.append(file_path)
                    continue
                else:
                    print(f"  Chunk-level instrumentation failed, trying file-level fallback...")

            # Small file OR chunk-level failed: use file-level instrumentation
            try:
                print(f"  Using file-level instrumentation ({file_size} lines)")
                instrumented_code = self.instrument_file(file_path, workdir=workdir)
                # Ensure #include <iostream> is present
                code_lines = instrumented_code.splitlines(keepends=True)
                self._ensure_iostream_include(code_lines)
                instrumented_code = "".join(code_lines)
                with open(file_path, 'w') as f:
                    f.write(instrumented_code)
                modified_files.append(file_path)
            except Exception as e:
                print(f"  Error instrumenting {file_path}: {e}")

        return modified_files

    def _find_instrumented_functions(self, lines):
        """
        Find all functions in the file that contain instrumentation markers
        (@@INST_START_ or @@INST_END_).

        Returns a list of (func_start, func_end) tuples (0-indexed line indices),
        sorted by func_start ascending.
        """
        # Find all lines containing instrumentation markers
        marker_lines = []
        for i, line in enumerate(lines):
            if '@@INST_START_' in line or '@@INST_END_' in line:
                marker_lines.append(i)

        if not marker_lines:
            return []

        # Find enclosing functions for each marker line
        func_ranges = {}
        for marker_idx in marker_lines:
            result = self._find_enclosing_function(lines, marker_idx)
            if result:
                # Merge overlapping ranges
                merged = False
                for existing_key in list(func_ranges.keys()):
                    if (existing_key[0] <= result[0] <= existing_key[1] or
                        result[0] <= existing_key[0] <= result[1]):
                        new_key = (min(existing_key[0], result[0]),
                                   max(existing_key[1], result[1]))
                        func_ranges[new_key] = True
                        if new_key != existing_key:
                            del func_ranges[existing_key]
                        merged = True
                        break
                if not merged:
                    func_ranges[result] = True

        return sorted(func_ranges.keys(), key=lambda x: x[0])

    def _repair_instrumented_functions(self, func_name, original_func_text,
                                       previous_attempt, file_path,
                                       current_code, compilation_error="",
                                       backup_dir=None, func_range=None):
        """
        Ask LLM to fix an instrumented function and splice the fix back.

        Parameters:
            func_range: Optional (start, end) 0-indexed line range of the
                        function to repair.  When provided the method repairs
                        exactly that range; otherwise it falls back to
                        searching for all instrumented functions via markers.
        Returns:
            Always a 2-tuple ``(fixed_code, updated_range)``. When
            ``func_range`` is provided, ``updated_range`` is the new
            ``(start, end)`` 0-indexed line range after splice + any
            hoisted ``#include`` insertions. When ``func_range`` is
            None (multi-target mode), ``updated_range`` is ``None``.
        """
        lines = current_code.splitlines(keepends=True)

        # Determine which function(s) to repair.
        if func_range is not None:
            func_start, func_end = func_range
            targets = [(func_start, func_end)]
        else:
            targets = self._find_instrumented_functions(lines)
            if not targets:
                print(f"    No instrumented functions found in "
                      f"{os.path.basename(file_path)}, skipping")
                return current_code, None

        relevant_errors = compilation_error.strip()
        if not relevant_errors:
            return current_code, func_range

        # Track the updated function range (only meaningful when
        # func_range is provided, i.e. single-target mode).
        updated_range = func_range

        # Process in reverse order to preserve upper line numbers.
        for func_idx, (func_start, func_end) in enumerate(reversed(targets)):

            if self.is_c_project:
                from instrumentation.c_instrumentation_prompts import C_REPAIR_PROMPT_TEMPLATE
                repair_prompt = C_REPAIR_PROMPT_TEMPLATE.format(
                    func_name=func_name,
                    relevant_errors=relevant_errors,
                    original_func_text=original_func_text,
                    previous_attempt=previous_attempt,
                )
            else:
                repair_prompt = f"""Fix the compilation errors in this instrumented C++ function.

FUNCTION: {func_name}

COMPILATION ERRORS:
```
{relevant_errors}
```

ORIGINAL FUNCTION (before instrumentation — compiles fine):
```cpp
{original_func_text}
```

CURRENT ATTEMPT (has compilation errors):
```cpp
{previous_attempt}
```

INSTRUCTIONS:
1. First, check whether this function is declared as `constexpr` or `consteval`.
   - If it IS constexpr/consteval, wrap ALL instrumentation printing code inside
     `if consteval {{ }} else {{ /* printing here */ }}` blocks. Example:
     ```
     constexpr int foo(int x) {{
         if (__builtin_is_constant_evaluated()) {{
             (void)0;
         }} else {{
             std::cout << "runtime: " << x << "\\n";
         }}
         return x * 2;
     }}
     ```
   - If it is NOT constexpr, proceed normally.
2. Read the compilation errors carefully and fix exactly what they say.
3. Preserve all instrumentation markers (@@INST_START_/@@INST_END_ and @@FUNC_ID_...@@).
4. Preserve the original program logic — only fix the instrumentation code.
5. CUSTOM TYPES: operator<< overloads have been pre-generated for user-defined types. Assume any such type is printable with <<.
6. POINTER DEREFERENCES (CRITICAL): The null-check + length-cap pattern applies ONLY when printing a POINTER. For non-pointer types (int, double, std::string, enums, user-defined types with operator<<), print directly: `std::cout << "x = " << x << std::endl;`. Do NOT cast non-pointers to (void*) and do NOT apply `*value` — this will fail to compile. When the value IS a pointer, use:
   - char*/const char*: null-check outer, construct std::string from it, cap at 200 chars.
   - char**/const char**: null-check BOTH levels, construct std::string from *ptr, cap at 200 chars.
   - Other pointers (int*, MyType*): print address only with `(void*)ptr`; do NOT dereference.
   ```cpp
   // char** example
   if (ptr && *ptr) {{
       std::string __s(*ptr);
       if (__s.size() > 200) __s = __s.substr(0, 200) + "...";
       std::cout << " (*ret = " << __s << ")";
   }}
   ```
   Printing (const char*)nullptr is UB; printing a non-null const char* with no terminator dumps memory and swallows @@INST_END@@ markers, breaking the analysis.
7. If a specific variable causes a compilation error and you cannot figure out how to print it, commenting out that particular std::cout line is also a valid fix. For example:
   `// std::cout << "x = " << x << std::endl;  // commented out: unprintable type`
7. Return ONLY the fixed function with NO explanation, NO markdown formatting, NO additional text.

CRITICAL: The code MUST compile."""

            # Log repair prompt before API call
            rf_log = None
            if backup_dir:
                rf_log = self._begin_prompt_log(
                    backup_dir=backup_dir,
                    source_file=file_path,
                    call_type='repair_file',
                    prompt=repair_prompt,
                    func_name=f'lines_{func_start + 1}_{func_end + 1}',
                    attempt=func_idx + 1,
                )

            try:

                response = self.agent.get_response(repair_prompt)
                fixed_func = self._parse_response(response)

                fixed_func, hoisted_includes = self._strip_includes(fixed_func)
                new_func_lines = fixed_func.splitlines(keepends=True)
                if new_func_lines and not new_func_lines[-1].endswith('\n'):
                    new_func_lines[-1] += '\n'

                lines[func_start:func_end + 1] = new_func_lines
                n_includes = self._insert_includes_at_top(
                    lines, hoisted_includes)

                # Update tracked range: includes shift start/end down,
                # body size change shifts end only.
                new_start = func_start + n_includes
                new_end = new_start + len(new_func_lines) - 1
                updated_range = (new_start, new_end)


                if rf_log:
                    self._finish_prompt_log(rf_log, response=response)
            except Exception as e:
                if rf_log:
                    self._finish_prompt_log(rf_log)
                print(f"    Error repairing function at lines "
                      f"{func_start + 1}-{func_end + 1}: {e}")
                continue

        result_code = "".join(lines)
        return result_code, updated_range

    def _parse_response(self, response):
        """Extract code from LLM response, stripping markdown blocks."""
        response = response.strip()

        # Case 1: Complete code block with opening and closing fences
        code_block_pattern = r'```(?:c\+\+|cpp|c)?\s*(.*?)```'
        matches = re.findall(code_block_pattern, response, re.DOTALL | re.IGNORECASE)
        if matches:
            return matches[0].strip()

        # Case 2: Opening fence but no closing fence (truncated output)
        open_fence = re.match(r'^```(?:c\+\+|cpp|c)?\s*\n', response, re.IGNORECASE)
        if open_fence:
            response = response[open_fence.end():]
            if response.rstrip().endswith('```'):
                response = response.rstrip()[:-3]
            return response.strip()

        # Case 3: No code blocks found, assume the entire response is code
        return response

    def _filter_compilation_errors(self, error_output, max_errors=50, max_chars=15000):
        """
        Filter and summarize compilation errors to avoid token limit issues.

        Returns:
            Tuple of (filtered_error_string, set_of_files_with_errors)
        """
        lines = error_output.split('\n')

        errors = []
        warnings = []
        notes = []
        file_error_map = {}

        error_pattern = re.compile(r'(.+?):(\d+):(\d+):\s*(error|warning|note):\s*(.+)')

        for line in lines:
            match = error_pattern.match(line)
            if match:
                file_path, line_num, col_num, msg_type, message = match.groups()
                file_path = file_path.strip()

                if msg_type == 'error':
                    error_entry = f"{file_path}:{line_num}:{col_num}: error: {message}"
                    errors.append(error_entry)
                    if file_path not in file_error_map:
                        file_error_map[file_path] = []
                    file_error_map[file_path].append(message)

                elif msg_type == 'warning':
                    if len(warnings) < 10 and message not in [w.split(': ')[-1] for w in warnings]:
                        warnings.append(f"{file_path}:{line_num}: warning: {message}")

                elif msg_type == 'note':
                    if len(notes) < 20:
                        notes.append(f"{file_path}:{line_num}: note: {message}")

        # Build filtered output
        filtered_parts = []

        if file_error_map:
            filtered_parts.append("=== COMPILATION ERROR SUMMARY ===")
            filtered_parts.append(f"Total files with errors: {len(file_error_map)}")
            for fp, el in file_error_map.items():
                filtered_parts.append(f"  {fp}: {len(el)} error(s)")
            filtered_parts.append("")

        if errors:
            filtered_parts.append("=== ERRORS ===")
            unique_errors = []
            seen_messages = set()

            for error in errors[:max_errors]:
                msg = error.split('error: ')[-1] if 'error: ' in error else error
                if msg not in seen_messages or len(unique_errors) < 20:
                    unique_errors.append(error)
                    seen_messages.add(msg)

            filtered_parts.extend(unique_errors)

            if len(errors) > max_errors:
                filtered_parts.append(f"... and {len(errors) - max_errors} more errors")
            filtered_parts.append("")

        if warnings:
            filtered_parts.append("=== SAMPLE WARNINGS ===")
            filtered_parts.extend(warnings[:5])
            if len(warnings) > 5:
                filtered_parts.append(f"... and {len(warnings) - 5} more warnings")
            filtered_parts.append("")

        if notes and errors:
            filtered_parts.append("=== RELEVANT NOTES ===")
            filtered_parts.extend(notes[:10])
            filtered_parts.append("")

        result = '\n'.join(filtered_parts)

        if len(result) > max_chars:
            result = result[:max_chars] + f"\n\n[... truncated {len(result) - max_chars} characters ...]"

        return result, set(file_error_map.keys())

    # ========================================================================================
    # FUNCTION-LEVEL INSTRUMENTATION METHODS (for large files)
    # ========================================================================================

    def count_lines(self, file_path):
        """Count lines in a file."""
        try:
            with open(file_path, 'r') as f:
                return len(f.readlines())
        except:
            return 0

    def get_compile_flags_from_db(self, file_path):
        """
        Parse compile_commands.json to get compile flags for a specific file.

        When the file is found in compile_commands.json, returns its exact flags.
        When not found (e.g. header files like chrono.h), collects the union
        of -I, -isystem, and -D flags from ALL entries so that header includes
        can still be resolved.
        """
        if not self.compile_db_path or not os.path.exists(self.compile_db_path):
            return None

        import json
        import shlex

        try:
            with open(self.compile_db_path, 'r') as f:
                compile_db = json.load(f)

            file_path_abs = os.path.abspath(file_path)

            # --- Try exact match first ---
            for entry in compile_db:
                entry_file = os.path.abspath(entry.get('file', ''))
                if entry_file == file_path_abs:
                    command = entry.get('command', '')
                    if command:
                        return self._extract_clang_flags(command)

            # --- File not in compile_commands (e.g. header file) ---
            # Collect the union of include paths and defines from all entries
            # so the AST parser can still resolve #include directives.
            all_flags = set()       # deduplicated individual flags
            isystem_paths = set()   # -isystem needs pairing

            for entry in compile_db:
                command = entry.get('command', '')
                if not command:
                    continue
                args = shlex.split(command)
                skip_next = False
                for i, arg in enumerate(args):
                    if skip_next:
                        skip_next = False
                        continue
                    if arg in ['/usr/bin/c++', 'c++', 'g++', 'clang++', '-o', '-c']:
                        continue
                    if arg.startswith('-o'):
                        continue
                    if arg == entry.get('file', ''):
                        continue
                    if arg.startswith('-I') or arg.startswith('-D') or arg.startswith('-std='):
                        all_flags.add(arg)
                    elif arg in ['-isystem'] and i + 1 < len(args):
                        isystem_paths.add(args[i + 1])
                        skip_next = True

            if not all_flags and not isystem_paths:
                return None

            result = sorted(all_flags)
            for path in sorted(isystem_paths):
                result.extend(['-isystem', path])
            return result

        except Exception:
            return None

    @staticmethod
    def _extract_clang_flags(command):
        """Extract -D, -I, -std=, -isystem flags from a compile command string."""
        import shlex
        args = shlex.split(command)
        clang_flags = []
        skip_next = False
        for i, arg in enumerate(args):
            if skip_next:
                skip_next = False
                continue
            if arg in ['/usr/bin/c++', 'c++', 'g++', 'clang++', '-o', '-c']:
                continue
            if arg.startswith('-o'):
                continue
            if arg.startswith('-D') or arg.startswith('-I') or arg.startswith('-std='):
                clang_flags.append(arg)
            elif arg in ['-isystem']:
                clang_flags.append(arg)
                if i + 1 < len(args):
                    clang_flags.append(args[i + 1])
                    skip_next = True
        return clang_flags
