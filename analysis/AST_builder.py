#!/usr/bin/env python3
import json
import shlex
import os
import sys
from clang import cindex
from analysis.Data_analyzer import extract_ddg, extract_def_tree
from instrumentation.InstrumentationCoordinator import InstrumentationCoordinator
DEBUG_FLAG = False

# Set libclang if necessary.
cindex.Config.set_library_file("/usr/lib/llvm-15/lib/libclang-15.so.1")

def is_system_file(file_path):
    """
    Check if a file is from system includes/libraries.
    We skip analyzing these to focus on user code.
    """
    if not file_path:
        return True

    system_prefixes = [
        '/usr/include/',
        '/usr/lib/',
        '/usr/local/include/',
        '/usr/local/lib/',
        '/opt/',
        '/lib/',
    ]

    # Add Emscripten SDK path from environment if available
    emsdk_path = os.environ.get('EMSDK')
    if emsdk_path:
        system_prefixes.append(emsdk_path)

    # Fallback: also check for common emsdk location patterns
    if os.path.exists(os.path.expanduser('~/emsdk')):
        system_prefixes.append(os.path.expanduser('~/emsdk'))

    # Check if it's a system file
    abs_path = os.path.abspath(file_path)
    return any(abs_path.startswith(prefix) for prefix in system_prefixes)

# Global symbol index: maps a USR to a dictionary with definition info and references.
global_index = {}

# Global list to store assignment/instrumentation candidate nodes.
# Each entry is a dict:
# { "usr": ..., "spelling": ..., "file": <filename>,
#   "line": <line>, "column": <col>, "node": <cursor>,
#   "assigned_var": <variable name>, "used_vars": set([...]) }
assignment_nodes = []

def reset_global_state():
    """Reset all global state before starting a new analysis"""
    global global_index, assignment_nodes, global_stmt_id
    global_index = {}
    assignment_nodes = []
    global_stmt_id = 0

def add_to_global_index(cursor):
    """
    Add a symbol definition (the full definition) to the global index using its USR.
    """
    usr = cursor.get_usr()
    if not usr:
        return
    loc = cursor.location
    entry = {
        "spelling": cursor.spelling,
        "kind": str(cursor.kind),
        "file": loc.file.name if loc and loc.file else None,
        "line": loc.line if loc else None,
        "column": loc.column if loc else None,
        "references": global_index.get(usr, {}).get("references", [])
    }
    global_index[usr] = entry

def get_used_variables(node):
    """
    Recursively traverse node and return a set of variable names (spelling)
    that are used (i.e. DECL_REF_EXPR).
    For CALL_EXPR nodes, we also tokenize the extent as a fallback.
    """
    # print("node: ", node.spelling)
    used = set()
    try:
        # If the node is a direct reference, add it.
        if node.kind == cindex.CursorKind.DECL_REF_EXPR and node.spelling != "__assert_fail":
            used.add(node.spelling)
    except ValueError:
        print("Value Error for get used variable")
        return used

    # Special handling for CALL_EXPR: its children might not include all arguments.
    if node.kind == cindex.CursorKind.CALL_EXPR:
        # Fall back to tokenization.
        tokens = list(node.get_tokens())
        for tok in tokens:
            # We assume that identifiers that are not the function name are used variables.
            # For example, if the function is "assert", skip that.
            # if tok.kind == cindex.TokenKind.IDENTIFIER and tok.spelling != "assert":
            # print("tok: ", tok)
            if tok.kind == cindex.TokenKind.IDENTIFIER and tok.spelling != "__assert_fail":
                used.add(tok.spelling)
    # print("node kind: ", node.kind.name)
    # Recurse over children.
    for child in node.get_children():
        used |= get_used_variables(child)
    return used


def collect_assignment_nodes(cursor):
    """
    Traverse the AST and, for each assignment-like instruction,
    record its information in the global assignment_nodes list.
    We consider two cases:
      1. VAR_DECL with an initializer.
      2. BINARY_OPERATOR where one token is an assignment operator and the
         first child (LHS) is a variable.
    """
    try:
        node_kind = cursor.kind
    except ValueError:
        return

    # If this is a declaration that is a definition, record it in the symbol index.
    if cursor.kind.is_declaration() and cursor.is_definition():
        add_to_global_index(cursor)

    # Case 1: variable declaration with initializer.
    if cursor.kind == cindex.CursorKind.VAR_DECL:
        # Check if it has any children (an initializer).
        children = list(cursor.get_children())
        if children:
            loc = cursor.extent.end
            if loc.file and os.path.abspath(loc.file.name) == os.path.abspath(current_filename):
                entry = {
                    "usr": cursor.get_usr(),
                    "spelling": cursor.spelling,
                    "file": loc.file.name,
                    "line": loc.line,
                    "column": loc.column,
                    "node": cursor,
                    "assigned_var": cursor.spelling,
                    # "used_vars": get_used_variables(children[0])
                    "used_vars": set()
                }
                assignment_nodes.append(entry)

    # Case 2: binary assignment operator.
    # For simplicity, we check if any child uses TARGET_VAR? Now we instrument any assignment.
    if cursor.kind in (cindex.CursorKind.BINARY_OPERATOR, cindex.CursorKind.COMPOUND_ASSIGNMENT_OPERATOR):
        # Get tokens to check for assignment operator.
        tokens = list(cursor.get_tokens())
        # If one of the tokens is one of the assignment operators, we treat it as an assignment.
        if any(tok.spelling in ["=", "+=", "-=", "*=", "/="] for tok in tokens):
            children = list(cursor.get_children())
            if children:
                lhs = children[0]
                # For our purposes, if lhs has a non-empty spelling, assume it is a variable.
                if lhs.spelling:
                    loc = cursor.extent.end
                    if loc.file and os.path.abspath(loc.file.name) == os.path.abspath(current_filename):
                        entry = {
                            "usr": cursor.get_usr(),
                            "spelling": cursor.spelling,
                            "file": loc.file.name,
                            "line": loc.line,
                            "column": loc.column,
                            "node": cursor,
                            "assigned_var": lhs.spelling,
                            # "used_vars": get_used_variables(children[1]) if len(children) > 1 else set()
                            "used_vars": set()
                        }
                        assignment_nodes.append(entry)
    # Recurse on children.
    for child in cursor.get_children():
        collect_assignment_nodes(child)


def instrument_assignment(node_entry):
    """
    Generate an instrumentation string that prints the value of the assigned variable.
    Returns a single string (inserted AFTER the statement).
    """
    import hashlib

    defines = node_entry["defines"]
    uses = node_entry["uses"]

    vars_to_print = set(defines + uses)
    line = node_entry["line"]
    file = node_entry["file"]

    # Create a unique marker for this instrumentation point for reliable parsing
    marker = hashlib.md5(f"{file}:{line}".encode()).hexdigest()[:8]

    return f'''/*[Instrumented] Location in the original program: {file}, line {line};
Variables to print: {vars_to_print}; (If the set is empty, please conservatively infer from the context.)
Marker: INST_{marker}
--
TODO: Please print the specified variables with the following EXACT format:
    std::cout << "@@INST_START_{marker}@@" << std::endl;
    // Print each variable and its value here (one per line, e.g., "varname = value")
    std::cout << "@@INST_END_{marker}@@" << std::endl;
Make sure to include the marker strings EXACTLY as shown above, including the @@ symbols.
Please make sure the instrumented code compiles and runs correctly -- Otherwise, the analysis will fail!
*/
'''


def instrument_function_entry(func_name, params, file_path, start_line,
                              output_to_stderr=False, is_c_project=False):
    """
    Generate a TODO block for printing function parameters at entry.
    Inserted right after the opening { of the function body.

    Args:
        output_to_stderr: If True, use ``std::cerr`` instead of ``std::cout``
            in the template.
    """
    import hashlib
    marker = hashlib.md5(f"{file_path}:{start_line}:entry".encode()).hexdigest()[:8]
    func_id = hashlib.md5(f"{file_path}:{start_line}:{func_name}".encode()).hexdigest()[:8]
    params_desc = ", ".join(f"{p['name']}: {p['type']}" for p in params) if params else "(none)"
    # Sanitize comment-header fields against '*/' closing the block early.
    _fn_safe = (func_name or '').replace('*/', '* /')
    _fp_safe = (file_path or '').replace('*/', '* /')
    _pd_safe = params_desc.replace('*/', '* /')

    if is_c_project:
        stream = "stderr" if output_to_stderr else "stdout"
        other_stream = "stdout" if output_to_stderr else "stderr"
        return f'''/*[Instrumented] Function ENTRY: {_fn_safe} at {_fp_safe}, line {start_line};
Parameters: {_pd_safe};
Marker: INST_{marker}
--
TODO: Print the function parameters at entry with the following EXACT format:
    fprintf({stream}, "@@INST_START_{marker}@@\\n");
    fprintf({stream}, "@@FUNC_ID_{func_name}_{func_id}@@\\n");
    fprintf({stream}, "+++Below are Input+++\\n");
    // Print each parameter and its value (one per line, e.g., "paramname = value\\n")
    // For signed integers cast to (long long) and use %lld.
    // For unsigned integers cast to (unsigned long long) and use %llu.
    // For floats/doubles cast to (double) and use %g.
    // For char* use %.200s with a NULL guard: fprintf({stream}, "p=%.200s\\n", p ? p : "(null)");
    // For other pointers use %p: fprintf({stream}, "p=%p\\n", (void*)p);
    // For user-defined struct types call the pre-generated print_<kind>_<name>(FILE*, const ...) helper.
    fprintf({stream}, "@@INST_END_{marker}@@\\n");
Make sure to include the marker strings EXACTLY as shown above, including the @@ symbols.
Use fprintf({stream}, ...) for ALL printing statements (not fprintf({other_stream}, ...)).
Do NOT use std::cout, std::cerr, or any C++ iostream — this is a C project.
Please make sure the instrumented code compiles and runs correctly -- Otherwise, the analysis will fail!
*/
'''

    os = "std::cerr" if output_to_stderr else "std::cout"
    return f'''/*[Instrumented] Function ENTRY: {_fn_safe} at {_fp_safe}, line {start_line};
Parameters: {_pd_safe};
Marker: INST_{marker}
--
TODO: Print the function parameters at entry with the following EXACT format:
    {os} << "@@INST_START_{marker}@@" << std::endl;
    {os} << "@@FUNC_ID_{func_name}_{func_id}@@" << std::endl;
    {os} << "+++Below are Input+++" << std::endl;
    // Print each parameter and its value (one per line, e.g., "paramname = value")
    {os} << "@@INST_END_{marker}@@" << std::endl;
Make sure to include the marker strings EXACTLY as shown above, including the @@ symbols.
Use {os} for ALL printing statements (not {"std::cout" if output_to_stderr else "std::cerr"}).
Please make sure the instrumented code compiles and runs correctly -- Otherwise, the analysis will fail!
*/
'''


def _instrument_function_exit_c(func_name, return_type, return_expr,
                                file_path, ret_line, marker, func_id,
                                params_desc, output_to_stderr):
    """C-mode variant of instrument_function_exit.

    Uses ``fprintf`` with C format specifiers instead of iostream. The
    return-value capture uses a C block-expression pattern: a temporary
    local variable stores the result, prints happen, then the variable
    is returned. Pointer classification mirrors the C++ variant but
    uses ``%.200s`` / ``%p`` format specifiers.
    """
    stream = "stderr" if output_to_stderr else "stdout"
    other_stream = "stdout" if output_to_stderr else "stderr"

    _fn_safe = (func_name or '').replace('*/', '* /')
    _fp_safe = (file_path or '').replace('*/', '* /')
    _rt_safe = (return_type or '').replace('*/', '* /')
    _re_safe = (return_expr or '').replace('*/', '* /')
    _pd_safe = params_desc.replace('*/', '* /')

    _rt = (return_type or '').strip()
    _ptr_depth = 0
    _rt_noptr = _rt
    while _rt_noptr.endswith('*'):
        _rt_noptr = _rt_noptr[:-1].rstrip()
        _ptr_depth += 1
    _is_char_like = ('char' in _rt_noptr) if _ptr_depth >= 1 else False

    if _ptr_depth == 0:
        # Non-pointer: provide a concrete default (signed integer via long
        # long cast). The LLM must swap the specifier/cast if the actual
        # return type is unsigned, floating, or a user-defined struct.
        # NOTE: inline ``// ...`` comments here, never ``/* ... */`` — the
        # whole TODO template is emitted inside a ``/*[Instrumented]...*/``
        # block, so a nested ``*/`` would prematurely terminate the comment.
        ret_block = (
            f'      fprintf({stream}, "ret=%lld\\n", '
            f'(long long)__retval_{marker});  '
            f'// ADJUST: %llu+(unsigned long long) for unsigned, '
            f'%g+(double) for float/double, or call '
            f'print_<kind>_<name>({stream}, &__retval_{marker}); '
            f'then fprintf({stream}, "\\n"); for a user-defined struct'
        )
        ret_guidance = (
            "Return type is NOT a pointer. The template above defaults to "
            "%lld with a (long long) cast, which is correct for all signed "
            "integer types. If the return type is unsigned, change the "
            "specifier to %llu and the cast to (unsigned long long). If "
            "it is a float or double, use %g with a (double) cast. If it "
            "is a user-defined struct/union/enum, replace the fprintf "
            "line with a call to the pre-generated "
            f"print_<kind>_<name>({stream}, &__retval_{marker}); helper."
        )
    elif _ptr_depth == 1 and _is_char_like:
        ret_block = (
            f'      fprintf({stream}, "ret=%p\\n", (const void*)__retval_{marker});\n'
            f'      // char or const char pointer: guard null, cap length to 200.\n'
            f'      if (__retval_{marker}) {{\n'
            f'          fprintf({stream}, " (*ret=%.200s)\\n", __retval_{marker});\n'
            f'      }}'
        )
        ret_guidance = (
            "Return type is a char pointer. Print the address with %p, then "
            "if non-null print the C-string with %.200s (length-capped)."
        )
    elif _ptr_depth >= 2 and _is_char_like:
        ret_block = (
            f'      fprintf({stream}, "ret=%p\\n", (const void*)__retval_{marker});\n'
            f'      if (__retval_{marker} && *__retval_{marker}) {{\n'
            f'          fprintf({stream}, " (*ret=%.200s)\\n", *__retval_{marker});\n'
            f'      }}'
        )
        ret_guidance = (
            "Return type is a char pointer-to-pointer. Print the outer "
            "address with %p, then if both levels are non-null print the "
            "inner C-string with %.200s."
        )
    else:
        ret_block = (
            f'      fprintf({stream}, "ret=%p\\n", (const void*)__retval_{marker});'
        )
        ret_guidance = (
            "Return type is a non-char pointer. Print the address only "
            "with %p. Do NOT dereference."
        )

    return f'''/*[Instrumented] Function EXIT: {_fn_safe} at {_fp_safe}, line {ret_line};
Return type: {_rt_safe}; Return expression: {_re_safe};
Parameters (print again -- may have been mutated via pointer/reference): {_pd_safe};
Marker: INST_{marker}
--
TODO: Capture the return value AND re-print the input parameters. Replace the return statement below with:
    {{
      // C block-expression substitute: declare, print, then return.
      __typeof__({return_expr}) __retval_{marker} = ({return_expr});
      fprintf({stream}, "@@INST_START_{marker}@@\\n");
      fprintf({stream}, "@@FUNC_ID_{func_name}_{func_id}@@\\n");
      fprintf({stream}, "---Below are Outputs---\\n");
{ret_block}
      // Also print each input parameter again (one per line, e.g., "paramname=value\\n")
      // This captures mutations made through pointers/references during the function call.
      fprintf({stream}, "@@INST_END_{marker}@@\\n");
      return __retval_{marker};
    }}
Guidance for this return type: {ret_guidance}
If a parameter is a pointer and you need to print the pointee, ALWAYS null-check
first and cap any string output to 200 chars using fprintf({stream}, "%.200s", ptr).
This avoids UB from printing NULL and from dumping unbounded buffers that
swallow subsequent markers.
Make sure to include the marker strings EXACTLY as shown above, including the @@ symbols.
Use fprintf({stream}, ...) for ALL printing statements (not fprintf({other_stream}, ...)).
Do NOT use std::cout, std::cerr, or any C++ iostream -- this is a C project.
Note: __typeof__ is a GCC/clang extension that works in C mode. If the
compiler rejects it, replace it with the literal return type {return_type}.
Please make sure the instrumented code compiles and runs correctly -- Otherwise, the analysis will fail!
*/
'''


def instrument_function_exit(func_name, return_type, return_expr, file_path, ret_line,
                             params=None, func_start_line=None,
                             output_to_stderr=False, is_c_project=False):
    """
    Generate a TODO block for capturing the return value before a return statement.
    Also prints input parameters again (they may have been mutated via pointer/reference).
    Inserted right before the return statement. The LLM should wrap the return.

    The "ret=" print is tailored to the return type:
      - Non-pointer     → plain `<< __retval`.
      - char* / const char* → address + guarded C-string with 200-char cap.
      - char** / const char** → address + guarded deref-to-C-string with cap.
      - Other pointer types → address only (no dereference).

    Args:
        func_start_line: The function's definition start line (for func_id hash).
                         Falls back to ret_line if not provided.
        output_to_stderr: If True, use ``std::cerr`` instead of ``std::cout``
            in the template.
    """
    import hashlib
    marker = hashlib.md5(f"{file_path}:{ret_line}:exit".encode()).hexdigest()[:8]
    _start = func_start_line if func_start_line is not None else ret_line
    func_id = hashlib.md5(f"{file_path}:{_start}:{func_name}".encode()).hexdigest()[:8]
    params_desc = ", ".join(f"{p['name']}: {p['type']}" for p in params) if params else "(none)"

    if is_c_project:
        return _instrument_function_exit_c(
            func_name, return_type, return_expr, file_path, ret_line,
            marker, func_id, params_desc, output_to_stderr)

    os = "std::cerr" if output_to_stderr else "std::cout"

    # Sanitize any field spliced into the surrounding /* ... */ block so
    # that a stray '*/' in the text cannot prematurely terminate the
    # comment. (Note: `return_expr` is spliced twice -- once as plain text
    # inside the comment header, once as actual C++ inside `auto& __retval
    # = (...)`. The C++ splice stays verbatim; the comment splice uses the
    # sanitized copy.)
    def _sanitize(s):
        return (s or '').replace('*/', '* /')
    _fn_safe = _sanitize(func_name)
    _fp_safe = _sanitize(file_path)
    _rt_safe = _sanitize(return_type)
    _re_safe = _sanitize(return_expr)
    _pd_safe = _sanitize(params_desc)

    # Classify return type: count trailing '*' for pointer depth and check
    # for char-like pointee to decide whether dereference is safe.
    _rt = (return_type or '').strip()
    _ptr_depth = 0
    _rt_noptr = _rt
    while _rt_noptr.endswith('*'):
        _rt_noptr = _rt_noptr[:-1].rstrip()
        _ptr_depth += 1
    _is_char_like = ('char' in _rt_noptr) if _ptr_depth >= 1 else False

    # Build the `ret=` print block conditionally.
    if _ptr_depth == 0:
        ret_block = f'      {os} << "ret=" << __retval_{marker} << std::endl;'
        ret_guidance = "Return type is NOT a pointer — print the value directly."
    elif _ptr_depth == 1 and _is_char_like:
        ret_block = (
            f'      {os} << "ret=" << (void*)__retval_{marker};\n'
            f'      // char or const char pointer: guard null, cap length to 200.\n'
            f'      if (__retval_{marker}) {{\n'
            f'          std::string __s(__retval_{marker});\n'
            f'          if (__s.size() > 200) __s = __s.substr(0, 200) + "...";\n'
            f'          {os} << " (*ret = " << __s << ")";\n'
            f'      }}\n'
            f'      {os} << std::endl;'
        )
        ret_guidance = (
            "Return type is a char pointer (char or const char pointer). "
            "Print the address, then IF non-null, construct a std::string "
            "from it and cap at 200 chars."
        )
    elif _ptr_depth >= 2 and _is_char_like:
        ret_block = (
            f'      {os} << "ret=" << (void*)__retval_{marker};\n'
            f'      // char** or const char**: guard BOTH levels, cap length to 200.\n'
            f'      if (__retval_{marker} && *__retval_{marker}) {{\n'
            f'          std::string __s(*__retval_{marker});\n'
            f'          if (__s.size() > 200) __s = __s.substr(0, 200) + "...";\n'
            f'          {os} << " (*ret = " << __s << ")";\n'
            f'      }}\n'
            f'      {os} << std::endl;'
        )
        ret_guidance = (
            "Return type is a char pointer-to-pointer (char ** or const char **). "
            "Print the address, then IF both pointer and target are non-null, "
            "construct a std::string from the inner pointer and cap at 200 chars."
        )
    else:
        # Non-char pointer: only address, do NOT dereference (would be UB
        # or produce garbage for non-string pointees).
        ret_block = (
            f'      {os} << "ret=" << (void*)__retval_{marker} << std::endl;'
        )
        ret_guidance = (
            "Return type is a non-char pointer. Print the address only. "
            "Do NOT dereference -- the pointee is not guaranteed to be a string."
        )

    return f'''/*[Instrumented] Function EXIT: {_fn_safe} at {_fp_safe}, line {ret_line};
Return type: {_rt_safe}; Return expression: {_re_safe};
Parameters (print again -- may have been mutated via pointer/reference): {_pd_safe};
Marker: INST_{marker}
--
TODO: Capture the return value AND re-print the input parameters. Replace the return statement below with:
    {{ const auto& __retval_{marker} = ({return_expr});
      {os} << "@@INST_START_{marker}@@" << std::endl;
      {os} << "@@FUNC_ID_{func_name}_{func_id}@@" << std::endl;
      {os} << "---Below are Outputs---" << std::endl;
{ret_block}
      // Also print each input parameter again (one per line, e.g., "paramname = value")
      // This captures mutations made through pointers/references during the function call.
      {os} << "@@INST_END_{marker}@@" << std::endl;
      return __retval_{marker}; }}
Guidance for this return type: {ret_guidance}
If a parameter is a pointer and you need to dereference it for printing,
ALWAYS null-check every level AND cap any string output to 200 chars:
    if (ptr && *ptr) {{
        std::string __s(*ptr);
        if (__s.size() > 200) __s = __s.substr(0, 200) + "...";
        {os} << " (*p = " << __s << ")";
    }}
This avoids UB from printing (const char*)nullptr and from dumping
unbounded buffers that swallow subsequent markers.
Make sure to include the marker strings EXACTLY as shown above, including the @@ symbols.
Use {os} for ALL printing statements (not {"std::cout" if output_to_stderr else "std::cerr"}).
Please make sure the instrumented code compiles and runs correctly -- Otherwise, the analysis will fail!
*/
'''


def instrument_void_exit(func_name, file_path, exit_line, params=None,
                         func_start_line=None, output_to_stderr=False,
                         is_c_project=False):
    """
    Generate a TODO block for void function exit (before closing } or void return).
    Also prints input parameters again (they may have been mutated via pointer/reference).

    Args:
        func_start_line: The function's definition start line (for func_id hash).
                         Falls back to exit_line if not provided.
        output_to_stderr: If True, use ``std::cerr`` instead of ``std::cout``
            in the template.
    """
    import hashlib
    marker = hashlib.md5(f"{file_path}:{exit_line}:exit".encode()).hexdigest()[:8]
    _start = func_start_line if func_start_line is not None else exit_line
    func_id = hashlib.md5(f"{file_path}:{_start}:{func_name}".encode()).hexdigest()[:8]
    params_desc = ", ".join(f"{p['name']}: {p['type']}" for p in params) if params else "(none)"
    _fn_safe = (func_name or '').replace('*/', '* /')
    _fp_safe = (file_path or '').replace('*/', '* /')
    _pd_safe = params_desc.replace('*/', '* /')

    if is_c_project:
        stream = "stderr" if output_to_stderr else "stdout"
        other_stream = "stdout" if output_to_stderr else "stderr"
        return f'''/*[Instrumented] Function EXIT (void): {_fn_safe} at {_fp_safe}, line {exit_line};
Parameters (print again -- may have been mutated via pointer/reference): {_pd_safe};
Marker: INST_{marker}
--
TODO: Print a marker for void function exit AND re-print the input parameters:
    fprintf({stream}, "@@INST_START_{marker}@@\\n");
    fprintf({stream}, "@@FUNC_ID_{func_name}_{func_id}@@\\n");
    fprintf({stream}, "---Below are Outputs---\\n");
    fprintf({stream}, "ret=VOID\\n");
    // Also print each input parameter again (one per line, e.g., "paramname=value\\n").
    // Use %lld/(long long) for signed ints, %llu/(unsigned long long) for unsigned ints,
    // %g/(double) for floats, %.200s with NULL guard for char*, %p for other pointers,
    // or print_<kind>_<name>(FILE*, ...) for user-defined struct types.
    fprintf({stream}, "@@INST_END_{marker}@@\\n");
Make sure to include the marker strings EXACTLY as shown above, including the @@ symbols.
Use fprintf({stream}, ...) for ALL printing statements (not fprintf({other_stream}, ...)).
Do NOT use std::cout, std::cerr, or any C++ iostream -- this is a C project.
Please make sure the instrumented code compiles and runs correctly -- Otherwise, the analysis will fail!
*/
'''

    os = "std::cerr" if output_to_stderr else "std::cout"
    return f'''/*[Instrumented] Function EXIT (void): {_fn_safe} at {_fp_safe}, line {exit_line};
Parameters (print again -- may have been mutated via pointer/reference): {_pd_safe};
Marker: INST_{marker}
--
TODO: Print a marker for void function exit AND re-print the input parameters:
    {os} << "@@INST_START_{marker}@@" << std::endl;
    {os} << "@@FUNC_ID_{func_name}_{func_id}@@" << std::endl;
    {os} << "---Below are Outputs---" << std::endl;
    {os} << "ret=VOID" << std::endl;
    // Also print each input parameter again (one per line, e.g., "paramname = value")
    // This captures mutations made through pointers/references during the function call.
    {os} << "@@INST_END_{marker}@@" << std::endl;
Make sure to include the marker strings EXACTLY as shown above, including the @@ symbols.
Use {os} for ALL printing statements (not {"std::cout" if output_to_stderr else "std::cerr"}).
Please make sure the instrumented code compiles and runs correctly -- Otherwise, the analysis will fail!
*/
'''


def _strip_all_comments(lines):
    """
    Return a copy of ``lines`` where every character inside a ``//`` line
    comment or a ``/* ... */`` block comment (including multi-line block
    comments that span multiple list entries) is replaced by a space.
    Newlines are preserved, so line and column indices in the returned
    list align 1:1 with the original. String and character literal
    contents are left untouched.

    Use whenever you need to scan source for braces / parens / semicolons
    without being fooled by comment prose (Doxygen, commented-out code,
    etc.). All other AST_builder helpers that inspect raw punctuation
    should operate on this view rather than the raw lines.
    """
    out = []
    in_block = False
    in_str = False
    str_char = None
    for line in lines:
        buf = list(line)
        n = len(buf)
        i = 0
        while i < n:
            ch = buf[i]
            if in_block:
                if ch == '*' and i + 1 < n and buf[i + 1] == '/':
                    buf[i] = ' '
                    buf[i + 1] = ' '
                    in_block = False
                    i += 2
                    continue
                if ch != '\n':
                    buf[i] = ' '
                i += 1
                continue
            if in_str:
                if ch == '\\' and i + 1 < n:
                    i += 2
                    continue
                if ch == str_char:
                    in_str = False
                i += 1
                continue
            if ch in ('"', "'"):
                in_str = True
                str_char = ch
                i += 1
                continue
            if ch == '/' and i + 1 < n:
                nxt = buf[i + 1]
                if nxt == '/':
                    for k in range(i, n):
                        if buf[k] == '\n':
                            break
                        buf[k] = ' '
                    break
                if nxt == '*':
                    buf[i] = ' '
                    buf[i + 1] = ' '
                    in_block = True
                    i += 2
                    continue
            i += 1
        out.append(''.join(buf))
    return out


def find_opening_brace(lines, start_line):
    """
    Find the line number (1-indexed) of the opening brace of a function body.
    Scans from start_line, tracking paren depth to handle multi-line
    signatures. Braces inside ``//`` line comments and ``/* ... */`` block
    comments (including multi-line Doxygen blocks) are ignored entirely.

    Args:
        lines: list of source lines (0-indexed)
        start_line: 1-indexed line where the function starts

    Returns:
        1-indexed line number of the opening brace, or None if not found.
    """
    scan = _strip_all_comments(lines)
    paren_depth = 0
    for i in range(start_line - 1, min(start_line - 1 + 30, len(scan))):
        for ch in scan[i]:
            if ch == '(':
                paren_depth += 1
            elif ch == ')':
                paren_depth -= 1
            elif ch == '{' and paren_depth == 0:
                return i + 1  # 1-indexed
    return None


def _strip_line_comment(s):
    """Strip C++ line comments (``// ...``) not inside string literals."""
    in_str = False
    for idx in range(len(s) - 1):
        if s[idx] == '"' and (idx == 0 or s[idx-1] != '\\'):
            in_str = not in_str
        if not in_str and s[idx:idx+2] == '//':
            return s[:idx].rstrip()
    return s


def _strip_block_comments(s):
    """Strip ``/* ... */`` block comments from a single line.

    Used during return-expression extraction: when a source line looks like
    ``return expr; /* trailing note */``, the raw ``return_expr`` captured by
    :func:`find_return_statements` would otherwise include the trailing
    comment (including its ``*/``). That ``*/`` then corrupts the
    ``/*[Instrumented]...*/`` TODO comment block and makes the generated
    code unparseable. Strip block comments *before* trimming the semicolon
    so the expression ends cleanly at ``;``.

    Stateful scanner that preserves string and character literal contents
    (so ``return strcmp(s, "/* not a comment */");`` is left alone) and
    respects backslash escapes inside literals. Handles only non-nested
    ``/* ... */`` on a single line; multi-line block comments are not
    combined here — the caller in :func:`find_return_statements` already
    skips lines whose stripped form starts with ``/*``.

    - Unterminated ``/*`` with no matching ``*/``: left as-is (invalid C
      anyway; the function does not try to repair it).
    - Matches :func:`_strip_line_comment`'s string-awareness style.
    """
    out = []
    i = 0
    n = len(s)
    in_str = False
    str_char = ''   # '"' or "'"
    while i < n:
        ch = s[i]
        if in_str:
            out.append(ch)
            # Handle backslash escapes inside string/char literals so that
            # an escaped quote doesn't terminate the literal early.
            if ch == '\\' and i + 1 < n:
                out.append(s[i + 1])
                i += 2
                continue
            if ch == str_char:
                in_str = False
                str_char = ''
            i += 1
            continue
        # Not in a literal.
        if ch == '"' or ch == "'":
            in_str = True
            str_char = ch
            out.append(ch)
            i += 1
            continue
        if ch == '/' and i + 1 < n and s[i + 1] == '*':
            # Find the matching */. If missing, bail out and keep the rest
            # verbatim — this is an unterminated comment, which is invalid
            # C; we don't try to fix it.
            end = s.find('*/', i + 2)
            if end == -1:
                out.append(s[i:])
                i = n
                break
            i = end + 2   # skip past the closing */
            continue
        out.append(ch)
        i += 1
    return ''.join(out)


def find_return_statements(lines, body_start, body_end):
    """
    Find return statements within a function body.

    If a ``return`` is embedded mid-line (e.g. ``if (...) return expr;``),
    the source line is **rewritten in place** to wrap the return in braces
    on its own line so that instrumentation can replace it cleanly::

        // Before:
        if (cond) return expr;  // comment
        // After:
        if (cond) {
            return expr;  // comment
        }

    Args:
        lines: list of source lines (0-indexed).  **May be mutated.**
        body_start: 1-indexed line of opening brace
        body_end: 1-indexed line of closing brace

    Returns:
        List of (line_number_1indexed, return_expression_string) tuples.
        Line numbers account for any lines inserted by mid-line splitting.
    """
    import re
    results = []
    line_offset = 0  # cumulative lines inserted by splitting

    # For single-line functions like "auto begin() { return this; }",
    # body_start == body_end (both point to the brace line).
    # Just extract the return expression — the caller
    # (_register_callee_instrumentation) is responsible for splitting
    # the line BEFORE registering TODOs, so that subsequent functions'
    # line numbers aren't corrupted by the split.
    if body_start == body_end or body_start >= body_end:
        brace_line = lines[body_start - 1] if body_start - 1 < len(lines) else ''
        m = re.search(r'\breturn\b\s+(.+?)\s*;\s*\}', brace_line)
        if m:
            ret_expr = m.group(1).strip()
            results.append((body_start, ret_expr))
        return results

    i = body_start   # start scanning from the line after opening brace

    while i < body_end - 1 + line_offset:  # don't scan the closing brace line
        line = lines[i]
        stripped = line.strip()

        # Skip comments and preprocessor directives
        if stripped.startswith('//') or stripped.startswith('#') or stripped.startswith('/*'):
            i += 1
            continue

        # Look for 'return' keyword (not inside a string)
        match = re.search(r'\breturn\b', stripped)
        if match:
            # Check if the return is mid-line (e.g. "if (...) return expr;")
            # by seeing if there's non-whitespace before the 'return'.
            prefix = stripped[:match.start()].rstrip()
            if prefix:
                # Mid-line return — split into braced block.
                # Original: "  if (cond) return expr;  // comment\n"
                # Becomes:
                #   "  if (cond) {\n"
                #   "      return expr;  // comment\n"
                #   "  }\n"
                indent = line[:len(line) - len(line.lstrip())]
                inner_indent = indent + "    "

                # Everything from 'return' onward (including comment/newline)
                orig_line = lines[i]
                # Find the position of 'return' in the original line
                ret_pos = orig_line.find('return', len(indent) + len(prefix) - len(prefix.lstrip()))
                if ret_pos == -1:
                    ret_pos = orig_line.lower().find('return')
                before_return = orig_line[:ret_pos].rstrip()
                return_part = orig_line[ret_pos:]  # "return expr;  // comment\n"

                new_lines = [
                    before_return + " {\n",
                    inner_indent + return_part.lstrip(),
                    indent + "}\n",
                ]
                # Ensure the return_part line ends with \n
                if not new_lines[1].endswith('\n'):
                    new_lines[1] += '\n'

                lines[i:i+1] = new_lines
                added = len(new_lines) - 1
                line_offset += added

                # Now the return is on lines[i+1]; re-scan from there
                i += 1
                continue

            # Normal return (at start of line) — extract expression.
            # Strip BOTH ``// ...`` line comments and ``/* ... */`` block
            # comments from each source line before scanning for the
            # semicolon. A trailing ``/* note */`` after the semicolon
            # would otherwise be captured in ``full_expr`` and — because
            # its ``*/`` closes the outer ``/*[Instrumented]*/`` TODO
            # comment — corrupt the generated TODO block downstream.
            after_return = stripped[match.end():].strip()
            after_return_no_comment = _strip_line_comment(
                _strip_block_comments(after_return))

            # Handle multi-line return expressions
            expr_parts = [after_return_no_comment]
            scan_line = i
            while not after_return_no_comment.rstrip().endswith(';') \
                    and scan_line < body_end - 2 + line_offset:
                scan_line += 1
                next_line = _strip_line_comment(
                    _strip_block_comments(lines[scan_line].strip()))
                expr_parts.append(next_line)
                after_return_no_comment = next_line

            full_expr = ' '.join(expr_parts).strip()
            # Remove trailing semicolon
            if full_expr.endswith(';'):
                full_expr = full_expr[:-1].strip()

            # 'return;' for void returns
            if not full_expr:
                full_expr = "void"

            results.append((i + 1, full_expr))  # 1-indexed line number
        i += 1

    return results

class ASTBuilder:
    """
    Processes a single file using libclang.
    It builds the AST, collects assignment nodes, and later we use these
    to instrument data dependencies.
    """
    def __init__(self, filename, args, working_directory):
        self.filename = filename
        self.args = args
        self.working_directory = working_directory
        self.tu = None
        self.source = None
        self.modifications = []  # List of tuples: (line_no, insertion_text, position)

    def build_ast(self):
        index = cindex.Index.create()
        full_path = os.path.join(self.working_directory, self.filename) \
                    if not os.path.isabs(self.filename) else self.filename
        self.tu = index.parse(full_path, args=self.args,
                              options=cindex.TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD)
        with open(full_path, "r") as f:
            self.source = f.read()
        # print("root tu.cursor: ", self.tu.cursor.kind)
        # print("working dir: ", self.working_directory)
        # input()

    def is_already_instrumented(self):
        """Check if this file has already been instrumented"""
        return "// [INST:" in self.source or "/*[Instrumented]" in self.source

    def collect_assignments(self):
        global current_filename
        current_filename = os.path.abspath(self.filename)
        collect_assignment_nodes(self.tu.cursor)

    def instrument_source(self):
        # Split the source into a list of lines (preserving newline characters).
        lines = self.source.splitlines(keepends=True)

        # NOTE: Compact one-line function bodies are pre-expanded in
        # Phase 1a-ii.5 (see expand_oneliners.expand_files, driven by
        # the "auto_expansion" flag in metadata.json). No in-builder
        # fallback is performed here.

        def _line_inside_macro(src_lines, target_1indexed):
            """Return True if the 1-indexed line sits inside the body
            of a backslash-continued ``#define`` macro.

            Walk backward from the line above the target across any
            contiguous chain of lines ending with ``\\``.  If ANY line
            in that chain (including the chain's head) begins with
            ``#define``, the target is inside a macro body — inserting
            a multi-line ``/* ... */`` TODO comment there would break
            the preprocessor's line-continuation and make the expansion
            unparseable."""
            idx = target_1indexed - 2
            while idx >= 0:
                raw = src_lines[idx].rstrip('\n')
                stripped = raw.rstrip()
                if stripped.lstrip().startswith('#define'):
                    return True
                if not stripped.endswith('\\'):
                    return False
                idx -= 1
            return False

        # Sort modifications in reverse order by line number so that inserting
        # a later modification does not affect the positions for earlier ones.
        # For the same line, process "after" before "before" (bottom-up insertion
        # means "after" at a higher offset is processed first, then "before").
        self.modifications.sort(
            key=lambda mod: (mod[0], 0 if mod[2] == "after" else -1),
            reverse=True
        )
        # For each modification, insert the instrumentation at the correct position.
        for line_no, instr, position in self.modifications:
            if position == "before":
                # Insert BEFORE the statement's start line (line_no is 1-indexed)
                insert_at = line_no - 1
            else:
                # Insert AFTER the statement's end line (original behavior)
                insert_at = self._find_statement_end_line(lines, line_no)
            # Final-insertion macro guard: _find_statement_end_line can
            # skip past the real function brace when paren-depth gets
            # confused by a multi-line signature, landing the TODO
            # many lines later — potentially inside a ``#define \``
            # macro body.  Even if the originally-registered line was
            # safe, the calculated ``insert_at`` is what matters for
            # correctness.  Drop the insertion if the target line
            # lives in a macro body.
            # insert_at is 0-indexed; convert to 1-indexed for the check.
            if _line_inside_macro(lines, insert_at + 1):
                print(f"  [instrument_source] Dropping "
                      f"instrumentation at {self.filename}:"
                      f"{insert_at + 1} "
                      f"(inside a #define macro body — TODO "
                      f"would break macro continuation)")
                continue
            lines.insert(insert_at, instr + "\n")
        # Join the list of lines back into a single string.
        return "".join(lines)

    def _find_statement_end_line(self, lines, start_line):
        """
        Find the correct line index to insert instrumentation after a statement.

        For function definitions that span multiple lines, finds the line with '{'.
        For regular statements, finds the line ending with ';'.

        Args:
            lines: List of source lines (0-indexed)
            start_line: 1-indexed line number where the statement starts

        Returns:
            Index for lines.insert() - inserts AFTER the statement end
        """
        # Convert 1-indexed to 0-indexed for array access
        idx = start_line - 1

        if idx < 0 or idx >= len(lines):
            return start_line  # Fallback to original behavior

        # Fast path for function-entry registrations: the caller
        # (_register_callee_instrumentation) passes brace_line — the
        # exact line already holding the function-body ``{``.  Trust
        # that.  The old forward-scan-with-paren-depth logic gets
        # confused by a multi-line signature's closing ``)`` that
        # lands on this same line (e.g. ``... expected_ec) {``):
        # paren_depth goes negative before the ``{`` is seen, so the
        # ``{`` gets skipped and the scanner walks into subsequent
        # lines — frequently landing inside a nested macro body.
        if '{' in lines[idx]:
            return idx + 1

        # Track parentheses to handle multi-line function signatures
        paren_depth = 0
        in_string = False
        string_char = None

        # Scan from start_line forward to find statement end
        for i in range(idx, min(idx + 30, len(lines))):  # Look up to 30 lines ahead
            line = lines[i]

            for j, char in enumerate(line):
                # Handle string literals (skip characters inside strings)
                if char in ('"', "'") and (j == 0 or line[j-1] != '\\'):
                    if not in_string:
                        in_string = True
                        string_char = char
                    elif char == string_char:
                        in_string = False
                    continue

                if in_string:
                    continue

                # Track parentheses depth
                if char == '(':
                    paren_depth += 1
                elif char == ')':
                    paren_depth -= 1
                elif char == '{' and paren_depth == 0:
                    # Found opening brace - insert AFTER this line (inside function body)
                    return i + 1
                elif char == ';' and paren_depth == 0:
                    # Found statement end - insert after this line
                    return i + 1

        # Fallback: insert after original line
        return start_line

    def process_file(self, dep_stmts):
        self.build_ast()

        # Check if file is already instrumented
        if self.is_already_instrumented():
            print(f"File {self.filename} is already instrumented - skipping")
            return ""

        self.collect_assignments()
        # For each assignment node in our slice (that belongs to this file),
        # add a modification at the appropriate position.
        for entry in dep_stmts:
            if os.path.abspath(entry["file"]) == os.path.abspath(self.filename):
                line = entry['line']
                position = entry.get('position', 'after')
                # Use pre-generated instrumentation if available, otherwise generate it
                instr = entry.get('instrumentation') or instrument_assignment(entry)
                if isinstance(instr, dict):
                    # Split before/after from instrument_assignment for call entries
                    if 'before' in instr:
                        self.modifications.append((line, instr['before'], 'before'))
                    if 'after' in instr:
                        self.modifications.append((line, instr['after'], 'after'))
                else:
                    self.modifications.append((line, instr, position))
        if self.modifications:
            modified_source = self.instrument_source()
            return modified_source
        else:
            print(f"No modifications found in {self.filename}.")
            return ""

class ProjectASTBuilder:
    """
    Loads compile_commands.json and processes every file entry.
    Uses the compilation commands to obtain the right flags and working directory,
    and then after building ASTs for all files, performs data-dependence slicing.
    """
    def __init__(self, compile_commands_path, root_file, root_line, root_func_name):
        self.compile_commands_path = compile_commands_path
        self.commands = []
        self.root_file = os.path.abspath(root_file)
        self.root_line = int(root_line)
        self.root_func_name = root_func_name
        # Mapping from filename to ASTBuilder instance.
        self.file_builders = {}
        self.builder = None
        self.ddg = extract_ddg(compdb_path=self.compile_commands_path, root_dir="")
        self.coordinator = InstrumentationCoordinator()  # Multi-file coordinator
        self.function_metadata = {}  # func_id -> per-function metadata dict

    def load_compile_commands(self, filter_system_files=True):
        """
        Load compile_commands.json and optionally filter out system files.

        Args:
            filter_system_files: If True, skip files from system directories
        """
        with open(self.compile_commands_path, "r") as f:
            all_commands = json.load(f)

        if filter_system_files:
            # Filter out system files
            filtered_commands = []
            skipped_system = 0

            for entry in all_commands:
                file_path = entry["file"]
                # Make it absolute
                if not os.path.isabs(file_path):
                    file_path = os.path.join(entry["directory"], file_path)

                if is_system_file(file_path):
                    skipped_system += 1
                else:
                    filtered_commands.append(entry)

            self.commands = filtered_commands
            print(f"Loaded {len(all_commands)} compilation commands")
            print(f"Filtered out {skipped_system} system files")
            print(f"Processing {len(self.commands)} user source files")
        else:
            self.commands = all_commands
            print(f"Loaded {len(all_commands)} compilation commands")

    def get_enclosing_function(self, cursor):
        while cursor is not None:
            if cursor.kind == cindex.CursorKind.FUNCTION_DECL:
                return cursor
            cursor = cursor.semantic_parent  # or use lexical_parent if appropriate
        return None

    def _fallback_instrument_enclosing_function(self, root_cursor, root_builder):
        """
        Fallback when extract_def_tree returns None (e.g., macro lines like EXPECT_EQ).

        Finds the enclosing function of root_cursor and collects all DDG statements
        within that function's line range, so they can still be instrumented.

        Returns:
            List of dep_stmt dicts, or empty list if nothing found.
        """
        enclosing = self.get_enclosing_function(root_cursor)
        if not enclosing:
            print("  Could not find enclosing function for fallback.")
            return []

        func_name = enclosing.spelling
        start_line = enclosing.extent.start.line
        end_line = enclosing.extent.end.line
        print(f"  Enclosing function: {func_name} (lines {start_line}-{end_line})")

        # Collect all DDG statements within the enclosing function's line range
        fallback_stmts = []
        for stmt in self.ddg.get("statements", []):
            if stmt["file"] == self.root_file and start_line <= stmt["line"] <= end_line:
                fallback_stmts.append(stmt)

        if fallback_stmts:
            print(f"  Found {len(fallback_stmts)} DDG statement(s) in enclosing function")
        else:
            # If DDG has nothing either, create a minimal entry for the target line
            print(f"  No DDG statements in function range, creating minimal entry for line {self.root_line}")
            fallback_stmts = [{
                "file": self.root_file,
                "line": self.root_line,
                "defines": [],
                "uses": [],
                "calls": [],
            }]

        return fallback_stmts

    def _collect_compile_args_from_commands(self):
        """
        Extract the union of -I, -D, -std flags from all entries in self.commands.
        Returns (args_list, directory) suitable for ASTBuilder or index.parse().
        """
        args_set = set()
        args = []
        directory = None
        std_flag = None

        for cmd in self.commands:
            cmd_args = shlex.split(cmd["command"])[1:]

            i = 0
            while i < len(cmd_args):
                arg = cmd_args[i]

                if arg.startswith("-I") and len(arg) > 2:
                    if arg not in args_set:
                        args.append(arg)
                        args_set.add(arg)
                elif arg == "-I" and i + 1 < len(cmd_args):
                    combined = f"-I{cmd_args[i + 1]}"
                    if combined not in args_set:
                        args.append("-I")
                        args.append(cmd_args[i + 1])
                        args_set.add(combined)
                    i += 1
                elif arg.startswith("-D"):
                    if arg not in args_set:
                        args.append(arg)
                        args_set.add(arg)
                elif arg.startswith("-std="):
                    std_flag = arg

                i += 1

            if directory is None:
                directory = cmd["directory"]

        if std_flag:
            args.append(std_flag)
        elif "-std=c++17" not in args:
            args.append("-std=c++17")

        return args, directory or ""

    @staticmethod
    def _collect_decl_refs_on_line(cursor, line, file_path):
        """
        Walk cursor's AST subtree and return DECL_REF_EXPR nodes whose
        location matches the given line and file.
        """
        results = []
        if cursor.location.file and os.path.abspath(cursor.location.file.name) == file_path:
            if cursor.kind == cindex.CursorKind.DECL_REF_EXPR and cursor.location.line == line:
                results.append(cursor)
        for child in cursor.get_children():
            results.extend(ProjectASTBuilder._collect_decl_refs_on_line(child, line, file_path))
        return results

    @staticmethod
    def _find_call_exprs_in_subtree(cursor):
        """
        Walk cursor's AST subtree and return all CALL_EXPR nodes.
        """
        results = []
        if cursor.kind == cindex.CursorKind.CALL_EXPR:
            results.append(cursor)
        for child in cursor.get_children():
            results.extend(ProjectASTBuilder._find_call_exprs_in_subtree(child))
        return results

    @staticmethod
    def _find_function_definition_in_tu(tu, func_name):
        """
        Walk a translation unit's AST for a FUNCTION_DECL or CXX_METHOD
        whose spelling matches func_name and is_definition() is True.

        Prefers definitions located in the **main file** (the .cc file
        being parsed) over definitions from included headers, to avoid
        picking up unrelated system functions with the same name.

        Returns the cursor or None.
        """
        _FUNC_KINDS = (
            cindex.CursorKind.FUNCTION_DECL,
            cindex.CursorKind.CXX_METHOD,
            cindex.CursorKind.CONSTRUCTOR,
            cindex.CursorKind.DESTRUCTOR,
            cindex.CursorKind.FUNCTION_TEMPLATE,
        )
        main_file = tu.spelling  # the .cc file path

        best = None       # best match from any file
        best_local = None  # best match from the main file

        def _walk(cursor):
            nonlocal best, best_local
            if (cursor.kind in _FUNC_KINDS
                    and cursor.spelling == func_name
                    and cursor.is_definition()):
                if best is None:
                    best = cursor
                loc_file = cursor.location.file
                if loc_file and os.path.abspath(loc_file.name) == os.path.abspath(main_file):
                    if best_local is None:
                        best_local = cursor
            for child in cursor.get_children():
                _walk(child)

        _walk(tu.cursor)
        return best_local or best

    def _trace_variable_producing_calls(self, dep_stmts, root_cursor, call_graph):
        """
        Trace variables used on the root line back to their definitions,
        and ensure any function calls involved in producing those variables
        are included in the instrumentation plan.

        Uses the libclang AST (DECL_REF_EXPR → referenced VAR_DECL → child
        CALL_EXPR) instead of regex.

        Args:
            dep_stmts: Current list of dependency statements.
            root_cursor: The AST cursor for the root line.
            call_graph: The call_graph dict (may be None).

        Returns:
            dep_stmts extended with any new entries for variable-producing calls.
        """
        enclosing = self.get_enclosing_function(root_cursor)
        if not enclosing:
            return dep_stmts

        root_file_abs = os.path.abspath(self.root_file)

        # Collect DECL_REF_EXPR nodes on the root line
        decl_refs = self._collect_decl_refs_on_line(enclosing, self.root_line, root_file_abs)
        if not decl_refs:
            return dep_stmts

        # Collect function names already covered in dep_stmts
        covered_func_names = set()
        for stmt in dep_stmts:
            for call in stmt.get('calls', []):
                covered_func_names.add(call.get('name', ''))

        new_entries = []
        seen_vars = set()

        for ref in decl_refs:
            referenced = ref.referenced
            if referenced is None:
                continue
            # Skip parameters — we only want local variable definitions
            if referenced.kind == cindex.CursorKind.PARM_DECL:
                continue
            if referenced.kind != cindex.CursorKind.VAR_DECL:
                continue

            var_name = referenced.spelling
            if var_name in seen_vars:
                continue
            seen_vars.add(var_name)

            var_def_line = referenced.extent.start.line

            # Walk the VAR_DECL subtree for CALL_EXPR nodes
            call_exprs = self._find_call_exprs_in_subtree(referenced)
            for call_cursor in call_exprs:
                # Determine the called function's name
                called = call_cursor.referenced
                if called is not None:
                    func_name = called.spelling
                else:
                    func_name = call_cursor.spelling

                if not func_name or func_name in covered_func_names:
                    continue

                call_entry = {'name': func_name, 'usr': None, 'file': None,
                              'line': None, 'kind': 'CALL_EXPR'}

                # Try to resolve USR from libclang referenced cursor
                if called is not None and called.get_usr():
                    call_entry['usr'] = called.get_usr()
                    if called.location.file:
                        call_entry['file'] = os.path.abspath(called.location.file.name)
                        call_entry['line'] = called.location.line

                # Fall back to call_graph lookup
                if call_entry['usr'] is None and call_graph:
                    cg_functions = call_graph.get('functions', {})
                    for usr, finfo in cg_functions.items():
                        if finfo.get('name') == func_name and finfo.get('is_definition'):
                            call_entry['usr'] = usr
                            call_entry['file'] = finfo.get('file')
                            call_entry['line'] = finfo.get('line')
                            break

                synthetic_entry = {
                    'file': self.root_file,
                    'line': var_def_line,
                    'defines': [var_name],
                    'uses': [],
                    'calls': [call_entry],
                }
                new_entries.append(synthetic_entry)
                covered_func_names.add(func_name)

                print(f"  [variable-trace] Variable '{var_name}' at line {var_def_line} -> {func_name}()")

        if new_entries:
            print(f"  [variable-trace] Found {len(new_entries)} additional function call(s) via variable tracing")

        return dep_stmts + new_entries

    def _supplement_root_line_calls(self, dep_stmts, root_cursor, call_graph):
        """Scan the AST at the root line for all call expressions.

        The DDG-based dependency slice only captures calls that participate
        in a data-flow chain.  Void side-effect calls (e.g. ``fmt::print``)
        are invisible to the DDG because they don't define any variable.

        This method walks the AST for every ``CALL_EXPR`` whose location
        is on the root line, and adds synthetic dep_stmt entries for any
        calls not already covered.

        Returns the (possibly extended) dep_stmts list.
        """
        # Try enclosing function first; fall back to the full TU cursor
        # (needed when the root line is inside a macro like TEST() which
        # creates a CXX_METHOD whose semantic_parent chain is broken).
        root_builder = self.file_builders.get(os.path.abspath(self.root_file))
        if not root_builder:
            return dep_stmts
        search_root = self.get_enclosing_function(root_cursor)
        if not search_root:
            search_root = root_builder.tu.cursor

        root_file_abs = os.path.abspath(self.root_file)

        # Collect names already covered
        covered = set()
        for stmt in dep_stmts:
            for c in stmt.get('calls', []):
                covered.add(c.get('name', ''))

        # Walk the enclosing function AST for CALL_EXPRs on the root line.
        # We check the *spelling* location so that calls inside macro
        # expansions are attributed to the line where the macro was invoked.
        call_cursors = []

        def _collect(cursor):
            loc = cursor.location
            if loc.file and os.path.abspath(loc.file.name) == root_file_abs:
                if loc.line == self.root_line and cursor.kind == cindex.CursorKind.CALL_EXPR:
                    call_cursors.append(cursor)
            for child in cursor.get_children():
                _collect(child)

        _collect(search_root)

        cg_functions = (call_graph or {}).get('functions', {})
        new_entries = []

        for call_cursor in call_cursors:
            called = call_cursor.referenced
            func_name = (called.spelling if called else call_cursor.spelling)
            if not func_name or func_name in covered:
                continue

            call_entry = {
                'name': func_name,
                'usr': None,
                'file': None,
                'line': None,
                'kind': str(call_cursor.kind),
            }

            # Resolve from libclang
            if called and called.get_usr():
                call_entry['usr'] = called.get_usr()
                if called.location.file:
                    call_entry['file'] = os.path.abspath(
                        called.location.file.name)
                    call_entry['line'] = called.location.line

            # Fall back to call_graph lookup
            if call_entry['usr'] is None:
                for usr, finfo in cg_functions.items():
                    if finfo.get('name') == func_name and finfo.get('is_definition'):
                        call_entry['usr'] = usr
                        call_entry['file'] = finfo.get('file')
                        call_entry['line'] = finfo.get('line')
                        break

            new_entries.append({
                'file': self.root_file,
                'line': self.root_line,
                'defines': [],
                'uses': [],
                'calls': [call_entry],
            })
            covered.add(func_name)
            print(f"  [root-line-calls] Found call to "
                  f"'{func_name}' at root line {self.root_line}")

        if new_entries:
            print(f"  [root-line-calls] Added {len(new_entries)} "
                  f"call(s) from direct AST scan of root line")

        return dep_stmts + new_entries

    def _resolve_declaration_to_definition(self, func_name, decl_info, cg_functions):
        """
        When a function has is_definition=False (declaration only, e.g. in a .h),
        try to find its definition by parsing candidate .cc/.cpp files with
        libclang and walking the AST for a matching FUNCTION_DECL.

        Args:
            func_name: The function name to search for.
            decl_info: The call_graph entry for the declaration.
            cg_functions: All functions from the call graph (unused, kept for API compat).

        Returns:
            A dict with file, line, end_line, name, params, return_type, etc.
            or None if not found.
        """
        decl_file = decl_info.get('file', '')
        if not decl_file:
            return None

        # Only resolve .h/.hpp declarations
        _, ext = os.path.splitext(decl_file)
        if ext not in ('.h', '.hpp', '.hxx'):
            return None

        # Look for matching source files
        base_dir = os.path.dirname(decl_file)
        stem = os.path.splitext(os.path.basename(decl_file))[0]
        source_exts = ['.cc', '.cpp', '.cxx', '.c']

        candidates = []
        for src_ext in source_exts:
            candidate = os.path.join(base_dir, stem + src_ext)
            if os.path.exists(candidate):
                candidates.append(candidate)

        # Also search parent/sibling directories and common source
        # directories (src/, lib/) relative to the project root.
        # The project root is inferred from the compile_commands.json
        # location (typically build_native/ or build/).
        cc_dir = os.path.dirname(os.path.abspath(
            self.compile_commands_path))
        project_root = os.path.dirname(cc_dir)
        search_dirs = [os.path.dirname(base_dir), base_dir]
        for subdir in ('src', 'lib', 'source'):
            sd = os.path.join(project_root, subdir)
            if os.path.isdir(sd):
                search_dirs.append(sd)
        for search_dir in search_dirs:
            if not os.path.isdir(search_dir):
                continue
            for src_ext in source_exts:
                candidate = os.path.join(search_dir, stem + src_ext)
                if os.path.exists(candidate) and candidate not in candidates:
                    candidates.append(candidate)
            for entry in os.listdir(search_dir):
                full = os.path.join(search_dir, entry)
                if os.path.isdir(full):
                    for src_ext in source_exts:
                        candidate = os.path.join(full, stem + src_ext)
                        if os.path.exists(candidate) and candidate not in candidates:
                            candidates.append(candidate)

        if not candidates:
            return None

        # Collect compile args for parsing candidate files
        compile_args, _ = self._collect_compile_args_from_commands()

        index = cindex.Index.create()

        for src_file in candidates:
            try:
                tu = index.parse(src_file, args=compile_args)
            except Exception:
                continue

            func_cursor = self._find_function_definition_in_tu(tu, func_name)
            if func_cursor is None:
                continue

            start_line = func_cursor.extent.start.line
            end_line = func_cursor.extent.end.line

            # Extract params from the AST cursor
            params = []
            for arg in func_cursor.get_arguments():
                params.append({
                    'name': arg.spelling,
                    'type': arg.type.spelling,
                })
            return_type = func_cursor.result_type.spelling

            abs_src = os.path.abspath(src_file)
            print(f"    [resolve] Found definition of '{func_name}' "
                  f"in {os.path.basename(abs_src)}:{start_line}-{end_line}")

            return {
                'name': func_name,
                'qualified_name': decl_info.get('qualified_name', func_name),
                'file': abs_src,
                'line': start_line,
                'end_line': end_line,
                'is_definition': True,
                'params': params,
                'return_type': return_type,
            }

        return None

    def _register_callee_instrumentation(self, dep_entry, call_graph,
                                          output_to_stderr=False,
                                          is_c_project=False):
        """
        For a dep_stmt with function calls, instrument inside the called function bodies.

        For each called function found in the call_graph:
          - Add entry instrumentation (print params) after opening {
          - Add exit instrumentation (print return value) before each return statement

        Args:
            dep_entry: A dep_stmt dict with non-empty 'calls' field.
            output_to_stderr: If True, generated TODO blocks use std::cerr.
            call_graph: The loaded call_graph.json dict.
        """
        cg_functions = call_graph.get('functions', {})
        instrumented_usrs = set()  # avoid instrumenting same function twice

        for call_info in dep_entry.get('calls', []):
            call_usr = call_info.get('usr')
            call_name = call_info.get('name', 'unknown')

            if not call_usr or call_usr in instrumented_usrs:
                continue

            func_info = cg_functions.get(call_usr)
            if not func_info:
                print(f"    [callee] Function '{call_name}' not found in call graph, skipping")
                continue

            if not func_info.get('is_definition'):
                # Try to find definition in a matching source file
                resolved = self._resolve_declaration_to_definition(
                    call_name, func_info, cg_functions)
                if resolved:
                    func_info = resolved
                    print(f"    [callee] Resolved '{call_name}' declaration to "
                          f"definition at {os.path.basename(resolved['file'])}:"
                          f"{resolved['line']}")
                else:
                    print(f"    [callee] Function '{call_name}' has no definition body, skipping")
                    continue

            func_file = func_info['file']
            if is_system_file(func_file):
                continue

            # # Skip gtest framework files — no need to instrument test infrastructure
            func_file_abs = os.path.abspath(func_file)
            # func_file_lower = func_file_abs.lower()
            # if ('gtest' in func_file_lower or 'gmock' in func_file_lower or
            #         '/gtest/' in func_file_lower or '/gmock/' in func_file_lower):
            #     print(f"    [callee] Skipping gtest/gmock function '{call_name}' "
            #           f"in {os.path.basename(func_file)}")
            #     continue

            func_name = func_info.get('qualified_name', func_info['name'])
            params = func_info.get('params', [])
            return_type = func_info.get('return_type', 'void')
            start_line = func_info['line']
            end_line = func_info['end_line']
            # Preserve pre-split (original-file) line numbers so that
            # Phase 4 (which runs against files reverted to their
            # pristine pre-instrumentation state) can locate functions
            # by their original positions.
            original_start_line = start_line
            original_end_line = end_line

            # Apply accumulated line offset from single-line function
            # splits in the same file.  Each split adds 6 lines;
            # functions below the split point need their line numbers
            # adjusted.
            if not hasattr(self, '_file_line_offsets'):
                self._file_line_offsets = {}  # file -> [(split_line, offset)]
            for split_line, offset in self._file_line_offsets.get(
                    func_file_abs, []):
                if start_line >= split_line:
                    start_line += offset
                if end_line >= split_line:
                    end_line += offset

            # Read the source file
            try:
                with open(func_file_abs, 'r') as f:
                    lines = f.readlines()
            except Exception as e:
                print(f"    [callee] Could not read {func_file_abs}: {e}")
                continue

            # ---- Guard: skip implicit constructors / destructors ----
            # Compilers auto-generate copy/move ctors, default ctors, and
            # dtors.  libclang reports them with is_definition=True but
            # their location points to the *class* definition line, not a
            # real function body.  Instrumenting the class body as if it
            # were a function body corrupts the source.
            #
            # Detection: CallGraphBuilder sets is_implicit=True when a
            # ctor/dtor/method's source location is identical (line AND
            # column) to its parent class definition.
            if func_info.get('is_implicit', False):
                print(f"    [callee] Skipping implicit '{call_name}' at "
                      f"{os.path.basename(func_file)}:{start_line} "
                      f"(compiler-generated, no real function body)")
                continue

            # Skip explicitly defaulted/deleted special members — they
            # have no body to instrument either.
            if start_line == end_line:
                _line_text = lines[start_line - 1] if start_line <= len(lines) else ''
                if '= default' in _line_text or '= delete' in _line_text:
                    print(f"    [callee] Skipping defaulted/deleted "
                          f"'{call_name}' at "
                          f"{os.path.basename(func_file)}:{start_line}")
                    continue

            # Note: constexpr/consteval specifiers are stripped by the build system
            # during instrumentation, so we proceed with instrumenting these functions.
            decl_text = ''.join(lines[max(0, start_line - 3):start_line + 3])
            _decl_lower = decl_text.lower()
            if ('constexpr' in _decl_lower or 'consteval' in _decl_lower or
                    'FMT_CONSTEVAL' in decl_text or 'FMT_CONSTEXPR' in decl_text):
                print(f"    [callee] constexpr/consteval function '{call_name}' "
                      f"at {os.path.basename(func_file)}:{start_line} — "
                      f"build system will strip constexpr for instrumentation")

            # Find opening brace of function body
            brace_line = find_opening_brace(lines, start_line)
            if brace_line is None:
                print(f"    [callee] Could not find opening brace for {call_name} at {func_file}:{start_line}")
                continue

            # ---- Guard: verify brace belongs to a function, not a class ----
            # Scan text from start_line to brace_line for parentheses —
            # every real function signature has '(' .. ')' before '{'.
            _sig_text = ''.join(lines[start_line - 1:brace_line])
            if '(' not in _sig_text:
                print(f"    [callee] Skipping '{call_name}' at "
                      f"{os.path.basename(func_file)}:{start_line} "
                      f"(no function signature before opening brace)")
                continue

            # ---- Guard: skip if the insertion site is inside a
            # backslash-continued macro body. Multi-line /* */ TODO
            # comments placed inside a ``#define ... \`` block break
            # the preprocessor: the comment's non-``\``-ended lines
            # terminate the macro mid-body, producing a malformed
            # expansion that fails to compile in Step 1b.  Detect by
            # scanning backward for an unbroken chain of lines ending
            # in ``\`` until we hit either a non-continued line
            # (safe) or a ``#define`` (inside a macro — skip).
            def _is_inside_macro(src_lines, target_1indexed):
                """Return True when ``target_1indexed`` lives inside a
                backslash-continued ``#define`` macro body.

                Walk backward from the line above the target while
                each line ends with ``\\``.  Check EVERY line (not
                just the one that ends the chain) for a leading
                ``#define`` — the ``#define`` line itself ends with
                ``\\`` so it IS part of the continued chain.  Stop at
                the first line that does not end with ``\\``."""
                idx = target_1indexed - 2   # line above the target
                while idx >= 0:
                    raw = src_lines[idx].rstrip('\n')
                    stripped = raw.rstrip()
                    if stripped.lstrip().startswith('#define'):
                        return True
                    if not stripped.endswith('\\'):
                        # Chain broken by a non-continued line — we
                        # never saw a ``#define``, so not in a macro.
                        return False
                    idx -= 1
                return False

            if _is_inside_macro(lines, brace_line):
                print(f"    [callee] Skipping '{call_name}' at "
                      f"{os.path.basename(func_file)}:{start_line} "
                      f"(inside a #define macro body — "
                      f"TODO comments would break macro continuation)")
                continue

            # Recompute end_line by counting braces from the opening
            # brace.  Clang's cursor.extent.end.line can be wrong for
            # short inline methods (reports the signature line, not the
            # closing brace), causing find_return_statements to miss
            # return statements that are one line below.
            _depth = 0
            _actual_end = end_line
            for _bi in range(brace_line - 1, len(lines)):
                for _ch in lines[_bi]:
                    if _ch == '{':
                        _depth += 1
                    elif _ch == '}':
                        _depth -= 1
                        if _depth == 0:
                            _actual_end = _bi + 1  # 1-indexed
                            break
                if _depth == 0:
                    break
            end_line = _actual_end

            # ---- Split single-line functions BEFORE registering TODOs ----
            # For "auto begin() -> iterator { return this; }" where the
            # entire body is on one line, split into 3 lines so that
            # both entry and exit instrumentation have room.  This MUST
            # happen before any TODO registration because it shifts all
            # subsequent line numbers in the file.
            if brace_line == end_line:
                import re as _re
                _brace_text = lines[brace_line - 1]
                _ret_m = _re.search(
                    r'\breturn\b\s+(.+?)\s*;\s*\}', _brace_text)
                if _ret_m:
                    _indent = len(_brace_text) - len(_brace_text.lstrip())
                    _inner = ' ' * (_indent + 4)
                    _brace_pos = _brace_text.find('{')
                    _sig = _brace_text[:_brace_pos + 1]
                    _ret_expr = _ret_m.group(1).strip()
                    # Add 2 blank lines after `{` and 2 blank lines
                    # before `}` to give TODO blocks room and avoid
                    # line-offset collisions with neighboring TODOs.
                    lines[brace_line - 1:brace_line] = [
                        _sig + '\n',
                        '\n',
                        '\n',
                        f'{_inner}return {_ret_expr};\n',
                        '\n',
                        '\n',
                        ' ' * _indent + '}\n',
                    ]
                    end_line = brace_line + 6  # now 7 lines
                    # Write back immediately so the file on disk
                    # matches the in-memory state for this and
                    # all subsequent functions in the same file.
                    with open(func_file_abs, 'w') as _wf:
                        _wf.writelines(lines)
                    # Record the offset so subsequent functions in
                    # the same file adjust their line numbers.
                    if not hasattr(self, '_file_line_offsets'):
                        self._file_line_offsets = {}
                    self._file_line_offsets.setdefault(
                        func_file_abs, []).append(
                            (brace_line, 6))  # +6 lines added

            print(f"    [callee] Instrumenting {call_name} at {func_file}:{start_line}-{end_line}")

            # Entry instrumentation: after opening {
            entry_instr = instrument_function_entry(
                func_name, params, func_file_abs, start_line,
                output_to_stderr=output_to_stderr,
                is_c_project=is_c_project)
            if self.coordinator.can_instrument(func_file_abs, brace_line, "after"):
                self.coordinator.register_instrumentation(func_file_abs, brace_line, entry_instr, "after")

            # Find return statements
            returns = find_return_statements(lines, brace_line, end_line)

            def _skip_if_macro(ret_ln, kind_label):
                """Skip the TODO for this exit site when it lands inside
                a ``#define ... \\`` macro body. ``find_return_statements``
                picks up every ``return`` keyword in the function's
                brace-matched extent, including those belonging to
                nested lambdas that happen to sit inside a macro — a
                TODO comment there would break macro continuation."""
                if _is_inside_macro(lines, ret_ln):
                    print(f"    [callee] Skipping exit "
                          f"({kind_label}) for '{call_name}' at "
                          f"{os.path.basename(func_file)}:{ret_ln} "
                          f"(inside a #define macro body)")
                    return True
                return False

            if return_type and return_type != 'void':
                # Non-void function: instrument each return statement
                for ret_line, ret_expr in returns:
                    if _skip_if_macro(ret_line, 'non-void return'):
                        continue
                    exit_instr = instrument_function_exit(
                        func_name, return_type, ret_expr, func_file_abs, ret_line,
                        params=params, func_start_line=start_line,
                        output_to_stderr=output_to_stderr,
                        is_c_project=is_c_project)
                    if self.coordinator.can_instrument(func_file_abs, ret_line, "before"):
                        self.coordinator.register_instrumentation(func_file_abs, ret_line, exit_instr, "before")
            else:
                # Void function: instrument each explicit return
                for ret_line, ret_expr in returns:
                    if _skip_if_macro(ret_line, 'void return'):
                        continue
                    exit_instr = instrument_void_exit(
                        func_name, func_file_abs, ret_line,
                        params=params, func_start_line=start_line,
                        output_to_stderr=output_to_stderr,
                        is_c_project=is_c_project)
                    if self.coordinator.can_instrument(func_file_abs, ret_line, "before"):
                        self.coordinator.register_instrumentation(func_file_abs, ret_line, exit_instr, "before")

                # Check if the last 2 lines before the closing } contain a return.
                # If not, the function can fall through without returning, so add
                # an exit block before the closing }.
                last_2_lines = [lines[i].strip() for i in range(max(brace_line, end_line - 3), end_line - 1)
                                if i < len(lines)]
                import re
                has_return_at_end = any(re.search(r'\breturn\b', l) for l in last_2_lines)
                if not has_return_at_end and not _skip_if_macro(
                        end_line, 'void fall-through'):
                    exit_instr = instrument_void_exit(
                        func_name, func_file_abs, end_line,
                        params=params, func_start_line=start_line,
                        output_to_stderr=output_to_stderr,
                        is_c_project=is_c_project)
                    if self.coordinator.can_instrument(func_file_abs, end_line, "before"):
                        self.coordinator.register_instrumentation(func_file_abs, end_line, exit_instr, "before")

            # ---- Build structured metadata for downstream consumers ----
            import hashlib as _hl
            _func_id = _hl.md5(
                f"{func_file_abs}:{start_line}:{func_name}".encode()
            ).hexdigest()[:8]

            meta_blocks = []
            # Entry block
            _entry_marker = _hl.md5(
                f"{func_file_abs}:{start_line}:entry".encode()
            ).hexdigest()[:8]
            meta_blocks.append({
                "block_type": "entry",
                "marker": _entry_marker,
                "return_expr": None,
                "return_type": None,
            })

            # Exit blocks (mirrors the registration logic above)
            if return_type and return_type != 'void':
                for ret_line, ret_expr in returns:
                    _exit_marker = _hl.md5(
                        f"{func_file_abs}:{ret_line}:exit".encode()
                    ).hexdigest()[:8]
                    meta_blocks.append({
                        "block_type": "exit",
                        "marker": _exit_marker,
                        "return_expr": ret_expr,
                        "return_type": return_type,
                    })
            else:
                for ret_line, _ret_expr in returns:
                    _exit_marker = _hl.md5(
                        f"{func_file_abs}:{ret_line}:exit".encode()
                    ).hexdigest()[:8]
                    meta_blocks.append({
                        "block_type": "void_exit",
                        "marker": _exit_marker,
                        "return_expr": None,
                        "return_type": None,
                    })
                # Fall-through void exit (same condition as above)
                if not has_return_at_end:
                    _exit_marker = _hl.md5(
                        f"{func_file_abs}:{end_line}:exit".encode()
                    ).hexdigest()[:8]
                    meta_blocks.append({
                        "block_type": "void_exit",
                        "marker": _exit_marker,
                        "return_expr": None,
                        "return_type": None,
                    })

            self.function_metadata[_func_id] = {
                "func_name": func_name,
                "func_id": _func_id,
                "file_path": func_file_abs,
                "params": params,
                "return_type": return_type,
                "start_line": start_line,
                "end_line": end_line,
                "original_start_line": original_start_line,
                "original_end_line": original_end_line,
                "blocks": meta_blocks,
            }

            instrumented_usrs.add(call_usr)

    def process_project(self, var_name="", calls_only=False, call_graph=None,
                        output_to_stderr=False, additional_function_usrs=None,
                        is_c_project=False):
        """
        Process the project: build ASTs, run dependency analysis, and instrument.

        Args:
            var_name: Variable name filter (unused currently).
            calls_only: If True, only instrument function-call statements.
            call_graph: If provided (dict), use callee-side instrumentation:
                        instrument inside called function bodies (entry + exit)
                        instead of at call sites.
            output_to_stderr: If True, generated instrumentation uses
                ``std::cerr`` instead of ``std::cout``.
            additional_function_usrs: Optional iterable of USRs for functions
                that should be instrumented even if they are not reachable from
                the root line's dependency slice.  Used to force instrumentation
                of test-case-scope helpers (signal handlers, setup helpers,
                etc.) that sit in the test body but have no data dependency on
                the failed line.
        """
        # Reset global state for clean analysis
        reset_global_state()

        self.load_compile_commands()
        # Process every file and build ASTs.
        from tqdm import tqdm
        global current_filename
        for entry in tqdm(self.commands,
                          desc="  [1a-i] Building ASTs",
                          unit="file", leave=True):
            directory = entry["directory"]
            file = entry["file"]
            command = entry["command"]
            args = shlex.split(command)[1:]
            if "-o" in args:
                idx = args.index("-o")
                del args[idx:idx+2]
            if file in args:
                args.remove(file)
            self.builder = ASTBuilder(file, args, directory)
            self.builder.build_ast()
            # Set global current_filename for assignment collection.
            current_filename = os.path.abspath(file)
            self.builder.collect_assignments()
            self.file_builders[os.path.abspath(file)] = self.builder

        # Now, find the root instruction.
        root_builder = self.file_builders.get(self.root_file)
        if not root_builder:
            print(f"Root file {self.root_file} was not processed.")
            return
        root_cursor = self.find_top_level_cursor_by_line(root_builder.tu.cursor, self.root_line, self.root_file)
        if not root_cursor:
            print(f"No AST node found at {self.root_file}:{self.root_line}")
            return

        # Extract compile args from the root file for on-demand parsing
        compile_args = root_builder.args if root_builder.args else ["-std=c++17"]
        print("  [1a-i] Extracting dependency tree...")
        dep_stmts = extract_def_tree(self.root_file, self.root_line, self.ddg, flatten=True,
                                      include_call_deps=True, compile_args=compile_args)
        print(f"  [1a-i] Dependency tree: {len(dep_stmts) if dep_stmts else 0} statement(s)")

        if not dep_stmts:
            print(f"WARNING: No dependency slice found at {self.root_file}:{self.root_line}")
            print("  This can happen when the target line is a macro (e.g., EXPECT_EQ).")
            print("  Falling back to instrumenting the enclosing function.")
            dep_stmts = self._fallback_instrument_enclosing_function(root_cursor, root_builder)

        # Supplement with variable-origin call tracing
        if dep_stmts:
            print("  [1a-i] Tracing variable-producing calls...")
            dep_stmts = self._trace_variable_producing_calls(dep_stmts, root_cursor, call_graph)
            print(f"  [1a-i] After call tracing: {len(dep_stmts)} statement(s)")

        # Supplement with ALL call expressions found on the root line.
        # The DDG only tracks data dependencies, so void side-effect calls
        # (e.g. fmt::print) that don't produce a variable are missed.
        # This pass scans the AST directly for any call at the root line.
        print("  [1a-i] Scanning root line for additional calls...")
        dep_stmts = self._supplement_root_line_calls(
            dep_stmts or [], root_cursor, call_graph)

        # Collect the unique set of functions referenced by dep_stmts so that
        # callers (e.g. DiffTraceAnalysis) can feed them to TypeParser.
        # NOTE: implicit constructors (is_implicit=True) are kept here so
        # that TypeParser can discover the parent class type for operator<<
        # generation.  They are only skipped in _register_callee_instrumentation
        # (no TODO markers placed inside the class body).
        seen = set()
        self.instrumented_functions = []
        for stmt in (dep_stmts or []):
            for call in stmt.get('calls', []):
                name = call.get('name')
                if not name:
                    continue
                key = (call.get('file'), name, call.get('line'))
                if key in seen:
                    continue
                seen.add(key)
                self.instrumented_functions.append({
                    'name': name,
                    'file': call.get('file', ''),
                    'line': call.get('line'),
                })

        # Use coordinator to manage multi-file instrumentation
        print("\n" + "="*60)
        print("INSTRUMENTATION PLANNING")
        print("="*60)

        # Register all dependencies and plan instrumentation
        skipped_non_call = 0
        for entry in dep_stmts:
            # When calls_only=True, skip statements that don't involve function calls
            if calls_only and not entry.get('calls'):
                skipped_non_call += 1
                continue

            if call_graph and entry.get('calls'):
                # Callee-side instrumentation: instrument inside called function bodies
                self._register_callee_instrumentation(
                    entry, call_graph,
                    output_to_stderr=output_to_stderr,
                    is_c_project=is_c_project)
            else:
                # Caller-side instrumentation: instrument at the statement location
                file_path = os.path.abspath(entry['file'])
                line = entry['line']
                instr = instrument_assignment(entry)
                if self.coordinator.can_instrument(file_path, line, "after"):
                    self.coordinator.register_instrumentation(file_path, line, instr, "after")

        if calls_only and skipped_non_call > 0:
            print(f"  [calls_only] Skipped {skipped_non_call} non-function-call statements")

        # Register any explicit test-case-scope functions that don't show up
        # in the DDG-derived dep_stmts (signal handlers, setup helpers,
        # functions passed as pointers, etc.).  These are fed from
        # DiffTraceAnalysis which knows the test case line range.
        if call_graph and additional_function_usrs:
            _cg_functions = call_graph.get('functions', {})
            _added_extra = 0
            for _usr in additional_function_usrs:
                _info = _cg_functions.get(_usr)
                if not _info:
                    continue
                if not _info.get('is_definition', False):
                    continue
                _file = _info.get('file', '')
                if not _file or is_system_file(_file):
                    continue
                synthetic = {
                    'calls': [{
                        'name': _info.get('name', ''),
                        'usr': _usr,
                        'file': _file,
                        'line': _info.get('line'),
                        'kind': 'CALL_EXPR',
                    }],
                }
                self._register_callee_instrumentation(
                    synthetic, call_graph,
                    output_to_stderr=output_to_stderr,
                    is_c_project=is_c_project)
                # Also reflect in instrumented_functions so the
                # downstream BFS seeds from these.
                _key = (_file, _info.get('name', ''), _info.get('line'))
                if _key not in seen:
                    seen.add(_key)
                    self.instrumented_functions.append({
                        'name': _info.get('name', ''),
                        'file': _file,
                        'line': _info.get('line'),
                    })
                _added_extra += 1
            if _added_extra:
                print(f"  [extras] Registered {_added_extra} test-case-scope "
                      f"function(s) for instrumentation")

        # Transitively instrument callees via call_graph edges.
        # Walk the call graph from already-instrumented functions and
        # instrument their callees (and their callees, etc.).
        if call_graph:
            cg_functions = call_graph.get('functions', {})
            call_edges = call_graph.get('call_edges', {})
            already = set(self.coordinator.get_instrumented_usrs()
                          if hasattr(self.coordinator, 'get_instrumented_usrs')
                          else [])
            # Collect USRs of already-instrumented functions from
            # the instrumented_functions list.
            for func in self.instrumented_functions:
                fname = func.get('name', '')
                ffile = func.get('file', '')
                fbase = os.path.splitext(
                    os.path.basename(ffile))[0] if ffile else ''
                # Add ALL USRs matching the name+basename — don't stop
                # at the first match.  Clang records call edges on the
                # primary template definition, not on implicit
                # specializations.  For a templated function like
                # ``scan_to<>(FILE *, ..., int &, int &)`` both entries
                # share the same name and file, but only the primary
                # template has real call edges — stopping early can
                # pick the specialization (0 callees) and kill the
                # whole transitive walk.
                for usr, info in cg_functions.items():
                    if info.get('name') != fname and \
                       info.get('qualified_name') != fname:
                        continue
                    cg_base = os.path.splitext(
                        os.path.basename(info.get('file', '')))[0]
                    # Match by stem (ignoring .h/.cc extension difference)
                    if fbase and cg_base == fbase:
                        already.add(usr)
                        continue
                    # Match by name alone if no file info
                    if not fbase:
                        already.add(usr)

            # BFS through call_edges
            queue = list(already)
            visited = set(already)
            while queue:
                caller_usr = queue.pop(0)
                for callee_usr in call_edges.get(caller_usr, []):
                    if callee_usr in visited:
                        continue
                    visited.add(callee_usr)
                    callee_info = cg_functions.get(callee_usr, {})
                    if not callee_info:
                        continue
                    # Skip system files
                    callee_file = callee_info.get('file', '')
                    if is_system_file(callee_file):
                        continue
                    # Build a synthetic dep_entry for this callee
                    synthetic = {
                        'calls': [{
                            'name': callee_info.get('name', ''),
                            'usr': callee_usr,
                            'file': callee_file,
                            'line': callee_info.get('line'),
                            'kind': 'CALL_EXPR',
                        }],
                    }
                    self._register_callee_instrumentation(
                        synthetic, call_graph,
                        output_to_stderr=output_to_stderr,
                        is_c_project=is_c_project)
                    queue.append(callee_usr)

            # Update instrumented_functions with transitively discovered callees
            for usr in visited - already:
                info = cg_functions.get(usr, {})
                if info and not is_system_file(info.get('file', '')):
                    name = info.get('name', '')
                    key = (info.get('file'), name, info.get('line'))
                    if key not in seen:
                        seen.add(key)
                        self.instrumented_functions.append({
                            'name': name,
                            'file': info.get('file', ''),
                            'line': info.get('line'),
                        })

        # Resolve header conflicts
        strategies = self.coordinator.resolve_header_conflicts()

        # Print summary
        print(self.coordinator.get_summary())

        # Group by file for actual instrumentation
        files_to_instrument = self.coordinator.group_by_file()

        # Defensive filter: never instrument files inside build output or
        # backup directories.  Upstream analysis (TypeParser / dependency
        # extraction) can occasionally pick up a header via a relative
        # include path that resolves inside .pre_instrumentation_originals
        # or instrumented_files — those are snapshots, not source.
        _SKIP_DIR_MARKERS = (
            os.sep + 'build_native' + os.sep,
            os.sep + 'build_wasm' + os.sep,
            os.sep + 'build' + os.sep,
            os.sep + '.pre_instrumentation_originals' + os.sep,
            os.sep + '.instrumentation_backups' + os.sep,
            os.sep + 'instrumented_files' + os.sep,
        )

        def _should_skip_path(fp: str) -> bool:
            ap = os.path.abspath(fp) + os.sep
            return any(m in ap for m in _SKIP_DIR_MARKERS)

        _skipped_dirs = [fp for fp in files_to_instrument if _should_skip_path(fp)]
        if _skipped_dirs:
            print(f"  Skipping {len(_skipped_dirs)} file(s) in "
                  f"build/backup directories:")
            for fp in _skipped_dirs[:5]:
                print(f"    - {fp}")
            if len(_skipped_dirs) > 5:
                print(f"    ... and {len(_skipped_dirs) - 5} more")
        files_to_instrument = {
            fp: pts for fp, pts in files_to_instrument.items()
            if not _should_skip_path(fp)
        }

        # Now instrument each file
        modified_files = []
        instrumented_files = set()

        for file_path, instrumentation_points in files_to_instrument.items():
            # Skip if already instrumented (can happen with headers included multiple times)
            if file_path in instrumented_files:
                print(f"Skipping {file_path} - already instrumented")
                continue

            # Convert instrumentation_points to dep_stmts format for process_file
            file_deps = []
            for line, instr, position in instrumentation_points:
                # Reconstruct entry format expected by process_file
                file_deps.append({
                    'file': file_path,
                    'line': line,
                    'defines': [],  # Will be populated if needed
                    'uses': [],
                    'instrumentation': instr,  # Pass pre-generated instrumentation
                    'position': position
                })

            # Check if we have a builder for this file
            if file_path in self.file_builders:
                builder = self.file_builders[file_path]
                source = builder.process_file(file_deps)
                if source:  # Only add files that were actually modified
                    modified_files.append((builder.filename, source))
                    instrumented_files.add(file_path)
            else:
                # This file has dependencies but no builder (likely a header file)
                print(f"Warning: File {file_path} has dependencies but wasn't in compile_commands.json")
                print(f"  Attempting to create builder for header file...")

                args, directory = self._collect_compile_args_from_commands()

                # Try to create a builder for this header file
                try:
                    builder = ASTBuilder(file_path, args, directory)
                    source = builder.process_file(file_deps)
                    if source:
                        modified_files.append((file_path, source))
                        instrumented_files.add(file_path)
                        print(f"  Successfully instrumented header file: {file_path}")
                except Exception as e:
                    print(f"  Error creating builder for {file_path}: {e}")
                    # Fallback: direct text insertion (no AST needed)
                    print(f"  Falling back to direct text insertion...")
                    try:
                        source = self._direct_text_instrument(file_path, file_deps)
                        if source:
                            modified_files.append((file_path, source))
                            instrumented_files.add(file_path)
                            print(f"  Successfully instrumented via text insertion: {file_path}")
                    except Exception as e2:
                        print(f"  Direct text insertion also failed: {e2}")
                        print(f"  Skipping instrumentation for this file.")

        # Export instrumentation report
        report_path = os.path.join(os.path.dirname(self.compile_commands_path), "instrumentation_report.json")
        self.coordinator.export_report(report_path)

        return modified_files

    def _direct_text_instrument(self, file_path, file_deps):
        """
        Fallback: insert instrumentation blocks directly into a source file
        without AST parsing. Used for header files that fail to parse.

        Args:
            file_path: Absolute path to the source file.
            file_deps: List of dicts with 'line', 'instrumentation', 'position' keys.

        Returns:
            Modified source string, or empty string if nothing was inserted.
        """
        with open(file_path, 'r') as f:
            lines = f.readlines()

        # Check if already instrumented
        content = ''.join(lines)
        if "/*[Instrumented]" in content:
            print(f"  File {file_path} is already instrumented - skipping")
            return ""

        # Sort modifications: reverse by line, "after" before "before" for same line
        mods = []
        for dep in file_deps:
            line = dep['line']
            instr = dep.get('instrumentation', '')
            position = dep.get('position', 'after')
            if instr:
                mods.append((line, instr, position))

        if not mods:
            return ""

        # NOTE: Compact one-line function bodies are pre-expanded in
        # Phase 1a-ii.5 (expand_oneliners.expand_files), not here.

        mods.sort(key=lambda x: (x[0], 0 if x[2] == "after" else -1), reverse=True)

        for line_no, instr, position in mods:
            if position == "before":
                insert_at = line_no - 1  # 0-indexed, before this line
            else:
                insert_at = line_no  # 0-indexed, after this line (line_no is 1-indexed)
            if 0 <= insert_at <= len(lines):
                lines.insert(insert_at, instr + "\n")

        return "".join(lines)

    def find_cursor_by_line(self, cursor, target_line, target_file): # this should be deprecated since it doesn't get the top level cursor of a specific line of code.
        """
        Traverse the AST recursively to find a cursor in target_file
        whose start location matches target_line. (This is a heuristic.)
        """
        try:
            loc = cursor.location
            if loc.file and os.path.abspath(loc.file.name) == os.path.abspath(target_file):
                if loc.line == target_line and cursor.kind == cindex.CursorKind.PAREN_EXPR: # Assume it is calling the assert. We shall consider more general case here.
                    return cursor
        except ValueError:
            pass
        for child in cursor.get_children():
            result = self.find_cursor_by_line(child, target_line, target_file)
            if result:
                return result
        return None

    def find_top_level_cursor_by_line(self, root_cursor, target_line, target_file):
        """
        Find the top-level cursor that contains code at target_line in target_file.
        """
        target_file_abs = os.path.abspath(target_file)
        
        # First, collect all cursors that contain the target line
        matching_cursors = []
        
        def collect_containing_cursors(cursor):
            try:
                if cursor.location.file and os.path.abspath(cursor.location.file.name) == target_file_abs:
                    # Check if this cursor's extent contains the target line
                    if hasattr(cursor, 'extent') and cursor.extent.start.line <= target_line <= cursor.extent.end.line:
                        matching_cursors.append(cursor)
                
                # Recursively collect from children
                for child in cursor.get_children():
                    collect_containing_cursors(child)
                    
            except (ValueError, AttributeError):
                # Skip cursors that cause errors
                pass
        
        # Collect all cursors containing the target line
        collect_containing_cursors(root_cursor)
        
        # No matches found
        if not matching_cursors:
            return None
        
        # Find statements among the matching cursors
        stmt_cursors = [c for c in matching_cursors if self.is_statement_or_declaration(c)]
        
        # If we found statements, use those, otherwise use all matching cursors
        candidates = stmt_cursors if stmt_cursors else matching_cursors
        
        # Sort by the size of their extent (smaller difference between start and end means more specific)
        # We want the cursor with the smallest extent that still contains our line
        candidates.sort(key=lambda c: c.extent.end.line - c.extent.start.line)
        
        # Return the most specific cursor
        return candidates[0] if candidates else None

    def is_statement_or_declaration(self, cursor):
        """Helper method to check if a cursor is a statement or declaration"""
        stmt_kinds = [
            cindex.CursorKind.COMPOUND_STMT,
            cindex.CursorKind.IF_STMT,
            cindex.CursorKind.WHILE_STMT,
            cindex.CursorKind.DO_STMT,
            cindex.CursorKind.FOR_STMT,
            cindex.CursorKind.SWITCH_STMT,
            cindex.CursorKind.CASE_STMT,
            cindex.CursorKind.DEFAULT_STMT,
            cindex.CursorKind.BREAK_STMT,
            cindex.CursorKind.CONTINUE_STMT,
            cindex.CursorKind.RETURN_STMT,
            cindex.CursorKind.DECL_STMT,
            cindex.CursorKind.CALL_EXPR,
            cindex.CursorKind.BINARY_OPERATOR,
            cindex.CursorKind.PAREN_EXPR,
            # Add other statement kinds as needed
        ]
        
        decl_kinds = [
            cindex.CursorKind.FUNCTION_DECL,
            cindex.CursorKind.VAR_DECL,
            cindex.CursorKind.PARM_DECL,
            # Add other declaration kinds as needed
        ]
        
        return cursor.kind in stmt_kinds or cursor.kind in decl_kinds


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python instrument_project.py <path_to_compile_commands.json> <root_file> <root_line> [<root_func_name>]")
        sys.exit(1)
    compile_commands_path = sys.argv[1]
    root_file = sys.argv[2]
    root_line = sys.argv[3]
    root_func_name = sys.argv[4] if len(sys.argv) > 4 else ""
    project_builder = ProjectASTBuilder(compile_commands_path, root_file, root_line, root_func_name)
    project_builder.process_project()
