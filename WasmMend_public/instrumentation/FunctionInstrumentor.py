#!/usr/bin/env python3
"""
FunctionInstrumentor: Generates instrumentation code for C/C++ functions
so that their entry parameters and exit return values can be printed at
runtime via ``std::cout``.

Mimics the design of :class:`OstreamInstrumentor.generate_instrumentation_plan`:
given a call graph and the list of functions discovered by Step 1a, it
produces a plan dict that contains ready-to-insert ``std::cout`` code for
each TODO block — no LLM needed for the initial generation.

Usage (programmatic)::

    from instrumentation.FunctionInstrumentor import FunctionInstrumentor

    instrumentor = FunctionInstrumentor(call_graph_path, compile_commands_path)
    plan = instrumentor.generate_instrumentation_plan(func_list)
"""

import hashlib
import json
import os
import shlex

from clang import cindex


# ============================================================================
# PrintabilityClassifier (for function parameters)
# ============================================================================

PRINTABLE_PRIMITIVES = frozenset({
    "int", "unsigned int", "signed int",
    "short", "unsigned short", "short int", "unsigned short int",
    "long", "unsigned long", "long int", "unsigned long int",
    "long long", "unsigned long long",
    "long long int", "unsigned long long int",
    "float", "double", "long double",
    "char", "signed char", "unsigned char",
    "wchar_t", "char16_t", "char32_t", "char8_t",
    "bool",
    "size_t", "std::size_t", "ssize_t",
    "ptrdiff_t", "std::ptrdiff_t",
    "int8_t", "int16_t", "int32_t", "int64_t",
    "uint8_t", "uint16_t", "uint32_t", "uint64_t",
    "intptr_t", "uintptr_t",
})

PRINTABLE_STD_TYPES = frozenset({
    "std::string",
    "std::basic_string<char>",
    "std::basic_string<char, std::char_traits<char>, std::allocator<char>>",
    "std::string_view",
    "std::basic_string_view<char>",
    "std::basic_string_view<char, std::char_traits<char>>",
})

SIZED_CONTAINER_PREFIXES = [
    "std::vector", "std::map", "std::unordered_map",
    "std::set", "std::unordered_set", "std::multiset",
    "std::multimap", "std::unordered_multimap", "std::unordered_multiset",
    "std::list", "std::deque",
    "std::queue", "std::stack", "std::priority_queue",
    "std::array", "std::bitset",
]

NON_PRINTABLE_PREFIXES = [
    "std::forward_list",
    "std::pair", "std::tuple",
    "std::unique_ptr", "std::shared_ptr", "std::weak_ptr",
    "std::optional", "std::variant", "std::any",
    "std::function", "std::mutex", "std::recursive_mutex",
    "std::thread", "std::atomic",
]


def _strip_cv(type_str):
    """Strip leading const / volatile qualifiers."""
    result = type_str.strip()
    for prefix in ("const ", "volatile ", "const volatile ",
                    "volatile const "):
        if result.startswith(prefix):
            result = result[len(prefix):]
    return result.strip()


def _is_pointer_type(type_str):
    """Return True if the type is a pointer (ends with '*')."""
    s = _strip_cv(type_str).rstrip()
    return s.endswith('*')


def _is_reference_type(type_str):
    """Return True if the type is a reference (ends with '&')."""
    s = _strip_cv(type_str).rstrip()
    return s.endswith('&') or s.endswith('&&')


def _strip_reference(type_str):
    """Remove trailing & or && from a type string."""
    s = type_str.rstrip()
    if s.endswith('&&'):
        return s[:-2].rstrip()
    if s.endswith('&'):
        return s[:-1].rstrip()
    return s


def _is_char_pointer(type_str):
    """Check if the type is a char pointer (prints as string)."""
    stripped = _strip_cv(type_str).rstrip()
    return stripped in ("char *", "char*",
                        "const char *", "const char*",
                        "wchar_t *", "wchar_t*",
                        "const wchar_t *", "const wchar_t*")


def classify_param_type(type_str):
    """
    Classify a parameter type string for printability.

    Returns one of:
        ``"printable"``          – can be printed with ``std::cout << var``
        ``"printable_pointer"``  – pointer to printable type (needs null guard)
        ``"char_pointer"``       – char*/wchar_t* (print as string, no deref)
        ``"sized_container"``    – has ``.size()``; print size
        ``"non_printable"``      – cannot be printed
    """
    stripped = _strip_cv(type_str)

    # References: classify the underlying type.
    if _is_reference_type(stripped):
        stripped = _strip_cv(_strip_reference(stripped))

    # Char pointers print as strings.
    if _is_char_pointer(type_str):
        return "char_pointer"

    # Other pointers.
    if _is_pointer_type(stripped):
        return "printable_pointer"  # assume user-defined types have operator<<

    # Arrays.
    if '[' in stripped:
        return "non_printable"

    # Primitives.
    if stripped in PRINTABLE_PRIMITIVES:
        return "printable"

    # Known std types.
    if stripped in PRINTABLE_STD_TYPES:
        return "printable"

    # Containers with .size() — print the size instead of the full content.
    for prefix in SIZED_CONTAINER_PREFIXES:
        if stripped.startswith(prefix):
            return "sized_container"

    # Known non-printable types (no operator<< or .size()).
    for prefix in NON_PRINTABLE_PREFIXES:
        if stripped.startswith(prefix):
            return "non_printable"

    # Enums are printable (cast to int by default).
    # User-defined types: assume operator<< was generated in Step 1a-iii.
    return "printable"


# ============================================================================
# Code generation helpers
# ============================================================================

def _gen_print_var_c(var_name, var_type, indent="    ", label=None,
                     output_to_stderr=False):
    """C-mode variant of _gen_print_var. Emits fprintf statements.

    Classification is C-appropriate: no std::string, no sized_container, no
    operator<<. Primitives map to ``%lld``/(long long) for signed ints,
    ``%llu``/(unsigned long long) for unsigned ints, ``%g``/(double) for
    floats, ``%.200s`` for char*, ``%p`` for other pointers. User-defined
    struct members get a placeholder line (non-printable) since emitting
    a generic struct printer from here is not possible.
    """
    stream = "stderr" if output_to_stderr else "stdout"
    if label is None:
        label = var_name

    stripped = _strip_cv(var_type)
    if _is_reference_type(stripped):
        stripped = _strip_cv(_strip_reference(stripped))

    # char*/const char* with NULL guard + 200-char cap via %.200s.
    if _is_char_pointer(var_type):
        return [
            f'{indent}fprintf({stream}, "{label} = %.200s\\n", '
            f'({var_name}) ? ({var_name}) : "(null)");',
        ]

    # Signed integer primitives -> %lld with (long long) cast.
    if stripped in (
        "char", "signed char",
        "short", "signed short", "short int", "signed short int",
        "int", "signed", "signed int",
        "long", "signed long", "long int", "signed long int",
        "long long", "signed long long",
        "long long int", "signed long long int",
        "int8_t", "int16_t", "int32_t", "int64_t",
        "intptr_t", "ptrdiff_t", "intmax_t",
    ):
        return [
            f'{indent}fprintf({stream}, "{label} = %lld\\n", '
            f'(long long)({var_name}));',
        ]

    # Unsigned integer primitives -> %llu with (unsigned long long) cast.
    if stripped in (
        "unsigned char",
        "unsigned short", "unsigned short int",
        "unsigned", "unsigned int",
        "unsigned long", "unsigned long int",
        "unsigned long long", "unsigned long long int",
        "uint8_t", "uint16_t", "uint32_t", "uint64_t",
        "uintptr_t", "size_t", "uintmax_t",
    ):
        return [
            f'{indent}fprintf({stream}, "{label} = %llu\\n", '
            f'(unsigned long long)({var_name}));',
        ]

    # Floats -> %g with (double) cast.
    if stripped in ("float", "double"):
        return [
            f'{indent}fprintf({stream}, "{label} = %g\\n", '
            f'(double)({var_name}));',
        ]
    if stripped == "long double":
        return [
            f'{indent}fprintf({stream}, "{label} = %Lg\\n", ({var_name}));',
        ]

    # C99/C11 bool.
    if stripped in ("_Bool", "bool"):
        return [
            f'{indent}fprintf({stream}, "{label} = %d\\n", (int)({var_name}));',
        ]

    # Other pointer types -> address only via %p.
    if _is_pointer_type(stripped):
        return [
            f'{indent}fprintf({stream}, "{label} = %p\\n", '
            f'(const void*)({var_name}));',
        ]

    # Arrays and unknown/user-defined types: print a placeholder so the
    # code compiles. Downstream repair (LLM) can substitute a call to
    # the pre-generated print_<kind>_<name> helper if the type has one.
    return [
        f'{indent}fprintf({stream}, "{label} = (non-printable:{var_type})\\n");',
    ]


def _gen_print_var(var_name, var_type, indent="    ", label=None,
                   output_to_stderr=False):
    """
    Generate std::cout/std::cerr statement(s) for printing a single variable.

    Args:
        var_name: The C++ expression to print (used in the generated code).
        var_type: The C++ type string (used for classification).
        indent:   Indentation prefix for each generated line.
        label:    The label shown in the output (defaults to *var_name*).
        output_to_stderr: If True, use ``std::cerr`` instead of ``std::cout``.

    Returns a list of code lines (without trailing newline).
    """
    os = "std::cerr" if output_to_stderr else "std::cout"
    if label is None:
        label = var_name
    classification = classify_param_type(var_type)

    if classification == "char_pointer":
        # char* prints as string, but guard against null AND cap the
        # length.  Unterminated/garbage-aimed C strings can dump the
        # rest of memory, swallowing instrumentation markers.
        return [
            f'{indent}if ({var_name}) {{',
            f'{indent}    std::string __s_{label}({var_name});',
            f'{indent}    if (__s_{label}.size() > 200) __s_{label} = __s_{label}.substr(0, 200) + "...";',
            f'{indent}    {os} << "{label} = " << __s_{label} << std::endl;',
            f'{indent}}} else {{ {os} << "{label} = (nullptr)" << std::endl; }}',
        ]

    if classification == "printable_pointer":
        # Print address; dereference only when non-null AND the target
        # is also non-null (handles ptr-to-ptr-to-string cases like
        # const char**).  Cap string output at 200 chars.
        return [
            f'{indent}{os} << "{label} = " << (void*){var_name};',
            f'{indent}if ({var_name}) {{',
            f'{indent}    // Safe dereference: non-null pointer; if pointee is',
            f'{indent}    // itself a C-string, cap output to avoid dumps.',
            f'{indent}    using __T_{label} = typename std::remove_reference<decltype(*{var_name})>::type;',
            f'{indent}    if constexpr (std::is_convertible<__T_{label}, const char*>::value) {{',
            f'{indent}        if (*{var_name}) {{',
            f'{indent}            std::string __s_{label}(*{var_name});',
            f'{indent}            if (__s_{label}.size() > 200) __s_{label} = __s_{label}.substr(0, 200) + "...";',
            f'{indent}            {os} << " (*{label} = " << __s_{label} << ")";',
            f'{indent}        }}',
            f'{indent}    }} else {{',
            f'{indent}        {os} << " (*{label} = " << *{var_name} << ")";',
            f'{indent}    }}',
            f'{indent}}}',
            f'{indent}{os} << std::endl;',
        ]

    if classification == "sized_container":
        return [
            f'{indent}{os} << "{label}.size() = " << {var_name}.size() << std::endl;',
        ]

    if classification == "non_printable":
        return [
            f'{indent}{os} << "{label} = [unprintable]" << std::endl;',
        ]

    # printable
    return [
        f'{indent}{os} << "{label} = " << {var_name} << std::endl;',
    ]


def _make_marker(file_path, line, suffix=""):
    """Create an 8-char MD5 marker, matching AST_builder.py's convention."""
    key = f"{file_path}:{line}"
    if suffix:
        key += f":{suffix}"
    return hashlib.md5(key.encode()).hexdigest()[:8]


def _make_func_id(file_path, start_line, func_name):
    """Create a func_id hash, matching AST_builder.py's convention."""
    return hashlib.md5(
        f"{file_path}:{start_line}:{func_name}".encode()
    ).hexdigest()[:8]


# ============================================================================
# Per-function code generators
# ============================================================================

def generate_entry_code(func_name, params, file_path=None, start_line=None,
                        indent="    ", marker=None, func_id=None,
                        owner_type=None, output_to_stderr=False,
                        is_c_project=False):
    """
    Generate the instrumentation code to replace a Function ENTRY TODO block.

    Returns the code as a string (ready to be spliced into the function body).

    When *marker* and/or *func_id* are provided they override the values that
    would otherwise be computed from *file_path* / *start_line*.  This is used
    when the caller already knows the marker (e.g. parsed from an existing TODO
    block whose line numbers may have shifted).

    Args:
        owner_type: If this function is a member of a class/struct, the
            owning type name (e.g. ``"mmap_file_open"``).  When set, a
            ``@@TYPE_NAME_...@@`` line is emitted so that Phase 3 can
            associate the event with its defining type.
        output_to_stderr: If True, use ``std::cerr`` instead of ``std::cout``.
    """
    if marker is None:
        marker = _make_marker(file_path, start_line, "entry")
    if func_id is None:
        func_id = _make_func_id(file_path, start_line, func_name)

    if is_c_project:
        stream = "stderr" if output_to_stderr else "stdout"
        lines = [
            f'{indent}fprintf({stream}, "@@INST_START_{marker}@@\\n");',
            f'{indent}fprintf({stream}, "@@FUNC_ID_{func_name}_{func_id}@@\\n");',
        ]
        if owner_type:
            lines.append(
                f'{indent}fprintf({stream}, "@@TYPE_NAME_{owner_type}@@\\n");')
        lines.append(
            f'{indent}fprintf({stream}, "+++Below are Input+++\\n");')
        if params:
            for p in params:
                lines.extend(_gen_print_var_c(
                    p['name'], p['type'], indent,
                    output_to_stderr=output_to_stderr))
        else:
            lines.append(f'{indent}/* (no parameters) */')
        lines.append(
            f'{indent}fprintf({stream}, "@@INST_END_{marker}@@\\n");')
        return "\n".join(lines)

    os = "std::cerr" if output_to_stderr else "std::cout"
    lines = [
        f'{indent}{os} << "@@INST_START_{marker}@@" << std::endl;',
        f'{indent}{os} << "@@FUNC_ID_{func_name}_{func_id}@@" << std::endl;',
    ]
    if owner_type:
        lines.append(
            f'{indent}{os} << "@@TYPE_NAME_{owner_type}@@" << std::endl;')
    lines.append(
        f'{indent}{os} << "+++Below are Input+++" << std::endl;')

    if params:
        for p in params:
            lines.extend(_gen_print_var(p['name'], p['type'], indent,
                                        output_to_stderr=output_to_stderr))
    else:
        lines.append(f'{indent}// (no parameters)')

    lines.append(f'{indent}{os} << "@@INST_END_{marker}@@" << std::endl;')

    return "\n".join(lines)


def generate_exit_code(func_name, params, file_path=None, ret_line=None,
                       start_line=None, return_expr="", return_type="void",
                       indent="    ", marker=None, func_id=None,
                       owner_type=None, output_to_stderr=False,
                       is_c_project=False):
    """
    Generate the instrumentation code for a Function EXIT (non-void return).

    The generated block **replaces** the original ``return expr;`` statement.
    It captures the return expression into a local variable with ``auto``,
    prints the value and parameters, then returns the captured variable.

    When *marker* and/or *func_id* are provided they override the values
    computed from *file_path* / *ret_line* / *start_line*.

    Args:
        owner_type: Owning class/struct name, if any (see
            :func:`generate_entry_code`).
        output_to_stderr: If True, use ``std::cerr`` instead of ``std::cout``.
    """
    if marker is None:
        marker = _make_marker(file_path, ret_line, "exit")
    if func_id is None:
        func_id = _make_func_id(file_path, start_line, func_name)
    retval_var = f"__retval_{marker}"
    inner = indent + "  "

    if is_c_project:
        stream = "stderr" if output_to_stderr else "stdout"
        lines = [
            f'{indent}{{',
            f'{inner}__typeof__({return_expr}) {retval_var} = ({return_expr});',
            f'{inner}fprintf({stream}, "@@INST_START_{marker}@@\\n");',
            f'{inner}fprintf({stream}, "@@FUNC_ID_{func_name}_{func_id}@@\\n");',
        ]
        if owner_type:
            lines.append(
                f'{inner}fprintf({stream}, "@@TYPE_NAME_{owner_type}@@\\n");')
        lines.append(
            f'{inner}fprintf({stream}, "---Below are Outputs---\\n");')
        lines.extend(_gen_print_var_c(
            retval_var, return_type, inner, label="ret",
            output_to_stderr=output_to_stderr))
        if params:
            for p in params:
                lines.extend(_gen_print_var_c(
                    p['name'], p['type'], inner,
                    output_to_stderr=output_to_stderr))
        lines.append(
            f'{inner}fprintf({stream}, "@@INST_END_{marker}@@\\n");')
        lines.append(f'{inner}return {retval_var};')
        lines.append(f'{indent}}}')
        return "\n".join(lines)

    os = "std::cerr" if output_to_stderr else "std::cout"
    lines = [
        f'{indent}{{ const auto& {retval_var} = ({return_expr});',
        f'{inner}{os} << "@@INST_START_{marker}@@" << std::endl;',
        f'{inner}{os} << "@@FUNC_ID_{func_name}_{func_id}@@" << std::endl;',
    ]
    if owner_type:
        lines.append(
            f'{inner}{os} << "@@TYPE_NAME_{owner_type}@@" << std::endl;')
    lines.append(
        f'{inner}{os} << "---Below are Outputs---" << std::endl;')

    # Print the captured return value
    lines.extend(_gen_print_var(retval_var, return_type, inner, label="ret",
                                output_to_stderr=output_to_stderr))

    # Print (possibly modified) input parameters
    if params:
        for p in params:
            lines.extend(_gen_print_var(p['name'], p['type'], inner,
                                        output_to_stderr=output_to_stderr))

    lines.append(f'{inner}{os} << "@@INST_END_{marker}@@" << std::endl;')
    lines.append(f'{inner}return {retval_var}; }}')

    return "\n".join(lines)


def generate_void_exit_code(func_name, params, file_path=None, exit_line=None,
                            start_line=None, indent="    ",
                            marker=None, func_id=None, owner_type=None,
                            output_to_stderr=False, member_vars=None,
                            is_c_project=False):
    """
    Generate the instrumentation code to replace a Function EXIT (void) TODO
    block.

    When *marker* and/or *func_id* are provided they override the values
    computed from *file_path* / *exit_line* / *start_line*.

    Args:
        owner_type: Owning class/struct name, if any (see
            :func:`generate_entry_code`).
        output_to_stderr: If True, use ``std::cerr`` instead of ``std::cout``.
        member_vars: For constructors — list of dicts with ``name`` and
            ``type_spelling`` keys.  When provided, ``this->member``
            values are printed at exit to capture the constructed
            object's state.
    """
    if marker is None:
        marker = _make_marker(file_path, exit_line, "exit")
    if func_id is None:
        func_id = _make_func_id(file_path, start_line, func_name)

    if is_c_project:
        stream = "stderr" if output_to_stderr else "stdout"
        lines = [
            f'{indent}fprintf({stream}, "@@INST_START_{marker}@@\\n");',
            f'{indent}fprintf({stream}, "@@FUNC_ID_{func_name}_{func_id}@@\\n");',
        ]
        if owner_type:
            lines.append(
                f'{indent}fprintf({stream}, "@@TYPE_NAME_{owner_type}@@\\n");')
        lines.extend([
            f'{indent}fprintf({stream}, "---Below are Outputs---\\n");',
            f'{indent}fprintf({stream}, "ret = VOID\\n");',
        ])
        if params:
            for p in params:
                lines.extend(_gen_print_var_c(
                    p['name'], p['type'], indent,
                    output_to_stderr=output_to_stderr))
        # C has no constructors / member_vars; ignored in C mode.
        lines.append(
            f'{indent}fprintf({stream}, "@@INST_END_{marker}@@\\n");')
        return "\n".join(lines)

    os = "std::cerr" if output_to_stderr else "std::cout"
    lines = [
        f'{indent}{os} << "@@INST_START_{marker}@@" << std::endl;',
        f'{indent}{os} << "@@FUNC_ID_{func_name}_{func_id}@@" << std::endl;',
    ]
    if owner_type:
        lines.append(
            f'{indent}{os} << "@@TYPE_NAME_{owner_type}@@" << std::endl;')
    lines.extend([
        f'{indent}{os} << "---Below are Outputs---" << std::endl;',
        f'{indent}{os} << "ret = VOID" << std::endl;',
    ])

    if params:
        for p in params:
            lines.extend(_gen_print_var(p['name'], p['type'], indent,
                                        output_to_stderr=output_to_stderr))

    # For constructors: print member variables to capture the
    # constructed object's state.
    if member_vars:
        for m in member_vars:
            m_name = m['name']
            m_type = m.get('type_spelling', '')
            lines.extend(_gen_print_var(
                f'this->{m_name}', m_type, indent,
                label=m_name,
                output_to_stderr=output_to_stderr))

    lines.append(f'{indent}{os} << "@@INST_END_{marker}@@" << std::endl;')

    return "\n".join(lines)


# ============================================================================
# FunctionInstrumentor (main class)
# ============================================================================

class FunctionInstrumentor:
    """
    Generates an instrumentation plan for function entry/exit printing,
    analogous to ``OstreamInstrumentor.generate_instrumentation_plan()``.

    Uses libclang AST to locate function bodies and return statements
    (no regex).
    """

    def __init__(self, call_graph_path, compile_commands_path=None):
        """
        Args:
            call_graph_path: Path to the call_graph.json produced by
                :class:`CallGraphBuilder`.
            compile_commands_path: Path to compile_commands.json.  Needed so
                libclang can resolve ``#include`` directives when parsing
                source files.  If ``None``, files are parsed with minimal
                flags (``-std=c++11``).
        """
        with open(call_graph_path, 'r') as f:
            self.call_graph = json.load(f)
        self.cg_functions = self.call_graph.get('functions', {})

        # Per-file compile args and a default fallback
        self._compile_args = {}   # abs_file_path -> [args]
        self._default_args = ['-std=c++11']
        if compile_commands_path:
            self._load_compile_commands(compile_commands_path)

        # Cache parsed translation units per file
        self._tu_cache = {}       # abs_file_path -> TranslationUnit

    # -----------------------------------------------------------------
    # Compile-commands loading
    # -----------------------------------------------------------------

    def _load_compile_commands(self, path):
        """Parse compile_commands.json and store per-file arg lists."""
        with open(path) as f:
            commands = json.load(f)

        all_include_flags = set()

        for entry in commands:
            file_path = entry['file']
            directory = entry.get('directory', '.')
            if not os.path.isabs(file_path):
                file_path = os.path.join(directory, file_path)
            file_path = os.path.abspath(file_path)

            raw_args = shlex.split(entry['command'])[1:]  # drop compiler name

            # Remove -o <output> and the source filename
            if '-o' in raw_args:
                idx = raw_args.index('-o')
                del raw_args[idx:idx + 2]
            if file_path in raw_args:
                raw_args.remove(file_path)
            basename = os.path.basename(entry['file'])
            if basename in raw_args:
                raw_args.remove(basename)

            self._compile_args[file_path] = raw_args

            # Collect include flags for the default fallback
            for arg in raw_args:
                if arg.startswith('-I') or arg.startswith('-D') or arg.startswith('-std='):
                    all_include_flags.add(arg)

        if all_include_flags:
            self._default_args = sorted(all_include_flags)

    # -----------------------------------------------------------------
    # libclang AST helpers
    # -----------------------------------------------------------------

    def _get_tu(self, file_path):
        """Parse *file_path* with libclang (cached per file)."""
        abs_path = os.path.abspath(file_path)
        if abs_path in self._tu_cache:
            return self._tu_cache[abs_path]

        args = list(self._compile_args.get(abs_path, self._default_args))

        # Header files must be parsed as C++ (libclang defaults .h to C)
        if abs_path.endswith(('.h', '.hpp', '.hxx')):
            args.append('-xc++')

        index = cindex.Index.create()

        try:
            tu = index.parse(
                abs_path, args=args,
                options=cindex.TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD)
            self._tu_cache[abs_path] = tu
            return tu
        except Exception as e:
            print(f"  Warning: failed to parse {abs_path}: {e}")
            return None

    def _find_function_cursor(self, tu, file_path, start_line):
        """
        Walk the AST to find the function definition cursor at
        *file_path*:*start_line*.
        """
        abs_path = os.path.abspath(file_path)

        def _visit(cursor):
            # Only look at nodes in the target file
            loc_file = cursor.location.file
            if loc_file:
                if os.path.abspath(loc_file.name) != abs_path:
                    return None

            if cursor.kind in (cindex.CursorKind.FUNCTION_DECL,
                               cindex.CursorKind.CXX_METHOD,
                               cindex.CursorKind.CONSTRUCTOR,
                               cindex.CursorKind.DESTRUCTOR):
                if cursor.is_definition() and cursor.location.line == start_line:
                    return cursor

            for child in cursor.get_children():
                result = _visit(child)
                if result is not None:
                    return result
            return None

        return _visit(tu.cursor)

    def _find_body_cursor(self, func_cursor):
        """Return the COMPOUND_STMT child (function body), or ``None``."""
        for child in func_cursor.get_children():
            if child.kind == cindex.CursorKind.COMPOUND_STMT:
                return child
        return None

    def _collect_return_stmts(self, body_cursor, source_lines):
        """
        Traverse *body_cursor* to find every ``return`` statement.

        Skips returns inside nested lambdas / local function declarations.

        Returns:
            list of ``(start_line, end_line, return_expression_string)``
            tuples (both 1-indexed).
        """
        results = []

        def _visit(cursor):
            # Don't recurse into nested function-like constructs
            if cursor.kind in (cindex.CursorKind.LAMBDA_EXPR,
                               cindex.CursorKind.FUNCTION_DECL,
                               cindex.CursorKind.CXX_METHOD):
                return

            if cursor.kind == cindex.CursorKind.RETURN_STMT:
                ret_line = cursor.location.line      # 1-indexed
                ret_end  = cursor.extent.end.line    # 1-indexed
                children = list(cursor.get_children())
                if children:
                    expr_text = self._extent_text(children[0].extent,
                                                  source_lines)
                    # libclang sometimes gives the child an extent that
                    # includes the "return" keyword itself — strip it.
                    if expr_text.startswith('return '):
                        expr_text = expr_text[len('return '):].strip()
                    elif expr_text.startswith('return'):
                        expr_text = expr_text[len('return'):].strip()
                else:
                    expr_text = "void"
                results.append((ret_line, ret_end, expr_text))
                return  # don't recurse into the expression

            for child in cursor.get_children():
                _visit(child)

        _visit(body_cursor)
        return results

    def _has_fall_through(self, body_cursor):
        """
        Check whether the function body falls through (no ``return`` at
        the very end).  Uses AST: the last top-level statement in the
        COMPOUND_STMT must be a RETURN_STMT for there to be no
        fall-through.
        """
        children = list(body_cursor.get_children())
        if not children:
            return True
        last = children[-1]
        if last.kind == cindex.CursorKind.RETURN_STMT:
            return False
        return True

    @staticmethod
    def _extent_text(extent, source_lines):
        """Extract the source text covered by a clang *extent*."""
        sl = extent.start.line - 1   # 0-indexed
        sc = extent.start.column - 1
        el = extent.end.line - 1
        ec = extent.end.column - 1

        if sl == el:
            text = source_lines[sl][sc:ec]
        else:
            parts = [source_lines[sl][sc:]]
            for i in range(sl + 1, el):
                parts.append(source_lines[i])
            parts.append(source_lines[el][:ec])
            text = ' '.join(p.strip() for p in parts)

        text = text.strip()
        if text.endswith(';'):
            text = text[:-1].strip()
        return text

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def generate_instrumentation_plan(self, func_list):
        """
        Generate an instrumentation plan for the given functions.

        Returns:
            A plan dict (see class docstring for schema).
        """
        matched = 0
        seen_usrs = set()
        functions_list = []
        files_needing_include = set()

        for func_spec in func_list:
            func_info = self._resolve_function(func_spec)
            if func_info is None:
                continue

            usr = func_info.get('usr', '')
            if usr and usr in seen_usrs:
                continue
            if usr:
                seen_usrs.add(usr)

            if not func_info.get('is_definition'):
                resolved = self._resolve_to_definition(func_info)
                if resolved:
                    func_info = resolved
                else:
                    continue

            func_file = func_info['file']
            if not func_file or not os.path.exists(func_file):
                continue

            func_file_abs = os.path.abspath(func_file)  # for file I/O
            func_file_raw = func_file                    # for markers (matches call graph)
            func_file_lower = func_file_abs.lower()
            if ('gtest' in func_file_lower or 'gmock' in func_file_lower or
                    '/gtest/' in func_file_lower or '/gmock/' in func_file_lower):
                continue

            func_name = func_info.get('qualified_name', func_info['name'])
            params = func_info.get('params', [])
            return_type = func_info.get('return_type', 'void') or 'void'
            start_line = func_info['line']
            end_line = func_info.get('end_line', start_line)

            # --- Read source file ---
            try:
                with open(func_file_abs, 'r') as f:
                    source_lines = f.readlines()
            except Exception:
                continue

            # --- Parse with libclang and locate function body ---
            tu = self._get_tu(func_file_abs)
            if tu is None:
                continue

            func_cursor = self._find_function_cursor(tu, func_file_abs,
                                                     start_line)
            if func_cursor is None:
                continue

            # Skip constructors, destructors, and operator<< overloads
            if func_cursor.kind in (cindex.CursorKind.CONSTRUCTOR,
                                    cindex.CursorKind.DESTRUCTOR):
                continue
            if func_info.get('name', '').startswith('operator<<'):
                continue

            body_cursor = self._find_body_cursor(func_cursor)
            if body_cursor is None:
                continue

            brace_line = body_cursor.location.line  # 1-indexed

            # --- Generate entry code ---
            entry_code = generate_entry_code(
                func_name, params, func_file_raw, start_line)

            # --- Find return statements via AST ---
            returns = self._collect_return_stmts(body_cursor, source_lines)
            is_void = (return_type == 'void')

            exit_points = []
            for ret_line, ret_end, ret_expr in returns:
                if is_void:
                    code = generate_void_exit_code(
                        func_name, params, func_file_raw, ret_line,
                        start_line)
                else:
                    code = generate_exit_code(
                        func_name, params, func_file_raw, ret_line,
                        start_line, return_expr=ret_expr,
                        return_type=return_type)
                exit_points.append({
                    "line": ret_line,
                    "end_line": ret_end,
                    "return_expr": ret_expr,
                    "code": code,
                    "is_void": is_void,
                })

            # Fall-through detection via AST
            has_fall_through = False
            if is_void and self._has_fall_through(body_cursor):
                has_fall_through = True
                code = generate_void_exit_code(
                    func_name, params, func_file_raw, end_line,
                    start_line)
                exit_points.append({
                    "line": end_line,
                    "return_expr": "",
                    "code": code,
                    "is_void": True,
                })

            files_needing_include.add(func_file_abs)
            matched += 1

            functions_list.append({
                "name": func_info['name'],
                "qualified_name": func_name,
                "file": func_file_abs,
                "start_line": start_line,
                "end_line": end_line,
                "brace_line": brace_line,
                "return_type": return_type,
                "params": params,
                "entry_code": entry_code,
                "exit_points": exit_points,
                "has_fall_through_exit": has_fall_through,
            })

        plan = {
            "source_functions": func_list,
            "functions": functions_list,
            "files_needing_include": sorted(files_needing_include),
            "total_functions": len(functions_list),
            "total_provided": len(func_list),
        }

        print(
            f"FunctionInstrumentor: generated plan for "
            f"{len(functions_list)} function(s) from "
            f"{matched}/{len(func_list)} provided."
        )
        return plan

    # -----------------------------------------------------------------
    # Call-graph resolution helpers
    # -----------------------------------------------------------------

    def _resolve_function(self, func_spec):
        """
        Resolve a function spec dict to a call graph entry.

        Tries three strategies:
          1. Exact match by (file, line)
          2. Match by (file, name)
          3. Match by name only
        """
        target_name = func_spec.get('name', '')
        target_file = func_spec.get('file', '')
        target_line = func_spec.get('line')

        if target_file:
            target_file = os.path.abspath(target_file)

        best = None

        for _usr, info in self.cg_functions.items():
            info_file = info.get('file', '')
            if info_file:
                info_file = os.path.abspath(info_file)

            # Strategy 1: exact file + line
            if (target_file and target_line and
                    info_file == target_file and
                    info.get('line') == target_line):
                return info

            # Strategy 2: file + name
            if (target_file and info_file == target_file and
                    info.get('name') == target_name):
                best = info

            # Strategy 3: name only (weakest)
            if not best and info.get('name') == target_name:
                best = info

        return best

    def _resolve_to_definition(self, func_info):
        """
        If *func_info* is a declaration, try to find its definition in
        the call graph.
        """
        target_name = func_info.get('name', '')
        target_qname = func_info.get('qualified_name', target_name)

        for _usr, info in self.cg_functions.items():
            if not info.get('is_definition'):
                continue
            if (info.get('qualified_name') == target_qname or
                    info.get('name') == target_name):
                info_file = info.get('file', '')
                if info_file and os.path.exists(info_file):
                    return info
        return None



# ============================================================================
# CLI entry point
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate function instrumentation plan from call graph")
    parser.add_argument("call_graph", help="Path to call_graph.json")
    parser.add_argument("func_list", help="Path to JSON file with function list")
    parser.add_argument("-c", "--compile-commands",
                        help="Path to compile_commands.json")
    parser.add_argument("-o", "--output",
                        default="function_instrumentation_plan.json",
                        help="Output plan JSON path")
    args = parser.parse_args()

    with open(args.func_list, 'r') as f:
        funcs = json.load(f)

    instr = FunctionInstrumentor(args.call_graph, args.compile_commands)
    plan = instr.generate_instrumentation_plan(funcs)

    with open(args.output, 'w') as f:
        json.dump(plan, f, indent=2)
    print(f"Plan written to {args.output}")
