#!/usr/bin/env python3
"""
Call Graph Builder for C/C++ projects.
This module analyzes a project to build a complete call graph showing which functions call which.
"""
import json
import shlex
import os
from clang import cindex
from collections import defaultdict
from multiprocessing import Pool, cpu_count

# Set libclang if necessary
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
    if any(abs_path.startswith(prefix) for prefix in system_prefixes):
        return True

    # Treat files inside build directories as system files.
    # These include fetched dependencies (e.g. catch2 in
    # build_native/_deps/) that should not be instrumented.
    for build_dir in ('/build_native/', '/build_wasm/', '/build/'):
        if build_dir in abs_path:
            return True

    return False


class FunctionInfo:
    """Stores information about a function definition"""
    def __init__(self, cursor):
        self.name = cursor.spelling
        self.qualified_name = cursor.displayname
        self.usr = cursor.get_usr()  # Unique identifier
        self.file = cursor.location.file.name if cursor.location.file else None
        self.line = cursor.location.line if cursor.location else None
        self.end_line = cursor.extent.end.line if cursor.extent else self.line  # Function end line
        self.column = cursor.location.column if cursor.location else None
        self.is_definition = cursor.is_definition()
        self.return_type = cursor.result_type.spelling if hasattr(cursor, 'result_type') else None
        # Store parameter types
        self.params = []
        for arg in cursor.get_arguments():
            self.params.append({
                'name': arg.spelling,
                'type': arg.type.spelling
            })

        # Detect compiler-generated implicit functions (copy/move ctors,
        # default ctors, dtors, assignment operators).  These have their
        # source location identical to the parent class definition — they
        # don't correspond to real source text and must not be instrumented.
        self.is_implicit = False
        if cursor.kind in (cindex.CursorKind.CONSTRUCTOR,
                           cindex.CursorKind.DESTRUCTOR,
                           cindex.CursorKind.CXX_METHOD):
            parent = cursor.semantic_parent
            if (parent and
                    parent.kind in (cindex.CursorKind.CLASS_DECL,
                                    cindex.CursorKind.STRUCT_DECL,
                                    cindex.CursorKind.CLASS_TEMPLATE,
                                    cindex.CursorKind.CLASS_TEMPLATE_PARTIAL_SPECIALIZATION) and
                    cursor.location.line == parent.location.line and
                    cursor.location.column == parent.location.column):
                self.is_implicit = True

    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        return {
            'name': self.name,
            'qualified_name': self.qualified_name,
            'usr': self.usr,
            'file': self.file,
            'line': self.line,
            'end_line': self.end_line,
            'column': self.column,
            'is_definition': self.is_definition,
            'is_implicit': self.is_implicit,
            'return_type': self.return_type,
            'params': self.params
        }


def _process_file_worker(args_tuple):
    """
    Worker function for parallel file processing.
    Must be a module-level function for multiprocessing.

    Args:
        args_tuple: (file_path, compile_args, working_directory, file_num, total_files)

    Returns:
        dict: Extracted data (functions, call_edges, classes, etc.)
    """
    file_path, compile_args, working_directory, file_num, total_files = args_tuple

    try:
        # Each worker creates its own Index (thread-safe)
        index = cindex.Index.create()
        full_path = os.path.join(working_directory, file_path) \
                    if not os.path.isabs(file_path) else file_path

        tu = index.parse(
            full_path,
            args=compile_args,
            options=cindex.TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD
        )

        current_file_abs = os.path.abspath(full_path)

        # Initialize temporary storage for this file
        result = {
            'functions': {},
            'call_edges': defaultdict(set),
            'reverse_call_edges': defaultdict(set),
            'file_to_functions': defaultdict(set),
            'classes': {},
            'class_bases': defaultdict(set),
            'class_derived': defaultdict(set),
            'method_to_class': {},
            'file_path': os.path.basename(file_path),
            'file_num': file_num,
            'total_files': total_files
        }

        # Extract data using helper functions
        _extract_class_hierarchy_worker(tu.cursor, current_file_abs, result)
        _extract_function_definitions_worker(tu.cursor, current_file_abs, result)
        _extract_function_calls_worker(tu.cursor, None, current_file_abs, result)

        # Convert sets to lists for JSON serialization
        result['call_edges'] = {k: list(v) for k, v in result['call_edges'].items()}
        result['reverse_call_edges'] = {k: list(v) for k, v in result['reverse_call_edges'].items()}
        result['file_to_functions'] = {k: list(v) for k, v in result['file_to_functions'].items()}
        result['class_bases'] = {k: list(v) for k, v in result['class_bases'].items()}
        result['class_derived'] = {k: list(v) for k, v in result['class_derived'].items()}

        return result

    except Exception as e:
        return {
            'error': str(e),
            'file_path': file_path,
            'file_num': file_num,
            'total_files': total_files
        }


def _extract_class_hierarchy_worker(cursor, current_file, result):
    """Worker version of extract_class_hierarchy"""
    try:
        if cursor.location.file:
            cursor_file = os.path.abspath(cursor.location.file.name)
            if is_system_file(cursor_file):
                return

        if cursor.kind in (cindex.CursorKind.CLASS_DECL,
                         cindex.CursorKind.STRUCT_DECL,
                         cindex.CursorKind.CLASS_TEMPLATE):
            class_usr = cursor.get_usr()
            if class_usr:
                result['classes'][class_usr] = cursor.spelling

                for base in cursor.get_children():
                    if base.kind == cindex.CursorKind.CXX_BASE_SPECIFIER:
                        base_type = base.type
                        if base_type:
                            base_decl = base_type.get_declaration()
                            if base_decl:
                                base_usr = base_decl.get_usr()
                                if base_usr:
                                    result['class_bases'][class_usr].add(base_usr)
                                    result['class_derived'][base_usr].add(class_usr)
    except (ValueError, AttributeError):
        pass

    for child in cursor.get_children():
        _extract_class_hierarchy_worker(child, current_file, result)


def _extract_function_definitions_worker(cursor, current_file, result):
    """Worker version of extract_function_definitions"""
    try:
        if cursor.location.file:
            cursor_file = os.path.abspath(cursor.location.file.name)
            if is_system_file(cursor_file):
                return

        if cursor.kind in (cindex.CursorKind.FUNCTION_DECL,
                         cindex.CursorKind.CXX_METHOD,
                         cindex.CursorKind.CONSTRUCTOR,
                         cindex.CursorKind.DESTRUCTOR,
                         cindex.CursorKind.FUNCTION_TEMPLATE):
            usr = cursor.get_usr()
            if usr:
                if usr not in result['functions']:
                    result['functions'][usr] = FunctionInfo(cursor).to_dict()
                elif cursor.is_definition():
                    result['functions'][usr] = FunctionInfo(cursor).to_dict()

                if cursor.location.file:
                    file_path = os.path.abspath(cursor.location.file.name)
                    result['file_to_functions'][file_path].add(usr)

                if cursor.kind in (cindex.CursorKind.CXX_METHOD,
                                   cindex.CursorKind.FUNCTION_TEMPLATE):
                    parent = cursor.semantic_parent
                    if parent and parent.kind in (cindex.CursorKind.CLASS_DECL,
                                                  cindex.CursorKind.STRUCT_DECL,
                                                  cindex.CursorKind.CLASS_TEMPLATE):
                        class_usr = parent.get_usr()
                        if class_usr:
                            result['method_to_class'][usr] = class_usr
    except (ValueError, AttributeError):
        pass

    for child in cursor.get_children():
        _extract_function_definitions_worker(child, current_file, result)


def _extract_function_calls_worker(cursor, current_function_usr, current_file, result):
    """Worker version of extract_function_calls

    Enhanced to capture base class and delegating constructor calls in initializer lists.
    """
    try:
        if cursor.location.file:
            cursor_file = os.path.abspath(cursor.location.file.name)
            if is_system_file(cursor_file):
                return

        if cursor.kind in (cindex.CursorKind.FUNCTION_DECL,
                         cindex.CursorKind.CXX_METHOD,
                         cindex.CursorKind.CONSTRUCTOR,
                         cindex.CursorKind.DESTRUCTOR,
                         cindex.CursorKind.FUNCTION_TEMPLATE):
            current_function_usr = cursor.get_usr()

            # NEW: For constructors, explicitly process initializer list first
            if cursor.kind == cindex.CursorKind.CONSTRUCTOR and current_function_usr:
                children = list(cursor.get_children())
                for child in children:
                    # Stop when we hit the function body
                    if child.kind == cindex.CursorKind.COMPOUND_STMT:
                        break
                    # Process CALL_EXPR in initializer list (base class or delegating ctor)
                    if child.kind == cindex.CursorKind.CALL_EXPR:
                        referenced = child.referenced
                        if referenced:
                            callee_usr = referenced.get_usr()
                            if callee_usr:
                                result['call_edges'][current_function_usr].add(callee_usr)
                                result['reverse_call_edges'][callee_usr].add(current_function_usr)

                                if callee_usr not in result['functions']:
                                    try:
                                        if referenced.location.file:
                                            ref_file = os.path.abspath(referenced.location.file.name)
                                            if not is_system_file(ref_file):
                                                result['functions'][callee_usr] = FunctionInfo(referenced).to_dict()
                                                result['file_to_functions'][ref_file].add(callee_usr)
                                    except (ValueError, AttributeError):
                                        pass
                    # Process TYPE_REF in initializer list (base class or delegating ctor)
                    elif child.kind == cindex.CursorKind.TYPE_REF:
                        referenced = child.referenced
                        if referenced:
                            ref_file = str(referenced.location.file) if referenced.location.file else None
                            if ref_file and not is_system_file(ref_file):
                                # For TYPE_REF, find constructors in the referenced class
                                if referenced.kind in (cindex.CursorKind.CLASS_DECL,
                                                      cindex.CursorKind.STRUCT_DECL,
                                                      cindex.CursorKind.CLASS_TEMPLATE):
                                    for type_child in referenced.get_children():
                                        if type_child.kind == cindex.CursorKind.CONSTRUCTOR:
                                            ctor_file = str(type_child.location.file) if type_child.location.file else None
                                            if ctor_file and not is_system_file(ctor_file):
                                                callee_usr = type_child.get_usr()
                                                if callee_usr:
                                                    result['call_edges'][current_function_usr].add(callee_usr)
                                                    result['reverse_call_edges'][callee_usr].add(current_function_usr)

                                                    if callee_usr not in result['functions']:
                                                        try:
                                                            result['functions'][callee_usr] = FunctionInfo(type_child).to_dict()
                                                            result['file_to_functions'][ctor_file].add(callee_usr)
                                                        except (ValueError, AttributeError):
                                                            pass

        if cursor.kind == cindex.CursorKind.CALL_EXPR:
            referenced = cursor.referenced
            if referenced:
                callee_usr = referenced.get_usr()
                if callee_usr and current_function_usr:
                    result['call_edges'][current_function_usr].add(callee_usr)
                    result['reverse_call_edges'][callee_usr].add(current_function_usr)

                    if callee_usr not in result['functions']:
                        try:
                            if referenced.location.file:
                                ref_file = os.path.abspath(referenced.location.file.name)
                                if not is_system_file(ref_file):
                                    result['functions'][callee_usr] = FunctionInfo(referenced).to_dict()
                                    result['file_to_functions'][ref_file].add(callee_usr)
                        except (ValueError, AttributeError):
                            pass

    except (ValueError, AttributeError):
        pass

    for child in cursor.get_children():
        _extract_function_calls_worker(child, current_function_usr, current_file, result)


class CallGraphBuilder:
    """
    Builds a call graph for a C/C++ project using libclang.

    The call graph maps functions to the functions they call.
    """

    def __init__(self, compile_commands_path, parallel=True, max_workers=None):
        """
        Initialize the call graph builder.

        Args:
            compile_commands_path: Path to compile_commands.json
            parallel: If True, use parallel processing (default: True)
            max_workers: Maximum number of worker processes (default: cpu_count())
        """
        self.compile_commands_path = compile_commands_path
        self.commands = []
        self.parallel = parallel
        self.max_workers = max_workers or cpu_count()

        # Maps USR -> FunctionInfo
        self.functions = {}

        # Maps caller_usr -> set of callee_usrs
        self.call_edges = defaultdict(set)

        # Maps file -> set of function_usrs defined in that file
        self.file_to_functions = defaultdict(set)

        # Maps function_usr -> set of caller_usrs (reverse call graph)
        self.reverse_call_edges = defaultdict(set)

        # Class hierarchy tracking
        # Maps class_usr -> set of base_class_usrs (direct bases only)
        self.class_bases = defaultdict(set)

        # Maps class_usr -> set of derived_class_usrs (direct derived only)
        self.class_derived = defaultdict(set)

        # Maps class_usr -> class name (for debugging)
        self.classes = {}

        # Virtual method tracking
        # Maps method_usr -> set of override_usrs (all overrides in derived classes)
        self.virtual_method_overrides = defaultdict(set)

        # Maps method_usr -> class_usr (which class this method belongs to)
        self.method_to_class = {}

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

    def extract_class_hierarchy(self, cursor, current_file=None):
        """
        Recursively traverse AST to extract class hierarchy information.

        Args:
            cursor: Current AST cursor
            current_file: Absolute path of file being processed (for filtering)
        """
        try:
            # Only process nodes in the current file
            if cursor.location.file:
                cursor_file = os.path.abspath(cursor.location.file.name)

                # Skip system files
                if is_system_file(cursor_file):
                    return

            # Check if this is a class or struct declaration
            if cursor.kind in (cindex.CursorKind.CLASS_DECL,
                             cindex.CursorKind.STRUCT_DECL,
                             cindex.CursorKind.CLASS_TEMPLATE):
                class_usr = cursor.get_usr()
                if class_usr:
                    # Store class info
                    self.classes[class_usr] = cursor.spelling

                    # Extract base classes
                    for base in cursor.get_children():
                        if base.kind == cindex.CursorKind.CXX_BASE_SPECIFIER:
                            # Get the base class
                            base_type = base.type
                            if base_type:
                                # Get the declaration of the base class
                                base_decl = base_type.get_declaration()
                                if base_decl:
                                    base_usr = base_decl.get_usr()
                                    if base_usr:
                                        # Record inheritance relationship
                                        self.class_bases[class_usr].add(base_usr)
                                        self.class_derived[base_usr].add(class_usr)

        except (ValueError, AttributeError):
            # Skip problematic cursors
            pass

        # Recurse into children
        for child in cursor.get_children():
            self.extract_class_hierarchy(child, current_file)

    def extract_function_definitions(self, cursor, current_file=None):
        """
        Recursively traverse AST to find all function definitions.

        Args:
            cursor: Current AST cursor
            current_file: Absolute path of file being processed (for filtering)
        """
        try:
            # Only process nodes in the current file (avoid system headers)
            if cursor.location.file:
                cursor_file = os.path.abspath(cursor.location.file.name)

                # Skip system files entirely (but allow user headers!)
                if is_system_file(cursor_file):
                    return

                # NOTE: We removed the current_file != cursor_file check
                # This allows us to process header files that are included
                # Deduplication is handled by USR checking (line 136)

            # Check for USING_DECLARATION to handle inherited constructors
            # Example: using base_class::base_class;
            if cursor.kind == cindex.CursorKind.USING_DECLARATION:
                # Get the referenced declaration
                for child in cursor.get_children():
                    if child.kind == cindex.CursorKind.OVERLOADED_DECL_REF:
                        # This references overloaded constructors
                        # Iterate through all overloads
                        for overload_child in child.get_children():
                            if overload_child.kind == cindex.CursorKind.CONSTRUCTOR:
                                usr = overload_child.get_usr()
                                if usr and usr not in self.functions:
                                    # Create entry for the inherited constructor
                                    self.functions[usr] = FunctionInfo(overload_child)
                                    if overload_child.location.file:
                                        file_path = os.path.abspath(overload_child.location.file.name)
                                        self.file_to_functions[file_path].add(usr)

            # Check if this is a function declaration/definition
            if cursor.kind in (cindex.CursorKind.FUNCTION_DECL,
                             cindex.CursorKind.CXX_METHOD,
                             cindex.CursorKind.CONSTRUCTOR,
                             cindex.CursorKind.DESTRUCTOR,
                             cindex.CursorKind.FUNCTION_TEMPLATE):
                usr = cursor.get_usr()
                if usr:
                    # Store or update function info
                    if usr not in self.functions:
                        self.functions[usr] = FunctionInfo(cursor)
                    elif cursor.is_definition():
                        # Update with definition if we only had a declaration
                        self.functions[usr] = FunctionInfo(cursor)

                    # Track which file this function is in
                    if cursor.location.file:
                        file_path = os.path.abspath(cursor.location.file.name)
                        self.file_to_functions[file_path].add(usr)

                    # Track virtual methods and their class membership
                    if cursor.kind in (cindex.CursorKind.CXX_METHOD,
                                       cindex.CursorKind.FUNCTION_TEMPLATE):
                        # Find the parent class
                        parent = cursor.semantic_parent
                        if parent and parent.kind in (cindex.CursorKind.CLASS_DECL,
                                                      cindex.CursorKind.STRUCT_DECL,
                                                      cindex.CursorKind.CLASS_TEMPLATE):
                            class_usr = parent.get_usr()
                            if class_usr:
                                self.method_to_class[usr] = class_usr

                        # Check if this is a virtual method
                        if cursor.is_virtual_method():
                            # Track this as a virtual method
                            # We'll build the override relationships after all classes are processed
                            pass

        except (ValueError, AttributeError):
            # Skip problematic cursors
            pass

        # Recurse into children
        for child in cursor.get_children():
            self.extract_function_definitions(child, current_file)

    def _add_implicit_destructor_edge(self, constructor_cursor,
                                       current_function_usr):
        """Infer and add an implicit destructor call edge.

        C++ destructors run implicitly at scope exit for stack objects,
        on ``delete`` for heap objects, and at end-of-full-expression
        for temporaries.  libclang doesn't emit ``CALL_EXPR`` for any
        of these.  When we see a constructor call, we add a
        corresponding destructor edge from the same calling function —
        a sound over-approximation that ensures the destructor is in
        the reachable set for instrumentation.

        Args:
            constructor_cursor: The libclang cursor for the constructor
                being called.  Its ``semantic_parent`` is the class
                that owns the destructor.
            current_function_usr: USR of the function that triggers the
                construction (and thus the eventual destruction).
        """
        if not current_function_usr:
            return
        parent_class = constructor_cursor.semantic_parent
        if not parent_class:
            return
        for child in parent_class.get_children():
            try:
                if child.kind != cindex.CursorKind.DESTRUCTOR:
                    continue
            except ValueError:
                continue
            dtor_usr = child.get_usr()
            if not dtor_usr:
                continue
            self.call_edges[current_function_usr].add(dtor_usr)
            self.reverse_call_edges[dtor_usr].add(current_function_usr)
            if dtor_usr not in self.functions:
                try:
                    if child.location.file:
                        dtor_file = os.path.abspath(child.location.file.name)
                        if not is_system_file(dtor_file):
                            self.functions[dtor_usr] = FunctionInfo(child)
                            self.file_to_functions[dtor_file].add(dtor_usr)
                except (ValueError, AttributeError):
                    pass
            break  # at most one destructor per class

    def _infer_implicit_destructor_edges(self):
        """Post-processing: add destructor edges from constructor callers.

        Runs AFTER all parallel/sequential results are merged into
        ``self.functions`` and ``self.call_edges``.  For every
        constructor that has callers, finds the matching destructor in
        the same class and adds edges from every constructor-caller to
        the destructor.

        This is idempotent — adding an edge that already exists is a
        no-op because ``call_edges`` stores ``set``s.
        """
        from collections import defaultdict

        # Build name → [USR] index for fast destructor lookup.
        name_to_usrs = defaultdict(list)
        for usr, func_info in self.functions.items():
            info = (func_info if isinstance(func_info, dict)
                    else func_info.to_dict())
            name_to_usrs[info.get('name', '')].append(usr)

        added = 0
        for usr in list(self.functions.keys()):
            func_info = self.functions[usr]
            info = (func_info if isinstance(func_info, dict)
                    else func_info.to_dict())
            name = info.get('name', '')
            if not name or name.startswith('~'):
                continue  # skip destructors / nameless

            # Only process USRs that look like constructors: the
            # function name equals the class name from the USR's
            # scope (c:@...@S@ClassName@F@ClassName#...).
            # Quick heuristic: split on @F@ and check the last
            # scope segment.
            parts = usr.rsplit('@F@', 1)
            if len(parts) != 2:
                continue
            scope = parts[0]            # e.g. c:@N@el@N@base@S@PErrorWriter
            scope_parts = scope.split('@')
            if not scope_parts:
                continue
            class_name = scope_parts[-1]   # e.g. PErrorWriter
            if name != class_name:
                continue  # not a constructor

            # No callers → no edges to propagate.
            callers = self.reverse_call_edges.get(usr, set())
            if not callers:
                continue

            # Find matching destructor: ~ClassName in the same class.
            dtor_name = '~' + name
            dtor_usr = None
            for candidate_usr in name_to_usrs.get(dtor_name, []):
                # Same class scope → same prefix before @F@.
                cand_parts = candidate_usr.rsplit('@F@', 1)
                if len(cand_parts) == 2 and cand_parts[0] == scope:
                    dtor_usr = candidate_usr
                    break

            if not dtor_usr:
                continue  # no user-defined destructor (compiler-generated)

            # Copy every constructor caller → destructor.
            for caller_usr in callers:
                if dtor_usr not in self.call_edges.get(caller_usr, set()):
                    self.call_edges[caller_usr].add(dtor_usr)
                    self.reverse_call_edges[dtor_usr].add(caller_usr)
                    added += 1

        if added:
            print(f"\nAdding implicit destructor edges...")
            print(f"Added {added} implicit destructor call edge(s)")

    def extract_function_calls(self, cursor, current_function_usr=None, current_file=None):
        """
        Recursively traverse AST to find all function calls.

        Enhanced to capture base class and delegating constructor calls in initializer lists.

        Args:
            cursor: Current AST cursor
            current_function_usr: USR of the enclosing function (caller)
            current_file: Absolute path of file being processed (for filtering)
        """
        try:
            # Only process nodes in the current file (but allow user headers)
            if cursor.location.file:
                cursor_file = os.path.abspath(cursor.location.file.name)

                # Skip recursing into system files
                if is_system_file(cursor_file):
                    return

                # NOTE: We removed the current_file != cursor_file check
                # This allows us to capture calls from inline/template functions in headers
                # Deduplication is handled by set membership in call_edges

            # Update current function context if we enter a function
            if cursor.kind in (cindex.CursorKind.FUNCTION_DECL,
                             cindex.CursorKind.CXX_METHOD,
                             cindex.CursorKind.CONSTRUCTOR,
                             cindex.CursorKind.DESTRUCTOR,
                             cindex.CursorKind.FUNCTION_TEMPLATE):
                current_function_usr = cursor.get_usr()

                # NEW: For constructors, explicitly process initializer list first
                if cursor.kind == cindex.CursorKind.CONSTRUCTOR and current_function_usr:
                    children = list(cursor.get_children())
                    for child in children:
                        # Stop when we hit the function body
                        if child.kind == cindex.CursorKind.COMPOUND_STMT:
                            break
                        # Process CALL_EXPR in initializer list (base class or delegating ctor)
                        if child.kind == cindex.CursorKind.CALL_EXPR:
                            referenced = child.referenced
                            if referenced:
                                callee_usr = referenced.get_usr()
                                if callee_usr:
                                    self.call_edges[current_function_usr].add(callee_usr)
                                    self.reverse_call_edges[callee_usr].add(current_function_usr)

                                    if callee_usr not in self.functions:
                                        try:
                                            if referenced.location.file:
                                                ref_file = os.path.abspath(referenced.location.file.name)
                                                if not is_system_file(ref_file):
                                                    self.functions[callee_usr] = FunctionInfo(referenced)
                                                    self.file_to_functions[ref_file].add(callee_usr)
                                        except (ValueError, AttributeError):
                                            pass
                        # Process TYPE_REF in initializer list (base class or delegating ctor)
                        elif child.kind == cindex.CursorKind.TYPE_REF:
                            referenced = child.referenced
                            if referenced:
                                ref_file = str(referenced.location.file) if referenced.location.file else None
                                if ref_file and not is_system_file(ref_file):
                                    # For TYPE_REF, find constructors in the referenced class
                                    if referenced.kind in (cindex.CursorKind.CLASS_DECL,
                                                          cindex.CursorKind.STRUCT_DECL,
                                                          cindex.CursorKind.CLASS_TEMPLATE):
                                        for type_child in referenced.get_children():
                                            if type_child.kind == cindex.CursorKind.CONSTRUCTOR:
                                                ctor_file = str(type_child.location.file) if type_child.location.file else None
                                                if ctor_file and not is_system_file(ctor_file):
                                                    callee_usr = type_child.get_usr()
                                                    if callee_usr:
                                                        self.call_edges[current_function_usr].add(callee_usr)
                                                        self.reverse_call_edges[callee_usr].add(current_function_usr)

                                                        if callee_usr not in self.functions:
                                                            try:
                                                                self.functions[callee_usr] = FunctionInfo(type_child)
                                                                self.file_to_functions[ctor_file].add(callee_usr)
                                                            except (ValueError, AttributeError):
                                                                pass

            # Check if this is a function call
            if cursor.kind == cindex.CursorKind.CALL_EXPR:
                # Get the referenced function (callee)
                referenced = cursor.referenced
                if referenced:
                    callee_usr = referenced.get_usr()
                    if callee_usr and current_function_usr:
                        # Add edge: caller -> callee (even if callee is a system function)
                        # We want to know that user code calls system functions
                        self.call_edges[current_function_usr].add(callee_usr)
                        # Add reverse edge: callee <- caller
                        self.reverse_call_edges[callee_usr].add(current_function_usr)

                        # If this function isn't in our functions dict yet, try to add it
                        # This handles template instantiations and other edge cases
                        if callee_usr not in self.functions:
                            try:
                                # Try to create a function info from the referenced cursor
                                if referenced.location.file:
                                    ref_file = os.path.abspath(referenced.location.file.name)
                                    # Only add if it's not a system file
                                    if not is_system_file(ref_file):
                                        self.functions[callee_usr] = FunctionInfo(referenced)
                                        self.file_to_functions[ref_file].add(callee_usr)
                            except (ValueError, AttributeError):
                                pass

            # Check for implicit constructor calls in variable declarations
            # Example: MyClass obj; or MyClass obj(args);
            if cursor.kind == cindex.CursorKind.VAR_DECL and current_function_usr:
                # Look for constructor calls in the children
                for child in cursor.get_children():
                    if child.kind == cindex.CursorKind.CXX_CONSTRUCT_EXPR:
                        # This is a constructor call
                        # Get the constructor being called
                        constructor = child.referenced
                        if not constructor:
                            # Try to get it from the type
                            var_type = cursor.type
                            if var_type:
                                type_decl = var_type.get_declaration()
                                if type_decl:
                                    # Find the constructor in the children
                                    for type_child in type_decl.get_children():
                                        if type_child.kind == cindex.CursorKind.CONSTRUCTOR:
                                            constructor = type_child
                                            break

                        if constructor:
                            constructor_usr = constructor.get_usr()
                            if constructor_usr:
                                self.call_edges[current_function_usr].add(constructor_usr)
                                self.reverse_call_edges[constructor_usr].add(current_function_usr)
                                # Try to add constructor info if not already present
                                if constructor_usr not in self.functions:
                                    try:
                                        if constructor.location.file:
                                            ctor_file = os.path.abspath(constructor.location.file.name)
                                            if not is_system_file(ctor_file):
                                                self.functions[constructor_usr] = FunctionInfo(constructor)
                                                self.file_to_functions[ctor_file].add(constructor_usr)
                                    except (ValueError, AttributeError):
                                        pass
                                # Implicit destructor edge: scope exit
                                # calls ~T() for stack-allocated objects.
                                self._add_implicit_destructor_edge(
                                    constructor, current_function_usr)
                        break  # Only process the first constructor call

            # Check for constructor calls in new expressions
            # Example: new MyClass(args);
            if cursor.kind == cindex.CursorKind.CXX_NEW_EXPR and current_function_usr:
                # Look for the constructor call
                for child in cursor.get_children():
                    if child.kind == cindex.CursorKind.CXX_CONSTRUCT_EXPR:
                        # Find the constructor being called
                        for construct_child in child.get_children():
                            if construct_child.kind == cindex.CursorKind.CALL_EXPR:
                                referenced = construct_child.referenced
                                if referenced:
                                    constructor_usr = referenced.get_usr()
                                    if constructor_usr:
                                        self.call_edges[current_function_usr].add(constructor_usr)
                                        self.reverse_call_edges[constructor_usr].add(current_function_usr)
                                        # Try to add constructor info if not already present
                                        if constructor_usr not in self.functions:
                                            try:
                                                if referenced.location.file:
                                                    ctor_file = os.path.abspath(referenced.location.file.name)
                                                    if not is_system_file(ctor_file):
                                                        self.functions[constructor_usr] = FunctionInfo(referenced)
                                                        self.file_to_functions[ctor_file].add(constructor_usr)
                                            except (ValueError, AttributeError):
                                                pass
                                        # Implicit destructor for new expr
                                        self._add_implicit_destructor_edge(
                                            referenced,
                                            current_function_usr)

                        # If no explicit call, try to get constructor directly
                        if child.type:
                            type_decl = child.type.get_declaration()
                            if type_decl:
                                for type_child in type_decl.get_children():
                                    if type_child.kind == cindex.CursorKind.CONSTRUCTOR:
                                        constructor_usr = type_child.get_usr()
                                        if constructor_usr:
                                            self.call_edges[current_function_usr].add(constructor_usr)
                                            self.reverse_call_edges[constructor_usr].add(current_function_usr)
                                            # Try to add constructor info if not already present
                                            if constructor_usr not in self.functions:
                                                try:
                                                    if type_child.location.file:
                                                        ctor_file = os.path.abspath(type_child.location.file.name)
                                                        if not is_system_file(ctor_file):
                                                            self.functions[constructor_usr] = FunctionInfo(type_child)
                                                            self.file_to_functions[ctor_file].add(constructor_usr)
                                                except (ValueError, AttributeError):
                                                    pass
                                            # Implicit destructor for new
                                            # expr (fallback path).
                                            self._add_implicit_destructor_edge(
                                                type_child,
                                                current_function_usr)
                                            break
                        break

            # Check for implicit copy constructors in CXX_CONSTRUCT_EXPR
            # This catches copy constructors used when passing by value
            if cursor.kind == cindex.CursorKind.CXX_CONSTRUCT_EXPR and current_function_usr:
                # Check if this is a copy or move constructor
                constructor_kind = cursor.get_num_arguments()
                if constructor_kind > 0:  # Has arguments, might be copy/move
                    # Try to get the constructor being called
                    if cursor.type:
                        type_decl = cursor.type.get_declaration()
                        if type_decl:
                            # Look for constructors in the class
                            for type_child in type_decl.get_children():
                                if type_child.kind == cindex.CursorKind.CONSTRUCTOR:
                                    # Check if it's a copy or move constructor by looking at parameters
                                    params = list(type_child.get_arguments())
                                    if len(params) == 1:  # Copy/move constructors have 1 param
                                        constructor_usr = type_child.get_usr()
                                        if constructor_usr:
                                            self.call_edges[current_function_usr].add(constructor_usr)
                                            self.reverse_call_edges[constructor_usr].add(current_function_usr)
                                            # Try to add constructor info if not already present
                                            if constructor_usr not in self.functions:
                                                try:
                                                    if type_child.location.file:
                                                        ctor_file = os.path.abspath(type_child.location.file.name)
                                                        if not is_system_file(ctor_file):
                                                            self.functions[constructor_usr] = FunctionInfo(type_child)
                                                            self.file_to_functions[ctor_file].add(constructor_usr)
                                                except (ValueError, AttributeError):
                                                    pass
                                            # Implicit destructor for
                                            # copy/move-constructed temp.
                                            self._add_implicit_destructor_edge(
                                                type_child,
                                                current_function_usr)
                                            break

        except (ValueError, AttributeError):
            # Skip problematic cursors
            pass

        # Recurse into children
        for child in cursor.get_children():
            self.extract_function_calls(child, current_function_usr, current_file)

    def fill_missing_function_entries(self):
        """
        Post-processing step to ensure all called functions have entries in the functions dict.
        This handles edge cases where functions are called but weren't captured during traversal.
        Creates minimal stub entries for these functions.
        """
        print("\nFilling missing function entries...")

        # Find all USRs that are in call edges but not in functions dict
        all_referenced_usrs = set()
        for caller_usr in self.call_edges:
            all_referenced_usrs.update(self.call_edges[caller_usr])
        for callee_usr in self.reverse_call_edges:
            all_referenced_usrs.add(callee_usr)

        missing_usrs = all_referenced_usrs - set(self.functions.keys())

        # Filter out system functions (we don't need entries for those)
        user_missing_usrs = []
        for usr in missing_usrs:
            # Quick heuristic: if USR contains user code namespace/path indicators
            # This is imperfect but better than nothing
            if '@N@std@' not in usr and '@N@__gnu_cxx@' not in usr:
                user_missing_usrs.append(usr)

        if user_missing_usrs:
            print(f"  Found {len(user_missing_usrs)} user functions that are called but missing from functions dict")
            print(f"  Note: These are template instantiations or inherited functions that couldn't be")
            print(f"  fully resolved. They remain in the call graph but without complete metadata.")

        # Note: We don't create stub entries because we don't have enough information
        # The important fix is that we now capture these during the traversal phase
        # This method serves mainly as a diagnostic

    def add_virtual_call_edges(self):
        """
        Add call edges for virtual method calls after virtual_method_overrides is built.
        For any existing call edge to a virtual method, add edges to all its overrides.
        """
        print("\nAdding virtual call edges...")

        new_edges = []  # Store new edges to add (to avoid modifying dict during iteration)

        for caller_usr, callee_set in self.call_edges.items():
            for callee_usr in callee_set:
                # Check if this callee is a virtual method with overrides
                if callee_usr in self.virtual_method_overrides:
                    # Add edges to all overrides
                    for override_usr in self.virtual_method_overrides[callee_usr]:
                        if override_usr != callee_usr:  # Don't add self-edge
                            new_edges.append((caller_usr, override_usr))

        # Add all new edges
        for caller_usr, override_usr in new_edges:
            self.call_edges[caller_usr].add(override_usr)
            self.reverse_call_edges[override_usr].add(caller_usr)

        print(f"Added {len(new_edges)} virtual call edges")

    def _get_func_attr(self, func_info, attr):
        """
        Helper to get attribute from func_info, which may be a FunctionInfo object or dict.
        Handles both parallel (dict) and sequential (object) modes.
        """
        if isinstance(func_info, dict):
            return func_info.get(attr)
        else:
            return getattr(func_info, attr, None)

    def build_virtual_method_overrides(self):
        """
        Build virtual method override relationships after all classes are processed.
        For each virtual method, find all overrides in derived classes.
        """
        print("\nBuilding virtual method override relationships...")

        # For each method, check if it's virtual and find overrides
        for method_usr, func_info in self.functions.items():
            # Check if this is a virtual method
            if method_usr not in self.method_to_class:
                continue  # Not a method

            class_usr = self.method_to_class[method_usr]

            # Get all derived classes (recursively)
            def get_all_derived(cls_usr, visited=None):
                if visited is None:
                    visited = set()
                if cls_usr in visited:
                    return set()
                visited.add(cls_usr)

                result = set(self.class_derived.get(cls_usr, []))
                for derived in list(result):
                    result.update(get_all_derived(derived, visited))
                return result

            all_derived = get_all_derived(class_usr)

            # Find methods with the same name in derived classes
            method_name = self._get_func_attr(func_info, 'name')
            for derived_usr in all_derived:
                # Find all methods in this derived class
                for other_method_usr, other_class_usr in self.method_to_class.items():
                    if other_class_usr == derived_usr:
                        other_func_info = self.functions.get(other_method_usr)
                        if other_func_info:
                            other_name = self._get_func_attr(other_func_info, 'name')
                            if other_name == method_name:
                                # This is likely an override (name matches, in derived class)
                                # Add bidirectional relationship
                                self.virtual_method_overrides[method_usr].add(other_method_usr)
                                self.virtual_method_overrides[other_method_usr].add(method_usr)

        total_virtual_methods = len([usr for usr in self.functions.keys()
                                     if usr in self.virtual_method_overrides])
        print(f"Found {total_virtual_methods} virtual methods with overrides")

    def process_file(self, file_path, args, working_directory, file_num, total_files):
        """
        Process a single source file to extract functions and calls.

        Args:
            file_path: Path to the source file
            args: Compilation arguments
            working_directory: Working directory for compilation
            file_num: Current file number
            total_files: Total number of files
        """
        print(f"  [{file_num}/{total_files}] Processing: {os.path.basename(file_path)}", flush=True)

        try:
            # Build AST
            index = cindex.Index.create()
            full_path = os.path.join(working_directory, file_path) \
                        if not os.path.isabs(file_path) else file_path

            tu = index.parse(
                full_path,
                args=args,
                options=cindex.TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD
            )

            current_file_abs = os.path.abspath(full_path)

            # First pass: extract class hierarchy
            self.extract_class_hierarchy(tu.cursor, current_file_abs)

            # Second pass: extract function definitions
            self.extract_function_definitions(tu.cursor, current_file_abs)

            # Third pass: extract function calls
            self.extract_function_calls(tu.cursor, None, current_file_abs)

        except Exception as e:
            print(f"    Error processing {file_path}: {e}")

    def discover_and_process_header_files(self):
        """
        Discover user header files that may contain template definitions
        or other functions that weren't captured during compilation unit processing.

        This mirrors the approach in ProjectASTBuilder.py (lines 416-479).
        """
        print("\n" + "="*60)
        print("DISCOVERING HEADER FILES")
        print("="*60)

        # Get project root from compile_commands
        if not self.commands:
            return

        project_root = os.path.dirname(os.path.dirname(self.compile_commands_path))

        # Find all user header files in the project directory
        header_files = []
        _skip_dirs = {
            'build', 'build_native', 'build_wasm',
            '.git', '__pycache__',
            # Pipeline-internal snapshot / output dirs. Walking into
            # them adds duplicate function entries under backup paths,
            # which later get whitelist-filtered (since the whitelist
            # treats these dirs as implicit-skip), silently dropping
            # the real instrumentation plan entries.
            '.pre_instrumentation_originals',
            '.instrumentation_backups',
            'instrumented_files',
            'my_custom_cache',
        }
        for root, dirs, files in os.walk(project_root):
            # Skip system directories and build directories
            dirs[:] = [d for d in dirs if d not in _skip_dirs]

            for file in files:
                if file.endswith(('.hpp', '.h')):
                    full_path = os.path.abspath(os.path.join(root, file))
                    if not is_system_file(full_path):
                        # Check if this header has few/no functions captured
                        existing_funcs = len(self.file_to_functions.get(full_path, set()))
                        if existing_funcs == 0:
                            header_files.append(full_path)

        if not header_files:
            print("No additional header files to process")
            return

        print(f"Found {len(header_files)} user header files with 0 functions captured")
        print("Processing headers as separate translation units...")

        # Extract compilation flags from all compile commands
        args_set = set()
        args = []
        std_flag = None
        working_directory = self.commands[0]["directory"] if self.commands else "."

        # Collect -I, -D, and -std flags from all commands
        for cmd in self.commands:
            cmd_args = shlex.split(cmd["command"])[1:]

            i = 0
            while i < len(cmd_args):
                arg = cmd_args[i]

                # Handle -I with path attached: -I/path/to/include
                if arg.startswith("-I") and len(arg) > 2:
                    if arg not in args_set:
                        args.append(arg)
                        args_set.add(arg)
                # Handle -I as separate argument: -I /path/to/include
                elif arg == "-I" and i + 1 < len(cmd_args):
                    combined = f"-I{cmd_args[i + 1]}"
                    if combined not in args_set:
                        args.append("-I")
                        args.append(cmd_args[i + 1])
                        args_set.add(combined)
                    i += 1  # Skip next arg since we processed it
                # Handle -D defines
                elif arg.startswith("-D"):
                    if arg not in args_set:
                        args.append(arg)
                        args_set.add(arg)
                # Handle -std flag (use most recent/highest version)
                elif arg.startswith("-std="):
                    std_flag = arg  # Keep updating to get highest version

                i += 1

        # Add std flag at the end
        if std_flag:
            args.append(std_flag)
        elif "-std=c++17" not in args:
            args.append("-std=c++17")

        # Process each header file (parallelize if many headers)
        if self.parallel and len(header_files) > 10:
            print(f"Using parallel processing with {self.max_workers} workers for headers")

            # Prepare tasks for header processing
            header_tasks = []
            for idx, header_path in enumerate(header_files, start=1):
                header_tasks.append((header_path, args, working_directory, idx, len(header_files)))

            # Process headers in parallel
            with Pool(processes=self.max_workers) as pool:
                header_results = list(pool.imap_unordered(_process_file_worker, header_tasks))

            # Merge header results
            processed_count = 0
            functions_found = 0

            for result in header_results:
                if 'error' in result:
                    # Silently skip - headers often fail to parse
                    continue

                funcs_added = len(result['functions'])
                if funcs_added > 0:
                    # Merge into main data structures
                    for usr, func_info in result['functions'].items():
                        if usr not in self.functions:
                            self.functions[usr] = func_info
                        elif func_info['is_definition']:
                            self.functions[usr] = func_info

                    for caller, callees in result['call_edges'].items():
                        self.call_edges[caller].update(callees)
                    for callee, callers in result['reverse_call_edges'].items():
                        self.reverse_call_edges[callee].update(callers)

                    for file_path, funcs in result['file_to_functions'].items():
                        self.file_to_functions[file_path].update(funcs)

                    self.classes.update(result['classes'])
                    for class_usr, bases in result['class_bases'].items():
                        self.class_bases[class_usr].update(bases)
                    for base_usr, derived in result['class_derived'].items():
                        self.class_derived[base_usr].update(derived)

                    self.method_to_class.update(result['method_to_class'])

                    processed_count += 1
                    functions_found += funcs_added

        else:
            # Sequential processing for small number of headers
            processed_count = 0
            functions_found = 0

            for header_idx, header_path in enumerate(header_files, start=1):
                if header_idx % 10 == 0:
                    print(f"  Processed {header_idx}/{len(header_files)} headers...", flush=True)

                try:
                    # Build AST for this header as a translation unit
                    index = cindex.Index.create()
                    tu = index.parse(
                        header_path,
                        args=args,
                        options=cindex.TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD
                    )

                    current_file_abs = os.path.abspath(header_path)

                    # Track function count before processing
                    funcs_before = len(self.functions)

                    # Extract class hierarchy
                    self.extract_class_hierarchy(tu.cursor, current_file_abs)

                    # Extract function definitions (including template definitions)
                    self.extract_function_definitions(tu.cursor, current_file_abs)

                    # Extract function calls
                    self.extract_function_calls(tu.cursor, None, current_file_abs)

                    # Track function count after processing
                    funcs_after = len(self.functions)
                    funcs_added = funcs_after - funcs_before

                    if funcs_added > 0:
                        processed_count += 1
                        functions_found += funcs_added

                except Exception as e:
                    # Silently skip headers that fail to parse (common for incomplete templates)
                    pass

        print(f"\n✓ Processed {processed_count} headers")
        print(f"✓ Found {functions_found} additional functions in headers")

    def build_call_graph(self):
        """
        Build the complete call graph for the project.

        Returns:
            dict: The call graph data structure
        """
        print("\n" + "="*60)
        print("BUILDING CALL GRAPH")
        print("="*60)

        # Load compilation database
        self.load_compile_commands()

        total_files = len(self.commands)
        print(f"Processing {total_files} files...")

        if self.parallel and total_files > 1:
            print(f"Using parallel processing with {self.max_workers} workers\n")

            # Prepare tasks for worker pool
            tasks = []
            for file_num, entry in enumerate(self.commands, start=1):
                directory = entry["directory"]
                file = entry["file"]
                command = entry["command"]

                # Parse compilation arguments
                args = shlex.split(command)[1:]
                # Remove output file arguments
                if "-o" in args:
                    idx = args.index("-o")
                    del args[idx:idx+2]
                if file in args:
                    args.remove(file)

                tasks.append((file, args, directory, file_num, total_files))

            # Process files in parallel
            with Pool(processes=self.max_workers) as pool:
                results = list(pool.imap_unordered(_process_file_worker, tasks))

            # Merge results from all workers
            print(f"\nMerging results from {len(results)} workers...")
            processed = 0
            errors = 0

            for result in results:
                if 'error' in result:
                    errors += 1
                    print(f"  Error processing {result.get('file_path', 'unknown')}: {result['error']}")
                    continue

                # Merge functions (prefer definitions over declarations)
                for usr, func_info in result['functions'].items():
                    if usr not in self.functions:
                        self.functions[usr] = func_info
                    elif func_info['is_definition']:
                        self.functions[usr] = func_info

                # Merge call edges (convert back to sets)
                for caller, callees in result['call_edges'].items():
                    self.call_edges[caller].update(callees)
                for callee, callers in result['reverse_call_edges'].items():
                    self.reverse_call_edges[callee].update(callers)

                # Merge file_to_functions
                for file_path, funcs in result['file_to_functions'].items():
                    self.file_to_functions[file_path].update(funcs)

                # Merge class hierarchy
                self.classes.update(result['classes'])
                for class_usr, bases in result['class_bases'].items():
                    self.class_bases[class_usr].update(bases)
                for base_usr, derived in result['class_derived'].items():
                    self.class_derived[base_usr].update(derived)

                # Merge method_to_class
                self.method_to_class.update(result['method_to_class'])

                processed += 1

            print(f"✓ Successfully processed {processed} files")
            if errors > 0:
                print(f"⚠ {errors} files had errors")

        else:
            # Sequential processing (for single file or if parallel=False)
            print("Using sequential processing\n")
            for file_num, entry in enumerate(self.commands, start=1):
                directory = entry["directory"]
                file = entry["file"]
                command = entry["command"]

                # Parse compilation arguments
                args = shlex.split(command)[1:]
                # Remove output file arguments
                if "-o" in args:
                    idx = args.index("-o")
                    del args[idx:idx+2]
                if file in args:
                    args.remove(file)

                # Process the file
                self.process_file(file, args, directory, file_num, total_files)

        # Discover and process user header files that weren't captured
        self.discover_and_process_header_files()

        # Build virtual method override relationships
        self.build_virtual_method_overrides()

        # Add virtual call edges to the call graph
        self.add_virtual_call_edges()

        # Check for any missing function entries (diagnostic)
        self.fill_missing_function_entries()

        # Add implicit destructor edges.
        # C++ destructors are called implicitly at scope exit, on
        # ``delete``, or at end-of-full-expression. libclang doesn't
        # emit CALL_EXPR for any of these. For every constructor that
        # has call edges, find the matching destructor in the same
        # class and copy the caller set to the destructor. This
        # ensures destructors are reachable for instrumentation.
        self._infer_implicit_destructor_edges()

        # Print summary
        print("\n" + "="*60)
        print("CALL GRAPH STATISTICS")
        print("="*60)
        print(f"Total functions found: {len(self.functions)}")
        print(f"Total call edges: {sum(len(callees) for callees in self.call_edges.values())}")
        print(f"  (includes direct calls, virtual dispatches, and implicit constructor calls)")
        print(f"Files analyzed: {len(self.file_to_functions)}")
        print(f"Total classes found: {len(self.classes)}")
        print(f"Classes with inheritance: {len([c for c in self.class_bases if self.class_bases[c]])}")
        print(f"Virtual methods with overrides: {len(self.virtual_method_overrides)}")

        # Find root functions (functions that are never called)
        root_functions = set(self.functions.keys()) - set(self.reverse_call_edges.keys())
        print(f"Root functions (never called): {len(root_functions)}")

        # Find leaf functions (functions that don't call anything)
        leaf_functions = set(self.functions.keys()) - set(self.call_edges.keys())
        print(f"Leaf functions (call nothing): {len(leaf_functions)}")

        return {
            'functions': self.functions,
            'call_edges': self.call_edges,
            'reverse_call_edges': self.reverse_call_edges,
            'file_to_functions': self.file_to_functions
        }

    def discover_template_dependent_calls(self):
        """Discover call edges inside template bodies that clang cannot
        resolve because the callee depends on a template parameter.

        Inside a template body like ``template<class T> void f() {
        AudioFile<T> w; w.save(...); }``, the call ``w.save()`` appears
        as a ``CALL_EXPR`` with ``type='<dependent type>'`` and
        ``referenced=None``.  The normal call-edge extraction skips it.

        This method re-parses each TU and, for every such unresolved
        call inside a template function, extracts the member name from
        the token stream (``obj . member``) and matches it against
        known class/struct member functions in the call graph.

        Gated by ``"discover_template": "True"`` in metadata; call
        after ``build_call_graph()``.
        """
        from collections import defaultdict as _dd

        def _get(info, key, default=''):
            """Access FunctionInfo attr or dict key."""
            if isinstance(info, dict):
                return info.get(key, default)
            return getattr(info, key, default)

        import re as _re

        def _class_from_usr(usr: str) -> str:
            """Extract the innermost class/struct name from a member
            function USR.  Examples:
              c:@S@Foo@F@bar#               -> 'Foo'
              c:@N@kgr@S@container@F@...    -> 'container'
              c:@ST>1#T@AudioFile@F@save#   -> 'AudioFile'
            """
            # Last @S@NAME or @ST>...@NAME before @F@ / @FT@
            m = _re.search(r'@S@([A-Za-z_]\w*)(?=@F|@FT|@ST|$)', usr)
            if m:
                return m.group(1)
            m = _re.search(r'@ST>\d+#[^@]*@([A-Za-z_]\w*)@F', usr)
            if m:
                return m.group(1)
            return ''

        def _class_from_type_spelling(tspell: str) -> str:
            """Pull the innermost type name from a cursor.type.spelling,
            stripping namespaces, template args, ptr/ref qualifiers."""
            if not tspell:
                return ''
            t = tspell
            # Strip reference/pointer/cv qualifiers
            t = _re.sub(r'\b(const|volatile)\b', '', t)
            t = t.replace('&', '').replace('*', '').strip()
            # Drop template args
            t = _re.sub(r'<.*', '', t)
            # Drop namespace qualifiers — keep the last segment
            if '::' in t:
                t = t.rsplit('::', 1)[-1]
            return t.strip()

        # Build lookup: member_name -> [(class_name, usr), ...] for
        # class/struct member functions (both template and non-template).
        member_lookup = _dd(list)
        for usr, info in self.functions.items():
            if '@S@' not in usr and '@ST@' not in usr:
                continue
            name = _get(info, 'name')
            if not name:
                continue
            cls = _class_from_usr(usr)
            member_lookup[name].append((cls, usr))

        if not member_lookup:
            print("  [discover_template] No class member functions")
            return

        # Identify template functions (USR contains @FT@).
        template_usrs = {
            usr for usr in self.functions if '@FT@' in usr
        }
        if not template_usrs:
            print("  [discover_template] No function templates found")
            return

        print(f"  [discover_template] Scanning {len(template_usrs)} "
              f"template function(s) for dependent calls...")

        def _pick_matches(member_name, obj_type_cls):
            """Return list of USRs for `member_name` whose class
            matches `obj_type_cls` (or all if obj class unknown and
            the name is globally unique)."""
            candidates = member_lookup.get(member_name, [])
            if not candidates:
                return []
            if obj_type_cls:
                precise = [u for (c, u) in candidates
                           if c == obj_type_cls]
                if precise:
                    return precise
            # No class info or no class match: only return when the
            # name is globally unique across classes, so we don't
            # flood common method names (find, size, begin, ...)
            # with false-positive edges.
            if len(candidates) == 1:
                return [candidates[0][1]]
            return []

        added = 0
        for entry in self.commands:
            directory = entry["directory"]
            file_path = entry["file"]
            command = entry.get("command", "")
            args = shlex.split(command)[1:] if command else []
            if "-o" in args:
                idx = args.index("-o")
                del args[idx:idx + 2]
            full_path = (file_path if os.path.isabs(file_path)
                         else os.path.join(directory, file_path))
            full_path = os.path.abspath(full_path)
            if file_path in args:
                args.remove(file_path)
            if full_path in args:
                args.remove(full_path)

            try:
                index = cindex.Index.create()
                tu = index.parse(
                    full_path, args=args,
                    options=cindex.TranslationUnit
                    .PARSE_DETAILED_PROCESSING_RECORD)
            except Exception:
                continue

            def _extract_obj_class(member_ref_cursor):
                """Given a MEMBER_REF_EXPR cursor, return the simple
                class name of the object the member is called on,
                or '' when the call is implicit-this (no typed object
                precedes the member name in the AST).

                Skip cursor kinds that represent the callee/member
                name itself (OVERLOADED_DECL_REF, DECL_REF_EXPR for
                overloaded lookup) or template-argument refs
                (TYPE_REF to a template parameter). Only accept an
                explicit typed object (DECL_REF_EXPR to a variable,
                CALL_EXPR, MEMBER_REF_EXPR, UNEXPOSED_EXPR, etc.).
                """
                for sub in member_ref_cursor.get_children():
                    k = sub.kind
                    # Skip the member-name / overload lookup itself.
                    if k in (cindex.CursorKind.OVERLOADED_DECL_REF,
                             cindex.CursorKind.NAMESPACE_REF):
                        continue
                    # TYPE_REF to a template type parameter is a
                    # template-argument, not the receiver.
                    if k == cindex.CursorKind.TYPE_REF:
                        try:
                            ref = sub.referenced
                            if ref and ref.kind == \
                                    cindex.CursorKind\
                                    .TEMPLATE_TYPE_PARAMETER:
                                continue
                        except (ValueError, AttributeError):
                            continue
                        # A real class TYPE_REF still isn't the
                        # receiver in `obj.member` — it appears for
                        # template args. Skip.
                        continue
                    try:
                        tspell = sub.type.spelling if sub.type else ''
                    except (ValueError, AttributeError):
                        tspell = ''
                    if not tspell or tspell == '<dependent type>':
                        # Dependent receiver — can't narrow by type.
                        continue
                    cls = _class_from_type_spelling(tspell)
                    if cls:
                        return cls
                return ''

            def _walk(cursor, cur_func_usr):
                nonlocal added
                try:
                    if cursor.location.file:
                        if is_system_file(
                                os.path.abspath(
                                    cursor.location.file.name)):
                            return
                except (ValueError, AttributeError):
                    pass

                kind = cursor.kind
                if kind in (cindex.CursorKind.FUNCTION_DECL,
                            cindex.CursorKind.CXX_METHOD,
                            cindex.CursorKind.CONSTRUCTOR,
                            cindex.CursorKind.DESTRUCTOR,
                            cindex.CursorKind.FUNCTION_TEMPLATE):
                    cur_func_usr = cursor.get_usr()

                # Detect dependent member calls: CALL_EXPR with
                # type='<dependent type>' and referenced=None, inside
                # a template function.
                if (kind == cindex.CursorKind.CALL_EXPR
                        and cur_func_usr
                        and cur_func_usr in template_usrs):
                    ref = cursor.referenced
                    type_sp = (cursor.type.spelling
                               if cursor.type else '')
                    if ref is None and '<dependent' in type_sp:
                        for child in cursor.get_children():
                            if child.kind != \
                                    cindex.CursorKind.MEMBER_REF_EXPR:
                                continue
                            # Primary: cursor.referenced.spelling is
                            # set when clang already knows the member
                            # name (implicit this->member, unqualified
                            # lookup against overloaded decl refs).
                            # This covers free-function-style calls
                            # like `definition<T>()` that have no '.'
                            # or '->' in source.
                            member_name = None
                            try:
                                cref = child.referenced
                                if cref and cref.spelling:
                                    member_name = cref.spelling
                            except (ValueError, AttributeError):
                                pass
                            # Fallback: explicit obj.member / obj->member
                            # in source — parse the token stream.
                            if not member_name:
                                tokens = list(child.get_tokens())
                                tok_strs = [t.spelling for t in tokens]
                                for ti, ts in enumerate(tok_strs):
                                    if ts in ('.', '->') \
                                            and ti + 1 < len(tok_strs):
                                        member_name = tok_strs[ti + 1]
                                        break
                            if not member_name:
                                break
                            obj_cls = _extract_obj_class(child)
                            # For implicit this->member calls the
                            # MEMBER_REF_EXPR child is an OVERLOADED_
                            # DECL_REF (not a typed object), so
                            # _extract_obj_class returns ''. Use the
                            # enclosing class of the caller template
                            # as the object class in that case.
                            if not obj_cls and cur_func_usr:
                                obj_cls = _class_from_usr(cur_func_usr)
                            for callee_usr in _pick_matches(
                                    member_name, obj_cls):
                                existing = self.call_edges.get(
                                    cur_func_usr, set())
                                if callee_usr not in existing:
                                    self.call_edges[
                                        cur_func_usr].add(callee_usr)
                                    self.reverse_call_edges[
                                        callee_usr].add(cur_func_usr)
                                    added += 1
                            break  # only need the first child

                for child in cursor.get_children():
                    _walk(child, cur_func_usr)

            _walk(tu.cursor, None)

        if added:
            print(f"  [discover_template] Added {added} dependent "
                  f"call edge(s)")
        else:
            print("  [discover_template] No new edges discovered")

    def export_to_json(self, output_path):
        """
        Export the call graph to JSON format.

        Args:
            output_path: Path to output JSON file
        """
        print(f"\nExporting call graph to: {output_path}")

        # Convert to JSON-serializable format
        export_data = {
            'functions': {
                usr: (func_info if isinstance(func_info, dict) else func_info.to_dict())
                for usr, func_info in self.functions.items()
            },
            'call_edges': {
                caller_usr: list(callee_usrs)
                for caller_usr, callee_usrs in self.call_edges.items()
            },
            'reverse_call_edges': {
                callee_usr: list(caller_usrs)
                for callee_usr, caller_usrs in self.reverse_call_edges.items()
            },
            'file_to_functions': {
                file_path: list(func_usrs)
                for file_path, func_usrs in self.file_to_functions.items()
            },
            'classes': self.classes,
            'class_bases': {
                class_usr: list(base_usrs)
                for class_usr, base_usrs in self.class_bases.items()
            },
            'class_derived': {
                class_usr: list(derived_usrs)
                for class_usr, derived_usrs in self.class_derived.items()
            },
            'virtual_method_overrides': {
                method_usr: list(override_usrs)
                for method_usr, override_usrs in self.virtual_method_overrides.items()
            },
            'method_to_class': self.method_to_class
        }

        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)

        print(f"✓ Call graph exported successfully")

    def get_callers(self, function_usr):
        """
        Get all functions that call the given function.

        Args:
            function_usr: USR of the function

        Returns:
            set: Set of caller USRs
        """
        return self.reverse_call_edges.get(function_usr, set())

    def get_callees(self, function_usr):
        """
        Get all functions called by the given function.

        Args:
            function_usr: USR of the function

        Returns:
            set: Set of callee USRs
        """
        return self.call_edges.get(function_usr, set())

    def find_function_by_name(self, function_name):
        """
        Find all functions matching the given name.

        Args:
            function_name: Name of the function to find

        Returns:
            list: List of (usr, FunctionInfo) tuples
        """
        matches = []
        for usr, func_info in self.functions.items():
            if func_info.name == function_name:
                matches.append((usr, func_info))
        return matches

    def find_function_at_location(self, file_path, line):
        """
        Find the function at a specific file:line location.

        Args:
            file_path: Path to the source file
            line: Line number

        Returns:
            tuple: (usr, FunctionInfo) or (None, None) if not found
        """
        file_path_abs = os.path.abspath(file_path)
        func_usrs = self.file_to_functions.get(file_path_abs, set())

        for usr in func_usrs:
            func_info = self.functions[usr]
            # Check if the line is within the function (simple heuristic)
            if func_info.line == line:
                return (usr, func_info)

        return (None, None)

    def get_call_chain_to_root(self, function_usr, max_depth=10):
        """
        Find all call chains from this function back to root functions.
        Useful for finding what calls lead to a specific function.

        Args:
            function_usr: USR of the target function
            max_depth: Maximum depth to search

        Returns:
            list: List of call chains, where each chain is a list of USRs
        """
        chains = []

        def dfs(current_usr, path, depth):
            if depth > max_depth:
                return

            # Get callers of current function
            callers = self.get_callers(current_usr)

            if not callers:
                # This is a root function, add the complete path
                chains.append(list(reversed(path)))
            else:
                # Continue searching through callers
                for caller_usr in callers:
                    if caller_usr not in path:  # Avoid cycles
                        dfs(caller_usr, path + [caller_usr], depth + 1)

        dfs(function_usr, [function_usr], 0)
        return chains

    def print_function_info(self, function_usr, show_calls=True, show_callers=True):
        """
        Print detailed information about a function.

        Args:
            function_usr: USR of the function
            show_calls: Whether to show functions this one calls
            show_callers: Whether to show functions that call this one
        """
        if function_usr not in self.functions:
            print(f"Function with USR {function_usr} not found")
            return

        func_info = self.functions[function_usr]

        print(f"\nFunction: {func_info.name}")
        print(f"  Qualified name: {func_info.qualified_name}")
        print(f"  Location: {func_info.file}:{func_info.line}")
        print(f"  Return type: {func_info.return_type}")
        print(f"  Parameters: {', '.join(p['type'] + ' ' + p['name'] for p in func_info.params)}")

        if show_callers:
            callers = self.get_callers(function_usr)
            print(f"  Called by ({len(callers)}):")
            for caller_usr in list(callers)[:10]:  # Show first 10
                caller_info = self.functions.get(caller_usr)
                if caller_info:
                    print(f"    - {caller_info.name} at {caller_info.file}:{caller_info.line}")
            if len(callers) > 10:
                print(f"    ... and {len(callers) - 10} more")

        if show_calls:
            callees = self.get_callees(function_usr)
            print(f"  Calls ({len(callees)}):")
            for callee_usr in list(callees)[:10]:  # Show first 10
                callee_info = self.functions.get(callee_usr)
                if callee_info:
                    print(f"    - {callee_info.name} at {callee_info.file}:{callee_info.line}")
            if len(callees) > 10:
                print(f"    ... and {len(callees) - 10} more")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python CallGraphBuilder.py <path_to_compile_commands.json> [output.json]")
        sys.exit(1)

    compile_commands_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "call_graph.json"

    # Build call graph
    builder = CallGraphBuilder(compile_commands_path)
    call_graph = builder.build_call_graph()

    # Export to JSON
    builder.export_to_json(output_path)

    print("\n" + "="*60)
    print("DONE")
    print("="*60)
