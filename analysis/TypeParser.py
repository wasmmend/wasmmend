#!/usr/bin/env python3
"""
TypeParser: Extracts user-defined type definitions from a C/C++ project,
builds a dependency graph among them, detects cyclic dependencies via
Tarjan's SCC algorithm, and exports results to JSON.

Usage:
    python TypeParser.py <compile_commands.json> [output.json]
"""
import json
import re
import shlex
import os
import sys
from clang import cindex
from collections import defaultdict
import multiprocessing as _mp
from tqdm import tqdm

# Python 3.14 defaults to 'forkserver' which fails when the main module
# isn't a real file (e.g. invoked via -c or stdin).  Use 'fork' explicitly
# to avoid this.
_mp_ctx = _mp.get_context('fork')
Pool = _mp_ctx.Pool

# Use config.py for portable libclang auto-detection (calls setup_libclang()
# at import time -- see src/config.py:69).
from repair.config import setup_libclang  # noqa: F401 — side-effect import


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def is_system_file(file_path):
    """Check if *file_path* belongs to a system include / library directory."""
    if not file_path:
        return True
    system_prefixes = [
        "/usr/include/",
        "/usr/lib/",
        "/usr/local/include/",
        "/usr/local/lib/",
        "/opt/",
        "/lib/",
    ]
    emsdk_path = os.environ.get("EMSDK")
    if emsdk_path:
        system_prefixes.append(emsdk_path)
    home_emsdk = os.path.expanduser("~/emsdk")
    if os.path.exists(home_emsdk):
        system_prefixes.append(home_emsdk)
    abs_path = os.path.abspath(file_path)
    return any(abs_path.startswith(p) for p in system_prefixes)


def _is_system_header(file_path):
    """Lightweight cursor-level check for system headers."""
    if file_path is None:
        return True
    return (
        file_path.startswith("/usr/")
        or "/lib/clang" in file_path
        or "/include/c++" in file_path
        or "/lib/gcc/" in file_path
        or "/../lib/gcc" in file_path
    )


# ---------------------------------------------------------------------------
# Multiprocessing worker (module-level so it is picklable)
# ---------------------------------------------------------------------------

def _parse_one_file_worker(args):
    """Process a single compile-command entry in a worker process.

    Returns (types, function_info, function_type_usage, file_type_usage,
    warnings) where each dict mirrors the corresponding TypeParser attribute
    for the *single* TU that was parsed.
    """
    compile_commands_path, entry = args

    # Create a throwaway TypeParser — its methods do the heavy lifting.
    tp = TypeParser(compile_commands_path)

    directory = entry["directory"]
    file_path = entry["file"]
    command = entry.get("command", "")

    cli_args = shlex.split(command)[1:] if command else []
    if "-o" in cli_args:
        o_idx = cli_args.index("-o")
        cli_args[o_idx: o_idx + 2] = []
    full_path = (
        file_path if os.path.isabs(file_path)
        else os.path.join(directory, file_path)
    )
    full_path = os.path.abspath(full_path)
    if file_path in cli_args:
        cli_args.remove(file_path)
    if full_path in cli_args:
        cli_args.remove(full_path)

    warnings = []
    try:
        index = cindex.Index.create()
        tu = index.parse(
            full_path,
            args=cli_args,
            options=cindex.TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD,
        )
        if tu is None:
            warnings.append(
                f"Warning: failed to parse {full_path} (tu is None)")
            return (full_path, ({}, {}, {}, {}, warnings))
        for diag in tu.diagnostics:
            if diag.severity >= cindex.Diagnostic.Error:
                warnings.append(
                    f"Clang error in {os.path.basename(full_path)}: {diag}")
        tp._current_source_file = full_path
        tp._walk_tu(tu.cursor)
    except Exception as e:
        warnings.append(f"Error parsing {full_path}: {e}")
        return (full_path, ({}, {}, {}, {}, warnings))

    return (
        full_path,
        (
            tp.types,
            tp.function_info,
            dict(tp.function_type_usage),
            dict(tp.file_type_usage),
            warnings,
        ),
    )


# ---------------------------------------------------------------------------
# TypeInfo — data holder for a single user-defined type
# ---------------------------------------------------------------------------

class TypeInfo:
    """Stores everything we know about one user-defined type."""

    _KIND_MAP = {
        cindex.CursorKind.STRUCT_DECL: "struct",
        cindex.CursorKind.CLASS_DECL: "class",
        cindex.CursorKind.ENUM_DECL: "enum",
        cindex.CursorKind.UNION_DECL: "union",
        cindex.CursorKind.TYPEDEF_DECL: "typedef",
        cindex.CursorKind.TYPE_ALIAS_DECL: "type_alias",
        cindex.CursorKind.TYPE_ALIAS_TEMPLATE_DECL: "type_alias_template",
        cindex.CursorKind.CLASS_TEMPLATE: "class_template",
        cindex.CursorKind.CLASS_TEMPLATE_PARTIAL_SPECIALIZATION: "class_template_partial_spec",
    }

    def __init__(self, usr, cursor=None):
        self.usr = usr
        self.spelling = ""
        self.qualified_name = ""
        self.kind = ""
        self.file_path = None       # absolute path
        self.start_line = None
        self.end_line = None
        self.body = None             # source text (filled later)
        self.is_definition = False
        self.depends_on = set()      # set of USRs
        self.scc_groups = []         # list of lists of qualified names
        self._reconstructed_body = None  # AST-based body (filled Phase 1)
        self.macro_generated = False     # True if body was from a macro
        self.member_variables = []       # list of member-variable dicts
        self.template_parameters = []    # list of template type-param names
        self.has_ostream_operator = False  # True if operator<< already exists

        if cursor is not None:
            self._populate_from_cursor(cursor)

    # ---- internal helpers ------------------------------------------------

    def _populate_from_cursor(self, cursor):
        self.spelling = cursor.spelling or ""
        self.qualified_name = self._build_qualified_name(cursor)
        self.kind = self._KIND_MAP.get(cursor.kind, str(cursor.kind))
        self.is_definition = cursor.is_definition()
        try:
            if cursor.location and cursor.location.file:
                self.file_path = os.path.abspath(cursor.location.file.name)
        except (ValueError, AttributeError):
            pass
        try:
            if cursor.extent:
                self.start_line = cursor.extent.start.line
                self.end_line = cursor.extent.end.line
        except (ValueError, AttributeError):
            pass

    @staticmethod
    def _build_qualified_name(cursor):
        """Walk the semantic_parent chain to produce a fully-qualified name."""
        parts = []
        c = cursor
        while c is not None and c.kind != cindex.CursorKind.TRANSLATION_UNIT:
            name = c.spelling
            if name:
                parts.append(name)
            else:
                # Anonymous type — use a synthetic tag
                try:
                    loc = c.location
                    if loc and loc.file:
                        fname = os.path.basename(loc.file.name)
                        parts.append(f"(anon@{fname}:{loc.line})")
                    else:
                        parts.append("(anon)")
                except (ValueError, AttributeError):
                    parts.append("(anon)")
            c = c.semantic_parent
        return "::".join(reversed(parts)) if parts else "(unknown)"


# ---------------------------------------------------------------------------
# TypeParser — main class
# ---------------------------------------------------------------------------

class TypeParser:
    """Parse a C/C++ project for user-defined types, dependencies, and cycles."""

    # CursorKinds that represent type declarations we care about.
    TYPE_DECL_KINDS = frozenset([
        cindex.CursorKind.STRUCT_DECL,
        cindex.CursorKind.CLASS_DECL,
        cindex.CursorKind.ENUM_DECL,
        cindex.CursorKind.UNION_DECL,
        cindex.CursorKind.TYPEDEF_DECL,
        cindex.CursorKind.TYPE_ALIAS_DECL,
        cindex.CursorKind.TYPE_ALIAS_TEMPLATE_DECL,
        cindex.CursorKind.CLASS_TEMPLATE,
        cindex.CursorKind.CLASS_TEMPLATE_PARTIAL_SPECIALIZATION,
    ])

    # CursorKinds for function definitions whose bodies we walk for type refs.
    FUNC_KINDS = frozenset([
        cindex.CursorKind.FUNCTION_DECL,
        cindex.CursorKind.CXX_METHOD,
        cindex.CursorKind.CONSTRUCTOR,
        cindex.CursorKind.DESTRUCTOR,
        cindex.CursorKind.FUNCTION_TEMPLATE,
    ])

    # Cast expression kinds whose target type we inspect.
    CAST_KINDS = frozenset([
        cindex.CursorKind.CXX_STATIC_CAST_EXPR,
        cindex.CursorKind.CXX_DYNAMIC_CAST_EXPR,
        cindex.CursorKind.CXX_REINTERPRET_CAST_EXPR,
        cindex.CursorKind.CXX_CONST_CAST_EXPR,
        cindex.CursorKind.CSTYLE_CAST_EXPR,
        cindex.CursorKind.CXX_FUNCTIONAL_CAST_EXPR,
    ])

    # Maximum recursion depth for _resolve_type to avoid runaway template nesting.
    _MAX_RESOLVE_DEPTH = 20

    # Regex for detecting genuine type-definition keywords in source text.
    _TYPE_KEYWORD_RE = re.compile(
        r'\b(struct|class|enum|union|typedef|using|template)\b'
    )

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, compile_commands_path):
        self.compile_commands_path = os.path.abspath(compile_commands_path)
        self.commands = []                         # filtered compile entries
        self.types = {}                            # USR -> TypeInfo
        self.function_type_usage = defaultdict(set)  # func USR -> set of type USRs
        self.function_info = {}                    # func USR -> {name, file, line, displayname}
        self.file_type_usage = defaultdict(set)    # source file abs path -> set of type USRs
        self._current_source_file = None           # set per-TU during parsing
        self._file_cache = {}                      # abs path -> list of lines

    # ------------------------------------------------------------------
    # Compile-commands loading
    # ------------------------------------------------------------------

    def load_compile_commands(self, filter_system_files=True):
        """Load *compile_commands.json* and optionally filter out system files."""
        with open(self.compile_commands_path, "r") as f:
            all_commands = json.load(f)

        if filter_system_files:
            self.commands = [
                entry for entry in all_commands
                if not is_system_file(
                    entry["file"]
                    if os.path.isabs(entry["file"])
                    else os.path.join(entry["directory"], entry["file"])
                )
            ]
        else:
            self.commands = list(all_commands)

        print(
            f"TypeParser: loaded {len(all_commands)} compile commands, "
            f"processing {len(self.commands)} (after filtering)"
        )

    # ------------------------------------------------------------------
    # Merging results from parallel workers
    # ------------------------------------------------------------------

    def _merge_worker_result(self, w_types, w_func_info, w_func_usage,
                             w_file_usage):
        """Merge one worker's parse results into *self*."""
        for usr, ti in w_types.items():
            if usr not in self.types:
                self.types[usr] = ti
            elif ti.is_definition and not self.types[usr].is_definition:
                # Upgrade forward decl → definition, keep accumulated deps.
                old_deps = self.types[usr].depends_on
                self.types[usr] = ti
                self.types[usr].depends_on.update(old_deps)
            else:
                # Same USR seen again — just accumulate deps.
                self.types[usr].depends_on.update(ti.depends_on)

        for func_usr, info in w_func_info.items():
            if func_usr not in self.function_info:
                self.function_info[func_usr] = info

        for func_usr, type_usrs in w_func_usage.items():
            self.function_type_usage[func_usr].update(type_usrs)

        for fpath, type_usrs in w_file_usage.items():
            self.file_type_usage[fpath].update(type_usrs)

    # ------------------------------------------------------------------
    # Phase 1 — parse every TU
    # ------------------------------------------------------------------

    def parse_all_files(self, parallel=True, max_workers=None):
        """Parse all translation units and collect type info + dependencies.

        Args:
            parallel: Use multiprocessing to parse files concurrently.
            max_workers: Maximum worker processes (default: cpu_count).
        """
        self.load_compile_commands()
        total = len(self.commands)

        if parallel and total > 1:
            if max_workers is None:
                max_workers = min(os.cpu_count() or 4, total)

            worker_args = [
                (self.compile_commands_path, entry)
                for entry in self.commands
            ]

            # Collect results with the source file they came from,
            # then merge in a deterministic order (sorted by file
            # path). The merge is first-wins on USRs, so without a
            # stable ordering the plan differs run-to-run when
            # imap_unordered returns worker results in different
            # orders
            all_warnings = []
            collected = []  # list of (tu_file_path, result_tuple)
            with Pool(max_workers) as pool:
                for (tu_file, result) in tqdm(
                    pool.imap_unordered(_parse_one_file_worker, worker_args),
                    total=total,
                    desc="  Parsing types",
                    unit="file",
                ):
                    collected.append((tu_file, result))

            collected.sort(key=lambda x: x[0] or "")
            for _tu, result in collected:
                w_types, w_func_info, w_func_usage, w_file_usage, warnings = result
                self._merge_worker_result(
                    w_types, w_func_info, w_func_usage, w_file_usage)
                all_warnings.extend(warnings)

            # Print any warnings/errors collected from workers.
            for w in all_warnings:
                print(f"  {w}")
        else:
            # Sequential fallback (small projects or parallel=False).
            for idx, entry in enumerate(self.commands, 1):
                directory = entry["directory"]
                file_path = entry["file"]
                command = entry.get("command", "")

                args = shlex.split(command)[1:] if command else []
                if "-o" in args:
                    o_idx = args.index("-o")
                    args[o_idx: o_idx + 2] = []
                full_path = (
                    file_path
                    if os.path.isabs(file_path)
                    else os.path.join(directory, file_path)
                )
                full_path = os.path.abspath(full_path)
                if file_path in args:
                    args.remove(file_path)
                if full_path in args:
                    args.remove(full_path)

                print(
                    f"  [{idx}/{total}] Parsing types in: "
                    f"{os.path.basename(full_path)}",
                    flush=True,
                )

                try:
                    index = cindex.Index.create()
                    tu = index.parse(
                        full_path,
                        args=args,
                        options=(
                            cindex.TranslationUnit
                            .PARSE_DETAILED_PROCESSING_RECORD),
                    )
                    if tu is None:
                        print(f"    Warning: failed to parse "
                              f"{full_path} (tu is None)")
                        continue
                    for diag in tu.diagnostics:
                        if diag.severity >= cindex.Diagnostic.Error:
                            print(f"    Clang error: {diag}")
                    self._current_source_file = full_path
                    self._walk_tu(tu.cursor)
                except Exception as e:
                    print(f"    Error parsing {full_path}: {e}")

        # Phase 2 — read source text for every type definition we found.
        self._extract_all_bodies()
        # Phase 2.5 — detect macro-generated types and fix their bodies.
        self._detect_and_fix_macro_types()

    # ------------------------------------------------------------------
    # AST walking
    # ------------------------------------------------------------------

    def _walk_tu(self, cursor):
        """Recursively walk one TU, registering types and collecting usage."""
        # Filter out system headers early.
        try:
            loc_file = cursor.location.file
            if loc_file and _is_system_header(loc_file.name):
                return
        except (ValueError, AttributeError):
            pass

        try:
            kind = cursor.kind
        except ValueError:
            return

        # --- Register type declarations ---
        if kind in self.TYPE_DECL_KINDS:
            self._register_type(cursor)
            usr = cursor.get_usr()
            if usr and self._current_source_file:
                self.file_type_usage[self._current_source_file].add(usr)
            # If this is a definition, collect intra-type dependencies right now
            # (we still have the cursor alive).
            if cursor.is_definition():
                if usr and usr in self.types:
                    dep_usrs = set()
                    self._collect_type_dependencies(cursor, dep_usrs)
                    dep_usrs.discard(usr)  # remove self-dependency
                    self.types[usr].depends_on.update(dep_usrs)

        # --- For function definitions, collect types used ---
        if kind in self.FUNC_KINDS and cursor.is_definition():
            func_usr = cursor.get_usr()
            if func_usr:
                # Record function metadata for later lookup.
                if func_usr not in self.function_info:
                    try:
                        func_file = None
                        func_line = None
                        if cursor.location and cursor.location.file:
                            func_file = os.path.abspath(cursor.location.file.name)
                            func_line = cursor.location.line
                        self.function_info[func_usr] = {
                            "name": cursor.spelling or "",
                            "displayname": cursor.displayname or "",
                            "file": func_file,
                            "line": func_line,
                        }
                    except (ValueError, AttributeError):
                        pass

                self._collect_types_from_function(cursor, func_usr)
                # Also record these types under the current source file.
                if self._current_source_file:
                    self.file_type_usage[self._current_source_file].update(
                        self.function_type_usage[func_usr]
                    )

        # --- Recurse into children ---
        for child in cursor.get_children():
            self._walk_tu(child)

    # ------------------------------------------------------------------
    # Type registration (USR-based dedup, prefer definitions)
    # ------------------------------------------------------------------

    def _register_type(self, cursor):
        """Register a type.  Definitions supersede forward declarations."""
        usr = cursor.get_usr()
        if not usr:
            return

        _RECORD_LIKE = (
            cindex.CursorKind.STRUCT_DECL,
            cindex.CursorKind.CLASS_DECL,
            cindex.CursorKind.UNION_DECL,
            cindex.CursorKind.CLASS_TEMPLATE,
            cindex.CursorKind.CLASS_TEMPLATE_PARTIAL_SPECIALIZATION,
        )
        _TEMPLATE_LIKE = (
            cindex.CursorKind.CLASS_TEMPLATE,
            cindex.CursorKind.CLASS_TEMPLATE_PARTIAL_SPECIALIZATION,
        )

        if usr not in self.types:
            self.types[usr] = TypeInfo(usr, cursor)
            if cursor.is_definition():
                self.types[usr]._reconstructed_body = (
                    self._reconstruct_type_body(cursor)
                )
                if cursor.kind in _RECORD_LIKE:
                    self.types[usr].member_variables = (
                        self._extract_member_variables(cursor)
                    )
                    self.types[usr].has_ostream_operator = (
                        self._has_existing_ostream_operator(cursor)
                    )
                if cursor.kind in _TEMPLATE_LIKE:
                    self.types[usr].template_parameters = (
                        self._extract_template_parameters(cursor)
                    )
        elif cursor.is_definition() and not self.types[usr].is_definition:
            # Upgrade: replace forward-decl entry with the real definition,
            # but preserve any dependency info already collected.
            old_deps = self.types[usr].depends_on
            self.types[usr] = TypeInfo(usr, cursor)
            self.types[usr].depends_on = old_deps
            self.types[usr]._reconstructed_body = (
                self._reconstruct_type_body(cursor)
            )
            if cursor.kind in _RECORD_LIKE:
                self.types[usr].member_variables = (
                    self._extract_member_variables(cursor)
                )
                self.types[usr].has_ostream_operator = (
                    self._has_existing_ostream_operator(cursor)
                )
            if cursor.kind in _TEMPLATE_LIKE:
                self.types[usr].template_parameters = (
                    self._extract_template_parameters(cursor)
                )

    # ------------------------------------------------------------------
    # Collecting types referenced by a function (signature + body)
    # ------------------------------------------------------------------

    def _collect_types_from_function(self, func_cursor, func_usr):
        """Gather every user-defined type referenced by *func_cursor*."""
        found_usrs = set()

        # For constructors / destructors, the enclosing class type itself
        # is needed — instrumentation prints the constructed/destructed
        # object, so we need operator<< for the class.
        if func_cursor.kind in (cindex.CursorKind.CONSTRUCTOR,
                                cindex.CursorKind.DESTRUCTOR):
            parent = func_cursor.semantic_parent
            if (parent and
                    parent.kind in self.TYPE_DECL_KINDS):
                parent_usr = parent.get_usr()
                if parent_usr:
                    found_usrs.add(parent_usr)
                    # Also register the type if not already known.
                    if parent_usr not in self.types:
                        self._register_type(parent)

        # Return type.
        try:
            self._resolve_type(func_cursor.result_type, found_usrs)
        except (ValueError, AttributeError):
            pass

        # Parameters.
        try:
            for arg in func_cursor.get_arguments():
                self._resolve_type(arg.type, found_usrs)
        except (ValueError, AttributeError):
            pass

        # Walk the body recursively.
        self._walk_function_body(func_cursor, found_usrs)

        self.function_type_usage[func_usr].update(found_usrs)

    def _walk_function_body(self, cursor, found_usrs):
        """Recursively walk a function body collecting type references."""
        try:
            kind = cursor.kind
        except ValueError:
            return

        if kind == cindex.CursorKind.TYPE_REF:
            self._handle_type_or_template_ref(cursor, found_usrs)

        elif kind == cindex.CursorKind.TEMPLATE_REF:
            self._handle_type_or_template_ref(cursor, found_usrs)

        elif kind in (
            cindex.CursorKind.VAR_DECL,
            cindex.CursorKind.PARM_DECL,
            cindex.CursorKind.FIELD_DECL,
        ):
            try:
                self._resolve_type(cursor.type, found_usrs)
            except (ValueError, AttributeError):
                pass

        elif kind in self.CAST_KINDS:
            try:
                self._resolve_type(cursor.type, found_usrs)
            except (ValueError, AttributeError):
                pass

        elif kind == cindex.CursorKind.CXX_NEW_EXPR:
            try:
                self._resolve_type(cursor.type, found_usrs)
            except (ValueError, AttributeError):
                pass

        elif kind == cindex.CursorKind.MEMBER_REF_EXPR:
            try:
                self._resolve_type(cursor.type, found_usrs)
            except (ValueError, AttributeError):
                pass

        elif kind == cindex.CursorKind.CXX_BASE_SPECIFIER:
            try:
                self._resolve_type(cursor.type, found_usrs)
            except (ValueError, AttributeError):
                pass

        for child in cursor.get_children():
            self._walk_function_body(child, found_usrs)

    def _handle_type_or_template_ref(self, cursor, found_usrs):
        """Handle TYPE_REF / TEMPLATE_REF — register + record the referenced type."""
        try:
            ref = cursor.referenced
            if ref is None:
                return
            if ref.kind in self.TYPE_DECL_KINDS:
                loc_file = ref.location.file
                if loc_file and not _is_system_header(loc_file.name):
                    usr = ref.get_usr()
                    if usr:
                        found_usrs.add(usr)
                        self._register_type(ref)
        except (ValueError, AttributeError):
            pass

    # ------------------------------------------------------------------
    # Deep type unwrapping
    # ------------------------------------------------------------------

    def _resolve_type(self, clang_type, found_usrs, _depth=0):
        """Recursively unwrap a clang Type to find user-defined declarations."""
        if _depth > self._MAX_RESOLVE_DEPTH:
            return
        if clang_type is None or clang_type.kind == cindex.TypeKind.INVALID:
            return

        kind = clang_type.kind

        # Pointer / reference — unwrap pointee.
        if kind in (
            cindex.TypeKind.POINTER,
            cindex.TypeKind.LVALUEREFERENCE,
            cindex.TypeKind.RVALUEREFERENCE,
        ):
            self._resolve_type(clang_type.get_pointee(), found_usrs, _depth + 1)
            return

        # Arrays — unwrap element type.
        if kind in (
            cindex.TypeKind.CONSTANTARRAY,
            cindex.TypeKind.INCOMPLETEARRAY,
            cindex.TypeKind.VARIABLEARRAY,
            cindex.TypeKind.DEPENDENTSIZEDARRAY,
        ):
            try:
                self._resolve_type(clang_type.element_type, found_usrs, _depth + 1)
            except (ValueError, AttributeError):
                pass
            return

        # Template specializations — resolve each template argument.
        try:
            n_args = clang_type.get_num_template_arguments()
            if n_args > 0:
                for i in range(n_args):
                    try:
                        arg_type = clang_type.get_template_argument_type(i)
                        if arg_type.kind != cindex.TypeKind.INVALID:
                            self._resolve_type(arg_type, found_usrs, _depth + 1)
                    except (ValueError, AttributeError):
                        pass
        except (ValueError, AttributeError):
            pass

        # Get the declaration of this type.
        try:
            decl = clang_type.get_declaration()
            if decl and decl.kind != cindex.CursorKind.NO_DECL_FOUND:
                if decl.kind in self.TYPE_DECL_KINDS:
                    loc_file = decl.location.file
                    if loc_file and not _is_system_header(loc_file.name):
                        usr = decl.get_usr()
                        if usr:
                            found_usrs.add(usr)
                            self._register_type(decl)
        except (ValueError, AttributeError):
            pass

        # Also chase the canonical type if it differs (resolves typedefs).
        try:
            canonical = clang_type.get_canonical()
            if canonical.kind != kind:
                self._resolve_type(canonical, found_usrs, _depth + 1)
        except (ValueError, AttributeError):
            pass

    # ------------------------------------------------------------------
    # Intra-type dependency collection
    # ------------------------------------------------------------------

    def _collect_type_dependencies(self, type_cursor, dep_usrs):
        """Walk the children of a type *definition* to find type dependencies."""
        for child in type_cursor.get_children():
            try:
                ck = child.kind
            except ValueError:
                continue

            if ck == cindex.CursorKind.CXX_BASE_SPECIFIER:
                try:
                    self._resolve_type(child.type, dep_usrs)
                except (ValueError, AttributeError):
                    pass

            elif ck == cindex.CursorKind.FIELD_DECL:
                try:
                    self._resolve_type(child.type, dep_usrs)
                except (ValueError, AttributeError):
                    pass

            elif ck == cindex.CursorKind.TYPE_REF:
                self._handle_type_or_template_ref(child, dep_usrs)

            elif ck == cindex.CursorKind.TEMPLATE_REF:
                self._handle_type_or_template_ref(child, dep_usrs)

            elif ck == cindex.CursorKind.TYPEDEF_DECL:
                # The typedef's underlying type is a dependency of the outer type.
                try:
                    self._resolve_type(child.underlying_typedef_type, dep_usrs)
                except (ValueError, AttributeError):
                    pass

            elif ck == cindex.CursorKind.TYPE_ALIAS_DECL:
                try:
                    self._resolve_type(child.underlying_typedef_type, dep_usrs)
                except (ValueError, AttributeError):
                    pass

            elif ck == cindex.CursorKind.TYPE_ALIAS_TEMPLATE_DECL:
                # Template alias (e.g. template<...> using X = Y<...>).
                # Walk into children to find the TYPE_ALIAS_DECL child
                # and resolve its underlying type.
                for grandchild in child.get_children():
                    try:
                        gk = grandchild.kind
                    except ValueError:
                        continue
                    if gk == cindex.CursorKind.TYPE_ALIAS_DECL:
                        try:
                            self._resolve_type(
                                grandchild.underlying_typedef_type, dep_usrs)
                        except (ValueError, AttributeError):
                            pass
                    elif gk == cindex.CursorKind.TEMPLATE_REF:
                        self._handle_type_or_template_ref(grandchild, dep_usrs)

            elif ck in self.TYPE_DECL_KINDS:
                # Nested type definition — register it but don't add as dep.
                self._register_type(child)

            elif ck in self.FUNC_KINDS:
                # Method declarations inside the class — skip for dependency
                # purposes (we handle function-level usage separately).
                pass

            else:
                # Recurse deeper (e.g., access specifiers, anonymous structs).
                self._collect_type_dependencies(child, dep_usrs)

    # ------------------------------------------------------------------
    # AST-based type body reconstruction (for macro-generated types)
    # ------------------------------------------------------------------

    def _reconstruct_type_body(self, cursor):
        """Reconstruct a type definition from AST children.

        Must be called during Phase 1 while the cursor is alive.
        Returns a source-like string, or *None* if reconstruction fails.
        """
        try:
            kind = cursor.kind
        except ValueError:
            return None

        if kind in (cindex.CursorKind.STRUCT_DECL,
                    cindex.CursorKind.CLASS_DECL,
                    cindex.CursorKind.UNION_DECL):
            return self._reconstruct_record_body(cursor)
        if kind == cindex.CursorKind.ENUM_DECL:
            return self._reconstruct_enum_body(cursor)
        if kind == cindex.CursorKind.TYPEDEF_DECL:
            return self._reconstruct_typedef_body(cursor)
        if kind == cindex.CursorKind.TYPE_ALIAS_DECL:
            return self._reconstruct_type_alias_body(cursor)
        if kind == cindex.CursorKind.TYPE_ALIAS_TEMPLATE_DECL:
            return self._reconstruct_type_alias_template_body(cursor)
        if kind in (cindex.CursorKind.CLASS_TEMPLATE,
                    cindex.CursorKind.CLASS_TEMPLATE_PARTIAL_SPECIALIZATION):
            return self._reconstruct_class_template_body(cursor)
        return None

    # ---- record (struct / class / union) --------------------------------

    _RECORD_KEYWORD = {
        cindex.CursorKind.STRUCT_DECL: "struct",
        cindex.CursorKind.CLASS_DECL: "class",
        cindex.CursorKind.UNION_DECL: "union",
        cindex.CursorKind.CLASS_TEMPLATE: "class",
        cindex.CursorKind.CLASS_TEMPLATE_PARTIAL_SPECIALIZATION: "class",
    }

    def _reconstruct_record_body(self, cursor):
        """Reconstruct a struct / class / union definition."""
        keyword = self._RECORD_KEYWORD.get(cursor.kind, "struct")
        name = cursor.spelling or ""

        # -- base specifiers --
        bases = []
        for child in cursor.get_children():
            try:
                if child.kind == cindex.CursorKind.CXX_BASE_SPECIFIER:
                    access = self._access_keyword(child.access_specifier)
                    base_type = child.type.spelling if child.type else "(unknown)"
                    bases.append(f"{access} {base_type}" if access else base_type)
            except (ValueError, AttributeError):
                pass

        header = f"{keyword} {name}" if name else keyword
        if bases:
            header += " : " + ", ".join(bases)

        members = self._collect_record_members(cursor)
        if not members:
            return f"{header} {{}};"

        lines = [f"{header} {{"]
        lines.extend(f"    {m}" for m in members)
        lines.append("};")
        return "\n".join(lines)

    def _collect_record_members(self, cursor):
        """Walk *cursor*'s children and return one text line per member."""
        members = []
        for child in cursor.get_children():
            try:
                ck = child.kind
            except ValueError:
                continue

            if ck == cindex.CursorKind.CXX_BASE_SPECIFIER:
                continue  # handled in header

            if ck == cindex.CursorKind.CXX_ACCESS_SPEC_DECL:
                access = self._access_keyword(child.access_specifier)
                if access:
                    members.append(f"{access}:")
            elif ck == cindex.CursorKind.FIELD_DECL:
                members.append(self._reconstruct_field(child))
            elif ck == cindex.CursorKind.CXX_METHOD:
                members.append(self._reconstruct_method(child))
            elif ck == cindex.CursorKind.CONSTRUCTOR:
                members.append(self._reconstruct_constructor(child))
            elif ck == cindex.CursorKind.DESTRUCTOR:
                members.append(self._reconstruct_destructor(child))
            elif ck == cindex.CursorKind.CONVERSION_FUNCTION:
                members.append(self._reconstruct_method(child))
            elif ck == cindex.CursorKind.FUNCTION_TEMPLATE:
                members.append(self._reconstruct_method(child))
            elif ck == cindex.CursorKind.VAR_DECL:
                # Static data member.
                type_name = child.type.spelling if child.type else "int"
                members.append(f"static {type_name} {child.spelling};")
            elif ck in self.TYPE_DECL_KINDS:
                nested = self._reconstruct_type_body(child)
                if nested:
                    members.append(nested)
            elif ck == cindex.CursorKind.USING_DECLARATION:
                members.append(f"using {child.spelling};")
            # Silently skip everything else (attributes, friend decls, etc.)

        return members

    # ---- structured member-variable extraction --------------------------

    def _extract_member_variables(self, cursor):
        """Extract structured member variable info from a record/template cursor.

        Must be called during Phase 1 while the cursor is alive.
        Returns a list of dicts, one per FIELD_DECL or static VAR_DECL child.
        """
        members = []
        for child in cursor.get_children():
            try:
                ck = child.kind
            except ValueError:
                continue
            if ck == cindex.CursorKind.FIELD_DECL:
                members.append(self._build_member_dict(child, is_static=False))
            elif ck == cindex.CursorKind.VAR_DECL:
                members.append(self._build_member_dict(child, is_static=True))
        return members

    @staticmethod
    def _build_member_dict(child, is_static):
        """Build a member-variable info dict from a FIELD_DECL or VAR_DECL cursor."""
        ctype = child.type
        type_spelling = ctype.spelling if ctype else "(unknown)"
        name = child.spelling or "(unnamed)"

        # Access specifier
        access = ""
        try:
            _ACCESS_MAP = {
                cindex.AccessSpecifier.PUBLIC: "public",
                cindex.AccessSpecifier.PROTECTED: "protected",
                cindex.AccessSpecifier.PRIVATE: "private",
            }
            access = _ACCESS_MAP.get(child.access_specifier, "")
        except (ValueError, AttributeError):
            pass

        # Type classification flags
        is_pointer = False
        is_reference = False
        is_array = False
        is_const = False
        is_dependent = False
        pointee_type_spelling = ""
        pointee_type_usr = ""
        type_kind_name = ""
        type_usr = ""

        if ctype:
            kind = ctype.kind
            type_kind_name = kind.name if hasattr(kind, "name") else str(kind)
            is_const = ctype.is_const_qualified()

            if kind == cindex.TypeKind.POINTER:
                is_pointer = True
                try:
                    pointee = ctype.get_pointee()
                    pointee_type_spelling = pointee.spelling if pointee else ""
                    if pointee:
                        pdecl = pointee.get_declaration()
                        if pdecl and pdecl.kind != cindex.CursorKind.NO_DECL_FOUND:
                            pointee_type_usr = pdecl.get_usr() or ""
                except (ValueError, AttributeError):
                    pass
            elif kind in (cindex.TypeKind.LVALUEREFERENCE,
                          cindex.TypeKind.RVALUEREFERENCE):
                is_reference = True
            elif kind in (cindex.TypeKind.CONSTANTARRAY,
                          cindex.TypeKind.INCOMPLETEARRAY,
                          cindex.TypeKind.VARIABLEARRAY,
                          cindex.TypeKind.DEPENDENTSIZEDARRAY):
                is_array = True

            if kind in (cindex.TypeKind.DEPENDENT,
                        cindex.TypeKind.UNEXPOSED):
                is_dependent = True

            try:
                decl = ctype.get_declaration()
                if decl and decl.kind != cindex.CursorKind.NO_DECL_FOUND:
                    type_usr = decl.get_usr() or ""
            except (ValueError, AttributeError):
                pass

        # Bitfield
        is_bitfield = False
        bitfield_width = None
        try:
            if child.is_bitfield():
                is_bitfield = True
                bitfield_width = child.get_bitfield_width()
        except (ValueError, AttributeError):
            pass

        return {
            "name": name,
            "type_spelling": type_spelling,
            "access": access,
            "is_static": is_static,
            "is_const": is_const,
            "is_pointer": is_pointer,
            "is_reference": is_reference,
            "is_array": is_array,
            "pointee_type_spelling": pointee_type_spelling,
            "pointee_type_usr": pointee_type_usr,
            "is_bitfield": is_bitfield,
            "bitfield_width": bitfield_width,
            "type_kind": type_kind_name,
            "type_usr": type_usr,
            "is_dependent_type": is_dependent,
        }

    @staticmethod
    def _extract_template_parameters(cursor):
        """Return a list of template type-parameter names from a template cursor."""
        tparams = []
        for child in cursor.get_children():
            try:
                ck = child.kind
            except ValueError:
                continue
            if ck == cindex.CursorKind.TEMPLATE_TYPE_PARAMETER:
                tparams.append(child.spelling or "")
        return tparams

    @staticmethod
    def _has_existing_ostream_operator(cursor):
        """Check if a record type already defines an ``operator<<``.

        Detects:
          - Friend ``operator<<`` declarations inside the class
          - Member ``operator<<`` methods

        Must be called during Phase 1 while the cursor is alive.
        """
        for child in cursor.get_children():
            try:
                ck = child.kind
            except ValueError:
                continue

            # Friend declaration containing operator<<.
            if ck == cindex.CursorKind.FRIEND_DECL:
                for grandchild in child.get_children():
                    try:
                        gk = grandchild.kind
                    except ValueError:
                        continue
                    if gk in (cindex.CursorKind.FUNCTION_DECL,
                              cindex.CursorKind.FUNCTION_TEMPLATE):
                        if grandchild.spelling == "operator<<":
                            return True

            # Member method operator<<.
            if ck == cindex.CursorKind.CXX_METHOD:
                if child.spelling == "operator<<":
                    return True

        return False

    # ---- individual member helpers --------------------------------------

    @staticmethod
    def _reconstruct_field(child):
        type_name = child.type.spelling if child.type else "int"
        name = child.spelling or "(unnamed)"
        try:
            if child.is_bitfield():
                width = child.get_bitfield_width()
                return f"{type_name} {name} : {width};"
        except (ValueError, AttributeError):
            pass
        return f"{type_name} {name};"

    @staticmethod
    def _reconstruct_method(child):
        parts = []
        try:
            if child.is_static_method():
                parts.append("static")
        except (ValueError, AttributeError):
            pass
        try:
            if child.is_virtual_method():
                parts.append("virtual")
        except (ValueError, AttributeError):
            pass

        try:
            ret = child.result_type.spelling
        except (ValueError, AttributeError):
            ret = "void"
        parts.append(ret)

        name = child.spelling or "(unnamed)"
        params = TypeParser._format_params(child)
        parts.append(f"{name}({params})")

        try:
            if child.is_const_method():
                parts.append("const")
        except (ValueError, AttributeError):
            pass
        try:
            if child.is_pure_virtual_method():
                parts.append("= 0")
        except (ValueError, AttributeError):
            pass

        return " ".join(parts) + ";"

    @staticmethod
    def _reconstruct_constructor(child):
        name = child.spelling or "(unnamed)"
        params = TypeParser._format_params(child)
        return f"{name}({params});"

    @staticmethod
    def _reconstruct_destructor(child):
        prefix = ""
        try:
            if child.is_virtual_method():
                prefix = "virtual "
        except (ValueError, AttributeError):
            pass
        name = child.spelling or "~(unnamed)"
        return f"{prefix}{name}();"

    @staticmethod
    def _format_params(cursor):
        """Format a function's parameter list as a comma-separated string."""
        params = []
        try:
            for arg in cursor.get_arguments():
                arg_type = arg.type.spelling if arg.type else "int"
                arg_name = arg.spelling or ""
                if arg_name:
                    params.append(f"{arg_type} {arg_name}")
                else:
                    params.append(arg_type)
        except (ValueError, AttributeError):
            pass
        return ", ".join(params)

    # ---- enum -----------------------------------------------------------

    def _reconstruct_enum_body(self, cursor):
        name = cursor.spelling or ""
        scoped = ""
        try:
            if cursor.is_scoped_enum():
                scoped = "class "
        except (ValueError, AttributeError):
            pass

        constants = []
        for child in cursor.get_children():
            try:
                if child.kind == cindex.CursorKind.ENUM_CONSTANT_DECL:
                    c_name = child.spelling or "(unnamed)"
                    try:
                        c_val = child.enum_value
                        constants.append(f"{c_name} = {c_val}")
                    except (ValueError, AttributeError):
                        constants.append(c_name)
            except ValueError:
                pass

        header = f"enum {scoped}{name}".strip()
        if not constants:
            return f"{header} {{}};"
        return f"{header} {{ {', '.join(constants)} }};"

    # ---- typedef / type alias -------------------------------------------

    @staticmethod
    def _reconstruct_typedef_body(cursor):
        name = cursor.spelling or "(unnamed)"
        try:
            underlying = cursor.underlying_typedef_type.spelling
        except (ValueError, AttributeError):
            underlying = "(unknown)"
        return f"typedef {underlying} {name};"

    @staticmethod
    def _reconstruct_type_alias_body(cursor):
        name = cursor.spelling or "(unnamed)"
        try:
            underlying = cursor.underlying_typedef_type.spelling
        except (ValueError, AttributeError):
            underlying = "(unknown)"
        return f"using {name} = {underlying};"

    @staticmethod
    def _reconstruct_type_alias_template_body(cursor):
        # Extract template parameters
        tparams = []
        alias_child = None
        for child in cursor.get_children():
            try:
                ck = child.kind
            except ValueError:
                continue
            if ck == cindex.CursorKind.TEMPLATE_TYPE_PARAMETER:
                tparams.append(f"typename {child.spelling}" if child.spelling
                               else "typename")
            elif ck == cindex.CursorKind.TEMPLATE_NON_TYPE_PARAMETER:
                tparams.append(child.spelling or "auto")
            elif ck == cindex.CursorKind.TYPE_ALIAS_DECL:
                alias_child = child

        tparam_str = ", ".join(tparams) if tparams else "..."
        name = cursor.spelling or "(unnamed)"
        underlying = "(unknown)"
        if alias_child:
            try:
                underlying = alias_child.underlying_typedef_type.spelling
            except (ValueError, AttributeError):
                pass
        return f"template <{tparam_str}>\nusing {name} = {underlying};"

    # ---- class template -------------------------------------------------

    def _reconstruct_class_template_body(self, cursor):
        tparams = []
        for child in cursor.get_children():
            try:
                ck = child.kind
            except ValueError:
                continue
            if ck == cindex.CursorKind.TEMPLATE_TYPE_PARAMETER:
                tparams.append(
                    f"typename {child.spelling}" if child.spelling else "typename"
                )
            elif ck == cindex.CursorKind.TEMPLATE_NON_TYPE_PARAMETER:
                ptype = child.type.spelling if child.type else "int"
                tparams.append(
                    f"{ptype} {child.spelling}" if child.spelling else ptype
                )

        prefix = (f"template <{', '.join(tparams)}>"
                  if tparams else "template <>")

        record = self._reconstruct_record_body(cursor)
        if record is None:
            name = cursor.spelling or "(unnamed)"
            record = f"class {name} {{}};"
        return f"{prefix}\n{record}"

    # ---- access keyword helper ------------------------------------------

    @staticmethod
    def _access_keyword(access_specifier):
        """Convert an AccessSpecifier enum value to its keyword string."""
        _MAP = {
            cindex.AccessSpecifier.PUBLIC: "public",
            cindex.AccessSpecifier.PROTECTED: "protected",
            cindex.AccessSpecifier.PRIVATE: "private",
        }
        return _MAP.get(access_specifier, "")

    # ------------------------------------------------------------------
    # Phase 2 — extract source text for every type definition
    # ------------------------------------------------------------------

    def _extract_all_bodies(self):
        """Read the source text for each type that has a definition."""
        for type_info in self.types.values():
            if not type_info.is_definition:
                continue
            if not type_info.file_path or not type_info.start_line or not type_info.end_line:
                continue
            type_info.body = self._read_source_range(
                type_info.file_path, type_info.start_line, type_info.end_line
            )

    def _read_source_range(self, file_path, start_line, end_line):
        """Return lines *start_line* .. *end_line* (1-indexed, inclusive) from *file_path*."""
        try:
            if file_path not in self._file_cache:
                with open(file_path, "r", errors="replace") as f:
                    self._file_cache[file_path] = f.readlines()
            lines = self._file_cache[file_path]
            start_idx = max(0, start_line - 1)
            end_idx = min(len(lines), end_line)
            return "".join(lines[start_idx:end_idx])
        except (OSError, IOError) as e:
            print(f"    Warning: could not read {file_path}: {e}")
            return None

    # ------------------------------------------------------------------
    # Phase 2.5 — detect macro-generated types and fix their bodies
    # ------------------------------------------------------------------

    def _detect_and_fix_macro_types(self):
        """Detect macro-generated types and replace bodies with AST reconstructions.

        A type is considered macro-generated if its source body text (before
        the first ``{`` or ``;``) does not contain any C/C++ type-declaration
        keyword (struct, class, enum, union, typedef, using, template).
        """
        fixed = 0
        for type_info in self.types.values():
            if not type_info.is_definition or not type_info.body:
                continue

            # Grab prefix before the first '{' or ';'.
            body = type_info.body
            brace = body.find('{')
            semi = body.find(';')
            if brace == -1:
                brace = len(body)
            if semi == -1:
                semi = len(body)
            prefix = body[:min(brace, semi)]

            if self._TYPE_KEYWORD_RE.search(prefix):
                continue  # genuine type definition

            type_info.macro_generated = True
            if type_info._reconstructed_body:
                type_info.body = type_info._reconstructed_body
                fixed += 1

        macro_count = sum(1 for t in self.types.values() if t.macro_generated)
        if macro_count:
            print(
                f"TypeParser: detected {macro_count} macro-generated type(s), "
                f"replaced {fixed} body(ies) with AST reconstructions"
            )

    # ------------------------------------------------------------------
    # Cycle detection — Tarjan's SCC
    # ------------------------------------------------------------------

    def _find_cycles(self):
        """Find strongly connected components in the type dependency graph.

        Only SCCs with more than one member are reported (those represent
        actual cyclic dependencies).

        Returns the list of SCCs (each SCC is a list of USRs).
        """
        index_counter = [0]
        stack = []
        lowlink = {}
        index = {}
        on_stack = {}
        sccs = []

        def _strongconnect(v):
            index[v] = index_counter[0]
            lowlink[v] = index_counter[0]
            index_counter[0] += 1
            stack.append(v)
            on_stack[v] = True

            v_info = self.types.get(v)
            neighbours = v_info.depends_on if v_info else set()
            for w in neighbours:
                if w not in self.types:
                    continue
                if w not in index:
                    _strongconnect(w)
                    lowlink[v] = min(lowlink[v], lowlink[w])
                elif on_stack.get(w, False):
                    lowlink[v] = min(lowlink[v], index[w])

            if lowlink[v] == index[v]:
                scc = []
                while True:
                    w = stack.pop()
                    on_stack[w] = False
                    scc.append(w)
                    if w == v:
                        break
                if len(scc) > 1:
                    sccs.append(scc)

        # Use sys.setrecursionlimit to handle large type graphs safely.
        old_limit = sys.getrecursionlimit()
        needed = len(self.types) + 500
        if needed > old_limit:
            sys.setrecursionlimit(needed)
        try:
            for v in self.types:
                if v not in index:
                    _strongconnect(v)
        finally:
            sys.setrecursionlimit(old_limit)

        # Annotate each TypeInfo with the SCC groups it belongs to.
        for scc in sccs:
            scc_names = sorted(
                self.types[u].qualified_name
                for u in scc
                if u in self.types
            )
            for u in scc:
                if u in self.types:
                    self.types[u].scc_groups.append(scc_names)

        return sccs

    # ------------------------------------------------------------------
    # JSON export
    # ------------------------------------------------------------------

    def export_to_json(self, output_path):
        """Write type information to *output_path* as JSON.

        Output structure (grouped by the file where each type is *defined*)::

            {
              "/path/to/header.h": {
                "TypeName": { "body": ..., "kind": ..., ... },
                ...
              },
              ...
            }

        Each type appears exactly once, under its definition file.
        """
        # Build USR -> qualified_name lookup.
        usr_to_name = {
            usr: info.qualified_name for usr, info in self.types.items()
        }

        # Group types by their definition file.
        by_file = defaultdict(dict)

        for usr, type_info in self.types.items():
            if not type_info.is_definition:
                continue
            if not type_info.file_path:
                continue

            key = type_info.qualified_name
            file_types = by_file[type_info.file_path]
            # Disambiguate duplicate qualified names within the same file.
            if key in file_types:
                key = (
                    f"{key} [{type_info.kind}@"
                    f"{os.path.basename(type_info.file_path)}"
                    f":{type_info.start_line}]"
                )

            depends_on_names = sorted(
                usr_to_name.get(dep_usr, dep_usr)
                for dep_usr in type_info.depends_on
                if dep_usr in self.types and self.types[dep_usr].is_definition
            )

            file_types[key] = {
                "body": type_info.body or "",
                "kind": type_info.kind,
                "location": {
                    "start_line": type_info.start_line,
                    "end_line": type_info.end_line,
                },
                "depends_on": depends_on_names,
                "cycles": type_info.scc_groups,
                "macro_generated": type_info.macro_generated,
                "has_ostream_operator": type_info.has_ostream_operator,
                "member_variables": type_info.member_variables,
                "template_parameters": type_info.template_parameters,
            }

        # Sort by file path for stable output.
        output = dict(sorted(by_file.items()))
        total_exported = sum(len(v) for v in output.values())

        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)

        print(
            f"TypeParser: exported {total_exported} type definitions "
            f"across {len(output)} files to {output_path}"
        )
        return output

    # ------------------------------------------------------------------
    # Type lookup by name (for implicit constructor fallback)
    # ------------------------------------------------------------------

    def _find_type_by_name_and_file(self, name, file_path):
        """Find a type USR whose spelling matches *name*.

        Tries (file, name) first, then name-only.  Returns the USR of
        the first matching *definition* or ``None``.
        """
        abs_file = os.path.abspath(file_path) if file_path else None
        # Prefer types defined in the same file.
        best = None
        for usr, ti in self.types.items():
            if ti.spelling != name:
                continue
            if not ti.is_definition:
                continue
            if abs_file and ti.file_path == abs_file:
                return usr  # exact file match — return immediately
            if best is None:
                best = usr
        return best

    # ------------------------------------------------------------------
    # Transitive type expansion
    # ------------------------------------------------------------------

    def _expand_types_transitively(self, type_usrs):
        """Given a set of type USRs, follow depends_on edges transitively.

        Returns the full set of reachable type USRs (including the input set).
        """
        visited = set()
        worklist = list(type_usrs)
        while worklist:
            usr = worklist.pop()
            if usr in visited:
                continue
            visited.add(usr)
            type_info = self.types.get(usr)
            if type_info:
                for dep_usr in type_info.depends_on:
                    if dep_usr not in visited and dep_usr in self.types:
                        worklist.append(dep_usr)
        return visited

    # ------------------------------------------------------------------
    # Function-types extraction (given a function list)
    # ------------------------------------------------------------------

    def extract_function_types(self, func_list):
        """Extract all transitively-dependent types for a list of functions.

        Args:
            func_list: A list of dicts, each with keys ``name``, ``file``
                (optional), and ``line`` (optional)::

                    [
                      {"name": "func_name", "file": "/path/to/file.cc", "line": 42},
                      ...
                    ]

        Returns:
            A dict keyed by source file, then function name, containing
            all transitively-dependent type definitions grouped by the
            file where the type is defined::

                {
                  "/path/to/func_file.cc": {
                    "func_name": {
                      "has_cycles": bool,
                      "cycles": [...],
                      "types": {
                        "/path/to/header.h": {
                          "TypeA": {"body": ..., "kind": ..., ...},
                        }
                      }
                    }
                  }
                }
        """
        # Build lookup indexes for matching.
        # Index 1: (abs_file, line) -> func_usr  (most precise)
        by_file_line = {}
        # Index 2: (abs_file, name) -> [func_usr, ...]
        by_file_name = defaultdict(list)
        # Index 3: name -> [func_usr, ...]
        by_name = defaultdict(list)

        for func_usr, info in self.function_info.items():
            if info["file"] and info["line"]:
                by_file_line[(info["file"], info["line"])] = func_usr
            if info["file"] and info["name"]:
                by_file_name[(info["file"], info["name"])].append(func_usr)
            if info["name"]:
                by_name[info["name"]].append(func_usr)

        usr_to_name = {
            usr: info.qualified_name for usr, info in self.types.items()
        }

        output = {}
        matched = 0

        for entry in tqdm(func_list, desc="  Extracting types",
                          unit="func"):
            func_name = entry.get("name", "")
            func_file = entry.get("file", "")
            func_line = entry.get("line", None)

            if func_file:
                func_file = os.path.abspath(func_file)

            # Try to match the function — most precise first.
            func_usr = None

            # 1. Exact match by (file, line).
            if func_file and func_line is not None:
                func_usr = by_file_line.get((func_file, func_line))

            # 2. Match by (file, name).
            if func_usr is None and func_file and func_name:
                candidates = by_file_name.get((func_file, func_name), [])
                if len(candidates) == 1:
                    func_usr = candidates[0]
                elif len(candidates) > 1 and func_line is not None:
                    # Pick the one closest to the given line.
                    func_usr = min(
                        candidates,
                        key=lambda u: abs((self.function_info[u]["line"] or 0) - func_line),
                    )

            # 3. Match by name only (last resort).
            if func_usr is None and func_name:
                candidates = by_name.get(func_name, [])
                if len(candidates) == 1:
                    func_usr = candidates[0]

            if func_usr is None:
                # Fallback for implicit constructors / destructors whose
                # location points to the class definition (invisible in
                # the AST).  If the unmatched function name is the same
                # as a known type, treat it as a constructor call and
                # include the class type.
                type_usr = self._find_type_by_name_and_file(
                    func_name, func_file)
                if type_usr:
                    matched += 1
                    all_usrs = self._expand_types_transitively({type_usr})
                else:
                    print(
                        f"  Warning: could not match function "
                        f"{func_name} at {func_file}:{func_line}"
                    )
                    continue
            else:
                matched += 1

                # Get direct types and expand transitively.
                direct_usrs = self.function_type_usage.get(func_usr, set())
                all_usrs = self._expand_types_transitively(direct_usrs)

            # Group types by their definition file.
            by_def_file = defaultdict(dict)
            for u in all_usrs:
                if u not in self.types:
                    continue
                ti = self.types[u]
                if not ti.is_definition or not ti.file_path:
                    continue

                dep_names = sorted(
                    usr_to_name.get(d, d)
                    for d in ti.depends_on
                    if d in self.types and self.types[d].is_definition
                )

                by_def_file[ti.file_path][ti.qualified_name] = {
                    "body": ti.body or "",
                    "kind": ti.kind,
                    "location": {
                        "start_line": ti.start_line,
                        "end_line": ti.end_line,
                    },
                    "depends_on": dep_names,
                    "macro_generated": ti.macro_generated,
                    "has_ostream_operator": ti.has_ostream_operator,
                    "member_variables": ti.member_variables,
                    "template_parameters": ti.template_parameters,
                }

            # Sort the def-file keys and type names within each file.
            func_types = {}
            for def_file in sorted(by_def_file):
                func_types[def_file] = dict(
                    sorted(by_def_file[def_file].items())
                )

            # Collect cycle info from all types this function depends on.
            seen_cycles = set()
            cycles = []
            for u in all_usrs:
                ti = self.types.get(u)
                if ti is None:
                    continue
                for scc_group in ti.scc_groups:
                    scc_key = tuple(sorted(scc_group))
                    if scc_key not in seen_cycles:
                        seen_cycles.add(scc_key)
                        cycles.append(scc_group)

            out_file = entry.get("file", func_file)
            if out_file not in output:
                output[out_file] = {}
            output[out_file][func_name] = {
                "has_cycles": len(cycles) > 0,
                "cycles": cycles,
                "types": func_types,
            }

        print(
            f"TypeParser: matched {matched}/{len(func_list)} functions"
        )
        return output

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def run(self, output_path="types.json", parallel=True, max_workers=None):
        """Run the complete pipeline: parse -> cycles -> export."""
        self.parse_all_files(parallel=parallel, max_workers=max_workers)
        sccs = self._find_cycles()
        if sccs:
            print(f"TypeParser: found {len(sccs)} cyclic dependency group(s)")
        else:
            print("TypeParser: no cyclic dependencies found")
        self.export_to_json(output_path)


# ---------------------------------------------------------------------------
# CLI entry point
# Example usage: python3 TypeParser.py ../tests/fmt/build/compile_commands.json -o test_types.json --functions fmt_functions.json --functions-output fmt_function_types.json 
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Extract user-defined C/C++ types from a project."
    )
    ap.add_argument("compile_commands", help="Path to compile_commands.json")
    ap.add_argument(
        "-o", "--output", default="types.json",
        help="Output JSON for type definitions (default: types.json)",
    )
    ap.add_argument(
        "--functions", default=None,
        help=(
            "Path to a JSON file listing functions to analyze. "
            "Format: [{\"name\": ..., \"file\": ..., \"line\": ...}, ...]. "
            "Produces a second JSON with transitive type dependencies per function."
        ),
    )
    ap.add_argument(
        "--functions-output", default="function_types.json",
        help="Output JSON for per-function types (default: function_types.json)",
    )
    args = ap.parse_args()

    parser = TypeParser(args.compile_commands)
    parser.run(args.output)

    with open(args.functions, 'r') as f:
        func_list = json.load(f)

    types_output = parser.extract_function_types(func_list)

    with open(args.functions_output, 'w') as f:
        json.dump(types_output, f, indent=2)
