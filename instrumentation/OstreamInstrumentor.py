#!/usr/bin/env python3
"""
OstreamInstrumentor: Generates and injects ``operator<<`` overloads for
user-defined C/C++ types so that they can be printed via ``std::ostream``
at runtime for logging / debugging.

Designed to consume the structured type information produced by
:class:`TypeParser.TypeParser` (specifically ``TypeInfo.member_variables``
and ``TypeInfo.template_parameters``).

Usage (programmatic)::

    from analysis.TypeParser import TypeParser
    from instrumentation.OstreamInstrumentor import OstreamInstrumentor

    tp = TypeParser("compile_commands.json")
    tp.run("types.json")

    instrumentor = OstreamInstrumentor(tp)
    instrumentor.instrument(output_manifest="ostream_manifest.json")

Usage (CLI)::

    python OstreamInstrumentor.py compile_commands.json -o ostream_manifest.json
"""
import pdb
import json
import os
import re
from collections import defaultdict


# ============================================================================
# PrintabilityClassifier
# ============================================================================

class PrintabilityClassifier:
    """Determines whether a member variable's type is directly printable
    via ``operator<<`` on a ``std::ostream``.

    Classification results
    ----------------------
    ``"printable"``
        The member can be emitted with ``os << obj.member``.
    ``"printable_pointer"``
        The member is a pointer to a printable type — needs a null guard.
    ``"dependent"``
        The member's type depends on a template parameter — use the
        SFINAE ``try_print`` helper at compile time.
    ``"non_printable"``
        The member cannot be printed; emit a comment instead.
    """

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

    NON_PRINTABLE_PREFIXES = [
        "std::vector", "std::map", "std::unordered_map",
        "std::set", "std::unordered_set", "std::multiset",
        "std::multimap", "std::unordered_multimap", "std::unordered_multiset",
        "std::list", "std::forward_list",
        "std::deque", "std::queue", "std::stack", "std::priority_queue",
        "std::array", "std::pair", "std::tuple",
        "std::unique_ptr", "std::shared_ptr", "std::weak_ptr",
        "std::optional", "std::variant", "std::any",
        "std::function", "std::mutex", "std::recursive_mutex",
        "std::thread", "std::atomic",
        "std::bitset",
    ]

    def __init__(self, type_graph=None):
        """
        Args:
            type_graph: dict mapping USR -> TypeInfo (or a dict with at least
                ``is_definition`` and ``kind`` keys).  Used to decide whether
                a user-defined type will also receive a generated
                ``operator<<``.
        """
        self.type_graph = type_graph or {}
        self._generated_usrs = set()

    def register_generated(self, usr):
        """Mark that ``operator<<`` will be generated for the type with *usr*."""
        self._generated_usrs.add(usr)

    # -----------------------------------------------------------------

    def classify(self, member_dict):
        """Classify a member-variable dict.

        Returns one of ``"printable"``, ``"printable_pointer"``,
        ``"dependent"``, or ``"non_printable"``.
        """
        type_spelling = member_dict.get("type_spelling", "")
        stripped = self._strip_cv(type_spelling)

        # 1. Arrays are never directly printable.
        if member_dict.get("is_array"):
            return "non_printable"

        # 2. Bitfields are always integral → printable.
        if member_dict.get("is_bitfield"):
            return "printable"

        # 3. Template-dependent types → SFINAE try_print.
        if member_dict.get("is_dependent_type"):
            return "dependent"

        # 4. Pointers.
        if member_dict.get("is_pointer"):
            return self._classify_pointer(member_dict)

        # 5. References — classify the underlying type.
        if member_dict.get("is_reference"):
            return self._classify_reference(member_dict)

        # 6. Primitives.
        if stripped in self.PRINTABLE_PRIMITIVES:
            return "printable"

        # 7. Known printable std types.
        if stripped in self.PRINTABLE_STD_TYPES:
            return "printable"

        # 8. Known non-printable patterns (STL containers, smart ptrs, …).
        for prefix in self.NON_PRINTABLE_PREFIXES:
            if stripped.startswith(prefix):
                return "non_printable"

        # 9. Enums.
        if member_dict.get("type_kind") == "ENUM":
            return "printable"

        # 10. User-defined types whose operator<< will be generated.
        type_usr = member_dict.get("type_usr", "")
        if type_usr and type_usr in self._generated_usrs:
            return "printable"

        # 11. Default.
        return "non_printable"

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------

    def _classify_pointer(self, member_dict):
        pointee = self._strip_cv(member_dict.get("pointee_type_spelling", ""))
        # char* / const char* — ostream handles natively.
        if pointee in ("char", "signed char", "wchar_t"):
            return "printable"
        # Pointer to a printable primitive.
        if pointee in self.PRINTABLE_PRIMITIVES:
            return "printable_pointer"
        # Pointer to a printable std type.
        if pointee in self.PRINTABLE_STD_TYPES:
            return "printable_pointer"
        # Pointer to a user-defined type with generated operator<<.
        type_usr = member_dict.get("pointee_type_usr", "") or member_dict.get("type_usr", "")
        if type_usr and type_usr in self._generated_usrs:
            # If the pointee is in a dependency cycle, its operator<< may not
            # be defined before ours → can't safely dereference.
            ti = self.type_graph.get(type_usr)
            if ti and getattr(ti, "scc_groups", []):
                return "non_printable"
            return "printable_pointer"
        return "non_printable"

    def _classify_reference(self, member_dict):
        """References are printed just like the underlying type."""
        type_spelling = member_dict.get("type_spelling", "")
        base = self._strip_cv(type_spelling).rstrip("&").rstrip().rstrip("&").strip()
        if base in self.PRINTABLE_PRIMITIVES:
            return "printable"
        if base in self.PRINTABLE_STD_TYPES:
            return "printable"
        type_usr = member_dict.get("type_usr", "")
        if type_usr and type_usr in self._generated_usrs:
            return "printable"
        return "non_printable"

    @staticmethod
    def _strip_cv(type_str):
        """Strip leading ``const`` / ``volatile`` qualifiers."""
        result = type_str.strip()
        for prefix in ("const volatile ", "volatile const ",
                       "const ", "volatile "):
            if result.startswith(prefix):
                result = result[len(prefix):]
        return result.strip()


# ============================================================================
# OstreamCodeGenerator
# ============================================================================

class OstreamCodeGenerator:
    """Generates ``operator<<`` code strings for types."""

    _SFINAE_HELPER = '''\
namespace _instrumentation {
template<typename _T>
inline auto try_print(std::ostream& os, const char* label, const _T& val, int)
    -> decltype(os << val, void()) {
    os << label << val;
}
template<typename _T>
inline void try_print(std::ostream& os, const char* label, const _T&, ...) {
    os << label << "(non-printable)";
}
} // namespace _instrumentation'''

    def __init__(self, classifier):
        self.classifier = classifier

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def generate_for_record(self, type_info):
        """Generate a ``friend operator<<`` for a struct/class.

        Returns the code string to be injected inside the class body
        (before the closing ``};``).
        """
        name = type_info.spelling or type_info.qualified_name.split("::")[-1]
        members = type_info.member_variables
        lines = self._build_operator_body(name, members, is_template=False)
        return self._wrap_friend(name, lines)

    def generate_for_enum(self, type_info):
        """Generate a free ``inline operator<<`` for an enum.

        Returns the code string to be injected *after* the enum definition.
        """
        name = type_info.spelling or type_info.qualified_name.split("::")[-1]
        return (
            f"inline std::ostream& operator<<(std::ostream& os, {name} _val) {{\n"
            f"    os << \"{name}(\" << static_cast<int>(_val) << \")\";\n"
            f"    return os;\n"
            f"}}"
        )

    def generate_for_template(self, type_info):
        """Generate a ``friend operator<<`` for a class template.

        Template-parameter-dependent members use SFINAE ``try_print``.
        Returns the code string to be injected inside the class body.
        """
        name = type_info.spelling or type_info.qualified_name.split("::")[-1]
        members = type_info.member_variables
        lines = self._build_operator_body(name, members, is_template=True)
        return self._wrap_friend(name, lines)

    def generate_sfinae_helper(self):
        """Return the SFINAE ``_instrumentation::try_print`` helper block."""
        return self._SFINAE_HELPER

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------

    def _wrap_friend(self, class_name, body_lines):
        """Wrap body lines in a ``friend std::ostream& operator<<`` function."""
        parts = [
            f"    friend std::ostream& operator<<(std::ostream& os, const {class_name}& obj) {{",
        ]
        parts.extend(body_lines)
        parts.append("        return os;")
        parts.append("    }")
        return "\n".join(parts)

    def _build_operator_body(self, class_name, members, is_template):
        """Build the statement lines between the function braces."""
        lines = [f'        os << "{class_name}{{";']

        is_first = True
        for member in members:
            classification = self.classifier.classify(member)
            stmt = self._format_member_print(
                member, class_name, classification, is_first, is_template,
            )
            lines.append(stmt)
            # Only advance is_first for lines that actually print something
            # visible (not comments).
            if classification != "non_printable":
                is_first = False

        lines.append('        os << "}";')
        return lines

    def _format_member_print(self, member, class_name, classification,
                             is_first, is_template):
        prefix = " " if is_first else ", "
        name = member["name"]
        type_spelling = member["type_spelling"]

        if member.get("is_static"):
            accessor = f"{class_name}::{name}"
            label_prefix = "[static] "
        else:
            accessor = f"obj.{name}"
            label_prefix = ""

        if classification == "printable":
            return (
                f'        os << "{prefix}{label_prefix}{name}=" << {accessor};'
            )

        if classification == "printable_pointer":
            return (
                f'        os << "{prefix}{label_prefix}{name}=";\n'
                f'        if ({accessor}) os << *{accessor}; else os << "(nullptr)";'
            )

        if classification == "dependent":
            if is_template:
                return (
                    f'        _instrumentation::try_print(os, "{prefix}{label_prefix}{name}=", {accessor}, 0);'
                )
            # Fallback: treat as regular printable (shouldn't normally happen
            # for non-templates, but be safe).
            return (
                f'        os << "{prefix}{label_prefix}{name}=" << {accessor};'
            )

        # non_printable — emit as a comment.
        return (
            f'        // {label_prefix}{name}: {type_spelling} (non-printable)'
        )


# ============================================================================
# InstrumentationManifest
# ============================================================================

class InstrumentationManifest:
    """Records all ``operator<<`` injections for traceability."""

    def __init__(self):
        self.entries = []

    def record(self, file_path, line_number, injection_type, type_name, code):
        """Record a single injection.

        Args:
            file_path:      Absolute path to the source file.
            line_number:    Original (pre-injection) 1-based line number.
            injection_type: One of ``"friend_operator"``,
                ``"free_operator"``, ``"sfinae_helper"``, ``"include"``.
            type_name:      Qualified type name (``""`` for includes/helpers).
            code:           The injected code string.
        """
        self.entries.append({
            "file": file_path,
            "original_line": line_number,
            "injection_type": injection_type,
            "type_name": type_name,
            "code": code,
        })

    def to_dict(self):
        by_file = defaultdict(list)
        for entry in self.entries:
            by_file[entry["file"]].append(entry)
        return {
            "total_injections": len(self.entries),
            "by_file": dict(sorted(by_file.items())),
        }

    def save(self, output_path):
        with open(output_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


# ============================================================================
# SourceInjector
# ============================================================================

class SourceInjector:
    """Injects generated ``operator<<`` code into source files.

    All mutations are accumulated in memory and written back atomically
    via :meth:`write_all`.
    """

    OSTREAM_INCLUDE = "#include <iostream>"
    INSTRUMENTATION_TAG = "// [ostream-instrumentation]"

    def __init__(self):
        self.manifest = InstrumentationManifest()
        self._file_lines = {}          # abs path -> list of line strings
        self._injections = []          # list of pending injection dicts
        self._sfinae_injected = set()  # file paths where SFINAE helper is done

    # -----------------------------------------------------------------
    # File loading
    # -----------------------------------------------------------------

    def load_file(self, file_path):
        """Load a file into the line buffer (lazy, idempotent)."""
        if file_path not in self._file_lines:
            with open(file_path, "r", errors="replace") as f:
                self._file_lines[file_path] = f.readlines()

    # -----------------------------------------------------------------
    # Injection scheduling
    # -----------------------------------------------------------------

    def inject_friend_into_class(self, file_path, end_line, code, type_name=""):
        """Schedule injection of a friend function before the class closing ``};``.

        Args:
            file_path: Absolute path.
            end_line:  1-based line number of the closing ``};``.
            code:      The friend function code to inject.
            type_name: Qualified type name (for manifest).
        """
        self.load_file(file_path)
        lines = self._file_lines[file_path]
        target_line = lines[end_line - 1] if end_line <= len(lines) else ""

        # Detect single-line definitions like "struct Tag { int id; };"
        # where the opening '{' and closing '};' are on the same line.
        # These need to be expanded into multi-line before injecting.
        needs_expand = '{' in target_line and '};' in target_line

        self._injections.append({
            "file": file_path,
            "line": end_line,         # inject *before* this line
            "mode": "before",
            "code": code,
            "injection_type": "friend_operator",
            "type_name": type_name,
            "_expand_single_line": needs_expand,
        })

    def inject_after_type(self, file_path, end_line, code, type_name=""):
        """Schedule injection of a free function after a type definition.

        Args:
            file_path: Absolute path.
            end_line:  1-based line number of the closing ``};``.
            code:      The free function code to inject.
            type_name: Qualified type name (for manifest).
        """
        self.load_file(file_path)
        self._injections.append({
            "file": file_path,
            "line": end_line,
            "mode": "after",
            "code": code,
            "injection_type": "free_operator",
            "type_name": type_name,
        })

    def inject_sfinae_helper(self, file_path, before_line, code):
        """Schedule injection of the SFINAE helper namespace (once per file).

        Args:
            file_path:   Absolute path.
            before_line: 1-based line number to insert before.
            code:        The SFINAE helper code.
        """
        if file_path in self._sfinae_injected:
            return
        self._sfinae_injected.add(file_path)
        self.load_file(file_path)
        self._injections.append({
            "file": file_path,
            "line": before_line,
            "mode": "before",
            "code": code,
            "injection_type": "sfinae_helper",
            "type_name": "",
        })

    def ensure_iostream_include(self, file_path):
        """Add ``#include <iostream>`` if not already present."""
        self.load_file(file_path)
        lines = self._file_lines[file_path]
        content = "".join(lines)

        if self.OSTREAM_INCLUDE in content:
            return  # already present

        # Find the last #include line and insert after it.
        last_include_idx = -1
        for i, line in enumerate(lines):
            if re.match(r'\s*#\s*include\b', line):
                last_include_idx = i

        if last_include_idx >= 0:
            # Insert after the last existing #include line.
            self._injections.append({
                "file": file_path,
                "line": last_include_idx + 1 + 1,   # 0-based -> 1-based
                "mode": "after",
                "code": self.OSTREAM_INCLUDE,
                "injection_type": "include",
                "type_name": "",
            })
        else:
            # No existing includes — insert at the very top of the file.
            self._injections.append({
                "file": file_path,
                "line": 1,
                "mode": "before",
                "code": self.OSTREAM_INCLUDE,
                "injection_type": "include",
                "type_name": "",
            })

    # -----------------------------------------------------------------
    # Write-back
    # -----------------------------------------------------------------

    def apply_injections(self):
        """Apply all scheduled injections to the in-memory line buffers.

        Injections are processed **bottom-up** (highest line number first)
        within each file so that earlier line numbers remain valid.
        """
        by_file = defaultdict(list)
        for inj in self._injections:
            by_file[inj["file"]].append(inj)

        for file_path, injections in by_file.items():
            lines = self._file_lines[file_path]
            # Sort descending by line number for bottom-up processing.
            # For same line, "before" should be processed after "after"
            # (so that "before" inserts end up above "after" inserts).
            injections.sort(
                key=lambda x: (-x["line"], x["mode"] == "before"),
            )
            for inj in injections:
                target = inj["line"] - 1   # 0-based index

                # Handle single-line definitions: expand "struct X { ... };"
                # into multiple lines so the friend function goes inside.
                if inj.get("_expand_single_line"):
                    line = lines[target]
                    brace_close_idx = line.rfind("};")
                    if brace_close_idx >= 0:
                        # Detect indentation of the original line.
                        indent = ""
                        for ch in line:
                            if ch in " \t":
                                indent += ch
                            else:
                                break
                        # Everything before "};" stays on the current line.
                        before = line[:brace_close_idx].rstrip() + "\n"
                        # "};" goes on a new line with the same indentation.
                        closing = indent + "};\n"
                        lines[target] = before
                        lines.insert(target + 1, closing)
                        # The new target for "before" is the fresh "};" line.
                        target = target + 1

                code_lines = [
                    ln + "\n" if not ln.endswith("\n") else ln
                    for ln in inj["code"].split("\n")
                ]
                # Tag the first injected line for identification.
                code_lines[0] = code_lines[0].rstrip("\n") + (
                    f"  {self.INSTRUMENTATION_TAG}\n"
                )

                if inj["mode"] == "before":
                    lines[target:target] = code_lines
                else:  # "after"
                    lines[target + 1:target + 1] = code_lines

                self.manifest.record(
                    file_path=file_path,
                    line_number=inj["line"],
                    injection_type=inj["injection_type"],
                    type_name=inj["type_name"],
                    code=inj["code"],
                )

    def write_all(self):
        """Apply pending injections and write all modified files to disk."""
        self.apply_injections()
        for file_path, lines in self._file_lines.items():
            with open(file_path, "w") as f:
                f.writelines(lines)

    def write_manifest(self, output_path):
        """Write the injection manifest JSON to *output_path*."""
        self.manifest.save(output_path)


# ============================================================================
# OstreamInstrumentor  (orchestrator)
# ============================================================================

class OstreamInstrumentor:
    """Top-level orchestrator: takes a :class:`TypeParser` instance,
    generates ``operator<<`` overloads for each instrumentable type, and
    injects them into the source files.
    """

    # Types that receive an operator<<.
    _INSTRUMENTABLE_KINDS = frozenset({
        "struct", "class", "enum",
        "class_template", "class_template_partial_spec",
    })

    def __init__(self, type_parser):
        """
        Args:
            type_parser: A :class:`TypeParser` instance that has already
                been run (``parse_all_files`` completed).
        """
        self.types = type_parser.types      # USR -> TypeInfo
        self.type_parser = type_parser

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def instrument(self, output_manifest="ostream_manifest.json"):
        """Run the full instrumentation pipeline.

        1. Determine which types are instrumentable.
        2. Register them with the classifier (topological order).
        3. Generate code for each type.
        4. Schedule injections.
        5. Write modified files + manifest.
        """
        # Phase 1 — topological ordering + classifier setup.
        ordered_usrs = self._topological_order()
        classifier = PrintabilityClassifier(self.types)
        instrumentable = []
        for usr in ordered_usrs:
            ti = self.types[usr]
            if self._should_instrument(ti):
                classifier.register_generated(usr)
                instrumentable.append((usr, ti))

        # Phase 2 — code generation + injection scheduling.
        generator = OstreamCodeGenerator(classifier)
        injector = SourceInjector()
        needs_sfinae = {}   # file_path -> earliest start_line

        for usr, ti in instrumentable:
            if not ti.file_path or not ti.end_line:
                continue

            if ti.kind in ("struct", "class"):
                code = generator.generate_for_record(ti)
                injector.inject_friend_into_class(
                    ti.file_path, ti.end_line, code,
                    type_name=ti.qualified_name,
                )
            elif ti.kind == "enum":
                code = generator.generate_for_enum(ti)
                # For enums nested inside a class (USR contains @S@
                # before the @E@ segment), the free operator<< must be
                # injected after the *enclosing class*, not after the
                # enum itself (which is still inside the class body).
                inject_line = ti.end_line
                import re as _re
                _parent_classes = _re.findall(r'@S@([^@]+)', usr)
                if _parent_classes:
                    # Try type database first.
                    _found_parent = False
                    for _pu, _pt in self.types.items():
                        if (_pt.spelling == _parent_classes[-1]
                                and _pt.kind in ("struct", "class",
                                                  "class_template")
                                and _pt.file_path == ti.file_path
                                and _pt.end_line
                                and _pt.end_line > ti.end_line):
                            inject_line = _pt.end_line
                            _found_parent = True
                            break
                    # Fallback: scan the source file for the enclosing
                    # closing brace by tracking brace depth from the
                    # enum's end line.
                    if not _found_parent and ti.file_path:
                        _enc = self._find_enclosing_class_end(
                            ti.file_path, ti.end_line)
                        if _enc:
                            inject_line = _enc
                injector.inject_after_type(
                    ti.file_path, inject_line, code,
                    type_name=ti.qualified_name,
                )
            elif ti.kind in ("class_template", "class_template_partial_spec"):
                code = generator.generate_for_template(ti)
                injector.inject_friend_into_class(
                    ti.file_path, ti.end_line, code,
                    type_name=ti.qualified_name,
                )
                # Track whether this file needs the SFINAE helper.
                if any(m.get("is_dependent_type")
                       for m in ti.member_variables):
                    cur = needs_sfinae.get(ti.file_path)
                    if cur is None or ti.start_line < cur:
                        needs_sfinae[ti.file_path] = ti.start_line

        # Phase 3 — SFINAE helpers (once per file).
        sfinae_code = generator.generate_sfinae_helper()
        for file_path, earliest_line in needs_sfinae.items():
            injector.inject_sfinae_helper(file_path, earliest_line, sfinae_code)

        # Phase 4 — #include <iostream>.
        seen_files = {ti.file_path for _, ti in instrumentable if ti.file_path}
        for file_path in seen_files:
            injector.ensure_iostream_include(file_path)

        # Phase 5 — write.
        injector.write_all()
        injector.write_manifest(output_manifest)

        print(
            f"OstreamInstrumentor: instrumented {len(instrumentable)} type(s) "
            f"across {len(seen_files)} file(s).  Manifest → {output_manifest}"
        )
        return injector.manifest

    def generate_instrumentation_plan(self, func_list, output_path,
                                       function_types=None):
        """Generate an instrumentation plan as JSON without modifying files.

        For a set of input functions, resolves their transitive type
        dependencies, generates ``operator<<`` code for each instrumentable
        type, and writes a JSON file containing both original bodies and
        generated code in topological (dependency) order.

        Args:
            func_list: list of dicts with keys ``name``, ``file`` (optional),
                ``line`` (optional).
            output_path: path to write the output JSON.
            function_types: optional pre-computed result from
                ``TypeParser.extract_function_types(func_list)``.  When
                provided the method skips its own function-matching step
                and derives type USRs directly from this dict.

        Returns:
            The plan dict (also written to *output_path*).
        """
        # --- Step 1: resolve target types ---
        if function_types is not None:
            target_usrs, matched, total = \
                self._usrs_from_function_types(function_types, len(func_list))
        else:
            target_usrs, matched, total = self._resolve_target_usrs(func_list)

        # --- Step 2: filter through _should_instrument ---
        instrumentable_usrs = set()
        for usr in target_usrs:
            ti = self.types.get(usr)
            if ti and self._should_instrument(ti):
                instrumentable_usrs.add(usr)

        # --- Step 3: topological sort (scoped) ---
        ordered_usrs = self._topological_order(target_usrs=instrumentable_usrs)

        # --- Step 4: classify + generate ---
        classifier = PrintabilityClassifier(self.types)
        generator = OstreamCodeGenerator(classifier)
        needs_sfinae = {}       # file_path -> earliest start_line
        files_with_types = set()

        # USR -> qualified_name map for dependency resolution.
        usr_to_name = {
            usr: ti.qualified_name
            for usr, ti in self.types.items()
        }

        types_list = []
        for order_idx, usr in enumerate(ordered_usrs):
            ti = self.types[usr]
            classifier.register_generated(usr)

            # Generate code.
            if ti.kind in ("struct", "class"):
                code = generator.generate_for_record(ti)
                injection_type = "friend_operator"
            elif ti.kind == "enum":
                code = generator.generate_for_enum(ti)
                injection_type = "free_operator"
            elif ti.kind in ("class_template", "class_template_partial_spec"):
                code = generator.generate_for_template(ti)
                injection_type = "friend_operator"
                if any(m.get("is_dependent_type")
                       for m in ti.member_variables):
                    cur = needs_sfinae.get(ti.file_path)
                    if cur is None or ti.start_line < cur:
                        needs_sfinae[ti.file_path] = ti.start_line
            else:
                continue

            if ti.file_path:
                files_with_types.add(ti.file_path)

            # Detect single-line definition.
            needs_expand = False
            if injection_type == "friend_operator" and ti.start_line == ti.end_line:
                needs_expand = True

            # Dependency names (only those in our instrumentable set).
            dep_names = sorted(
                usr_to_name.get(d, d)
                for d in ti.depends_on
                if d in instrumentable_usrs
            )

            types_list.append({
                "order": order_idx,
                "usr": usr,
                "qualified_name": ti.qualified_name,
                "spelling": ti.spelling,
                "kind": ti.kind,
                "file": ti.file_path,
                "start_line": ti.start_line,
                "end_line": ti.end_line,
                "depends_on": dep_names,
                "cycles": ti.scc_groups,
                "member_variables": ti.member_variables,
                "template_parameters": ti.template_parameters,
                "has_ostream_operator": ti.has_ostream_operator,
                "original_body": ti.body or "",
                "injection_type": injection_type,
                "generated_code": code,
                "needs_single_line_expansion": needs_expand,
            })

        # --- Step 5: build auxiliary info ---
        sfinae_code = generator.generate_sfinae_helper()
        auxiliary = {
            "sfinae_helper_code": sfinae_code,
            "files_needing_sfinae": needs_sfinae,
            "files_needing_include": sorted(files_with_types),
            "include_code": SourceInjector.OSTREAM_INCLUDE,
        }

        # --- Step 6: assemble and write ---
        plan = {
            "source_functions": func_list,
            "types": types_list,
            "auxiliary": auxiliary,
            "total_types": len(types_list),
            "total_functions_matched": matched,
            "total_functions_provided": total,
        }

        # with open(output_path, "w") as f:
        #     json.dump(plan, f, indent=2)

        print(
            f"OstreamInstrumentor: generated plan for {len(types_list)} type(s) "
            f"from {matched}/{total} function(s).  Plan → {output_path}"
        )
        return plan

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------

    @staticmethod
    def _find_enclosing_class_end(file_path, inner_end_line):
        """Find the closing ``};`` of the class that encloses *inner_end_line*.

        *inner_end_line* is the 1-based line of the nested type's own
        closing ``};``.  We scan **forward** from the line after it,
        starting at depth 0 (we just exited the inner type's scope and
        are now in the enclosing class body).  Every ``{`` increases
        depth, every ``}`` decreases it.  When depth drops below 0, we
        have found the enclosing class's closing brace.

        Returns the 1-based line number of the enclosing ``};``, or
        None if the type is not nested.
        """
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
        except OSError:
            return None

        # Scan forward from the line after the inner type's };
        depth = 0
        for i in range(inner_end_line, len(lines)):  # inner_end_line is 1-based → 0-indexed = inner_end_line
            line = lines[i]
            depth += line.count('{') - line.count('}')
            if depth < 0:
                return i + 1  # 1-based
        return None

    def _should_instrument(self, type_info):
        """Return True if this type should receive an ``operator<<``."""
        if not type_info.is_definition:
            return False
        if type_info.kind not in self._INSTRUMENTABLE_KINDS:
            return False
        if not type_info.file_path:
            return False
        # Skip anonymous types — they have no valid name for operator<<.
        if not type_info.spelling:
            return False
        # Skip macro-generated types whose line info may be unreliable.
        if type_info.macro_generated:
            return False
        # Skip types that already have an operator<< (detected during AST walk).
        if type_info.has_ostream_operator:
            return False
        # Skip types that have a free operator<< (e.g. enums with external def).
        if self._has_free_ostream_operator(type_info):
            return False
        # Skip implicit template specializations — they share the same source
        # location as the class template, which is already instrumented.
        if type_info.kind in ("struct", "class") and self._is_implicit_specialization(type_info):
            return False
        return True

    def _has_free_ostream_operator(self, type_info):
        """Check ``function_info`` for a free ``operator<<`` that takes *type_info*.

        This catches out-of-class definitions and enum operator<< overloads
        that the in-class friend-decl scan cannot detect.
        """
        spelling = type_info.spelling
        if not spelling:
            return False
        for info in self.type_parser.function_info.values():
            if info["name"] == "operator<<":
                displayname = info.get("displayname", "")
                # Check if the type's spelling appears in the parameter list.
                if spelling in displayname:
                    return True
        return False

    def _is_implicit_specialization(self, type_info):
        """Return True if *type_info* is an implicit template specialization.

        Implicit specializations (e.g. ``Holder<int>`` from ``Holder<T>``)
        share the same file and line range as their class template.  The
        class template itself is already instrumented, so we skip these.
        """
        for other in self.types.values():
            if other is type_info:
                continue
            if other.kind not in ("class_template", "class_template_partial_spec"):
                continue
            if (other.file_path == type_info.file_path
                    and other.start_line == type_info.start_line
                    and other.end_line == type_info.end_line):
                return True
        return False

    def _resolve_target_usrs(self, func_list):
        """Resolve function specs to the union of their transitive type USRs.

        Uses the same multi-level matching strategy as
        ``TypeParser.extract_function_types``.

        Returns:
            tuple(target_usrs: set, matched: int, total: int)
        """
        tp = self.type_parser

        # Build lookup indexes.
        by_file_line = {}
        by_file_name = defaultdict(list)
        by_name = defaultdict(list)

        for func_usr, info in tp.function_info.items():
            if info["file"] and info["line"]:
                by_file_line[(info["file"], info["line"])] = func_usr
            if info["file"] and info["name"]:
                by_file_name[(info["file"], info["name"])].append(func_usr)
            if info["name"]:
                by_name[info["name"]].append(func_usr)

        all_direct_usrs = set()
        matched = 0

        for entry in func_list:
            func_name = entry.get("name", "")
            func_file = entry.get("file", "")
            func_line = entry.get("line", None)

            if func_file:
                func_file = os.path.abspath(func_file)

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
                    func_usr = min(
                        candidates,
                        key=lambda u: abs(
                            (tp.function_info[u]["line"] or 0) - func_line
                        ),
                    )

            # 3. Match by name only.
            if func_usr is None and func_name:
                candidates = by_name.get(func_name, [])
                if len(candidates) == 1:
                    func_usr = candidates[0]

            if func_usr is None:
                print(
                    f"  Warning: could not match function "
                    f"{func_name} at {func_file}:{func_line}"
                )
                continue

            matched += 1
            direct_usrs = tp.function_type_usage.get(func_usr, set())
            all_direct_usrs.update(direct_usrs)

        # Expand transitively.
        target_usrs = tp._expand_types_transitively(all_direct_usrs)
        return target_usrs, matched, len(func_list)

    def _usrs_from_function_types(self, function_types, total):
        """Derive the set of type USRs from a pre-computed ``function_types`` dict.

        ``function_types`` is the result of
        ``TypeParser.extract_function_types(func_list)`` — a nested dict
        keyed by source file -> function name -> {types: {header: {qname: ...}}}.

        We collect every qualified type name from the dict, then reverse-map
        each name to its USR(s) in ``self.types``.

        Returns:
            tuple(target_usrs: set, matched: int, total: int)
        """
        # Build reverse map: qualified_name -> list of USRs
        qname_to_usrs = defaultdict(list)
        for usr, ti in self.types.items():
            qname_to_usrs[ti.qualified_name].append(usr)

        # Collect all qualified names from the function_types result
        target_usrs = set()
        matched = 0
        for file_funcs in function_types.values():
            matched += len(file_funcs)
            for func_info in file_funcs.values():
                for header_types in func_info.get('types', {}).values():
                    for qname in header_types:
                        target_usrs.update(qname_to_usrs.get(qname, []))

        return target_usrs, matched, total

    def _topological_order(self, target_usrs=None):
        """Return type USRs in dependency-respecting order.

        Types with no dependencies come first so that when we process a
        type that depends on another, the dependency already has its
        ``operator<<`` registered with the classifier.

        Args:
            target_usrs: optional set of USRs to restrict to.  When provided,
                only these USRs are included in the result.  Dependencies
                outside this set are still used for ordering but not emitted.
        """
        # Kahn's algorithm for topological sort.
        in_degree = defaultdict(int)
        dependents = defaultdict(list)   # USR -> list of USRs that depend on it

        all_usrs = set()
        for usr, ti in self.types.items():
            if not ti.is_definition:
                continue
            if target_usrs is not None and usr not in target_usrs:
                continue
            all_usrs.add(usr)
            for dep in ti.depends_on:
                if dep in self.types and self.types[dep].is_definition:
                    if target_usrs is not None and dep not in target_usrs:
                        continue
                    in_degree[usr] += 1
                    dependents[dep].append(usr)

        # Seed the queue with zero-in-degree nodes.
        queue = [u for u in all_usrs if in_degree[u] == 0]
        ordered = []
        while queue:
            u = queue.pop(0)
            ordered.append(u)
            for v in dependents[u]:
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    queue.append(v)

        # If there are cycles, append the remaining nodes (the SCC members)
        # in arbitrary order — the SFINAE helper handles the case where a
        # dependent's operator<< might not exist yet.
        remaining = all_usrs - set(ordered)
        ordered.extend(sorted(remaining))

        return ordered


# ============================================================================
# CLI entry point
# ============================================================================
"""
Example usage of only generating the instrumentation plan without modifying files:

  inst = OstreamInstrumentor(tp)
  plan = inst.generate_instrumentation_plan(
      [                                        # input functions
          {"name": "process_packet", "file": "processor.cpp", "line": 3},
          {"name": "handle_connection"},       # file/line are optional
      ],
      "output/plan.json"                       # output path
  )

"""
if __name__ == "__main__":
    import argparse
    from analysis.TypeParser import TypeParser

    ap = argparse.ArgumentParser(
        description=(
            "Generate and inject operator<< overloads for user-defined "
            "C/C++ types."
        ),
    )
    ap.add_argument("compile_commands", help="Path to compile_commands.json")
    ap.add_argument(
        "-o", "--output", default="ostream_manifest.json",
        help="Output manifest JSON (default: ostream_manifest.json)",
    )
    ap.add_argument(
        "--types-output", default="types.json",
        help="TypeParser output JSON (default: types.json)",
    )
    args = ap.parse_args()

    tp = TypeParser(args.compile_commands)
    tp.run(args.types_output)

    instrumentor = OstreamInstrumentor(tp)
    instrumentor.instrument(output_manifest=args.output)
