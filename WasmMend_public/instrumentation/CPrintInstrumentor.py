#!/usr/bin/env python3
"""
CPrintInstrumentor: generates ``static inline void print_<Type>(FILE*, ...)``
helper functions for C struct / union / enum types so they can be serialized
to a FILE* stream at runtime.

Parallel to :class:`OstreamInstrumentor.OstreamInstrumentor` but for pure C
projects where ``std::cout``/``operator<<`` is not available. Consumes the
same :class:`TypeParser.TypeParser` output and emits plans with the same
top-level shape as ``OstreamInstrumentor.generate_instrumentation_plan``,
so the caller (the helper-generation step in the preprocessor) can reuse
the existing injection / repair loop.

Key differences from the C++ path:

* No ``friend`` / ``operator<<`` — helpers are free ``static inline``
  functions injected **after** the type definition (not inside it).
* Format specifiers are chosen per primitive type (``%lld`` for signed,
  ``%llu`` for unsigned, ``%g`` for floats, ``%.200s`` with NULL guard
  for ``char*``, ``%p`` for other pointers).
* No templates, no SFINAE, no namespaces.
* User-defined struct/union members recurse through the generated helper
  of the nested type. Enums pass by value; structs/unions pass by pointer.
"""
import os
from collections import defaultdict


# ============================================================================
# CPrintabilityClassifier
# ============================================================================


class CPrintabilityClassifier:
    """Classifies whether a member is printable in C via fprintf.

    For primitives, returns the format specifier and a lambda that wraps the
    accessor expression with the appropriate cast. For user-defined types,
    defers to the set of types for which helpers have been registered.
    """

    # Signed integer primitives — cast to (long long), print with %lld.
    _SIGNED_INTS = frozenset({
        "char", "signed char",
        "short", "signed short", "short int", "signed short int",
        "int", "signed", "signed int",
        "long", "signed long", "long int", "signed long int",
        "long long", "signed long long",
        "long long int", "signed long long int",
        "int8_t", "int16_t", "int32_t", "int64_t",
        "intptr_t", "ptrdiff_t", "intmax_t",
    })

    # Unsigned integer primitives — cast to (unsigned long long), print %llu.
    _UNSIGNED_INTS = frozenset({
        "unsigned char",
        "unsigned short", "unsigned short int",
        "unsigned", "unsigned int",
        "unsigned long", "unsigned long int",
        "unsigned long long", "unsigned long long int",
        "uint8_t", "uint16_t", "uint32_t", "uint64_t",
        "uintptr_t", "size_t", "uintmax_t",
    })

    # Floats — cast to (double), print %g.
    _FLOATS = frozenset({"float", "double"})

    # Bool
    _BOOLS = frozenset({"_Bool", "bool"})

    def __init__(self, types):
        """
        Args:
            types: USR -> TypeInfo dict from TypeParser.
        """
        self.types = types
        self._generated_usrs = set()

    def register_generated(self, usr):
        self._generated_usrs.add(usr)

    def is_helper_generated(self, usr):
        return usr in self._generated_usrs

    @staticmethod
    def strip_cv(type_str):
        result = type_str.strip()
        for prefix in ("const volatile ", "volatile const ",
                       "const ", "volatile "):
            if result.startswith(prefix):
                result = result[len(prefix):]
        return result.strip()

    def format_for_primitive(self, type_spelling):
        """Return (fmt_spec, cast_prefix) for a primitive C type, or None.

        ``cast_prefix`` is a C cast like ``"(long long)"`` to be prepended
        to the accessor expression. Empty string means no cast needed.
        """
        t = self.strip_cv(type_spelling)

        if t in self._SIGNED_INTS:
            return ("%lld", "(long long)")
        if t in self._UNSIGNED_INTS:
            return ("%llu", "(unsigned long long)")
        if t in self._FLOATS:
            return ("%g", "(double)")
        if t == "long double":
            return ("%Lg", "")
        if t in self._BOOLS:
            return ("%d", "(int)")
        return None


# ============================================================================
# CPrintCodeGenerator
# ============================================================================


class CPrintCodeGenerator:
    """Generates ``static inline void print_<Type>(...)`` helper functions."""

    def __init__(self, classifier, types):
        """
        Args:
            classifier: CPrintabilityClassifier instance.
            types:      USR -> TypeInfo dict from TypeParser.
        """
        self.classifier = classifier
        self.types = types

    # -----------------------------------------------------------------
    # Public helper for name mangling (used elsewhere too)
    # -----------------------------------------------------------------

    @staticmethod
    def helper_name(type_info):
        """Return the C identifier used for *type_info*'s print helper.

        Format: ``print_<kind>_<flat_qname>``. Mangling by kind guards
        against the (rare) case of a ``struct Foo`` and ``enum Foo`` in
        the same file. Flattening ``::`` handles nested scopes even
        though C doesn't normally use them.
        """
        kind = type_info.kind  # "struct" / "union" / "enum"
        flat = type_info.qualified_name.replace("::", "_")
        return f"print_{kind}_{flat}"

    @staticmethod
    def c_type_expr(type_info):
        """Return the C type expression for *type_info*.

        ``struct Foo`` / ``union Foo`` / ``enum Foo`` — prefixed with the
        kind keyword so the code works regardless of whether a typedef
        exists. We use the struct tag (``spelling``) which is always
        present for non-anonymous types.
        """
        kind = type_info.kind
        name = type_info.spelling or type_info.qualified_name.split("::")[-1]
        return f"{kind} {name}"

    # -----------------------------------------------------------------
    # Record (struct / union) generation
    # -----------------------------------------------------------------

    def generate_for_record(self, type_info):
        """Generate a ``print_<kind>_<name>`` helper for a struct/union."""
        fname = self.helper_name(type_info)
        ctype = self.c_type_expr(type_info)
        members = type_info.member_variables or []
        short = type_info.qualified_name

        lines = [
            f"static inline void {fname}(FILE *out, const {ctype} *obj) {{",
            f"    if (!obj) {{ fprintf(out, \"{short}(null)\"); return; }}",
            f"    fprintf(out, \"{short}{{\");",
        ]

        first = True
        for member in members:
            sep = "" if first else ", "
            stmt = self._format_member(member, sep)
            if stmt is None:
                continue
            lines.append(stmt)
            first = False

        lines.append(f"    fprintf(out, \"}}\");")
        lines.append("}")
        return "\n".join(lines)

    def generate_for_enum(self, type_info):
        """Generate a ``print_enum_<name>`` helper for an enum."""
        fname = self.helper_name(type_info)
        ctype = self.c_type_expr(type_info)
        short = type_info.qualified_name
        return (
            f"static inline void {fname}(FILE *out, {ctype} val) {{\n"
            f"    fprintf(out, \"{short}(%d)\", (int)val);\n"
            f"}}"
        )

    # -----------------------------------------------------------------
    # Internal: per-member fprintf emission
    # -----------------------------------------------------------------

    def _format_member(self, member, sep):
        """Generate the fprintf statement(s) for a single member.

        Returns a multi-line C code string (indented with 4 spaces), or
        None if the member should be skipped entirely (e.g., nameless).
        """
        name = member.get("name", "")
        if not name:
            return None

        type_spelling = member.get("type_spelling", "")
        accessor = f"obj->{name}"

        # 1. Arrays — too complex to print generically.
        if member.get("is_array"):
            return (f'    fprintf(out, "{sep}{name}='
                    f'(array:{type_spelling})");')

        # 2. Bitfields — integral, always printable as signed long long.
        if member.get("is_bitfield"):
            return (f'    fprintf(out, "{sep}{name}=%lld", '
                    f'(long long)({accessor}));')

        # 3. Pointers.
        if member.get("is_pointer"):
            return self._format_pointer(member, sep, name, accessor)

        # 4. Primitives.
        spec = self.classifier.format_for_primitive(type_spelling)
        if spec is not None:
            fmt, cast = spec
            cast_expr = f"{cast}({accessor})" if cast else f"({accessor})"
            return f'    fprintf(out, "{sep}{name}={fmt}", {cast_expr});'

        # 5. User-defined value type with a registered helper.
        type_usr = member.get("type_usr", "")
        if type_usr and self.classifier.is_helper_generated(type_usr):
            ti = self.types.get(type_usr)
            if ti is not None:
                helper = self.helper_name(ti)
                if ti.kind == "enum":
                    return (
                        f'    fprintf(out, "{sep}{name}=");\n'
                        f"    {helper}(out, {accessor});"
                    )
                else:
                    return (
                        f'    fprintf(out, "{sep}{name}=");\n'
                        f"    {helper}(out, &({accessor}));"
                    )

        # 6. Enum fields via ``type_kind`` hint (TypeParser exposes this).
        if member.get("type_kind") == "ENUM":
            return (f'    fprintf(out, "{sep}{name}=%d", '
                    f'(int)({accessor}));')

        # 7. Default — non-printable.
        return (f'    fprintf(out, "{sep}{name}='
                f'(non-printable:{type_spelling})");')

    def _format_pointer(self, member, sep, name, accessor):
        """Generate fprintf for a pointer-typed member."""
        pointee = self.classifier.strip_cv(
            member.get("pointee_type_spelling", ""))

        # char* / signed char* — print as length-capped string, NULL-safe.
        if pointee in ("char", "signed char"):
            return (
                f'    fprintf(out, "{sep}{name}=%.200s", '
                f'({accessor}) ? ({accessor}) : "(null)");'
            )

        # Pointer to a user-defined instrumentable type.
        pointee_usr = (member.get("pointee_type_usr", "")
                       or member.get("type_usr", ""))
        if pointee_usr and self.classifier.is_helper_generated(pointee_usr):
            ti = self.types.get(pointee_usr)
            if ti is not None and ti.kind in ("struct", "union"):
                helper = CPrintCodeGenerator.helper_name(ti)
                return (
                    f'    if ({accessor}) {{\n'
                    f'        fprintf(out, "{sep}{name}=");\n'
                    f"        {helper}(out, {accessor});\n"
                    f"    }} else {{\n"
                    f'        fprintf(out, "{sep}{name}=(null)");\n'
                    f"    }}"
                )

        # Fallback — print address only.
        return (f'    fprintf(out, "{sep}{name}=%p", '
                f'(const void*)({accessor}));')


# ============================================================================
# CPrintInstrumentor (orchestrator)
# ============================================================================


class CPrintInstrumentor:
    """Top-level orchestrator. Takes a TypeParser instance and returns a
    plan dict matching ``OstreamInstrumentor.generate_instrumentation_plan``
    so the caller can reuse its injection loop.

    The caller is responsible for the compile-repair loop (mirrors the
    ``_generate_ostream_operators`` flow in ``diff_trace_analysis.py``).
    """

    _INSTRUMENTABLE_KINDS = frozenset({"struct", "union", "enum"})

    def __init__(self, type_parser):
        self.type_parser = type_parser
        self.types = type_parser.types  # USR -> TypeInfo

    # -----------------------------------------------------------------
    # Public API — parallel to OstreamInstrumentor
    # -----------------------------------------------------------------

    def generate_instrumentation_plan(self, func_list, output_path,
                                      function_types=None):
        """Generate an instrumentation plan as a dict; do not modify files.

        The returned dict has the same top-level shape as
        ``OstreamInstrumentor.generate_instrumentation_plan`` so the
        downstream injection loop in the preprocessor can be shared between
        backends (only the ``injection_type`` and ``include_code`` fields
        differ in semantics).
        """
        # --- Step 1: resolve target USRs.
        if function_types is not None:
            target_usrs, matched, total = self._usrs_from_function_types(
                function_types, len(func_list))
        else:
            target_usrs, matched, total = self._resolve_target_usrs(func_list)

        # --- Step 2: filter via _should_instrument.
        instrumentable_usrs = {
            u for u in target_usrs
            if self._should_instrument(self.types.get(u))
        }

        # --- Step 3: topological sort (Kahn's algorithm).
        ordered_usrs = self._topological_order(
            target_usrs=instrumentable_usrs)

        # --- Step 4: classify + generate code.
        classifier = CPrintabilityClassifier(self.types)
        generator = CPrintCodeGenerator(classifier, self.types)
        files_with_types = set()

        usr_to_name = {u: t.qualified_name
                       for u, t in self.types.items()}

        types_list = []
        for order_idx, usr in enumerate(ordered_usrs):
            ti = self.types[usr]
            classifier.register_generated(usr)

            if ti.kind in ("struct", "union"):
                code = generator.generate_for_record(ti)
                injection_type = "c_print_helper_after"
            elif ti.kind == "enum":
                code = generator.generate_for_enum(ti)
                injection_type = "c_print_helper_after"
            else:
                continue

            if ti.file_path:
                files_with_types.add(ti.file_path)

            dep_names = sorted(
                usr_to_name.get(d, d)
                for d in ti.depends_on
                if d in instrumentable_usrs
            )

            # Detect single-line definitions (e.g.
            # ``struct Foo { int x; };`` on one line) — the caller may
            # need to expand them before injection.
            needs_expand = (ti.start_line is not None
                            and ti.start_line == ti.end_line)

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
                "helper_name": CPrintCodeGenerator.helper_name(ti),
            })

        # --- Step 5: auxiliary info (C path uses <stdio.h>, no SFINAE).
        auxiliary = {
            "sfinae_helper_code": "",
            "files_needing_sfinae": {},
            "files_needing_include": sorted(files_with_types),
            "include_code": "#include <stdio.h>",
        }

        plan = {
            "source_functions": func_list,
            "types": types_list,
            "auxiliary": auxiliary,
            "total_types": len(types_list),
            "total_functions_matched": matched,
            "total_functions_provided": total,
        }

        print(
            f"CPrintInstrumentor: generated plan for {len(types_list)} "
            f"type(s) from {matched}/{total} function(s). "
            f"Plan -> {output_path}"
        )
        return plan

    # -----------------------------------------------------------------
    # Internal helpers (largely mirror OstreamInstrumentor)
    # -----------------------------------------------------------------

    def _should_instrument(self, type_info):
        """Return True if *type_info* should receive a print helper."""
        if type_info is None:
            return False
        if not type_info.is_definition:
            return False
        if type_info.kind not in self._INSTRUMENTABLE_KINDS:
            return False
        if not type_info.file_path:
            return False
        if not type_info.spelling:
            # Anonymous types have no tag name and no safe way to
            # reference them from a helper function signature.
            return False
        if type_info.macro_generated:
            return False
        return True

    def _usrs_from_function_types(self, function_types, total):
        """Derive target type USRs from a precomputed function_types dict.

        Mirrors ``OstreamInstrumentor._usrs_from_function_types``.
        """
        qname_to_usrs = defaultdict(list)
        for usr, ti in self.types.items():
            qname_to_usrs[ti.qualified_name].append(usr)

        target_usrs = set()
        matched = 0
        for file_funcs in function_types.values():
            matched += len(file_funcs)
            for func_info in file_funcs.values():
                for header_types in func_info.get("types", {}).values():
                    for qname in header_types:
                        target_usrs.update(qname_to_usrs.get(qname, []))

        return target_usrs, matched, total

    def _resolve_target_usrs(self, func_list):
        """Resolve function specs to the union of their transitive type USRs.

        Uses the TypeParser's lookup indexes and the same multi-level
        matching strategy as ``OstreamInstrumentor._resolve_target_usrs``.
        """
        tp = self.type_parser

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

            if func_file and func_line is not None:
                func_usr = by_file_line.get((func_file, func_line))

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

        target_usrs = tp._expand_types_transitively(all_direct_usrs)
        return target_usrs, matched, len(func_list)

    def _topological_order(self, target_usrs=None):
        """Kahn's algorithm — return USRs in dependency-respecting order."""
        in_degree = defaultdict(int)
        dependents = defaultdict(list)

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

        queue = [u for u in all_usrs if in_degree[u] == 0]
        ordered = []
        while queue:
            u = queue.pop(0)
            ordered.append(u)
            for v in dependents[u]:
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    queue.append(v)

        remaining = all_usrs - set(ordered)
        ordered.extend(sorted(remaining))
        return ordered
