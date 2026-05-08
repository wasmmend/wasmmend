"""
Multi-file instrumentation coordinator
Handles complex scenarios with headers, templates, and cross-file dependencies
"""

import os
import json
from typing import Dict, List, Tuple, Set
from collections import defaultdict


class InstrumentationCoordinator:
    """
    Coordinates instrumentation across multiple files to ensure:
    1. No duplicate instrumentation
    2. Correct handling of headers included by multiple sources
    3. Dependency tracking across file boundaries
    4. Conflict resolution for shared code
    """

    def __init__(self):
        self.instrumented_locations = set()  # (file_path, line, position) tuples
        self.file_dependencies = defaultdict(set)  # file -> set of files it depends on
        self.header_includers = defaultdict(set)  # header -> set of source files including it
        self.instrumentation_plan = []  # List of (file, line, instrumentation, position) tuples

    def register_dependency(self, source_file: str, depends_on_file: str):
        """Register that source_file depends on depends_on_file"""
        self.file_dependencies[source_file].add(depends_on_file)

        # Track header inclusions
        if depends_on_file.endswith(('.h', '.hpp', '.hxx')):
            self.header_includers[depends_on_file].add(source_file)

    def is_header_file(self, file_path: str) -> bool:
        """Check if file is a header"""
        return file_path.endswith(('.h', '.hpp', '.hxx', '.hh'))

    def can_instrument(self, file_path: str, line: int, position: str = "after") -> bool:
        """
        Check if a location can be instrumented
        Returns False if already instrumented
        """
        key = (os.path.abspath(file_path), line, position)
        return key not in self.instrumented_locations

    # Directories that must never be instrumented.  These are either
    # compiler build outputs (build*/) or internal snapshots kept by the
    # pipeline itself (.pre_instrumentation_originals/,
    # .instrumentation_backups/, instrumented_files/).  Headers in these
    # dirs can get pulled in if a sibling copy of a real source tree
    # lives next to the project (common after a previous run), and
    # letting them through would corrupt the snapshot and duplicate
    # instrumentation.
    _FORBIDDEN_DIR_MARKERS = (
        os.sep + 'build_native' + os.sep,
        os.sep + 'build_wasm' + os.sep,
        os.sep + 'build' + os.sep,
        os.sep + '.pre_instrumentation_originals' + os.sep,
        os.sep + '.instrumentation_backups' + os.sep,
        os.sep + 'instrumented_files' + os.sep,
    )

    @classmethod
    def _is_forbidden_path(cls, file_path: str) -> bool:
        ap = os.path.abspath(file_path) + os.sep
        return any(m in ap for m in cls._FORBIDDEN_DIR_MARKERS)

    def register_instrumentation(self, file_path: str, line: int, instrumentation: str, position: str = "after"):
        """
        Register an instrumentation point.
        Deduplicates and tracks what has been instrumented.

        Args:
            position: "before" to insert before the statement, "after" to insert after.
        """
        file_path = os.path.abspath(file_path)

        if self._is_forbidden_path(file_path):
            # Silent refusal: these paths come from stale snapshots or
            # build outputs; they should never be instrumented.
            return False

        key = (file_path, line, position)

        if key in self.instrumented_locations:
            print(f"  Skipping duplicate instrumentation: {file_path}:{line} ({position})")
            return False

        self.instrumented_locations.add(key)
        self.instrumentation_plan.append((file_path, line, instrumentation, position))
        return True

    def get_instrumentation_strategy(self, file_path: str) -> str:
        """
        Determine the best instrumentation strategy for a file
        - Header with single includer: instrument in the header
        - Header with multiple includers: warn, instrument carefully
        - Source file: instrument normally
        """
        file_path = os.path.abspath(file_path)

        if not self.is_header_file(file_path):
            return "source_file"

        includer_count = len(self.header_includers.get(file_path, set()))

        if includer_count == 0:
            return "orphan_header"  # Header not included by any analyzed source
        elif includer_count == 1:
            return "single_includer_header"
        else:
            return "multi_includer_header"

    def resolve_header_conflicts(self) -> Dict[str, str]:
        """
        Resolve conflicts for headers included by multiple source files
        Returns strategy for each file
        """
        strategies = {}

        for file_path in self.file_dependencies.keys():
            strategy = self.get_instrumentation_strategy(file_path)
            strategies[file_path] = strategy

            if strategy == "multi_includer_header":
                includers = self.header_includers[file_path]
                print(f"  Warning: {file_path} is included by multiple sources:")
                for includer in includers:
                    print(f"    - {includer}")
                print(f"    Will instrument once in the header file")

        return strategies

    def group_by_file(self) -> Dict[str, List[Tuple[int, str, str]]]:
        """
        Group instrumentation points by file
        Returns: {file_path: [(line, instrumentation, position), ...]}
        """
        grouped = defaultdict(list)

        for file_path, line, instrumentation, position in self.instrumentation_plan:
            grouped[file_path].append((line, instrumentation, position))

        # Sort by line number (reverse) for bottom-up insertion.
        # For the same line, process "after" before "before" so that the
        # before-block ends up above the after-block in the final source.
        for file_path in grouped:
            grouped[file_path].sort(
                key=lambda x: (x[0], 0 if x[2] == "after" else -1),
                reverse=True
            )

        return dict(grouped)

    def export_report(self, output_path: str):
        """Export instrumentation report for analysis"""
        report = {
            "total_locations": len(self.instrumented_locations),
            "files_modified": len(set(fp for fp, _, _, _ in self.instrumentation_plan)),
            "header_files": {},
            "source_files": {},
            "instrumentation_points": []
        }

        # Categorize files
        for file_path, line, instr, position in self.instrumentation_plan:
            point = {
                "file": file_path,
                "line": line,
                "position": position,
                "instrumentation": instr
            }
            report["instrumentation_points"].append(point)

            if self.is_header_file(file_path):
                if file_path not in report["header_files"]:
                    report["header_files"][file_path] = {
                        "includers": list(self.header_includers.get(file_path, set())),
                        "points": []
                    }
                report["header_files"][file_path]["points"].append(line)
            else:
                if file_path not in report["source_files"]:
                    report["source_files"][file_path] = {"points": []}
                report["source_files"][file_path]["points"].append(line)

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

        print(f"\nInstrumentation report exported to: {output_path}")
        print(f"  Total locations: {report['total_locations']}")
        print(f"  Files modified: {report['files_modified']}")
        print(f"  Header files: {len(report['header_files'])}")
        print(f"  Source files: {len(report['source_files'])}")

    def get_summary(self) -> str:
        """Get a human-readable summary of the instrumentation plan"""
        summary = []
        summary.append("="*60)
        summary.append("INSTRUMENTATION SUMMARY")
        summary.append("="*60)

        grouped = self.group_by_file()

        summary.append(f"\nTotal files to modify: {len(grouped)}")
        summary.append(f"Total instrumentation points: {len(self.instrumented_locations)}")

        summary.append("\nFiles by type:")
        headers = [f for f in grouped.keys() if self.is_header_file(f)]
        sources = [f for f in grouped.keys() if not self.is_header_file(f)]

        summary.append(f"  Headers: {len(headers)}")
        for h in headers:
            includers = len(self.header_includers.get(h, set()))
            summary.append(f"    - {h} (included by {includers} file(s))")

        summary.append(f"  Sources: {len(sources)}")
        for s in sources:
            summary.append(f"    - {s}")

        summary.append("\n" + "="*60)

        return "\n".join(summary)
