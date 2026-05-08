#!/usr/bin/env python3
"""
Dynamic Trace Analysis for Root Cause Identification

Uses exact execution traces to find root causes of discrepancies
between native and WASM execution.

Key principle: A discrepancy is a ROOT CAUSE if it has no dependencies
on earlier discrepancies in the execution trace.
"""

import re
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class DiscrepancyType(Enum):
    """Types of discrepancies."""
    EXECUTION_COUNT = "execution_count"  # Different number of executions
    VALUE_MISMATCH = "value_mismatch"    # Same count, different values
    MISSING_IN_NATIVE = "missing_in_native"  # Only in WASM
    MISSING_IN_WASM = "missing_in_wasm"    # Only in native


@dataclass
class ExecutionEvent:
    """Single execution event in the trace."""
    position: int           # Sequential position in execution (0, 1, 2, ...)
    marker: str            # Instrumentation marker (e.g., "ef1f889d")
    file_path: str         # Source file
    line: int              # Line number
    function_usr: str      # Function USR (from call graph)
    values: List[str]      # Captured variable values

    def location_key(self) -> str:
        """Unique key for this location."""
        return f"{self.file_path}:{self.line}"


@dataclass
class Discrepancy:
    """A detected discrepancy between native and WASM."""
    location_key: str      # "file:line"
    file_path: str
    line: int
    function_usr: str
    function_name: str
    discrepancy_type: DiscrepancyType
    position: int          # Position in trace where first detected
    native_summary: str    # Summary of native behavior
    wasm_summary: str      # Summary of WASM behavior
    is_root_cause: bool = False  # Determined by analysis
    depends_on: Set[str] = None  # Location keys this depends on

    def __post_init__(self):
        if self.depends_on is None:
            self.depends_on = set()


class ExecutionTraceParser:
    """Parses execution logs to extract ordered execution events."""

    def __init__(self, marker_map: Dict[str, Dict]):
        """
        Initialize parser with marker mapping.

        Args:
            marker_map: Dict mapping marker -> {file, line, variables}
                       (from instrumentation_report.json)
        """
        self.marker_map = marker_map

    def parse_log(self, log_file_path: str, filter_test_block: bool = True) -> List[ExecutionEvent]:
        """
        Parse execution log to get ordered sequence of events.

        Args:
            log_file_path: Path to execution_output_*.log
            filter_test_block: If True, only parse events within @@TEST BLOCK START@@
                              and @@TEST BLOCK END@@ markers

        Returns:
            List of ExecutionEvent in execution order
        """
        with open(log_file_path, 'r') as f:
            full_content = f.read()

        # If filter_test_block is enabled, extract only content within test block
        if filter_test_block:
            test_start_marker = "@@TEST BLOCK START@@" # TODO: Shall we consider also adding an identifier?
            test_end_marker = "@@TEST BLOCK END@@"

            start_idx = full_content.find(test_start_marker)
            end_idx = full_content.find(test_end_marker)

            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                # Extract content between markers
                content = full_content[start_idx + len(test_start_marker):end_idx]
                print(f"    Filtering to test block ({end_idx - start_idx} chars)")
            else:
                # No test block markers found, use full content
                print(f"    No test block markers found, using full log")
                content = full_content
        else:
            content = full_content

        # Extract all instrumentation blocks in order
        pattern = r'@@INST_START_([a-f0-9]{8})@@(.*?)@@INST_END_\1@@'
        matches = re.finditer(pattern, content, re.DOTALL)

        events = []
        for position, match in enumerate(matches):
            marker = match.group(1)
            block_content = match.group(2).strip()

            # Look up marker info
            if marker not in self.marker_map:
                continue

            info = self.marker_map[marker]

            # Parse values
            values = [line.strip() for line in block_content.split('\n') if line.strip()]

            event = ExecutionEvent(
                position=position,
                marker=marker,
                file_path=info['file'],
                line=info['line'],
                function_usr=info.get('usr', 'unknown'),  # Will be filled later
                values=values
            )

            events.append(event)

        return events


class DependencyAnalyzer:
    """Analyzes dependencies between execution locations using call graph."""

    def __init__(self, call_graph: Dict):
        """
        Initialize with call graph.

        Args:
            call_graph: Call graph data with 'functions' and 'call_edges'
        """
        self.functions = call_graph.get('functions', {})
        self.call_edges = {
            usr: set(callees) for usr, callees in call_graph.get('call_edges', {}).items()
        }
        self.reverse_call_edges = {
            usr: set(callers) for usr, callers in call_graph.get('reverse_call_edges', {}).items()
        }

    def get_function_for_location(self, file_path: str, line: int) -> Optional[str]:
        """
        Find the function USR containing the given location.

        Args:
            file_path: Source file path
            line: Line number

        Returns:
            Function USR or None
        """
        import os
        file_path_abs = os.path.abspath(file_path)

        for usr, func_info in self.functions.items():
            func_file = func_info.get('file', '')
            if not func_file:
                continue

            func_file_abs = os.path.abspath(func_file)
            if func_file_abs != file_path_abs:
                continue

            func_start = func_info.get('line')
            func_end = func_info.get('end_line')

            if func_start is None:
                continue

            # Check if line is in function range
            if func_end is None:
                # Fallback: assume any line >= start is in function
                if line >= func_start:
                    return usr
            elif func_start <= line <= func_end:
                return usr

        return None

    def get_function_name(self, usr: str) -> str:
        """Get function name from USR."""
        return self.functions.get(usr, {}).get('name', 'unknown')

    def calls(self, caller_usr: str, callee_usr: str) -> bool:
        """Check if caller calls callee (directly)."""
        callees = self.call_edges.get(caller_usr, set())
        return callee_usr in callees

    def calls_transitively(self, caller_usr: str, callee_usr: str) -> bool:
        """Check if caller calls callee (transitively)."""
        visited = set()
        stack = [caller_usr]

        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)

            if current == callee_usr:
                return True

            callees = self.call_edges.get(current, set())
            stack.extend(callees)

        return False

    def is_called_by(self, callee_usr: str, caller_usr: str) -> bool:
        """Check if callee is called by caller."""
        return self.calls(caller_usr, callee_usr)


class RootCauseAnalyzer:
    """
    Main analyzer: finds root causes using position-wise trace alignment.

    THEORETICAL BASIS:
    - Execution traces are temporal sequences of events
    - Causality respects temporal order: event at position p1 can only
      affect events at positions p2 > p1
    - The FIRST divergence point in execution order is the root cause
    - All subsequent divergences are symptoms (consequences)
    """

    def __init__(self, call_graph: Dict, marker_map: Dict[str, Dict]):
        """
        Initialize analyzer.

        Args:
            call_graph: Call graph JSON data
            marker_map: Marker mapping from instrumentation report
        """
        self.dependency_analyzer = DependencyAnalyzer(call_graph)
        self.trace_parser = ExecutionTraceParser(marker_map)

    def analyze(
        self,
        native_log_path: str,
        wasm_log_path: str
    ) -> Tuple[List[Discrepancy], List[Discrepancy]]:
        """
        Analyze traces to find root causes using position-wise alignment.

        ALGORITHM:
        1. Parse execution traces (ordered sequences of events)
        2. Align traces position-by-position
        3. Find first divergence point → ROOT CAUSE
        4. All subsequent differences → SYMPTOMS

        Args:
            native_log_path: Path to execution_output_native.log
            wasm_log_path: Path to execution_output_wasm.log

        Returns:
            Tuple of (root_causes, symptoms)
        """
        # Step 1: Parse both traces
        print("Parsing execution traces...")
        native_trace = self.trace_parser.parse_log(native_log_path)
        wasm_trace = self.trace_parser.parse_log(wasm_log_path)

        # Enrich with function USRs
        self._enrich_with_function_usrs(native_trace)
        self._enrich_with_function_usrs(wasm_trace)

        print(f"  Native trace: {len(native_trace)} events")
        print(f"  WASM trace: {len(wasm_trace)} events")

        # Step 2: Find root cause by position-wise alignment
        print("\nAnalyzing trace divergence...")
        root_cause, symptoms = self._find_divergence_point(native_trace, wasm_trace)

        if root_cause:
            print(f"  Root cause found at position {root_cause.position}")
            print(f"  Symptoms: {len(symptoms)} subsequent divergences")
        else:
            print("  No divergence found - traces are identical")

        return ([root_cause] if root_cause else [], symptoms)

    def _enrich_with_function_usrs(self, trace: List[ExecutionEvent]):
        """Add function USR information to events."""
        for event in trace:
            usr = self.dependency_analyzer.get_function_for_location(
                event.file_path,
                event.line
            )
            event.function_usr = usr if usr else 'unknown'

    def _infer_dynamic_call_chain(self, trace: List[ExecutionEvent]) -> List[Tuple[str, str, int]]:
        """
        Infer actual call relationships from execution trace.

        Uses execution order + call graph validation to build the actual
        call chain that occurred during execution.

        Returns:
            List of (caller_usr, callee_usr, position) tuples
        """
        call_chain = []
        prev_func_usr = None

        # Also collect all functions that actually executed
        executed_functions = set()

        for event in trace:
            func_usr = event.function_usr

            if func_usr != 'unknown':
                executed_functions.add(func_usr)

            if func_usr != prev_func_usr and func_usr != 'unknown':
                if prev_func_usr is not None and prev_func_usr != 'unknown':
                    # Transition: prev_func_usr → func_usr
                    # Verify this is a valid call in the call graph
                    if self.dependency_analyzer.calls(prev_func_usr, func_usr):
                        call_chain.append((prev_func_usr, func_usr, event.position))

                prev_func_usr = func_usr

        print(f"    Dynamic call chain: {len(call_chain)} transitions")
        print(f"    Functions executed: {len(executed_functions)}")

        return call_chain

    def _find_caller_function(self, trace: List[ExecutionEvent], last_event: ExecutionEvent) -> Optional[str]:
        """
        Find the function that called the function at last_event.

        Strategy:
        1. Build call chain from trace
        2. Find transition where callee is last_event's function
        3. Return the caller

        Args:
            trace: Execution trace
            last_event: The last event in the trace

        Returns:
            Caller function USR or None
        """
        call_chain = self._infer_dynamic_call_chain(trace)
        last_func_usr = last_event.function_usr

        # Find the most recent call to last_func_usr
        for (caller, callee, pos) in reversed(call_chain):
            if callee == last_func_usr:
                return caller

        # Fallback: use static call graph to find potential callers
        # among executed functions
        executed_functions = {event.function_usr for event in trace if event.function_usr != 'unknown'}

        for exec_func in executed_functions:
            if exec_func == last_func_usr:
                continue
            # Check if exec_func calls last_func_usr
            if self.dependency_analyzer.calls(exec_func, last_func_usr):
                return exec_func

        return None

    def _backward_slice_from_termination(
        self,
        trace: List[ExecutionEvent],
        termination_func_usr: str
    ) -> Set[str]:
        """
        Perform backward slice to find functions relevant to termination.

        Starting from the termination point, walk backward through the
        dynamic call chain to find all functions that led to termination.

        FALLBACK: If dynamic call chain is empty/insufficient, use static
        call graph within executed functions.

        Args:
            trace: Execution trace
            termination_func_usr: Function where trace terminated

        Returns:
            Set of function USRs in the execution path to termination
        """
        # Build dynamic call chain
        call_chain = self._infer_dynamic_call_chain(trace)

        # Collect all functions that actually executed
        executed_functions = {event.function_usr for event in trace if event.function_usr != 'unknown'}

        # Start with termination function
        relevant_functions = {termination_func_usr}

        # Walk backward through dynamic call chain
        for (caller, callee, pos) in reversed(call_chain):
            if callee in relevant_functions:
                relevant_functions.add(caller)

        # FALLBACK: If we didn't find many relevant functions via dynamic chain,
        # use static backward slice within executed functions
        if len(relevant_functions) < 3 and len(executed_functions) > 3:
            print(f"    Dynamic call chain incomplete, using static call graph fallback")

            # Find all executed functions that are transitively called by termination function
            # (i.e., termination_func_usr is the caller, exec_func is the callee)
            for exec_func in executed_functions:
                if exec_func == termination_func_usr:
                    continue
                # Check if termination function transitively calls exec_func
                # (backward slice: include callees of termination function)
                if self.dependency_analyzer.calls_transitively(termination_func_usr, exec_func):
                    relevant_functions.add(exec_func)

        print(f"  Backward slice from {self.dependency_analyzer.get_function_name(termination_func_usr)}:")
        print(f"    Found {len(relevant_functions)} relevant functions in execution path")

        return relevant_functions

    def _is_pointer_value_difference(self, divergence: Discrepancy) -> bool:
        """
        Heuristic to detect if divergence is likely a benign pointer difference.

        Checks:
        - Values look like hexadecimal addresses (0x...)
        - Variable names suggest pointers (ptr, addr, in, etc.)

        Returns:
            True if likely a benign pointer difference
        """
        if divergence.discrepancy_type != DiscrepancyType.VALUE_MISMATCH:
            return False

        # Check if values look like pointers
        native_val = divergence.native_summary
        wasm_val = divergence.wasm_summary

        # Simple heuristic: both values are hex numbers
        hex_pattern = r'0x[0-9a-fA-F]+'
        import re

        native_has_hex = re.search(hex_pattern, native_val) is not None
        wasm_has_hex = re.search(hex_pattern, wasm_val) is not None

        if native_has_hex and wasm_has_hex:
            # Both look like addresses
            return True

        return False

    def _find_divergence_point(
        self,
        native_trace: List[ExecutionEvent],
        wasm_trace: List[ExecutionEvent]
    ) -> Tuple[Optional[Discrepancy], List[Discrepancy]]:
        """
        Find the first RELEVANT PROBLEMATIC divergence using backward slice.

        STRATEGY:
        1. Detect early termination (structural divergence)
        2. Check if last position has divergence
           - If YES: that's the root cause
           - If NO: bug is in CALLER function (after successful return)
        3. Perform backward slice from root cause to find relevant functions
        4. Search for first divergence in relevant functions only
        5. Deprioritize benign pointer differences

        Returns:
            Tuple of (root_cause, symptoms)
            - root_cause: First relevant problematic divergence
            - symptoms: Other divergences (in and out of relevant functions)
        """
        min_len = min(len(native_trace), len(wasm_trace))

        # Phase 1: Check for early termination and identify root cause location
        relevant_functions = None
        termination_divergence = None

        if len(native_trace) != len(wasm_trace):
            # Early termination detected
            last_pos = min_len - 1

            # CRITICAL: Check if last position has a divergence
            last_divergence = None
            if last_pos >= 0:
                last_divergence = self._check_divergence(
                    native_trace[last_pos],
                    wasm_trace[last_pos],
                    last_pos
                )

            if last_divergence and not self._is_pointer_value_difference(last_divergence):
                # Last position HAS a real divergence → that's where the bug is
                print(f"  → Early termination with divergence at position {last_pos}")
                print(f"     {last_divergence.location_key}")
                termination_divergence = last_divergence
                termination_func = termination_divergence.function_usr
            else:
                # Last position has NO divergence (or only benign pointer difference)
                # → Bug is in the CALLER, after this call returned successfully
                last_event = wasm_trace[last_pos] if last_pos >= 0 else wasm_trace[-1]
                caller_func = self._find_caller_function(wasm_trace, last_event)

                print(f"  → Early termination at position {last_pos}")
                print(f"     Last successful call: {last_event.location_key()}")
                print(f"     No divergence in last call → bug is in CALLER after return")

                if caller_func:
                    caller_name = self.dependency_analyzer.get_function_name(caller_func)
                    print(f"     Caller function: {caller_name}")

                    # Create discrepancy for the caller
                    termination_divergence = Discrepancy(
                        location_key=f"{caller_name} (caller, after {last_event.location_key()})",
                        file_path=last_event.file_path,
                        line=last_event.line,
                        function_usr=caller_func,
                        function_name=caller_name,
                        discrepancy_type=DiscrepancyType.MISSING_IN_WASM,
                        position=last_pos,
                        native_summary=f"{len(native_trace)} total executions",
                        wasm_summary=f"{len(wasm_trace)} total executions (terminated in {caller_name} after {last_event.location_key()} returned)"
                    )
                    termination_func = caller_func
                else:
                    # Fallback: use the last executed function
                    print(f"     Could not identify caller, using last executed function")
                    termination_divergence = self._handle_early_termination(native_trace, wasm_trace, min_len)
                    termination_func = termination_divergence.function_usr

            # Perform backward slice to find relevant functions
            relevant_functions = self._backward_slice_from_termination(wasm_trace, termination_func)

        # Phase 2: Look for first relevant divergence (within relevant functions)
        first_relevant_divergence = None
        all_divergences = []

        for pos in range(min_len):
            n_event = native_trace[pos]
            w_event = wasm_trace[pos]

            divergence = self._check_divergence(n_event, w_event, pos)

            if divergence:
                all_divergences.append(divergence)

                # Check if this divergence is in a relevant function
                is_relevant = (relevant_functions is None or
                              divergence.function_usr in relevant_functions)

                # Skip benign pointer differences (unless no other option)
                is_likely_benign = self._is_pointer_value_difference(divergence)

                if is_relevant and not is_likely_benign:
                    if first_relevant_divergence is None:
                        first_relevant_divergence = divergence
                        print(f"  → Found first relevant divergence at position {pos}")
                        print(f"     {divergence.location_key} in {self.dependency_analyzer.get_function_name(divergence.function_usr)}")

        # Phase 3: Determine root cause
        if first_relevant_divergence:
            # Found a non-pointer divergence in relevant functions
            first_relevant_divergence.is_root_cause = True
            root_cause = first_relevant_divergence

            # Mark others as symptoms
            symptoms = [d for d in all_divergences if d != root_cause]
            for s in symptoms:
                s.is_root_cause = False

            # Add termination as symptom if it exists
            if termination_divergence:
                termination_divergence.is_root_cause = False
                symptoms.append(termination_divergence)

        elif termination_divergence:
            # No other relevant divergence found, termination itself is root cause
            termination_divergence.is_root_cause = True
            root_cause = termination_divergence

            # All other divergences are symptoms
            symptoms = all_divergences
            for s in symptoms:
                s.is_root_cause = False

        else:
            # No termination, check for control flow or value divergences
            # (This is the case when traces are same length)
            return self._find_divergence_same_length_traces(native_trace, wasm_trace, min_len)

        return root_cause, symptoms

    def _find_divergence_same_length_traces(
        self,
        native_trace: List[ExecutionEvent],
        wasm_trace: List[ExecutionEvent],
        min_len: int
    ) -> Tuple[Optional[Discrepancy], List[Discrepancy]]:
        """
        Handle case where traces have same length but different values/control flow.

        This handles cases without early termination.
        """
        first_control_divergence = None
        first_value_divergence = None
        all_divergences = []

        for pos in range(min_len):
            n_event = native_trace[pos]
            w_event = wasm_trace[pos]

            divergence = self._check_divergence(n_event, w_event, pos)

            if divergence:
                all_divergences.append(divergence)

                # Prioritize control flow divergence over value divergence
                if divergence.discrepancy_type == DiscrepancyType.EXECUTION_COUNT:
                    # Control flow divergence (different locations at same position)
                    if first_control_divergence is None:
                        first_control_divergence = divergence
                else:
                    # Value divergence - skip if benign pointer
                    if not self._is_pointer_value_difference(divergence):
                        if first_value_divergence is None:
                            first_value_divergence = divergence

        # Choose root cause: prefer control flow divergence over value divergence
        if first_control_divergence:
            first_control_divergence.is_root_cause = True
            print(f"  → Root cause at position {first_control_divergence.position}: Control flow divergence")
            print(f"     {first_control_divergence.location_key}")
            symptoms = [d for d in all_divergences if d != first_control_divergence]
            for s in symptoms:
                s.is_root_cause = False
            return first_control_divergence, symptoms

        elif first_value_divergence:
            first_value_divergence.is_root_cause = True
            print(f"  → Root cause at position {first_value_divergence.position}: Value divergence")
            print(f"     {first_value_divergence.location_key}")
            symptoms = [d for d in all_divergences if d != first_value_divergence]
            for s in symptoms:
                s.is_root_cause = False
            return first_value_divergence, symptoms

        # No divergence found
        return None, []

    def _collect_all_value_divergences(
        self,
        native_trace: List[ExecutionEvent],
        wasm_trace: List[ExecutionEvent],
        min_len: int,
        skip_pos: Optional[int] = None
    ) -> List[Discrepancy]:
        """Collect all value divergences as symptoms (excluding skip_pos if specified)."""
        symptoms = []
        for pos in range(min_len):
            if skip_pos is not None and pos == skip_pos:
                continue

            n_event = native_trace[pos]
            w_event = wasm_trace[pos]
            divergence = self._check_divergence(n_event, w_event, pos)

            if divergence:
                divergence.is_root_cause = False
                symptoms.append(divergence)

        return symptoms

    def _check_divergence(
        self,
        n_event: ExecutionEvent,
        w_event: ExecutionEvent,
        position: int
    ) -> Optional[Discrepancy]:
        """
        Check if two events at the same position diverge.

        Returns:
            Discrepancy if events differ, None if identical
        """
        # Check 1: Control flow divergence (different locations)
        if n_event.location_key() != w_event.location_key():
            return Discrepancy(
                location_key=f"{n_event.location_key()} vs {w_event.location_key()}",
                file_path=n_event.file_path,
                line=n_event.line,
                function_usr=n_event.function_usr,
                function_name=self.dependency_analyzer.get_function_name(n_event.function_usr),
                discrepancy_type=DiscrepancyType.EXECUTION_COUNT,  # Control flow difference
                position=position,
                native_summary=f"At {n_event.location_key()}",
                wasm_summary=f"At {w_event.location_key()}"
            )

        # Check 2: Value divergence (same location, different values)
        if n_event.values != w_event.values:
            return Discrepancy(
                location_key=n_event.location_key(),
                file_path=n_event.file_path,
                line=n_event.line,
                function_usr=n_event.function_usr,
                function_name=self.dependency_analyzer.get_function_name(n_event.function_usr),
                discrepancy_type=DiscrepancyType.VALUE_MISMATCH,
                position=position,
                native_summary=" | ".join(n_event.values) if n_event.values else "(no values)",
                wasm_summary=" | ".join(w_event.values) if w_event.values else "(no values)"
            )

        # No divergence
        return None

    def _handle_early_termination(
        self,
        native_trace: List[ExecutionEvent],
        wasm_trace: List[ExecutionEvent],
        aligned_len: int
    ) -> Optional[Discrepancy]:
        """
        Handle case where one trace terminates early.

        STRATEGY:
        - If WASM trace is shorter: last event in WASM likely caused exception/exit
        - If native trace is shorter: unusual, but same logic applies

        Returns:
            Discrepancy representing the termination point
        """
        if len(native_trace) == len(wasm_trace):
            return None

        # Determine which trace is shorter
        shorter_trace = wasm_trace if len(wasm_trace) < len(native_trace) else native_trace
        longer_trace = native_trace if len(native_trace) > len(wasm_trace) else wasm_trace

        is_wasm_shorter = len(wasm_trace) < len(native_trace)

        # The termination point
        termination_pos = len(shorter_trace) - 1

        if termination_pos < 0:
            # One trace is completely empty
            return None

        last_event = shorter_trace[termination_pos]

        # Create discrepancy for early termination
        return Discrepancy(
            location_key=last_event.location_key(),
            file_path=last_event.file_path,
            line=last_event.line,
            function_usr=last_event.function_usr,
            function_name=self.dependency_analyzer.get_function_name(last_event.function_usr),
            discrepancy_type=DiscrepancyType.MISSING_IN_WASM if is_wasm_shorter else DiscrepancyType.MISSING_IN_NATIVE,
            position=termination_pos,
            native_summary=f"{len(native_trace)} total executions",
            wasm_summary=f"{len(wasm_trace)} total executions (terminated early)"
        )


    def print_analysis_report(self, root_causes: List[Discrepancy], symptoms: List[Discrepancy]):
        """Print human-readable analysis report."""
        import os

        print("\n" + "="*80)
        print("ROOT CAUSE ANALYSIS REPORT")
        print("="*80)
        print("\nTHEORETICAL BASIS:")
        print("  - Execution traces are temporal sequences")
        print("  - First divergence point = root cause (by causality)")
        print("  - All subsequent divergences = symptoms")
        print("="*80)

        print(f"\nTotal discrepancies detected: {len(root_causes) + len(symptoms)}")
        print(f"  Root causes: {len(root_causes)}")
        print(f"  Symptoms: {len(symptoms)}")

        if root_causes:
            print("\n" + "="*80)
            print("ROOT CAUSE (First divergence in execution)")
            print("="*80)

            for disc in root_causes:
                print(f"\n✗ {disc.function_name}")
                print(f"   Location: {os.path.basename(disc.file_path)}:{disc.line}")
                print(f"   Position in trace: {disc.position}")
                print(f"   Divergence type: {disc.discrepancy_type.value}")
                print(f"   Native: {disc.native_summary}")
                print(f"   WASM: {disc.wasm_summary}")
                print(f"\n   → This is the FIRST point where traces diverge")
                print(f"   → All subsequent differences are consequences of this")

        if symptoms:
            print("\n" + "="*80)
            print("SYMPTOMS (Subsequent divergences, caused by root cause)")
            print("="*80)

            for i, disc in enumerate(symptoms, 1):
                print(f"\n{i}. {disc.function_name}")
                print(f"   Location: {os.path.basename(disc.file_path)}:{disc.line}")
                print(f"   Position in trace: {disc.position}")
                print(f"   Type: {disc.discrepancy_type.value}")
