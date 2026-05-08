import os
import json
from clang import cindex
# cindex.Config.set_library_file("/usr/lib/llvm-15/lib/libclang.so")
cindex.Config.set_library_file("/usr/lib/llvm-15/lib/libclang-15.so.1")
from collections import defaultdict, deque
import json

# Set to True to enable verbose debug prints
DATA_ANALYZER_DEBUG = False

def _debug_print(*args, **kwargs):
    if DATA_ANALYZER_DEBUG:
        _debug_print(*args, **kwargs)

# --- CFG classes for explicit control-flow graph construction ---

class CFGNode:
    def __init__(self, id):
        self.id = id
        self.statements = []       # List of AST cursors (statements) in this block.
        self.successors = []       # List of successor CFGNode objects.
        self.predecessors = []     # List of predecessor CFGNode objects.

    def __repr__(self):
        return f"CFGNode({self.id}, stmts={len(self.statements)}, succ={[n.id for n in self.successors]}, pred={[n.id for n in self.predecessors]})"

class CFGBuilder:
    def __init__(self, func_cursor):
        self.func_cursor = func_cursor
        self.nodes = []
        self.next_id = 0

    def new_node(self):
        node = CFGNode(self.next_id)
        self.nodes.append(node)
        self.next_id += 1
        return node

    def build(self):
        body = None
        initializers = []

        # Collect both member initializers and the function body
        children = list(self.func_cursor.get_children())
        for i, child in enumerate(children):
            if child.kind == cindex.CursorKind.COMPOUND_STMT:
                body = child
                break  # Stop collecting initializers after finding body

            # Collect constructor member/base initializers
            elif child.kind == cindex.CursorKind.MEMBER_REF:
                # MEMBER_REF represents a member being initialized
                # The next sibling will be its value expression
                initializers.append(child)
                # Also append the value expression so we can process them together
                if i + 1 < len(children):
                    initializers.append(children[i + 1])

            elif child.kind == cindex.CursorKind.CALL_EXPR:
                # CALL_EXPR before COMPOUND_STMT is a base class constructor call
                # Check if this is NOT already added as part of a MEMBER_REF
                if i > 0 and children[i - 1].kind != cindex.CursorKind.MEMBER_REF:
                    # This is a base class constructor, add it as a statement
                    initializers.append(child)

            elif child.kind == cindex.CursorKind.TYPE_REF:
                # TYPE_REF represents a base class or delegating constructor call
                # This is how base class constructors appear in initializer lists
                initializers.append(child)
                # Also collect the next child which contains the arguments
                if i + 1 < len(children):
                    next_child = children[i + 1]
                    # The next child could be UNEXPOSED_EXPR containing arguments
                    if next_child.kind != cindex.CursorKind.COMPOUND_STMT:
                        initializers.append(next_child)

        # If neither body nor initializers exist, return None
        if not body and not initializers:
            return None

        entry = self.new_node()

        # Add initializers as statements in the entry node
        for init in initializers:
            entry.statements.append(init)

        # Process body if it exists
        if body:
            self._build_from_stmt(body, entry)

        return entry

    def _build_from_stmt(self, stmt, current_node):
        """
        Build CFG from a statement, handling control flow constructs.

        Now supports:
        - COMPOUND_STMT (blocks)
        - IF_STMT (if/else branches) with proper return value handling
        - FOR_STMT (for loops with back edges) with proper return value handling
        - WHILE_STMT (while loops with back edges) with proper return value handling
        - DO_STMT (do-while loops with back edges)
        - SWITCH_STMT (switch/case statements)
        - Other statements (treated as straight-line code)

        Returns:
            The final node after processing all statements (important for nested control flow)
        """
        if stmt.kind == cindex.CursorKind.COMPOUND_STMT:
            for child in stmt.get_children():

                # Handle IF statements
                if child.kind == cindex.CursorKind.IF_STMT:
                    if_node = self.new_node()
                    if_node.statements.append(child)
                    current_node.successors.append(if_node)
                    if_node.predecessors.append(current_node)

                    children_of_if = list(child.get_children())
                    then_branch = children_of_if[1] if len(children_of_if) >= 2 else None
                    else_branch = children_of_if[2] if len(children_of_if) >= 3 else None

                    # Process then branch and capture final node
                    then_node = self.new_node()
                    if then_branch:
                        final_then_node = self._build_from_stmt(then_branch, then_node)
                    else:
                        final_then_node = then_node
                    if_node.successors.append(then_node)
                    then_node.predecessors.append(if_node)

                    if else_branch:
                        # Process else branch and capture final node
                        else_node = self.new_node()
                        final_else_node = self._build_from_stmt(else_branch, else_node)
                        if_node.successors.append(else_node)
                        else_node.predecessors.append(if_node)

                        # Merge both branches
                        merge_node = self.new_node()
                        final_then_node.successors.append(merge_node)
                        merge_node.predecessors.append(final_then_node)
                        final_else_node.successors.append(merge_node)
                        merge_node.predecessors.append(final_else_node)
                        current_node = merge_node
                    else:
                        # No else branch - just create new block after then
                        new_block = self.new_node()
                        final_then_node.successors.append(new_block)
                        new_block.predecessors.append(final_then_node)
                        current_node = new_block

                # Handle FOR loops
                elif child.kind == cindex.CursorKind.FOR_STMT:
                    """
                    FOR loop structure:
                        for (init; condition; increment) { body; }

                    CFG:
                        current_node → init_node → condition_node
                                                       ↓ (true)    ↓ (false - exit)
                                                   body_node → increment_node → after_loop
                                                       ↑               ↓
                                                       └─── back edge ──┘
                    """
                    children_of_for = list(child.get_children())

                    # Extract components (may be None if not present)
                    init_stmt = children_of_for[0] if len(children_of_for) >= 1 else None
                    condition = children_of_for[1] if len(children_of_for) >= 2 else None
                    increment = children_of_for[2] if len(children_of_for) >= 3 else None
                    body = children_of_for[3] if len(children_of_for) >= 4 else None

                    # Create init node (if init exists)
                    if init_stmt and init_stmt.kind != cindex.CursorKind.NULL_STMT:
                        init_node = self.new_node()
                        init_node.statements.append(init_stmt)
                        current_node.successors.append(init_node)
                        init_node.predecessors.append(current_node)
                        current_node = init_node

                    # Create condition node
                    condition_node = self.new_node()
                    if condition:
                        condition_node.statements.append(condition)
                    current_node.successors.append(condition_node)
                    condition_node.predecessors.append(current_node)

                    # Create body node and process it (FIXED: capture return value)
                    body_node = self.new_node()
                    if body:
                        final_body_node = self._build_from_stmt(body, body_node)
                    else:
                        final_body_node = body_node
                    condition_node.successors.append(body_node)
                    body_node.predecessors.append(condition_node)

                    # Create increment node (if increment exists)
                    if increment and increment.kind != cindex.CursorKind.NULL_STMT:
                        increment_node = self.new_node()
                        increment_node.statements.append(increment)
                        # FIXED: Connect from final_body_node, not body_node!
                        final_body_node.successors.append(increment_node)
                        increment_node.predecessors.append(final_body_node)
                        # BACK EDGE: increment → condition
                        increment_node.successors.append(condition_node)
                        condition_node.predecessors.append(increment_node)
                    else:
                        # No increment: body directly loops back to condition
                        # BACK EDGE: final_body_node → condition
                        final_body_node.successors.append(condition_node)
                        condition_node.predecessors.append(final_body_node)

                    # Create after-loop node (exit path)
                    after_loop_node = self.new_node()
                    condition_node.successors.append(after_loop_node)
                    after_loop_node.predecessors.append(condition_node)
                    current_node = after_loop_node

                # Handle WHILE loops
                elif child.kind == cindex.CursorKind.WHILE_STMT:
                    """
                    WHILE loop structure:
                        while (condition) { body; }

                    CFG:
                        current_node → condition_node
                                           ↓ (true)    ↓ (false - exit)
                                       body_node → after_loop
                                           ↓
                                       back edge
                                           ↓
                                       condition_node
                    """
                    children_of_while = list(child.get_children())
                    condition = children_of_while[0] if len(children_of_while) >= 1 else None
                    body = children_of_while[1] if len(children_of_while) >= 2 else None

                    # Create condition node
                    condition_node = self.new_node()
                    if condition:
                        condition_node.statements.append(condition)
                    current_node.successors.append(condition_node)
                    condition_node.predecessors.append(current_node)

                    # Create body node and process it (FIXED: capture return value)
                    body_node = self.new_node()
                    if body:
                        final_body_node = self._build_from_stmt(body, body_node)
                    else:
                        final_body_node = body_node
                    condition_node.successors.append(body_node)
                    body_node.predecessors.append(condition_node)

                    # BACK EDGE: final_body_node → condition (FIXED!)
                    final_body_node.successors.append(condition_node)
                    condition_node.predecessors.append(final_body_node)

                    # Create after-loop node (exit path)
                    after_loop_node = self.new_node()
                    condition_node.successors.append(after_loop_node)
                    after_loop_node.predecessors.append(condition_node)
                    current_node = after_loop_node

                # Handle DO-WHILE loops (NEW!)
                elif child.kind == cindex.CursorKind.DO_STMT:
                    """
                    DO-WHILE loop structure:
                        do { body; } while (condition);

                    CFG:
                        current_node → body_node → condition_node
                                           ↑             ↓ (true - back edge)
                                           └─────────────┘
                                                         ↓ (false - exit)
                                                    after_loop_node
                    """
                    children_of_do = list(child.get_children())
                    body = children_of_do[0] if len(children_of_do) >= 1 else None
                    condition = children_of_do[1] if len(children_of_do) >= 2 else None

                    # Create body node and process it
                    body_node = self.new_node()
                    if body:
                        final_body_node = self._build_from_stmt(body, body_node)
                    else:
                        final_body_node = body_node
                    current_node.successors.append(body_node)
                    body_node.predecessors.append(current_node)

                    # Create condition node
                    condition_node = self.new_node()
                    if condition:
                        condition_node.statements.append(condition)
                    final_body_node.successors.append(condition_node)
                    condition_node.predecessors.append(final_body_node)

                    # BACK EDGE: condition → body (if true)
                    condition_node.successors.append(body_node)
                    body_node.predecessors.append(condition_node)

                    # Create after-loop node (exit path when condition is false)
                    after_loop_node = self.new_node()
                    condition_node.successors.append(after_loop_node)
                    after_loop_node.predecessors.append(condition_node)
                    current_node = after_loop_node

                # Handle CXX_FOR_RANGE loops (range-based for loops)
                elif child.kind == cindex.CursorKind.CXX_FOR_RANGE_STMT:
                    """
                    Range-based FOR loop structure:
                        for (declaration : range_expression) { body; }

                    Clang's AST structure:
                        children[0]: VAR_DECL - the loop variable (e.g., 'spec')
                        children[1]: Expression - the container (e.g., 'spec_list')
                        children[2]: COMPOUND_STMT - the loop body

                    CFG:
                        current_node → range_init_node → body_node
                                                            ↓
                                                        (process body recursively)
                                                            ↓
                                                        after_loop_node
                                           ↑               ↓
                                           └─── back edge ──┘
                    """
                    children_of_for_range = list(child.get_children())

                    # Extract components
                    loop_var = children_of_for_range[0] if len(children_of_for_range) >= 1 else None
                    container = children_of_for_range[1] if len(children_of_for_range) >= 2 else None
                    body = children_of_for_range[2] if len(children_of_for_range) >= 3 else None

                    # Create range init node (loop variable + container expression)
                    range_init_node = self.new_node()
                    if loop_var:
                        range_init_node.statements.append(loop_var)
                    if container:
                        range_init_node.statements.append(container)
                    current_node.successors.append(range_init_node)
                    range_init_node.predecessors.append(current_node)

                    # Create body node and recursively process it
                    body_node = self.new_node()
                    if body:
                        final_body_node = self._build_from_stmt(body, body_node)
                    else:
                        final_body_node = body_node
                    range_init_node.successors.append(body_node)
                    body_node.predecessors.append(range_init_node)

                    # BACK EDGE: body → range_init (for next iteration)
                    final_body_node.successors.append(range_init_node)
                    range_init_node.predecessors.append(final_body_node)

                    # Create after-loop node (exit path)
                    after_loop_node = self.new_node()
                    range_init_node.successors.append(after_loop_node)
                    after_loop_node.predecessors.append(range_init_node)
                    current_node = after_loop_node

                # Handle SWITCH statements (NEW!)
                elif child.kind == cindex.CursorKind.SWITCH_STMT:
                    """
                    SWITCH statement structure:
                        switch (expr) {
                            case A: stmts; break;
                            case B: stmts; // fall through
                            case C: stmts; break;
                            default: stmts;
                        }

                    CFG:
                        current_node → switch_expr_node → case_A_node → case_B_node ...
                                                         → case_C_node → ...
                                                         → default_node → ...
                                       (all paths) → after_switch_node

                    Note: Basic implementation treats all cases as straight-line with fall-through.

                    IMPORTANT: Macros like EXPECT_THAT may generate degenerate SWITCH_STMT
                    structures where the body is not a COMPOUND_STMT. In such cases, treat
                    the entire SWITCH_STMT as a single statement.
                    """
                    children_of_switch = list(child.get_children())
                    switch_expr = children_of_switch[0] if len(children_of_switch) >= 1 else None
                    switch_body = children_of_switch[1] if len(children_of_switch) >= 2 else None

                    # FIX: Check if this is a proper switch statement with COMPOUND_STMT body
                    # If not, treat the entire SWITCH_STMT as a single statement (macro case)
                    if not switch_body or switch_body.kind != cindex.CursorKind.COMPOUND_STMT:
                        # Macro-generated or degenerate switch: treat as single statement
                        current_node.statements.append(child)
                    else:
                        # Normal switch statement: decompose into CFG nodes
                        # Create switch expression node
                        switch_expr_node = self.new_node()
                        if switch_expr:
                            switch_expr_node.statements.append(switch_expr)
                        current_node.successors.append(switch_expr_node)
                        switch_expr_node.predecessors.append(current_node)

                        # Create after-switch node (all cases converge here)
                        after_switch_node = self.new_node()

                        # Process switch body (contains CASE_STMT and DEFAULT_STMT)
                        case_nodes = []
                        prev_case_node = None

                        for case_child in switch_body.get_children():
                            if case_child.kind in (cindex.CursorKind.CASE_STMT,
                                                  cindex.CursorKind.DEFAULT_STMT):
                                # Create node for this case
                                case_node = self.new_node()
                                case_node.statements.append(case_child)

                                # Connect from switch expression
                                switch_expr_node.successors.append(case_node)
                                case_node.predecessors.append(switch_expr_node)

                                # Handle fall-through: connect from previous case
                                if prev_case_node:
                                    prev_case_node.successors.append(case_node)
                                    case_node.predecessors.append(prev_case_node)

                                case_nodes.append(case_node)
                                prev_case_node = case_node

                        # Connect all case nodes to after-switch (simplified - no BREAK handling)
                        for case_node in case_nodes:
                            case_node.successors.append(after_switch_node)
                            after_switch_node.predecessors.append(case_node)

                        current_node = after_switch_node

                # All other statements (straight-line code)
                else:
                    current_node.statements.append(child)

            return current_node
        else:
            current_node.statements.append(stmt)
            return current_node

# --- DDG Builder that uses the explicit CFG ---
class Statement:
    def __init__(self, stmt_id, defines=None, uses=None, file=None, line=None, column=None, source=None, cursor=None, calls=None):
        self.id = stmt_id
        # Ensure defines is stored as a list.
        if defines is None:
            self.defines = []
        elif isinstance(defines, list):
            self.defines = defines
        else:
            self.defines = [defines]
        self.uses = uses or []       # List of variables used
        self.calls = calls or []     # List of functions/constructors called (NEW)
        self.file = file             # File where the statement appears
        self.line = line             # Line number
        self.column = column         # Column number
        self.source = source         # Original statement text
        self.cursor = cursor
    def __repr__(self):
        return f"Statement({self.id}: def={self.defines}, uses={self.uses}, calls={self.calls})"

class BasicBlock:
    def __init__(self, block_id):
        self.id = block_id
        self.statements = []         # List of Statement objects
        self.succ = []               # Successor block IDs
        self.preds = []              # Predecessor block IDs
        self.gen_set = set()         # GEN set for reaching definitions
        self.kill_set = set()        # KILL set for reaching definitions

functions_of_files = set()  # e.g., {filename: set(funcs)}

class DDGAnalyzer:
    def __init__(self):
        self.blocks = {}             # Mapping from block id to BasicBlock
        self.definition_map = {}     # Map: variable -> set of statement IDs that define it
        self.def_id_to_vars = {}     # Map: statement ID -> list of variables defined
        self.IN = {}                 # IN set per block
        self.OUT = {}                # OUT set per block
        self.dependencies = defaultdict(set)  # Data dependence edges

    def analyze(self, entry_block_id):
        # 1. Initialize IN/OUT sets and fill predecessor lists.
        for block in self.blocks.values():
            self.IN[block.id] = set()
            self.OUT[block.id] = set()
        for block in self.blocks.values():
            for succ_id in block.succ:
                self.blocks[succ_id].preds.append(block.id)

        # 2. Build definition map.
        self.definition_map = defaultdict(set)
        for block in self.blocks.values():
            for stmt in block.statements:
                for var in stmt.defines:
                    self.definition_map[var].add(stmt.id)
                self.def_id_to_vars[stmt.id] = stmt.defines
        # 3. Compute GEN and KILL sets.
        for block in self.blocks.values():
            last_def_in_block = {}  # var -> last stmt id that defines var in this block
            for stmt in block.statements:
                for var in stmt.defines:
                    last_def_in_block[var] = stmt.id
            block.gen_set = set(last_def_in_block.values())
            block.kill_set = set()
            for var, def_ids in self.definition_map.items():
                if var in last_def_in_block:
                    for def_id in def_ids:
                        if def_id != last_def_in_block[var]:
                            block.kill_set.add(def_id)
        # 4. Worklist algorithm to compute reaching definitions.
        worklist = deque(self.blocks.keys())
        while worklist:
            block_id = worklist.popleft()
            block = self.blocks[block_id]
            new_in = set()
            for p in block.preds:
                new_in |= self.OUT[p]
            self.IN[block_id] = new_in
            new_out = block.gen_set | (new_in - block.kill_set)
            if new_out != self.OUT[block_id]:
                self.OUT[block_id] = new_out
                for succ in block.succ:
                    worklist.append(succ)
        # 5. Per-statement analysis.
        for block in self.blocks.values():
            reaching_defs = set(self.IN[block.id])
            for stmt in block.statements:
                for var in stmt.uses:
                    for def_id in list(reaching_defs):
                        if var in self.def_id_to_vars.get(def_id, []):
                            self.dependencies[stmt.id].add(def_id)
                if stmt.cursor.kind == cindex.CursorKind.CALL_EXPR:
                    for var in stmt.uses:
                        reaching_defs = {d for d in reaching_defs if var not in self.def_id_to_vars.get(d, [])}
                    reaching_defs.add(stmt.id)
                elif stmt.defines:
                    for var in stmt.defines:
                        reaching_defs = {d for d in reaching_defs if var not in self.def_id_to_vars.get(d, [])}
                    reaching_defs.add(stmt.id)
        # Now self.dependencies holds the DDG edges.

def get_source_at_line(file_path, line_number):
    """Read the actual source code at a specific line"""
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            if 0 < line_number <= len(lines):
                return lines[line_number - 1].strip()
    except Exception as e:
        print(f"Error reading {file_path}:{line_number}: {e}")
    return None

def is_likely_macro(cursor):
    """
    Check if cursor is likely from a macro expansion by comparing
    the cursor's extent with the original source line.
    """
    if not cursor.location.file:
        return False
    
    loc = cursor.location
    extent = cursor.extent
    
    # If cursor has no tokens, likely a macro
    tokens = list(cursor.get_tokens())
    if not tokens:
        return True
    
    # Check if the extent spans way more than the original source suggests
    # For macros like TAO_PEGTL_TEST_ASSERT(...), the DO_STMT extent is much larger
    # than the single line macro call
    if extent.start.line and extent.end.line and loc.line:
        # If extent spans multiple lines but we expect a single line statement
        original_source = get_source_at_line(str(loc.file), loc.line)
        if original_source:
            # Simple heuristic: if it looks like a single-line macro call
            # but the cursor extent is large, it's probably a macro
            if extent.end.line - extent.start.line > 0:
                # Multi-line extent but source is single line
                if original_source.count('\n') == 0:
                    return True
    
    return False

def get_location_info(cursor):
    # If the cursor is an UNEXPOSED_EXPR, try to unwrap it.
    if cursor.kind == cindex.CursorKind.UNEXPOSED_EXPR:
        for child in cursor.get_children():
            file, line, col = get_location_info(child)
            if file:
                return file, line, col
    loc = cursor.location
    if loc and loc.file:
        return str(loc.file), loc.line, loc.column
    return "", 0, 0

def get_source_text(cursor):
    tokens = list(cursor.get_tokens())
    if not tokens:
        # Fallback: try to read from file
        if cursor.extent.start.file and cursor.extent.start.file == cursor.extent.end.file:
            try:
                with open(str(cursor.extent.start.file), 'r') as f:
                    lines = f.readlines()
                    start_line = cursor.extent.start.line - 1
                    end_line = cursor.extent.end.line - 1
                    if start_line == end_line:
                        line = lines[start_line]
                        return line[cursor.extent.start.column-1:cursor.extent.end.column-1].strip()
            except:
                pass
        return ""
    return " ".join([t.spelling for t in tokens])


# --- AST-to-Statement Conversion Functions ---
global_stmt_id = 0

def extract_uses(cursor):
    """Recursively extract all variable references (DECL_REF_EXPR)"""
    uses = []
    if cursor.kind == cindex.CursorKind.DECL_REF_EXPR:
        uses.append(cursor.spelling)
    for child in cursor.get_children():
        uses.extend(extract_uses(child))
    return uses

def extract_calls(cursor):
    """
    Recursively extract all function/constructor calls.
    Returns a list of call info dicts with name, kind, file, line, usr.
    Filters out system library calls.

    Note: Regular function calls appear as CALL_EXPR nodes.
    Base class and delegating constructor calls appear as TYPE_REF nodes in initializer lists.
    """
    calls = []

    if cursor.kind == cindex.CursorKind.CALL_EXPR:
        # Get the function/constructor being called
        referenced = cursor.referenced
        if referenced:
            ref_file = str(referenced.location.file) if referenced.location.file else None

            # Filter out system library calls
            if ref_file and not _is_system_header(ref_file):
                call_info = {
                    'name': referenced.spelling,
                    'kind': str(referenced.kind),
                    'file': ref_file,
                    'line': referenced.location.line if referenced.location else None,
                    'usr': referenced.get_usr() if referenced else None
                }
                calls.append(call_info)

    elif cursor.kind == cindex.CursorKind.TYPE_REF:
        # TYPE_REF in constructor initializer lists represent base class or delegating constructor calls
        # Get the referenced type/class
        referenced = cursor.referenced
        if referenced:
            ref_file = str(referenced.location.file) if referenced.location.file else None

            _debug_print(f"[DEBUG extract_calls] TYPE_REF found: {cursor.spelling}")
            _debug_print(f"  Referenced class: {referenced.spelling} at {ref_file}")
            _debug_print(f"  Is system header: {_is_system_header(ref_file) if ref_file else 'N/A'}")

            # Filter out system library calls
            if ref_file and not _is_system_header(ref_file):
                # For TYPE_REF, we need to find the constructor being called
                # Look for constructors in the referenced class/struct
                if referenced.kind in (cindex.CursorKind.CLASS_DECL,
                                      cindex.CursorKind.STRUCT_DECL,
                                      cindex.CursorKind.CLASS_TEMPLATE):
                    # Find constructors in this class
                    # We'll add the class itself as the call target, and the specific constructor
                    # will be resolved later based on arguments
                    constructors_found = []
                    _debug_print(f"  Looking for direct constructors in {referenced.spelling}...")
                    for child in referenced.get_children():
                        if child.kind == cindex.CursorKind.CONSTRUCTOR:
                            # Check if constructor is deleted or defaulted
                            tokens = list(child.get_tokens())
                            token_strings = [t.spelling for t in tokens]
                            is_deleted = False
                            if '=' in token_strings:
                                eq_idx = token_strings.index('=')
                                if eq_idx + 1 < len(token_strings) and token_strings[eq_idx + 1] == 'delete':
                                    is_deleted = True
                                    _debug_print(f"    Skipping deleted constructor at line {child.location.line}")

                            if not is_deleted:
                                # Add this constructor as a potential call target
                                ctor_file = str(child.location.file) if child.location.file else None
                                if ctor_file and not _is_system_header(ctor_file):
                                    _debug_print(f"    Found direct constructor: {child.spelling} at {ctor_file}:{child.location.line}")
                                    call_info = {
                                        'name': child.spelling,
                                        'kind': str(child.kind),
                                        'file': ctor_file,
                                        'line': child.location.line if child.location else None,
                                        'usr': child.get_usr() if child else None
                                    }
                                    calls.append(call_info)
                                    constructors_found.append(child)

                    # If no direct constructors found, check for inherited constructors via 'using' declarations
                    if not constructors_found:
                        _debug_print(f"  No direct constructors found. Checking for USING_DECLARATION nodes...")
                        for child in referenced.get_children():
                            _debug_print(f"    Child node: {child.kind} - {child.spelling}")
                            if child.kind == cindex.CursorKind.USING_DECLARATION:
                                _debug_print(f"    ✓ Found USING_DECLARATION: {child.spelling}")
                                # Found a 'using' declaration - follow it to the base class
                                # Look for TEMPLATE_REF or TYPE_REF children that point to the base class
                                for using_child in child.get_children():
                                    _debug_print(f"      Using child: {using_child.kind} - {using_child.spelling}")
                                    if using_child.kind in (cindex.CursorKind.TEMPLATE_REF, cindex.CursorKind.TYPE_REF):
                                        _debug_print(f"      Following {using_child.kind} to base class...")
                                        base_class = using_child.referenced
                                        if base_class:
                                            _debug_print(f"        Base class: {base_class.spelling} (kind: {base_class.kind})")
                                            if base_class.kind in (cindex.CursorKind.CLASS_DECL,
                                                                    cindex.CursorKind.STRUCT_DECL,
                                                                    cindex.CursorKind.CLASS_TEMPLATE,
                                                                    cindex.CursorKind.CLASS_TEMPLATE_PARTIAL_SPECIALIZATION):
                                                # Find constructors in the base class
                                                _debug_print(f"        Looking for base class constructors...")

                                                # For CLASS_TEMPLATE nodes (forward declarations), we need to find
                                                # the actual specializations which contain the constructors
                                                # Walk up to the translation unit and search for partial specializations
                                                classes_to_search = [base_class]
                                                if base_class.kind == cindex.CursorKind.CLASS_TEMPLATE:
                                                    _debug_print(f"        Base is CLASS_TEMPLATE - searching for partial specializations...")
                                                    # Get translation unit
                                                    tu_cursor = base_class
                                                    while tu_cursor.semantic_parent:
                                                        tu_cursor = tu_cursor.semantic_parent

                                                    # Find all partial specializations with matching name
                                                    def find_specializations(cursor):
                                                        results = []
                                                        if (cursor.kind == cindex.CursorKind.CLASS_TEMPLATE_PARTIAL_SPECIALIZATION and
                                                            cursor.spelling == base_class.spelling):
                                                            results.append(cursor)
                                                        for child in cursor.get_children():
                                                            results.extend(find_specializations(child))
                                                        return results

                                                    specializations = find_specializations(tu_cursor)
                                                    _debug_print(f"        Found {len(specializations)} specialization(s)")
                                                    classes_to_search.extend(specializations)

                                                # Search all classes (base + specializations)
                                                for cls in classes_to_search:
                                                    for base_child in cls.get_children():
                                                        # Check for both CONSTRUCTOR and FUNCTION_TEMPLATE nodes
                                                        # Template constructors appear as FUNCTION_TEMPLATE
                                                        is_constructor = (base_child.kind == cindex.CursorKind.CONSTRUCTOR or
                                                                         (base_child.kind == cindex.CursorKind.FUNCTION_TEMPLATE and
                                                                          base_child.spelling.startswith(cls.spelling)))

                                                        if is_constructor:
                                                            # Check if deleted
                                                            tokens = list(base_child.get_tokens())
                                                            token_strings = [t.spelling for t in tokens]
                                                            is_deleted = False
                                                            if '=' in token_strings:
                                                                eq_idx = token_strings.index('=')
                                                                if eq_idx + 1 < len(token_strings) and token_strings[eq_idx + 1] == 'delete':
                                                                    is_deleted = True

                                                            if not is_deleted:
                                                                ctor_file = str(base_child.location.file) if base_child.location.file else None
                                                                if ctor_file and not _is_system_header(ctor_file):
                                                                    _debug_print(f"          ✓ Found inherited constructor: {base_child.spelling} at {ctor_file}:{base_child.location.line}")
                                                                    call_info = {
                                                                        'name': base_child.spelling,
                                                                        'kind': str(base_child.kind),
                                                                        'file': ctor_file,
                                                                        'line': base_child.location.line if base_child.location else None,
                                                                        'usr': base_child.get_usr() if base_child else None
                                                                    }
                                                                    calls.append(call_info)
                                        else:
                                            print(f"        WARNING: referenced is None")
                    else:
                        _debug_print(f"  Found {len(constructors_found)} direct constructor(s), skipping using declaration check")
                    # Note: We add all constructors because we can't easily determine
                    # which overload is called without deep type analysis
                    # The dependency analysis will handle this

    elif cursor.kind == cindex.CursorKind.TEMPLATE_REF:
        # TEMPLATE_REF in constructor initializer lists represent template base class constructor calls
        # Similar to TYPE_REF but for template classes
        # Get the referenced template/class
        referenced = cursor.referenced
        if referenced:
            ref_file = str(referenced.location.file) if referenced.location.file else None

            _debug_print(f"[DEBUG extract_calls] TEMPLATE_REF found: {cursor.spelling}")
            _debug_print(f"  Referenced template: {referenced.spelling} at {ref_file}")
            _debug_print(f"  Is system header: {_is_system_header(ref_file) if ref_file else 'N/A'}")

            # Filter out system library calls
            if ref_file and not _is_system_header(ref_file):
                # For TEMPLATE_REF, we need to find the constructor being called
                # Look for constructors in the referenced class/struct/template
                if referenced.kind in (cindex.CursorKind.CLASS_DECL,
                                      cindex.CursorKind.STRUCT_DECL,
                                      cindex.CursorKind.CLASS_TEMPLATE):
                    # Find constructors in this template class
                    constructors_found = []
                    _debug_print(f"  Looking for direct constructors in {referenced.spelling}...")
                    for child in referenced.get_children():
                        if child.kind == cindex.CursorKind.CONSTRUCTOR:
                            # Check if constructor is deleted or defaulted
                            tokens = list(child.get_tokens())
                            token_strings = [t.spelling for t in tokens]
                            is_deleted = False
                            if '=' in token_strings:
                                eq_idx = token_strings.index('=')
                                if eq_idx + 1 < len(token_strings) and token_strings[eq_idx + 1] == 'delete':
                                    is_deleted = True
                                    _debug_print(f"    Skipping deleted constructor at line {child.location.line}")

                            if not is_deleted:
                                # Add this constructor as a potential call target
                                ctor_file = str(child.location.file) if child.location.file else None
                                if ctor_file and not _is_system_header(ctor_file):
                                    _debug_print(f"    Found direct constructor: {child.spelling} at {ctor_file}:{child.location.line}")
                                    call_info = {
                                        'name': child.spelling,
                                        'kind': str(child.kind),
                                        'file': ctor_file,
                                        'line': child.location.line if child.location else None,
                                        'usr': child.get_usr() if child else None
                                    }
                                    calls.append(call_info)
                                    constructors_found.append(child)

                    # If no direct constructors found, check for inherited constructors via 'using' declarations
                    if not constructors_found:
                        _debug_print(f"  No direct constructors found. Checking for USING_DECLARATION nodes...")
                        for child in referenced.get_children():
                            _debug_print(f"    Child node: {child.kind} - {child.spelling}")
                            if child.kind == cindex.CursorKind.USING_DECLARATION:
                                _debug_print(f"    ✓ Found USING_DECLARATION: {child.spelling}")
                                # Found a 'using' declaration - follow it to the base class
                                # Look for TEMPLATE_REF or TYPE_REF children that point to the base class
                                for using_child in child.get_children():
                                    _debug_print(f"      Using child: {using_child.kind} - {using_child.spelling}")
                                    if using_child.kind in (cindex.CursorKind.TEMPLATE_REF, cindex.CursorKind.TYPE_REF):
                                        _debug_print(f"      Following {using_child.kind} to base class...")
                                        base_class = using_child.referenced
                                        if base_class:
                                            _debug_print(f"        Base class: {base_class.spelling} (kind: {base_class.kind})")
                                            if base_class.kind in (cindex.CursorKind.CLASS_DECL,
                                                                    cindex.CursorKind.STRUCT_DECL,
                                                                    cindex.CursorKind.CLASS_TEMPLATE,
                                                                    cindex.CursorKind.CLASS_TEMPLATE_PARTIAL_SPECIALIZATION):
                                                # Find constructors in the base class
                                                _debug_print(f"        Looking for base class constructors...")

                                                # For CLASS_TEMPLATE nodes (forward declarations), we need to find
                                                # the actual specializations which contain the constructors
                                                # Walk up to the translation unit and search for partial specializations
                                                classes_to_search = [base_class]
                                                if base_class.kind == cindex.CursorKind.CLASS_TEMPLATE:
                                                    _debug_print(f"        Base is CLASS_TEMPLATE - searching for partial specializations...")
                                                    # Get translation unit
                                                    tu_cursor = base_class
                                                    while tu_cursor.semantic_parent:
                                                        tu_cursor = tu_cursor.semantic_parent

                                                    # Find all partial specializations with matching name
                                                    def find_specializations(cursor):
                                                        results = []
                                                        if (cursor.kind == cindex.CursorKind.CLASS_TEMPLATE_PARTIAL_SPECIALIZATION and
                                                            cursor.spelling == base_class.spelling):
                                                            results.append(cursor)
                                                        for child in cursor.get_children():
                                                            results.extend(find_specializations(child))
                                                        return results

                                                    specializations = find_specializations(tu_cursor)
                                                    _debug_print(f"        Found {len(specializations)} specialization(s)")
                                                    classes_to_search.extend(specializations)

                                                # Search all classes (base + specializations)
                                                for cls in classes_to_search:
                                                    for base_child in cls.get_children():
                                                        # Check for both CONSTRUCTOR and FUNCTION_TEMPLATE nodes
                                                        # Template constructors appear as FUNCTION_TEMPLATE
                                                        is_constructor = (base_child.kind == cindex.CursorKind.CONSTRUCTOR or
                                                                         (base_child.kind == cindex.CursorKind.FUNCTION_TEMPLATE and
                                                                          base_child.spelling.startswith(cls.spelling)))

                                                        if is_constructor:
                                                            # Check if deleted
                                                            tokens = list(base_child.get_tokens())
                                                            token_strings = [t.spelling for t in tokens]
                                                            is_deleted = False
                                                            if '=' in token_strings:
                                                                eq_idx = token_strings.index('=')
                                                                if eq_idx + 1 < len(token_strings) and token_strings[eq_idx + 1] == 'delete':
                                                                    is_deleted = True

                                                            if not is_deleted:
                                                                ctor_file = str(base_child.location.file) if base_child.location.file else None
                                                                if ctor_file and not _is_system_header(ctor_file):
                                                                    _debug_print(f"          ✓ Found inherited constructor: {base_child.spelling} at {ctor_file}:{base_child.location.line}")
                                                                    call_info = {
                                                                        'name': base_child.spelling,
                                                                        'kind': str(base_child.kind),
                                                                        'file': ctor_file,
                                                                        'line': base_child.location.line if base_child.location else None,
                                                                        'usr': base_child.get_usr() if base_child else None
                                                                    }
                                                                    calls.append(call_info)
                                        else:
                                            print(f"        WARNING: referenced is None")
                    else:
                        _debug_print(f"  Found {len(constructors_found)} direct constructor(s), skipping using declaration check")

    # Recurse into children
    for child in cursor.get_children():
        calls.extend(extract_calls(child))

    return calls

def _is_system_header(file_path):
    """Check if a file is from a system library"""
    if file_path is None:
        return False
    return (file_path.startswith("/usr/") or
            "/lib/clang" in file_path or
            "/include/c++" in file_path or
            "/lib/gcc/" in file_path or
            "/../lib/gcc" in file_path)

def process_unexposed_expr(cursor):
    for child in cursor.get_children():
        if child.kind == cindex.CursorKind.CALL_EXPR:
            return child
    return None

def extract_macro_def_uses(cursor, original_source):
    """
    Extract def, uses, and calls from macro by recursively traversing the expanded AST
    and collecting all references (variables, functions, and calls).

    Note: This function extracts only actual variable references (DECL_REF_EXPR),
    NOT type names (TYPE_REF, TEMPLATE_REF). Type names represent compile-time
    entities and are not runtime data dependencies.

    Also filters out compiler/library internal variables (starting with _).
    """
    defines = []
    uses = []
    calls = []

    # Recursively collect all references
    def collect_all_references(node, depth=0):
        # Uncomment for debugging:
        # print(f"{'  ' * depth}Node: {node.kind}, spelling: '{node.spelling}', type: {node.type.spelling if node.type else 'N/A'}")
        
        if node.kind == cindex.CursorKind.DECL_REF_EXPR:
            # Variable or function reference
            uses.append(node.spelling)
            
        elif node.kind == cindex.CursorKind.CALL_EXPR:
            # Function call - track both uses and calls
            referenced = node.referenced
            if referenced:
                ref_file = str(referenced.location.file) if referenced.location.file else None

                # Filter out system library calls
                if ref_file and not _is_system_header(ref_file):
                    call_info = {
                        'name': referenced.spelling,
                        'kind': str(referenced.kind),
                        'file': ref_file,
                        'line': referenced.location.line if referenced.location else None,
                        'usr': referenced.get_usr() if referenced else None
                    }
                    calls.append(call_info)

            # NOTE: We do NOT add node.spelling to uses here because for constructors,
            # the spelling is the type/class name (e.g., "locale" for std::locale(x)),
            # which should not be tracked as a variable dependency.
            #
            # The actual variable arguments are extracted via recursive traversal below.
            # Function names for regular calls are already tracked in the 'calls' list above.
                        
        elif node.kind == cindex.CursorKind.MEMBER_REF_EXPR:
            # Member access
            uses.append(node.spelling)

        # TYPE_REF and TEMPLATE_REF removed: These represent type names, not variables.
        # Type names (like std::locale, std::runtime_error, template parameters) are
        # compile-time entities and should not be tracked as runtime data dependencies.
        # They are correctly handled in extract_calls() for constructor identification.
        #
        # elif node.kind == cindex.CursorKind.TYPE_REF:
        #     # REMOVED: Type reference - this is a type name, not a variable
        #     uses.append(node.spelling)
        #
        # elif node.kind == cindex.CursorKind.TEMPLATE_REF:
        #     # REMOVED: Template reference - this is a type name, not a variable
        #     uses.append(node.spelling)
            
        # Constructor expression handling removed: These also extract type names, not variables.
        # Constructors calls are correctly tracked in extract_calls() via CALL_EXPR nodes.
        #
        # elif (hasattr(cindex.CursorKind, 'CXX_FUNCTIONAL_CAST_EXPR') and
        #       node.kind == cindex.CursorKind.CXX_FUNCTIONAL_CAST_EXPR):
        #     # REMOVED: Functional cast - extracts type name
        #     ...
        #
        # elif (hasattr(cindex.CursorKind, 'CXX_TEMPORARY_OBJECT_EXPR') and
        #       node.kind == cindex.CursorKind.CXX_TEMPORARY_OBJECT_EXPR):
        #     # REMOVED: Temporary object - extracts type name
        #     ...
        #
        # elif node.type and node.type.spelling and node.kind.name.startswith('CXX_'):
        #     # REMOVED: CXX_ fallback - extracts type names from expression types
        #     ...
        
        # Recursively traverse all children
        for child in node.get_children():
            collect_all_references(child, depth + 1)
    
    collect_all_references(cursor)

    # Remove duplicates while preserving order
    uses = list(dict.fromkeys(uses))

    # Filter out compiler/library internal variables (starting with _ or __)
    # These are implementation details that should not be tracked as user-level dependencies
    uses = [u for u in uses if u and not u.startswith('_')]

    # Remove duplicate calls based on USR
    seen_usrs = set()
    unique_calls = []
    for call in calls:
        usr = call.get('usr')
        if usr and usr not in seen_usrs:
            seen_usrs.add(usr)
            unique_calls.append(call)
        elif not usr:  # If no USR, keep it anyway
            unique_calls.append(call)

    return defines, uses, unique_calls

def extract_def_and_uses(cursor):
    defines = []
    uses = []
    calls = []

    if cursor.kind in (cindex.CursorKind.VAR_DECL, cindex.CursorKind.PARM_DECL):
        defines.append(cursor.spelling)
        # Also extract initializer uses and calls
        for child in cursor.get_children():
            uses.extend(extract_uses(child))
            calls.extend(extract_calls(child))
    elif cursor.kind == cindex.CursorKind.MEMBER_REF:
        # Constructor member initializer: member(value)
        # NOTE: This branch is kept for backward compatibility but is not used
        # because MEMBER_REF nodes don't have semantic_parent set correctly.
        # The actual handling is in convert_ast_to_statement() with next_sibling parameter.
        defines.append(cursor.spelling)
        # Uses and calls are extracted via next_sibling in convert_ast_to_statement
    elif cursor.kind == cindex.CursorKind.IF_STMT:
        children = list(cursor.get_children())
        if children:
            cond = children[0]
            uses.extend(extract_uses(cond))
            calls.extend(extract_calls(cond))
    elif cursor.kind == cindex.CursorKind.DO_STMT:
        # DO_STMT often comes from macro expansion
        # Extract uses and calls from the condition and body
        for child in cursor.get_children():
            child_uses = extract_uses(child)
            uses.extend(child_uses)
            calls.extend(extract_calls(child))
    elif cursor.kind == cindex.CursorKind.WHILE_STMT:
        # Similar handling for while statements
        children = list(cursor.get_children())
        if children:
            # First child is condition
            uses.extend(extract_uses(children[0]))
            calls.extend(extract_calls(children[0]))
            # Process body
            for child in children[1:]:
                child_def, child_uses, child_calls = extract_def_and_uses(child)
                defines.extend(child_def)
                uses.extend(child_uses)
                calls.extend(child_calls)
    elif cursor.kind == cindex.CursorKind.UNEXPOSED_EXPR:
        call_child = process_unexposed_expr(cursor)
        if call_child:
            return extract_def_and_uses(call_child)
        else:
            for child in cursor.get_children():
                child_def, child_uses, child_calls = extract_def_and_uses(child)
                defines.extend(child_def)
                uses.extend(child_uses)
                calls.extend(child_calls)
            return defines, uses, calls
    elif cursor.kind == cindex.CursorKind.TYPE_REF:
        # TYPE_REF represents a base class or delegating constructor call in initializer list
        # Extract the constructor calls and uses from arguments
        calls.extend(extract_calls(cursor))
        # Extract uses from all children (which contain the arguments)
        for child in cursor.get_children():
            uses.extend(extract_uses(child))
        return defines, uses, calls
    elif cursor.kind == cindex.CursorKind.CALL_EXPR:
        # Extract uses from all arguments and track the call
        calls.extend(extract_calls(cursor))
        children = list(cursor.get_children())

        # For base class constructor calls, extract uses from all children (including the callee)
        # For regular calls, skip the first child (the callee) and only process arguments
        if cursor.referenced and cursor.referenced.kind == cindex.CursorKind.CONSTRUCTOR:
            # Base constructor call - extract uses from all arguments
            for arg in children:
                uses.extend(extract_uses(arg))
        else:
            # Regular function call - skip callee, process arguments
            for arg in children[1:]:
                uses.extend(extract_uses(arg))

        return defines, uses, calls
    elif cursor.kind in (cindex.CursorKind.BINARY_OPERATOR, cindex.CursorKind.COMPOUND_ASSIGNMENT_OPERATOR):
        tokens = list(cursor.get_tokens())
        token_strs = [t.spelling for t in tokens]
        if any(tok in token_strs for tok in ["=", "+=", "-=", "*=", "/="]):
            children = list(cursor.get_children())
            if len(children) >= 2:
                lhs = children[0]
                rhs = children[1]
                if lhs.kind == cindex.CursorKind.DECL_REF_EXPR:
                    defines.append(lhs.spelling)
                    # FIX: For compound assignments (+=, -=, etc.), LHS is also USED (read-modify-write)
                    if any(tok in token_strs for tok in ["+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=", "<<=", ">>="]):
                        uses.append(lhs.spelling)
                uses.extend(extract_uses(rhs))
                calls.extend(extract_calls(rhs))
            else:
                for child in cursor.get_children():
                    child_def, child_uses, child_calls = extract_def_and_uses(child)
                    defines.extend(child_def)
                    uses.extend(child_uses)
                    calls.extend(child_calls)
        else:
            for child in cursor.get_children():
                child_def, child_uses, child_calls = extract_def_and_uses(child)
                defines.extend(child_def)
                uses.extend(child_uses)
                calls.extend(child_calls)
    elif cursor.kind == cindex.CursorKind.CXX_FOR_RANGE_STMT:
        # FIX: Special handling for range-based for loops to capture the container variable
        # Range-for loop structure: for (declaration : range_expression) statement
        #
        # Clang's AST structure (verified):
        #   children[0]: VAR_DECL - the loop variable (e.g., 'item')
        #   children[1]: Expression - the container (e.g., DECL_REF_EXPR 'months', or CALL_EXPR, etc.)
        #   children[2]: COMPOUND_STMT - the loop body
        #
        # The container expression (child[1]) was being lost because DECL_REF_EXPR
        # nodes have no children, so the default recursive traversal couldn't find them.
        # We must use extract_uses() which explicitly handles DECL_REF_EXPR.
        children = list(cursor.get_children())

        if len(children) >= 1:
            # Extract loop variable as a definition
            loop_var = children[0]
            child_def, child_uses, child_calls = extract_def_and_uses(loop_var)
            defines.extend(child_def)
            uses.extend(child_uses)
            calls.extend(child_calls)

        if len(children) >= 2:
            # Extract container expression - use extract_uses() to properly handle DECL_REF_EXPR
            # This works for any expression type: DECL_REF_EXPR, CALL_EXPR, MEMBER_REF_EXPR, etc.
            container = children[1]
            uses.extend(extract_uses(container))
            calls.extend(extract_calls(container))

        if len(children) >= 3:
            # Extract from loop body
            body = children[2]
            child_def, child_uses, child_calls = extract_def_and_uses(body)
            defines.extend(child_def)
            uses.extend(child_uses)
            calls.extend(child_calls)

        return defines, uses, calls
    else:
        for child in cursor.get_children():
            child_def, child_uses, child_calls = extract_def_and_uses(child)
            defines.extend(child_def)
            uses.extend(child_uses)
            calls.extend(child_calls)

    return defines, uses, calls

def convert_ast_to_statement(cursor, next_sibling=None):
    """
    Convert an AST cursor to a Statement object.

    Args:
        cursor: The AST cursor to convert
        next_sibling: Optional next sibling cursor (used for member initializers)
    """
    global global_stmt_id

    file, line, column = get_location_info(cursor)
    original_source = None

    # Get original source if available
    if file and line:
        original_source = get_source_at_line(file, line)

    # Check if this is likely a macro
    if is_likely_macro(cursor) and original_source:
        defines, uses, calls = extract_macro_def_uses(cursor, original_source)

        if not uses:  # If we still couldn't extract anything useful
            _debug_print(f"Warning: Macro at {file}:{line} has no extractable uses")

        stmt = Statement(
            global_stmt_id,
            defines=defines,
            uses=uses,
            calls=calls,
            file=file,
            line=line,
            column=column,
            source=original_source,
            cursor=cursor
        )
        global_stmt_id += 1
        return stmt

    # Special handling for member initializers in constructor initializer lists
    if cursor.kind == cindex.CursorKind.MEMBER_REF and next_sibling:
        defines = [cursor.spelling]
        uses = []
        calls = []

        # The next sibling should be the value expression
        if next_sibling.kind in (cindex.CursorKind.UNEXPOSED_EXPR,
                                cindex.CursorKind.CALL_EXPR,
                                cindex.CursorKind.INIT_LIST_EXPR):
            uses = extract_uses(next_sibling)
            calls = extract_calls(next_sibling)

        source = get_source_text(cursor) or original_source
        stmt = Statement(global_stmt_id, defines=defines, uses=uses, calls=calls,
                        file=file, line=line, column=column,
                        source=source, cursor=cursor)
        global_stmt_id += 1
        return stmt

    # Standard processing for non-macro statements
    if cursor.kind == cindex.CursorKind.UNEXPOSED_EXPR:
        call_child = process_unexposed_expr(cursor)
        if call_child:
            cursor = call_child

    defines, uses, calls = extract_def_and_uses(cursor)
    uses = list(set(uses))  # Remove duplicates
    source = get_source_text(cursor) or original_source

    # Safety check
    if not source or not source.strip():
        if original_source:
            source = original_source
        else:
            _debug_print(f"Warning: Empty source at {file}:{line}")
            return None

    stmt = Statement(global_stmt_id, defines=defines, uses=uses, calls=calls,
                    file=file, line=line, column=column,
                    source=source, cursor=cursor)
    global_stmt_id += 1
    return stmt

def convert_cfgnode_to_basicblock(cfgnode):
    bb = BasicBlock(cfgnode.id)
    statements = cfgnode.statements
    i = 0
    while i < len(statements):
        ast_stmt = statements[i]
        next_sibling = statements[i + 1] if i + 1 < len(statements) else None

        # For MEMBER_REF, pass the next sibling (the value expression)
        if ast_stmt.kind == cindex.CursorKind.MEMBER_REF and next_sibling:
            stmt = convert_ast_to_statement(ast_stmt, next_sibling)
            if stmt:
                bb.statements.append(stmt)
            # Skip the next sibling since we already processed it as part of the initializer
            i += 2
        else:
            stmt = convert_ast_to_statement(ast_stmt)
            if stmt:
                bb.statements.append(stmt)
            i += 1
    return bb

def filter_compile_args(args, source_file):
    filtered = []
    skip_next = False
    for arg in args:
        if skip_next:
            skip_next = False
            continue
        if arg in ("clang++", "clang", "-o") or arg == source_file:
            if arg == "-o":
                skip_next = True
            continue
        if arg.startswith("--driver-mode"):
            continue
        filtered.append(arg)
    return filtered

# ============================================================
# Auxiliary Functions for Modularization and Aggregation
# ============================================================

def process_function(cursor, ddg_analyzer, index, root_dir, ddg_global_results, cfg_global_results):
    global global_stmt_id  # Declare at function start to avoid scope issues

    def _is_system_header(file_path: str) -> bool:
        if file_path is None:
            return False
        return (file_path.startswith("/usr/") or
                "/lib/clang" in file_path or
                "/include/c++" in file_path or
                "/lib/gcc/" in file_path)

    if _is_system_header(str(cursor.location.file)):
        return

    # Check for deleted or defaulted functions
    tokens = list(cursor.get_tokens())
    token_strings = [t.spelling for t in tokens]
    if '=' in token_strings:
        eq_idx = token_strings.index('=')
        if eq_idx + 1 < len(token_strings):
            next_token = token_strings[eq_idx + 1]
            if next_token in ('delete', 'default'):
                _debug_print(f"Skipping {next_token}d function: {cursor.spelling}")
                return

    functions_of_files.add(cursor.spelling)
    _debug_print(f"\nProcessing function: {cursor.spelling} at {cursor.location.file}:{cursor.location.line}")
    cfg_builder = CFGBuilder(cursor)
    entry_node = cfg_builder.build()
    if not entry_node:
        return

    cfg_json = []
    for node in cfg_builder.nodes:
        cfg_json.append({
            "id": node.id,
            "statements": [get_source_text(stmt) for stmt in node.statements],
            "successors": [succ.id for succ in node.successors],
            "predecessors": [pred.id for pred in node.predecessors]
        })
    cfg_global_results[cursor.spelling] = {"cfg": cfg_json, "file": str(cursor.location.file)}

    for node in cfg_builder.nodes:
        bb = convert_cfgnode_to_basicblock(node)
        ddg_analyzer.blocks[bb.id] = bb

    for node in cfg_builder.nodes:
        bb = ddg_analyzer.blocks[node.id]
        for succ in node.successors:
            bb.succ.append(succ.id)

    _debug_print("Basic Blocks:")
    for bb in ddg_analyzer.blocks.values():
        _debug_print(f"Block {bb.id}:")
        for stmt in bb.statements:
            _debug_print(f"   {stmt}")

    ddg_analyzer.analyze(entry_node.id)
    _debug_print("Data Dependence Edges:")
    for use_stmt, def_set in ddg_analyzer.dependencies.items():
        _debug_print(f"Statement {use_stmt} depends on defs: {sorted(def_set)}")

    # CRITICAL FIX: Add a statement for the function declaration itself
    # This allows calls to be matched to this function by USR for cross-file data flow
    func_decl_stmt_id = global_stmt_id
    global_stmt_id += 1

    ddg_global_results["statements"].append({
        "id": func_decl_stmt_id,
        "defines": [],  # Function declarations don't define variables in the data flow sense
        "uses": [],     # Function declarations don't use variables
        "calls": [],    # Function declarations don't make calls (the body does)
        "file": str(cursor.location.file),
        "line": cursor.location.line,
        "column": cursor.location.column,
        "source": cursor.spelling,  # Function name
        "kind": str(cursor.kind),   # FUNCTION_DECL, CXX_METHOD, or CONSTRUCTOR
        "usr": cursor.get_usr()     # Critical: USR for matching calls to this function
    })
    _debug_print(f"Added FUNCTION_DECL statement for {cursor.spelling} (USR: {cursor.get_usr()[:60]}...)")

    # CRITICAL FIX PART 2: Link the FUNCTION_DECL to the function body
    # Create dependencies from FUNCTION_DECL to entry statements (params + first body statements)
    # This allows DFS in extract_def_tree to traverse into the function body
    entry_stmt_ids = []

    # Collect all parameter declarations and first statements from the entry block
    for block in ddg_analyzer.blocks.values():
        for stmt in block.statements:
            # Include parameter declarations
            if stmt.cursor.kind == cindex.CursorKind.PARM_DECL:
                entry_stmt_ids.append(stmt.id)

    # Add entry block's first statements (those with no dependencies within the function)
    # These are typically the statements that start the function's data flow
    if entry_node.id in ddg_analyzer.blocks:
        entry_block = ddg_analyzer.blocks[entry_node.id]
        for stmt in entry_block.statements[:3]:  # First few statements of entry block
            if stmt.id not in entry_stmt_ids:
                entry_stmt_ids.append(stmt.id)

    # Create dependency edges from FUNCTION_DECL to entry statements
    if entry_stmt_ids:
        ddg_analyzer.dependencies[func_decl_stmt_id] = set(entry_stmt_ids)
        _debug_print(f"  Linked FUNCTION_DECL to {len(entry_stmt_ids)} entry statement(s)")

    # Add statements from all blocks
    has_statements = False
    for block in ddg_analyzer.blocks.values():
        for stmt in block.statements:
            # Get USR for function/constructor definitions
            usr = None
            if stmt.cursor.kind in (cindex.CursorKind.FUNCTION_DECL,
                                     cindex.CursorKind.CXX_METHOD,
                                     cindex.CursorKind.CONSTRUCTOR):
                usr = stmt.cursor.get_usr()

            ddg_global_results["statements"].append({
                "id": stmt.id,
                "defines": stmt.defines,
                "uses": stmt.uses,
                "calls": stmt.calls,
                "file": stmt.file,
                "line": stmt.line,
                "column": stmt.column,
                "source": stmt.source,
                "kind": str(stmt.cursor.kind),
                "usr": usr  # NEW: Add USR for matching
            })
            has_statements = True

    # For constructors with empty bodies, add a placeholder statement
    if not has_statements and cursor.kind == cindex.CursorKind.CONSTRUCTOR:
        loc = cursor.location
        # Extract constructor parameters as "uses"
        uses = []
        for child in cursor.get_children():
            if child.kind == cindex.CursorKind.PARM_DECL:
                uses.append(child.spelling)

        ddg_global_results["statements"].append({
            "id": global_stmt_id,
            "defines": [],  # Constructors don't "define" in the traditional sense
            "uses": uses,
            "calls": [],  # Empty bodies have no calls
            "file": str(loc.file) if loc.file else "",
            "line": loc.line if loc else 0,
            "column": loc.column if loc else 0,
            "source": f"{cursor.spelling}",
            "kind": str(cursor.kind),
            "usr": cursor.get_usr()  # Add USR for matching
        })
        global_stmt_id += 1

    for use_id, def_set in ddg_analyzer.dependencies.items():
        ddg_global_results["dependencies"][str(use_id)] = list(def_set)
    
    ddg_analyzer.blocks.clear()
    ddg_analyzer.IN.clear()
    ddg_analyzer.OUT.clear()
    ddg_analyzer.definition_map.clear()
    ddg_analyzer.def_id_to_vars.clear()
    ddg_analyzer.dependencies.clear()

def find_base_class_constructors(cursor, tu):
    """
    For a class/struct with 'using BaseClass::BaseClass' declarations,
    find the constructors from the base class.
    Returns a list of constructor cursors.
    """
    constructors = []

    # Look for USING_DECLARATION children
    for child in cursor.get_children():
        if child.kind == cindex.CursorKind.USING_DECLARATION:
            # Check if this is inheriting constructors
            # Usually spelled as the base class name
            referenced = child.referenced
            if referenced and referenced.kind == cindex.CursorKind.CONSTRUCTOR:
                # This is an inherited constructor
                _debug_print(f"    Found inherited constructor via using declaration: {child.spelling}")
                constructors.append(referenced)
            elif referenced and referenced.kind in (cindex.CursorKind.CLASS_DECL,
                                                      cindex.CursorKind.STRUCT_DECL,
                                                      cindex.CursorKind.CLASS_TEMPLATE):
                # This references the base class itself
                # Find all constructors in that base class
                for base_child in referenced.get_children():
                    if base_child.kind == cindex.CursorKind.CONSTRUCTOR:
                        _debug_print(f"    Found inherited constructor from base: {base_child.spelling}")
                        constructors.append(base_child)

        # Also check base class specifiers
        elif child.kind == cindex.CursorKind.CXX_BASE_SPECIFIER:
            base_class = child.referenced
            if base_class:
                # Check if there's a using declaration that matches this base class
                has_using_declaration = False
                for using_decl in cursor.get_children():
                    if using_decl.kind == cindex.CursorKind.USING_DECLARATION:
                        # Check if the using declaration refers to this base class
                        # (spelling might include template params, so check if base class name is in it)
                        if base_class.spelling in using_decl.spelling:
                            has_using_declaration = True
                            break

                # If there's a using declaration, add all constructors from the base class
                if has_using_declaration:
                    for base_child in base_class.get_children():
                        if base_child.kind == cindex.CursorKind.CONSTRUCTOR:
                            _debug_print(f"    Found inherited constructor from base specifier: {base_child.spelling}")
                            constructors.append(base_child)

    return constructors

def process_file(file_path, ddg_analyzer, index, root_dir, ddg_global_results, cfg_global_results):
    compile_cmd = None
    for cmd in ddg_analyzer.compdb.getAllCompileCommands():
        if cmd.filename == file_path:
            compile_cmd = cmd
            break

    if not compile_cmd:
        _debug_print(f"Warning: No compile command found for {file_path}")
        return

    args = [str(arg) for arg in compile_cmd.arguments]
    filtered_args = filter_compile_args(args, file_path)

    _debug_print(f"\nProcessing file: {file_path} with args: {filtered_args}")
    tu = index.parse(file_path, args=filtered_args)

    normalized_file_path = os.path.abspath(file_path)
    _debug_print(f"Looking for functions in: {normalized_file_path}")

    function_count = 0
    
    # Recursive function to find all functions in the AST
    def find_functions(cursor, depth=0):
        """Recursively find all function definitions in the AST"""
        nonlocal function_count

        # Always process the current node if it's a function definition
        if cursor.location.file:
            cursor_file = str(cursor.location.file)
            cursor_file_abs = os.path.abspath(cursor_file)

            # Process function definitions from our target file
            if cursor.kind == cindex.CursorKind.FUNCTION_DECL and cursor.is_definition():
                # Only process functions from the main source file, not headers
                if cursor_file_abs == normalized_file_path:
                    _debug_print(f"  >>> PROCESSING function: {cursor.spelling} at {cursor.location.line}")
                    process_function(cursor, ddg_analyzer, index, root_dir, ddg_global_results, cfg_global_results)
                    function_count += 1

            # Also process member functions inside classes/structs from any project file
            # (This catches constructors and methods that might be defined inline in headers)
            elif cursor.kind == cindex.CursorKind.CXX_METHOD and cursor.is_definition():
                # Skip system headers but include project headers
                if not cursor_file.startswith("/usr/") and not cursor_file.startswith("/lib/"):
                    _debug_print(f"  >>> PROCESSING method: {cursor.spelling} at {cursor_file}:{cursor.location.line}")
                    process_function(cursor, ddg_analyzer, index, root_dir, ddg_global_results, cfg_global_results)
                    function_count += 1

            # Process constructors (including those in templates)
            elif cursor.kind == cindex.CursorKind.CONSTRUCTOR:
                # For constructors, check if it's defined (has a body) or if it's in a template
                # Template constructors might not report is_definition() correctly
                has_body = False
                for child in cursor.get_children():
                    if child.kind == cindex.CursorKind.COMPOUND_STMT:
                        has_body = True
                        break

                if cursor.is_definition() or has_body:
                    if not cursor_file.startswith("/usr/") and not cursor_file.startswith("/lib/"):
                        _debug_print(f"  >>> PROCESSING constructor: {cursor.spelling} at {cursor_file}:{cursor.location.line}")
                        process_function(cursor, ddg_analyzer, index, root_dir, ddg_global_results, cfg_global_results)
                        function_count += 1

            # Process class/struct templates and their members
            elif cursor.kind in (cindex.CursorKind.CLASS_DECL,
                                  cindex.CursorKind.STRUCT_DECL,
                                  cindex.CursorKind.CLASS_TEMPLATE):
                # Skip system headers
                if not cursor_file.startswith("/usr/") and not cursor_file.startswith("/lib/"):
                    # Check for inherited constructors via using declarations
                    _debug_print(f"  Found class/struct: {cursor.spelling} at {cursor_file}:{cursor.location.line}")
                    inherited_constructors = find_base_class_constructors(cursor, tu)
                    for ctor in inherited_constructors:
                        ctor_file = str(ctor.location.file) if ctor.location.file else ""
                        if not ctor_file.startswith("/usr/") and not ctor_file.startswith("/lib/"):
                            _debug_print(f"  >>> PROCESSING inherited constructor: {ctor.spelling} at {ctor_file}:{ctor.location.line}")
                            process_function(ctor, ddg_analyzer, index, root_dir, ddg_global_results, cfg_global_results)
                            function_count += 1

        # Always recurse into all children to find functions
        # (functions can be nested in namespaces, classes, etc.)
        for child in cursor.get_children():
            find_functions(child, depth + 1)
    
    # Start recursive traversal from the translation unit root
    find_functions(tu.cursor)
    
    _debug_print(f"Total functions found: {function_count}")

def export_ddg_to_json(ddg_global_results, output_path):
    with open(output_path, "w") as f:
        json.dump(ddg_global_results, f, indent=2)

def export_cfg_to_json(cfg_global_results, output_path):
    with open(output_path, "w") as f:
        json.dump(cfg_global_results, f, indent=2)
    _debug_print(f"CFG results exported to {output_path}")

def extract_ddg(compdb_path, root_dir):
    index = cindex.Index.create()

    # clang's fromDirectory() always reads "compile_commands.json"
    # from the directory, ignoring the actual filename.  If the user
    # provides a custom-named file (e.g. compile_commands_filtered.json),
    # we need to temporarily symlink/copy it or load manually.
    import json as _json, shutil as _shutil
    _compdb_dir = os.path.dirname(os.path.abspath(compdb_path))
    _compdb_basename = os.path.basename(compdb_path)
    _canonical = os.path.join(_compdb_dir, 'compile_commands.json')
    _needs_restore = False
    _backup_canonical = _canonical + '._ddg_backup'

    if _compdb_basename != 'compile_commands.json':
        # The provided file has a non-standard name.  Temporarily
        # swap it in so fromDirectory() picks it up.
        if os.path.exists(_canonical):
            _shutil.copy2(_canonical, _backup_canonical)
            _needs_restore = True
        _shutil.copy2(compdb_path, _canonical)

    try:
        compdb = cindex.CompilationDatabase.fromDirectory(_compdb_dir)
    finally:
        if _needs_restore:
            _shutil.copy2(_backup_canonical, _canonical)
            os.remove(_backup_canonical)
        elif _compdb_basename != 'compile_commands.json':
            # We created compile_commands.json from the custom file;
            # remove it to avoid leaving stale files.
            if os.path.exists(_canonical):
                os.remove(_canonical)

    ddg_analyzer = DDGAnalyzer()
    ddg_analyzer.compdb = compdb

    if compdb.getAllCompileCommands() is None:
        raise RuntimeError("No compilation commands found in the database.")

    ddg_global_results = {"statements": [], "dependencies": {}}
    cfg_global_results = {}

    all_cmds = list(compdb.getAllCompileCommands())
    from tqdm import tqdm
    for compile_cmd in tqdm(all_cmds,
                            desc="  [1a-i] Building DDG",
                            unit="file", leave=True):
        file_path = compile_cmd.filename
        _debug_print(f"Compiling file: {file_path} in extract_ddg")
        process_file(file_path, ddg_analyzer, index, root_dir, ddg_global_results, cfg_global_results)
    
    export_ddg_to_json(ddg_global_results, "ddg.json")
    export_cfg_to_json(cfg_global_results, "cfg.json")
    _debug_print("DDG results exported to ddg.json")
    return ddg_global_results

def get_stmt_by_loc(root_file, root_line, ddg):
    """Get the first statement at the given location."""
    for stmt in ddg.get("statements", []):
        if stmt["file"] == root_file and stmt["line"] == root_line:
            return stmt
    return None

def get_all_stmts_by_loc(root_file, root_line, ddg):
    """
    Get ALL statements at the given location.

    This is important for macros that expand to multiple statements on the same line.
    For example, EXPECT_THAT() might generate multiple statements at the macro call site.
    """
    stmts = []
    for stmt in ddg.get("statements", []):
        if stmt["file"] == root_file and stmt["line"] == root_line:
            stmts.append(stmt)
    return stmts

# File-level translation unit cache for on-demand analysis.
# Avoids re-parsing the same header when multiple missing functions
# are in the same file.
_on_demand_tu_cache = {}   # file_path -> (tu, index)


def analyze_missing_function_on_demand(call_info, compile_args):
    """
    Parse a header file on-demand to analyze a missing function/constructor.
    Enhanced to handle inherited constructors via 'using' declarations.

    Args:
        call_info: Dict with 'file', 'line', 'usr', 'name', 'kind'
        compile_args: List of compilation arguments to use for parsing

    Returns:
        List of statement dicts to add to DDG, or empty list if analysis fails
    """
    call_file = call_info.get('file')
    call_usr = call_info.get('usr')
    call_name = call_info.get('name')
    call_line = call_info.get('line')
    call_kind = call_info.get('kind', '')

    if not call_file:
        return []

    _debug_print(f"  [ON-DEMAND] Analyzing missing function '{call_name}' in {os.path.basename(call_file)} at line {call_line}")

    try:
        # Parse the header file (cached per file to avoid re-parsing)
        abs_file = os.path.abspath(call_file)
        if abs_file in _on_demand_tu_cache:
            tu, index = _on_demand_tu_cache[abs_file]
        else:
            # Force C++ when parsing a header.  Without ``-x c++-header``
            # clang's ``from_source`` infers the language from the
            # extension — ``.h`` defaults to plain C, which fails to
            # parse C++-only headers (e.g. doctest.h, Catch2) and
            # raises ``TranslationUnitLoadError``.
            parse_args = list(compile_args or [])
            _ext = os.path.splitext(call_file)[1].lower()
            if _ext in ('.h', '.hpp', '.hxx', '.hh', '.inl'):
                # Prepend so explicit -x flags later in compile_args
                # still win if the caller provided one.
                if '-x' not in parse_args:
                    parse_args = ['-x', 'c++-header'] + parse_args
            index = cindex.Index.create()
            tu = index.parse(call_file, args=parse_args,
                            options=cindex.TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD)
            _on_demand_tu_cache[abs_file] = (tu, index)

        # STEP 1: Check if the call is to an inherited constructor via 'using' declaration
        # If call_line points to a 'using ClassName::ClassName' declaration, follow it to the base class
        inherited_constructor_info = None
        if 'CONSTRUCTOR' in call_kind and call_line:
            def find_using_declaration_at_line(cursor):
                """Find a using declaration at the specific line"""
                if cursor.kind == cindex.CursorKind.USING_DECLARATION:
                    if cursor.location.file and cursor.location.line == call_line:
                        file_path = str(cursor.location.file)
                        if os.path.abspath(file_path) == os.path.abspath(call_file):
                            return cursor
                for child in cursor.get_children():
                    result = find_using_declaration_at_line(child)
                    if result:
                        return result
                return None

            using_decl = find_using_declaration_at_line(tu.cursor)
            if using_decl:
                _debug_print(f"    Found 'using' declaration at line {call_line}")
                _debug_print(f"    using_decl.spelling: {using_decl.spelling}")

                # Traverse children to find the base class reference
                # The structure is: USING_DECLARATION -> TEMPLATE_REF (for template classes) or TYPE_REF (for regular classes)
                base_class_cursor = None
                for child in using_decl.get_children():
                    _debug_print(f"      using_decl child: {child.kind} - {child.spelling}")
                    if child.kind == cindex.CursorKind.TEMPLATE_REF:
                        # This references the base class template
                        base_class_cursor = child.referenced
                        _debug_print(f"      TEMPLATE_REF references: {base_class_cursor.kind} - {base_class_cursor.spelling}")
                        break
                    elif child.kind == cindex.CursorKind.TYPE_REF:
                        # This references the base class
                        base_class_cursor = child.referenced
                        _debug_print(f"      TYPE_REF references: {base_class_cursor.kind} - {base_class_cursor.spelling}")
                        break

                # Get the referenced base class or constructor
                if base_class_cursor and base_class_cursor.kind in (cindex.CursorKind.CLASS_TEMPLATE,
                                                                      cindex.CursorKind.CLASS_DECL,
                                                                      cindex.CursorKind.STRUCT_DECL):
                    # Found base class - find matching constructor
                    _debug_print(f"    Found base class: {base_class_cursor.spelling}")
                    _debug_print(f"    Looking for constructor with name: {call_name}")

                    # Find all constructors in the base class
                    for child in base_class_cursor.get_children():
                        if child.kind == cindex.CursorKind.CONSTRUCTOR:
                            _debug_print(f"      Found constructor: {child.spelling} at {os.path.basename(str(child.location.file))}:{child.location.line}")
                            # Match by constructor name (might include template params)
                            # For template classes, constructor spelling includes template params like "mmap_input<P, Eol>"
                            # but call_name is just "mmap_input", so check if constructor name starts with call_name
                            if child.spelling.startswith(call_name):
                                inherited_constructor_info = {
                                    'file': str(child.location.file) if child.location.file else None,
                                    'line': child.location.line,
                                    'usr': child.get_usr(),
                                    'name': child.spelling,
                                    'kind': str(child.kind)
                                }
                                _debug_print(f"    ✓ Matched constructor at {os.path.basename(inherited_constructor_info['file'])}:{inherited_constructor_info['line']}")
                                break

        # If we found an inherited constructor, recursively analyze that instead
        if inherited_constructor_info and inherited_constructor_info['file'] != call_file:
            _debug_print(f"    Recursively analyzing inherited constructor...")
            return analyze_missing_function_on_demand(inherited_constructor_info, compile_args)

        # STEP 2: Find the function/constructor with matching USR or location
        statements_to_add = []
        matched_functions = []

        def find_matching_functions(cursor, depth=0):
            """Find all functions matching by USR, location, or name (for templates)"""
            # Check if this is a function/method/constructor
            if cursor.kind in (cindex.CursorKind.FUNCTION_DECL,
                              cindex.CursorKind.CXX_METHOD,
                              cindex.CursorKind.CONSTRUCTOR,
                              cindex.CursorKind.FUNCTION_TEMPLATE):
                cursor_usr = cursor.get_usr()
                cursor_name = cursor.spelling
                cursor_file = str(cursor.location.file) if cursor.location.file else None
                cursor_line = cursor.location.line if cursor.location else None

                # Skip system headers
                if cursor_file and _is_system_header(cursor_file):
                    return

                # Match by USR (exact match)
                if cursor_usr and cursor_usr == call_usr:
                    matched_functions.append(cursor)
                    return

                # Match by location (file + line)
                elif call_line and cursor_file and cursor_line:
                    call_file_abs = os.path.abspath(call_file)
                    cursor_file_abs = os.path.abspath(cursor_file)
                    if call_file_abs == cursor_file_abs and cursor_line == call_line:
                        matched_functions.append(cursor)
                        return

                # For templates: match by name if USR doesn't match exactly
                # Template instantiations have different USRs than template definitions
                elif call_name and cursor_name == call_name:
                    # Additional check: both must be templates or have template in USR
                    if (cursor.kind == cindex.CursorKind.FUNCTION_TEMPLATE or
                        (call_usr and ('<' in call_usr or 'template' in call_usr.lower())) or
                        (cursor_usr and ('<' in cursor_usr or 'template' in cursor_usr.lower()))):
                        # This is likely a template definition matching a template instantiation call
                        _debug_print(f"    Fuzzy-matched template: {cursor_name} at {os.path.basename(cursor_file or 'unknown')}:{cursor_line}")
                        matched_functions.append(cursor)
                        # Don't return - keep looking for exact matches

            # Recurse into children
            for child in cursor.get_children():
                find_matching_functions(child, depth + 1)

        # Search for the function
        find_matching_functions(tu.cursor)

        # Analyze all matched functions
        for func_cursor in matched_functions:
            global global_stmt_id  # Access the global statement ID counter

            _debug_print(f"    Analyzing function: {func_cursor.spelling} at line {func_cursor.location.line}")

            # Check if function has a definition (body)
            has_definition = func_cursor.is_definition()
            if not has_definition:
                # For templates, check if there are children indicating a body
                for child in func_cursor.get_children():
                    if child.kind == cindex.CursorKind.COMPOUND_STMT:
                        has_definition = True
                        break

            if not has_definition:
                _debug_print(f"    Skipping: no definition found")
                continue

            # Build CFG for this function
            cfg_builder = CFGBuilder(func_cursor)
            entry_node = cfg_builder.build()

            if entry_node:
                # Convert CFG nodes to statements
                for cfg_node in cfg_builder.nodes:
                    bb = convert_cfgnode_to_basicblock(cfg_node)
                    for stmt in bb.statements:
                        func_usr = func_cursor.get_usr()
                        stmt_dict = {
                            "id": stmt.id,
                            "defines": stmt.defines,
                            "uses": stmt.uses,
                            "calls": stmt.calls,
                            "file": stmt.file,
                            "line": stmt.line,
                            "column": stmt.column,
                            "source": stmt.source,
                            "kind": str(stmt.cursor.kind),
                            "usr": func_usr if stmt.cursor.kind in (cindex.CursorKind.FUNCTION_DECL,
                                                                       cindex.CursorKind.CXX_METHOD,
                                                                       cindex.CursorKind.CONSTRUCTOR) else None
                        }
                        statements_to_add.append(stmt_dict)
                        global_stmt_id += 1

                _debug_print(f"    Added {len(statements_to_add)} statements")
            else:
                _debug_print(f"    No CFG entry node (empty body)")

        if not matched_functions:
            _debug_print(f"    Warning: Could not find function with USR {call_usr[:60] if call_usr else 'N/A'}...")
            _debug_print(f"    Call info: name={call_name}, file={os.path.basename(call_file)}, line={call_line}")

        return statements_to_add

    except Exception as e:
        print(f"    Error analyzing {call_file}: {e}")
        import traceback
        traceback.print_exc()
        return []

def extract_def_tree(root_file, root_line, ddg, flatten=True, include_call_deps=True, compile_args=None):
    """
    Extract dependency tree from DDG with on-demand analysis of missing functions.

    Args:
        root_file: File path of the root statement
        root_line: Line number of the root statement
        ddg: DDG data structure
        flatten: Whether to flatten the tree into a list
        include_call_deps: Whether to include call dependencies (NEW)
        compile_args: Compilation arguments for on-demand parsing (NEW)

    Returns:
        Dependency tree or flattened list of statements
    """
    # Default compilation args if not provided
    if compile_args is None:
        compile_args = ["-std=c++17"]

    stmt_by_id = {stmt["id"]: stmt for stmt in ddg.get("statements", [])}

    # Build reverse indexes for matching calls to definitions
    function_by_location = {}  # {(file, line): [stmt_ids]}
    function_by_usr = {}        # {usr: [stmt_ids]} - most reliable matching
    analyzed_functions_cache = set()  # Cache of USRs that have been analyzed on-demand

    # Count unique user-header files referenced by calls (upper bound
    # for on-demand file parses).
    _all_call_files = set()
    for stmt in ddg.get("statements", []):
        for call in stmt.get("calls", []):
            cf = call.get("file", "")
            if cf and '/usr/' not in cf and not cf.startswith('/include/') \
                    and '/lib/gcc/' not in cf and '/lib/clang/' not in cf:
                _all_call_files.add(os.path.abspath(cf))
    # Files already in the DDG don't need on-demand analysis.
    _ddg_files = {os.path.abspath(s["file"])
                  for s in ddg.get("statements", []) if s.get("file")}
    _potential_ondemand_files = _all_call_files - _ddg_files
    from tqdm import tqdm as _tqdm
    _ondemand_pbar = _tqdm(
        total=max(len(_potential_ondemand_files), 1),
        desc="  [1a-i] On-demand file analysis",
        unit="file", leave=True) if _potential_ondemand_files else None
    _ondemand_files_done = set()

    for stmt in ddg.get("statements", []):
        if stmt.get("kind") in ["CursorKind.CONSTRUCTOR", "CursorKind.FUNCTION_DECL", "CursorKind.CXX_METHOD"]:
            # Index by location
            key = (stmt["file"], stmt["line"])
            if key not in function_by_location:
                function_by_location[key] = []
            function_by_location[key].append(stmt["id"])

            # Index by USR (Unified Symbol Resolution) for exact matching
            usr = stmt.get("usr")
            if usr:
                if usr not in function_by_usr:
                    function_by_usr[usr] = []
                function_by_usr[usr].append(stmt["id"])

    # FIX: Handle macros that expand to multiple statements at the same line
    # Instead of just getting the first statement, get ALL statements at this line
    root_stmts = get_all_stmts_by_loc(root_file, root_line, ddg)

    # Check if we have statements with actual data flow info
    has_dataflow = any(
        stmt.get('defines') or stmt.get('uses') or stmt.get('calls')
        for stmt in root_stmts
    ) if root_stmts else False

    # If no statements at exact line OR statements have no data flow,
    # parse the source line to extract variables and find their definitions.
    # This handles cases like macros (EXPECT_EQ) or statements inside control flow blocks.
    if not root_stmts or not has_dataflow:
        if not root_stmts:
            _debug_print(f"No statement found at {root_file}:{root_line}")
        else:
            _debug_print(f"[Data Flow] Found {len(root_stmts)} statement(s) but no data flow info")

        _debug_print(f"[Data Flow] Parsing source line to extract variables...")

        # Extract variables from the source line by re-parsing it
        variables_at_line = set()
        source_line = ""

        # First, read the source line from file
        try:
            with open(root_file, 'r') as f:
                lines = f.readlines()
                if 0 < root_line <= len(lines):
                    source_line = lines[root_line - 1].strip()
                    _debug_print(f"[Data Flow] Source line {root_line}: {source_line[:80]}...")
                else:
                    _debug_print(f"[Data Flow] Invalid line number {root_line} (file has {len(lines)} lines)")
        except Exception as e:
            print(f"[Data Flow] Failed to read source file: {e}")

        # Try to parse the source line with clang
        if source_line:
            try:
                index_temp = cindex.Index.create()
                # Parse just this line (wrapped in a minimal context)
                # This is a heuristic approach - wrap in a function to make it parseable
                temp_code = f"void temp() {{ {source_line} }}"
                tu = index_temp.parse('temp.cc', unsaved_files=[('temp.cc', temp_code)], args=compile_args)

                # Extract variables from the parsed line
                def find_decl_refs(cursor):
                    refs = set()
                    if cursor.kind == cindex.CursorKind.DECL_REF_EXPR:
                        refs.add(cursor.spelling)
                    for child in cursor.get_children():
                        refs.update(find_decl_refs(child))
                    return refs

                variables_at_line = find_decl_refs(tu.cursor)
                # Filter out operators and system functions
                variables_at_line = {
                    v for v in variables_at_line
                    if not v.startswith('operator') and not v.startswith('__')
                }
                _debug_print(f"[Data Flow] Extracted variables from source (clang): {variables_at_line}")
            except Exception as e:
                print(f"[Data Flow] Failed to parse source line with clang: {e}")

        # Fallback: If clang parsing returned empty (common with macros like EXPECT_EQ),
        # TODO: Should consider if we can improve clang parsing to avoid using regex fallback.
        # use regex-based extraction to find potential variable names
        if not variables_at_line and source_line:
            _debug_print(f"[Data Flow] Clang parsing returned empty, trying regex-based extraction...")
            import re
            # Extract all C++ identifiers from the line
            all_identifiers = set(re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b', source_line))

            # Filter out common keywords, macros, namespaces, and type names
            cpp_keywords = {
                'if', 'else', 'for', 'while', 'do', 'switch', 'case', 'break', 'continue',
                'return', 'void', 'int', 'char', 'float', 'double', 'bool', 'auto', 'const',
                'static', 'class', 'struct', 'enum', 'union', 'public', 'private', 'protected',
                'virtual', 'override', 'final', 'new', 'delete', 'this', 'nullptr', 'true', 'false',
                'sizeof', 'typeid', 'typename', 'template', 'namespace', 'using', 'typedef',
                'throw', 'try', 'catch', 'noexcept', 'constexpr', 'inline', 'explicit',
                'unsigned', 'signed', 'long', 'short', 'volatile', 'mutable', 'register', 'extern'
            }
            common_macros = {
                'EXPECT_EQ', 'EXPECT_NE', 'EXPECT_TRUE', 'EXPECT_FALSE', 'EXPECT_LT', 'EXPECT_LE',
                'EXPECT_GT', 'EXPECT_GE', 'EXPECT_STREQ', 'EXPECT_STRNE', 'EXPECT_THAT',
                'ASSERT_EQ', 'ASSERT_NE', 'ASSERT_TRUE', 'ASSERT_FALSE', 'ASSERT_LT', 'ASSERT_LE',
                'ASSERT_GT', 'ASSERT_GE', 'ASSERT_STREQ', 'ASSERT_STRNE', 'ASSERT_THAT',
                'TEST', 'TEST_F', 'TEST_P', 'TYPED_TEST', 'INSTANTIATE_TEST_SUITE_P',
                'NULL', 'EOF', 'NDEBUG', 'DEBUG'
            }
            # common_namespaces = {'std', 'fmt', 'testing', 'gtest', 'chrono'}
            common_namespaces = {'std', 'gtest'}
            common_functions = {'format', 'runtime', 'to_string', 'make_pair', 'make_tuple', 'move', 'forward'}

            # Get all variable names defined in the DDG (for this file) to match against
            known_variables = set()
            for stmt in ddg.get("statements", []):
                if stmt.get("file") == root_file:
                    known_variables.update(stmt.get('defines', []))

            # Filter: keep identifiers that are either known variables or look like variables
            # (not keywords, not macros, not namespaces)
            filtered_identifiers = set()
            for ident in all_identifiers:
                if ident in cpp_keywords or ident in common_macros or ident in common_namespaces:
                    continue
                if ident in common_functions:
                    continue
                # Keep if it's a known variable OR if it looks like a local variable (lowercase start)
                if ident in known_variables or (ident[0].islower() and len(ident) > 1):
                    filtered_identifiers.add(ident)

            variables_at_line = filtered_identifiers
            _debug_print(f"[Data Flow] Extracted variables from source (regex): {variables_at_line}")

            # END OF TODO: Case of handling macros with rule-based fall back.

        # Find definition statements for these variables
        if variables_at_line:
            _debug_print(f"[Data Flow] Finding definitions for variables: {variables_at_line}")
            def_stmts = []
            for stmt in ddg.get("statements", []):
                # Look for statements that define any of our variables
                stmt_defines = set(stmt.get('defines', []))
                if stmt_defines & variables_at_line:  # Intersection
                    _debug_print(f"[Data Flow]   Found definition at line {stmt['line']}: defines {stmt_defines & variables_at_line}")
                    def_stmts.append(stmt)

            if def_stmts:
                _debug_print(f"[Data Flow] Adding {len(def_stmts)} definition statement(s) as roots")
                root_stmts.extend(def_stmts)
            else:
                _debug_print(f"[Data Flow] No definitions found for variables: {variables_at_line}")
        else:
            _debug_print(f"[Data Flow] No variables extracted from source line")

        # If still no root statements, return None
        if not root_stmts:
            _debug_print(f"[Data Flow] ERROR: Could not find any statements related to {root_file}:{root_line}")
            return None
    else:
        _debug_print(f"[Data Flow] Found {len(root_stmts)} statement(s) at {root_file}:{root_line}")
        for i, stmt in enumerate(root_stmts):
            _debug_print(f"  [{i}] defines={stmt.get('defines', [])}, uses={stmt.get('uses', [])}, calls={len(stmt.get('calls', []))}")

    def dfs(stmt_id, path):
        if stmt_id in path:
            return {"stmt": stmt_by_id.get(stmt_id, {}), "deps": ["cycle detected"]}
        new_path = path | {stmt_id}
        children = []

        # Add data dependencies (traditional DDG)
        for dep in ddg.get("dependencies", {}).get(str(stmt_id), []):
            dep_id = int(dep)
            children.append(dfs(dep_id, new_path))

        # Add call dependencies (NEW)
        if include_call_deps:
            stmt = stmt_by_id.get(stmt_id, {})
            for call_info in stmt.get("calls", []):
                # Find the statement that defines this function/constructor
                call_file = call_info.get("file")
                call_line = call_info.get("line")
                call_usr = call_info.get("usr")
                call_name = call_info.get("name")

                func_stmt_ids = []

                # Try matching by USR first (most reliable)
                if call_usr and call_usr in function_by_usr:
                    func_stmt_ids = function_by_usr[call_usr]

                # Fall back to location-based matching
                if not func_stmt_ids and call_file and call_line:
                    key = (call_file, call_line)
                    func_stmt_ids = function_by_location.get(key, [])

                # If still no match, and this is a constructor, look for constructors with the same name
                # This handles inherited constructors where the call points to a 'using' declaration
                if not func_stmt_ids and call_name and 'CONSTRUCTOR' in call_info.get('kind', ''):
                    # Find all constructors with matching name
                    for stmt_id_check, stmt_obj in stmt_by_id.items():
                        if ('CONSTRUCTOR' in stmt_obj.get('kind', '') and
                            call_name in stmt_obj.get('source', '')):
                            func_stmt_ids.append(stmt_id_check)

                # ON-DEMAND ANALYSIS: If still no match, analyze the header file on-demand
                if not func_stmt_ids and call_usr:
                    # Check if we've already analyzed this function (prevent infinite loops)
                    if call_usr in analyzed_functions_cache:
                        _debug_print(f"  Function {call_name} already analyzed (cached), skipping...")
                        continue

                    # Skip system headers — they won't have
                    # instrumentable functions.
                    if call_file and (
                            '/usr/' in call_file
                            or call_file.startswith('/include/')
                            or '/lib/gcc/' in call_file
                            or '/lib/clang/' in call_file):
                        analyzed_functions_cache.add(call_usr)
                        continue

                    _debug_print(f"  Function not found in DDG: {call_name} (USR: {call_usr[:60]}...)")
                    analyzed_functions_cache.add(call_usr)  # Mark as being analyzed
                    # Update progress bar when a new file is parsed.
                    _cf_abs = os.path.abspath(call_file) if call_file else ''
                    if _ondemand_pbar and _cf_abs \
                            and _cf_abs not in _ondemand_files_done:
                        _ondemand_files_done.add(_cf_abs)
                        _ondemand_pbar.update(1)
                        _ondemand_pbar.set_postfix_str(
                            os.path.basename(call_file))
                    new_statements = analyze_missing_function_on_demand(call_info, compile_args)

                    if new_statements:
                        # Add new statements to DDG and indexes
                        for new_stmt in new_statements:
                            stmt_by_id[new_stmt["id"]] = new_stmt
                            ddg["statements"].append(new_stmt)

                            # Update indexes
                            if new_stmt.get("kind") in ["CursorKind.CONSTRUCTOR", "CursorKind.FUNCTION_DECL", "CursorKind.CXX_METHOD"]:
                                key = (new_stmt["file"], new_stmt["line"])
                                if key not in function_by_location:
                                    function_by_location[key] = []
                                function_by_location[key].append(new_stmt["id"])

                                usr = new_stmt.get("usr")
                                if usr:
                                    if usr not in function_by_usr:
                                        function_by_usr[usr] = []
                                    function_by_usr[usr].append(new_stmt["id"])
                                    _debug_print(f"  DEBUG: Indexed new stmt {new_stmt['id']} with USR {usr[:60]}...")

                        # Now try matching again - try both exact USR and fuzzy name matching
                        if call_usr in function_by_usr:
                            func_stmt_ids = function_by_usr[call_usr]
                            _debug_print(f"  ✓ On-demand analysis succeeded (USR match): found {len(func_stmt_ids)} statement(s)")
                        else:
                            # Fuzzy match: match by name and file for templates
                            _debug_print(f"  DEBUG: Call USR not in index, trying fuzzy match...")
                            _debug_print(f"  DEBUG: Looking for function {call_name} at {call_info.get('file')}:{call_info.get('line')}")
                            for new_stmt in new_statements:
                                # Match this statement if it's from the right function
                                # For templates, the statement inside the function won't have a USR,
                                # but we want to include them as dependencies
                                func_stmt_ids.append(new_stmt["id"])
                                _debug_print(f"  DEBUG: Fuzzy-matched statement {new_stmt['id']} from {call_name}")
                            if func_stmt_ids:
                                _debug_print(f"  ✓ On-demand analysis succeeded (fuzzy match): found {len(func_stmt_ids)} statement(s)")

                # Add all matching function definitions to dependencies
                for func_stmt_id in func_stmt_ids:
                    if func_stmt_id not in new_path:  # Avoid cycles
                        children.append(dfs(func_stmt_id, new_path))

        return {"stmt": stmt_by_id.get(stmt_id, {}), "deps": children}

    # FIX: Build dependency tree from ALL root statements (handles macros)
    # Merge all dependencies from all statements at the target line
    _debug_print(f"[Data Flow] Building dependency tree from {len(root_stmts)} root statement(s)...")
    all_deps = []
    for root_stmt in root_stmts:
        _debug_print(f"[Data Flow]   Processing root stmt {root_stmt['id']} with uses={root_stmt.get('uses', [])}")
        dep_subtree = dfs(root_stmt["id"], set())
        all_deps.append(dep_subtree)

    # If only one root statement, return its tree directly
    if len(all_deps) == 1:
        def_tree = all_deps[0]
    else:
        # Multiple root statements: create a synthetic root that includes all of them
        def_tree = {
            "stmt": {
                "id": -1,  # Synthetic ID
                "file": root_file,
                "line": root_line,
                "kind": "SYNTHETIC_ROOT",
                "defines": [],
                "uses": [],
                "calls": []
            },
            "deps": all_deps
        }

    if _ondemand_pbar:
        _ondemand_pbar.close()

    return def_tree if not flatten else flatten_def_tree(def_tree)

def flatten_def_tree(dep_tree):
    flat_list = []
    visited = set()

    def dfs(node):
        stmt = node.get("stmt", {})
        if stmt and "id" in stmt:
            stmt_id = stmt["id"]
            if stmt_id not in visited:
                visited.add(stmt_id)
                flat_list.append(stmt)
        for child in node.get("deps", []):
            if isinstance(child, dict):
                dfs(child)

    dfs(dep_tree)
    if dep_tree.get("stmt", {}) in flat_list:
        flat_list.remove(dep_tree.get("stmt", {}))
    return flat_list

def main():
    # Example debug entry point — pass paths via env or edit before running.
    compdb_path = os.environ.get("COMPDB_PATH", "compile_commands.json")
    root_dir = os.environ.get("PROJECT_ROOT", "./build")
    ddg = extract_ddg(compdb_path=compdb_path, root_dir=root_dir)
    _debug_print(extract_def_tree(os.path.join(root_dir, "test_normal.cpp"), 21, ddg, True))

if __name__ == "__main__":
    main()