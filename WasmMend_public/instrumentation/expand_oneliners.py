#!/usr/bin/env python3
"""Expand one-line C/C++ function bodies across multiple lines.

Copied from tests/OpenCC/scripts/expand_oneliners.py so the preprocessor
can import it directly instead of shelling out to a python subprocess.

Matches lines of the form:
    <indent><signature> { stmt1; stmt2; ... }
and rewrites them as:
    <indent><signature> {
    <indent>
    <indent>
    <indent>    stmt1;
    <indent>
    <indent>
    <indent>    stmt2;
    <indent>
    <indent>
    <indent>}

Rationale: extra blank lines between the opening `{`, each statement, and the
closing `}` give instrumentation pipelines room to insert their `/* ... */`
marker comments without silently swallowing adjacent statements. A previous
instrumentation pass on rapidjson's document.h stripped `++ptr_;` from
`operator++()` because the instrumenter collapsed everything between the
opening brace and the return statement -- leaving iterators that never
advance and infinite loops during JSON parsing.

What is skipped:
- Lines inside /* ... */ block comments.
- Heads that start with struct/class/enum/union/namespace/typedef/using or
  control-flow keywords (if/else/for/while/switch/return/throw/case).
- Heads without both `(` and `)` (i.e., not a function signature).
- Bodies with zero statements (empty `{}`).
- Lines with nested braces in the body (we only expand single-line brace
  pairs -- multi-line bodies are already expanded).

Public API:
    expand(path)                -> number of expansions applied
    expand_files(paths)         -> total expansions across all paths
    split_statements(body)      -> list of statements (used internally)
"""
import re
import sys


def split_statements(body):
    """Split body on top-level semicolons, preserving the `;` on each stmt."""
    stmts = []
    depth_paren = depth_angle = 0
    buf = []
    for ch in body:
        buf.append(ch)
        if ch == '(':
            depth_paren += 1
        elif ch == ')':
            depth_paren -= 1
        elif ch == '<':
            depth_angle += 1
        elif ch == '>':
            depth_angle = max(0, depth_angle - 1)
        elif ch == ';' and depth_paren == 0:
            stmts.append(''.join(buf).strip())
            buf = []
    tail = ''.join(buf).strip()
    if tail:
        stmts.append(tail)
    return stmts


PATTERN = re.compile(r'^(\s*)(.*?)\{([^{}]*?)\}\s*$')
SKIP_HEAD_KEYWORDS = {
    'struct', 'class', 'enum', 'union', 'namespace', 'typedef', 'using',
    'if', 'else', 'for', 'while', 'switch', 'return', 'throw', 'case', 'do',
}


def expand(path):
    """Expand compact one-line function bodies in *path* (in place).

    Returns the number of lines rewritten.
    """
    with open(path, 'r') as f:
        lines = f.readlines()

    out = []
    in_block_comment = False
    changes = 0

    for line in lines:
        stripped = line.rstrip('\n')

        if in_block_comment:
            if '*/' in stripped:
                in_block_comment = False
            out.append(line)
            continue

        has_open = '/*' in stripped
        has_close = '*/' in stripped
        if has_open and not has_close:
            in_block_comment = True
            out.append(line)
            continue

        # Separate trailing `//` comment from code to be matched.
        code = stripped
        comment_idx = -1
        in_str = False
        i = 0
        while i < len(code):
            c = code[i]
            if c == '"' and (i == 0 or code[i - 1] != '\\'):
                in_str = not in_str
            elif not in_str and c == '/' and i + 1 < len(code) and code[i + 1] == '/':
                comment_idx = i
                break
            i += 1
        match_target = code[:comment_idx] if comment_idx >= 0 else code
        trailing_comment = code[comment_idx:] if comment_idx >= 0 else ''

        m = PATTERN.match(match_target)
        if not m:
            out.append(line)
            continue

        indent, head, body = m.group(1), m.group(2), m.group(3)
        head_stripped = head.strip()
        if not head_stripped:
            out.append(line)
            continue
        if '(' not in head_stripped or ')' not in head_stripped:
            out.append(line)
            continue
        first_tok = re.match(r'\w+', head_stripped)
        if first_tok and first_tok.group(0) in SKIP_HEAD_KEYWORDS:
            out.append(line)
            continue

        stmts = split_statements(body)
        if not stmts:
            out.append(line)
            continue

        new_indent = indent + '    '
        new_lines = [f"{indent}{head.rstrip()} {{\n", "\n", "\n"]
        for s in stmts:
            new_lines.append(f"{new_indent}{s}\n")
            new_lines.append("\n")
            new_lines.append("\n")
        new_lines.append(f"{indent}}}")
        if trailing_comment:
            new_lines[-1] += ' ' + trailing_comment
        new_lines[-1] += '\n'
        out.extend(new_lines)
        changes += 1

    with open(path, 'w') as f:
        f.writelines(out)
    return changes


def expand_files(paths):
    """Run :func:`expand` on every path in *paths*; return total changes."""
    total = 0
    for p in paths:
        try:
            n = expand(p)
        except Exception as e:
            print(f"  [expand_oneliners] skip {p}: {e}")
            continue
        if n:
            print(f"  [expand_oneliners] {p}: expanded {n} one-line body")
        total += n
    return total


def main():
    if len(sys.argv) < 2:
        print("usage: expand_oneliners.py <file> [<file> ...]", file=sys.stderr)
        sys.exit(2)
    total = expand_files(sys.argv[1:])
    print(f"total: {total}")


if __name__ == '__main__':
    main()
