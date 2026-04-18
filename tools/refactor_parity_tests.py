#!/usr/bin/env python3
"""
Mechanical refactor of backend_parity Category A files from helper-fn pattern
to TEST_P + INSTANTIATE_BACKEND_TESTS pattern.

Transformations applied per file:
  1. Add `#include "../backend_test_fixture.hpp"` after parity_test_utils.hpp.
  2. Detect all `TEST(SuiteName, ...)` suites; for each, insert
     `class SuiteName : public BackendTest {};` once.
  3. Replace `TEST(SuiteName,` → `TEST_P(SuiteName,`.
  4. Strip the boilerplate skip block:
        auto backends = get_available_backends();
        if (backends.size() < 2) GTEST_SKIP();
        (or with inline body before/after the call)
  5. Replace `test_operation_parity(<args>)` and
     `test_operation_parity_backends(<args>)` with
     `test_operation_parity_single(<lambda>, <inputs>, device, <rtol>, <atol>, <name>)`
     by inserting `device,` at argument 3.
  6. Append `INSTANTIATE_BACKEND_TESTS(SuiteName);` for each detected suite,
     placed just before the `int main(...)` function.

The script reports per-file: TRANSFORMED | SKIPPED-COMPLEX | SKIPPED-MISSING-PATTERN
so manual review can focus on the rejected files.

Usage: python3 tools/refactor_parity_tests.py <file1> [<file2> ...]
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


# Match the boilerplate skip block including its trailing newline. The
# leading `[ \t]*` (no \n) is critical: if we let `\s*` consume a previous
# newline, a comment on the line above will get joined onto the line below
# and silently swallow code.
SKIP_RE = re.compile(
    r'[ \t]*auto\s+backends\s*=\s*get_available_backends\(\)\s*;[ \t]*\n'
    r'[ \t]*if\s*\(\s*backends\.size\(\)\s*<\s*2\s*\)\s*GTEST_SKIP\(\)\s*;[ \t]*\n',
)

# `test_operation_parity` and `test_operation_parity_backends` are recognized.
# We want to insert `device,` as the third argument. Match a call by paren
# pairing — naive token splitting is unsafe across nested parens / lambdas.

def find_test_suite_names(text: str) -> list[str]:
    """Return all distinct first-arg names from TEST(...) macros."""
    return sorted(set(re.findall(r'\bTEST\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*,', text)))


def add_fixture_include(text: str) -> str:
    """Insert backend_test_fixture.hpp include after parity_test_utils.hpp."""
    if '../backend_test_fixture.hpp' in text or 'backend_test_fixture.hpp' in text:
        return text
    return text.replace(
        '#include "parity_test_utils.hpp"',
        '#include "../backend_test_fixture.hpp"\n#include "parity_test_utils.hpp"',
        1,
    )


def add_class_definitions(text: str, suites: list[str]) -> str:
    """Insert `class Suite : public BackendTest {};` once per suite, after the
    last `using namespace ...` line."""
    # Find a good insertion point: after the last "using namespace" line.
    using_iter = list(re.finditer(r'using\s+namespace\s+[a-zA-Z0-9_:]+\s*;\s*\n', text))
    if not using_iter:
        # Fall back to top-of-file after includes.
        last_include = list(re.finditer(r'#include[^\n]*\n', text))
        if not last_include:
            return text
        insert_at = last_include[-1].end()
    else:
        insert_at = using_iter[-1].end()
    block = '\n'
    for s in suites:
        block += f'class {s} : public BackendTest {{}};\n'
    return text[:insert_at] + block + text[insert_at:]


def convert_test_macros(text: str) -> str:
    """Replace TEST(Suite, → TEST_P(Suite, for every TEST."""
    return re.sub(r'\bTEST\s*\(', 'TEST_P(', text)


def strip_skip_blocks(text: str) -> str:
    """Remove `auto backends = ...; if (backends.size() < 2) GTEST_SKIP();`."""
    return SKIP_RE.sub('', text)


def insert_device_arg(text: str, fn_name: str, new_name: str | None = None,
                       replace_third_arg: bool = False) -> str:
    """Rewrite every `fn_name(...)` call.

    By default: insert `device,` between args 2 and 3 →
        new_name(arg1, arg2, device, arg3, arg4, arg5).

    With replace_third_arg=True: REPLACE arg3 with `device` →
        new_name(arg1, arg2, device, arg4, arg5).
    Use this when the original third arg is `backends` (a vector) that the
    `_single` variant doesn't take.

    `{a, b}` initializer lists, `vector<T, U>` templates, and `[](...) { ... }`
    lambda bodies are all skipped over so we don't mistake their commas for
    the call's argument separators.
    """
    if new_name is None:
        new_name = fn_name
    out_chunks: list[str] = []
    cursor = 0
    pattern = re.compile(rf'\b{re.escape(fn_name)}\s*\(')
    while True:
        m = pattern.search(text, cursor)
        if m is None:
            out_chunks.append(text[cursor:])
            break
        # Walk depth of paren / brace / bracket / angle. A top-level comma is
        # one where ALL trackers are at zero (we start at paren=1 from the
        # opening of the fn_name call).
        i = m.end()
        depth_paren = 1
        depth_brace = 0
        depth_brack = 0
        depth_angle = 0
        commas: list[int] = []
        end_pos = -1
        while i < len(text):
            ch = text[i]
            # Skip char/string literals so commas inside don't count.
            if ch == '"' or ch == "'":
                quote = ch
                i += 1
                while i < len(text):
                    if text[i] == '\\':
                        i += 2
                        continue
                    if text[i] == quote:
                        break
                    i += 1
                i += 1
                continue
            # Skip line comments.
            if ch == '/' and i + 1 < len(text) and text[i+1] == '/':
                while i < len(text) and text[i] != '\n':
                    i += 1
                continue
            # Skip block comments.
            if ch == '/' and i + 1 < len(text) and text[i+1] == '*':
                i += 2
                while i + 1 < len(text) and not (text[i] == '*' and text[i+1] == '/'):
                    i += 1
                i += 2
                continue
            if ch == '(':
                depth_paren += 1
            elif ch == ')':
                depth_paren -= 1
                if depth_paren == 0:
                    end_pos = i
                    break
            elif ch == '{':
                depth_brace += 1
            elif ch == '}':
                depth_brace -= 1
            elif ch == '[':
                depth_brack += 1
            elif ch == ']':
                depth_brack -= 1
            elif ch == ',':
                if (depth_paren == 1 and depth_brace == 0
                        and depth_brack == 0 and depth_angle == 0):
                    commas.append(i)
            i += 1
        min_commas = 3 if replace_third_arg else 2
        if end_pos < 0 or len(commas) < min_commas:
            # Malformed, lambda swallowed depth, or too few args; leave alone.
            out_chunks.append(text[cursor:m.end()])
            cursor = m.end()
            continue
        call_start = m.start()
        out_chunks.append(text[cursor:call_start])
        out_chunks.append(new_name)
        out_chunks.append(text[m.end()-1:m.end()])  # the '('
        if replace_third_arg:
            # Emit args 1 and 2 (text up to and incl. comma2), then `device,`,
            # then args 4..N (text from after comma3 to ')').
            out_chunks.append(text[m.end():commas[1] + 1])  # up to comma2 (incl)
            out_chunks.append(' device,')
            out_chunks.append(text[commas[2] + 1:end_pos + 1])  # comma3+1..')'
        else:
            # Insert: keep all args, just splice `device,` after comma2.
            out_chunks.append(text[m.end():commas[1] + 1])  # up to comma2 (incl)
            out_chunks.append(' device,')
            out_chunks.append(text[commas[1] + 1:end_pos + 1])  # remaining args + ')'
        cursor = end_pos + 1
    return ''.join(out_chunks)


def append_instantiations(text: str, suites: list[str]) -> str:
    """Insert `INSTANTIATE_BACKEND_TESTS(Suite);` for each suite, just before
    `int main(`."""
    block = '\n'
    for s in suites:
        block += f'INSTANTIATE_BACKEND_TESTS({s});\n'
    block += '\n'
    main_match = re.search(r'^\s*int\s+main\s*\(', text, flags=re.MULTILINE)
    if main_match is None:
        return text + block  # no main(); just append at end.
    return text[:main_match.start()] + block + text[main_match.start():]


def refactor_file(path: Path, dry_run: bool = False,
                   conservative: bool = False) -> str:
    text = path.read_text()
    suites = find_test_suite_names(text)
    if not suites:
        return f'SKIPPED-MISSING-PATTERN ({path.name}): no TEST() found'

    if 'TEST_F(' in text or 'TEST_P(' in text:
        return f'SKIPPED-COMPLEX ({path.name}): already uses TEST_F/TEST_P'

    # Detect mixed-pattern files (manual backend loops). In conservative mode
    # we still convert these — the body keeps working because the test_*
    # helper functions don't depend on the fixture's `device`. In strict mode
    # we reject them.
    cleaned = SKIP_RE.sub('', text)
    cleaned = re.sub(r',\s*backends\s*,', ', BACKENDS_REPLACED,', cleaned)
    has_manual_loops = bool(re.search(
        r'\bbackends\.size\b|\bbackends\[\b|for\s*\(.*backends\b', cleaned))
    if has_manual_loops and not conservative:
        return (f'SKIPPED-COMPLEX ({path.name}): references `backends` outside '
                f'standard boilerplate; rerun with --conservative')

    new_text = text
    new_text = add_fixture_include(new_text)
    new_text = add_class_definitions(new_text, suites)
    if not conservative:
        # Strict mode: collapse the SKIP block AND rename helper calls.
        new_text = strip_skip_blocks(new_text)
        new_text = insert_device_arg(new_text, 'test_operation_parity_backends',
                                      'test_operation_parity_single',
                                      replace_third_arg=True)
        new_text = insert_device_arg(new_text, 'test_operation_parity',
                                      'test_operation_parity_single')
    # In conservative mode we leave the body untouched (the SKIP block and
    # the test_operation_parity calls keep working, just run N times instead
    # of once — wasteful but produces correct results and exposes per-backend
    # ctest entries).
    new_text = convert_test_macros(new_text)
    new_text = append_instantiations(new_text, suites)

    if not dry_run:
        path.write_text(new_text)
    mode = 'CONSERVATIVE' if conservative else 'STRICT'
    return f'TRANSFORMED-{mode} ({path.name}): suites={suites}'


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('files', nargs='+', help='C++ files to refactor')
    parser.add_argument('--dry-run', action='store_true', help='Preview only')
    parser.add_argument('--conservative', action='store_true',
                        help='Convert mixed-pattern files; leave bodies alone.')
    args = parser.parse_args()

    for fname in args.files:
        path = Path(fname)
        if not path.exists():
            print(f'MISSING: {fname}')
            continue
        report = refactor_file(path, dry_run=args.dry_run,
                                conservative=args.conservative)
        print(report)
    return 0


if __name__ == '__main__':
    sys.exit(main())
