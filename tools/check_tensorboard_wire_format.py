#!/usr/bin/env python3
"""
check_tensorboard_wire_format.py
================================

Drift check between the reference proto spec at
``docs/wire_formats/tensorboard_graph.proto`` and the hand-rolled wire-format
encoder in ``src/utils/tensorboard.cpp``.

Why this exists
---------------
The .proto is documentation only — it is not run through ``protoc`` and no
build target consumes it. That means the encoder can silently drift from the
spec (e.g. someone changes ``NodeDef.input`` from field 3 to field 4 in the
.proto but forgets to change ``write_string_field(node, 3, in)`` in the C++,
or vice versa).

What it checks
--------------
For each message that the encoder actually emits (``GraphDef``, ``NodeDef``),
the script:

1. Parses the .proto with a simple line-oriented regex parser (no protoc, no
   ``google.protobuf`` dependency). For every ``<type> <name> = <num>;`` line
   inside a message, it derives the wire type from the protobuf scalar type:

       wire 0 (varint)            : int32/int64/uint32/uint64/bool/enum
       wire 1 (64-bit)            : fixed64/sfixed64/double
       wire 2 (length-delimited)  : string/bytes/message/repeated-anything
       wire 5 (32-bit)            : fixed32/sfixed32/float

   ``repeated`` scalar fields without ``[packed = true]`` would normally keep
   their scalar wire type, but the encoder only emits length-delimited or
   string fields for the messages we cross-check, so this is a non-issue here.

2. Extracts the matching ``(field_number, wire_type)`` pairs from
   ``src/utils/tensorboard.cpp`` by scanning the bodies of ``encode_node_def``
   (→ NodeDef) and ``build_graph_def`` (→ GraphDef wrapping of NodeDef).

3. Compares the two sets per message and exits non-zero on any disagreement.

Run
---
    python tools/check_tensorboard_wire_format.py

Exit codes
----------
    0  all checked messages match
    1  drift detected (printed per-field) or required files missing
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# Locations
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
PROTO_PATH = REPO_ROOT / "docs" / "wire_formats" / "tensorboard_graph.proto"
ENCODER_PATH = REPO_ROOT / "src" / "utils" / "tensorboard.cpp"


# ---------------------------------------------------------------------------
# Wire-type derivation
# ---------------------------------------------------------------------------

# Scalar protobuf type -> wire type (per
# https://protobuf.dev/programming-guides/encoding/#structure).
SCALAR_WIRE = {
    "int32":    0, "int64":    0,
    "uint32":   0, "uint64":   0,
    "sint32":   0, "sint64":   0,
    "bool":     0,
    "enum":     0,
    "fixed64":  1, "sfixed64": 1, "double": 1,
    "string":   2, "bytes":    2,
    "fixed32":  5, "sfixed32": 5, "float":  5,
}


def wire_for(proto_type: str, repeated: bool, packed: bool) -> int:
    """Return the wire type used on the *outer* tag for a field.

    Packed-repeated scalars and any message-typed field are length-delimited
    (wire type 2). Non-packed repeated scalars keep their scalar wire type but
    are emitted once per element.
    """
    if packed and repeated and proto_type in SCALAR_WIRE:
        return 2
    if proto_type in SCALAR_WIRE:
        return SCALAR_WIRE[proto_type]
    # Anything else is a (nested) message reference -> length-delimited.
    return 2


# ---------------------------------------------------------------------------
# .proto parser
# ---------------------------------------------------------------------------

@dataclass
class ProtoField:
    name: str
    number: int
    wire: int


@dataclass
class ProtoMessage:
    name: str
    fields: dict[int, ProtoField] = field(default_factory=dict)


_FIELD_RE = re.compile(
    r"""
    ^\s*
    (?P<label>repeated\s+|optional\s+|required\s+)?          # optional label
    (?P<type>[A-Za-z_][A-Za-z0-9_.<>,\s]*?)                  # type (may be 'map<...>')
    \s+
    (?P<name>[A-Za-z_][A-Za-z0-9_]*)
    \s*=\s*
    (?P<num>\d+)
    \s*
    (?:\[(?P<opts>[^\]]*)\])?                                # optional [packed = true]
    \s*;
    """,
    re.VERBOSE,
)


def parse_proto(path: Path) -> dict[str, ProtoMessage]:
    """Very small proto parser. Handles:

    - ``message Foo { ... }`` blocks (and nested messages — flattened by
      tracking brace depth, but only top-level message names are exposed).
    - ``<type> <name> = <num>;`` lines, with optional ``[packed = true]``.
    - ``map<K, V> name = num;`` (treated as length-delimited, wire 2).
    - ``oneof`` blocks (their inner fields are treated as ordinary fields of
      the containing message; this matches the wire format).
    - Comments (``//`` and ``/* ... */``) are stripped before matching.
    """
    text = path.read_text()
    # Strip /* */ block comments.
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    # Strip // line comments.
    text = re.sub(r"//[^\n]*", "", text)

    messages: dict[str, ProtoMessage] = {}
    # Stack of (name, ProtoMessage) so we attach fields to the innermost
    # message currently open.
    stack: list[ProtoMessage] = []
    # We treat 'oneof <name> {' as a transparent wrapper — its fields go on
    # the enclosing message. Use a depth counter to know when to ignore the
    # matching closing brace.
    oneof_depth = 0

    lines = text.splitlines()
    for raw in lines:
        line = raw.strip()
        if not line:
            continue

        # Open block?
        m_msg = re.match(r"^message\s+([A-Za-z_][A-Za-z0-9_]*)\s*\{", line)
        if m_msg:
            msg = ProtoMessage(name=m_msg.group(1))
            # Only expose top-level messages (depth 0 before push).
            if not stack:
                messages[msg.name] = msg
            stack.append(msg)
            continue

        if re.match(r"^oneof\s+[A-Za-z_][A-Za-z0-9_]*\s*\{", line):
            oneof_depth += 1
            continue

        if line == "}":
            if oneof_depth > 0:
                oneof_depth -= 1
            elif stack:
                stack.pop()
            continue

        if not stack:
            continue  # top-level statements like `syntax`, `package`.

        # map<K, V> name = num;
        m_map = re.match(
            r"^map\s*<\s*[^>]+\s*>\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(\d+)\s*;",
            line,
        )
        if m_map:
            name = m_map.group(1)
            num = int(m_map.group(2))
            stack[-1].fields[num] = ProtoField(name=name, number=num, wire=2)
            continue

        m = _FIELD_RE.match(line)
        if not m:
            continue
        label = (m.group("label") or "").strip()
        proto_type = m.group("type").strip()
        name = m.group("name")
        num = int(m.group("num"))
        opts = m.group("opts") or ""
        packed = "packed" in opts and "true" in opts
        repeated = label == "repeated"
        wire = wire_for(proto_type, repeated, packed)
        stack[-1].fields[num] = ProtoField(name=name, number=num, wire=wire)

    return messages


# ---------------------------------------------------------------------------
# Encoder scraper
# ---------------------------------------------------------------------------

@dataclass
class EncoderField:
    number: int
    wire: int
    call: str  # The call form, for error messages.


def _slice_function(text: str, signature_prefix: str) -> str | None:
    """Return the body (between the first { and matching }) of the first
    function whose declaration starts with ``signature_prefix``."""
    idx = text.find(signature_prefix)
    if idx < 0:
        return None
    brace = text.find("{", idx)
    if brace < 0:
        return None
    depth = 0
    for i in range(brace, len(text)):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[brace + 1 : i]
    return None


# Buffer argument may be a plain identifier (`event`) or a member access
# (`out.bytes`, `impl_->buffer`), and there may be a `/*field_number=*/`-style
# inline comment in front of any literal argument.
_BUF = r"[A-Za-z_][\w.\->]*"
_INLINE_COMMENT = r"(?:/\*[^*]*\*/\s*)?"

_HEADER_RE = re.compile(
    rf"write_field_header\s*\(\s*{_BUF}\s*,\s*{_INLINE_COMMENT}(\d+)\s*,"
    rf"\s*{_INLINE_COMMENT}(\d+)\s*\)"
)
_STRING_RE = re.compile(
    rf"write_string_field\s*\(\s*{_BUF}\s*,\s*{_INLINE_COMMENT}(\d+)\s*,"
)
_LENDELIM_RE = re.compile(
    rf"write_length_delimited\s*\(\s*{_BUF}\s*,\s*{_INLINE_COMMENT}(\d+)\s*,"
)


def extract_encoder_fields(body: str) -> list[EncoderField]:
    out: list[EncoderField] = []
    for m in _HEADER_RE.finditer(body):
        out.append(EncoderField(int(m.group(1)), int(m.group(2)),
                                "write_field_header"))
    for m in _STRING_RE.finditer(body):
        out.append(EncoderField(int(m.group(1)), 2, "write_string_field"))
    for m in _LENDELIM_RE.finditer(body):
        out.append(EncoderField(int(m.group(1)), 2, "write_length_delimited"))
    return out


def encoder_field_sets(encoder_text: str) -> dict[str, list[EncoderField]]:
    """Map message name -> list of (field_number, wire_type) emitted by the
    encoder helper responsible for that message."""
    nd_body = _slice_function(encoder_text, "encode_node_def")
    gd_body = _slice_function(encoder_text, "build_graph_def")
    result: dict[str, list[EncoderField]] = {}
    if nd_body is not None:
        result["NodeDef"] = extract_encoder_fields(nd_body)
    if gd_body is not None:
        # build_graph_def only writes the *outer* wrapping of each NodeDef
        # into the GraphDef (field 1, length-delimited). It also calls
        # encode_node_def which we already attributed to NodeDef, so restrict
        # to the wrapper write here.
        wrapper: list[EncoderField] = []
        for m in _LENDELIM_RE.finditer(gd_body):
            wrapper.append(EncoderField(int(m.group(1)), 2,
                                        "write_length_delimited"))
        result["GraphDef"] = wrapper
    return result


# ---------------------------------------------------------------------------
# Cross-check
# ---------------------------------------------------------------------------

def check(messages: dict[str, ProtoMessage],
          encoder: dict[str, list[EncoderField]]) -> int:
    """Return number of failures."""
    failures = 0

    # Messages we expect the encoder to emit and the proto to define.
    REQUIRED = ("GraphDef", "NodeDef")

    for msg_name in REQUIRED:
        spec = messages.get(msg_name)
        emit = encoder.get(msg_name, [])
        if spec is None:
            print(f"[FAIL] {msg_name}: not found in {PROTO_PATH.name}")
            failures += 1
            continue
        if not emit:
            print(f"[FAIL] {msg_name}: encoder helper not found in "
                  f"{ENCODER_PATH.name}")
            failures += 1
            continue

        # Reduce the encoder list to the unique (number, wire) set, since
        # repeated fields get emitted in a loop.
        emitted_pairs = {(e.number, e.wire) for e in emit}

        msg_failures = 0
        for number, wire in sorted(emitted_pairs):
            spec_field = spec.fields.get(number)
            if spec_field is None:
                print(f"[FAIL] {msg_name}: encoder writes field {number} "
                      f"(wire {wire}) but {PROTO_PATH.name} has no such field")
                msg_failures += 1
                continue
            if spec_field.wire != wire:
                print(f"[FAIL] {msg_name}.{spec_field.name} (field "
                      f"{number}): spec wire={spec_field.wire}, "
                      f"encoder wire={wire}")
                msg_failures += 1

        if msg_failures == 0:
            emitted_str = ", ".join(
                f"{n}/{w}" for n, w in sorted(emitted_pairs)
            )
            print(f"[ OK ] {msg_name}: encoder emits fields {{{emitted_str}}} "
                  f"matching spec")
        failures += msg_failures

    # Informational: messages defined in the spec but not yet cross-checkable
    # because the encoder doesn't emit them. This is not a failure — it's a
    # heads-up for whoever extends the encoder.
    informational = sorted(set(messages) - set(REQUIRED))
    if informational:
        print(f"[INFO] spec messages with no encoder yet (no drift check "
              f"performed): {', '.join(informational)}")

    return failures


def main() -> int:
    if not PROTO_PATH.exists():
        print(f"[FAIL] missing spec: {PROTO_PATH}")
        return 1
    if not ENCODER_PATH.exists():
        print(f"[FAIL] missing encoder: {ENCODER_PATH}")
        return 1

    messages = parse_proto(PROTO_PATH)
    encoder = encoder_field_sets(ENCODER_PATH.read_text())
    failures = check(messages, encoder)

    if failures:
        print(f"\nFAIL: {failures} field(s) drifted between "
              f"{PROTO_PATH.relative_to(REPO_ROOT)} and "
              f"{ENCODER_PATH.relative_to(REPO_ROOT)}")
        return 1
    print(f"\nPASS: wire format in sync between "
          f"{PROTO_PATH.relative_to(REPO_ROOT)} and "
          f"{ENCODER_PATH.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
