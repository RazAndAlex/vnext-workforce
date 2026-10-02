"""Which field of a tool's input says what one approval is about.

A reviewer told only "WebFetch" cannot tell a docs page from an upload target,
so every approval carries one subject: the file for a read, the URL for a
fetch, the query for a search.  Picking that field by a fixed scan of candidate
names records the wrong one as soon as an input carries more than one of them.
A probe sent WebFetch a real ``url`` and a decoy ``file_path``, and the decoy
was what the audit record named.  So the field is chosen by the tool that is
asking, and the scan survives only as the answer for a tool nobody named here.

Both ends of the approval path read this one table: the Claude bridge, which
fills the envelope, and the scheduler, which records what it saw.  Deleting
this module and its two imports removes the whole mechanism.
"""

from __future__ import annotations

import unicodedata
from typing import Mapping

# One entry per tool whose input this project knows the shape of.  A tool with
# several keys names them in reading order and the subject carries each one
# that is present: a Grep is its pattern, and where it searched when it said.
SUBJECT_KEYS_BY_TOOL: dict[str, tuple[str, ...]] = {
    "WebFetch": ("url",),
    "WebSearch": ("query",),
    "Read": ("file_path",),
    "NotebookRead": ("notebook_path",),
    "Grep": ("pattern", "path"),
    "Glob": ("pattern", "path"),
}

# The answer for a tool this table does not name: an older bridge, a provider
# with tools of its own, a rename upstream.  A first match is a guess, which is
# why it is reached only when the tool itself said nothing.
UNNAMED_TOOL_SUBJECT_KEYS: tuple[str, ...] = (
    "file_path",
    "path",
    "url",
    "query",
    "pattern",
)


def approval_subject(
    tool_name: str, fields: Mapping[str, object], limit: int
) -> str | None:
    """Name what one tool call is about, or name nothing.

    Nothing is returned when the input said nothing, so a caller leaves the
    key absent rather than empty: an operator must not read a silent provider
    as a read of the root directory.  A malformed field is read as an absent
    one and never raises.
    """

    keys = SUBJECT_KEYS_BY_TOOL.get(tool_name)
    if keys is None:
        keys = UNNAMED_TOOL_SUBJECT_KEYS
        first_match_only = True
    else:
        first_match_only = False
    parts: list[str] = []
    for key in keys:
        value = fields.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(value)
            if first_match_only:
                break
    if not parts:
        return None
    return " ".join(parts)[:limit]


# A backslash is left alone: a Windows path is full of them and mangling
# "C:\\Program Files" to make a newline visible trades one unreadable subject
# for another.  Only the characters that can end a line or move a cursor are
# replaced.
_NAMED_ESCAPES = {"\n": "\\n", "\r": "\\r", "\t": "\\t", "\x0b": "\\v", "\x0c": "\\f"}
_UNPRINTABLE_CATEGORIES = frozenset({"Cc", "Zl", "Zp"})


def visible_on_one_line(text: str) -> str:
    """Render a subject so it cannot become two lines wherever it is shown.

    A provider-shaped subject is whatever the provider sent.  One with a
    newline in it turned a single audit line into two, the second of which read
    like a record of its own.  The raw value stays in the structured field; this
    is what gets read as a line.
    """

    rendered: list[str] = []
    for character in text:
        named = _NAMED_ESCAPES.get(character)
        if named is not None:
            rendered.append(named)
        elif unicodedata.category(character) in _UNPRINTABLE_CATEGORIES:
            code = ord(character)
            rendered.append(
                f"\\x{code:02x}" if code < 0x100 else f"\\u{code:04x}"
            )
        else:
            rendered.append(character)
    return "".join(rendered)
