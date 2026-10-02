"""Private, vNext-owned configuration for non-subscription providers.

The catalog asks this boundary only whether a provider is runnable.  Each
bridge is the sole consumer of its own API key, so a credential never becomes
session configuration, an adapter field, or a run-log payload.

Every provider here is read from the same vNext-owned file.  A key that also
exists in another product's configuration is copied into this one by hand:
vNext does not read another product's provider configuration at runtime.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


PROVIDER_CONFIG_PATH = Path.home() / ".vnext" / "providers.json"
ZAI_ENDPOINT = "https://api.z.ai/api/anthropic"
_MAX_PROVIDER_CONFIG_BYTES = 65_536


@dataclass(frozen=True, repr=False)
class ZaiProviderConfig:
    """Validated z.ai credential material whose representation stays secret-free."""

    api_key: str


@dataclass(frozen=True, repr=False)
class CommandCodeProviderConfig:
    """Validated Command Code credential material, equally secret-free."""

    api_key: str


def _read_api_key(provider: str, path: str | Path | None) -> str | None:
    """Return one provider's key from the vNext-owned file, or ``None``.

    Provider configuration is an optional local capability.  Missing,
    unreadable, oversized, malformed, or incomplete files all mean the same
    thing to the catalog: do not offer that provider's models.
    """

    target = Path(path) if path is not None else PROVIDER_CONFIG_PATH
    try:
        if target.stat().st_size > _MAX_PROVIDER_CONFIG_BYTES:
            return None
        value: Any = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    providers = value.get("providers") if isinstance(value, Mapping) else None
    entry = providers.get(provider) if isinstance(providers, Mapping) else None
    api_key = entry.get("api_key") if isinstance(entry, Mapping) else None
    if (
        not isinstance(api_key, str)
        or not api_key
        or api_key.strip() != api_key
        or len(api_key) > 16_384
    ):
        return None
    return api_key


KNOWN_PROVIDERS = ("zai", "commandcode")


def _display(path: Path) -> str:
    """The path as the README writes it, with the home folder as ``~``."""

    try:
        return f"~/{path.relative_to(Path.home())}"
    except ValueError:
        return str(path)


def describe_provider_config_problem(
    provider: str, path: str | Path | None = None
) -> str | None:
    """Say what is wrong with the provider file, or ``None`` when nothing is.

    The catalog needs one answer -- is this provider runnable -- and a person
    whose start failed needs another: which line of the file they hand-wrote is
    wrong.  ``_read_api_key`` answers the first by reading every fault as "no
    key", which left ``--check`` telling a user with a trailing comma in the
    file to go and configure a key.  This answers the second.  A missing file is
    not a fault: it is the normal state of a machine that uses no such provider.

    The key itself is never part of an answer.
    """

    target = Path(path) if path is not None else PROVIDER_CONFIG_PATH
    shown = _display(target)
    try:
        size = target.stat().st_size
    except OSError:
        # Absent, or a folder component that is not one: nothing to report.
        return None
    if size > _MAX_PROVIDER_CONFIG_BYTES:
        return (
            f"{shown} is larger than {_MAX_PROVIDER_CONFIG_BYTES} bytes, "
            f"so it was not read"
        )
    try:
        raw = target.read_text(encoding="utf-8")
    except UnicodeError:
        return f"{shown} could not be read: it is not valid UTF-8 text"
    except OSError as exc:
        return f"{shown} could not be read: {exc.strerror or exc}"
    try:
        value: Any = json.loads(raw)
    except json.JSONDecodeError as exc:
        line, column, reason = exc.lineno, exc.colno, exc.msg
        comma = _trailing_comma_before(raw, exc.pos)
        if comma is not None:
            # Python 3.13 names a trailing comma and points at it.  Earlier
            # versions point at the closing bracket after it, often a line
            # later, so name the comma here for every supported version.
            line = raw.count("\n", 0, comma) + 1
            column = comma - (raw.rfind("\n", 0, comma) + 1) + 1
            reason = "trailing comma before the closing bracket"
        return f"{shown} could not be read: line {line} column {column}: {reason}"
    if not isinstance(value, Mapping):
        return f"{shown} does not hold a JSON object"
    providers = value.get("providers")
    if not isinstance(providers, Mapping):
        return f"{shown} has no 'providers' object"
    if provider not in providers:
        unknown = sorted(name for name in providers if name not in KNOWN_PROVIDERS)
        if not unknown:
            return None
        named = ", ".join(repr(name) for name in unknown)
        verb = "is not a known provider" if len(unknown) == 1 else "are not known providers"
        return f"{shown}: {named} {verb} (known: {', '.join(KNOWN_PROVIDERS)})"
    entry = providers[provider]
    if not isinstance(entry, Mapping):
        return f"{shown}: providers.{provider} is not an object"
    api_key = entry.get("api_key")
    if not isinstance(api_key, str):
        found = ", ".join(sorted(str(name) for name in entry)) or "nothing"
        return (
            f"{shown}: providers.{provider} has no 'api_key' string "
            f"(found: {found})"
        )
    if not api_key:
        return f"{shown}: providers.{provider}.api_key is empty"
    if api_key.strip() != api_key:
        return (
            f"{shown}: providers.{provider}.api_key has a space or a newline "
            f"at one end"
        )
    if len(api_key) > 16_384:
        return f"{shown}: providers.{provider}.api_key is longer than 16384 characters"
    return None


def provider_config_problems() -> list[str]:
    """Every distinct fault the file holds, for a start that offers no model."""

    found: list[str] = []
    for provider in KNOWN_PROVIDERS:
        problem = describe_provider_config_problem(provider)
        if problem is not None and problem not in found:
            found.append(problem)
    return found


def load_zai_provider(path: str | Path | None = None) -> ZaiProviderConfig | None:
    """Return a usable z.ai entry, or quietly report that none is available."""

    api_key = _read_api_key("zai", path)
    return None if api_key is None else ZaiProviderConfig(api_key=api_key)


def load_commandcode_provider(
    path: str | Path | None = None,
) -> CommandCodeProviderConfig | None:
    """Return a usable Command Code entry, or quietly report that none is available.

    Another tool may already hold the same key in its own configuration.  vNext
    reads only this file, because a provider vNext runs is a provider vNext owns
    the credential for.
    """

    api_key = _read_api_key("commandcode", path)
    return None if api_key is None else CommandCodeProviderConfig(api_key=api_key)


def _trailing_comma_before(raw: str, position: int) -> int | None:
    """Index of a comma that only whitespace separates from a closing bracket at ``position``."""

    if not 0 <= position < len(raw) or raw[position] not in "}]":
        return None
    index = position - 1
    while index >= 0 and raw[index] in " \t\r\n":
        index -= 1
    return index if index >= 0 and raw[index] == "," else None
