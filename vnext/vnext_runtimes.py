"""Keep the Codex CLI and the Claude Agent SDK that vNext runs up to date.

vNext ships pinned to one Codex CLI and one Claude Agent SDK.  Both packages
release almost daily, and a new model usually arrives with a new release, so a
pinned install falls behind within days.  This module adds a side runtime: a
private virtual environment under ``~/.vnext/runtimes/<codex>-<sdk>/`` (or the
folder ``VNEXT_RUNTIMES_DIR`` names) that holds a newer pair.

* ``check_for_updates`` asks PyPI once a day which stable release is newest.
  ``VNEXT_NO_UPDATE_CHECK=1`` turns the question off.
* ``update_runtimes`` installs a pair, hashes both executables, asks each one
  for its models without sending a prompt, and writes ``active.json``.
* ``rollback`` deletes ``active.json`` and keeps the installed folders.

Nothing here restarts anything.  The server reads ``active.json`` when it
starts, so a change takes effect at the next restart.  With no ``active.json``
every function returns early and the pinned runtime runs as before.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from urllib.request import Request, urlopen

from .release_check import RELEASE_CODEX_VERSION

RUNTIMES_ENV = "VNEXT_RUNTIMES_DIR"
NO_UPDATE_CHECK_ENV = "VNEXT_NO_UPDATE_CHECK"
# The bridge reads these two when it runs on a side runtime's Python.
SDK_VERSION_ENV = "VNEXT_CLAUDE_SDK_VERSION"
CLI_PATH_ENV = "VNEXT_CLAUDE_CLI_PATH"
CODEX_PACKAGE = "openai-codex"
SDK_PACKAGE = "claude-agent-sdk"
PYPI_URL = "https://pypi.org/pypi/{name}/json"
ACTIVE_FILE = "active.json"
CACHE_FILE = "update-check.json"
UNTESTED = "not tested by vNext"
UPDATE_FLAG = "--update-runtimes"
ROLLBACK_FLAG = "--rollback"


def is_runtime_command(argv: Sequence[str]) -> bool:
    """True when the arguments ask to install or roll back a side runtime."""

    return UPDATE_FLAG in argv or ROLLBACK_FLAG in argv
RESTART_NOTE = (
    "This takes effect at the next restart of the vNext server. Restart when no "
    "workers are running: a restart stops running workers."
)
_STABLE = re.compile(r"^\d+\.\d+\.\d+$")
_FETCH_TIMEOUT_SECONDS = 4.0
_PROBE_TIMEOUT_SECONDS = 60.0


def _claude_sdk_pin() -> str:
    from .vnext_claude_bridge import CLAUDE_SDK_VERSION

    return CLAUDE_SDK_VERSION


def runtimes_dir() -> Path:
    """Where side runtimes live: ``VNEXT_RUNTIMES_DIR`` or ``~/.vnext/runtimes``."""

    configured = os.environ.get(RUNTIMES_ENV)
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".vnext" / "runtimes"


def _version_key(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in version.split("."))


def stable_newest(payload: Mapping[str, Any]) -> str | None:
    """The newest stable X.Y.Z release in a PyPI JSON answer.

    Alphas, betas, release candidates and fully yanked releases are left out.
    """

    releases = payload.get("releases") if isinstance(payload, Mapping) else None
    if not isinstance(releases, Mapping):
        return None
    stable = []
    for version, files in releases.items():
        if not isinstance(version, str) or not _STABLE.match(version):
            continue
        if isinstance(files, list) and files and all(
            isinstance(item, Mapping) and item.get("yanked") for item in files
        ):
            continue
        stable.append(version)
    return max(stable, key=_version_key) if stable else None


def _fetch_pypi(name: str) -> Mapping[str, Any]:
    request = Request(PYPI_URL.format(name=name), headers={"Accept": "application/json"})
    with urlopen(request, timeout=_FETCH_TIMEOUT_SECONDS) as response:  # noqa: S310 - fixed https URL
        return json.loads(response.read().decode("utf-8"))


def check_for_updates(
    *,
    fetch: Callable[[str], Mapping[str, Any]] | None = None,
    today: str | None = None,
    force: bool = False,
    retry_failed: bool = False,
    cached_only: bool = False,
) -> dict[str, str] | None:
    """Return ``{package: newest stable version}``, or None when not known.

    The answer is cached for the day in the runtimes folder.  A failed check,
    a network that is down included, gives None and is cached for the day as
    well, so an offline machine does not ask again at every start.
    ``retry_failed`` asks again after a failure cached today (``--check``
    does).  ``cached_only`` reads the day's cache and never the network.
    """

    if not force and os.environ.get(NO_UPDATE_CHECK_ENV) == "1":
        return None
    day = today or _dt.date.today().isoformat()
    cache = runtimes_dir() / CACHE_FILE
    if not force:
        try:
            cached = json.loads(cache.read_text(encoding="utf-8"))
            if cached.get("checked") == day and isinstance(cached.get("newest"), dict):
                return {str(k): str(v) for k, v in cached["newest"].items()}
            if cached.get("checked") == day and cached.get("failed") and not retry_failed:
                return None
        except (OSError, ValueError, AttributeError):
            pass
    if cached_only:
        return None
    reader = fetch or _fetch_pypi
    newest: dict[str, str] | None = {}
    try:
        for name in (CODEX_PACKAGE, SDK_PACKAGE):
            version = stable_newest(reader(name))
            if version is None:
                newest = None
                break
            newest[name] = version
    except Exception:  # noqa: BLE001 - an update check never stops a start
        newest = None
    record = {"checked": day, "newest": newest} if newest else {"checked": day, "failed": True}
    try:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(record), encoding="utf-8")
    except OSError:
        pass
    return newest or None


def read_active_runtime() -> dict[str, Any] | None:
    """The side runtime ``active.json`` names, or None for the pinned one."""

    path = runtimes_dir() / ACTIVE_FILE
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:
        print(f"vnext: {path} could not be read ({exc}); the pinned runtime runs", file=sys.stderr)
        return None
    codex = value.get("codex") if isinstance(value, Mapping) else None
    claude = value.get("claude") if isinstance(value, Mapping) else None
    if not _is_runtime_record(codex, claude):
        print(f"vnext: {path} is not a runtime record; the pinned runtime runs", file=sys.stderr)
        return None
    return dict(value)


def _is_runtime_record(codex: Any, claude: Any) -> bool:
    """True when both sections hold every field the server's selections read."""

    if not isinstance(codex, Mapping) or not isinstance(claude, Mapping):
        return False
    required = ((codex, ("executable", "sha256", "version")), (claude, ("python", "sdk_version")))
    for section, keys in required:
        if not all(isinstance(section.get(key), str) and section.get(key) for key in keys):
            return False
    # --update-runtimes always records the CLI it hashed.  A record without
    # one would let the SDK fall back to whichever claude the machine has.
    if not all(isinstance(claude.get(key), str) and claude.get(key) for key in ("cli_path", "cli_sha256")):
        return False
    return all(section.get("models") is None or isinstance(section.get("models"), list)
               for section in (codex, claude))


def runtime_view(active: Mapping[str, Any] | None = None) -> dict[str, str]:
    """Versions and source of the runtime the server runs."""

    if active is None:
        return {
            "codex": f"codex-cli {RELEASE_CODEX_VERSION}",
            "codex_package": RELEASE_CODEX_VERSION,
            "sdk": _claude_sdk_pin(),
            "source": "pinned",
        }
    codex = active["codex"]
    claude = active["claude"]
    return {
        "codex": str(codex.get("version")),
        "codex_package": str(codex.get("package_version")),
        "sdk": str(claude.get("sdk_version")),
        "source": f"side: {active.get('venv')}",
    }


def codex_runtime_selection(active: Mapping[str, Any] | None) -> dict[str, str] | None:
    """What ``resolve_session_runtime`` takes for the side Codex executable."""

    if active is None:
        return None
    codex = active["codex"]
    return {
        "executable": str(codex["executable"]),
        "sha256": str(codex["sha256"]),
        "version": str(codex["version"]),
    }


def claude_runtime_selection(active: Mapping[str, Any] | None) -> dict[str, str] | None:
    """What the Claude adapter needs to run its bridge on the side Python."""

    if active is None:
        return None
    claude = active["claude"]
    selection = {
        "python": str(claude["python"]),
        "sdk_version": str(claude["sdk_version"]),
        "source": f"side: {active.get('venv')}",
    }
    if claude.get("cli_path"):
        selection["cli_path"] = str(claude["cli_path"])
        selection["cli_sha256"] = str(claude.get("cli_sha256") or "")
    return selection


def session_runtime_config(active: Mapping[str, Any] | None) -> dict[str, Any]:
    """The ``SessionStartRequest.config`` keys that point a session at the side runtime."""

    if active is None:
        return {}
    return {
        "codex_runtime": codex_runtime_selection(active),
        "claude_runtime": claude_runtime_selection(active),
    }


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_claude_runtime(selection: Mapping[str, Any]) -> None:
    """Refuse a side Claude runtime whose Python or CLI moved since it was recorded."""

    python = Path(str(selection.get("python", "")))
    if not python.is_absolute() or not python.is_file() or not os.access(python, os.X_OK):
        raise RuntimeError("the side Claude runtime's Python is missing or cannot run; run vnext-mcp --update-runtimes again")
    cli = selection.get("cli_path")
    if not cli:
        raise RuntimeError("the side Claude runtime records no Claude CLI; run vnext-mcp --update-runtimes again")
    cli_path = Path(str(cli))
    if not cli_path.is_file():
        raise RuntimeError("the side Claude CLI is missing; run vnext-mcp --update-runtimes again")
    if _sha256(cli_path).lower() != str(selection.get("cli_sha256") or "").lower():
        raise RuntimeError("the side Claude CLI changed since it was installed")


def active_codex_models(active: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    if active is None:
        return []
    rows = active["codex"].get("models")
    return [dict(row) for row in rows if isinstance(row, Mapping) and isinstance(row.get("id"), str)] if isinstance(rows, list) else []


def active_codex_slugs(active: Mapping[str, Any] | None) -> frozenset[str] | None:
    """Every model id the side Codex runtime lists, or None when it lists none."""

    slugs = frozenset(row["id"] for row in active_codex_models(active))
    return slugs or None


def _is_untested_entry(entry: Mapping[str, Any]) -> bool:
    return any(
        UNTESTED in str(claim.get("statement", ""))
        for claim in entry.get("claims") or ()
        if isinstance(claim, Mapping)
    )


def untested_codex_models(
    active: Mapping[str, Any] | None, catalog: Sequence[Mapping[str, Any]]
) -> list[str]:
    """Visible models the side Codex runtime lists and the catalog does not."""

    known = {
        str(entry.get("model") or entry.get("model_id"))
        for entry in catalog
        if entry.get("provider") == "codex" and not _is_untested_entry(entry)
    }
    return sorted(
        row["id"] for row in active_codex_models(active)
        if not row.get("hidden") and row["id"] not in known
    )


def untested_catalog_entries(
    active: Mapping[str, Any] | None, catalog: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    return [
        {
            "provider": "codex",
            "model": model,
            "claims": [{
                "kind": "observation",
                "statement": (
                    f"{UNTESTED}: the side Codex runtime lists this model; "
                    "vNext has not run it and has no price for it"
                ),
            }],
        }
        for model in untested_codex_models(active, catalog)
    ]


def update_lines(view: Mapping[str, str], newest: Mapping[str, str] | None) -> list[str]:
    """One line per package that has a newer stable release than the one running."""

    if not newest:
        return []
    lines = []
    running = {CODEX_PACKAGE: view.get("codex_package", ""), SDK_PACKAGE: view.get("sdk", "")}
    for name in (CODEX_PACKAGE, SDK_PACKAGE):
        latest = newest.get(name)
        current = running[name]
        try:
            newer = bool(latest) and _version_key(latest) > _version_key(current)
        except ValueError:
            newer = bool(latest) and latest != current
        if newer:
            lines.append(
                f"{name} {latest} is out (vNext runs {current}). Tell your agent: "
                f"update vNext runtimes (vnext-mcp {UPDATE_FLAG})"
            )
    return lines


def runtime_notice_lines(
    catalog: Sequence[Mapping[str, Any]] = (),
    *,
    active: Mapping[str, Any] | None = None,
    newest: Mapping[str, str] | None = None,
    always: bool = True,
) -> list[str]:
    """The runtime line, the update lines and the untested-model line.

    ``always=False`` returns nothing when the pinned runtime runs and nothing
    newer is known, so a view that never showed a runtime keeps its shape.
    """

    view = runtime_view(active)
    updates = update_lines(view, newest)
    untested = untested_codex_models(active, catalog)
    if not always and active is None and not updates:
        return []
    lines = [f"runtime: Codex {view['codex']}, Claude SDK {view['sdk']} (source: {view['source']})"]
    lines.extend(updates)
    if untested:
        lines.append(f"models {UNTESTED}: {', '.join(untested)} (codex)")
    return lines


# ---------------------------------------------------------------------------
# Installing a side runtime


def _venv_python(venv: Path) -> Path:
    if os.name == "nt":
        return venv / "Scripts" / "python.exe"
    return venv / "bin" / "python"


_LOCATE_SCRIPT = r"""
import json, os
from pathlib import Path
import claude_agent_sdk, codex_cli_bin
name = "claude.exe" if os.name == "nt" else "claude"
claude = Path(claude_agent_sdk.__file__).resolve().parent / "_bundled" / name
print(json.dumps({
    "codex": str(codex_cli_bin.bundled_codex_path()),
    "sdk_version": getattr(claude_agent_sdk, "__version__", None),
    "claude": str(claude) if claude.is_file() else None,
}))
"""

# Starts the Claude CLI through the SDK without a prompt and prints its model
# table: the same connect() and get_server_info() the bridge makes.
_CLAUDE_PROBE_SCRIPT = r"""
import asyncio, json, sys
from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient
async def main():
    options = ClaudeAgentOptions(cwd=sys.argv[1], cli_path=sys.argv[2] or None)
    client = ClaudeSDKClient(options=options)
    try:
        await client.connect(None)
        info = await client.get_server_info()
    finally:
        await client.disconnect()
    rows = info.get("models") if isinstance(info, dict) else None
    print(json.dumps({"models": rows if isinstance(rows, list) else []}))
asyncio.run(main())
"""


def _run(command: Sequence[str], *, timeout: float = 900.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(list(command), capture_output=True, text=True, timeout=timeout, check=False)


def _install(venv: Path, codex_version: str, sdk_version: str, runner: Callable[..., Any]) -> str | None:
    """Make the venv and install the pair into it; a sentence on failure."""

    python = _venv_python(venv)
    if not python.exists():
        made = runner([sys.executable, "-m", "venv", str(venv)])
        if made.returncode != 0:
            return f"could not make a virtual environment at {venv}: {(made.stderr or made.stdout).strip()[-400:]}"
    packages = [f"{CODEX_PACKAGE}=={codex_version}", f"{SDK_PACKAGE}=={sdk_version}"]
    has_pip = runner([str(python), "-m", "pip", "--version"]).returncode == 0
    if has_pip:
        command = [str(python), "-m", "pip", "install", "--disable-pip-version-check", "-q", *packages]
    else:
        uv = shutil.which("uv")
        if uv is None:
            return f"the virtual environment at {venv} has no pip, and uv is not on PATH"
        command = [uv, "pip", "install", "--python", str(python), "-q", *packages]
    installed = runner(command)
    if installed.returncode != 0:
        return f"install of {' '.join(packages)} failed: {(installed.stderr or installed.stdout).strip()[-600:]}"
    return None


def _probe_codex(executable: Path, workspace: Path) -> tuple[list[dict[str, Any]], str | None]:
    """Start one app-server on the side executable and read model/list."""

    from .vnext_app_server import VNextAppServerAdapter

    adapter = None
    try:
        home = Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex")
        adapter = VNextAppServerAdapter(
            codex_executable=executable, codex_home=home, workspace=workspace,
            client_name="vnext_update",
        )
        adapter.initialize(timeout=_PROBE_TIMEOUT_SECONDS)
        rows: list[dict[str, Any]] = []
        cursor: str | None = None
        for _page in range(50):
            params: dict[str, Any] = {"includeHidden": True}
            if cursor:
                params["cursor"] = cursor
            answer = adapter.request("model/list", params, timeout=_PROBE_TIMEOUT_SECONDS)
            for item in answer.get("data") or ():
                if isinstance(item, Mapping) and isinstance(item.get("id"), str):
                    rows.append({
                        "id": item["id"],
                        "model": item.get("model") if isinstance(item.get("model"), str) else item["id"],
                        "hidden": bool(item.get("hidden")),
                    })
            cursor = answer.get("nextCursor")
            if not isinstance(cursor, str) or not cursor:
                break
        return rows, None
    except Exception as exc:  # noqa: BLE001 - reported to the person
        return [], f"the side Codex app-server did not answer model/list: {str(exc) or type(exc).__name__}"
    finally:
        if adapter is not None:
            try:
                adapter.close()
            except Exception:  # noqa: BLE001
                pass


def _probe_claude(python: Path, cli: Path | None, workspace: Path, runner: Callable[..., Any]) -> tuple[list[Any], str | None]:
    result = runner(
        [str(python), "-c", _CLAUDE_PROBE_SCRIPT, str(workspace), str(cli or "")],
        timeout=_PROBE_TIMEOUT_SECONDS,
    )
    if result.returncode != 0:
        return [], f"the side Claude SDK could not start the Claude CLI: {(result.stderr or result.stdout).strip()[-400:]}"
    try:
        rows = json.loads(result.stdout.strip().splitlines()[-1])["models"]
    except (ValueError, KeyError, IndexError) as exc:
        return [], f"the side Claude SDK answered something unreadable: {exc}"
    return list(rows) if isinstance(rows, list) else [], None


def _write_active(record: Mapping[str, Any]) -> Path:
    folder = runtimes_dir()
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / ACTIVE_FILE
    handle, temporary = tempfile.mkstemp(dir=folder, prefix=".active-", suffix=".json")
    with os.fdopen(handle, "w", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2)
    os.replace(temporary, target)
    return target


def update_runtimes(
    codex_version: str | None = None,
    sdk_version: str | None = None,
    *,
    out: Any = None,
    err: Any = None,
    runner: Callable[..., Any] = _run,
    fetch: Callable[[str], Mapping[str, Any]] | None = None,
    codex_probe: Callable[[Path, Path], tuple[list[dict[str, Any]], str | None]] = _probe_codex,
    claude_probe: Callable[..., tuple[list[Any], str | None]] | None = None,
) -> int:
    """Install a Codex CLI and Claude SDK pair beside vNext and make it active."""

    stream = out if out is not None else sys.stdout
    errors = err if err is not None else sys.stderr
    if codex_version is None or sdk_version is None:
        newest = check_for_updates(fetch=fetch, force=True)
        if newest is None:
            print("vnext: PyPI could not be read, so the newest versions are unknown; "
                  "pass --codex-version and --sdk-version", file=errors)
            return 1
        codex_version = codex_version or newest[CODEX_PACKAGE]
        sdk_version = sdk_version or newest[SDK_PACKAGE]
    for label, value in (("--codex-version", codex_version), ("--sdk-version", sdk_version)):
        if not _STABLE.match(value):
            print(f"vnext: {label} must be a stable X.Y.Z release (it was {value!r})", file=errors)
            return 2
    venv = runtimes_dir() / f"{codex_version}-{sdk_version}"
    print(f"installing {CODEX_PACKAGE} {codex_version} and {SDK_PACKAGE} {sdk_version} into {venv}", file=stream)
    problem = _install(venv, codex_version, sdk_version, runner)
    if problem:
        print(f"vnext: {problem}; nothing changed", file=errors)
        return 1
    python = _venv_python(venv)
    located = runner([str(python), "-c", _LOCATE_SCRIPT])
    try:
        paths = json.loads(located.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        print(f"vnext: the side runtime did not name its executables: {located.stderr.strip()[-400:]}; nothing changed", file=errors)
        return 1
    if paths.get("sdk_version") != sdk_version:
        print(f"vnext: the side Claude SDK reports {paths.get('sdk_version')!r}, not {sdk_version}; nothing changed", file=errors)
        return 1
    codex = Path(paths["codex"])
    codex_reply = runner([str(codex), "--version"], timeout=30)
    codex_stdout = (codex_reply.stdout or "").strip()
    if codex_reply.returncode != 0 or not codex_stdout:
        print(f"vnext: {codex} --version failed; nothing changed", file=errors)
        return 1
    if not paths.get("claude"):
        # Without its own CLI the SDK falls back to whichever claude it finds
        # on the machine.  vNext could not hash that one, and every executable
        # it starts is hash-checked, so the pair is refused.
        print(
            f"vnext: {SDK_PACKAGE} {sdk_version} ships no Claude CLI of its own, so its "
            "workers would run whichever claude the machine has, unchecked; nothing changed. "
            "Pick an SDK release that bundles the CLI with --sdk-version",
            file=errors,
        )
        return 1
    cli = Path(paths["claude"])
    cli_reply = runner([str(cli), "--version"], timeout=30)
    cli_version = (cli_reply.stdout or "").strip() or None
    workspace = venv
    codex_models, codex_problem = codex_probe(codex, workspace)
    if codex_problem:
        print(f"vnext: {codex_problem}; nothing changed", file=errors)
        return 1
    from .vnext_mcp_server import DEFAULT_CATALOG

    # The server refuses to start when a catalog model is missing from the
    # side Codex's list, so a pair that drops one is refused here instead.
    listed = {row["id"] for row in codex_models if isinstance(row, Mapping) and isinstance(row.get("id"), str)}
    missing = sorted({
        str(entry.get("model") or entry.get("model_id"))
        for entry in DEFAULT_CATALOG
        if entry.get("provider") == "codex" and not _is_untested_entry(entry)
    } - listed)
    if missing:
        print(
            f"vnext: Codex {codex_stdout} does not list {', '.join(missing)}, which vNext's "
            "catalog offers, so the server could not start on it; nothing changed. "
            "Pick another --codex-version",
            file=errors,
        )
        return 1
    probe = claude_probe or (lambda py, path, ws: _probe_claude(py, path, ws, runner))
    claude_models, claude_problem = probe(python, cli, workspace)
    if claude_problem:
        print(f"vnext: {claude_problem}; nothing changed", file=errors)
        return 1
    record = {
        "venv": str(venv),
        "installed_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "codex": {
            "executable": str(codex),
            "sha256": _sha256(codex),
            "version": codex_stdout,
            "package_version": codex_version,
            "models": codex_models,
        },
        "claude": {
            "python": str(python),
            "sdk_version": sdk_version,
            "cli_path": str(cli) if cli else None,
            "cli_sha256": _sha256(cli) if cli else None,
            "cli_version": cli_version,
            "models": claude_models,
        },
    }
    target = _write_active(record)
    view = runtime_view(record)
    print(f"active runtime: Codex {view['codex']}, Claude SDK {view['sdk']} (source: {view['source']})", file=stream)
    print(f"Codex lists {len(codex_models)} models; Claude lists {len(claude_models)}", file=stream)
    untested = untested_codex_models(record, DEFAULT_CATALOG)
    if untested:
        print(f"models {UNTESTED}: {', '.join(untested)} (codex)", file=stream)
    print(f"recorded in {target}", file=stream)
    print(RESTART_NOTE, file=stream)
    return 0


def rollback(*, out: Any = None) -> int:
    """Go back to the pinned runtime.  The side folders stay for a later switch."""

    stream = out if out is not None else sys.stdout
    target = runtimes_dir() / ACTIVE_FILE
    try:
        target.unlink()
    except FileNotFoundError:
        print("no side runtime is active; vNext already runs its pinned runtime", file=stream)
        return 0
    view = runtime_view(None)
    print(f"active runtime: Codex {view['codex']}, Claude SDK {view['sdk']} (source: pinned)", file=stream)
    print(f"the installed side runtimes stay in {runtimes_dir()}", file=stream)
    print(RESTART_NOTE, file=stream)
    return 0


def main(argv: Sequence[str] | None = None, *, out: Any = None, err: Any = None) -> int:
    parser = argparse.ArgumentParser(
        prog=f"vnext-mcp {UPDATE_FLAG}",
        description="Install a newer Codex CLI and Claude Agent SDK beside vNext, or go back to the pinned pair.",
    )
    parser.add_argument(UPDATE_FLAG, action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--codex-version", default=None, help="openai-codex release to install (default: newest stable)")
    parser.add_argument("--sdk-version", default=None, help="claude-agent-sdk release to install (default: newest stable)")
    parser.add_argument(ROLLBACK_FLAG, action="store_true", help="go back to the pinned runtime; installed folders stay")
    args = parser.parse_args(list(argv or []))
    if args.rollback:
        return rollback(out=out)
    return update_runtimes(args.codex_version, args.sdk_version, out=out, err=err)
