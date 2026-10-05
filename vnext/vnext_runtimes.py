"""Keep the Codex CLI and Claude Agent SDK that vNext runs up to date.

The daily PyPI check automatically installs and verifies newer stable pairs in
fresh timestamped folders under ``~/.vnext/runtimes/`` (or
``VNEXT_RUNTIMES_DIR``). ``auto_update`` writes ``staged.json`` only;
``promote_staged`` switches it in before the next server's startup reads.
Installers never modify an automatically installed pair in place. Nothing
restarts servers. Attempts are recorded before installation; failed builds are
removed, and promotion sweeps unreferenced runtime folders older than 14 days.
Promotion also checks the catalog this server will load and preserves the
rollback record across crashes.

``VNEXT_NO_UPDATE_CHECK=1`` disables checks and automatic installation.
``VNEXT_AUTO_UPDATE=0`` keeps the daily check and manual-update notices only.
``update_runtimes`` remains the explicit installer and writes ``active.json``.
``rollback`` restores the previous pair (or the pinned pair), declines the
rolled-back pair for automatic updates, and keeps all installed folders.
"""

from __future__ import annotations

import argparse
import atexit
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

from .release_check import RELEASE_CODEX_MODEL_COMPATIBILITY, RELEASE_CODEX_VERSION

RUNTIMES_ENV = "VNEXT_RUNTIMES_DIR"
NO_UPDATE_CHECK_ENV = "VNEXT_NO_UPDATE_CHECK"
AUTO_UPDATE_ENV = "VNEXT_AUTO_UPDATE"
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


def missing_codex_catalog_models(
    catalog: Sequence[Mapping[str, Any]], supported: Sequence[str] | frozenset[str],
) -> list[str]:
    """The startup gate checks Codex IDs only; Claude aliases need not have rows."""

    return sorted({
        model for entry in catalog if entry.get("provider") == "codex"
        and isinstance(model := entry.get("model") or entry.get("model_id"), str)
        and model not in supported
    })


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
    records = {}
    problems = []
    for name in ("declined.json", "staged.json", "previous.json", "last_update.json"):
        records[name], problem = _read_record_result(name)
        if problem:
            problems.append(f"runtime metadata: ignored {name} ({problem})")
    notice = None
    if os.environ.get(AUTO_UPDATE_ENV) != "0":
        state = records["last_update.json"] or {}
        staged = records["staged.json"]
        try:
            _identity, pid, live = _lock_details()
        except FileNotFoundError:
            live = False
        except (OSError, ValueError):
            live = False
            problems.append("runtime metadata: update.lock could not be read")
        if live and (staged or updates):
            notice = f"vNext update is waiting: update.lock is held by live process {pid}"
        elif state.get("state") == "failed":
            notice = (f"vNext could not update to {_pair_label(state.get('to', {}))}: "
                      f"{state.get('reason', 'unknown reason')}; still on {_pair_label(_pair(active))}")
        elif staged:
            notice = f"update to {_pair_label(_pair(staged))} is ready and applies at the next start"
        elif state.get("state") == "switched":
            notice = (f"vNext switched itself to {_pair_label(state.get('to', {}))} "
                      f"(was {_pair_label(state.get('from', {}))}); undo: "
                      f"vnext-mcp {UPDATE_FLAG} {ROLLBACK_FLAG}")
        elif state.get("state") == "installing":
            notice = f"vNext attempted an update to {_pair_label(state['to'])} today; another attempt waits until tomorrow"
        elif newest and updates:
            declined = records["declined.json"] or {}
            pair = {key: newest.get(key, "") for key in (CODEX_PACKAGE, SDK_PACKAGE)}
            if pair in declined.get("pairs", []):
                notice = f"update to {_pair_label(pair)} was declined; manual: vnext-mcp {UPDATE_FLAG}"
            elif auto_update_needed(newest):
                notice = (f"vNext will prepare an update to {_pair_label(newest)} automatically; "
                          "applies at the next start")
    untested = untested_codex_models(active, catalog)
    if not always and active is None and not updates and not notice and not problems:
        return []
    lines = [f"runtime: Codex {view['codex']}, Claude SDK {view['sdk']} (source: {view['source']})"]
    lines.extend([notice] if notice else updates)
    lines.extend(problems)
    if untested:
        lines.append(f"models {UNTESTED}: {', '.join(untested)} (codex)")
    return lines


def _pair_label(pair: Mapping[str, str]) -> str:
    return f"Codex {pair.get(CODEX_PACKAGE, '?')} and Claude SDK {pair.get(SDK_PACKAGE, '?')}"


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


def _write_record(name: str, record: Mapping[str, Any]) -> Path:
    folder = runtimes_dir()
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / name
    handle, temporary = tempfile.mkstemp(dir=folder, prefix=f".{name}-", suffix=".json")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(record, stream, indent=2)
        os.replace(temporary, target)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return target


def _write_active(record: Mapping[str, Any]) -> Path:
    return _write_record(ACTIVE_FILE, record)


def _build_runtime(
    venv: Path, codex_version: str, sdk_version: str, *,
    runner: Callable[..., Any],
    codex_probe: Callable[[Path, Path], tuple[list[dict[str, Any]], str | None]],
    claude_probe: Callable[..., tuple[list[Any], str | None]] | None,
) -> tuple[dict[str, Any] | None, str | None]:
    """Install and verify a pair; leave choosing the record's destination to its caller."""

    problem = _install(venv, codex_version, sdk_version, runner)
    if problem:
        return None, " ".join((f"{problem}; nothing changed").splitlines())

    python = _venv_python(venv)
    located = runner([str(python), "-c", _LOCATE_SCRIPT])
    try:
        paths = json.loads(located.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return None, " ".join((f"the side runtime did not name its executables: {located.stderr.strip()[-400:]}; nothing changed").splitlines())

    if paths.get("sdk_version") != sdk_version:
        return None, " ".join((f"the side Claude SDK reports {paths.get('sdk_version')!r}, not {sdk_version}; nothing changed").splitlines())

    codex = Path(paths["codex"])
    codex_reply = runner([str(codex), "--version"], timeout=30)
    codex_stdout = (codex_reply.stdout or "").strip()
    if codex_reply.returncode != 0 or not codex_stdout:
        return None, " ".join((f"{codex} --version failed; nothing changed").splitlines())

    if not paths.get("claude"):
        # Without its own CLI the SDK falls back to whichever claude it finds
        # on the machine.  vNext could not hash that one, and every executable
        # it starts is hash-checked, so the pair is refused.
        return None, " ".join((f"{SDK_PACKAGE} {sdk_version} ships no Claude CLI of its own, so its "
            "workers would run whichever claude the machine has, unchecked; nothing changed. "
            "Pick an SDK release that bundles the CLI with --sdk-version").splitlines())

    cli = Path(paths["claude"])
    cli_reply = runner([str(cli), "--version"], timeout=30)
    cli_version = (cli_reply.stdout or "").strip() or None
    workspace = venv
    codex_models, codex_problem = codex_probe(codex, workspace)
    if codex_problem:
        return None, " ".join((f"{codex_problem}; nothing changed").splitlines())

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
        return None, " ".join((f"Codex {codex_stdout} does not list {', '.join(missing)}, which vNext's "
            "catalog offers, so the server could not start on it; nothing changed. "
            "Pick another --codex-version").splitlines())

    probe = claude_probe or (lambda py, path, ws: _probe_claude(py, path, ws, runner))
    claude_models, claude_problem = probe(python, cli, workspace)
    if claude_problem:
        return None, " ".join((f"{claude_problem}; nothing changed").splitlines())

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
    return record, None


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
    record, problem = _build_runtime(
        venv, codex_version, sdk_version, runner=runner,
        codex_probe=codex_probe, claude_probe=claude_probe,
    )
    if problem:
        print(f"vnext: {problem}", file=errors)
        return 1
    assert record is not None
    from .vnext_mcp_server import DEFAULT_CATALOG

    codex_models = record["codex"]["models"]
    claude_models = record["claude"]["models"]
    _write_record("previous.json", read_active_runtime() or {"pinned": True})
    (runtimes_dir() / "staged.json").unlink(missing_ok=True)
    target = _write_active(record)
    (runtimes_dir() / "last_update.json").unlink(missing_ok=True)
    view = runtime_view(record)
    print(f"active runtime: Codex {view['codex']}, Claude SDK {view['sdk']} (source: {view['source']})", file=stream)
    print(f"Codex lists {len(codex_models)} models; Claude lists {len(claude_models)}", file=stream)
    untested = untested_codex_models(record, DEFAULT_CATALOG)
    if untested:
        print(f"models {UNTESTED}: {', '.join(untested)} (codex)", file=stream)
    print(f"recorded in {target}", file=stream)
    print(RESTART_NOTE, file=stream)
    return 0


def _valid_pair(value: Any) -> bool:
    return isinstance(value, Mapping) and all(
        isinstance(value.get(key), str) and _STABLE.fullmatch(value[key])
        for key in (CODEX_PACKAGE, SDK_PACKAGE)
    )


def _read_record_result(name: str) -> tuple[dict[str, Any] | None, str | None]:
    """Parse auxiliary runtime state at the file boundary, never trust its shape."""

    try:
        value = json.loads((runtimes_dir() / name).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None, None
    except (OSError, ValueError) as exc:
        return None, f"could not read it: {type(exc).__name__}"
    valid = isinstance(value, Mapping)
    if valid and name in ("staged.json", "previous.json"):
        valid = (name == "previous.json" and value.get("pinned") is True) or (
            _is_runtime_record(value.get("codex"), value.get("claude"))
            and isinstance(value.get("venv"), str)
            and isinstance(value["codex"].get("package_version"), str)
        )
    elif valid and name == "declined.json":
        valid = isinstance(value.get("pairs"), list) and all(_valid_pair(pair) for pair in value["pairs"])
    elif valid and name == "last_update.json":
        valid = (
            value.get("state") in {"installing", "staged", "switched", "failed"}
            and isinstance(value.get("day"), str)
            and all(isinstance(value.get(key), Mapping) and (
                not value[key] or _valid_pair(value[key])) for key in ("from", "to"))
            and isinstance(value.get("reason", ""), str)
            and isinstance(value.get("removed", []), list)
        )
    if not valid:
        return None, "invalid record"
    return dict(value), None


def _read_record(name: str) -> dict[str, Any] | None:
    return _read_record_result(name)[0]


def read_staged_runtime() -> dict[str, Any] | None:
    return _read_record("staged.json")


def _pair(active: Mapping[str, Any] | None) -> dict[str, str]:
    view = runtime_view(active)
    return {CODEX_PACKAGE: view["codex_package"], SDK_PACKAGE: view["sdk"]}


def auto_update_needed(newest: Mapping[str, str] | None) -> bool:
    """The cached answer can trigger a fresh attempt, without doing any work."""

    if (os.environ.get(NO_UPDATE_CHECK_ENV) == "1"
            or os.environ.get(AUTO_UPDATE_ENV) == "0" or not newest):
        return False
    if not all(isinstance(newest.get(name), str) and _STABLE.fullmatch(newest[name])
               for name in (CODEX_PACKAGE, SDK_PACKAGE)):
        return False
    pair = {name: newest[name] for name in (CODEX_PACKAGE, SDK_PACKAGE)}
    if not update_lines(runtime_view(read_active_runtime()), pair):
        return False
    declined = _read_record("declined.json") or {}
    if pair in declined.get("pairs", []):
        return False
    staged = _read_record("staged.json")
    if staged and _is_runtime_record(staged.get("codex"), staged.get("claude")) and _pair(staged) == pair:
        return False
    attempt = _read_record("last_update.json") or {}
    return not (attempt.get("to") == pair and attempt.get("day") == _dt.date.today().isoformat())


def _lock_details() -> tuple[os.stat_result, int | None, bool]:
    from .vnext_report import _pid_alive

    path = runtimes_dir() / "update.lock"
    identity = path.stat()
    age = _dt.datetime.now(_dt.timezone.utc).timestamp() - identity.st_mtime
    if age > 6 * 60 * 60:
        return identity, None, False  # Expiry does not depend on reading the PID.
    content = path.read_text(encoding="utf-8").strip()
    if not content and age < 10:
        return identity, None, True  # The claimant may not have written its PID yet.
    try:
        pid = int(content)
    except ValueError:
        pid = None
    live = pid is not None and pid > 0 and _pid_alive(pid)
    return identity, pid, live


def _acquire_update_lock() -> int | None:
    """Claim the PID lock; allow ten seconds for an empty claim to get its PID."""

    folder = runtimes_dir()
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "update.lock"
    for _ in range(3):
        try:
            handle = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            try:
                identity, _pid, live = _lock_details()
                if live:
                    return None
                current = path.stat()
                if (current.st_dev, current.st_ino) == (identity.st_dev, identity.st_ino):
                    path.unlink()
            except FileNotFoundError:
                pass
            continue
        try:
            os.write(handle, str(os.getpid()).encode("ascii"))
        except BaseException:
            os.close(handle)
            path.unlink(missing_ok=True)
            raise
        return handle
    return None


def _release_update_lock(handle: int) -> None:
    path = runtimes_dir() / "update.lock"
    try:
        owned = os.fstat(handle)
    finally:
        # Windows cannot unlink this file while the descriptor is open.
        os.close(handle)
    try:
        current = path.stat()
        if (current.st_dev, current.st_ino) == (owned.st_dev, owned.st_ino):
            path.unlink()
    except FileNotFoundError:
        pass


def _record_update(
    state: str, before: Mapping[str, str], after: Mapping[str, str], reason: str = "", *,
    removed: Sequence[str] = (),
) -> None:
    _write_record("last_update.json", {
        "state": state, "from": dict(before), "to": dict(after),
        "reason": " ".join(reason.splitlines()), "day": _dt.date.today().isoformat(),
        "removed": list(removed),
    })


def _new_runtime_folder(pair: Mapping[str, str]) -> Path:
    stamp = _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0)
    while True:
        folder = runtimes_dir() / f"{pair[CODEX_PACKAGE]}-{pair[SDK_PACKAGE]}-{stamp:%Y%m%d%H%M%S}"
        try:
            folder.mkdir()
            return folder
        except FileExistsError:
            # An interrupted build, even from the same second, is never reused.
            stamp += _dt.timedelta(seconds=1)


def auto_update(
    newest: Mapping[str, str] | None, *,
    runner: Callable[..., Any] | None = None,
    codex_probe: Callable[[Path, Path], tuple[list[dict[str, Any]], str | None]] | None = None,
    claude_probe: Callable[..., tuple[list[Any], str | None]] | None = None,
) -> None:
    """Record the attempt first, then build a staged pair without changing active."""

    handle = None
    folder: Path | None = None
    staged_written = False
    interrupted = False
    before: dict[str, str] = {}
    after: dict[str, str] = {}
    try:
        if not auto_update_needed(newest):
            return
        assert newest is not None
        after = {name: newest[name] for name in (CODEX_PACKAGE, SDK_PACKAGE)}
        before = _pair(read_active_runtime())
        handle = _acquire_update_lock()
        if handle is None or not auto_update_needed(newest):
            return
        _record_update("installing", before, after, "installation started")
        folder = _new_runtime_folder(after)
        record, problem = _build_runtime(
            folder, after[CODEX_PACKAGE], after[SDK_PACKAGE], runner=runner or _run,
            codex_probe=codex_probe or _probe_codex, claude_probe=claude_probe,
        )
        if problem:
            _record_update("failed", before, after, problem)
        else:
            assert record is not None
            _write_record("staged.json", record)
            staged_written = True
            _record_update("staged", before, after)
    except Exception as exc:  # noqa: BLE001 - updates never kill the server
        try:
            _record_update("failed", before, after, str(exc) or type(exc).__name__)
        except Exception:  # noqa: BLE001
            pass
    except BaseException:
        interrupted = True
        raise
    finally:
        # A killed process leaves its partial folder for the age-based sweep.
        # A completed failed build owns this fresh folder and can remove it now.
        if folder is not None and not staged_written and not interrupted:
            try:
                shutil.rmtree(folder)
            except OSError:
                pass
        if handle is not None:
            try:
                _release_update_lock(handle)
            except OSError:
                pass


def _promotion_catalog(catalog: str | Path | Sequence[Mapping[str, Any]] | None) -> Sequence[Mapping[str, Any]]:
    if catalog is None:
        from .vnext_mcp_server import DEFAULT_CATALOG

        return DEFAULT_CATALOG
    if isinstance(catalog, (str, Path)):
        value = json.loads(Path(catalog).read_text(encoding="utf-8"))
        catalog = value.get("models") if isinstance(value, Mapping) else value
    if not isinstance(catalog, (list, tuple)) or not all(isinstance(entry, Mapping) for entry in catalog):
        raise RuntimeError("the startup catalog is not a models list")
    return catalog


def _missing_catalog_models(staged: Mapping[str, Any], catalog: Sequence[Mapping[str, Any]]) -> list[str]:
    supported = active_codex_slugs(staged)
    if supported is None:
        supported = RELEASE_CODEX_MODEL_COMPATIBILITY[RELEASE_CODEX_VERSION]
    return [f"{model} (codex)" for model in missing_codex_catalog_models(catalog, supported)]


def register_runtime_in_use(active: Mapping[str, Any] | None) -> None:
    """Record this server's runtime until process exit, without blocking startup."""

    marker = runtimes_dir() / "in-use" / f"{os.getpid()}.json"
    folder = active.get("venv") if active else sys.prefix
    if not isinstance(folder, str):
        return

    def remove_marker() -> None:
        try:
            marker.unlink(missing_ok=True)
        except Exception:  # noqa: BLE001 - exit cleanup must never raise
            pass

    temporary = None
    try:
        marker.parent.mkdir(parents=True, exist_ok=True)
        handle, temporary = tempfile.mkstemp(dir=marker.parent, prefix=".in-use-", suffix=".tmp")
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump({"venv": str(Path(folder).resolve())}, stream)
        os.replace(temporary, marker)
        atexit.register(remove_marker)
    except Exception:  # noqa: BLE001 - runtime bookkeeping must not prevent startup
        pass
    finally:
        if temporary is not None:
            try:
                Path(temporary).unlink(missing_ok=True)
            except OSError:
                pass


def _cleanup_runtime_folders() -> list[str]:
    """Remove only old unreferenced runtime directories, never the pinned bundle."""

    from .vnext_report import _pid_alive

    root = runtimes_dir().resolve()
    protected = {Path(sys.prefix).resolve(), Path(__file__).resolve().parents[1]}
    for name in (ACTIVE_FILE, "previous.json", "staged.json"):
        record = read_active_runtime() if name == ACTIVE_FILE else _read_record(name)
        if record and isinstance(record.get("venv"), str):
            protected.add(Path(record["venv"]).resolve())
    for marker in (root / "in-use").glob("*.json"):
        try:
            pid = int(marker.stem)
            if pid <= 0:
                continue
            if not _pid_alive(pid):
                marker.unlink(missing_ok=True)
                continue
            record = json.loads(marker.read_text(encoding="utf-8"))
            if isinstance(record, Mapping) and isinstance(record.get("venv"), str):
                protected.add(Path(record["venv"]).resolve())
        except (OSError, ValueError):
            continue
    cutoff = _dt.datetime.now(_dt.timezone.utc).timestamp() - 14 * 24 * 60 * 60
    runtime_name = re.compile(r"^\d+\.\d+\.\d+-\d+\.\d+\.\d+(?:-\d{14})?$")
    removed = []
    try:
        folders = list(root.iterdir())
    except OSError:
        return []
    for folder in folders:
        try:
            if (runtime_name.fullmatch(folder.name) and not folder.is_symlink()
                    and folder.is_dir() and folder.resolve() not in protected
                    and not any(path.is_relative_to(folder.resolve()) for path in protected)
                    and folder.stat().st_mtime < cutoff):
                shutil.rmtree(folder)
                removed.append(folder.name)
        except OSError:
            continue
    return sorted(removed)


def promote_staged(catalog: str | Path | Sequence[Mapping[str, Any]] | None = None) -> None:
    """Validate the startup catalog and atomically preserve rollback before switching."""

    handle = None
    original: dict[str, Any] | None = None
    original_previous: dict[str, Any] | None = None
    before: dict[str, str] = {}
    after: dict[str, str] = {}
    previous_written = False
    switched = False
    invalid_runtime = False
    try:
        staged = read_staged_runtime()
        if staged is None:
            return
        handle = _acquire_update_lock()
        if handle is None:
            return
        # A second startup may have promoted between our read and lock claim.
        staged = read_staged_runtime()
        if staged is None:
            return
        original = read_active_runtime()
        original_previous = _read_record("previous.json")
        prior_selection = original
        if (original is None or original == staged) and original_previous:
            prior_selection = None if original_previous.get("pinned") else original_previous
        before = _pair(prior_selection)
        after = _pair(staged)
        missing = _missing_catalog_models(staged, _promotion_catalog(catalog))
        if missing:
            _record_update("failed", before, after, "startup catalog models missing: " + ", ".join(missing))
            return  # Keep this staged pair for a compatible future startup.
        invalid_runtime = True
        codex = Path(staged["codex"]["executable"])
        if not codex.is_file() or not os.access(codex, os.X_OK):
            raise RuntimeError("the staged Codex executable is missing or cannot run")
        if _sha256(codex).lower() != staged["codex"]["sha256"].lower():
            raise RuntimeError("the staged Codex executable changed since it was installed")
        verify_claude_runtime(claude_runtime_selection(staged) or {})
        invalid_runtime = False
        if original != staged:
            # Copy rather than move: active never disappears between the writes.
            # Recover the older move-based crash layout without losing its backup.
            previous = original or original_previous or {"pinned": True}
            _write_record("previous.json", previous)
            previous_written = True
            os.replace(runtimes_dir() / "staged.json", runtimes_dir() / ACTIVE_FILE)
            switched = True
        else:
            # An already selected duplicate must not overwrite the rollback link.
            (runtimes_dir() / "staged.json").unlink(missing_ok=True)
        _record_update("switched", before, after)
        removed = _cleanup_runtime_folders()
        if removed:
            try:
                _record_update("switched", before, after, removed=removed)
            except OSError:
                # Switching is already committed; a cleanup receipt cannot
                # restore a rollback link to a folder the sweep removed.
                pass
    except Exception as exc:  # noqa: BLE001 - any promotion failure permits startup
        reason = str(exc) or type(exc).__name__
        try:
            if switched:
                if original is None:
                    (runtimes_dir() / ACTIVE_FILE).unlink(missing_ok=True)
                else:
                    _write_active(original)
            if previous_written:
                if original_previous is None:
                    (runtimes_dir() / "previous.json").unlink(missing_ok=True)
                else:
                    _write_record("previous.json", original_previous)
            if invalid_runtime:
                (runtimes_dir() / "staged.json").unlink(missing_ok=True)
        except Exception as recovery:  # noqa: BLE001
            reason += f"; recovery: {str(recovery) or type(recovery).__name__}"
        try:
            _record_update("failed", before, after, reason)
        except Exception:  # noqa: BLE001
            pass
    finally:
        if handle is not None:
            try:
                _release_update_lock(handle)
            except Exception:  # noqa: BLE001
                pass


def rollback(*, out: Any = None) -> int:
    """Restore the previous runtime, declining the pair undone; retain side folders."""

    stream = out if out is not None else sys.stdout
    folder = runtimes_dir()
    target = folder / ACTIVE_FILE
    active = read_active_runtime()
    previous = _read_record("previous.json")
    (folder / "staged.json").unlink(missing_ok=True)
    if previous is not None:
        if previous.get("pinned"):
            target.unlink(missing_ok=True)
        else:
            _write_active(previous)
        (folder / "previous.json").unlink(missing_ok=True)
        if active is not None:
            declined = _read_record("declined.json") or {}
            pairs = declined.get("pairs", [])
            pair = _pair(active)
            if pair not in pairs:
                pairs.append(pair)
            _write_record("declined.json", {"pairs": pairs})
    else:
        try:
            target.unlink()
        except FileNotFoundError:
            print("no side runtime is active; vNext already runs its pinned runtime", file=stream)
            return 0
    # A switched notice no longer describes the selected runtime.
    (folder / "last_update.json").unlink(missing_ok=True)
    view = runtime_view(read_active_runtime())
    print(f"active runtime: Codex {view['codex']}, Claude SDK {view['sdk']} (source: {view['source']})", file=stream)
    print(f"the installed side runtimes stay in {folder}", file=stream)
    print(RESTART_NOTE, file=stream)
    return 0


def main(argv: Sequence[str] | None = None, *, out: Any = None, err: Any = None) -> int:
    parser = argparse.ArgumentParser(
        prog=f"vnext-mcp {UPDATE_FLAG}",
        description="Install a newer Codex CLI and Claude Agent SDK beside vNext, or restore the previous pair.",
    )
    parser.add_argument(UPDATE_FLAG, action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--codex-version", default=None, help="openai-codex release to install (default: newest stable)")
    parser.add_argument("--sdk-version", default=None, help="claude-agent-sdk release to install (default: newest stable)")
    parser.add_argument(ROLLBACK_FLAG, action="store_true", help="restore the previous runtime (or the pinned pair); installed folders stay")
    args = parser.parse_args(list(argv or []))
    if args.rollback:
        return rollback(out=out)
    return update_runtimes(args.codex_version, args.sdk_version, out=out, err=err)
