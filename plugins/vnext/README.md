# vnext — a workforce for Claude Code

This plugin points Claude Code at a vNext control tree. Claude Code stays the
root: it keeps its own conversation, its own permissions and its own user, and
gains thirteen manager tools with which it can start Codex and Claude workers in
this workspace, wait for them, steer them and judge what they return.

vNext never runs a turn on the root and never claims it can stop it. That is the
whole of the "external primary" mode this plugin depends on.

## Install

Python 3.11 or newer is required. Install
[uv](https://docs.astral.sh/uv/getting-started/installation/), then from the
repository root run:

```
uv tool install ".[claude]"
claude plugin marketplace add ./plugins
claude plugin install vnext@vnext
```

The `claude` extra installs `claude-agent-sdk==0.2.143`, which Claude and Z.ai
workers require. After a `git pull`, run `uv tool install --force ".[claude]"`
again from the clone to replace the server copy in uv's tool environment.
`claude plugin install` copies the skill into Claude Code's plugin cache, and
the copy stays as it is while the plugin's version number stays the same. After
a pull that changes the plugin, run `claude plugin uninstall vnext@vnext`
and `claude plugin install vnext@vnext` to refresh it.

The plugin needs `uv` on PATH. Claude Code starts `plugins/vnext/launch.py`
through `uv run`, and that small script finds vNext by itself. It tries these
places in order:

1. The `vnext-mcp` command in the `.venv` folder of the clone that holds the
   plugin.
2. A `vnext-mcp` command on PATH.
3. The clone itself, through `uv run --extra claude`. The first start of this
   kind builds the clone's `.venv`, and later starts use step 1.

When it finds none of them, it writes one message to the server's error output
and names the command that fixes it. Read that message with `claude --debug`.

`uv tool install` is still the way to install vNext for use outside a clone. It
puts the `vnext-mcp` launcher in `~/.local/bin` on macOS and in
`%USERPROFILE%\.local\bin` on Windows. Run `uv tool update-shell` once, so your
shell startup file holds that folder, then prove it with `vnext-mcp --help`.

Then restart Claude Code the whole way: quit the application and open it again
(Cmd-Q on macOS, and close every window on Windows). Opening one more window
beside the running process is not enough. A Claude Code that the Dock or the
Start menu started reads PATH once, at startup, out of your login shell, so a
process that started before `uv tool update-shell` ran keeps the PATH it was
born with.

`/mcp` should then list `vnext` with 14 tools: the thirteen manager tools and
`restart_server`, which the Node proxy adds. 13 tools means the restart proxy
did not start: Node or npm may be missing, or its installation may have failed.
Run `vnext-mcp --check` from the clone to see its current state and reason.

A start that fails shows as `CONNECTION_CLOSED`, which says only that the server
stopped. The reason is on the server's error output, where no client shows it.
Read it with `vnext-mcp --check`: it prints the models it would offer, or the
error that stopped the start, and exits non-zero when the start would fail. Two
causes cover almost every failure.

- PATH does not reach `uv`. Prove `uv --version` in a fresh terminal, then
  start Claude Code from that terminal, which hands it the PATH you just proved.
  "Running it by hand instead", below, needs no PATH at all.
- No provider has a login or a key, so there is no worker model to offer. Run
  `codex login` or `claude auth login`, or write a provider key into the provider
  file named below.

Sign in with `codex login` to offer Codex workers and with `claude auth login` to
offer Claude workers. Z.ai and Command Code models require their provider keys
in `~/.vnext/providers.json`; nothing creates that folder. On macOS,
run `mkdir -p ~/.vnext` before writing the file. On Windows, run
`New-Item -ItemType Directory -Force "$HOME\.vnext"` in PowerShell;
the file is `%USERPROFILE%\.vnext\providers.json`. The server reads it
from `Path.home()`, the current user's home folder. The model list omits
providers with no local login or key.
Codex workers share your normal login folder (`CODEX_HOME`, otherwise `~/.codex`);
set `CODEX_HOME` before starting Claude Code, or set `config.codex_home` for an
API-started session, to use another folder.

Both commands register at user scope, so the workforce is available in
every project, not only this one. Each project gets its own session with
itself as the workspace.

The plugin launches the server itself over stdio, so there is no port, no token
and nothing to start by hand. The server runs behind reloaderoo 1.1.5, pinned in
`plugins/vnext/reload/`. On first start the launcher runs `npm ci` once into
`~/.cache/vnext/reload/`; this needs Node and a network connection
once. Without Node it starts the server directly, as before. `VNEXT_RELOAD=0`
turns the proxy off. Behind the proxy a client reads this server's name as
`vnext-mcp-server-dev` (`vnext-mcp-server.exe-dev` on Windows) and its version as
`1.0.0-dev`, which reloaderoo makes up out of the filename of the command it
launches; without Node the client reads `vnext` and `1`. The `vnext-mcp` command runs the server with the Python
interpreter that `uv tool install` prepared. Keep the clone where it is: Claude
Code loads the plugin from the folder you added as a marketplace, reading it
again at every start. A clone that moves or goes away leaves the plugin failing
to load with `cache-miss`, and the vnext server disappears from `/mcp`. Moving
the folder back restores both.

Codex workers use the pinned CLI on Apple-silicon macOS (`darwin-arm64`) and
64-bit Windows (`win32-amd64`). Claude and Z.ai workers use the Claude Agent SDK
on hosts where Python 3.11 or newer and that SDK run. Command Code workers
currently use the same pinned Codex CLI as Codex workers, so their default
route has the same two-platform limit.

## Automatic wake

After a successful delegation, the plugin runs the returned `wake_command` in
an asynchronous Claude Code hook. When the worker stops, the hook wakes the
manager with the worker’s own report, verification flag, evidence count and
available token and cost totals. The manager can act on the notice and use
`inspect` for the full record. The plugin handles this wait automatically.

If the worker is still running after 30 minutes, the hook wakes the manager to
check it with `inspect` and supplies the command to keep waiting in the
background. Wait errors also wake the manager with their diagnostic. The hook
needs `uv` on PATH, just like the plugin's server launcher.

## What it writes

- The workforce writes in `${CLAUDE_PROJECT_DIR}`, under the workers' own
  permission systems.
- Everything the service writes lives under `${CLAUDE_PROJECT_DIR}/.vnext/`,
  one file per session rather than one per project, because several Claude Code
  windows open the same project at once and each runs its own server. They were
  all appending to one log, and overlapping appends left truncated records with
  nothing to say which run they came from. When vNext creates that folder it
  writes `.vnext/.gitignore` holding `*`, so the records stay out of Git. A
  `.vnext/.gitignore` you already have is left exactly as you wrote it, and so
  is a `.vnext` vNext has no permission to write in; in either case add the line
  `*` to that file yourself to keep the records untracked.
- `runs/<session>.jsonl` is the run record: every event, each tagged with its
  `session_id`. Worth reading after a real session.
- `status/<session>.json` holds that session's roster -- who is working right
  now, on what model, and how many have finished. It is rewritten in full
  whenever an agent changes, so a status line or a watcher reads it in one go
  instead of tailing the events. `live` goes false when the server shuts down
  cleanly. A shutdown that runs out of time also writes `live: false`, adds
  `stopping: true`, and keeps counting in `working` the agents that had not
  stopped yet. A server that is killed leaves its last roster behind, so check
  the `pid` before believing it.
- `outcomes/<session>.jsonl` holds one row per finished agent: role, model,
  effort, duration, token counts, an API-equivalent `cost_usd` where the model
  carries a price, and the agent's own `claimed_verified` / `claimed_outcome`.
  That is the file a rating pass reads. The claim fields are named as claims: a
  worker saying it verified its work is not a verdict on that work. A Claude
  worker's subagent has its tokens and cost counted in its parent's row; its own row says
  `in_parent`.

### When something goes wrong

Run this from the repository folder:

```
uv run python -m vnext.vnext_report "<path to your project>"
```

Give more than one path to read several projects in one run.

One line per session -- how many agents, how many records, how long ago -- and
under any that failed, the error, the cause it was raised from, and the last
lines the provider wrote to stderr before it died. `--full` adds the whole
traceback; `--failures` prints only the workspaces that went wrong and exits
non-zero, so a watcher can ask the question in a script. With no path it reads
the current directory, so give the path when you run it from the repository.

A worker's turn has 30 minutes. The worker reads that limit at the top of each
turn. A Claude or Z.ai turn that runs past it is stopped. A Codex or Command Code
turn runs on while it keeps working, and is stopped once 30 minutes pass with
nothing from it; `interrupt_agent` stops it sooner. Either way the worker's
blocker says the turn reached its limit. What it wrote stays in the workspace,
and `retry` continues the same session, so a long job can go on across turns.

When you quit the server, a provider can still send an error for a worker that
had already finished. The report shows that error as a note under the session.
The session still counts as a success, so `--failures` leaves it out.

A Claude worker runs a real Claude Code, which starts a vNext server of its own
in the same project. That server keeps its own session, and the report marks it
`started inside session`, naming the session it started under. Any other program
you start from a session carries the same mark. The mark tells you where a
session began. It does not say who asked for it.

A `provider.error` record in `runs/<session>.jsonl` holds the provider's own words
about a failure. For a Claude or Z.ai worker it also holds the traceback and the
bridge's own stderr. That text stays on this machine, beside the workspace it
came from. The failure receipt in `vnext_diagnostics` is a different record: it
can leave the machine, so it holds labels only.

### Showing it in the status line

A status line can read every `status/*.json` for the project. Skip any file
whose `pid` is gone. A status line needs three fields: `live`, `working` and
`updated`.

## Running it by hand instead

The same service listens over HTTP when you want one long-lived workforce that
outlives a Claude Code restart. Run this from the repository folder:

```
uv run --extra claude python -m vnext.vnext_mcp_server --workspace "<path to your project>" --port 8765
```

It prints the endpoint, the token, and the `claude mcp add` line that registers
them.

Useful arguments: `--catalog catalog.json` to choose which worker models the
root may delegate to, `--event-log PATH` to move the run record, `--status-file PATH` (or `none`) to move or silence the roster (`--outcome-log` does the same for the rating rows), `--token` (at least 16 characters) to
fix the credential across restarts.

## Models

`vnext-mcp --check` prints the models this server offers on your machine.
The root cannot delegate to a model outside the
catalog its server was started with. Pass `--catalog <path to a catalog JSON
file>` holding `{"models": [{"provider": "codex", "model": "..."}]}` to change
that.

A model is offered only when its provider has a login or a key. The Claude
models need `claude auth login` and the Codex models need `codex login`; the Z.ai
and Command Code models arrive once their key is in
`~/.vnext/providers.json`.
