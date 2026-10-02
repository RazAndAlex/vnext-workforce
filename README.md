# vNext

vNext lets Claude Code hand work to other AI agents and keeps a record of what
each one did.

You keep working in Claude Code as usual. vNext adds 14 tools when the restart
proxy starts, or 13 when it does not. With them, Claude Code can start a worker on another model, wait for the result,
send the worker a message, stop it, and read what it returned. A worker can
run on Claude, on OpenAI Codex, on Z.ai or on Command Code.

Claude Code stays in charge. It keeps its own conversation, its own permissions
and its own login. vNext never runs a turn on Claude Code itself.

## What you need

- Apple-silicon macOS or 64-bit Windows. These are the two platforms we test.
  We run the automatic tests on GitHub's Windows machines. Nobody has used
  vNext on a real Windows PC yet.
- Python 3.11 or later.
- [uv](https://docs.astral.sh/uv/getting-started/installation/), to install
  the `vnext-mcp` command. The two large downloads total about 200 MiB:
  the Codex CLI (about 114 MiB) and Claude Agent SDK (about 88 MiB).
  The install uses about 660 MB on disk, roughly half for each package.
- Claude Code.
- Node.js is optional. With Node, the server runs behind a small proxy, and
  Claude Code can restart the server without a restart of its own. The proxy
  installs from npm the first time, so that first start needs a network
  connection, and it keeps about 25 MB under `~/.cache/vnext/reload`.
  Without Node, the server starts directly. Behind the proxy a client reads the server's name as `vnext-mcp-server-dev` (on Windows,
  `vnext-mcp-server.exe-dev`) and its version as `1.0.0-dev`, both of which the
  proxy makes up out of the command it launches; without Node the same client
  reads `vnext` and `1`.
- A login or a key for each provider you want to use. See
  [Logins and keys](#logins-and-keys).

## Install

1. Install the command:

   ```
   uv tool install "vnext-workforce[claude] @ git+https://github.com/RazAndAlex/vnext-workforce"
   ```

   The `claude` extra adds the Claude Agent SDK. Claude and Z.ai workers need
   it. Codex and Command Code workers run without it. When a new version
   is out, run `uv tool install --force "vnext-workforce[claude] @
   git+https://github.com/RazAndAlex/vnext-workforce"` again to replace the
   installed server copy. Then run `claude plugin uninstall vnext@vnext` and
   `claude plugin install vnext@vnext` to refresh the cached skill. Neither
   `claude plugin update` nor `claude plugin marketplace update` refreshes it
   while the plugin's version number stays the same.
2. Put the uv tool folder on your PATH, once:

   ```
   uv tool update-shell
   ```

   This edits your shell startup file to add uv's bin folder to PATH; on a
   Mac using zsh, that file is `~/.zshenv`, and uv prints the name of the
   file it changed.
3. Open a new terminal and make sure the command runs:

   ```
   vnext-mcp --help
   ```

   If this prints a usage message, the install is correct. If the terminal
   cannot find `vnext-mcp`, the PATH change has not reached it yet.
4. Add the plugin to Claude Code:

   ```
   claude plugin marketplace add RazAndAlex/vnext-workforce
   claude plugin install vnext@vnext
   ```

5. Quit Claude Code fully and open it again. On macOS, use Cmd-Q. On Windows,
   close every window. Claude Code reads PATH only when it starts, so a new
   window of a running Claude Code does not see the new command.

## Check that it works

In Claude Code, type `/mcp`. The list shows `vnext` with 14 tools when the
restart proxy starts. 13 tools means the restart proxy did not start. This can
happen when Node or npm is missing, when the proxy installation fails, or when
`VNEXT_RELOAD=0` is set, which turns the proxy off on purpose.
Run `vnext-mcp --check` in a terminal to see its current proxy state and reason.
The fourteenth tool, `restart_server`, belongs to the proxy.

The plugin also installs a skill, `vnext-workforce`, which tells Claude how to
use these tools. Claude loads it when you ask it to delegate work, and you can
call it by name with `/vnext:vnext-workforce`.

If `/mcp` says the server failed to start, the message you see is usually
`CONNECTION_CLOSED`. That only says the server stopped. The reason goes to the
server's error output, which Claude Code does not show you. Read it in a
terminal:

```
vnext-mcp --check
```

This prints the models vNext would offer, or the error that stopped the start,
and it exits non-zero when the start would fail. Give the check the same
`--catalog <path to a catalog JSON file>` flag your server starts with, if it
starts with one: it answers for the command you pass it, so a flag you leave out
is a refusal it cannot see. Two causes cover almost every failure:

- **The command is not on PATH.** The terminal answers that `vnext-mcp` is not
  found. The PATH change in step 2 has not reached Claude Code yet. Run
  `uv tool update-shell`, then quit Claude Code the whole way and open it again.
- **No provider has a login or a key.** The check says a workforce needs at
  least one worker model. Run `claude auth login` or `codex login`, or write a
  provider key into `~/.vnext/providers.json`. See
  [Logins and keys](#logins-and-keys).

Then ask Claude Code for a first job, in your own words:

```
Use vNext to give a worker this job: read README.md and list every
command it tells a user to run. Wait for it, then show me what came back.
```

Claude Code turns that into a `delegate` call and an `await_children` call and
fills in the fields the tools ask for.

## Logins and keys

| Provider | Models | What it needs |
| --- | --- | --- |
| `codex` | `gpt-6.1-sol`, `gpt-6-sol`, `gpt-6-luna`, `gpt-6-astra`, `gpt-5.6-sol`, `gpt-5.6-terra`, `gpt-5.6-luna` | `codex login` |
| `claude` | `sonnet`, `opus`, `fable` | `claude auth login` |
| `zai` | `glm-5.3-flash`, `glm-5.3`, `glm-5.2` | a Z.ai API key |
| `commandcode` | `deepseek/deepseek-v4.1-flash` | a Command Code API key |

Codex workers use your normal Codex login folder. That is `~/.codex`, or the
folder that `CODEX_HOME` names when Claude Code starts.

The Z.ai and Command Code keys go in `~/.vnext/providers.json`. Nothing creates
that folder for you. On macOS, make it with:

```
mkdir -p ~/.vnext
```

On Windows, run this in PowerShell:

```
New-Item -ItemType Directory -Force "$HOME\.vnext"
```

On Windows the file is `%USERPROFILE%\.vnext\providers.json`. The server reads it
under `Path.home()`, which is the current user's home folder.

Then write the file:

```json
{
  "providers": {
    "zai": {"api_key": "your Z.ai key"},
    "commandcode": {"api_key": "your Command Code key"}
  }
}
```

That file holds your keys. On macOS, keep it readable by you alone:

```
chmod 600 ~/.vnext/providers.json
```

vNext offers only the models whose provider has a login or a key. A provider
with no login or key does not appear in the list.

## Platforms

| Provider | Runs through | Platforms |
| --- | --- | --- |
| `codex` | the Codex CLI | Apple-silicon macOS, 64-bit Windows |
| `commandcode` | the Codex CLI | Apple-silicon macOS, 64-bit Windows |
| `claude` | the Claude Agent SDK | Apple-silicon macOS, 64-bit Windows |
| `zai` | the Claude Agent SDK | Apple-silicon macOS, 64-bit Windows |

vNext pins one Codex CLI version and checks its checksum for `darwin-arm64` and
`win32-amd64` only. Codex and Command Code workers therefore do not start on
other platforms. Claude and Z.ai workers may run on Linux or on Intel Macs,
but we do not test them there.

## Where the records go

vNext writes its records into the project you work in, under `.vnext/`. It
also writes `.vnext/.gitignore`, so Git does not track the records.

- `runs/<session>.jsonl` has every event of a session.
- `status/<session>.json` shows who is working now.
- `outcomes/<session>.jsonl` has one row for each finished worker: the outcome
  text it reported, the model you asked for, the exact model that answered, its
  effort, its duration and its token counts. A Claude worker's subagent has its
  tokens and cost counted in its parent's row, and its own row says `in_parent`.

## When a worker fails

```
vnext-report "<path to your project>"
```

Step 2 installs this command beside `vnext-mcp`, so it runs in the environment
that is already there. From a clone,
`uv run python -m vnext.vnext_report "<path to your project>"` does the same
work.

This prints each session in that project. Under each failed session it shows
the error and the last lines the provider wrote before it stopped. Add
`--full` for the full traceback. Add `--failures` to print only the workspaces
with a failure. Either form exits non-zero when it finds a failure or a record
it could not read, so a script can check it. A path that is not a folder is an
error of its own: the command says so on its error output and exits 2.

With no path it reads the folder you are in, and its first line names that
folder so a report from the wrong directory reads as one.

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

## Running the tests

```
uv run --group dev pytest -q
```

The first run builds a separate environment in a `.venv` folder inside the
clone. `du` reports about 320 MB for it, and about 660 MB once you have also
run a `uv run --extra claude` command from the plugin README, because that adds
the Claude SDK to the same folder. On a Mac, uv shares these files with its
cache, so the free space you lose is far smaller than `du` shows. Delete the
folder when you no longer need it.

## Keeping up with new models

vNext installs one Codex CLI and one Claude Agent SDK, and checks their
checksums. A new model usually comes with a new release of one of them. Once a
day the server asks PyPI for the newest stable release of each. When one is
newer, `vnext-mcp --check` and the `inspect` tool show a line such as:

```
openai-codex 0.161.0 is out (vNext runs 0.160.0). Tell your agent: update vNext runtimes (vnext-mcp --update-runtimes)
```

To install the newest pair beside vNext, run:

```
vnext-mcp --update-runtimes
```

This makes a separate Python environment under `~/.vnext/runtimes/`, about
560 MB, and installs both packages there. Your vNext install does not change.
The command records the checksum of each program and asks each one for its
model list, with no prompt sent. A Codex model that the new CLI lists and
vNext has not run yet appears in the worker list as "not tested by vNext".

When the new pair cannot run vNext, for example when the new Codex CLI drops
a model that vNext offers, the command says why and changes nothing.

The change takes effect when the server restarts. Restart when no workers are
running, because a restart stops them. `--codex-version X` and
`--sdk-version Y` pick a release other than the newest.

To go back to the pair that vNext ships with, run:

```
vnext-mcp --update-runtimes --rollback
```

If a program in the new environment goes missing or changes on disk,
`vnext-mcp --check` names it and exits non-zero. Run the update again or roll
back.

The installed environments stay on disk. Set `VNEXT_NO_UPDATE_CHECK=1` to
stop the daily PyPI check. Set `VNEXT_RUNTIMES_DIR` to keep the environments in
another folder.

## More

- [plugins/vnext/README.md](plugins/vnext/README.md) has the full reference:
  the record files, running the server by hand over HTTP, and choosing which
  models Claude Code may use.
- [SECURITY.md](SECURITY.md) tells you how to report a security problem.
