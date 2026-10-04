# vnext-session-record

A Claude Code mod that records what the host session does with models, and shows it in a pane. It writes one JSON line per event.

Where the file goes:

- If the project root already has a `.vnext/` folder: `<root>/.vnext/host/<session-id>.jsonl`
- In every other folder: `~/.vnext/host/<slug>/<session-id>.jsonl`. The slug is the root path with each `/` and space changed to `-`. So the mod never adds a `.vnext/` folder to an unrelated repository.

The mod only observes. Every hook passes its event on unchanged, and a recording error never stops the session.

## What it records

| Row `kind` | When | Fields |
| :- | :- | :- |
| `start` | The session starts | session id, cwd, project root, main model, Claude Code version, interactive or not |
| `agent.spawn` | Claude Code is about to start a subagent (Agent tool) | subagent type, requested model, parent model, background, fork, description (first 120 characters) |
| `agent.spawn_result` | The subagent started | resolved model, agent id |
| `agent.complete` | A subagent's turn ended | agent id, reason, duration |
| `turn.step` | One model request finished, main loop or subagent | requested model, effort, agent id, answering model, token counts, stop reason |
| `vnext.delegate`, `vnext.delegate_result` | A call to a tool whose name matches `mcp__*vnext*__delegate` | `model_id`, `role`, `effort`, the child agent id the call returned |
| `measure` | Claude Code measures the session | context fill, rate-limit windows, cost |
| `end` | The session ends | reason, context fill, rate-limit windows, cost |

The mod stores no prompt text. It drops the subagent prompt and the delegate task text.

The mod rewrites the whole file after each row, because the mods API has no append. It stops at 3 MiB and adds one `truncated` row.

## The pane

Type `/vnext` to open the pane, and type it again to close it. The pane also opens by itself, without focus, the first time a subagent or a vNext delegate call starts. If you close it, it stays closed for the rest of the session.

```
claude-opus-5-5[1m]  182k of 1M  $4.10

Explore          claude-haiku-4-5-20251001  done    List files in directory
general-purpose  opus                       failed  Check it

vNext
worker  gpt-6-luna high  23:16  03b55785
```

The first line shows the session model, the context fill and the cost. Next come up to 8 subagents, newest first. The vNext delegate calls follow under their own heading. The word "failed" is the only text in colour.

## Load it

```bash
claude --plugin-dir /path/to/plugins/vnext-session-record
```

The mod also works with `claude -p`. Mods load only in a trusted workspace.

## Check it

```bash
claude plugin validate plugins/vnext-session-record --strict
claude plugin test plugins/vnext-session-record
```

Tested with Claude Code 2.1.287.

This mod is separate from the `vnext` plugin, because the vNext bridge turns that plugin off inside its own Claude workers.
