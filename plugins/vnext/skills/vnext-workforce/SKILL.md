---
name: vnext-workforce
description: Run a vNext workforce from Claude Code — delegate bounded work to Codex and Claude workers through the vnext MCP tools, wait for their evidence, and judge it. Use when a task splits into parallel pieces, when a piece deserves a different model, or when the user asks to delegate.
---

# Running a vNext workforce

The `vnext` MCP tools make you the Root Manager of a tree of worker agents that
run in this workspace. You keep your own conversation and your own permissions;
the workers are separate processes with their own. You decide what they do, wait
for what they produce, and judge it. Nothing judges it for you.

## The loop

1. `delegate` one child per bounded piece of work.
2. `await_children` with the ids you just created. This blocks.
3. `inspect` anything whose report you do not believe.
4. Decide: accept, `steer`, `retry`, `replace`, or `cancel_agent`.

## delegate

```json
{
  "role": "worker",
  "model_id": "gpt-5.6-sol",
  "objective": "One paragraph: what to produce, and where.",
  "task_contract": {"criteria": ["a checkable statement", "another one"]},
  "effort": "high"
}
```

- `role` is `worker` for work, `branch-manager` for a child that will itself
  delegate. A branch manager is worth it when a piece has its own subtree;
  otherwise it is a layer of telephone.
- `model_id` must be in the catalog this server was started with. `inspect` on
  yourself lists what is available.
- `criteria` are what you will check the result against, so write them as things
  that can be false. "Tests pass" is a criterion; "good code" is not.
- `objective` carries the whole packet. A worker sees no part of your
  conversation — name the files, the constraint, and what the last attempt got
  wrong if there was one.

The call returns an `agent_id`. That id, not the model name, is what every other
tool takes.

## await_children

```json
{"agent_ids": ["<id>", "<id>"]}
```

It returns when the children settle. Each child that has stopped carries what
it reported: `outcome` is the text the worker wrote when it finished, `verified`
is whether the worker claims it checked its own work, and `evidence` is the list
it pointed at. A child whose turn ended without a report carries its blocker or
its failure reason in `outcome` instead. A child still working carries none of
those fields. `inspect` returns the same three fields on the agent itself.

Long reports are cut at 4000 characters and the first 10 pieces of evidence are
kept; the whole text stays in `.vnext/outcomes/<session>.jsonl`.

If the wait is long it returns
`settled: false` with them still running — that is not a failure and they are
unaffected; call it again with the same ids. Waiting on a child that is
*blocked* is refused, because a blocked child is waiting on a decision from you.

## Restarting the server

The server runs behind a small proxy that keeps its connection to Claude Code
open, so the user never has to type `/mcp`. Call `restart_server` with no
arguments when a vnext call answers "Not connected" (the server stopped), or
when new vNext code has been merged and you want it running. A fresh server is
up in a few seconds.

A restart ends the vNext session. Every child still running is stopped, and
agent ids from before the restart no longer resolve. Restart when no child is
running, or once you have what you need from them.

## One writer per workspace

Parallel children share this workspace. Two of them writing the same file at the
same time corrupts both pieces of work. Give parallel writers disjoint file
scopes, or sequence them behind one `await_children`.

## Judging

A worker's own report is a claim. `inspect` with `deep: true` returns its
evidence. A run that says it verified something without evidence you can point
at has not verified it, and the right move is `steer` with a narrower ask rather
than accepting it.

When the whole objective is done and every direct child is terminal, call
`complete_session` with `decision`, a `summary`, and `criteria` as a map from
each criterion to true or false.

## What not to do

- Do not delegate work you could do faster yourself. A spawn costs real money
  and a whole context.
- Do not fan out a single build across several workers hoping for a better one.
  Parallelism is for pieces that are genuinely independent.
- Do not treat `await_children` returning `settled: false` as a timeout to
  escalate. Call it again.
