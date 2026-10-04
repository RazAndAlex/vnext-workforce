// Records the host session's own subagents, per-request model and usage
// into a JSONL file, and shows them in the /vnext pane. Every hook observes
// and passes the event on unchanged. No prompt text is ever stored.
//
// Where the file goes: <root>/.vnext/host/<session-id>.jsonl when the project
// root already has a .vnext folder, else ~/.vnext/host/<slug>/<session-id>.jsonl
// so that no .vnext folder appears in an unrelated repository.

const LIMIT_BYTES = 3 * 1024 * 1024
const DESCRIPTION_CHARS = 120
const PANE_DESCRIPTION_CHARS = 40
const PANE_SUBAGENTS = 8
const DELEGATE_TOOL = /^mcp__.*vnext.*__delegate$/
const PANE = 'vnext-session'

// Rows kept in memory; the whole file is rewritten after each new row,
// because $.fs has no append and one write may hold at most 4 MiB.
let rows = []
let bytes = 0
let truncated = false
let filePath = null
let writing = Promise.resolve()

// What the pane draws, kept in memory so ui.render reads no file
const view = {
  model: '',
  context: null,
  cost: null,
  subagents: [], // { toolUseId, agentId, type, model, status, description }
  delegates: [], // { toolUseId, role, modelId, effort, startedAt, agentId }
}
let paneAutoOpened = false
let paneClosedByUser = false

function reset() {
  rows = []
  bytes = 0
  truncated = false
  filePath = null
  view.subagents = []
  view.delegates = []
}

function utf8Length(text) {
  let n = 0
  for (let i = 0; i < text.length; i++) {
    const c = text.charCodeAt(i)
    if (c < 0x80) n += 1
    else if (c < 0x800) n += 2
    else if (c >= 0xd800 && c <= 0xdbff) { n += 4; i++ }
    else n += 3
  }
  return n
}

export function slugOf(root) {
  return root.replace(/[\/ ]/g, '-')
}

async function resolvePath($) {
  if (filePath) return filePath
  const root = await $.session.root()
  const id = await $.session.id()
  if (await $.fs.exists(root + '/.vnext')) {
    filePath = root + '/.vnext/host/' + id + '.jsonl'
  } else {
    const home = await $.env.get('HOME')
    filePath = home + '/.vnext/host/' + slugOf(root) + '/' + id + '.jsonl'
  }
  return filePath
}

// Adds one row and rewrites the file. Never throws.
async function record($, row) {
  try {
    if (truncated) return
    const line = JSON.stringify({ ts: new Date().toISOString(), ...row }) + '\n'
    const size = utf8Length(line)
    if (bytes + size > LIMIT_BYTES) {
      truncated = true
      const last = JSON.stringify({ ts: new Date().toISOString(), kind: 'truncated', limitBytes: LIMIT_BYTES }) + '\n'
      rows.push(last)
      bytes += utf8Length(last)
    } else {
      rows.push(line)
      bytes += size
    }
    const path = await resolvePath($)
    const text = rows.join('')
    // Writes run one after another, each with the whole file
    writing = writing.then(() => $.fs.write(path, text)).catch(() => {})
    await writing
  } catch {
    // A recording error never breaks the session
  }
}

async function recordUsage($, kind, extra) {
  try {
    const usage = await $.session.usage()
    await record($, { kind, ...extra, context: usage.context, rateLimits: usage.rateLimits, cost: usage.cost })
  } catch {
    await record($, { kind, ...extra, usageError: true })
  }
}

// Reads an agent id from a vNext delegate result, whatever shape it came in
function delegateAgentId(result) {
  try {
    const value = result && result.result
    const seen = []
    const visit = (v, depth) => {
      if (v == null || depth > 4) return
      if (typeof v === 'string') {
        try { visit(JSON.parse(v), depth + 1) } catch {}
        return
      }
      if (typeof v !== 'object') return
      if (typeof v.agent_id === 'string') seen.push(v.agent_id)
      for (const k of Object.keys(v)) visit(v[k], depth + 1)
    }
    visit(value, 0)
    return seen[0]
  } catch {
    return undefined
  }
}

// Opens the pane without focus the first time it has something to show,
// and never again in this session once the user has closed it
async function autoOpen($) {
  try {
    $.ui.invalidate('ui.render')
    if (paneAutoOpened || paneClosedByUser) return
    paneAutoOpened = true
    await $.ui.open({ id: PANE, title: 'vNext' })
  } catch {}
}

function tokens(n) {
  if (n >= 1000000) return +(n / 1000000).toFixed(1) + 'M'
  if (n >= 1000) return Math.round(n / 1000) + 'k'
  return String(n)
}

function cut(text, n) {
  return text.length > n ? text.slice(0, n - 1) + '…' : text
}

function two(n) {
  return String(n).padStart(2, '0')
}

// The pane's lines as data: each line is a list of [text, isError] parts
export function paneLines(v) {
  const header = [v.model]
  if (v.context && typeof v.context.tokens === 'number' && typeof v.context.window === 'number') {
    header.push(tokens(v.context.tokens) + ' of ' + tokens(v.context.window))
  }
  if (v.cost && typeof v.cost.usd === 'number') header.push('$' + v.cost.usd.toFixed(2))
  const lines = [[[header.filter(Boolean).join('  '), false]], [[' ', false]]]

  const subs = v.subagents.slice(-PANE_SUBAGENTS).reverse()
  if (subs.length > 0) {
    const wType = Math.max(...subs.map((s) => s.type.length))
    const wModel = Math.max(...subs.map((s) => s.model.length))
    const wStatus = Math.max(...subs.map((s) => s.status.length))
    for (const s of subs) {
      lines.push([
        [s.type.padEnd(wType) + '  ' + s.model.padEnd(wModel) + '  ', false],
        [s.status, s.status === 'failed'],
        [' '.repeat(wStatus - s.status.length) + '  ' + cut(s.description, PANE_DESCRIPTION_CHARS), false],
      ])
    }
  }

  if (v.delegates.length > 0) {
    if (subs.length > 0) lines.push([[' ', false]])
    lines.push([['vNext', false]])
    const cells = v.delegates.map((d) => {
      const at = new Date(d.startedAt)
      return [
        d.role || '',
        (d.modelId || '') + (d.effort ? ' ' + d.effort : ''),
        two(at.getHours()) + ':' + two(at.getMinutes()),
        (d.agentId || '').slice(0, 8),
      ]
    })
    const widths = [0, 1, 2].map((i) => Math.max(...cells.map((c) => c[i].length)))
    for (const c of cells) {
      lines.push([[c.map((cell, i) => (i < 3 ? cell.padEnd(widths[i]) : cell)).join('  ').trimEnd(), false]])
    }
  }

  if (subs.length === 0 && v.delegates.length === 0) lines.push([['No subagents yet.', false]])
  return lines
}

export function register(on) {
  on('session.start', async ($, e, next) => {
    try {
      await $.command.register({ name: 'vnext', description: 'Show or hide the vNext pane', immediate: true })
    } catch {}
    let version
    try { version = (await $.session.version()).version } catch {}
    try {
      const model = await $.session.model()
      view.model = model
      await record($, {
        kind: 'start',
        session: await $.session.id(),
        cwd: await $.session.cwd(),
        root: await $.session.root(),
        model,
        version,
        isInteractive: e.isInteractive,
      })
    } catch {}
    return next(e)
  })

  on('command.run', { command: 'vnext' }, async ($) => {
    try {
      const isOpen = (await $.ui.panes()).some((pane) => pane.id === PANE)
      if (isOpen) {
        paneClosedByUser = true
        await $.ui.close({ id: PANE })
      } else {
        await $.ui.open({ id: PANE, title: 'vNext', focus: true, closeOnEscape: true })
      }
    } catch {}
    return {}
  })

  on('ui.close', { id: PANE }, async ($, e, next) => {
    if (e.origin && e.origin.kind === 'person') paneClosedByUser = true
    return next(e)
  })

  on('ui.render', { component: 'Pane' }, async ($, e, next) => {
    if (e.requestId !== PANE) return next(e)
    const { Box, Text } = $.ui.resolve(e)
    return Box({
      flexDirection: 'column',
      children: paneLines(view).map((parts) =>
        parts.length === 1
          ? Text({ wrap: 'truncate-end', children: [parts[0][0]] })
          : Text({
              wrap: 'truncate-end',
              children: parts.map(([text, isError]) => (isError ? Text({ color: 'error', children: [text] }) : text)),
            }),
      ),
    })
  })

  on('agent.spawn', async ($, e, next) => {
    // Written before the subagent starts; the prompt is left out on purpose
    await record($, {
      kind: 'agent.spawn',
      tool_use_id: e.tool_use_id,
      subagentType: e.subagentType,
      model: e.model ?? null,
      parentModel: e.parentModel,
      parentAgentId: e.parentAgentId,
      background: e.background,
      fork: e.fork,
      name: e.name,
      description: typeof e.description === 'string' ? e.description.slice(0, DESCRIPTION_CHARS) : undefined,
      provider: e.provider,
    })
    const entry = {
      toolUseId: e.tool_use_id,
      agentId: undefined,
      type: e.subagentType || '',
      model: e.model || e.parentModel || '',
      status: 'running',
      description: typeof e.description === 'string' ? e.description : '',
    }
    view.subagents.push(entry)
    let result
    try {
      result = await next(e)
    } catch (error) {
      entry.status = 'failed'
      await autoOpen($)
      throw error
    }
    try {
      if (result && result.deny) entry.status = 'failed'
      if (result && result.model) entry.model = result.model
      if (result && result.agentId) entry.agentId = result.agentId
      await record($, {
        kind: 'agent.spawn_result',
        tool_use_id: e.tool_use_id,
        model: result && result.model,
        agentId: result && result.agentId,
        denied: Boolean(result && result.deny),
      })
    } catch {}
    await autoOpen($)
    return result
  })

  on('turn.complete', async ($, e, next) => {
    try {
      if (e.agentId) {
        const entry = view.subagents.find((s) => s.agentId === e.agentId)
        if (entry) {
          entry.status = e.reason === 'answer' ? 'done' : 'failed'
          await record($, { kind: 'agent.complete', agentId: e.agentId, reason: e.reason, durationMs: e.durationMs })
          $.ui.invalidate('ui.render')
        }
      }
    } catch {}
    return next(e)
  })

  on('turn.step', async function* ($, e, next) {
    const result = yield* next(e)
    try {
      const usage = result && result.usage
      await record($, {
        kind: 'turn.step',
        turnId: e.turnId,
        index: e.index,
        model: e.model,
        effort: e.effort,
        agentId: e.agentId,
        answeredBy: usage ? usage.model : undefined,
        usage: usage
          ? {
              input_tokens: usage.input_tokens,
              output_tokens: usage.output_tokens,
              cache_read_input_tokens: usage.cache_read_input_tokens,
              cache_creation_input_tokens: usage.cache_creation_input_tokens,
            }
          : null,
        stopReason: result ? result.stopReason : null,
      })
    } catch {}
    return result
  })

  on('tool.call', { tool: DELEGATE_TOOL }, async ($, e, next) => {
    await record($, {
      kind: 'vnext.delegate',
      tool: e.tool,
      tool_use_id: e.tool_use_id,
      agentId: e.agentId,
      model_id: e.model_id,
      role: e.role,
      effort: e.effort,
    })
    const entry = {
      toolUseId: e.tool_use_id,
      role: typeof e.role === 'string' ? e.role : '',
      modelId: typeof e.model_id === 'string' ? e.model_id : '',
      effort: typeof e.effort === 'string' ? e.effort : '',
      startedAt: Date.now(),
      agentId: undefined,
    }
    view.delegates.push(entry)
    const result = await next(e)
    try {
      entry.agentId = delegateAgentId(result)
      await record($, {
        kind: 'vnext.delegate_result',
        tool_use_id: e.tool_use_id,
        delegatedAgentId: entry.agentId,
        denied: Boolean(result && result.deny),
        isError: Boolean(result && result.isError),
      })
    } catch {}
    await autoOpen($)
    return result
  })

  on('session.measure', async ($, e, next) => {
    try {
      const usage = await $.session.usage()
      view.context = usage.context
      view.cost = usage.cost
      $.ui.invalidate('ui.render')
      await record($, { kind: 'measure', changed: e.changed, context: usage.context, rateLimits: usage.rateLimits, cost: usage.cost })
    } catch {}
    return next(e)
  })

  on('session.end', async ($, e, next) => {
    await recordUsage($, 'end', { reason: e.reason, sessionId: e.sessionId })
    // /clear and /resume go on in the same process with another session id:
    // start a new file and an empty pane for it
    if (e.reason === 'clear' || e.reason === 'resume') reset()
    return next(e)
  })
}
