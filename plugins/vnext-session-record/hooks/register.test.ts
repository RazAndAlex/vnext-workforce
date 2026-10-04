import { expect, test } from 'claude-code/testing'

const SECRET = 'SECRET PROMPT TEXT that must never reach the file'

type Write = { path: string; text: string }

// Answers the session reads and keeps every file write the mod makes.
// No test touches the real file system: fs.write and fs.exists are stubs.
function stubSession(on, writes: Write[], opts: { hasVnext?: boolean; opens?: unknown[]; root?: string; id?: () => string } = {}) {
  const hasVnext = opts.hasVnext ?? true
  on('session.id', () => ({ value: opts.id ? opts.id() : 'sess-1' }))
  on('session.cwd', () => ({ value: '/work/sub' }))
  on('session.root', () => ({ value: opts.root ?? '/work' }))
  on('session.model', () => ({ value: 'claude-opus-5-5[1m]' }))
  on('session.version', () => ({ value: { version: '2.1.287', base: '2.1.287' } }))
  on('session.usage', () => ({
    value: { context: { tokens: 182000, window: 1000000, percent: 18 }, rateLimits: [], cost: { usd: 4.1 } },
  }))
  on('env.get', ($, e) => ({ value: e.name === 'HOME' ? '/home/test' : undefined }))
  on('fs.exists', ($, e) => ({ value: hasVnext && e.path === '/work/.vnext' }))
  on('fs.write', ($, e) => {
    writes.push({ path: e.path, text: e.text })
    return { value: undefined }
  })
  on('ui.open', ($, e) => {
    opts.opens?.push(e)
    return { value: { isPlaced: true } }
  })
}

function rowsOf(text: string) {
  return text.trim().split('\n').map((line) => JSON.parse(line))
}

const SPAWN = {
  tool_use_id: 'tu-1',
  prompt: SECRET,
  description: 'List the files here ' + 'x'.repeat(300),
  subagentType: 'Explore',
  provider: { plugin: 'engine', tier: 'core' },
  model: 'haiku',
  parentModel: 'claude-opus-5-5',
  background: false,
  fork: false,
}

// What Claude Code passes to a ui.render hook for this pane
const PANE = {
  plugin: 'vnext-session-record',
  component: 'Pane',
  requestId: 'vnext-session',
  surface: 'terminal',
  viewport: { columns: 100, rows: 30 },
  props: {
    title: 'vNext',
    isFocused: false,
    bodyColumns: 80,
    placement: 'inline',
    scroll: { offset: 0, bodyRows: 10 },
    view: {},
  },
} as const

// Flattens a drawn tree into its lines of text
function textOf(node): string {
  if (typeof node === 'string') return node
  if (!node || typeof node !== 'object') return ''
  const kids = (node.children ?? []).map(textOf)
  return node.type === 'Box' ? kids.join('\n') : kids.join('')
}

test('agent.spawn is written before next, with its model and no prompt text', async ($, on) => {
  const writes: Write[] = []
  let writesWhenSpawned = -1
  stubSession(on, writes)
  on('agent.spawn', () => {
    writesWhenSpawned = writes.length
    return { model: 'claude-haiku-4-5-20251001', agentId: 'agent-7' }
  })

  const result = await $.agent.spawn(SPAWN)

  expect(result.model).toBe('claude-haiku-4-5-20251001')
  // The spawn row was on disk before the subagent started
  expect(writesWhenSpawned).toBe(1)
  const last = writes.at(-1)!
  expect(last.text).not.toContain('SECRET PROMPT')
  const rows = rowsOf(last.text)
  expect(rows[0]).toMatchObject({ kind: 'agent.spawn', subagentType: 'Explore', model: 'haiku', background: false })
  expect(rows[0].description.length).toBe(120)
  expect(rows[0].prompt).toBeUndefined()
  expect(rows[1]).toMatchObject({ kind: 'agent.spawn_result', model: 'claude-haiku-4-5-20251001', agentId: 'agent-7' })
})

test('a project with .vnext gets the file under <root>/.vnext/host', async ($, on) => {
  const writes: Write[] = []
  stubSession(on, writes, { hasVnext: true })
  on('session.start', () => ({ cwd: '/work/sub' }))
  on('command.register', () => ({ value: undefined }))

  await $.session.start({ surface: null, isInteractive: false, cwd: '/work/sub' })

  expect(writes.at(-1)!.path).toBe('/work/.vnext/host/sess-1.jsonl')
})

test('a project without .vnext gets the file under ~/.vnext/host/<slug>', async ($, on) => {
  const writes: Write[] = []
  stubSession(on, writes, { hasVnext: false, root: '/home/dev/My Project' })
  on('session.start', () => ({ cwd: '/home/dev/My Project' }))
  on('command.register', () => ({ value: undefined }))

  await $.session.start({ surface: null, isInteractive: false, cwd: '/home/dev/My Project' })

  expect(writes.at(-1)!.path).toBe('/home/test/.vnext/host/-home-dev-My-Project/sess-1.jsonl')
})

test('session.end writes a usage row', async ($, on) => {
  const writes: Write[] = []
  stubSession(on, writes)
  on('session.end', () => ({ sessionId: 'sess-1' }))

  await $.session.end({ reason: 'other', sessionId: 'sess-1', resume: { id: 'sess-1' } })

  const rows = rowsOf(writes.at(-1)!.text)
  expect(rows.at(-1)).toMatchObject({
    kind: 'end',
    reason: 'other',
    context: { tokens: 182000 },
    rateLimits: [],
    cost: { usd: 4.1 },
  })
})

test('session.start writes one start row and registers /vnext', async ($, on) => {
  const writes: Write[] = []
  const commands: unknown[] = []
  stubSession(on, writes)
  on('session.start', () => ({ cwd: '/work/sub' }))
  on('command.register', ($, e) => {
    commands.push(e)
    return { value: undefined }
  })

  await $.session.start({ surface: null, isInteractive: false, cwd: '/work/sub' })

  const rows = rowsOf(writes.at(-1)!.text)
  expect(rows.length).toBe(1)
  expect(rows[0]).toMatchObject({ kind: 'start', session: 'sess-1', cwd: '/work/sub', root: '/work', model: 'claude-opus-5-5[1m]', version: '2.1.287' })
  expect(commands[0]).toMatchObject({ name: 'vnext', immediate: true })
})

test('turn.step records model, agent id and usage after the response', async ($, on) => {
  const writes: Write[] = []
  stubSession(on, writes)
  on('turn.step', async function* ($, e) {
    yield { kind: 'text', index: 0, text: 'ok' }
    return {
      turnId: e.turnId,
      index: e.index,
      answer: 'ok',
      toolUses: [],
      stopReason: 'end_turn',
      usage: { model: 'claude-haiku-4-5-20251001', input_tokens: 10, output_tokens: 2, cache_read_input_tokens: 0, cache_creation_input_tokens: 0 },
    }
  })

  const stream = $.turn.step({ turnId: 't', index: 0, model: 'claude-haiku-4-5-20251001', messageCount: 1, agentId: 'agent-7' })
  let step = await stream.next()
  while (step.done !== true) step = await stream.next()

  expect(step.value.answer).toBe('ok')
  const rows = rowsOf(writes.at(-1)!.text)
  expect(rows[0]).toMatchObject({
    kind: 'turn.step',
    model: 'claude-haiku-4-5-20251001',
    agentId: 'agent-7',
    answeredBy: 'claude-haiku-4-5-20251001',
    usage: { input_tokens: 10, output_tokens: 2 },
  })
})

test('a vNext delegate call is recorded and other tools pass straight through', async ($, on) => {
  const writes: Write[] = []
  stubSession(on, writes)
  on('tool.call', ($, e) =>
    e.tool === 'Bash' ? { result: 'listed' } : { result: { structuredContent: { agent_id: 'child-9abcdef0' } } },
  )

  await $.tool.call({ tool: 'Bash', command: 'ls' })
  expect(writes.length).toBe(0)

  await $.tool.call({ tool: 'mcp__plugin_vnext_vnext__delegate', model_id: 'gpt-6-luna', role: 'worker', task: SECRET })
  const text = writes.at(-1)!.text
  expect(text).not.toContain('SECRET PROMPT')
  const rows = rowsOf(text)
  expect(rows[0]).toMatchObject({ kind: 'vnext.delegate', model_id: 'gpt-6-luna', role: 'worker' })
  expect(rows[1]).toMatchObject({ kind: 'vnext.delegate_result', delegatedAgentId: 'child-9abcdef0' })
})

test('the pane shows the empty state, then one subagent line, and opens once without focus', async ($, on) => {
  const writes: Write[] = []
  const opens: any[] = []
  stubSession(on, writes, { opens })
  on('session.start', () => ({ cwd: '/work/sub' }))
  on('command.register', () => ({ value: undefined }))
  on('agent.spawn', () => ({ model: 'claude-haiku-4-5-20251001', agentId: 'agent-7' }))
  on('session.measure', ($, e) => ({ changed: e.changed }))

  await $.session.start({ surface: null, isInteractive: false, cwd: '/work/sub' })
  await $.session.measure({ context: { window: 1000000 }, rateLimits: [], changed: ['context'] })

  const empty = await $.ui.mount(PANE)
  const emptyText = textOf(await empty.find({ type: 'Box' }))
  console.log('--- empty pane\n' + emptyText)
  expect(emptyText).toBe('claude-opus-5-5[1m]  182k of 1M  $4.10\n \nNo subagents yet.')
  await empty.unmount()

  await $.agent.spawn(SPAWN)
  await $.agent.spawn({ ...SPAWN, tool_use_id: 'tu-2' })

  const one = await $.ui.mount(PANE)
  const oneText = textOf(await one.find({ type: 'Box' }))
  console.log('--- pane with subagents\n' + oneText)
  expect(oneText.split('\n')[2]).toBe('Explore  claude-haiku-4-5-20251001  running  List the files here xxxxxxxxxxxxxxxxxxx…')
  await one.unmount()

  // Opened once, by the first spawn, without focus fields
  expect(opens.length).toBe(1)
  expect(opens[0]).toEqual({ id: 'vnext-session', title: 'vNext' })
})

test('the pane colours a failed spawn and lists vNext delegates', async ($, on) => {
  const writes: Write[] = []
  stubSession(on, writes)
  on('session.start', () => ({ cwd: '/work/sub' }))
  on('command.register', () => ({ value: undefined }))
  on('agent.spawn', () => ({ deny: 'refused' }))
  on('tool.call', () => ({ result: JSON.stringify({ agent_id: '03b55785-545b-445b' }) }))

  await $.session.start({ surface: null, isInteractive: false, cwd: '/work/sub' })
  await $.agent.spawn({ ...SPAWN, subagentType: 'general-purpose', model: 'opus', description: 'Check it' })
  await $.tool.call({ tool: 'mcp__vnext__delegate', model_id: 'gpt-6-luna', role: 'worker', effort: 'high', task: SECRET })

  const ui = await $.ui.mount(PANE)
  const text = textOf(await ui.find({ type: 'Box' }))
  console.log('--- pane with a failed spawn and a delegate\n' + text)
  const lines = text.split('\n')
  expect(lines[2]).toBe('general-purpose  opus  failed  Check it')
  expect(lines[3]).toBe(' ')
  expect(lines[4]).toBe('vNext')
  expect(lines[5]).toMatch(/^worker  gpt-6-luna high  \d\d:\d\d  03b55785$/)
  // The word "failed" is its own Text inside the line, in the theme's error colour
  const coloured: { text: string; color: unknown }[] = []
  const walk = (node) => {
    if (!node || typeof node !== 'object') return
    if (node.type === 'Text' && node.props?.color) coloured.push({ text: textOf(node), color: node.props.color })
    for (const child of node.children ?? []) walk(child)
  }
  walk(await ui.find({ type: 'Box' }))
  expect(coloured).toEqual([{ text: 'failed', color: 'error' }])
})

test('after /clear the next session gets its own file and an empty start', async ($, on) => {
  const writes: Write[] = []
  let id = 'sess-1'
  stubSession(on, writes, { id: () => id })
  on('session.end', () => ({ sessionId: 'sess-1' }))
  on('agent.spawn', () => ({ model: 'claude-haiku-4-5-20251001', agentId: 'a1' }))

  await $.agent.spawn({ ...SPAWN, tool_use_id: 'tu-old' })
  await $.session.end({ reason: 'clear', sessionId: 'sess-1', resume: { id: 'sess-1' } })
  id = 'sess-2'
  await $.agent.spawn({ ...SPAWN, tool_use_id: 'tu-new' })

  const last = writes.at(-1)!
  expect(last.path).toBe('/work/.vnext/host/sess-2.jsonl')
  const rows = rowsOf(last.text)
  expect(rows.some((r) => r.tool_use_id === 'tu-old')).toBe(false)
  expect(rows.some((r) => r.tool_use_id === 'tu-new')).toBe(true)
})
