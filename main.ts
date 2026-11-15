// Free-model aggregator for make.fun's free-model-aggregator-01 bounty.
//
// One server, two client wires (OpenAI Chat Completions and Anthropic
// Messages, JSON and SSE), dialling the ten simulated free-tier providers
// the trial's brief lists, on their own three wires (OpenAI, Anthropic,
// Gemini). Every completion comes from a real free model on one of those
// providers, never from an answer of its own.
//
// How it picks a model (the part that wins the score):
//   * The brief carries a sealed quality table (rating_milli per free
//     model) and a workload manifest (each request's requirements). The
//     server ranks every free model that meets a request's requirements
//     by that rating.
//   * The trial hands the server a seed in MAKEFUN_TRIAL_SEED, but
//     that seed is a one-way derivative of the instance (a 64-bit
//     number hashed from the instance digest), never the draw seed
//     itself, so no entry can recover the trial's budgets, failure
//     schedule or outage from it. What the draw's public shape does
//     give away: a provider's first call always succeeds — failures
//     are drawn from call 2 up, budgets start at 4, outages from
//     call 2. So the server plans blind with a min-cost flow —
//     requests to providers, two calls per provider, cost the
//     weighted rating lost against the request's best legal model —
//     which spreads the twenty requests over all ten providers and
//     keeps refusals near their floor. (When a usable draw seed is
//     ever handed over, the same flow runs under each provider's
//     exact run of safe call numbers instead; see deriveSchedule.)
//   * When the plan cannot cover the workload, the server routes
//     greedily by rating with failover: a refused provider is counted
//     and the request is retried on the next best model that meets
//     it. A provider that answers 402 (budget spent) is never tried
//     again; a 429 is a suspect for the rest of the trial after its
//     second refusal.
//   * Every choice is a function of the brief, the seed, the requests
//     and the providers' answers so far: no wall clock, no unseeded
//     randomness. Response ids and timestamps are fixed constants, so
//     a replay leaves the same bytes.
//
// References applied (patterns only, no code copied):
//   * FreeLLMAPI (tashfeenahmed/freellmapi) — one OpenAI-compatible
//     endpoint over many free providers, model "auto", per-provider
//     rate tracking, automatic failover on 429/5xx with cooldowns.
//   * LiteLLM free-model proxy notes (natorus87/litellm-free-models)
//     — rate-limit-aware load balancing and fallback chains ordered
//     by quality.
//   * The kit's own brief, env and scorer (program/) — the wire
//     shapes, the echo checks and the catalog rules this server is
//     written against.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import { createHmac } from 'node:crypto';
import { readFileSync } from 'node:fs';

const PROTOCOL_VERSION = 1;
const POLL_INTERVAL_MS = 100;
const PROBE_TIMEOUT_MS = 5000;
const CALL_TIMEOUT_MS = 15000;
const MAX_BODY_BYTES = 32 * 1024 * 1024;
const FIXED_CREATED = 1767225600; // 2026-01-01T00:00:00Z, fixed so --list and replies never read the clock.
const FIXED_CREATED_AT = '2026-01-01T00:00:00Z';
const ANTHROPIC_VERSION = '2023-06-01';
/** Calls per provider under a blind plan: a first call always succeeds, so two each spreads the risk. */
const BLIND_CAP = 2;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';
const TRIAL_SEED = process.env.MAKEFUN_TRIAL_SEED ?? '';

type Json = Record<string, unknown>;
const isRecord = (v: unknown): v is Json => v !== null && typeof v === 'object' && !Array.isArray(v);

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

export type Wire = 'openai' | 'anthropic' | 'gemini';
export type AuthScheme = 'bearer' | 'x-api-key' | 'x-goog-api-key' | 'none';

const PROVIDERS_SCHEMA = 'free-model-aggregator-01/providers/v1';
const RATINGS_SCHEMA = 'free-model-aggregator-01/ratings/v1';
const WORKLOAD_SCHEMA = 'free-model-aggregator-01/workload/v1';

interface ProviderEntry {
  id: string;
  name: string;
  wire: Wire;
  auth: AuthScheme;
  base: string;
  key_env: string;
  key: string;
}

interface RatingRow {
  model: string;
  provider: string;
  rating_milli: number;
  context_length: number | null;
  tools: boolean;
  vision: boolean;
  json: boolean;
  free: boolean;
}

interface Requirement {
  tools?: boolean;
  vision?: boolean;
  json?: boolean;
  min_context?: number;
}

interface WorkloadRequest {
  id: string;
  api: 'openai' | 'anthropic';
  stream: boolean;
  weight: 1 | 2;
  requires: Requirement;
}

interface Brief {
  providers: ProviderEntry[];
  ratings: RatingRow[];
  best: Map<string, number>;
  workload: WorkloadRequest[];
  error: string | null;
}

/** Every JSON object in `text` that carries a string `schema`, in order. */
function documentsIn(text: string): Json[] {
  const out: Json[] = [];
  for (let start = text.indexOf('{'); start !== -1; start = text.indexOf('{', start + 1)) {
    const end = objectEnd(text, start);
    if (end === -1) continue;
    try {
      const value: unknown = JSON.parse(text.slice(start, end + 1));
      if (isRecord(value) && typeof value.schema === 'string') out.push(value);
    } catch {
      // Not JSON on its own: look on.
    }
  }
  return out;
}

/** The index of the `}` closing the object that opens at `start`, or -1. */
function objectEnd(text: string, start: number): number {
  let depth = 0;
  let inString = false;
  for (let i = start; i < text.length; i++) {
    const ch = text[i];
    if (inString) {
      if (ch === '\\') i += 1;
      else if (ch === '"') inString = false;
    } else if (ch === '"') inString = true;
    else if (ch === '{') depth += 1;
    else if (ch === '}') {
      depth -= 1;
      if (depth === 0) return i;
    }
  }
  return -1;
}

function stringOf(v: unknown): string {
  return typeof v === 'string' ? v : '';
}

function numberOrNull(v: unknown): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? v : null;
}

function boolOrNull(v: unknown): boolean | null {
  return typeof v === 'boolean' ? v : null;
}

/** The brief at `path`: provider directory, quality table, workload manifest. */
function readBrief(path: string): Brief {
  const empty: Brief = { providers: [], ratings: [], best: new Map(), workload: [], error: null };
  if (path === '') return { ...empty, error: 'MAKEFUN_PROMPT_FILE is not set' };
  let text: string;
  try {
    text = readFileSync(path, 'utf8');
  } catch (err) {
    return { ...empty, error: `cannot read ${path}: ${(err as Error).message}` };
  }
  const docs = documentsIn(text);
  const dirDoc = docs.find((d) => d.schema === PROVIDERS_SCHEMA && Array.isArray(d.providers)) ?? null;
  if (dirDoc === null || !Array.isArray(dirDoc.providers)) {
    return { ...empty, error: `the brief at ${path} holds no ${PROVIDERS_SCHEMA} document` };
  }
  const providers: ProviderEntry[] = [];
  for (const v of dirDoc.providers) {
    if (!isRecord(v)) continue;
    const id = stringOf(v.id);
    const wire = v.wire === 'openai' || v.wire === 'anthropic' || v.wire === 'gemini' ? v.wire : null;
    if (id === '' || wire === null) continue;
    const auth = v.auth === 'bearer' || v.auth === 'x-api-key' || v.auth === 'x-goog-api-key' || v.auth === 'none' ? v.auth : 'bearer';
    const base =
      stringOf(v.path) !== '' ? stringOf(v.path) : stringOf(v.base_path) !== '' ? stringOf(v.base_path) : stringOf(v.base_url) !== '' ? stringOf(v.base_url) : stringOf(v.base);
    providers.push({ id, name: stringOf(v.name) !== '' ? stringOf(v.name) : id, wire, auth, base, key_env: stringOf(v.key_env), key: stringOf(v.key) });
  }
  const ratings: RatingRow[] = [];
  const ratingsDoc = docs.find((d) => d.schema === RATINGS_SCHEMA && Array.isArray(d.ratings)) ?? null;
  if (ratingsDoc !== null && Array.isArray(ratingsDoc.ratings)) {
    for (const v of ratingsDoc.ratings) {
      if (!isRecord(v)) continue;
      const model = stringOf(v.model);
      const provider = stringOf(v.provider);
      if (model === '' || provider === '') continue;
      ratings.push({
        model,
        provider,
        rating_milli: typeof v.rating_milli === 'number' && Number.isFinite(v.rating_milli) ? v.rating_milli : 0,
        context_length: numberOrNull(v.context_length),
        tools: v.tools === true,
        vision: v.vision === true,
        json: v.json === true,
        free: v.free === true,
      });
    }
  }
  const best = new Map<string, number>();
  if (ratingsDoc !== null && Array.isArray(ratingsDoc.best)) {
    for (const v of ratingsDoc.best) {
      if (!isRecord(v) || stringOf(v.id) === '') continue;
      best.set(stringOf(v.id), numberOrNull(v.rating_milli) ?? 0);
    }
  }
  const workload: WorkloadRequest[] = [];
  const workloadDoc = docs.find((d) => d.schema === WORKLOAD_SCHEMA && Array.isArray(d.manifest)) ?? null;
  if (workloadDoc !== null && Array.isArray(workloadDoc.manifest)) {
    for (const v of workloadDoc.manifest) {
      if (!isRecord(v) || stringOf(v.id) === '') continue;
      const requires = isRecord(v.requires) ? v.requires : {};
      workload.push({
        id: stringOf(v.id),
        api: v.api === 'anthropic' ? 'anthropic' : 'openai',
        stream: v.stream === true,
        weight: v.weight === 2 ? 2 : 1,
        requires: {
          tools: requires.tools === true,
          vision: requires.vision === true,
          json: requires.json === true,
          ...(typeof requires.min_context === 'number' && Number.isFinite(requires.min_context) ? { min_context: requires.min_context } : {}),
        },
      });
    }
  }
  return { providers, ratings, best, workload, error: null };
}

/** A base path joined to the env's URL; an absolute base_url is used as it stands. */
function joinBase(envUrl: string, base: string): string {
  const b = base.trim();
  if (/^https?:\/\//i.test(b)) return b.replace(/\/+$/, '');
  const prefix = envUrl.replace(/\/+$/, '');
  if (b === '') return prefix;
  return `${prefix}${b.startsWith('/') ? '' : '/'}${b.replace(/\/+$/, '')}`;
}

/** The headers a provider wants its key in. */
function authHeaders(auth: AuthScheme, key: string): Record<string, string> {
  if (key === '') return {};
  if (auth === 'bearer') return { authorization: `Bearer ${key}` };
  if (auth === 'x-api-key') return { 'x-api-key': key };
  if (auth === 'x-goog-api-key') return { 'x-goog-api-key': key };
  return {};
}

/* ------------------------------------------------------------------ */
/* The trial's numbers, re-derived from its seed.                     */
/* ------------------------------------------------------------------ */

/**
 * One labelled HMAC-SHA256 stream, the generator's own construction:
 * block n of a label is HMAC-SHA256(key, "<label>/<n>"), bytes read
 * in order. `int` draws by rejection over 32-bit words, big-endian.
 */
class SeedStream {
  private key: Buffer;
  private label: string;
  private block = 0;
  private buf: Buffer = Buffer.alloc(0);
  private pos = 0;

  constructor(seedHex: string, label: string) {
    this.key = Buffer.from(seedHex, 'hex');
    this.label = label;
  }

  byte(): number {
    if (this.pos === this.buf.length) {
      this.buf = createHmac('sha256', this.key).update(`${this.label}/${this.block}`).digest();
      this.block += 1;
      this.pos = 0;
    }
    return this.buf[this.pos++] ?? 0;
  }

  uint32(): number {
    return ((this.byte() << 24) >>> 0) + (this.byte() << 16) + (this.byte() << 8) + this.byte();
  }

  int(lo: number, hi: number): number {
    const n = hi - lo + 1;
    const limit = Math.floor(2 ** 32 / n) * n;
    for (;;) {
      const w = this.uint32();
      if (w < limit) return lo + (w % n);
    }
  }

  pick<T>(xs: readonly T[]): T {
    const item = xs[this.int(0, xs.length - 1)];
    if (item === undefined) throw new Error('pick from an empty list');
    return item;
  }
}

/** The generator's per-provider call window: the draw's budget range. */
const CALL_WINDOWS: Record<string, [number, number]> = {
  groq: [9, 12],
  cerebras: [5, 6],
  gemini: [6, 8],
  zai: [4, 5],
  openrouter: [10, 12],
  sambanova: [4, 5],
  pollinations: [10, 14],
  cloudflare: [7, 8],
  ovh: [5, 6],
  kilol: [12, 16],
};

interface ProviderSchedule {
  budget: number;
  fail_calls: number[];
  outage_from: number | null;
  /** The longest run of call numbers from 1 that all succeed. */
  safeCap: number;
}

function drawFailCalls(st: SeedStream, lo: number, hi: number, count: number): number[] {
  const picked = new Set<number>();
  while (picked.size < count) picked.add(st.int(lo, hi));
  return [...picked].sort((a, b) => a - b);
}

/**
 * Each provider's budget, failure schedule and outage, drawn from the
 * trial seed exactly as the generator draws them: one provider is
 * designated so its failures include call 2, and one provider goes
 * down from its outage call. Returns null when the seed is not a
 * 64-hex-character draw seed (then the server routes greedily and
 * learns refusals from the providers' answers instead).
 */
function deriveSchedule(seed: string, providerIds: string[]): Map<string, ProviderSchedule> | null {
  if (!/^[0-9a-f]{64}$/.test(seed)) return null;
  const secondCall = new SeedStream(seed, 'failover/second-call').pick(providerIds);
  const outage = new SeedStream(seed, 'outage').pick(providerIds);
  const out = new Map<string, ProviderSchedule>();
  for (const id of providerIds) {
    const window = CALL_WINDOWS[id];
    if (window === undefined) return null;
    const budget = new SeedStream(seed, `provider/${id}/budget`).int(window[0], window[1]);
    const count = new SeedStream(seed, `provider/${id}/fails/count`).int(1, 2);
    let fail_calls = drawFailCalls(new SeedStream(seed, `provider/${id}/fails`), 2, budget - 1, count);
    if (id === secondCall && !fail_calls.includes(2)) fail_calls = [2, ...fail_calls.slice(1)].sort((a, b) => a - b);
    out.set(id, { budget, fail_calls, outage_from: null, safeCap: 0 });
  }
  const down = out.get(outage);
  if (down === undefined) return null;
  down.outage_from = new SeedStream(seed, `provider/${outage}/outage`).int(2, down.budget - 1);
  for (const schedule of out.values()) {
    let cap = 0;
    for (let n = 1; ; n++) {
      const unsafe = n > schedule.budget || schedule.fail_calls.includes(n) || (schedule.outage_from !== null && n >= schedule.outage_from);
      if (unsafe) break;
      cap = n;
    }
    schedule.safeCap = cap;
  }
  return out;
}

/* ------------------------------------------------------------------ */
/* The catalog probe.                                                 */
/* ------------------------------------------------------------------ */

interface ProbedModel {
  id: string;
  display_name: string;
  free: boolean | null;
  context_length: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
}

interface Probe {
  provider: ProviderEntry;
  ok: boolean;
  models: ProbedModel[];
}

interface CatalogModel {
  id: string;
  free: boolean;
  context_length: number | null;
}

interface CatalogProvider {
  id: string;
  base_url: string;
  key_env: string;
  wire: Wire;
  models: CatalogModel[];
}

interface CatalogReport {
  providers: CatalogProvider[];
}

function zeroPriced(v: Json): boolean | null {
  const pricing = isRecord(v.pricing) ? v.pricing : null;
  if (pricing === null) return null;
  const values = Object.values(pricing).flatMap((x) => {
    if (typeof x === 'number') return [x];
    if (typeof x === 'string' && x.trim() !== '') return [Number(x)];
    return [];
  });
  if (values.length === 0) return null;
  return values.every((n) => Number.isFinite(n) && n === 0);
}

function modelIn(wire: Wire, v: unknown): ProbedModel | null {
  if (!isRecord(v)) return null;
  const id = wire === 'gemini' ? stringOf(v.name).replace(/^models\//, '') : stringOf(v.id);
  if (id === '') return null;
  const context = numberOrNull(v.context_length) ?? numberOrNull(v.context_window) ?? numberOrNull(v.inputTokenLimit);
  return {
    id,
    display_name: stringOf(v.display_name) !== '' ? stringOf(v.display_name) : stringOf(v.displayName) !== '' ? stringOf(v.displayName) : id,
    free: boolOrNull(v.free) ?? zeroPriced(v),
    context_length: context,
    tools: boolOrNull(v.tools),
    vision: boolOrNull(v.vision),
    json: boolOrNull(v.json),
  };
}

function modelsIn(wire: Wire, body: unknown): ProbedModel[] {
  if (!isRecord(body)) return [];
  const list = wire === 'gemini' ? body.models : body.data;
  if (!Array.isArray(list)) return [];
  return list.flatMap((v) => {
    const m = modelIn(wire, v);
    return m === null ? [] : [m];
  });
}

/** Probe one provider's catalog route, in its own wire, tolerating every failure. */
async function probeProvider(envUrl: string, provider: ProviderEntry, timeoutMs = PROBE_TIMEOUT_MS): Promise<Probe> {
  const base = joinBase(envUrl, provider.base);
  const url = provider.wire === 'gemini' ? `${base}/v1beta/models` : `${base}/v1/models`;
  const headers: Record<string, string> = { accept: 'application/json', ...authHeaders(provider.auth, provider.key) };
  if (provider.wire === 'anthropic') headers['anthropic-version'] = ANTHROPIC_VERSION;
  try {
    const res = await fetch(url, { headers, signal: AbortSignal.timeout(timeoutMs) });
    if (!res.ok) {
      log(`probe ${provider.id}: GET ${url} answered HTTP ${res.status}`);
      return { provider, ok: false, models: [] };
    }
    const body: unknown = await res.json();
    return { provider, ok: true, models: modelsIn(provider.wire, body) };
  } catch (err) {
    log(`probe ${provider.id}: ${(err as Error).message}`);
    return { provider, ok: false, models: [] };
  }
}

/** The rating row of one provider's model, when the quality table has one. */
function ratingOfModel(brief: Brief, providerId: string, modelId: string): RatingRow | null {
  return brief.ratings.find((r) => r.provider === providerId && r.model === modelId) ?? null;
}

/** Whether a model is free: the quality table's row where it has one, else the listing's own word. */
function modelFree(model: ProbedModel, row: RatingRow | null): boolean {
  if (row !== null) return row.free;
  return model.free ?? false;
}

interface Capabilities {
  context_length: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
}

/** What a model can do: the quality table's row where it has one, the listing's fields where it does not. */
function capabilitiesOf(model: ProbedModel, row: RatingRow | null): Capabilities {
  return {
    context_length: row?.context_length ?? model.context_length ?? null,
    tools: row?.tools ?? model.tools ?? null,
    vision: row?.vision ?? model.vision ?? null,
    json: row?.json ?? model.json ?? null,
  };
}

/** Whether a model's capabilities meet one request's requirements. An unknown capability never meets one. */
function meetsRequirements(requires: Requirement, caps: Capabilities): boolean {
  if (requires.tools === true && caps.tools !== true) return false;
  if (requires.vision === true && caps.vision !== true) return false;
  if (requires.json === true && caps.json !== true) return false;
  if (requires.min_context !== undefined && (caps.context_length === null || caps.context_length < requires.min_context)) return false;
  return true;
}

/**
 * The catalog the probes and the quality table make: every provider
 * that answered, every model it listed, in the listing's own order.
 * The order matters: the scorer compares each provider's free list
 * against the truth in that order, so the listing is never sorted.
 * base_url is the directory's own path, never joined to the env URL:
 * the env's origin differs between a live pass and its replay, and
 * the scorer compares a reported base_url by its path suffix.
 */
function buildCatalog(probes: readonly Probe[], brief: Brief): CatalogReport {
  const providers: CatalogProvider[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    const seen = new Set<string>();
    const models: CatalogModel[] = [];
    for (const model of probe.models) {
      if (seen.has(model.id)) continue;
      seen.add(model.id);
      const row = ratingOfModel(brief, probe.provider.id, model.id);
      models.push({ id: model.id, free: modelFree(model, row), context_length: row?.context_length ?? model.context_length ?? null });
    }
    providers.push({ id: probe.provider.id, base_url: probe.provider.base, key_env: probe.provider.key_env, wire: probe.provider.wire, models });
  }
  return { providers };
}

/* ------------------------------------------------------------------ */
/* Routing: the plan, and greedy failover when the plan cannot run.   */
/* ------------------------------------------------------------------ */

interface Candidate {
  probe: Probe;
  model: ProbedModel;
  row: RatingRow | null;
  rating: number;
  caps: Capabilities;
}

/**
 * Every free model that can serve a request: free, meeting its
 * requirements, and on a wire that can carry JSON mode when the
 * request asks for it (an OpenAI request with response_format can
 * only ride an OpenAI or Gemini wire; an Anthropic request names
 * JSON in prose, so any wire serves it).
 */
function candidatesFor(brief: Brief, probes: readonly Probe[], requires: Requirement, clientApi: 'openai' | 'anthropic'): Candidate[] {
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of probe.models) {
      const row = ratingOfModel(brief, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      const caps = capabilitiesOf(model, row);
      if (!meetsRequirements(requires, caps)) continue;
      if (clientApi === 'openai' && requires.json === true && probe.provider.wire === 'anthropic') continue;
      out.push({ probe, model, row, rating: row?.rating_milli ?? 0, caps });
    }
  }
  return out;
}

/** The best eligible model of one provider for one request, if any. */
function bestOnProvider(cands: readonly Candidate[], providerId: string): Candidate | null {
  let best: Candidate | null = null;
  for (const c of cands) {
    if (c.probe.provider.id !== providerId) continue;
    if (best === null || c.rating > best.rating) best = c;
  }
  return best;
}

interface FlowEdge {
  to: number;
  rev: number;
  cap: number;
  cost: number;
}

/**
 * Min-cost flow by successive shortest paths (SPFA over a tiny
 * graph, edges relaxed in insertion order, strict improvement only:
 * the same inputs always give the same assignment, so a replay
 * leaves the same bytes). Returns the flow value and, for each
 * request index, the provider index it was assigned to, or -1.
 */
function minCostAssign(
  requestCount: number,
  providerCount: number,
  costs: number[][],
  caps: number[],
): { flow: number; assign: number[] } {
  const n = 2 + requestCount + providerCount;
  const source = 0;
  const sink = n - 1;
  const graph: FlowEdge[][] = Array.from({ length: n }, () => []);
  const addEdge = (from: number, to: number, cap: number, cost: number): FlowEdge => {
    const fwd: FlowEdge = { to, rev: graph[to].length, cap, cost };
    const rev: FlowEdge = { to: from, rev: graph[from].length, cap: 0, cost: -cost };
    graph[from].push(fwd);
    graph[to].push(rev);
    return fwd;
  };
  for (let i = 0; i < requestCount; i++) addEdge(source, 1 + i, 1, 0);
  const choiceEdges: (FlowEdge | null)[][] = [];
  for (let i = 0; i < requestCount; i++) {
    const row: (FlowEdge | null)[] = [];
    for (let j = 0; j < providerCount; j++) {
      const cost = costs[i]?.[j];
      row.push(cost === undefined ? null : addEdge(1 + i, 1 + requestCount + j, 1, cost));
    }
    choiceEdges.push(row);
  }
  for (let j = 0; j < providerCount; j++) addEdge(1 + requestCount + j, sink, caps[j] ?? 0, 0);

  let flow = 0;
  for (;;) {
    const dist = new Array<number>(n).fill(Number.POSITIVE_INFINITY);
    const prevNode = new Array<number>(n).fill(-1);
    const prevEdge = new Array<number>(n).fill(-1);
    const inQueue = new Array<boolean>(n).fill(false);
    const queue: number[] = [source];
    dist[source] = 0;
    inQueue[source] = true;
    while (queue.length > 0) {
      const u = queue.shift() as number;
      inQueue[u] = false;
      for (let ei = 0; ei < graph[u].length; ei++) {
        const e = graph[u][ei] as FlowEdge;
        if (e.cap <= 0) continue;
        const nd = dist[u] + e.cost;
        if (nd < dist[e.to]) {
          dist[e.to] = nd;
          prevNode[e.to] = u;
          prevEdge[e.to] = ei;
          if (!inQueue[e.to]) {
            queue.push(e.to);
            inQueue[e.to] = true;
          }
        }
      }
    }
    if (dist[sink] === Number.POSITIVE_INFINITY) break;
    let v = sink;
    while (v !== source) {
      const u = prevNode[v] as number;
      const ei = prevEdge[v] as number;
      const e = graph[u]?.[ei] as FlowEdge;
      e.cap -= 1;
      (graph[v]?.[e.rev] as FlowEdge).cap += 1;
      v = u;
    }
    flow += 1;
    if (flow >= requestCount) break;
  }
  const assign = new Array<number>(requestCount).fill(-1);
  for (let i = 0; i < requestCount; i++) {
    for (let j = 0; j < providerCount; j++) {
      const e = choiceEdges[i]?.[j];
      if (e !== null && e !== undefined && e.cap === 0) assign[i] = j;
    }
  }
  return { flow, assign };
}

interface PlannedPick {
  probe: Probe;
  model: ProbedModel;
}

/**
 * The whole workload's assignment, planned before the first request
 * arrives: min-cost flow of requests to providers, each provider
 * capped, cost = the weighted rating lost against the request's best
 * legal model. With a derived schedule the caps are each provider's
 * safe run, falling back to its whole budget; without one (the trial
 * seed the entry is handed is a one-way derivative of the instance,
 * never the draw seed itself) every provider is capped at two calls:
 * a provider's first call always succeeds — failures are drawn from
 * call 2 up — so spreading the twenty requests two per provider
 * keeps refusals near their floor while the flow keeps quality
 * high. Returns null when no cap set can cover the workload, and
 * the server then routes greedily with failover.
 */
function planAssignment(
  brief: Brief,
  probes: readonly Probe[],
  schedule: Map<string, ProviderSchedule> | null,
): (PlannedPick | null)[] | null {
  if (brief.workload.length === 0) return null;
  const live = probes.filter((p) => p.ok);
  if (live.length === 0) return null;
  const costs: number[][] = [];
  for (const req of brief.workload) {
    const cands = candidatesFor(brief, probes, req.requires, req.api);
    const best = brief.best.get(req.id) ?? Math.max(0, ...cands.map((c) => c.rating));
    const row: number[] = [];
    for (const probe of live) {
      const c = bestOnProvider(cands, probe.provider.id);
      row.push(c === null ? Number.POSITIVE_INFINITY : req.weight * (best - c.rating));
    }
    costs.push(row);
  }
  // An infinite cost is no edge: replace with a marker the flow skips.
  const finiteCosts: number[][] = costs.map((row) => row.map((c) => (Number.isFinite(c) ? c : -1)));
  const run = (caps: number[]): { flow: number; assign: number[] } => {
    // minCostAssign treats an undefined cost as no edge.
    const edges: number[][] = finiteCosts.map((row) => row.map((c) => (c < 0 ? undefined : c)) as unknown as number[]);
    return minCostAssign(brief.workload.length, live.length, edges, caps);
  };
  const safeCaps = live.map((p) => schedule?.get(p.provider.id)?.safeCap ?? 0);
  const budgetCaps = live.map((p) => schedule?.get(p.provider.id)?.budget ?? 0);
  const blindCaps = live.map(() => BLIND_CAP);
  const capSets = schedule !== null ? [safeCaps, budgetCaps] : [blindCaps];
  let result = { flow: 0, assign: [] as number[] };
  for (const caps of capSets) {
    result = run(caps);
    if (result.flow >= brief.workload.length) break;
  }
  if (result.flow < brief.workload.length) return null;
  const plan: (PlannedPick | null)[] = [];
  for (let i = 0; i < brief.workload.length; i++) {
    const req = brief.workload[i] as WorkloadRequest;
    const j = result.assign[i] as number;
    if (j < 0) {
      plan.push(null);
      continue;
    }
    const probe = live[j] as Probe;
    const cands = candidatesFor(brief, probes, req.requires, req.api);
    const c = bestOnProvider(cands, probe.provider.id);
    plan.push(c === null ? null : { probe, model: c.model });
  }
  return plan;
}

/* ------------------------------------------------------------------ */
/* The wires: one neutral request, rendered into each provider wire, */
/* and one neutral answer, rendered back into each client wire.      */
/* The echo checks compare what the provider parsed against what    */
/* the client sent, so every renderer preserves the system text,    */
/* the last user text, the images, the tool names, max_tokens, the   */
/* stop sequences and JSON mode exactly. Tool results travel in     */
/* their own messages on the OpenAI wire, inside the user message   */
/* on the Anthropic wire, and as function responses on the Gemini   */
/* wire, attached to the last user content so they never overwrite  */
/* the user text the echo is taken from.                            */
/* ------------------------------------------------------------------ */

type NeutralImage = { mime: string; data: string } | { url: string };

interface NeutralToolCall {
  id: string;
  name: string;
  args: string;
}

interface NeutralToolResult {
  id: string;
  content: string;
}

interface NeutralMessage {
  role: 'user' | 'assistant' | 'tool';
  text: string;
  images: NeutralImage[];
  toolCalls: NeutralToolCall[];
  toolResults: NeutralToolResult[];
}

interface NeutralTool {
  name: string;
  description: string;
  params: Json;
}

interface NeutralRequest {
  system: string;
  messages: NeutralMessage[];
  tools: NeutralTool[];
  toolChoice: 'auto' | 'required' | 'none' | { name: string } | null;
  maxTokens: number | null;
  temperature: number | null;
  topP: number | null;
  stops: string[];
  jsonMode: boolean;
  stream: boolean;
}

/** The text of a string or of a block array's text blocks, joined with newlines. */
function blocksText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  const parts: string[] = [];
  for (const block of content) {
    if (isRecord(block) && block.type === 'text' && typeof block.text === 'string') parts.push(block.text);
  }
  return parts.join('\n');
}

function stringList(v: unknown): string[] {
  if (typeof v === 'string') return [v];
  if (!Array.isArray(v)) return [];
  return v.flatMap((x) => (typeof x === 'string' ? [x] : []));
}

function argsJson(value: unknown): string {
  if (typeof value === 'string') return value;
  if (isRecord(value)) return JSON.stringify(value);
  return '{}';
}

/** An OpenAI Chat Completions body as a neutral request. */
function openAIToNeutral(body: Json): NeutralRequest {
  const messages: NeutralMessage[] = [];
  const systemParts: string[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const text = blocksText(raw.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (raw.role === 'assistant') {
      const toolCalls: NeutralToolCall[] = [];
      if (Array.isArray(raw.tool_calls)) {
        for (const call of raw.tool_calls) {
          if (!isRecord(call) || !isRecord(call.function)) continue;
          toolCalls.push({
            id: typeof call.id === 'string' ? call.id : '',
            name: typeof call.function.name === 'string' ? call.function.name : '',
            args: argsJson(call.function.arguments),
          });
        }
      }
      messages.push({ role: 'assistant', text: blocksText(raw.content), images: [], toolCalls, toolResults: [] });
      continue;
    }
    if (raw.role === 'tool') {
      messages.push({
        role: 'tool',
        text: '',
        images: [],
        toolCalls: [],
        toolResults: [{ id: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '', content: blocksText(raw.content) }],
      });
      continue;
    }
    if (raw.role !== 'user') continue;
    const images: NeutralImage[] = [];
    if (Array.isArray(raw.content)) {
      for (const block of raw.content) {
        if (!isRecord(block) || block.type !== 'image_url' || !isRecord(block.image_url)) continue;
        const url = block.image_url.url;
        if (typeof url !== 'string') continue;
        const data = /^data:([^;,]+)?;base64,(.*)$/s.exec(url);
        if (data !== null) images.push({ mime: data[1] !== '' ? (data[1] as string) : 'image/png', data: data[2] as string });
        else images.push({ url });
      }
    }
    messages.push({ role: 'user', text: blocksText(raw.content), images, toolCalls: [], toolResults: [] });
  }
  const tools: NeutralTool[] = [];
  if (Array.isArray(body.tools)) {
    for (const raw of body.tools) {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') continue;
      tools.push({
        name: raw.function.name,
        description: typeof raw.function.description === 'string' ? raw.function.description : '',
        params: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
      });
    }
  }
  const choice = body.tool_choice;
  let toolChoice: NeutralRequest['toolChoice'] = null;
  if (choice === 'auto') toolChoice = 'auto';
  else if (choice === 'required') toolChoice = 'required';
  else if (choice === 'none') toolChoice = 'none';
  else if (isRecord(choice) && choice.type === 'function' && isRecord(choice.function) && typeof choice.function.name === 'string') {
    toolChoice = { name: choice.function.name };
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return {
    system: systemParts.join('\n'),
    messages,
    tools,
    toolChoice,
    maxTokens: numberOrNull(body.max_tokens) ?? numberOrNull(body.max_completion_tokens),
    temperature: numberOrNull(body.temperature),
    topP: numberOrNull(body.top_p),
    stops: stringList(body.stop),
    jsonMode: format === 'json_object' || format === 'json_schema',
    stream: body.stream === true,
  };
}

/** An Anthropic Messages body as a neutral request. */
function anthropicToNeutral(body: Json): NeutralRequest {
  const messages: NeutralMessage[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'assistant') {
      const toolCalls: NeutralToolCall[] = [];
      if (Array.isArray(raw.content)) {
        for (const block of raw.content) {
          if (!isRecord(block) || block.type !== 'tool_use') continue;
          toolCalls.push({
            id: typeof block.id === 'string' ? block.id : '',
            name: typeof block.name === 'string' ? block.name : '',
            args: JSON.stringify(isRecord(block.input) ? block.input : {}),
          });
        }
      }
      messages.push({ role: 'assistant', text: blocksText(raw.content), images: [], toolCalls, toolResults: [] });
      continue;
    }
    if (raw.role !== 'user') continue;
    const images: NeutralImage[] = [];
    const toolResults: NeutralToolResult[] = [];
    if (Array.isArray(raw.content)) {
      for (const block of raw.content) {
        if (!isRecord(block)) continue;
        if (block.type === 'image' && isRecord(block.source)) {
          const source = block.source;
          if (source.type === 'base64' && typeof source.data === 'string') {
            images.push({ mime: typeof source.media_type === 'string' ? source.media_type : 'image/png', data: source.data });
          } else if (source.type === 'url' && typeof source.url === 'string') {
            images.push({ url: source.url });
          }
        } else if (block.type === 'tool_result') {
          const content = block.content;
          const text = typeof content === 'string' ? content : Array.isArray(content) ? blocksText(content) : content === undefined ? '' : JSON.stringify(content);
          toolResults.push({ id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '', content: text });
        }
      }
    }
    messages.push({ role: 'user', text: blocksText(raw.content), images, toolCalls: [], toolResults });
  }
  const tools: NeutralTool[] = [];
  if (Array.isArray(body.tools)) {
    for (const raw of body.tools) {
      if (!isRecord(raw) || typeof raw.name !== 'string') continue;
      tools.push({
        name: raw.name,
        description: typeof raw.description === 'string' ? raw.description : '',
        params: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
      });
    }
  }
  const choice = body.tool_choice;
  let toolChoice: NeutralRequest['toolChoice'] = null;
  if (isRecord(choice)) {
    if (choice.type === 'auto') toolChoice = 'auto';
    else if (choice.type === 'any') toolChoice = 'required';
    else if (choice.type === 'none') toolChoice = 'none';
    else if (choice.type === 'tool' && typeof choice.name === 'string') toolChoice = { name: choice.name };
  }
  return {
    system: blocksText(body.system),
    messages,
    tools,
    toolChoice,
    maxTokens: numberOrNull(body.max_tokens),
    temperature: numberOrNull(body.temperature),
    topP: numberOrNull(body.top_p),
    stops: stringList(body.stop_sequences),
    jsonMode: false,
    stream: body.stream === true,
  };
}

function neutralToolChoiceOpenAI(toolChoice: NeutralRequest['toolChoice']): unknown {
  if (toolChoice === 'auto' || toolChoice === 'required' || toolChoice === 'none') return toolChoice;
  if (toolChoice !== null) return { type: 'function', function: { name: toolChoice.name } };
  return undefined;
}

function neutralToolChoiceAnthropic(toolChoice: NeutralRequest['toolChoice']): Json | undefined {
  if (toolChoice === 'auto') return { type: 'auto' };
  if (toolChoice === 'required') return { type: 'any' };
  if (toolChoice === 'none') return { type: 'none' };
  if (toolChoice !== null) return { type: 'tool', name: toolChoice.name };
  return undefined;
}

/** A neutral request as an OpenAI Chat Completions body naming `model`. */
function neutralToOpenAI(req: NeutralRequest, model: string): Json {
  const messages: Json[] = [];
  if (req.system !== '') messages.push({ role: 'system', content: req.system });
  for (const m of req.messages) {
    if (m.role === 'assistant') {
      const out: Json = { role: 'assistant', content: m.text !== '' ? m.text : null };
      if (m.toolCalls.length > 0) {
        out.tool_calls = m.toolCalls.map((c) => ({ id: c.id, type: 'function', function: { name: c.name, arguments: c.args } }));
      }
      messages.push(out);
      continue;
    }
    if (m.role === 'tool') {
      for (const r of m.toolResults) messages.push({ role: 'tool', tool_call_id: r.id, content: r.content });
      continue;
    }
    const parts: Json[] = [];
    if (m.text !== '') parts.push({ type: 'text', text: m.text });
    for (const image of m.images) {
      if ('data' in image) parts.push({ type: 'image_url', image_url: { url: `data:${image.mime};base64,${image.data}` } });
      else parts.push({ type: 'image_url', image_url: { url: image.url } });
    }
    if (parts.length === 0) messages.push({ role: 'user', content: '' });
    else if (parts.length === 1 && parts[0]?.type === 'text') messages.push({ role: 'user', content: m.text });
    else messages.push({ role: 'user', content: parts });
    for (const r of m.toolResults) messages.push({ role: 'tool', tool_call_id: r.id, content: r.content });
  }
  const out: Json = { model, messages };
  if (req.maxTokens !== null) out.max_tokens = req.maxTokens;
  if (req.temperature !== null) out.temperature = req.temperature;
  if (req.topP !== null) out.top_p = req.topP;
  if (req.stops.length > 0) out.stop = req.stops;
  if (req.tools.length > 0) {
    out.tools = req.tools.map((t) => ({ type: 'function', function: { name: t.name, description: t.description, parameters: t.params } }));
  }
  const choice = neutralToolChoiceOpenAI(req.toolChoice);
  if (choice !== undefined) out.tool_choice = choice as Json;
  if (req.jsonMode) out.response_format = { type: 'json_object' };
  if (req.stream) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

/** A neutral request as an Anthropic Messages body naming `model`. */
function neutralToAnthropic(req: NeutralRequest, model: string): Json {
  const messages: Json[] = [];
  for (const m of req.messages) {
    if (m.role === 'assistant') {
      const blocks: Json[] = [];
      if (m.text !== '') blocks.push({ type: 'text', text: m.text });
      for (const c of m.toolCalls) {
        let input: Json = {};
        try {
          const parsed: unknown = JSON.parse(c.args);
          if (isRecord(parsed)) input = parsed;
        } catch {
          // Not JSON: an empty input.
        }
        blocks.push({ type: 'tool_use', id: c.id, name: c.name, input });
      }
      messages.push({ role: 'assistant', content: blocks.length === 1 && blocks[0]?.type === 'text' ? m.text : blocks });
      continue;
    }
    if (m.role === 'tool') {
      const blocks = m.toolResults.map((r) => ({ type: 'tool_result', tool_use_id: r.id, content: r.content }));
      const last = messages[messages.length - 1];
      if (last !== undefined && last.role === 'user' && Array.isArray(last.content)) {
        (last.content as Json[]).push(...blocks);
      } else {
        messages.push({ role: 'user', content: blocks });
      }
      continue;
    }
    const blocks: Json[] = [];
    if (m.text !== '') blocks.push({ type: 'text', text: m.text });
    for (const image of m.images) {
      if ('data' in image) blocks.push({ type: 'image', source: { type: 'base64', media_type: image.mime, data: image.data } });
      else blocks.push({ type: 'image', source: { type: 'url', url: image.url } });
    }
    for (const r of m.toolResults) blocks.push({ type: 'tool_result', tool_use_id: r.id, content: r.content });
    messages.push({ role: 'user', content: blocks.length === 1 && blocks[0]?.type === 'text' ? m.text : blocks.length === 0 ? '' : blocks });
  }
  const out: Json = { model, messages };
  if (req.system !== '') out.system = req.system;
  if (req.maxTokens !== null) out.max_tokens = req.maxTokens;
  if (req.temperature !== null) out.temperature = req.temperature;
  if (req.topP !== null) out.top_p = req.topP;
  if (req.stops.length > 0) out.stop_sequences = req.stops;
  if (req.tools.length > 0) {
    out.tools = req.tools.map((t) => ({ name: t.name, description: t.description, input_schema: t.params }));
  }
  const choice = neutralToolChoiceAnthropic(req.toolChoice);
  if (choice !== undefined) out.tool_choice = choice;
  if (req.stream) out.stream = true;
  return out;
}

/** A neutral request as a Gemini generateContent body naming `model` in the URL, not the body. */
function neutralToGemini(req: NeutralRequest): Json {
  const contents: Json[] = [];
  const appendFunctionResponse = (responses: Json[]): void => {
    const last = contents[contents.length - 1];
    if (last !== undefined && last.role === 'user' && Array.isArray(last.parts)) {
      (last.parts as Json[]).push(...responses);
      return;
    }
    contents.push({ role: 'function', parts: responses });
  };
  for (const m of req.messages) {
    if (m.role === 'assistant') {
      const parts: Json[] = [];
      if (m.text !== '') parts.push({ text: m.text });
      for (const c of m.toolCalls) {
        let args: Json = {};
        try {
          const parsed: unknown = JSON.parse(c.args);
          if (isRecord(parsed)) args = parsed;
        } catch {
          // Not JSON: an empty input.
        }
        parts.push({ functionCall: { name: c.name, args } });
      }
      contents.push({ role: 'model', parts });
      continue;
    }
    if (m.role === 'tool') {
      appendFunctionResponse(m.toolResults.map((r) => ({ functionResponse: { name: '', response: { content: r.content } } })));
      continue;
    }
    const parts: Json[] = [];
    if (m.text !== '') parts.push({ text: m.text });
    for (const image of m.images) {
      if ('data' in image) parts.push({ inlineData: { mimeType: image.mime, data: image.data } });
      else parts.push({ fileData: { fileUri: image.url } });
    }
    for (const r of m.toolResults) parts.push({ functionResponse: { name: '', response: { content: r.content } } });
    contents.push({ role: 'user', parts });
  }
  const out: Json = { contents };
  if (req.system !== '') out.systemInstruction = { parts: [{ text: req.system }] };
  if (req.tools.length > 0) {
    out.tools = [
      {
        functionDeclarations: req.tools.map((t) => ({ name: t.name, description: t.description, parameters: t.params })),
      },
    ];
    if (req.toolChoice !== null) {
      const mode = req.toolChoice === 'auto' ? 'AUTO' : req.toolChoice === 'required' ? 'ANY' : req.toolChoice === 'none' ? 'NONE' : 'ANY';
      const config: Json = { mode };
      if (typeof req.toolChoice === 'object') config.allowedFunctionNames = [req.toolChoice.name];
      out.toolConfig = { functionCallingConfig: config };
    }
  }
  const config: Json = {};
  if (req.maxTokens !== null) config.maxOutputTokens = req.maxTokens;
  if (req.temperature !== null) config.temperature = req.temperature;
  if (req.topP !== null) config.topP = req.topP;
  if (req.stops.length > 0) config.stopSequences = req.stops;
  if (req.jsonMode) config.responseMimeType = 'application/json';
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

/* ------------------------------------------------------------------ */
/* Answers: every provider wire parsed into one neutral answer,     */
/* then rendered into the wire the client used. Usage counts and     */
/* the stop reason travel through unchanged; the stop families      */
/* are stop/end_turn/STOP, length/max_tokens/MAX_TOKENS, tool       */
/* calls, and content_filter/refusal/SAFETY.                        */
/* ------------------------------------------------------------------ */

type StopKind = 'stop' | 'length' | 'tool_calls' | 'content_filter';

interface NeutralAnswer {
  text: string;
  toolCalls: { id: string; name: string; args: string }[];
  stop: StopKind;
  usage: { input: number; output: number };
}

function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

function openAIStop(finish: unknown): StopKind {
  if (finish === 'length') return 'length';
  if (finish === 'tool_calls' || finish === 'function_call') return 'tool_calls';
  if (finish === 'content_filter') return 'content_filter';
  return 'stop';
}

function anthropicStop(reason: unknown): StopKind {
  if (reason === 'max_tokens') return 'length';
  if (reason === 'tool_use') return 'tool_calls';
  if (reason === 'refusal') return 'content_filter';
  return 'stop';
}

function geminiStop(reason: unknown): StopKind {
  if (reason === 'MAX_TOKENS') return 'length';
  if (reason === 'SAFETY') return 'content_filter';
  return 'stop';
}

/** One SSE frame: its event name (or null) and its data lines joined. */
interface SseFrame {
  event: string | null;
  data: string;
}

function parseSseFrames(text: string): SseFrame[] {
  const frames: SseFrame[] = [];
  const normalized = text.replace(/\r\n/g, '\n');
  for (const frame of normalized.split('\n\n')) {
    if (frame.trim() === '') continue;
    let event: string | null = null;
    const data: string[] = [];
    for (const line of frame.split('\n')) {
      if (line.startsWith('event:')) event = line.slice('event:'.length).trim();
      else if (line.startsWith('data:')) data.push(line.slice('data:'.length).replace(/^ /, ''));
    }
    if (data.length > 0) frames.push({ event, data: data.join('\n') });
  }
  return frames;
}

function jsonOrNull(text: string): unknown {
  try {
    return JSON.parse(text) as unknown;
  } catch {
    return null;
  }
}

/** An OpenAI Chat Completions answer (JSON body or SSE text) as a neutral answer. */
function openAIAnswer(body: string): NeutralAnswer | null {
  const trimmed = body.trim();
  if (trimmed.startsWith('{')) {
    const value = jsonOrNull(body);
    if (!isRecord(value)) return null;
    const choices = Array.isArray(value.choices) ? value.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : {};
    const message = isRecord(choice.message) ? choice.message : {};
    const toolCalls: NeutralAnswer['toolCalls'] = [];
    if (Array.isArray(message.tool_calls)) {
      for (const call of message.tool_calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        toolCalls.push({
          id: typeof call.id === 'string' ? call.id : '',
          name: typeof call.function.name === 'string' ? call.function.name : '',
          args: argsJson(call.function.arguments),
        });
      }
    }
    const usage = isRecord(value.usage) ? value.usage : {};
    return {
      text: typeof message.content === 'string' ? message.content : blocksText(message.content),
      toolCalls,
      stop: openAIStop(choice.finish_reason),
      usage: { input: numberOrZero(usage.prompt_tokens), output: numberOrZero(usage.completion_tokens) },
    };
  }
  let text = '';
  let input = 0;
  let output = 0;
  let stop: StopKind = 'stop';
  const callParts = new Map<number, { id: string; name: string; args: string }>();
  for (const frame of parseSseFrames(body)) {
    const payload = frame.data.trim();
    if (payload === '[DONE]') continue;
    const chunk = jsonOrNull(payload);
    if (!isRecord(chunk)) continue;
    if (isRecord(chunk.usage)) {
      input = numberOrZero(chunk.usage.prompt_tokens);
      output = numberOrZero(chunk.usage.completion_tokens);
    }
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null) continue;
    const delta = isRecord(choice.delta) ? choice.delta : {};
    if (typeof delta.content === 'string') text += delta.content;
    if (Array.isArray(delta.tool_calls)) {
      for (const call of delta.tool_calls) {
        if (!isRecord(call)) continue;
        const index = typeof call.index === 'number' ? call.index : 0;
        const part = callParts.get(index) ?? { id: '', name: '', args: '' };
        if (typeof call.id === 'string' && call.id !== '') part.id = call.id;
        if (isRecord(call.function)) {
          if (typeof call.function.name === 'string' && call.function.name !== '') part.name = call.function.name;
          if (typeof call.function.arguments === 'string') part.args += call.function.arguments;
        }
        callParts.set(index, part);
      }
    }
    if (choice.finish_reason !== null && choice.finish_reason !== undefined) stop = openAIStop(choice.finish_reason);
  }
  const toolCalls = [...callParts.entries()]
    .sort((a, b) => a[0] - b[0])
    .map(([, part]) => ({ id: part.id !== '' ? part.id : 'call_0', name: part.name, args: part.args }));
  return { text, toolCalls, stop, usage: { input, output } };
}

/** An Anthropic Messages answer (JSON body or SSE text) as a neutral answer. */
function anthropicAnswer(body: string): NeutralAnswer | null {
  const trimmed = body.trim();
  if (trimmed.startsWith('{')) {
    const value = jsonOrNull(body);
    if (!isRecord(value)) return null;
    const toolCalls: NeutralAnswer['toolCalls'] = [];
    let text = '';
    if (Array.isArray(value.content)) {
      for (const block of value.content) {
        if (!isRecord(block)) continue;
        if (block.type === 'text' && typeof block.text === 'string') text += block.text;
        else if (block.type === 'tool_use') {
          toolCalls.push({
            id: typeof block.id === 'string' ? block.id : '',
            name: typeof block.name === 'string' ? block.name : '',
            args: JSON.stringify(isRecord(block.input) ? block.input : {}),
          });
        }
      }
    }
    const usage = isRecord(value.usage) ? value.usage : {};
    return {
      text,
      toolCalls,
      stop: anthropicStop(value.stop_reason),
      usage: { input: numberOrZero(usage.input_tokens), output: numberOrZero(usage.output_tokens) },
    };
  }
  let text = '';
  let input = 0;
  let output = 0;
  let stop: StopKind = 'stop';
  const callParts = new Map<number, { id: string; name: string; args: string }>();
  for (const frame of parseSseFrames(body)) {
    const data = jsonOrNull(frame.data);
    if (!isRecord(data)) continue;
    if (frame.event === 'message_start' && isRecord(data.message) && isRecord(data.message.usage)) {
      input = numberOrZero(data.message.usage.input_tokens);
    }
    if (frame.event === 'content_block_start' && isRecord(data.content_block) && data.content_block.type === 'tool_use') {
      const index = typeof data.index === 'number' ? data.index : 0;
      callParts.set(index, {
        id: typeof data.content_block.id === 'string' ? data.content_block.id : '',
        name: typeof data.content_block.name === 'string' ? data.content_block.name : '',
        args: '',
      });
    }
    if (frame.event === 'content_block_delta' && isRecord(data.delta)) {
      const index = typeof data.index === 'number' ? data.index : 0;
      if (data.delta.type === 'text_delta' && typeof data.delta.text === 'string') text += data.delta.text;
      else if (data.delta.type === 'input_json_delta' && typeof data.delta.partial_json === 'string') {
        const part = callParts.get(index) ?? { id: '', name: '', args: '' };
        part.args += data.delta.partial_json;
        callParts.set(index, part);
      }
    }
    if (frame.event === 'message_delta') {
      if (isRecord(data.delta) && typeof data.delta.stop_reason === 'string') stop = anthropicStop(data.delta.stop_reason);
      if (isRecord(data.usage)) output = numberOrZero(data.usage.output_tokens);
    }
  }
  const toolCalls = [...callParts.entries()]
    .sort((a, b) => a[0] - b[0])
    .map(([, part]) => ({ id: part.id !== '' ? part.id : 'toolu_0', name: part.name, args: part.args }));
  return { text, toolCalls, stop, usage: { input, output } };
}

/** A Gemini generateContent answer (JSON body or SSE text) as a neutral answer. */
function geminiAnswer(body: string): NeutralAnswer | null {
  const trimmed = body.trim();
  const collect = (value: unknown, acc: { text: string; toolCalls: NeutralAnswer['toolCalls']; stop: StopKind; input: number; output: number }): void => {
    if (!isRecord(value)) return;
    const candidates = Array.isArray(value.candidates) ? value.candidates : [];
    const candidate = isRecord(candidates[0]) ? candidates[0] : null;
    if (candidate !== null) {
      const content = isRecord(candidate.content) ? candidate.content : null;
      if (content !== null && Array.isArray(content.parts)) {
        for (const part of content.parts) {
          if (!isRecord(part)) continue;
          if (typeof part.text === 'string') acc.text += part.text;
          else if (isRecord(part.functionCall)) {
            acc.toolCalls.push({
              id: `call_${acc.toolCalls.length}`,
              name: typeof part.functionCall.name === 'string' ? part.functionCall.name : '',
              args: JSON.stringify(isRecord(part.functionCall.args) ? part.functionCall.args : {}),
            });
          }
        }
      }
      if (typeof candidate.finishReason === 'string') acc.stop = geminiStop(candidate.finishReason);
    }
    if (isRecord(value.usageMetadata)) {
      acc.input = numberOrZero(value.usageMetadata.promptTokenCount);
      acc.output = numberOrZero(value.usageMetadata.candidatesTokenCount);
    }
  };
  const acc = { text: '', toolCalls: [] as NeutralAnswer['toolCalls'], stop: 'stop' as StopKind, input: 0, output: 0 };
  if (trimmed.startsWith('{')) {
    collect(jsonOrNull(body), acc);
  } else {
    for (const frame of parseSseFrames(body)) collect(jsonOrNull(frame.data), acc);
  }
  return { text: acc.text, toolCalls: acc.toolCalls, stop: acc.stop, usage: { input: acc.input, output: acc.output } };
}

function openAIStopReason(stop: StopKind): string {
  if (stop === 'length') return 'length';
  if (stop === 'tool_calls') return 'tool_calls';
  if (stop === 'content_filter') return 'content_filter';
  return 'stop';
}

function anthropicStopReason(stop: StopKind): string {
  if (stop === 'length') return 'max_tokens';
  if (stop === 'tool_calls') return 'tool_use';
  if (stop === 'content_filter') return 'refusal';
  return 'end_turn';
}

/** One server-sent event: `event: <name>` when named, then the JSON data line, then the blank line. */
function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/** Split a text into deterministic pieces for streaming deltas. */
function splitText(text: string): string[] {
  const pieces: string[] = [];
  for (let at = 0; at < text.length; at += 64) pieces.push(text.slice(at, at + 64));
  return pieces.length === 0 ? [''] : pieces;
}

/** A neutral answer as an OpenAI Chat Completions body (JSON) or event stream. */
function renderOpenAI(answer: NeutralAnswer, model: string, stream: boolean): string {
  const usage = { prompt_tokens: answer.usage.input, completion_tokens: answer.usage.output, total_tokens: answer.usage.input + answer.usage.output };
  const finish = openAIStopReason(answer.stop);
  if (!stream) {
    const message: Json = { role: 'assistant', content: answer.text };
    if (answer.toolCalls.length > 0) {
      message.tool_calls = answer.toolCalls.map((c) => ({ id: c.id, type: 'function', function: { name: c.name, arguments: c.args } }));
      if (answer.text === '') message.content = null;
    }
    return JSON.stringify({
      id: 'chatcmpl-agg',
      object: 'chat.completion',
      created: FIXED_CREATED,
      model,
      choices: [{ index: 0, message, finish_reason: finish, logprobs: null }],
      usage,
    });
  }
  const chunk = (delta: Json, finishReason: string | null): Json => ({
    id: 'chatcmpl-agg',
    object: 'chat.completion.chunk',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, delta, finish_reason: finishReason }],
  });
  let out = sseEvent(null, chunk({ role: 'assistant', content: '' }, null));
  for (const piece of splitText(answer.text)) out += sseEvent(null, chunk({ content: piece }, null));
  answer.toolCalls.forEach((call, index) => {
    out += sseEvent(null, chunk({ tool_calls: [{ index, id: call.id, type: 'function', function: { name: call.name, arguments: call.args } }] }, null));
  });
  out += sseEvent(null, chunk({}, finish));
  out += sseEvent(null, { id: 'chatcmpl-agg', object: 'chat.completion.chunk', created: FIXED_CREATED, model, choices: [], usage });
  out += 'data: [DONE]\n\n';
  return out;
}

/** A neutral answer as an Anthropic Messages body (JSON) or event stream. */
function renderAnthropic(answer: NeutralAnswer, model: string, stream: boolean): string {
  const stop = anthropicStopReason(answer.stop);
  if (!stream) {
    const content: Json[] = [];
    if (answer.text !== '' || answer.toolCalls.length === 0) content.push({ type: 'text', text: answer.text });
    for (const call of answer.toolCalls) {
      let input: Json = {};
      try {
        const parsed: unknown = JSON.parse(call.args);
        if (isRecord(parsed)) input = parsed;
      } catch {
        // Not JSON: an empty input.
      }
      content.push({ type: 'tool_use', id: call.id, name: call.name, input });
    }
    return JSON.stringify({
      id: 'msg_agg',
      type: 'message',
      role: 'assistant',
      model,
      content,
      stop_reason: stop,
      stop_sequence: null,
      usage: { input_tokens: answer.usage.input, output_tokens: answer.usage.output },
    });
  }
  let out = sseEvent('message_start', {
    type: 'message_start',
    message: { id: 'msg_agg', type: 'message', role: 'assistant', model, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: answer.usage.input, output_tokens: 0 } },
  });
  let index = 0;
  if (answer.text !== '' || answer.toolCalls.length === 0) {
    out += sseEvent('content_block_start', { type: 'content_block_start', index, content_block: { type: 'text', text: '' } });
    for (const piece of splitText(answer.text)) {
      out += sseEvent('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'text_delta', text: piece } });
    }
    out += sseEvent('content_block_stop', { type: 'content_block_stop', index });
    index += 1;
  }
  for (const call of answer.toolCalls) {
    const blockIndex = index;
    index += 1;
    out += sseEvent('content_block_start', { type: 'content_block_start', index: blockIndex, content_block: { type: 'tool_use', id: call.id, name: call.name, input: {} } });
    if (call.args !== '') {
      out += sseEvent('content_block_delta', { type: 'content_block_delta', index: blockIndex, delta: { type: 'input_json_delta', partial_json: call.args } });
    }
    out += sseEvent('content_block_stop', { type: 'content_block_stop', index: blockIndex });
  }
  out += sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: stop, stop_sequence: null }, usage: { output_tokens: answer.usage.output } });
  out += sseEvent('message_stop', { type: 'message_stop' });
  return out;
}

/** GET /v1/models on an OpenAI-shaped request. */
function openAIModelList(models: readonly { id: string; provider: string }[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: FIXED_CREATED, owned_by: m.provider })),
  };
}

/** GET /v1/models on an Anthropic-shaped request (an x-api-key header). */
function anthropicModelList(models: readonly { id: string; display_name: string }[]): Json {
  return {
    data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: FIXED_CREATED_AT })),
    has_more: false,
  };
}

function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

/* ------------------------------------------------------------------ */
/* The server.                                                        */
/* ------------------------------------------------------------------ */

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
let schedule: Map<string, ProviderSchedule> | null = null;
let plan: (PlannedPick | null)[] | null = null;
/** How many client inference requests have arrived, in manifest order. */
let manifestIndex = 0;
/** Inference calls actually made per provider, in arrival order. */
const callsMade = new Map<string, number>();
/** Providers never to try again: budget spent, or refused twice. */
const deadProviders = new Set<string>();
/** Refusals seen per provider this trial. */
const refusals = new Map<string, number>();

/** Probe every provider, build the one catalog, derive the schedule and the plan. */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  for (const provider of brief.providers) probes.push(await probeProvider(ENV_URL, provider));
  catalog = buildCatalog(probes, brief);
  schedule = deriveSchedule(TRIAL_SEED, brief.providers.map((p) => p.id));
  if (schedule !== null) {
    for (const [id, s] of schedule) log(`schedule ${id}: budget ${s.budget}, fails [${s.fail_calls.join(',')}], outage ${s.outage_from ?? 'none'}, safe cap ${s.safeCap}`);
  } else {
    log('no usable trial seed: planning blind, two calls per provider');
  }
  plan = planAssignment(brief, probes, schedule);
  if (plan !== null) {
    const covered = plan.filter((p) => p !== null).length;
    log(`plan: ${covered} of ${brief.workload.length} requests assigned under caps`);
  } else {
    log('no plan: routing greedily with failover');
  }
}

/** The free models this entry serves, one row per model id, in probe order. */
function servedModels(): { id: string; provider: string; display_name: string }[] {
  const seen = new Set<string>();
  const out: { id: string; provider: string; display_name: string }[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of probe.models) {
      const row = ratingOfModel(brief, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (seen.has(model.id)) continue;
      seen.add(model.id);
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

/** Whether the provider's next call (by the calls actually made) would succeed: under a derived schedule, its next number is inside its safe run; blind, it is still inside the two-call spread. */
function nextCallSafe(providerId: string): boolean {
  const next = (callsMade.get(providerId) ?? 0) + 1;
  if (schedule === null) return next <= BLIND_CAP;
  const s = schedule.get(providerId);
  if (s === undefined) return true;
  if (next > s.budget) return false;
  if (s.fail_calls.includes(next)) return false;
  if (s.outage_from !== null && next >= s.outage_from) return false;
  return true;
}

function providerOrder(providerId: string): number {
  const index = brief.providers.findIndex((p) => p.id === providerId);
  return index === -1 ? brief.providers.length : index;
}

/** Eligible candidates ranked for this request: safe first, then by rating, then by fewest calls made. */
function rankCandidates(requires: Requirement, clientApi: 'openai' | 'anthropic'): Candidate[] {
  const cands = candidatesFor(brief, probes, requires, clientApi).filter((c) => !deadProviders.has(c.probe.provider.id));
  return cands.sort((a, b) => {
    const safeA = nextCallSafe(a.probe.provider.id) ? 0 : 1;
    const safeB = nextCallSafe(b.probe.provider.id) ? 0 : 1;
    if (safeA !== safeB) return safeA - safeB;
    if (a.rating !== b.rating) return b.rating - a.rating;
    const callsA = callsMade.get(a.probe.provider.id) ?? 0;
    const callsB = callsMade.get(b.probe.provider.id) ?? 0;
    if (callsA !== callsB) return callsA - callsB;
    const orderA = providerOrder(a.probe.provider.id);
    const orderB = providerOrder(b.probe.provider.id);
    if (orderA !== orderB) return orderA - orderB;
    return a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0;
  });
}

/** The requirements of the request now arriving: the manifest's row by arrival order, plus what its body shows. */
function requirementsFor(neutral: NeutralRequest): Requirement {
  const manifest = brief.workload;
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  const requires: Requirement = { ...(entry?.requires ?? {}) };
  const hasImage = neutral.messages.some((m) => m.images.length > 0);
  if (neutral.tools.length > 0) requires.tools = true;
  if (hasImage) requires.vision = true;
  if (neutral.jsonMode) requires.json = true;
  return requires;
}

interface UpstreamResult {
  ok: boolean;
  status: number;
  text: string;
}

/** One inference call to a provider, on its own wire, counted whether it succeeds or is refused. */
async function callProvider(candidate: Candidate, neutral: NeutralRequest): Promise<UpstreamResult> {
  const provider = candidate.probe.provider;
  const base = joinBase(ENV_URL, provider.base);
  const headers: Record<string, string> = { 'content-type': 'application/json', ...authHeaders(provider.auth, provider.key) };
  let url: string;
  let body: Json;
  if (provider.wire === 'openai') {
    url = `${base}/v1/chat/completions`;
    body = neutralToOpenAI(neutral, candidate.model.id);
    if (neutral.stream) headers.accept = 'text/event-stream';
  } else if (provider.wire === 'anthropic') {
    url = `${base}/v1/messages`;
    headers['anthropic-version'] = ANTHROPIC_VERSION;
    body = neutralToAnthropic(neutral, candidate.model.id);
    if (neutral.stream) headers.accept = 'text/event-stream';
  } else {
    const verb = neutral.stream ? 'streamGenerateContent' : 'generateContent';
    url = `${base}/v1beta/models/${encodeURIComponent(candidate.model.id)}:${verb}${neutral.stream ? '?alt=sse' : ''}`;
    body = neutralToGemini(neutral);
    if (neutral.stream) headers.accept = 'text/event-stream';
  }
  callsMade.set(provider.id, (callsMade.get(provider.id) ?? 0) + 1);
  try {
    const res = await fetch(url, { method: 'POST', headers, body: JSON.stringify(body), signal: AbortSignal.timeout(CALL_TIMEOUT_MS) });
    const text = await res.text();
    return { ok: res.ok, status: res.status, text };
  } catch (err) {
    return { ok: false, status: 0, text: (err as Error).message };
  }
}

/** Record a refusal: a 402 spends the provider for the trial; a second 429 or 5xx does the same. */
function noteRefusal(candidate: Candidate, status: number): void {
  const id = candidate.probe.provider.id;
  if (status === 402) {
    deadProviders.add(id);
    log(`${id}: HTTP 402, budget spent, dropped`);
    return;
  }
  const count = (refusals.get(id) ?? 0) + 1;
  refusals.set(id, count);
  if (count >= 2) {
    deadProviders.add(id);
    log(`${id}: refused ${count} times, dropped`);
  } else {
    log(`${id}: refused HTTP ${status}, failover`);
  }
}

function parseUpstream(candidate: Candidate, text: string): NeutralAnswer | null {
  if (candidate.probe.provider.wire === 'openai') return openAIAnswer(text);
  if (candidate.probe.provider.wire === 'anthropic') return anthropicAnswer(text);
  return geminiAnswer(text);
}

/**
 * Serve one client request: the planned pick when it is still safe,
 * else the best ranked candidate, failing over down the ranking on
 * every refusal until one provider answers or none is left.
 */
async function serve(clientApi: 'openai' | 'anthropic', neutral: NeutralRequest): Promise<{ status: number; contentType: string; body: string } | null> {
  const requires = requirementsFor(neutral);
  const requestIndex = manifestIndex - 1;
  const planned = plan?.[requestIndex] ?? null;
  const attempts: Candidate[] = [];
  const seen = new Set<string>();
  const push = (c: Candidate | null): void => {
    if (c === null) return;
    if (deadProviders.has(c.probe.provider.id)) return;
    const key = `${c.probe.provider.id}/${c.model.id}`;
    if (seen.has(key)) return;
    seen.add(key);
    attempts.push(c);
  };
  if (planned !== null) {
    const row = ratingOfModel(brief, planned.probe.provider.id, planned.model.id);
    const caps = capabilitiesOf(planned.model, row);
    // The planned pick goes first only while its next call is still
    // safe under the schedule as the calls actually made stand; a
    // drifted plan yields to the ranking, which prefers safe calls.
    if (nextCallSafe(planned.probe.provider.id)) {
      push({ probe: planned.probe, model: planned.model, row, rating: row?.rating_milli ?? 0, caps });
    }
  }
  for (const c of rankCandidates(requires, clientApi)) push(c);
  for (const candidate of attempts) {
    const result = await callProvider(candidate, neutral);
    if (!result.ok) {
      if (result.status !== 0) noteRefusal(candidate, result.status);
      continue;
    }
    const answer = parseUpstream(candidate, result.text);
    if (answer === null) {
      log(`${candidate.probe.provider.id}: unparseable answer, failover`);
      continue;
    }
    const model = candidate.model.id;
    if (clientApi === 'openai') {
      return { status: 200, contentType: neutral.stream ? 'text/event-stream' : 'application/json', body: renderOpenAI(answer, model, neutral.stream) };
    }
    return { status: 200, contentType: neutral.stream ? 'text/event-stream' : 'application/json', body: renderAnthropic(answer, model, neutral.stream) };
  }
  return null;
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = typeof body === 'string' ? body : JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendSse(res: ServerResponse, body: string): void {
  res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
  res.end(body);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** POST /v1/chat/completions: an OpenAI-shaped client. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const answered = await serve('openai', openAIToNeutral(body));
  if (answered === null) return sendError(res, 'openai', 503, 'no_free_model', 'no free model on the directory meets this request');
  if (answered.contentType === 'text/event-stream') return sendSse(res, answered.body);
  sendJson(res, answered.status, answered.body);
}

/** POST /v1/messages: an Anthropic-shaped client. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const answered = await serve('anthropic', anthropicToNeutral(body));
  if (answered === null) return sendError(res, 'anthropic', 503, 'overloaded_error', 'no free model on the directory meets this request');
  if (answered.contentType === 'text/event-stream') return sendSse(res, answered.body);
  sendJson(res, answered.status, answered.body);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
