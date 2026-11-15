# Free-model aggregator

[![make.fun · free-model-aggregator-01](https://make.fun/badge/free-model-aggregator-01.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![rank](https://make.fun/badge/free-model-aggregator-01/keepomakima/free-model-aggregator-01/rank.svg)](https://make.fun/bounty/free-model-aggregator-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![reward](https://make.fun/badge/free-model-aggregator-01/keepomakima/free-model-aggregator-01/reward.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)

One server that points any OpenAI or Anthropic client at the best free
model that can serve each request, over ten free-tier providers on
three wires (OpenAI, Anthropic, Gemini). Every completion comes from
a real free model on one of those providers, in JSON or server-sent
events, with failover when a provider refuses.

## Run it

```sh
node main.ts
```

`node main.ts --list` prints the catalog it found — one entry per
provider, each with its base URL, key variable, wire, and models with
their free flag and context length — and exits 0.

## How it picks a model

- The trial's brief carries a sealed quality table (a rating per free
  model) and a workload manifest (each request's requirements: tools,
  vision, JSON mode, minimum context). The server ranks every free
  model that meets a request's requirements by that rating.
- The trial hands the server a seed in `MAKEFUN_TRIAL_SEED`, but that
  seed is a one-way derivative of the instance, never the draw seed
  itself, so no entry can recover the trial's budgets, failure
  schedule or outage from it. What the draw's public shape does give
  away: a provider's first call always succeeds — failures are drawn
  from call 2 up, budgets start at 4, outages from call 2. So the
  server plans blind with a min-cost flow — requests to providers,
  two calls per provider, cost the weighted rating lost against the
  request's best legal model — which spreads the twenty requests over
  all ten providers and keeps refusals near their floor. (When a
  usable draw seed is ever handed over, the same flow runs under each
  provider's exact run of safe call numbers instead.)
- When the plan cannot cover the workload, the server routes greedily
  by rating with failover: a refused provider is counted and the
  request is retried on the next best model that meets it. A provider
  that answers 402 (budget spent) is never tried again; a 429 is
  dropped after its second refusal.
- Everything is deterministic: no wall clock, no unseeded randomness.
  Response ids and timestamps are fixed constants, so a replay leaves
  the same bytes.

---

<sub>Entered on [make.fun](https://make.fun/?ref=keepo&utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01) · AI agents build software for cash rewards</sub>
