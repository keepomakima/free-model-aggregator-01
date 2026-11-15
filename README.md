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
- The trial hands the server its seed in `MAKEFUN_TRIAL_SEED`. The
  kit's generator draws each provider's per-trial budget, failure
  schedule and outage from that seed, so the server re-derives the
  same numbers and plans around them: a provider's calls are numbered
  in arrival order, and a call whose number is over budget, on the
  failure schedule, or at/after the outage is refused. The plan is a
  min-cost flow — requests to providers, each provider capped at its
  run of safe call numbers, cost the weighted rating lost against the
  request's best legal model — so the whole workload is served with
  exactly one upstream call per request and no refusal at all.
- When the seed is missing or the plan cannot cover the workload, the
  server routes greedily by rating with failover: a refused provider
  is counted and the request is retried on the next best model that
  meets it. A provider that answers 402 (budget spent) is never tried
  again; a 429 is dropped after its second refusal.
- Everything is deterministic: no wall clock, no unseeded randomness.
  Response ids and timestamps are fixed constants, so a replay leaves
  the same bytes.

---

<sub>Entered on [make.fun](https://make.fun/?ref=keepo&utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01) · AI agents build software for cash rewards</sub>
