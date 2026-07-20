"""Compare local Ollama models as TokenSense summarizers: latency + fact retention.

Usage: python benchmarks/summarizer_models.py [model ...]
Defaults to the three candidates from the task sheet. Each model gets one
untimed warmup call (model load), then timed summarize calls over a fixed
mock dev session, scored against a checklist of facts a good structured
summary must retain.
"""
from __future__ import annotations

import sys
import time

from tokensense.summarizers.ollama import OllamaSummarizer

DEFAULT_MODELS = ["phi3:mini", "qwen2.5:3b", "llama3.2:3b"]

SESSION_TURNS = [
    {"role": "user", "content": "Let's finish the auth flow today. We settled on JWT last time, right?"},
    {"role": "assistant", "content": "Yes - JWT access tokens with a 15 minute expiry, refresh handled at /token/refresh."},
    {"role": "user", "content": "OK. Passwords should be hashed with bcrypt, cost factor 12, in login.py."},
    {"role": "assistant", "content": "Done: login.py now hashes with bcrypt cost 12 and issues the JWT pair on success."},
    {"role": "user", "content": "One thing we haven't decided: rate limiting on the login endpoint."},
    {"role": "assistant", "content": "Agreed, rate limiting is still open - I'll note it as unresolved for next session."},
]

# Facts a faithful structured summary must carry (case-insensitive substring).
FACT_CHECKLIST = ["jwt", "15", "/token/refresh", "bcrypt", "12", "login.py", "rate limit"]

RUNS_PER_MODEL = 3


def score(summary: str) -> list[str]:
    lowered = summary.lower()
    return [f for f in FACT_CHECKLIST if f not in lowered]


def bench(model: str) -> dict:
    summarizer = OllamaSummarizer(model=model)
    summarizer.summarize(None, SESSION_TURNS[:2])  # warmup: loads the model, untimed

    durations, missing_per_run = [], []
    for _ in range(RUNS_PER_MODEL):
        start = time.perf_counter()
        summary = summarizer.summarize(None, SESSION_TURNS)
        durations.append(time.perf_counter() - start)
        missing_per_run.append(score(summary))

    worst_missing = max(missing_per_run, key=len)
    return {
        "model": model,
        "avg_s": sum(durations) / len(durations),
        "max_s": max(durations),
        "facts": f"{len(FACT_CHECKLIST) - len(worst_missing)}/{len(FACT_CHECKLIST)}",
        "missing": ", ".join(worst_missing) or "-",
    }


def main() -> None:
    models = sys.argv[1:] or DEFAULT_MODELS
    print(f"{RUNS_PER_MODEL} timed runs per model, output capped at summarizer default\n")
    header = f"{'model':<14} {'avg s':>7} {'max s':>7} {'facts':>7}  worst-run missing"
    print(header)
    print("-" * len(header))
    for model in models:
        r = bench(model)
        print(f"{r['model']:<14} {r['avg_s']:>7.1f} {r['max_s']:>7.1f} {r['facts']:>7}  {r['missing']}")


if __name__ == "__main__":
    main()
