"""Multi-session benchmark scenarios.

Each scenario is a sequence of sessions (each a list of user turns) followed by
a probe question asked in a fresh session. The probe checks that facts from
earlier sessions survive compression and come back through retrieval: `expect`
substrings must appear in the retrieved context injected into the probe's
payload.
"""

SCENARIOS = [
    {
        "name": "cross-session-decision-recall",
        "sessions": [
            [
                "We're building the payments service. Decision: use Stripe, not Adyen.",
                "Also decided: webhooks land on /hooks/stripe, secret rotates monthly.",
            ],
            [
                "Refactored the retry queue today, max attempts is now 7.",
            ],
        ],
        "probe": "Which payment provider did we pick, and what's the webhook path?",
        "expect": ["Stripe", "/hooks/stripe"],
    },
    {
        "name": "identifier-retention",
        "sessions": [
            [
                "The prod database is db-prod-cx41, the staging one is db-stg-9k2.",
                "Ticket PAY-1337 tracks the migration; deadline is March 3.",
            ],
        ],
        "probe": "What's the prod database identifier and the migration ticket?",
        "expect": ["db-prod-cx41", "PAY-1337"],
    },
    {
        "name": "long-session-compression",
        "sessions": [
            [f"Progress note {i}: step {i} of the rollout finished cleanly." for i in range(1, 13)]
            + ["Final note: rollout complete, version v2.4.1 is live everywhere."],
        ],
        "probe": "Which version is live?",
        "expect": ["v2.4.1"],
    },
]
