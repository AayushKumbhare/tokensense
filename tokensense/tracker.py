"""Token usage + CO2 tracking (see project doc: Token + CO2 Tracker / Methodology)."""
from __future__ import annotations

# kWh per 1000 tokens — conservative end of the public 0.001-0.01 range for LLM inference.
KWH_PER_1K_TOKENS = 0.001
# US average grid carbon intensity, kg CO2 per kWh.
GRID_CO2_PER_KWH = 0.4


class Tracker:
    def __init__(self):
        self.tokens_sent = 0
        self.tokens_baseline = 0
        self.sessions = 0

    def log_call(self, actual_tokens: int, baseline_tokens: int) -> None:
        self.tokens_sent += actual_tokens
        self.tokens_baseline += baseline_tokens

    def log_session_end(self) -> None:
        self.sessions += 1

    @property
    def tokens_saved(self) -> int:
        return max(self.tokens_baseline - self.tokens_sent, 0)

    @property
    def co2_saved_grams(self) -> float:
        kwh_saved = (self.tokens_saved / 1000) * KWH_PER_1K_TOKENS
        return kwh_saved * GRID_CO2_PER_KWH * 1000

    def stats(self) -> dict:
        return {
            "tokens_sent": self.tokens_sent,
            "tokens_baseline": self.tokens_baseline,
            "tokens_saved": self.tokens_saved,
            "co2_saved_grams": round(self.co2_saved_grams, 4),
            "sessions": self.sessions,
        }
