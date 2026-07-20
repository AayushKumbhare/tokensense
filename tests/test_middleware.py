from tokensense.middleware import SlidingWindow, build_payload
from tokensense.summarizers.base import BaseSummarizer


class FakeSummarizer(BaseSummarizer):
    def __init__(self):
        self.calls: list[str] = []

    def _complete(self, prompt: str) -> str:
        self.calls.append(prompt)
        return f"summary#{len(self.calls)}"


def test_window_keeps_last_n_verbatim():
    window = SlidingWindow(FakeSummarizer(), window_size=2)
    for i in range(4):
        window.add_turn({"role": "user", "content": f"turn {i}"})
    assert [t["content"] for t in window.verbatim_turns] == ["turn 2", "turn 3"]


def test_window_summarizes_evicted_turns():
    window = SlidingWindow(FakeSummarizer(), window_size=1)
    window.add_turn({"role": "user", "content": "a"})
    window.add_turn({"role": "user", "content": "b"})
    assert window.rolling_summary == "summary#1"


def test_finalize_anchors_to_verbatim_tail():
    summarizer = FakeSummarizer()
    window = SlidingWindow(summarizer, window_size=2)
    for i in range(3):
        window.add_turn({"role": "user", "content": f"turn {i}"})
    final = window.finalize()
    assert final == f"summary#{len(summarizer.calls)}"
    assert len(summarizer.calls) == 2  # one eviction call + one finalize call


def test_build_payload_orders_context_then_window_then_message():
    payload = build_payload(
        ["past chunk"], "rolling summary", [{"role": "user", "content": "hi"}], "now"
    )
    contents = [m["content"] for m in payload]
    assert contents[0].endswith("past chunk")
    assert contents[1].endswith("rolling summary")
    assert contents[-1] == "now"


def test_build_payload_omits_empty_sections():
    payload = build_payload([], None, [], "hi")
    assert payload == [{"role": "user", "content": "hi"}]
