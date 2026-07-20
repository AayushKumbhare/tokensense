from tokensense.tracker import Tracker


def test_tokens_saved_and_co2():
    tracker = Tracker()
    tracker.log_call(actual_tokens=100, baseline_tokens=1000)
    stats = tracker.stats()
    assert stats["tokens_sent"] == 100
    assert stats["tokens_baseline"] == 1000
    assert stats["tokens_saved"] == 900
    assert stats["co2_saved_grams"] > 0


def test_tokens_saved_floors_at_zero():
    tracker = Tracker()
    tracker.log_call(actual_tokens=500, baseline_tokens=100)
    assert tracker.stats()["tokens_saved"] == 0


def test_sessions_increments_on_log_session_end():
    tracker = Tracker()
    tracker.log_session_end()
    tracker.log_session_end()
    assert tracker.stats()["sessions"] == 2
