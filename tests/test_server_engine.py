"""Phases 3 & 4: engine session lifecycle, isolation under concurrency,
explicit-only project switching."""
import threading

from .fakes import make_engine


def _chat(engine, session, text):
    payload, retrieved = engine.prepare_payload(session, text)
    engine.record_turn(
        session, current_message=text, assistant_content=f"re: {text}", payload=payload, model="gpt-4o"
    )
    return retrieved


def test_session_binds_project_once(monkeypatch):
    engine = make_engine(monkeypatch)
    s1 = engine.get_session("s1", project_header="alpha")
    # Later turns with a different header (or changed env) must NOT retarget the session.
    monkeypatch.setenv("TOKENSENSE_PROJECT", "beta")
    s1_again = engine.get_session("s1", project_header="beta")
    assert s1_again is s1
    assert s1_again.project.name == "alpha"


def test_end_session_stores_memory_chunk(monkeypatch):
    engine = make_engine(monkeypatch)
    session = engine.get_session("s1", project_header="alpha")
    _chat(engine, session, "we decided to use pgvector")
    summary = engine.end_session("s1")
    assert summary is not None
    chunks = engine.store.memory_chunks
    assert len(chunks) == 1
    assert chunks[0].project_id == session.project.id
    assert engine.tracker.sessions == 1


def test_end_session_without_turns_stores_nothing(monkeypatch):
    engine = make_engine(monkeypatch)
    engine.get_session("s1", project_header="alpha")
    assert engine.end_session("s1") is None
    assert engine.store.memory_chunks == []


def test_retrieval_is_scoped_to_bound_project(monkeypatch):
    engine = make_engine(monkeypatch)
    a = engine.get_session("sa", project_header="alpha")
    b = engine.get_session("sb", project_header="beta")
    _chat(engine, a, "alpha fact: use port 8317")
    engine.end_session("sa")
    _chat(engine, b, "beta fact: use redis")
    engine.end_session("sb")

    a2 = engine.get_session("sa2", project_header="alpha")
    retrieved = _chat(engine, a2, "what port?")
    assert retrieved  # alpha's memory came back
    assert all("beta" not in chunk for chunk in retrieved)


def test_switch_project_ends_and_persists_old_session(monkeypatch):
    engine = make_engine(monkeypatch)
    session = engine.get_session("s1", project_header="alpha")
    old_project_id = session.project.id
    _chat(engine, session, "alpha work")

    new_session = engine.switch_project("s1", "beta")
    assert new_session.project.name == "beta"
    assert engine.bindings.get("s1") == new_session.project.id
    # Old session was summarized into its original project before the switch.
    assert [c.project_id for c in engine.store.memory_chunks] == [old_project_id]


def test_sweep_idle_ends_stale_sessions(monkeypatch):
    engine = make_engine(monkeypatch, idle_timeout_seconds=0.0)
    session = engine.get_session("s1", project_header="alpha")
    _chat(engine, session, "some work")
    session.last_active -= 1  # force past the timeout
    assert engine.sweep_idle() == ["s1"]
    assert engine.store.memory_chunks  # summarized on sweep
    assert engine.bindings.get("s1") is None


def test_end_all_flushes_every_session(monkeypatch):
    engine = make_engine(monkeypatch)
    for i, project in enumerate(["alpha", "beta"]):
        session = engine.get_session(f"s{i}", project_header=project)
        _chat(engine, session, f"{project} work")
    engine.end_all()
    assert len(engine.store.memory_chunks) == 2
    assert engine.bindings.active_sessions() == []


def test_concurrent_sessions_never_cross_contaminate(monkeypatch):
    """Phase 3 exit criterion: two sessions in different projects against one
    engine, under concurrent load, each retrieve only their own memory."""
    engine = make_engine(monkeypatch)
    # Seed memory in both projects.
    for project, fact in [("alpha", "alpha secret"), ("beta", "beta secret")]:
        s = engine.get_session(f"seed-{project}", project_header=project)
        _chat(engine, s, fact)
        engine.end_session(f"seed-{project}")

    errors = []

    def worker(project):
        for i in range(50):
            session = engine.get_session(f"{project}-{i}", project_header=project)
            retrieved = _chat(engine, session, "recall the secret")
            other = "beta" if project == "alpha" else "alpha"
            if any(other in chunk for chunk in retrieved):
                errors.append((project, retrieved))
            engine.end_session(f"{project}-{i}")

    threads = [threading.Thread(target=worker, args=(p,)) for p in ("alpha", "beta")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
