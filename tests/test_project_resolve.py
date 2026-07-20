"""Phase 3: resolution chain + session bindings."""
import threading

from tokensense.server.project_resolve import (
    DEFAULT_PROJECT_NAME,
    SessionBindings,
    infer_from_cwd,
    resolve_project_name,
)


def test_header_wins_over_everything():
    name = resolve_project_name(
        header="from-header", env={"TOKENSENSE_PROJECT": "from-env"}, cwd="/somewhere"
    )
    assert name == "from-header"


def test_env_wins_over_cwd(tmp_path):
    name = resolve_project_name(header=None, env={"TOKENSENSE_PROJECT": "from-env"}, cwd=tmp_path)
    assert name == "from-env"


def test_cwd_git_repo_name(tmp_path):
    repo = tmp_path / "my-repo"
    (repo / ".git").mkdir(parents=True)
    nested = repo / "src" / "deep"
    nested.mkdir(parents=True)
    assert infer_from_cwd(nested) == "my-repo"


def test_cwd_falls_back_to_folder_name(tmp_path):
    folder = tmp_path / "plain-folder"
    folder.mkdir()
    assert infer_from_cwd(folder) == "plain-folder"


def test_default_fallback():
    assert resolve_project_name(header=None, env={}, cwd="/") == DEFAULT_PROJECT_NAME


def test_bindings_resolve_once_then_cached():
    bindings = SessionBindings()
    calls = []

    def resolve():
        calls.append(1)
        return "proj-a"

    assert bindings.get_or_bind("s1", resolve) == "proj-a"
    assert bindings.get_or_bind("s1", lambda: "proj-b") == "proj-a"  # cached, resolver ignored
    assert len(calls) == 1


def test_bindings_release():
    bindings = SessionBindings()
    bindings.bind("s1", "proj-a")
    assert bindings.release("s1") == "proj-a"
    assert bindings.get("s1") is None


def test_concurrent_sessions_never_cross_bind():
    bindings = SessionBindings()
    errors = []

    def worker(session_id, project_id):
        for _ in range(200):
            bound = bindings.get_or_bind(session_id, lambda: project_id)
            if bound != project_id:
                errors.append((session_id, bound))

    threads = [
        threading.Thread(target=worker, args=(f"session-{i}", f"proj-{i}")) for i in range(8)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
