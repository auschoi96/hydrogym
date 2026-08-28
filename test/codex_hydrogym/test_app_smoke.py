"""Render the Streamlit app in its safe unconfigured state."""

from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_streamlit_app_renders_without_workspace_configuration(monkeypatch):
    for name in (
        "MLFLOW_EXPERIMENT_ID",
        "DATABRICKS_HOST",
        "CODEX_HYDROGYM_REVIEWER",
    ):
        monkeypatch.delenv(name, raising=False)
    app_path = Path(__file__).parents[2] / "codex_hydrogym" / "app" / "app.py"
    app = AppTest.from_file(str(app_path), default_timeout=20).run()

    assert not app.exception
    assert any("Domain-expert feedback" in item.value for item in app.subheader)
    assert any("Direct Unity AI Gateway model lab" in item.value for item in app.subheader)
    assert any("not attached" in item.value for item in app.code)
