"""Offline contracts for the opt-in Unity Catalog OTel destination."""

import json

import pytest

from codex_hydrogym.genai import tracing


def test_destination_is_off_by_default_and_is_a_noop(monkeypatch):
    monkeypatch.delenv(tracing.UC_TRACING_FLAG, raising=False)
    called = []
    monkeypatch.setattr(tracing, "importlib", type("Imports", (), {"import_module": lambda *_: called.append(1)})())

    assert tracing.configure_uc_trace_destination() is None
    assert called == []


def test_destination_uses_configured_catalog_and_schema(monkeypatch):
    captured = []

    class FakeLocation:
        def __init__(self, catalog, schema):
            self.catalog_name = catalog
            self.schema_name = schema

    fake_destination = type(
        "Destination",
        (),
        {"UCSchemaLocation": FakeLocation, "set_destination": captured.append},
    )()
    real_import = tracing.importlib.import_module
    monkeypatch.setattr(
        tracing.importlib,
        "import_module",
        lambda name: fake_destination if name == "mlflow.tracing.destination" else real_import(name),
    )
    env = {
        tracing.UC_TRACING_FLAG: "1",
        tracing.UC_CATALOG_ENV: "catalog_x",
        tracing.UC_SCHEMA_ENV: "schema_y",
        tracing.UC_WAREHOUSE_ENV: "warehouse_z",
    }

    location = tracing.configure_uc_trace_destination(environ=env)

    assert location.catalog_name == "catalog_x"
    assert location.schema_name == "schema_y"
    assert captured == [location]


def test_enabled_destination_fails_loudly_without_warehouse():
    env = {tracing.UC_TRACING_FLAG: "true"}

    with pytest.raises(RuntimeError, match=tracing.UC_WAREHOUSE_ENV):
        tracing.configure_uc_trace_destination(environ=env)


def test_preflight_reports_missing_configuration(capsys):
    result = tracing.preflight_uc_trace_destination(environ={tracing.UC_TRACING_FLAG: "on"})
    report = result.report()

    assert not result.ready
    assert tracing.UC_WAREHOUSE_ENV in result.missing
    assert tracing.DEFAULT_UC_CATALOG in result.as_dict()["catalog_name"]
    assert json.loads(report)["missing"] == [tracing.UC_WAREHOUSE_ENV]
