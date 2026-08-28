"""Packaging contracts for the codex_hydrogym JAX/PPO path."""

from pathlib import Path
import tomllib


def test_jax_extra_contains_every_dependency_required_by_the_ppo_example():
    pyproject = tomllib.loads(Path("pyproject.toml").read_text())
    dependencies = pyproject["tool"]["poetry"]["dependencies"]
    jax_extra = set(pyproject["tool"]["poetry"]["extras"]["jax"])
    required = {
        "jax",
        "jaxlib",
        "chex",
        "navix",
        "gymnax",
        "tree-math",
        "flax",
        "optax",
        "distrax",
        "matplotlib",
    }

    assert required <= jax_extra
    for package in required:
        assert dependencies[package]["optional"] is True


def test_genai_extra_contains_direct_gateway_memalign_and_gepa_dependencies():
    pyproject = tomllib.loads(Path("pyproject.toml").read_text())
    dependencies = pyproject["tool"]["poetry"]["dependencies"]
    genai_extra = set(pyproject["tool"]["poetry"]["extras"]["codex_hydrogym"])

    assert {"mlflow", "openai", "dspy", "gepa"} <= genai_extra
    assert all(dependencies[package]["optional"] is True for package in genai_extra)
    packaged_modules = {item["include"] for item in pyproject["tool"]["poetry"]["packages"]}
    assert {"hydrogym", "codex_hydrogym"} <= packaged_modules
    scripts = pyproject["tool"]["poetry"]["scripts"]
    assert {"codex-hydrogym-train", "codex-hydrogym-genai", "codex-hydrogym-model"} <= set(scripts)
