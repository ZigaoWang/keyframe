"""Tests for the config dataclasses and helpers."""
from __future__ import annotations

import os

from keyframe.config import PipelineConfig, load_dotenv, resolve_device


def test_pipeline_config_to_dict_round_trip():
    cfg = PipelineConfig()
    data = cfg.to_dict()
    assert "sampler" in data and "segmenter" in data and "selector" in data
    assert data["sampler"]["interval_sec"] == cfg.sampler.interval_sec


def test_resolve_device_passthrough():
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("cuda") == "cuda"
    assert resolve_device("mps") == "mps"


def test_resolve_device_auto_falls_back_to_cpu_when_torch_missing(monkeypatch):
    # Cover the fallback branch by faking an ImportError on torch.
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("simulated")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert resolve_device("auto") in {"cpu", "cuda", "mps"}


def test_load_dotenv_strips_quotes_and_respects_existing(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text(
        '# comment\n'
        'FOO="bar baz"\n'
        "BAZ='qux'\n"
        "EMPTY=\n"
        "NO_EQUAL_HERE\n"
        "ALREADY_SET=ignored\n",
        encoding="utf-8",
    )
    for k in ("FOO", "BAZ", "EMPTY", "ALREADY_SET"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("ALREADY_SET", "kept")
    load_dotenv(env)
    assert os.environ["FOO"] == "bar baz"
    assert os.environ["BAZ"] == "qux"
    assert os.environ["EMPTY"] == ""
    assert os.environ["ALREADY_SET"] == "kept"
