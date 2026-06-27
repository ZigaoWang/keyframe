"""Tests for the argparse surface."""
from __future__ import annotations

import pytest

from keyframe.cli import build_parser, main


def test_parser_run_subcommand_defaults():
    parser = build_parser()
    args = parser.parse_args(["run", "video.mp4"])
    assert args.command == "run"
    assert args.source == "video.mp4"
    assert args.sample_interval == 1.0
    assert args.sim_threshold == 0.96
    assert args.embedder == "yolov8n"
    assert args.no_caption is False


def test_parser_list_embedders():
    parser = build_parser()
    args = parser.parse_args(["list-embedders"])
    assert args.command == "list-embedders"


def test_parser_benchmark_defaults():
    parser = build_parser()
    args = parser.parse_args(["benchmark", "video.mp4"])
    assert args.command == "benchmark"
    assert args.embedders == ["yolov8n", "phash", "hsv"]


def test_main_list_embedders_prints(capsys):
    rc = main(["list-embedders"])
    assert rc == 0
    out = capsys.readouterr().out.splitlines()
    assert "phash" in out
    assert "hsv" in out


def test_parser_requires_subcommand():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])
