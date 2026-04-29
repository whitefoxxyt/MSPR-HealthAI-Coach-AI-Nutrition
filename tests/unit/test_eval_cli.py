from __future__ import annotations

import pytest

from scripts.eval_metrics import build_parser


def test_cli_parses_classifier_subcommand() -> None:
    parser = build_parser()
    args = parser.parse_args(
        ["classifier", "--n-food101", "100", "--seed", "42", "--output-dir", "docs/"]
    )

    assert args.command == "classifier"
    assert args.n_food101 == 100
    assert args.seed == 42
    assert str(args.output_dir).rstrip("/").endswith("docs")


def test_cli_parses_llm_subcommand() -> None:
    parser = build_parser()
    args = parser.parse_args(
        ["llm", "--n-generations", "100", "--seed", "7", "--output-dir", "docs/"]
    )

    assert args.command == "llm"
    assert args.n_generations == 100
    assert args.seed == 7


def test_cli_requires_subcommand() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])
