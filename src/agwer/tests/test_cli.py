"""CLI smoke tests."""

import json

from agwer.cli import main


def _write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


ROWS = [
    {"reference": "the cat sat", "corrected": "the cat sat",
     "nbest": ["the cat sad", "the cat sat"]},
    {"reference": "hello world", "corrected": "hello world",
     "nbest": ["hello world", "hello word"]},
]


def test_cli_human_output(tmp_path, capsys):
    p = tmp_path / "data.jsonl"
    _write_jsonl(p, ROWS)
    assert main([str(p)]) == 0
    out = capsys.readouterr().out
    assert "RIR" in out and "HER" in out


def test_cli_json_output(tmp_path, capsys):
    p = tmp_path / "data.jsonl"
    _write_jsonl(p, ROWS)
    assert main([str(p), "--json", "--her-granularity", "token"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["n"] == 2
    assert payload["edits"]["granularity"] == "token"


def test_cli_onebest_only(tmp_path, capsys):
    p = tmp_path / "data.jsonl"
    _write_jsonl(p, [
        {"reference": "a b c", "corrected": "a b c", "onebest": "a x c"},
    ])
    assert main([str(p), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["rir"] is None
    assert payload["wer_corrected"] == 0.0
