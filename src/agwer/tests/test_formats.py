"""Format adapters: every supported log shape yields the same records."""

import json

import pytest

from agwer.formats import load_records

REC = {"reference": "a b c", "corrected": "a b d", "nbest": ["a x c", "a b c"]}


def _write(tmp_path, rows, name="in.jsonl"):
    p = tmp_path / name
    p.write_text("\n".join(json.dumps(r) for r in rows))
    return str(p)


def test_native_jsonl(tmp_path):
    assert load_records(_write(tmp_path, [REC])) == [REC]


def test_openai_string_content(tmp_path):
    row = {"reference": REC["reference"], "nbest": REC["nbest"],
           "messages": [{"role": "system", "content": "correct the asr"},
                        {"role": "user", "content": "hypotheses ..."},
                        {"role": "assistant", "content": "a b d"}]}
    assert load_records(_write(tmp_path, [row]))[0] == REC
    # explicit flag agrees with auto-sniffing
    assert load_records(_write(tmp_path, [row]), "openai")[0] == REC


def test_openai_content_parts_and_structured(tmp_path):
    row = {"reference": REC["reference"], "onebest": "a x c",
           "messages": [{"role": "assistant", "content": [
               {"type": "text", "text": json.dumps({"corrected": "a b d"})}]}]}
    rec = load_records(_write(tmp_path, [row]))[0]
    assert rec["corrected"] == "a b d" and rec["onebest"] == "a x c"


def test_anthropic_response_blocks(tmp_path):
    row = {"reference": REC["reference"], "nbest": REC["nbest"],
           "response": {"role": "assistant", "stop_reason": "end_turn",
                        "content": [{"type": "text", "text": "a b d"}]}}
    assert load_records(_write(tmp_path, [row]))[0] == REC


def test_anthropic_structured_output(tmp_path):
    row = {"reference": REC["reference"], "nbest": REC["nbest"],
           "response": {"role": "assistant", "content": [
               {"type": "text",
                "text": json.dumps({"corrected": "a b d", "confidence": 0.9})}]}}
    assert load_records(_write(tmp_path, [row]), "anthropic")[0] == REC


def test_sharegpt(tmp_path):
    row = {"reference": REC["reference"], "nbest": REC["nbest"],
           "conversations": [{"from": "system", "value": "correct the asr"},
                             {"from": "human", "value": "hypotheses ..."},
                             {"from": "gpt", "value": "a b d"}]}
    assert load_records(_write(tmp_path, [row]))[0] == REC
    assert load_records(_write(tmp_path, [row]), "sharegpt")[0] == REC


def test_last_assistant_turn_wins(tmp_path):
    row = {"reference": REC["reference"], "nbest": REC["nbest"],
           "messages": [{"role": "assistant", "content": "draft"},
                        {"role": "user", "content": "try again"},
                        {"role": "assistant", "content": "a b d"}]}
    assert load_records(_write(tmp_path, [row]))[0]["corrected"] == "a b d"


def test_errors(tmp_path):
    with pytest.raises(SystemExit, match="missing 'reference'"):
        load_records(_write(tmp_path, [{"corrected": "x", "onebest": "x"}]))
    with pytest.raises(SystemExit, match="no assistant output"):
        load_records(_write(tmp_path, [
            {"reference": "a", "nbest": ["a"],
             "messages": [{"role": "user", "content": "hi"}]}]))
    with pytest.raises(SystemExit, match="'nbest'"):
        load_records(_write(tmp_path, [{"reference": "a", "corrected": "a"}]))
    with pytest.raises(SystemExit, match="unknown format"):
        load_records(_write(tmp_path, [REC]), "csv")


def test_parquet_roundtrip(tmp_path):
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq
    path = str(tmp_path / "in.parquet")
    pq.write_table(pa.table({k: [v] for k, v in REC.items()}), path)
    assert load_records(path) == [REC]              # auto by extension
    assert load_records(path, "parquet") == [REC]
