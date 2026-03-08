"""Tests for arithmetic task generation and parsing."""

from csr_llm.task import (
    generate_test_set,
    parse_generated_output,
    summarize_parsed,
)


def test_generate_test_set():
    problems = generate_test_set(100, ["+", "-"], (0, 9), seed=42)
    assert len(problems) == 100
    for p in problems:
        assert p.operation in ["+", "-"]
        assert p.answer == str(eval(p.expression))


def test_generate_deterministic():
    a = generate_test_set(50, ["+"], (0, 9), seed=123)
    b = generate_test_set(50, ["+"], (0, 9), seed=123)
    assert [p.full for p in a] == [p.full for p in b]


def test_parse_valid():
    raw = "3+5=8\n7-2=5\n4+1=5\n"
    parsed = parse_generated_output(raw)
    assert len(parsed) == 3
    assert all(e.is_valid for e in parsed)
    assert all(e.is_correct for e in parsed)


def test_parse_incorrect():
    raw = "3+5=9\n"
    parsed = parse_generated_output(raw)
    assert len(parsed) == 1
    assert parsed[0].is_valid
    assert not parsed[0].is_correct
    assert parsed[0].correct_answer == "8"


def test_parse_invalid():
    raw = "hello world\n3+5=8\ngarbage\n"
    parsed = parse_generated_output(raw)
    assert len(parsed) == 3
    assert not parsed[0].is_valid
    assert parsed[1].is_valid and parsed[1].is_correct
    assert not parsed[2].is_valid


def test_summarize():
    raw = "3+5=8\n7-2=5\n4+1=99\nbad line\n"
    parsed = parse_generated_output(raw)
    s = summarize_parsed(parsed)
    assert s["total"] == 4
    assert s["valid"] == 3
    assert s["correct"] == 2
    assert s["incorrect"] == 1
    assert s["unparseable"] == 1


def test_subtraction_no_negative():
    """Test that generated subtraction problems don't produce negative answers."""
    problems = generate_test_set(1000, ["-"], (0, 9), seed=42)
    for p in problems:
        assert int(p.answer) >= 0, f"Negative result: {p.full}"
