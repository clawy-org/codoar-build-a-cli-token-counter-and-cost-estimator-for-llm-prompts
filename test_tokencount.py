#!/usr/bin/env python3
"""Tests for tokencount.py"""

import json
import subprocess
import sys
import os
import tempfile

SCRIPT = os.path.join(os.path.dirname(__file__), "tokencount.py")


def run_tokencount(input_text="", args=None):
    """Helper to run tokencount.py with given input and args."""
    cmd = [sys.executable, SCRIPT] + (args or [])
    result = subprocess.run(cmd, input=input_text, capture_output=True, text=True, timeout=30)
    return result


def test_basic_token_counting():
    """Test that token counting works for a simple input."""
    result = run_tokencount("Hello, world!")
    assert result.returncode == 0
    output = result.stdout
    assert "Input:" in output
    assert "gpt-4o" in output
    assert "gpt-4o-mini" in output
    assert "claude-3-5-sonnet" in output
    assert "gemini-1.5-pro" in output
    print("✓ basic_token_counting")


def test_token_count_accuracy():
    """Test token count accuracy against known tiktoken results."""
    import tiktoken

    text = "The quick brown fox jumps over the lazy dog."

    # Verify gpt-4o encoding
    enc_o200k = tiktoken.get_encoding("o200k_base")
    expected_o200k = len(enc_o200k.encode(text))

    enc_cl100k = tiktoken.get_encoding("cl100k_base")
    expected_cl100k = len(enc_cl100k.encode(text))

    result = run_tokencount(text, ["--json"])
    assert result.returncode == 0
    data = json.loads(result.stdout)

    for model_data in data["models"]:
        if model_data["model"] in ("gpt-4o", "gpt-4o-mini"):
            assert model_data["input_tokens"] == expected_o200k, \
                f"{model_data['model']}: expected {expected_o200k}, got {model_data['input_tokens']}"
        else:
            assert model_data["input_tokens"] == expected_cl100k, \
                f"{model_data['model']}: expected {expected_cl100k}, got {model_data['input_tokens']}"
    print("✓ token_count_accuracy")


def test_cost_math():
    """Test cost calculation correctness."""
    text = "test"
    result = run_tokencount(text, ["--json", "--model", "gpt-4o", "--output-tokens", "0"])
    assert result.returncode == 0
    data = json.loads(result.stdout)
    model_data = data["models"][0]

    # gpt-4o: $2.50 per 1M input tokens
    expected_cost = model_data["input_tokens"] * 2.50 / 1_000_000
    assert abs(model_data["cost"] - expected_cost) < 0.000001, \
        f"Cost mismatch: expected {expected_cost}, got {model_data['cost']}"
    print("✓ cost_math")


def test_cost_with_output_tokens():
    """Test cost includes output token pricing."""
    text = "test"
    output_tokens = 1000
    result = run_tokencount(text, ["--json", "--model", "gpt-4o-mini", "--output-tokens", str(output_tokens)])
    assert result.returncode == 0
    data = json.loads(result.stdout)
    model_data = data["models"][0]

    # gpt-4o-mini: $0.15/1M input, $0.60/1M output
    input_cost = model_data["input_tokens"] * 0.15 / 1_000_000
    output_cost = output_tokens * 0.60 / 1_000_000
    expected = round(input_cost + output_cost, 6)
    assert abs(model_data["cost"] - expected) < 0.000001
    print("✓ cost_with_output_tokens")


def test_single_model_filter():
    """Test --model flag filters to one model."""
    result = run_tokencount("Hello", ["--json", "--model", "gpt-4o"])
    assert result.returncode == 0
    data = json.loads(result.stdout)
    assert len(data["models"]) == 1
    assert data["models"][0]["model"] == "gpt-4o"
    print("✓ single_model_filter")


def test_json_output():
    """Test --json produces valid JSON with expected structure."""
    result = run_tokencount("Hello, world!", ["--json"])
    assert result.returncode == 0
    data = json.loads(result.stdout)
    assert "input_text_length" in data
    assert "models" in data
    assert len(data["models"]) == 4
    for m in data["models"]:
        assert "model" in m
        assert "input_tokens" in m
        assert "output_tokens" in m
        assert "cost" in m
        assert isinstance(m["cost"], (int, float))
    print("✓ json_output")


def test_file_input():
    """Test reading from a file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("This is a test file for token counting.")
        f.flush()
        result = run_tokencount(args=[f.name, "--json"])
    os.unlink(f.name)
    assert result.returncode == 0
    data = json.loads(result.stdout)
    assert data["input_text_length"] > 0
    assert all(m["input_tokens"] > 0 for m in data["models"])
    print("✓ file_input")


def test_empty_input():
    """Test empty input is handled gracefully."""
    result = run_tokencount("")
    assert result.returncode == 0
    assert "0 tokens" in result.stdout
    print("✓ empty_input")


def test_custom_output_tokens():
    """Test --output-tokens flag changes output estimate."""
    result = run_tokencount("Hello", ["--json", "--output-tokens", "1000"])
    assert result.returncode == 0
    data = json.loads(result.stdout)
    for m in data["models"]:
        assert m["output_tokens"] == 1000
    print("✓ custom_output_tokens")


def test_table_format():
    """Test default table output format."""
    result = run_tokencount("Hello world")
    assert result.returncode == 0
    lines = result.stdout.strip().split("\n")
    assert any("Input:" in line for line in lines)
    assert any("Model" in line for line in lines)
    assert any("$" in line for line in lines)
    print("✓ table_format")


def test_stdin_pipeline():
    """Test piping works (simulates cat file | python tokencount.py)."""
    long_text = "word " * 100
    result = run_tokencount(long_text, ["--json"])
    assert result.returncode == 0
    data = json.loads(result.stdout)
    assert all(m["input_tokens"] > 50 for m in data["models"])
    print("✓ stdin_pipeline")


if __name__ == "__main__":
    tests = [
        test_basic_token_counting,
        test_token_count_accuracy,
        test_cost_math,
        test_cost_with_output_tokens,
        test_single_model_filter,
        test_json_output,
        test_file_input,
        test_empty_input,
        test_custom_output_tokens,
        test_table_format,
        test_stdin_pipeline,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__}: {e}")
            failed += 1

    print(f"\n{passed}/{passed + failed} tests passed")
    sys.exit(1 if failed else 0)
