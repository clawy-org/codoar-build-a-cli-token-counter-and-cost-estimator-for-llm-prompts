#!/usr/bin/env python3
"""CLI token counter and cost estimator for LLM prompts.

Counts tokens using tiktoken and estimates API costs for popular LLM models.
Reads input from stdin, a file argument, or clipboard (--clip).
"""

import argparse
import json
import sys

try:
    import tiktoken
except ImportError:
    print("Error: tiktoken is required. Install with: pip install tiktoken", file=sys.stderr)
    sys.exit(1)

# Prices per token as of 2025 Q1 (USD)
# Format: (input_price_per_1M_tokens, output_price_per_1M_tokens)
MODEL_PRICES = {
    "gpt-4o":            (2.50, 10.00),
    "gpt-4o-mini":       (0.15, 0.60),
    "claude-3-5-sonnet": (3.00, 15.00),
    "gemini-1.5-pro":    (1.25, 5.00),
}

# tiktoken encoding mapping
# Claude and Gemini don't have official tiktoken encodings,
# so we use cl100k_base as a reasonable approximation
MODEL_ENCODINGS = {
    "gpt-4o":            "o200k_base",
    "gpt-4o-mini":       "o200k_base",
    "claude-3-5-sonnet": "cl100k_base",
    "gemini-1.5-pro":    "cl100k_base",
}

DEFAULT_OUTPUT_TOKENS = 500


def count_tokens(text: str, encoding_name: str) -> int:
    """Count tokens in text using the specified encoding."""
    enc = tiktoken.get_encoding(encoding_name)
    return len(enc.encode(text))


def estimate_cost(input_tokens: int, output_tokens: int, model: str) -> float:
    """Estimate cost in USD for given token counts and model."""
    input_price, output_price = MODEL_PRICES[model]
    return (input_tokens * input_price + output_tokens * output_price) / 1_000_000


def get_input_text(args) -> str:
    """Get input text from file, clipboard, or stdin."""
    if args.clip:
        try:
            import subprocess
            # Try xclip, xsel, pbpaste in order
            for cmd in [["xclip", "-selection", "clipboard", "-o"], ["xsel", "--clipboard", "--output"], ["pbpaste"]]:
                try:
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
                    if result.returncode == 0:
                        return result.stdout
                except FileNotFoundError:
                    continue
            print("Error: No clipboard tool found (tried xclip, xsel, pbpaste)", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"Error reading clipboard: {e}", file=sys.stderr)
            sys.exit(1)

    if args.file:
        try:
            with open(args.file, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            print(f"Error: File not found: {args.file}", file=sys.stderr)
            sys.exit(1)
        except IOError as e:
            print(f"Error reading file: {e}", file=sys.stderr)
            sys.exit(1)

    # Read from stdin
    if sys.stdin.isatty():
        print("Reading from stdin (Ctrl+D to end):", file=sys.stderr)
    return sys.stdin.read()


def format_table(results: list, input_tokens_by_model: dict, output_tokens: int) -> str:
    """Format results as a clean ASCII table."""
    lines = []

    # Show input token count (use first model's count as representative)
    first_model = list(input_tokens_by_model.keys())[0]
    lines.append(f"Input: {input_tokens_by_model[first_model]} tokens")
    lines.append("")

    header = f"{'Model':<24} {'Input':>8} {'Output':>8} {'Est. Cost':>12}"
    separator = "─" * len(header)
    lines.append(header)
    lines.append(separator)

    for r in results:
        cost_str = f"${r['cost']:.4f}"
        lines.append(f"{r['model']:<24} {r['input_tokens']:>8} {output_tokens:>8} {cost_str:>12}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Count tokens and estimate API costs for LLM prompts.",
        epilog="Examples:\n"
               "  cat prompt.txt | python tokencount.py\n"
               "  python tokencount.py myfile.txt\n"
               "  python tokencount.py --clip --model gpt-4o\n"
               "  python tokencount.py myfile.txt --output-tokens 1000 --json",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("file", nargs="?", help="Input file to read")
    parser.add_argument("--clip", action="store_true", help="Read input from clipboard")
    parser.add_argument("--model", choices=list(MODEL_PRICES.keys()),
                        help="Filter to a specific model")
    parser.add_argument("--output-tokens", type=int, default=DEFAULT_OUTPUT_TOKENS,
                        help=f"Estimated output tokens (default: {DEFAULT_OUTPUT_TOKENS})")
    parser.add_argument("--json", action="store_true", dest="json_output",
                        help="Output in JSON format")

    args = parser.parse_args()

    text = get_input_text(args)

    if not text:
        print("Input: 0 tokens")
        print("\nNo input provided.")
        return

    models = [args.model] if args.model else list(MODEL_PRICES.keys())
    output_tokens = args.output_tokens

    results = []
    input_tokens_by_model = {}

    for model in models:
        encoding_name = MODEL_ENCODINGS[model]
        input_tokens = count_tokens(text, encoding_name)
        cost = estimate_cost(input_tokens, output_tokens, model)
        input_tokens_by_model[model] = input_tokens
        results.append({
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost": round(cost, 6),
        })

    if args.json_output:
        output = {
            "input_text_length": len(text),
            "models": results,
        }
        print(json.dumps(output, indent=2))
    else:
        print(format_table(results, input_tokens_by_model, output_tokens))


if __name__ == "__main__":
    main()
