# tokencount — CLI Token Counter & Cost Estimator for LLM Prompts

A command-line tool that counts tokens and estimates API costs for popular LLM models.

## Requirements

- Python 3.8+
- [tiktoken](https://github.com/openai/tiktoken)

## Installation

```bash
pip install tiktoken
```

No other dependencies required — single file, ready to run.

## Usage

### From a file
```bash
python tokencount.py prompt.txt
```

### From stdin (pipe)
```bash
cat prompt.txt | python tokencount.py
echo "Hello world" | python tokencount.py
```

### From clipboard
```bash
python tokencount.py --clip
```

### Filter to a specific model
```bash
python tokencount.py prompt.txt --model gpt-4o
```

### Custom output token estimate
```bash
python tokencount.py prompt.txt --output-tokens 1000
```

### JSON output
```bash
python tokencount.py prompt.txt --json
```

## Example Output

```
Input: 342 tokens

Model                     Input   Output    Est. Cost
──────────────────────────────────────────────────────
gpt-4o                      342      500       $0.0059
gpt-4o-mini                 342      500       $0.0004
claude-3-5-sonnet            342      500       $0.0085
gemini-1.5-pro               342      500       $0.0029
```

### JSON Output

```json
{
  "input_text_length": 1420,
  "models": [
    {
      "model": "gpt-4o",
      "input_tokens": 342,
      "output_tokens": 500,
      "cost": 0.005855
    }
  ]
}
```

## Supported Models & Pricing (2025 Q1)

| Model | Input ($/1M tokens) | Output ($/1M tokens) | Encoding |
|-------|---------------------|----------------------|----------|
| gpt-4o | $2.50 | $10.00 | o200k_base |
| gpt-4o-mini | $0.15 | $0.60 | o200k_base |
| claude-3-5-sonnet | $3.00 | $15.00 | cl100k_base* |
| gemini-1.5-pro | $1.25 | $5.00 | cl100k_base* |

*Claude and Gemini use cl100k_base as an approximation since their native tokenizers aren't available in tiktoken.

## Running Tests

```bash
python test_tokencount.py
```

## License

MIT
