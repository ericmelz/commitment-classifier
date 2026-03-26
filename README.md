# commitment-classifier

Classifies Slack messages from a workplace setting into structured categories — commitments, directives, decisions, or noise — using Anthropic's Claude models. Outputs a CSV with extracted fields for each message.

## Classification labels

| Label | Meaning | Example |
|---|---|---|
| `C1_BINDING` | Firm commitment | "I'll have it done by tomorrow" |
| `C2_SOFT` | Soft / hedged commitment | "I'll do it if I have time" |
| `D1_DIRECTIVE` | Request or directive | "Can you write the auth API?" |
| `P1_DECISION` | Decision made | "We're going API-first, mobile is deferred to Q4" |
| `N1_NONE` | None of the above | "who wants coffee?" |

## Output format

Results are written as CSV with these columns:

| Column | Description |
|---|---|
| `message_id` | ID of the message from the input file |
| `classification` | One of the labels above |
| `confidence` | `high`, `medium`, or `low` |
| `owner` | Person responsible (commitments only) |
| `deliverable` | What is to be done (all except N1_NONE) |
| `deadline` | Deadline if mentioned, else blank |

## Setup

**Requirements:** Python 3.13+, [uv](https://docs.astral.sh/uv/)

1. Clone the repo and install dependencies:

   ```bash
   git clone https://github.com/ericmelz/commitment-classifier.git
   cd commitment-classifier
   uv sync
   ```

2. Create a `.env` file (use `.env.example` as a template):

   ```bash
   cp .env.example .env
   ```

   Then edit `.env` and set your Anthropic API key:

   ```
   ANTHROPIC_API_KEY=sk-ant-...
   ```

## Usage

Place your input JSON file at `data/input/shadow_mode_test.json` (see [Input format](#input-format) below), then run:

```bash
uv run main.py
```

Output is written to `data/output/classified_messages.csv`.

## Input format

The input JSON must follow this structure:

```json
{
  "metadata": {
    "users": {
      "U001": { "name": "Alice Lee", "display_name": "alice.lee", "title": "Senior Backend Engineer" }
    }
  },
  "messages": [
    {
      "id": "MSG001",
      "channel_id": "C001",
      "user_id": "U001",
      "text": "I'll have the API spec ready by Friday.",
      "ts": "1711900800.000100",
      "thread_ts": null,
      "thread_context": null
    }
  ]
}
```

Thread replies include a `thread_context` array of prior messages. The classifier uses this context to interpret short replies like "On it" or "yep" correctly.

## Configuration

Settings are loaded from `.env` or environment variables via `pydantic-settings`:

| Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | *(required)* | Your Anthropic API key |
| `MODEL` | `claude-sonnet-4-6` | Claude model to use |
| `MAX_TOKENS` | `1024` | Max tokens per response |

## Project structure

```
commitment-classifier/
├── data/
│   ├── input/          # Input JSON files
│   └── output/         # Generated CSV output
├── src/
│   └── commitment_classifier/
│       ├── classifier.py   # Core classification logic
│       └── settings.py     # Pydantic settings
├── main.py             # Entry point
├── pyproject.toml
└── .env.example
```
