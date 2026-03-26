import csv
import json
from pathlib import Path

import anthropic
from pydantic import BaseModel

from .settings import Settings


SYSTEM_PROMPT = """\
You are an expert at classifying Slack messages from a workplace setting.
You track directives, decisions, and commitments.

Directive examples:
- give me the quarterly report by the end of the week
- write an authentication api

Commitment examples:
- Sure, I'll do it
- I'll have that done by the end of the day

Decision examples:
- We decided to do the API for the mobile UI
- We won't do the moonshot project

Classify each message into exactly one of these labels:
- C1_BINDING: binding commitment (example: I'll have it done by tomorrow)
- C2_SOFT: soft commitment (example: I'll do it if I have time)
- D1_DIRECTIVE: directive / request
- P1_DECISION: decision
- N1_NONE: none of the above

Respond with a single JSON object (no markdown fences) with these fields:
  classification: one of the labels above
  confidence: high | medium | low
  owner: person responsible (for C1/C2 commitments) — display_name or empty string
  deliverable: what is to be done — short phrase or empty string
  deadline: deadline if mentioned, else empty string

Rules:
- For N1_NONE: owner, deliverable, deadline must all be empty strings.
- For P1_DECISION and D1_DIRECTIVE: owner must be empty string; fill deliverable.
- Use thread_context to inform classification when the message is a reply.
"""


class ClassificationResult(BaseModel):
    message_id: str
    classification: str
    confidence: str
    owner: str
    deliverable: str
    deadline: str


class MessageClassifier:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings()
        self.client = anthropic.Anthropic(api_key=self.settings.anthropic_api_key)

    def _build_user_prompt(self, message: dict, users: dict) -> str:
        sender = users.get(message["user_id"], {})
        sender_name = sender.get("display_name", message["user_id"])
        sender_title = sender.get("title", "")

        lines = [
            f"message_id: {message['id']}",
            f"sender: {sender_name} ({sender_title})",
            f"text: {message['text']}",
        ]

        if message.get("thread_context"):
            lines.append("thread_context:")
            for ctx in message["thread_context"]:
                ctx_user = users.get(ctx["user_id"], {})
                ctx_name = ctx_user.get("display_name", ctx["user_id"])
                lines.append(f"  {ctx_name}: {ctx['text']}")

        return "\n".join(lines)

    def classify_message(self, message: dict, users: dict) -> ClassificationResult:
        user_prompt = self._build_user_prompt(message, users)

        response = self.client.messages.create(
            model=self.settings.model,
            max_tokens=self.settings.max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )

        data = json.loads(response.content[0].text)
        return ClassificationResult(
            message_id=message["id"],
            classification=data["classification"],
            confidence=data["confidence"],
            owner=data.get("owner", ""),
            deliverable=data.get("deliverable", ""),
            deadline=data.get("deadline", ""),
        )

    def classify_file(self, input_path: Path, output_path: Path) -> list[ClassificationResult]:
        with open(input_path) as f:
            data = json.load(f)

        users = data["metadata"]["users"]
        messages = data["messages"][: self.settings.max_messages]

        results = []
        for msg in messages:
            print(f"Classifying {msg['id']}...")
            result = self.classify_message(msg, users)
            results.append(result)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["message_id", "classification", "confidence", "owner", "deliverable", "deadline"],
            )
            writer.writeheader()
            for r in results:
                writer.writerow(r.model_dump())

        print(f"Wrote {len(results)} rows to {output_path}")
        return results
