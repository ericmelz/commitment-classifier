from pydantic import BaseModel
import anthropic

from .settings import Settings


class ClassificationResult(BaseModel):
    label: str
    confidence: str
    reasoning: str


class MessageClassifier:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings()
        self.client = anthropic.Anthropic(api_key=self.settings.anthropic_api_key)

    def classify(self, message: str, labels: list[str]) -> ClassificationResult:
        labels_str = ", ".join(f'"{label}"' for label in labels)
        prompt = (
            f"You are an expert at classifying slack messages generated in a\n"
            f"workplace setting.\n"
            f"You will be tracking directives, decisions, and commitments.\n"
            f"Directive examples:\n"
            f"- give me the quarterly report by the end of the week\n"
            f"- write an authentication api\n"
            f"Commitment examples:\n"
            f"- Sure, I'll do it\n"
            f"- I'll have that done by the end of the day\n"
            f"Decision examples:\n"
            f"- We decided to do the API for the mobile UI\n"
            f"- We won't do the moonshot project\n"
            f"Classify the following message into one of these labels: \n"
            f"- C1 binding commitments (example: I'll have it done by tomorrow)\n"
            f"- C2 soft commitment.  (example: I'll do it if I have time)\n"
            f"- P1 decision\n"
            f"- N1 none of the above\n"
            "Respond with JSON matching this schema:\n"
            '{"label": "<chosen label>", "confidence": "high|medium|low", "reasoning": "<brief explanation>"}'
        )

        response = self.client.messages.create(
            model=self.settings.model,
            max_tokens=self.settings.max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )

        import json
        text = response.content[0].text
        data = json.loads(text)
        return ClassificationResult(**data)
