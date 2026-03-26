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
            f"Classify the following message into one of these labels: {labels_str}.\n\n"
            f"Message: {message}\n\n"
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
