from src.commitment_classifier.classifier import MessageClassifier


def main():
    classifier = MessageClassifier()
    result = classifier.classify(
        message="I will finish the report by Friday.",
        labels=["commitment", "request", "question", "statement"],
    )
    print(f"Label:      {result.label}")
    print(f"Confidence: {result.confidence}")
    print(f"Reasoning:  {result.reasoning}")


if __name__ == "__main__":
    main()
