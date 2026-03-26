from pathlib import Path

from src.commitment_classifier.classifier import MessageClassifier


def main():
    classifier = MessageClassifier()
    classifier.classify_file(
        input_path=Path("data/input/shadow_mode_test.json"),
        output_path=Path("data/output/classified_messages.csv"),
    )


if __name__ == "__main__":
    main()
