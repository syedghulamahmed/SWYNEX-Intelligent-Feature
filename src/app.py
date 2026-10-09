from __future__ import annotations

import argparse
import sys

from feedback_model import classify, train_and_evaluate


def main() -> int:
    parser = argparse.ArgumentParser(description="Intelligent student-feedback triage.")
    parser.add_argument("--text", help="Feedback text to classify.")
    parser.add_argument("--threshold", type=float, default=0.45,
                        help="Flag predictions below this confidence for review (0-1).")
    args = parser.parse_args()
    if not 0 <= args.threshold <= 1:
        print("Error: --threshold must be between 0 and 1.", file=sys.stderr)
        return 2
    try:
        model, _ = train_and_evaluate()
        examples = [args.text] if args.text is not None else [
            "The mentor session was useful and well organized.",
            "Could you tell me when the next task is due?",
            "Please add a calendar to the dashboard.",
            "My submission disappeared and support has not replied.",
            "Blue quickly the window although tomorrow.",
            "",
        ]
        print("\nPredictions")
        for item in examples:
            try:
                result = classify(model, item, args.threshold)
                print(f"Input: {result['input']}")
                print(f"Category: {result['category']}")
                print(f"Confidence: {result['confidence']:.3f}")
                print(f"Human review: {'YES' if result['needs_human_review'] else 'no'}")
                if result["review_reason"]:
                    print(f"Reason: {result['review_reason']}")
                print()
            except ValueError as exc:
                print(f"Input error: {exc}\n", file=sys.stderr)
        return 0
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"Unable to run classifier: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Unexpected model error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
