"""CUB to NABirds few-shot fine-grained classification experiments."""


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Few-shot fine-grained experiments")
    parser.add_argument("command", choices=["train", "evaluate", "summarize", "select"])
    args, remaining = parser.parse_known_args()
    if args.command == "train":
        from pattern_recognition_coursework.engine.experiment import main as run
    elif args.command == "evaluate":
        from pattern_recognition_coursework.engine.evaluate import main as run
    elif args.command == "summarize":
        from pattern_recognition_coursework.analysis.summarize import main as run
    else:
        from pattern_recognition_coursework.analysis.select import main as run
    run(remaining)
