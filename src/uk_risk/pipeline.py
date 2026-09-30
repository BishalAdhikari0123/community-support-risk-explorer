"""Command-line entry point for rebuilding portfolio artefacts."""

from pathlib import Path

from .data import make_demo_dataset, validate_dataset
from .modeling import train_model


def main() -> None:
    frame = make_demo_dataset()
    validate_dataset(frame)
    result = train_model(frame)
    output = Path("reports")
    output.mkdir(exist_ok=True)
    frame.to_csv(output / "demo_dataset.csv", index=False)
    result.importance.to_csv(output / "feature_importance.csv", index=False)
    (output / "metrics.md").write_text(
        "# Model metrics\n\n" + "\n".join(f"- **{key}:** {value:.3f}" for key, value in result.metrics.items()),
        encoding="utf-8",
    )
    print("Built demo dataset and model reports in reports/")


if __name__ == "__main__":
    main()
