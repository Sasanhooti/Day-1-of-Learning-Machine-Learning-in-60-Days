"""Apply a manually parameterized linear function; this is not training."""

import numpy as np


def predict_linear(features, weights, bias: float) -> float:
    """Calculate dot(features, weights) + bias after checking dimensions."""
    x = np.asarray(features, dtype=float)
    w = np.asarray(weights, dtype=float)

    if x.ndim != 1 or w.ndim != 1:
        raise ValueError("features and weights must both be one-dimensional")
    if x.shape != w.shape:
        raise ValueError(
            f"Feature/weight length mismatch: {x.size} features vs {w.size} weights"
        )

    return float(np.dot(x, w) + bias)


def main() -> None:
    # Example feature order: Area, Rooms, Age, Distance
    example_house = [100, 3, 5, 4.5]

    # Chosen manually for demonstration only; these are not learned parameters.
    weights = [45, 250, -30, -70]
    bias = 1000

    result = predict_linear(example_house, weights, bias)
    print(f"Input features: {example_house}")
    print(f"Illustrative prediction: {result:.2f}")
    print("No training has taken place; the parameters were set manually.")


if __name__ == "__main__":
    main()
