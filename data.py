"""Generate a reproducible synthetic dataset for learning purposes."""

import numpy as np
import pandas as pd


def create_dataset(n_samples: int = 100, random_seed: int = 42) -> pd.DataFrame:
    """Return a synthetic house dataset.

    The generated values are illustrative, not real estate-market data.
    """
    if n_samples < 1:
        raise ValueError("n_samples must be at least 1")

    rng = np.random.default_rng(random_seed)

    area = rng.integers(50, 221, size=n_samples)
    rooms = np.clip(np.rint(area / 35), 1, 6).astype(int)
    age = rng.integers(0, 31, size=n_samples)
    distance = np.round(rng.uniform(1, 20, size=n_samples), 2)
    noise = rng.normal(0, 500, size=n_samples)

    price = (
        area * 45
        + rooms * 250
        - age * 30
        - distance * 70
        + 1000
        + noise
    )
    price = np.maximum(price, 500)

    return pd.DataFrame(
        {
            "Area": area,
            "Rooms": rooms,
            "Age": age,
            "Distance": distance,
            "Price": np.round(price, 2),
        }
    )


if __name__ == "__main__":
    print(create_dataset().head())
