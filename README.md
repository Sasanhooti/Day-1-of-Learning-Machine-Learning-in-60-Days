# Mini House ML — Week 1, Day 1 Final Project

## Goal
Build a small, well-organized Python project that introduces a tabular machine-learning problem. The project creates a **synthetic** house dataset and inspects its structure. It does not train a machine-learning model.

## Dataset
Each row represents one synthetic house. Features:
- `Area`: area in square meters
- `Rooms`: number of rooms
- `Age`: building age in years
- `Distance`: distance from the city center (illustrative units)

Target:
- `Price`: synthetic price units

These values are generated for practice and are not real market data.

## Setup
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

## Run
```bash
python main.py
```

Optional:
```bash
python analysis.py
python predict.py
```

## Project structure
```text
mini_house_ml/
├── data.py
├── analysis.py
├── predict.py
├── main.py
├── requirements.txt
└── README.md
```

## What you should learn
- How a dataset is represented in a pandas DataFrame
- How features and targets are separated
- How to inspect rows, columns, and shapes
- How a prediction function works
- Why manually choosing parameters is not the same as training

## Extension tasks
1. Add a `Price_per_sqm` column.
2. Print the first 10 rows.
3. Add validation to ensure the dataset has at least one row.
4. Explain in the README why synthetic data is useful for practice and why it cannot establish real-world house-price patterns.
