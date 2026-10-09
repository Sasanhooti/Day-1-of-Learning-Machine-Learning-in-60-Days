# Day 1 of Learning Machine Learning in 60 Days


---

# Week 1 — Day 1: Machine Learning Foundations

## 1. Learning Objectives

By the end of this lesson, you should be able to:

- Understand what Machine Learning (ML) is.
- Explain the difference between traditional programming and Machine Learning.
- Understand samples, features, targets, and datasets.
- Understand the meaning of `X`, `y`, and `y_hat`.
- Work with basic NumPy arrays.
- Understand models, parameters, predictions, and training.
- Explain the basic Machine Learning workflow.

---

## 2. What Is Machine Learning?

Machine Learning is a branch of Artificial Intelligence that allows computers to learn patterns from data and use those patterns to make predictions or decisions.

Instead of manually writing every rule, developers provide data and choose a learning algorithm that can identify useful patterns.

### Traditional Programming vs. Machine Learning

| Traditional Programming | Machine Learning |
|---|---|
| Developers write explicit rules. | Algorithms learn patterns from data. |
| Input + Rules = Output | Data + Learning Algorithm = Trained Model |
| Rules are manually updated. | Model parameters are adjusted during training. |
| Suitable for clearly defined rules. | Useful when patterns are difficult to describe manually. |

### Example: Spam Detection

In traditional programming, we might write:

```python
def is_spam(message):
    suspicious_words = ["win money", "free prize", "click here"]

    for word in suspicious_words:
        if word in message.lower():
            return True

    return False
```

This approach depends on manually defined rules.

A Machine Learning system, however, can learn patterns from thousands of messages labeled as spam or not spam.

**Important:** Machine Learning does not automatically understand information like a human. It learns patterns from the examples and objectives it receives.

---

## 3. Understanding Dataset Terminology

Before learning algorithms, you must understand how data is represented.

### 3.1 Sample

A sample is one individual example in a dataset.

For example, one house in a house-price dataset represents one sample.

### 3.2 Feature

A feature is an input variable that helps a model make a prediction.

Examples:

- `Area`: house area in square meters.
- `Rooms`: number of rooms.
- `Age`: building age.
- `Distance`: distance from the city center.

### 3.3 Target

The target is the value we want the model to predict.

For a house-price prediction problem, the target is `Price`.

### 3.4 Dataset

A dataset is a collection of samples.

Consider the following dataset:

| Area | Rooms | Age | Price |
|---:|---:|---:|---:|
| 70 | 2 | 10 | 95 |
| 95 | 3 | 5 | 135 |
| 120 | 4 | 2 | 180 |

In this example:

- Number of samples: 3
- Number of features: 3
- Number of targets: 1 target value per sample
- Features: `Area`, `Rooms`, `Age`
- Target: `Price`

The numbers above are illustrative and do not represent real estate market data.

---

## 4. Understanding X and y

In supervised Machine Learning, we commonly use the following notation:

- `X`: input features.
- `y`: actual target values.
- `y_hat`: predictions produced by a model.

For example:

```python
import numpy as np

X = np.array([
    [70, 2, 10],
    [95, 3, 5],
    [120, 4, 2]
])

y = np.array([95, 135, 180])

print("X:")
print(X)

print("y:")
print(y)

print("X shape:", X.shape)
print("y shape:", y.shape)
```

Expected output:

```text
X:
[[ 70   2  10]
 [ 95   3   5]
 [120   4   2]]

y:
[ 95 135 180]

X shape: (3, 3)
y shape: (3,)
```

### Understanding Array Shapes

For `X.shape == (3, 3)`:

- The first number represents the number of samples.
- The second number represents the number of features.

Therefore, the dataset contains three samples and three features.

For `y.shape == (3,)`, there are three target values.

In general, if a dataset has `n` samples and `p` features:

```python
X.shape == (n, p)
```

A one-dimensional target array commonly has this shape:

```python
y.shape == (n,)
```

**Remember:** rows usually represent samples, while columns represent features.

---

## 5. Working with NumPy Arrays

NumPy is a Python library used for efficient numerical computation.

Several useful properties of NumPy arrays are:

| Property | Meaning |
|---|---|
| `.shape` | Dimensions of the array |
| `.ndim` | Number of dimensions |
| `.size` | Total number of elements |
| `.dtype` | Data type of elements |

Example:

```python
print("Shape:", X.shape)
print("Dimensions:", X.ndim)
print("Number of elements:", X.size)
print("Data type:", X.dtype)
```

### Array Indexing

```python
# Select the first sample
first_sample = X[0, :]

# Select the first feature from all samples
areas = X[:, 0]

print("First sample:", first_sample)
print("Areas:", areas)
```

Explanation:

- `X[0, :]` selects the first row.
- `X[:, 0]` selects the first column.
- Python indexing starts at zero.

These concepts are essential when working with datasets in NumPy, Pandas, and Machine Learning libraries.

---

## 6. What Is a Machine Learning Model?

A model is a mathematical or computational function that maps input data to an output.

We can represent a model as:

\[
\hat{y} = f(X; \theta)
\]

Where:

- `X` represents the input features.
- `f` represents the model function.
- `θ` represents the model parameters.
- `y_hat` represents the predicted output.

Different algorithms use different model structures.

For example:

- Linear Regression uses a weighted mathematical function.
- Decision Trees use branching rules.
- Neural Networks use layers of interconnected mathematical operations.

### Model Parameters

Parameters are values used by a model to produce predictions.

In a simple linear model, the parameters are the weight and bias.

---

## 7. Understanding a Simple Linear Model

A simple linear model can be represented by:

\[
\hat{y} = wx + b
\]

Where:

- `x`: input value.
- `w`: weight or slope.
- `b`: bias or intercept.
- `y_hat`: predicted output.

Example:

```python
def linear_prediction(x, weight, bias):
    return weight * x + bias


area = 100
weight = 1.5
bias = 20

prediction = linear_prediction(
    area,
    weight,
    bias
)

print("Prediction:", prediction)
```

Output:

```text
Prediction: 170.0
```

The model calculates:

\[
\hat{y} = 1.5 \times 100 + 20 = 170
\]

In this example, the weight and bias were selected manually.

**Important:** This is not a trained Machine Learning model. It is simply a mathematical function with manually chosen parameters.

Later lessons will explain how algorithms learn suitable parameters from data.

---

## 8. What Does Training Mean?

Training is the process of using data and a learning algorithm to adjust model parameters.

A simplified Machine Learning workflow is:

1. Collect relevant data.
2. Inspect and prepare the dataset.
3. Select a suitable model.
4. Generate predictions.
5. Measure prediction errors.
6. Update model parameters.
7. Evaluate performance on unseen data.

### Prediction vs. Training

**Prediction** means using an existing model to generate an output.

**Training** means using data and a learning procedure to fit the model parameters.

For example:

```python
prediction = linear_prediction(100, 1.5, 20)
```

This line only calculates a prediction. It does not train a model.

Training will be studied in greater detail when we explore Linear Regression, Loss Functions, and Gradient Descent.

---

## 9. Types of Supervised Learning

Supervised Learning uses examples that include both inputs and known target values.

Two common types are Regression and Classification.

### 9.1 Regression

Regression predicts a numerical value.

Examples:

- Predicting house prices.
- Estimating delivery time.
- Predicting electricity consumption.

### 9.2 Classification

Classification predicts a category or class.

Examples:

- Spam vs. not spam.
- Cat vs. dog.
- Positive vs. negative review.

The important distinction is the type of output the problem is designed to predict.

---

## 10. Important Terms Summary

| Term | Definition |
|---|---|
| Sample | One example in a dataset |
| Feature | An input variable |
| Target | The value the model should predict |
| Dataset | A collection of samples |
| X | Input features |
| y | Actual target values |
| y_hat | Predicted values |
| Model | A function that maps inputs to outputs |
| Parameter | A value used by a model |
| Training | Fitting model parameters using data |
| Prediction | Producing an output using a model |
| Regression | Predicting a numerical value |
| Classification | Predicting a category |

---

## 11. Knowledge Check

Try to answer these questions before checking the answers.

1. What is Machine Learning?
2. What is the difference between traditional programming and Machine Learning?
3. What is a sample?
4. What is the difference between a feature and a target?
5. If `X.shape == (500, 4)`, how many samples and features are present?
6. What does `y_hat` represent?
7. What are the weight and bias in the equation `y_hat = wx + b`?
8. Does manually selecting a weight and bias mean that a model has been trained?
9. What is the difference between regression and classification?
10. Why do we evaluate a model using data it has not seen during training?

### Answers

<details>
<summary>Click to reveal the answers</summary>

1. Machine Learning is a field of computing in which systems learn patterns from data to make predictions or decisions.
2. Traditional programming relies on explicitly written rules, while Machine Learning uses data and learning algorithms to fit model parameters.
3. A sample is one individual example or record.
4. A feature is an input variable; a target is the value the model is intended to predict.
5. 500 samples and 4 features.
6. It represents a predicted value.
7. The weight controls the slope, and the bias shifts the output.
8. No. The parameters were selected manually rather than learned from data.
9. Regression predicts numerical values, while classification predicts categories.
10. To estimate how well the model generalizes to new examples.

</details>

---

## 12. Day 1 Completion Checklist

Before moving to Day 2, make sure you can:

- [ ] Explain Machine Learning in your own words.
- [ ] Identify samples, features, and targets.
- [ ] Understand `X`, `y`, and `y_hat`.
- [ ] Interpret NumPy array shapes.
- [ ] Explain what a model and its parameters are.
- [ ] Distinguish prediction from training.
- [ ] Explain the difference between regression and classification.

---

## 13. Next Lesson

**Week 1 — Day 2: Statistics for Machine Learning**

Topics:

- Mean and Median
- Variance and Standard Deviation
- Probability
- Data Distributions
- Correlation
- Basic statistical analysis with NumPy and Pandas

---

# Final Project: Mini House ML

The following project is separate from the theoretical lesson above.

Its purpose is to help you apply the concepts by creating and inspecting a small synthetic dataset.

## Project Goal

Create a Python project that generates house data, inspects its structure, and demonstrates a simple manually parameterized prediction function.

## Dataset Features

- `Area`: house area.
- `Rooms`: number of rooms.
- `Age`: building age.
- `Distance`: distance from the city center.
- `Price`: target value.

The dataset is synthetic and intended only for practice.

## Installation

```bash
pip install -r requirements.txt
```

## Run the Project

```bash
python main.py
```

You can also run:

```bash
python analysis.py
python predict.py
```

## Suggested Extensions

1. Add a `Price_per_sqm` column.
2. Print the first ten rows.
3. Check for missing values.
4. Explain why the current project does not train a model.
5. Describe two limitations of synthetic data.

**Success criterion:** You should understand the code and be able to explain how its dataset represents samples, features, and targets.


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

