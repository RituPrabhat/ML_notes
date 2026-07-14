# Feature Scaling

Feature Scaling is one of the **last steps in Feature Engineering**. After handling missing values, categorical variables, outliers, and other preprocessing tasks, we scale our numerical features before feeding them into a Machine Learning model.

It is one of the simplest yet most important preprocessing techniques because many Machine Learning algorithms are sensitive to the scale of features.

---

# What is Feature Scaling?

## Definition

Feature Scaling is the process of **bringing all independent (input) features to a similar range** without changing their underlying relationships.

In simple words,

> Feature Scaling changes the scale (range) of numerical features so that all features contribute equally while training the model.

It only affects **input features (X)** and **not the target/output variable (y).**

---

## Example

Suppose we want to predict a student's salary package after placement.

| IQ | CGPA | Package (LPA) |
|----|-------|---------------|
|110|8.2|12|
|125|9.1|18|
|95|7.4|8|

Here,

**Independent Features (X):**

- IQ
- CGPA

**Target Variable (y):**

- Package

Suppose,

- IQ ranges between **80–160**
- CGPA ranges between **0–10**

These two features have different scales.

After Feature Scaling, both will have approximately similar ranges.

---

# Why Do We Need Feature Scaling?

Many Machine Learning algorithms calculate the **distance between data points**.

Examples include:

- K-Nearest Neighbors (KNN)
- K-Means Clustering
- Support Vector Machine (SVM)
- Principal Component Analysis (PCA)

These algorithms usually rely on **Euclidean Distance**.

---

## Euclidean Distance

For two points,

\[
Distance=\sqrt{(x_2-x_1)^2+(y_2-y_1)^2}
\]

The feature with the larger numerical values contributes much more to the distance.

---

## Example

Consider the following dataset.

| Age | Salary |
|------|---------|
|20|40,000|
|25|80,000|

Difference in:

Age = **5**

Salary = **40,000**

During distance calculation,

Age Difference      = 5

Salary Difference   = 40,000

Obviously,

40,000 >> 5

Therefore, salary dominates the calculation while age contributes very little.

The model unintentionally gives much more importance to salary.

This leads to poor learning.

---

## After Feature Scaling

Suppose we transform the data.

| Age | Salary |
|------|---------|
|0.20|0.30|
|0.40|0.70|

Now both features have similar ranges.

During distance calculation,

- Age contributes fairly.
- Salary contributes fairly.

No feature dominates another.

---

# Which Algorithms Need Feature Scaling?

## Feature Scaling is Required

- K-Nearest Neighbors (KNN)
- K-Means Clustering
- Support Vector Machine (SVM)
- Principal Component Analysis (PCA)
- Neural Networks
- Logistic Regression
- Linear Regression (Gradient Descent implementation)

---

## Usually Not Required

Tree-based algorithms do not use distance.

Examples:

- Decision Tree
- Random Forest
- XGBoost
- LightGBM

These algorithms split data based on conditions rather than distances.

---

# Types of Feature Scaling

There are two major types.

Feature Scaling
│
├── Standardization
│
└── Normalization

Normalization itself has multiple techniques.

For example,

- Min-Max Scaling
- Robust Scaling
- Max-Abs Scaling

However, the most commonly used normalization technique is **Min-Max Scaling**.

---

# Standardization (Z-Score Scaling)

Standardization is the most popular Feature Scaling technique.

It transforms every value using the following formula.

\[
z=\frac{x-\mu}{\sigma}
\]

new_value = (value - mean) / standard_deviation

where,

- **x** = Original value
- **μ (mu)** = Mean of the feature
- **σ (sigma)** = Standard Deviation of the feature

The transformed value is called a **Z-score**.

---

# Step-by-Step Example

Suppose we have ages.

20
25
30
35
40

### Step 1

Calculate Mean.

Mean = 30

### Step 2

Calculate Standard Deviation.

Suppose,

Standard Deviation = 5

### Step 3

Apply the formula.

For Age = 20

\[
z=\frac{20-30}{5}
=-2
\]

For Age = 25

\[
z=\frac{25-30}{5}
=-1
\]

For Age = 30

\[
z=\frac{30-30}{5}
=0
\]

For Age = 35

\[
z=\frac{35-30}{5}
=1
\]

For Age = 40

\[
z=\frac{40-30}{5}
=2
\]

New column becomes

| Original Age | Standardized Age |
|---------------|------------------|
|20|-2|
|25|-1|
|30|0|
|35|1|
|40|2|

---

# Properties of Standardization

After applying Standardization,

### Property 1

The mean becomes

Mean = 0

---

### Property 2

The standard deviation becomes

Standard Deviation = 1

These two properties always hold after proper standardization.

---

# Understanding Standardization Intuitively

Suppose our original data looks like this.

20   25   30   35   40

Mean

30

The data is centered around **30**.

After Standardization,

-2  -1   0   1   2

Now,

the center shifts from **30 to 0**.

So instead of asking

> "How old is this person?"

the model now asks

> "How far is this person from the average age?"

That is exactly what a Z-score represents.

---

# Geometric Intuition of Standardization

Imagine plotting two features.

          Salary

             •
         •
      •
   •
____________________

        Age

Every point represents one data sample.

The center of this cloud is the **mean**.

---

## Step 1 — Mean Centering

First,

subtract the mean from every feature.

The whole cloud shifts so that its center becomes

(0,0)

Nothing else changes.

Only the location changes.

---

## Step 2 — Scaling

Next,

divide every feature by its standard deviation.

If the spread is very large,

the cloud shrinks.

If the spread is very small,

the cloud expands.

Finally,

every feature has

Mean = 0

Standard Deviation = 1

---

# Mean Centering

Subtracting the mean from every observation is called

> **Mean Centering**

Formula
New Value

=

Original Value

−

Mean

This shifts the data so that its center becomes zero.

---

# Scaling

After Mean Centering,

divide every value by the Standard Deviation.

Centered Value

÷

Standard Deviation

This ensures every feature has similar spread.

---

# What Does Standardization Actually Do?

Standardization performs **two operations**.

Original Data

↓

Mean Centering

↓

Scaling using Standard Deviation

↓

Standardized Data

So,

Standardization = Mean Centering + Scaling

---

# Does Standardization Change Relationships?

No.

Standardization **does not change the relationships between data points.**

It only changes the scale.

For example,

20 < 25 < 30

After Standardization,

-2 < -1 < 0

The ordering remains exactly the same.

---

# What Happens to Outliers?

Standardization **does not remove outliers**.

Instead,

outliers are also transformed using the same formula.

If a value was extremely far from the mean before,

it will still remain far after Standardization.

For example,

Original Ages

20
22
21
24
90

After Standardization,

-0.6
-0.4
-0.5
-0.2
7.1

The outlier still appears as an extreme value.

Therefore,

> **Always detect and treat outliers before applying Feature Scaling whenever appropriate.**

---

# Advantages of Standardization

- Easy to implement.
- Improves convergence of Gradient Descent.
- Prevents large-valued features from dominating.
- Makes distance calculations fair.
- Widely used in Machine Learning.

---

# Limitations

- Sensitive to outliers.
- Does not remove skewness.
- Does not normalize the distribution.

---

# Key Takeaways

- Feature Scaling is one of the last preprocessing steps before model training.
- It is applied only to input (independent) features.
- Many Machine Learning algorithms require features to be on similar scales.
- Standardization is the most commonly used Feature Scaling technique.
- Standardization converts data so that:
  - Mean = 0
  - Standard Deviation = 1
- Standardization performs two operations:
  - Mean Centering
  - Scaling
- It changes the scale of the data but preserves the relationships between observations.
- Standardization does **not** remove outliers.

---
# Standardization in Scikit-learn

Scikit-learn provides the **`StandardScaler`** class to perform Standardization automatically.

Instead of manually calculating:

- Mean
- Standard Deviation
- Applying the Z-score formula

`StandardScaler` does everything for us.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
```

---

# Correct Workflow for Standardization

A very important rule is:

> **Always split the dataset into training and testing sets before applying Feature Scaling.**

Correct workflow:

```
Dataset
   │
   ▼
Train-Test Split
   │
   ▼
Fit StandardScaler on Training Data
   │
   ▼
Transform Training Data
   │
   ▼
Transform Test Data
```

---

# Why Should We Split Before Scaling?

Suppose we have:

```
Total Dataset = 1000 samples

Training = 800

Testing = 200
```

If we calculate the mean using **all 1000 samples**, the model indirectly gains information about the test data.

This is called **Data Leakage**.

Since the model should never "see" the test data during training, we must avoid this.

---

# The Correct Procedure

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

Create the scaler.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
```

---

## Step 1: Fit on Training Data

```python
scaler.fit(X_train)
```

### What does `fit()` do?

It **does not change the data.**

Instead, it learns and stores:

- Mean (μ)
- Standard Deviation (σ)

for each feature.

For example,

```
Age Mean = 37.5

Age SD = 10.2

Salary Mean = 70000

Salary SD = 22000
```

These values are stored inside the scaler.

---

## Step 2: Transform the Training Data

```python
X_train_scaled = scaler.transform(X_train)
```

Now every value is converted using

\[
z=\frac{x-\mu}{\sigma}
\]

using the mean and standard deviation learned from the training data.

---

## Step 3: Transform the Test Data

```python
X_test_scaled = scaler.transform(X_test)
```

Notice something important:

We **do not call `fit()` again** on the test data.

Instead, we reuse the statistics learned from the training data.

---

# Never Do This

❌ Incorrect:

```python
scaler.fit(X_train)

scaler.fit(X_test)
```

This causes **Data Leakage**.

---

# Always Do This

✔ Correct

```python
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)

X_test_scaled = scaler.transform(X_test)
```

---

# Fit vs Transform

| Method | Purpose |
|---------|---------|
| `fit()` | Learns the mean and standard deviation from the training data. |
| `transform()` | Applies the learned statistics to scale the data. |
| `fit_transform()` | Performs both operations together (commonly used on training data). |

---

# Why Does `StandardScaler` Return a NumPy Array?

Suppose the input is a pandas DataFrame.

```python
X_train
```

Output after scaling:

```python
X_train_scaled
```

becomes a **NumPy array**.

This is normal because `StandardScaler` performs numerical computations and returns the transformed numerical values.

If you want a DataFrame again:

```python
import pandas as pd

X_train_scaled = pd.DataFrame(
    X_train_scaled,
    columns=X_train.columns
)
```

---

# How Can We Verify Standardization?

Use the `describe()` function before and after scaling.

### Before Standardization

| Statistic | Age |
|------------|-----|
| Mean | 37.9 |
| Std | 10.34 |

---

### After Standardization

| Statistic | Age |
|------------|-----|
| Mean | ≈ 0 |
| Std | ≈ 1 |

This confirms that Standardization has worked correctly.

---

# Visual Effect of Standardization

Suppose we plot:

- Age
- Salary

before scaling.

The data points may look like this:

```
Salary

      •
          •
               •


______________________

        Age
```

Notice:

- Salary has a much larger range.
- Age has a much smaller range.

After Standardization:

```
        •
     •
        •
   •

______________________
```

### Important Observation

- The **shape of the data remains the same.**
- The **relative positions of the points remain the same.**
- Only the **scale changes**.

---

# Distribution Before and After Scaling

A very important property of Standardization is:

> **It does not change the shape of the distribution.**

Suppose Age originally follows this distribution:

```
      /\

    /    \

__/        \__
```

After Standardization:

```
      /\

    /    \

__/        \__
```

The curve is exactly the same.

Only the x-axis values have changed.

Similarly, the Salary distribution also keeps its shape.

---

# Does Standardization Improve Accuracy?

The instructor demonstrated this using two models.

## Logistic Regression

Without Standardization:

```
Accuracy = 65%
```

With Standardization:

```
Accuracy = 86%
```

### Why?

Logistic Regression uses **Gradient Descent** for optimization.

Gradient Descent converges much faster and more effectively when features are on a similar scale.

---

## Decision Tree

Without Scaling:

```
Accuracy ≈ Same
```

With Scaling:

```
Accuracy ≈ Same
```

### Why?

Decision Trees do not calculate distances or optimize using Gradient Descent.

Instead, they split the data using threshold values (e.g., `Age < 30`), so feature scaling has almost no effect.

---

# Effect of Outliers on Standardization

Suppose the original Age values are:

```
18
22
25
30
35
```

Now we add an outlier:

```
18
22
25
30
35
120
```

After Standardization:

```
-0.7
-0.5
-0.3
0.1
0.3
6.8
```

The outlier is still an outlier.

### Important Point

> **Standardization does not remove outliers.**

It simply rescales them.

If outliers are present, they should usually be handled **before** applying Standardization.

---

# When Should You Use Standardization?

Standardization is recommended for algorithms that rely on:

- Distance calculations
- Gradient Descent
- Variance

### Use Standardization for:

### 1. K-Nearest Neighbors (KNN)

Uses Euclidean Distance.

Without scaling, features with larger values dominate the distance.

---

### 2. K-Means Clustering

Assigns points to clusters based on distance.

Scaling ensures every feature contributes equally.

---

### 3. Principal Component Analysis (PCA)

PCA identifies directions with maximum variance.

If one feature has a much larger scale, PCA becomes biased toward that feature.

Standardization makes PCA more reliable.

---

### 4. Gradient Descent Based Algorithms

Examples:

- Linear Regression (Gradient Descent)
- Logistic Regression
- Neural Networks

Scaling helps these algorithms converge faster and more efficiently.

---

# When is Standardization Not Required?

Tree-based algorithms generally do **not** require Feature Scaling.

Examples:

- Decision Tree
- Random Forest
- XGBoost
- LightGBM
- CatBoost

### Why?

These algorithms compare values using thresholds instead of calculating distances.

For example:

```
Age < 30
```

Whether Age is:

```
30
```

or

```
0.32
```

the comparison logic remains the same.

Hence, scaling has little or no impact.

---

# Algorithms That Require Standardization

| Algorithm | Standardization Needed? | Reason |
|-----------|--------------------------|--------|
| KNN | ✅ Yes | Distance-based |
| K-Means | ✅ Yes | Distance-based |
| SVM | ✅ Yes | Distance-based optimization |
| PCA | ✅ Yes | Variance-sensitive |
| Logistic Regression | ✅ Yes | Gradient Descent |
| Linear Regression (GD) | ✅ Yes | Gradient Descent |
| Neural Networks | ✅ Yes | Faster convergence |
| Decision Tree | ❌ No | Threshold-based |
| Random Forest | ❌ No | Threshold-based |
| XGBoost | ❌ No | Tree-based |
| LightGBM | ❌ No | Tree-based |

---

# Key Takeaways

- Always split the dataset before applying Standardization.
- Fit the scaler only on the training data.
- Transform both training and testing data using the same scaler.
- Standardization changes the scale but preserves the shape of the data distribution.
- Standardization does not remove outliers.
- Distance-based and Gradient Descent-based algorithms benefit greatly from Standardization.
- Tree-based algorithms usually do not require Standardization.


No. It preserves the relative ordering of observations while changing only their scale.
