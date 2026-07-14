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



### Q7. Does Standardization change the order of data?

No. It preserves the relative ordering of observations while changing only their scale.
