What is Feature Engineering?
<img width="862" height="362" alt="image" src="https://github.com/user-attachments/assets/642ae5a3-e7ca-429d-b48f-52f5020ce2ff" />

Before definition, let's understand the problem.
Suppose you have this dataset:
Age	Salary	Bought Laptop
22	25000	    No
45	90000	    Yes
You directly train a model.

Now another dataset:
Date of Birth	      Monthly Salary	      Bought Laptop
01-01-2002	           25000	            No
01-01-1980	           90000            	Yes

Humans immediately understand:
Age matters
Salary matters

But machine only sees numbers and text.

Feature Engineering is the process of converting raw data into more useful information for machine learning.
A feature is simply an input column.
Example: Age	Salary	Experience
Features:
Age
Salary
Experience

Target:
Purchased

In ML language:
X = Features
Y = Target

Why Feature Engineering Exists:
Real-world data is ugly. You may have:
Name
Address
Date
Gender
Salary
Comments

ML algorithms cannot properly understand many of these columns directly.
We transform them into meaningful numerical representations.
This transformation process is Feature Engineering.

A Famous Example:
Suppose we want to predict house prices.
Dataset:
House Size 1000
Model sees only size.
Now we create: Size	Bedrooms	Age
Performance improves. Why?Because we provided more useful information.This is feature engineering.

<img width="927" height="510" alt="image" src="https://github.com/user-attachments/assets/536c92e9-7bc7-43a9-bf31-a14982614d8d" />

# 1. Feature Transformation

Feature Transformation means changing an existing feature into a better representation that is easier for the model to learn from.

We already have the feature (column).

We simply transform it into another form.

Examples include:

- Missing Value Imputation
- Handling Categorical Variables
- Outlier Detection
- Feature Scaling
- Log Transformation
- Box-Cox Transformation

---

# Missing Value Imputation

## What are Missing Values?

Missing values are data points that are not available.

Example:

| Name | Age |
|------|-----|
| Ritu | 20 |
| Rahul | NaN |
| Aman | 22 |

Here, Rahul's age is missing.

---

## Why do Missing Values Occur?

Possible reasons include:

- User forgot to enter data
- Data collection error
- Sensor failure
- Database issue
- Data corruption

---

## Why Are Missing Values a Problem?

Most Machine Learning algorithms (especially in scikit-learn) cannot work directly with missing values.

If missing values are present, model training may fail or produce incorrect results.

Therefore, they must be handled first.

---

## Methods to Handle Missing Values

### 1. Remove Missing Data

Use when only a few rows contain missing values.

Example:

If only 5 out of 10,000 rows are missing, deleting them usually has little effect.

---

### 2. Fill Missing Values (Imputation)

Instead of deleting rows, replace missing values.

Common methods:

| Data Type | Replacement |
|------------|-------------|
| Numerical | Mean |
| Numerical (with outliers) | Median |
| Categorical | Mode |

Example:

```
10
15
NaN
20
25
```

Mean = 17.5

After imputation:

```
10
15
17.5
20
25
```

This process is called **Imputation**.

---

# Handling Categorical Variables

## What are Categorical Variables?

These contain labels instead of numbers.

Example:

| Animal |
|---------|
| Dog |
| Cat |
| Lion |
| Horse |

These are categories, not numerical values.

---

## Why Are They a Problem?

Most Machine Learning algorithms perform mathematical operations.

They cannot directly understand words like:

- Dog
- Cat
- Horse

Therefore, we must convert them into numbers.

This conversion process is called **Encoding**.

---

## Example: One-Hot Encoding

Original Data:

| Animal |
|---------|
| Dog |
| Cat |
| Horse |

After One-Hot Encoding:

| Dog | Cat | Horse |
|----|----|------|
|1|0|0|
|0|1|0|
|0|0|1|

Each category becomes a new binary column.

---

## Another Transformation: Binning

Sometimes numerical values are converted into categories.

Example:

Instead of storing exact age:

```
Age = 8
Age = 17
Age = 32
Age = 65
```

We group them into ranges:

| Age | Category |
|------|----------|
|8|Child|
|17|Teenager|
|32|Adult|
|65|Senior Citizen|

This process is called **Binning**.

---

# Outlier Detection

## What is an Outlier?

An outlier is a data point that is very different from the rest of the data.

Example:

Class marks:

```
48
50
49
52
51
980
```

Here,

**980** is an outlier because it is extremely different from all other values.

---

## Why Are Outliers Dangerous?

Many Machine Learning algorithms try to fit a pattern to all data points.

If extreme values exist, they can pull the model away from the true pattern.

---

## Example

Suppose we are fitting a Linear Regression line.

Without outlier:

```
● ● ● ● ●
 \\\\\
  Best Fit Line
```

With one extreme outlier:

```
● ● ● ● ●

            ● (Outlier)

Regression line bends toward the outlier.
```

The line becomes less accurate for the majority of data.

---

## Should We Always Remove Outliers?

No.

It depends on the problem.

Sometimes an outlier is:

- A measurement error ✅ Remove it.
- A genuine rare event ❌ Keep it.

Example:

A salary of ₹5 crore may be an outlier, but if it belongs to a CEO, it is valid.

Always understand the reason before removing outliers.

---


