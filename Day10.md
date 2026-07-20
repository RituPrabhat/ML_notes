## Introduction

Continuing with Feature Engineering. This is a new topic under **Feature Transformation**, called **Mathematical Transformations**.

Split across 2 parts:
1. **This video/note:** `FunctionTransformer` in sklearn (Log Transform, Reciprocal Transform, Square Transform, Square Root Transform)
2. **Next video/note:** `PowerTransformer` — covering **Box-Cox Transform** and **Yeo-Johnson Transform**

---

## Revision: What Has Been Covered So Far in Feature Engineering

So far, Feature Transformation has included:
- Encoding categorical columns
- Handling missing values
- Scaling data

Today's addition: **Mathematical Transformations**.

### Core idea
Very simple concept: you apply some **mathematical formula** to a column in your data, transforming it into something else.

### Transformations covered in this note (via `FunctionTransformer`):
1. **Log Transform**
2. **Reciprocal Transform**
3. **Power Transforms**: Square Transform and Square Root Transform

### Transformations covered in the *next* note (via `PowerTransformer`):
1. **Box-Cox Transform**
2. **Yeo-Johnson Transform**

---

## Why Do This At All? (The Goal)

**Core goal:** These transformations aim to convert your data's distribution into (or closer to) a **Normal Distribution**.

### What is Normal Distribution (quick recap)
If you plot the distribution of any numerical quantity and it forms the classic bell-curve shape (symmetric, centered), that's called a **Normal Distribution** (also referred to via its PDF — Probability Density Function — showing the probability of each value occurring).

Most real-world data is **NOT** normally distributed by default. The goal of these transformations is to convert (or push) your data closer to a normal distribution.

### Why does Normal Distribution matter for ML?

- In the field of **Statistics**, Normal Distribution is extremely important — when a statistician sees normally distributed data, calculations/problem-solving become much easier around it (many statistical formulas and tests are built assuming normality).
- In Machine Learning specifically, this matters mainly for **statistical/linear algorithms** like:
  - **Linear Regression**
  - **Logistic Regression**
  
  These algorithms tend to perform **better** when the input data is normally distributed.
  
- **Tree-based algorithms** (like Decision Trees, Random Forest) generally **don't care** whether the data's distribution is normal or not — it doesn't significantly affect their performance.

**Conclusion:** If you're working with linear/statistical ML algorithms and your data isn't normally distributed, applying these transformations can convert it closer to normal, which may improve your model's performance.

> Note: Besides the standard transforms covered here, you can even create your own **custom transformation** using any mathematical formula (e.g., `x² + 2x`) — the target is simply to see if it produces something closer to a normal distribution. It's all about experimentation.

---

## `FunctionTransformer` — Overview

In sklearn, there are 3 different mathematical transformer-related tools:
1. **`FunctionTransformer`** — most commonly used; covered in this note
2. **`PowerTransformer`** — covered in the next note (Box-Cox, Yeo-Johnson)
3. **`QuantileTransformer`** — not covered in this series (not commonly used, per personal experience)

Using `FunctionTransformer`, you can apply:
- Log Transform
- Reciprocal Transform
- Square Transform / Square Root Transform
- Any custom function you define yourself

---

## How to Check If Data Is Normally Distributed

Before applying any transform, you need a way to check whether your data is already normal or not, and whether a transform actually helped. Three common methods:

### Method 1: Distribution Plot (`distplot`)
```python
sns.distplot(df['column_name'])
Visually plots the distribution — gives a rough/intuitive idea of how close it is to a bell curve (normal shape) vs. skewed.
Method 2: Skewness (skew())

df['column_name'].skew()
If skewness value is close to 0 → distribution is roughly symmetric/normal.
Negative value → left-skewed.
Positive value → right-skewed.
Method 3: Q-Q Plot (Quantile-Quantile Plot) — Most Reliable Method
Considered the most reliable way to check if data is normally distributed.
(A dedicated, in-depth video on Q-Q plots is planned separately, since it's a statistics-heavy topic deserving its own explanation — this note only covers how to read one at a practical level.)
Understanding Q-Q Plots (How to Read Them)
A Q-Q plot has:

X-axis: Theoretical quantiles (from a true normal distribution)
Y-axis: Your actual data's sample quantiles
How to interpret it:
There's a reference straight line (typically at a 45° angle).
The more your data points lie exactly on this line, the more normally distributed your data is.
The more the points deviate from this line (and the more the shape/angle of scattered points curves away), the further your data is from being normally distributed.
Specific shape patterns to recognize:
Pattern	What it indicates
Points lie almost exactly along the line	Data is normally distributed
Data has 2 peaks (bimodal)	Points deviate from the line, causing a visible change in angle around the middle
Right-skewed data	Points tend to curve/rise away from the line on one side
Left-skewed data	Points deviate off the line from the lower end
Fat-tailed data	Points deviate from the line on both ends
Very close to normal	Points stay almost entirely on the line throughout
Simple rule of thumb: The closer your points hug the reference line, the closer your distribution is to normal. The more they drift away (and the more the slope/angle changes), the further from normal.

Q-Q plots will be used throughout this note to check whether a distribution is normal, before and after applying transformations.

Log Transform
Core idea
Very simple: take the logarithm of every value in a column.


df['column_log'] = np.log(df['column_name'])
E.g., if you have the Titanic dataset's Fare column, applying log transform means taking log(fare) (natural log, or log base 10 — either works) for every passenger's value.
Taking the log tends to make the data's distribution more normally distributed than before (not perfectly normal necessarily, but noticeably closer/better than the raw/original form).
When should you use Log Transform?
Cannot be applied to negative values — since you can't take the log of a negative number.
Works best on right-skewed data (data with a long tail toward higher values).
Common interview question: "What kind of data should Log Transform be applied to?" → Answer: Right-skewed data.

Why does Log Transform help? (Intuition)
Example: suppose your column has values like 1, 10, 100, 1000 (each 10x the previous).

On a linear/normal scale, plotting these would show a huge disparity — most values would cluster near zero visually, with a few extreme values stretching the range hugely.
After taking the log: log(1)=0, log(10)=1, log(100)=2, log(1000)=3 — these become equally spaced on a log scale.
So values that were previously very "far apart" (in raw scale) become much closer together after log transform, making the overall distribution look much more even/symmetric — closer to normal.
Key intuition: Log transform converts an additive scale into a multiplicative scale — very distant values on a normal scale become much closer together on a log scale, making the data look more evenly/normally distributed.
Why does this help Linear Models?
Since Linear Regression and Logistic Regression perform better with normally distributed input, applying log transform (which pushes skewed data toward normal) tends to improve their performance.
(This will be discussed in more depth in a dedicated practical/coding session later.)
Three More Transforms (Introduced, to Be Covered With Code)
1. Reciprocal Transform

df['column_reciprocal'] = 1 / df['column_name']
Formula: 1/x for each value x.
Effect: large values become small, and small values become large — essentially "inverts" the scale.
A very different kind of transform compared to log/square — sometimes helpful specifically for right-skewed data too, potentially producing a more normal-looking distribution depending on the data.
2. Square Transform

df['column_squared'] = df['column_name'] ** 2
Formula: x²
Specifically useful for a certain type of data pattern (explanation cut off mid-sentence in transcript — likely left-skewed data, to be confirmed/covered in the next part of the transcript).
3. Square Root Transform

df['column_sqrt'] = np.sqrt(df['column_name'])
Formula: √x
Another power-based transform, generally applicable to right-skewed data (similar spirit to log transform, but a "gentler" transformation).
