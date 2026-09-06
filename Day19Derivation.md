# Simple Linear Regression — Deriving 'm' and 'b' (OLS) + Coding from Scratch
---

## 1. Recap: What We're Solving For

- Data: CGPA (input) vs Package (output), **short-of-linear** (roughly linear, but with noise).
- Goal: find the **best fit line** → i.e., find the values of **m (slope)** and **b (intercept)** in:
$$y = mx + b$$
- "Best" = the line that makes **minimum error** across all points.

---

## 2. Two Ways to Calculate 'm' and 'b'

| Approach | What it is |
|---|---|
| **Closed-form solution** | A direct mathematical formula — plug in values, get the answer immediately. No iteration needed. |
| **Non-closed-form solution** | No direct formula exists — you use an iterative approximation technique instead. |

### 2.1 What is a "Closed-form Solution"?
- In math, an expression is in **closed form** if it can be written using a finite number of standard operations (+, −, ×, ÷, known functions) — **without needing iteration or approximation**.
- Example: solving a quadratic equation using the quadratic formula → plug in values → get the answer directly. That's closed-form.
- If no such direct formula exists, you must use **iterative techniques** to approach the answer step by step.

### 2.2 The Two Techniques for Linear Regression
1. **OLS (Ordinary Least Squares)** — the **closed-form / direct formula** approach. Simpler and faster for small numbers of input features (low-dimensional data). This is what **scikit-learn's `LinearRegression` class** uses internally.
2. **Gradient Descent** — the **iterative** approach. Needed when the number of dimensions/features becomes very large, because directly computing a closed-form formula becomes computationally difficult in high dimensions.

> **Note:** Scikit-learn has *two* separate classes:
> - `LinearRegression` → uses OLS internally.
> - `SGDRegressor` → uses Gradient Descent internally.
>
> Both solve the same underlying problem, just via different methods. This lecture focuses on **OLS**, and we will build it entirely from scratch (i.e., derive the formulas ourselves using calculus) before converting it into a proper Python class.

---

## 3. Deriving the OLS Formulas — Step by Step

### 3.1 Setting Up the Error (Loss) Function

- For every point $(x_i, y_i)$, there's a **difference (error)** between:
  - $y_i$ → the *actual* value (real package)
  - $\hat{y}_i = mx_i + b$ → the *predicted* value (what our line says)

- This difference for a single point is: $y_i - \hat{y}_i$

- **Why do we square this difference** instead of just adding them up directly?
  1. Since the differences can be **positive or negative** (points can lie above or below the line), simply summing them would let positive and negative errors cancel out, giving a misleadingly small total. Squaring makes everything **positive**.
  2. Compared to using **absolute value** ($|y_i - \hat y_i|$): the absolute value function is *continuous but not differentiable* at all points (it has a sharp corner), which makes it hard to apply calculus (differentiation) later. **Squaring is smooth and differentiable everywhere**, and also naturally **penalizes large errors (outliers) more heavily** than small ones.

- So, the **Total Error** across all $n$ points becomes:

$$E = \sum_{i=1}^{n} (y_i - \hat y_i)^2 = \sum_{i=1}^{n} (y_i - mx_i - b)^2$$

- This is called the **Loss Function** (or **Cost Function**) in ML — commonly denoted as $L$ or $E$.

### 3.2 The Optimization Goal

> **Find the values of $m$ and $b$ that MINIMIZE this loss function $E$.**

- Note: in this expression, $x_i$ and $y_i$ are **fixed/known data values** (you can't change them) — the *only* things you can vary are $m$ and $b$. That's why $E$ is described as being **"a function of $m$ and $b$"** — changing $m$ or $b$ is what changes the value of $E$.

- **Geometric intuition:** Any straight line in 2D space can be fully described by just $m$ and $b$. To go from any one line to any other line, you either:
  - Change $b$ (slide the line up/down while keeping slope fixed), or
  - Change $m$ (rotate the line around its y-intercept), or
  - Change both together.

### 3.3 Visualizing How E Depends on m and b

- If you fix $b$ and vary $m$ alone → plotting $E$ vs $m$ gives a **parabola-like curve** (U-shaped): error is high when $m$ is far from the "correct" value, dips to a minimum at the ideal $m$, then rises again.
- Similarly, if you fix $m$ and vary $b$ alone → plotting $E$ vs $b$ also gives a similar parabola-like curve.
- If you plot **both $m$ and $b$ together in 3D** (with $E$ as the height/vertical axis), you get a **bowl-shaped surface** — like a valley surrounded by hills. The lowest point of this bowl (the "valley floor") represents the combination of $m$ and $b$ that gives the **minimum total error** — that's exactly the best fit line we're looking for.

### 3.4 Using Calculus to Find the Minimum

- Recall from calculus (Maxima/Minima): **at a minimum point of a function, its derivative (slope of the tangent) equals zero.**
- So to find the $m$ and $b$ that minimize $E$, we:
  1. Take the **partial derivative** of $E$ with respect to $m$, set it to 0.
  2. Take the **partial derivative** of $E$ with respect to $b$, set it to 0.
  3. Solve these **two equations simultaneously** (2 equations, 2 unknowns: $m$ and $b$).

*(We use **partial derivatives** because $E$ is a function of two variables, $m$ and $b$ — not just one — so we differentiate holding one variable constant while varying the other.)*

---

### 3.5 Deriving 'b' (Step-by-Step)

Starting loss function:
$$E = \sum_{i=1}^{n} (y_i - mx_i - b)^2$$

**Partial derivative with respect to $b$:**

Using the chain rule:
$$\frac{\partial E}{\partial b} = \sum_{i=1}^{n} 2(y_i - mx_i - b) \cdot (-1) = 0$$

Simplify (divide both sides by −2):

$$\sum_{i=1}^{n} (y_i - mx_i - b) = 0$$

Split the summation into separate terms:

$$\sum y_i - m\sum x_i - \sum b = 0$$

Since $b$ is a constant added $n$ times: $\sum b = nb$

$$\sum y_i - m\sum x_i - nb = 0$$

Divide everything by $n$:

$$\frac{\sum y_i}{n} - m\frac{\sum x_i}{n} - b = 0$$

Recognize that $\frac{\sum y_i}{n} = \bar y$ (mean of y) and $\frac{\sum x_i}{n} = \bar x$ (mean of x):

$$\bar y - m\bar x - b = 0$$

**Final formula for b:**
$$\boxed{b = \bar y - m\bar x}$$

*(This matches the well-known OLS formula for intercept: mean of Y minus slope times mean of X.)*

---

### 3.6 Deriving 'm' (Step-by-Step)

Now substitute the value of $b$ back into the original loss function, then differentiate with respect to $m$:

$$E = \sum_{i=1}^{n} (y_i - mx_i - b)^2, \quad \text{where } b = \bar y - m\bar x$$

**Partial derivative with respect to $m$:**

$$\frac{\partial E}{\partial m} = \sum_{i=1}^{n} 2(y_i - mx_i - b)(-x_i) = 0$$

Substitute $b = \bar y - m\bar x$ inside:

$$\sum_{i=1}^{n} (y_i - mx_i - (\bar y - m\bar x))(-x_i) = 0$$

Divide both sides by −2 and simplify signs:

$$\sum_{i=1}^{n} (y_i - \bar y - mx_i + m\bar x)\, x_i = 0$$

$$\sum_{i=1}^{n} (y_i - \bar y)x_i - m\sum_{i=1}^{n} (x_i - \bar x)x_i = 0$$

Rearranging to isolate $m$:

$$\sum_{i=1}^{n} (y_i - \bar y)x_i = m\sum_{i=1}^{n} (x_i - \bar x)x_i$$

$$m = \frac{\sum_{i=1}^{n} (y_i - \bar y)x_i}{\sum_{i=1}^{n} (x_i - \bar x)x_i}$$

After further algebraic manipulation (replacing $x_i$ with $(x_i - \bar x)$ in the numerator too, which is a standard OLS algebra step that doesn't change the result), this simplifies to the standard, symmetric form:

**Final formula for m:**
$$\boxed{m = \frac{\sum_{i=1}^{n} (x_i - \bar x)(y_i - \bar y)}{\sum_{i=1}^{n} (x_i - \bar x)^2}}$$

### 3.7 Summary of the Two Final Formulas

$$m = \frac{\sum (x_i - \bar x)(y_i - \bar y)}{\sum (x_i - \bar x)^2} \qquad\qquad b = \bar y - m\bar x$$

- **Compute $m$ first** (since $b$ depends on $m$).
- Then plug $m$ into the $b$ formula.
- **Note:** These formulas are specific to **Simple Linear Regression** (a single input feature). Multiple Linear Regression uses a different (matrix-based) formula, covered separately.

---

## 4. Coding the Class from Scratch

### 4.1 Class Structure

Just like `LinearRegression` in scikit-learn, our custom class needs:
- An `__init__` method to initialize `m` and `b` (starting values, e.g., both set to `None` or some default).
- A `.fit(X_train, y_train)` method → computes and stores `m` and `b` using the training data.
- A `.predict(X_test)` method → uses the stored `m` and `b` to compute predictions for new inputs.

```python
class MeraLR:

    def __init__(self):
        self.m = None
        self.b = None

    def fit(self, X_train, y_train):
        num = 0
        den = 0
        for i in range(X_train.shape[0]):
            num = num + (X_train[i] - X_train.mean()) * (y_train[i] - y_train.mean())
            den = den + (X_train[i] - X_train.mean()) ** 2

        self.m = num / den
        self.b = y_train.mean() - (self.m * X_train.mean())

        print("Slope (m):", self.m)
        print("Intercept (b):", self.b)

    def predict(self, X_test):
        return self.m * X_test + self.b
```

### 4.2 Walkthrough of the `.fit()` Logic

1. **Numerator** ($\sum (x_i - \bar x)(y_i - \bar y)$):
   - Loop through every training point.
   - For each point, compute $(x_i - \bar x) \times (y_i - \bar y)$.
   - Keep adding this value to a running total called `num`.

2. **Denominator** ($\sum (x_i - \bar x)^2$):
   - Similarly, for each point, compute $(x_i - \bar x)^2$.
   - Keep adding this to a running total called `den`.

3. Once the loop finishes (having gone through all training rows):
   - `self.m = num / den`
   - `self.b = y_train.mean() - self.m * X_train.mean()`

4. These are now **stored as attributes** of the object (`self.m`, `self.b`) so `.predict()` can use them later.

### 4.3 The `.predict()` Method

- Simply applies the line equation using the already-learned `m` and `b`:
```python
def predict(self, X_test):
    return self.m * X_test + self.b
```

### 4.4 Using the Class

```python
import numpy as np

X = df['cgpa'].values
y = df['package'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

lr = MeraLR()
lr.fit(X_train, y_train)

print(lr.predict(X_test[0]))
```

### 4.5 Validation Against Scikit-learn

- The lecturer compares outputs: the custom class's `m` and `b` came out to the **same values** (≈0.55 for slope, ≈0.81 for intercept in the example) as scikit-learn's built-in `LinearRegression`.
- This confirms the manually-derived formulas and code are correctly replicating what scikit-learn does internally with OLS.

### 4.6 Important Limitation
> ⚠️ This custom class only works for **Simple Linear Regression** (a single input feature). It will **not work directly for Multiple Linear Regression** (multiple input features) — that requires a different, matrix-based formula and implementation, which will be covered when Multiple Linear Regression is discussed. Scikit-learn's actual `LinearRegression` class, in contrast, already handles both cases internally.

---

## 5. Key Takeaways (Summary)

| Concept | Meaning |
|---|---|
| **Closed-form solution** | Direct formula, no iteration needed |
| **OLS (Ordinary Least Squares)** | The closed-form method used to compute $m$, $b$ directly |
| **Gradient Descent** | Iterative alternative, needed for high-dimensional data |
| **Loss/Cost/Error Function** | $E = \sum (y_i - mx_i - b)^2$ — squared error, summed over all points |
| **Why squared error** | Keeps all errors positive + penalizes large errors + is differentiable (unlike absolute value) |
| **Minimization method** | Set partial derivatives of $E$ w.r.t. $m$ and $b$ to zero, solve simultaneously |
| **Formula for b** | $b = \bar y - m\bar x$ |
| **Formula for m** | $m = \dfrac{\sum (x_i-\bar x)(y_i-\bar y)}{\sum (x_i-\bar x)^2}$ |
| **Custom class** | `.fit()` computes and stores `m`, `b`; `.predict()` applies $y = mx+b$ |
| **Limitation** | Only works for Simple (single-feature) Linear Regression |

### Quick Mental Model
Think of the total error as a **bowl-shaped 3D surface** over all possible $(m, b)$ combinations. Calculus lets us find the exact bottom of that bowl (where the slope of the surface is zero in every direction) — and that bottom point *is* our best-fit $m$ and $b$, giving us the least possible total error.

---
