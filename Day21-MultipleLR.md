# Multiple Linear Regression — Introduction & Geometric Intuition

---

## 1. Recap: The Three Types of Linear Regression

1. **Simple Linear Regression** — 1 input column, 1 output column.
2. **Multiple Linear Regression** — **more than 1** input column, 1 output column. *(Today's topic)*
3. **Polynomial Regression** — a further extension (not covered yet).

### 1.1 Recap of Simple Linear Regression
- One input column, one output column.
- You plot the data (input vs output), check if the relationship looks linear.
- If yes, draw a **best fit line** — the line's equation is $y = mx + b$.
- The whole "job" of the algorithm was to find the right $m$ and $b$.

---

## 2. When Do We Need Multiple Linear Regression?

Multiple Linear Regression is needed whenever your dataset has **more than one input column**.

**Example:** Predicting Package (output) using:
- CGPA (input 1)
- IQ (input 2)
- *(could be Gender, 12th marks, etc. too — any number of inputs)*

> **Important practical note:** In real-world datasets, you will **almost always** have more than one input column. So even though Simple Linear Regression was taught first, **Multiple Linear Regression is what you'll actually use most of the time** in practice.

### 2.1 Why Learn Simple Linear Regression First, Then?
- Everything you learned in Simple Linear Regression **directly carries over** and extends into Multiple Linear Regression.
- Multiple Linear Regression can be thought of as a **generalization/expansion of Simple Linear Regression** — or conversely, Simple Linear Regression is just a **special case of Multiple Linear Regression** where you happen to have only 1 input instead of many.

---

## 3. Geometric Intuition — From 2D Line to 3D Plane (and Beyond)

### 3.1 Visualizing with 2 Inputs (3D Case)

With **2 inputs** (e.g., CGPA and IQ) and 1 output (Package), your data can be visualized in **3D space**:
- X-axis → CGPA ($X_1$)
- Y-axis → IQ ($X_2$)
- Z-axis → Package (output)

Just like in Simple Linear Regression you drew a **best fit line** through 2D data, here you draw a **best fit plane** through the 3D data — a flat plane that:
- Cuts through the cloud of data points.
- Some points sit above the plane, some below it.
- The plane tries to stay **as close as possible** to all the points overall — same "best fit" idea as before, just in one dimension higher.

### 3.2 Beyond 3D — "Hyperplane"

- With exactly 2 inputs, the best-fit "surface" is a **plane** (something we can visualize in 3D).
- But what if you have **3, 4, or more inputs**? Then your data technically lives in 4D, 5D, or higher-dimensional space — which we **cannot visually picture**, but the same underlying idea still applies mathematically.
- In these higher dimensions, this best-fit surface is called a **hyperplane** — a generalized term for "the flat surface that best fits the data," regardless of how many dimensions we're in.

> **Key takeaway:** Whether it's a line (2D), a plane (3D), or a hyperplane (4D+), the **core goal is identical** — find the flat surface that stays as close as possible to all data points.

---

## 4. Deriving the Multiple Linear Regression Equation

### 4.1 Starting Point: Simple Linear Regression's Equation

Recall:
$$y = mx + b$$

### 4.2 Extending to 2 Inputs

If output $y$ now depends on **two** inputs ($x_1$ and $x_2$), the equation naturally extends to:

$$y = m_1 x_1 + m_2 x_2 + b$$

For more consistent/standard notation (commonly used in literature), this is often rewritten using **beta ($\beta$)** notation:

$$y = \beta_0 + \beta_1 x_1 + \beta_2 x_2$$

Where:
- $\beta_0$ = the intercept (equivalent to $b$ before)
- $\beta_1$ = coefficient/weight for $x_1$ (e.g., CGPA)
- $\beta_2$ = coefficient/weight for $x_2$ (e.g., IQ)

This equation represents a **plane** (in 3D space).

### 4.3 Extending Further — 3 Inputs (4D)

If you had 3 inputs, the equation would naturally extend to:

$$y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \beta_3 x_3$$

...and so on — every additional input column adds one more $\beta$ term.

### 4.4 The General Formula (n Inputs)

For a dataset with **$n$ input columns**, the general Multiple Linear Regression equation is:

$$y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots + \beta_n x_n$$

Or, written more compactly using summation/dot product notation:

$$y = \beta_0 + \sum_{i=1}^{n} \beta_i x_i$$

- If there's only **1 input column** ($n=1$), this reduces exactly back to the Simple Linear Regression equation: $y = \beta_0 + \beta_1 x_1$ (same as $y = mx + b$, just renamed).

### 4.5 What You Need to Calculate

- For an **$n$-input** problem, you need to find/calculate **$(n+1)$ coefficients** in total — that is, $\beta_0, \beta_1, \beta_2, \ldots, \beta_n$.
  - Example: 2 input columns → 3 coefficients needed ($\beta_0, \beta_1, \beta_2$).
  - Example: 3 input columns → 4 coefficients needed ($\beta_0, \beta_1, \beta_2, \beta_3$).
- Using Linear Regression essentially means **solving for all these $(n+1)$ unknowns** simultaneously (the actual math/formula for this will be derived from scratch in the next lecture).

---

## 5. Interpreting the Coefficients (Human/Intuitive Meaning)

Just like in Simple Linear Regression, slope told you "how much the output depends on the input" — the same logic extends here, but **each $\beta$ tells you about its own specific input column**.

### 5.1 Example: $y = \beta_0 + \beta_1 (\text{CGPA}) + \beta_2 (\text{IQ})$

- **If $\beta_2$ (IQ's coefficient) is very small/close to 0:**
  → IQ has almost no impact on calculating Package. In other words, Package **doesn't really depend on IQ**.

- **If $\beta_1$ (CGPA's coefficient) is very small/close to 0:**
  → CGPA has almost no impact on Package. Package doesn't really depend on CGPA.

- **If $\beta_1$ is very large (e.g., much greater than others):**
  → CGPA plays a much more important/dominant role in determining Package compared to the other inputs.

> **In short: each $\beta_i$ tells you how important/influential that particular input column is in predicting the output** — a bigger magnitude coefficient means that feature has a bigger effect on the prediction.

### 5.2 Role of $\beta_0$ (Intercept)
- Just like in Simple Linear Regression, $\beta_0$ represents the baseline output value — i.e., what the prediction would be **if all input columns were 0**. Same interpretation as $b$ before, just carried forward.

### 5.3 Making a Prediction — How It Works

To predict Package for a new student:
1. Get their CGPA and IQ (the $x_1, x_2$ values).
2. Multiply CGPA by $\beta_1$, multiply IQ by $\beta_2$.
3. Add $\beta_0$ to the sum.
4. That gives you the predicted Package.

$$\text{Predicted Package} = \beta_0 + \beta_1(\text{CGPA}) + \beta_2(\text{IQ})$$

So, to make any prediction, you need: the new data point's input values, **plus** the learned $\beta_0, \beta_1, \beta_2, \ldots$ values from training.

---

## 6. Code Demo Walkthrough (Conceptual Summary)

The lecture demonstrates this using a synthetically generated dataset (via scikit-learn's `make_regression` function) with **2 input columns and 1 output column** (so it can still be visualized in 3D).

### Steps shown:
1. **Generate a synthetic dataset** with 2 input features and 1 output — using `make_regression()`, which can generate linear regression datasets of any number of dimensions.
2. **Convert to a DataFrame** for easier handling/visualization.
3. **Visualize the raw 3D data** using a 3D scatter plot (via the `plotly` library) — showing that the data does look roughly planar (i.e., a flat plane could reasonably fit through it).
4. **Train/Test split** the data — same process as with Simple Linear Regression.
5. **Fit a `LinearRegression` model** from scikit-learn on the training data (works exactly the same way regardless of how many input columns there are — `.fit(X_train, y_train)`).
6. **Make a prediction** using `.predict()` on the test set, and calculate the **R² Score** to evaluate performance (example from the lecture: R² ≈ 0.75, meaning ~75% of variance in the output is explained jointly by both input columns together).
7. **Visualize the fitted best-fit plane** overlaid on the 3D scatter plot — showing how the plane cuts through the cloud of points, trying to minimize overall distance from all of them (same "best fit" idea as the 2D line, just one dimension up).

### 6.1 Checking the Learned Coefficients
- Just like `.coef_` gave you a single slope value ($m$) in Simple Linear Regression, in Multiple Linear Regression, **`.coef_` returns an array/list of values** — one for **each** input column ($\beta_1, \beta_2, \ldots$).
  - With 2 inputs → you get 2 values back (e.g., $\beta_1$ and $\beta_2$).
  - With 3 inputs → you'd get 3 values back, and so on.
- `.intercept_` still gives you the single $\beta_0$ value, regardless of how many inputs there are.

---

## 7. Why This Lecture Was "High-Level" (No Full Math Yet)

- Since high-dimensional data (4D+) **cannot be visually plotted or graphed**, it's harder to build intuition purely from pictures once you go beyond 3 dimensions (2 inputs).
- This lecture intentionally focused on **conceptual/geometric understanding** and using the **existing scikit-learn class** to see Multiple Linear Regression in action.
- **The next lecture** will cover the **full mathematical derivation from scratch** — deriving the formula to calculate all the $\beta$ values (a matrix-based approach, since the simple calculus trick used for $m$ and $b$ in Simple Linear Regression doesn't scale cleanly to many dimensions).

> **Practical suggestion from the lecturer:** Try applying Multiple Linear Regression yourself on a real dataset (e.g., a housing dataset with 5 input columns) to get comfortable applying the scikit-learn class practically, even before the full math is covered.

---

## 8. Key Takeaways (Summary)

| Concept | Meaning |
|---|---|
| **Multiple Linear Regression** | Regression with more than 1 input column |
| **Relationship to Simple LR** | Simple LR is just a special case of Multiple LR (n=1) |
| **Geometric shape (2 inputs)** | A best-fit **plane** in 3D space |
| **Geometric shape (3+ inputs)** | A best-fit **hyperplane** in higher-dimensional space (not visualizable) |
| **General equation** | $y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots + \beta_n x_n$ |
| **Number of coefficients to find** | $n+1$ (one $\beta_i$ per input, plus $\beta_0$) |
| **Interpreting $\beta_i$** | How much that specific input influences the output; near-zero = little/no influence |
| **Interpreting $\beta_0$** | Baseline output value when all inputs = 0 |
| **`.coef_` in code** | Returns an array of coefficients (one per input column) |
| **`.intercept_` in code** | Returns the single $\beta_0$ value |

### Quick Mental Model
Think of Simple Linear Regression as fitting a **straight line** through a 2D scatter of points. Multiple Linear Regression just does the exact same thing, except the "line" becomes a **flat plane** (with 2 inputs) or a **flat hyperplane** (with 3+ inputs) — and instead of finding just one slope, you're now finding **one slope (coefficient) per input feature**, all packed into a single equation.

---
