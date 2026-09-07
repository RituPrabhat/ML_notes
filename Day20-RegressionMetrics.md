# Regression Evaluation Metrics — Complete Notes

---

## 0. Why We Need These Metrics

Once you apply a regression algorithm (e.g., Linear Regression) to a dataset, you need a way to **measure how well it's performing** — i.e., how close its predictions are to the actual values.

There are **5 major metrics** covered in this lecture:
1. **MAE** — Mean Absolute Error
2. **MSE** — Mean Squared Error
3. **RMSE** — Root Mean Squared Error
4. **R² Score** — Coefficient of Determination (a.k.a. "Goodness of Fit")
5. **Adjusted R² Score**

> **Important mindset:** No single metric is universally "best." Different metrics suit different datasets/situations — you should generally calculate **all of them together** and compare, rather than relying on just one.

---

## 1. MAE (Mean Absolute Error)

### 1.1 Concept
- For every data point, calculate the **absolute difference** between the actual value ($y_i$) and the predicted value ($\hat y_i$) — i.e., how far off the model's prediction is from reality.
- We use the **absolute value** (ignore the sign/direction) because errors can be positive (prediction too low) or negative (prediction too high) — we only care about the *magnitude* of the mistake, not the direction.
- Sum up these absolute errors across all points, then divide by the number of points ($n$) to get the **average/mean error**.

### 1.2 Formula
$$MAE = \frac{1}{n}\sum_{i=1}^{n} |y_i - \hat y_i|$$

### 1.3 Advantages
- **Same units as the output (y) column.** E.g., if predicting Package in LPA, the MAE value is also in LPA — making it very intuitive to interpret. If MAE = 1.5, you immediately know "on average, the model is off by 1.5 LPA."
- **Robust to outliers** (relatively) — because errors aren't squared, one unusually large error doesn't disproportionately dominate the metric as much as it would in MSE.

### 1.4 Disadvantages
- **Not differentiable at all points** — the absolute value function has a sharp "corner" at zero, which is a problem when you need to use calculus-based optimization techniques (like Gradient Descent) that require taking derivatives everywhere. This is the **biggest drawback** of MAE, and it's the main reason MSE was introduced as an alternative.

---

## 2. MSE (Mean Squared Error)

### 2.1 Concept
- Nearly identical to MAE, except instead of taking the **absolute value** of the error, you **square** it.
- Squaring naturally makes all errors positive (since a negative number squared becomes positive) — achieving the same "ignore the sign" goal as absolute value, but in a way that's **smooth and differentiable everywhere**.

### 2.2 Formula
$$MSE = \frac{1}{n}\sum_{i=1}^{n} (y_i - \hat y_i)^2$$

*(Geometrically: each error can be visualized as the side of a square; MSE is essentially summing up the areas of all these little squares — one per data point — and taking their average.)*

### 2.3 Advantages
- **Differentiable everywhere** — this is the biggest benefit, since it allows using derivative-based optimization techniques (like the one used to derive $m$ and $b$ for Linear Regression, or Gradient Descent) without any issues.

### 2.4 Disadvantages
- **Unit mismatch** — since errors are squared, the final MSE value is in **squared units** of the output (e.g., if y is in LPA, MSE will be in LPA²). This makes it **harder to interpret intuitively** — you'd need to explain to someone that the "real" error is the square root of this number.
- **Heavily penalizes outliers** — because squaring a large error makes it disproportionately larger, a model using MSE as its loss function will try very hard to avoid large errors on any single point, sometimes at the cost of overall fit. This means MSE is quite **sensitive to outliers** in the data.

---

## 3. RMSE (Root Mean Squared Error)

### 3.1 Concept
- Simply the **square root of MSE**. Introduced specifically to fix MSE's "unit mismatch" problem.

### 3.2 Formula
$$RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^{n} (y_i - \hat y_i)^2} = \sqrt{MSE}$$

### 3.3 Properties
- Inherits **all the same properties as MSE** (differentiable everywhere, sensitive to outliers), since it's just MSE's square root.

### 3.4 Advantage
- **Same units as y** — since we took the square root, the units match the original output column again (e.g., LPA), making interpretation much easier — same benefit MAE had.

### 3.5 Disadvantage
- **Still sensitive to outliers** — same issue as MSE, since it's derived directly from it.

### 3.6 Popularity
- **RMSE is the most commonly used metric overall**, especially in **Deep Learning**, because it combines differentiability (from MSE) with easy interpretability (matching y's units). However, in classical ML settings, you'll often see **MSE and MAE** used as well.

### 3.7 Practical Tip
- Since there's no universal "best" metric, **calculate MAE, MSE, and RMSE together** for any regression problem and compare — this gives a fuller picture of model performance from different angles. Sometimes optimizing for one metric might make another worse, so keep an eye on all of them, especially when tuning models (e.g., during plotting/visualization workflows).

---

## 4. R² Score (Coefficient of Determination)

### 4.1 Motivating Idea — The "Mean Baseline"

Imagine you have a dataset with CGPA and Package, but suppose you **didn't have the CGPA column** at all — only the Package values.
- If someone asked "what package will I get?", your **best possible guess**, with no other information, would be the **mean of all past packages** (e.g., "3.4 LPA on average").
- This gives you a **horizontal "mean line"** — a naive baseline prediction that ignores any input feature entirely.

Now, since you *do* have CGPA, you can do better: fit a **regression line** using CGPA as the input, and use that instead.

**R² Score essentially answers: "How much better is my regression line compared to just using the naive mean line?"**

### 4.2 Formula

$$R^2 = 1 - \frac{SS_{res}}{SS_{mean}}$$

Where:
- $SS_{res}$ = **Sum of Squared Residuals** (errors) of your **regression line**:
$$SS_{res} = \sum_{i=1}^{n} (y_i - \hat y_i)^2$$
- $SS_{mean}$ = **Sum of Squared Errors** of the naive **mean line**:
$$SS_{mean} = \sum_{i=1}^{n} (y_i - \bar y)^2$$

(Note: $SS_{res}$ is literally the numerator of MSE, without the $1/n$ division — same idea.)

### 4.3 Interpreting the Formula

- **In summary:** compare how much error your regression line makes vs. how much error the naive mean line makes.

**Case 1: $R^2 = 0$**
- This happens when $SS_{res} = SS_{mean}$, i.e., your regression line makes **exactly as many errors as** the naive mean line.
- Interpretation: your input feature (e.g., CGPA) is **useless** — the model gains nothing from using it. This is a **very bad model**.

**Case 2: $R^2 = 1$**
- This happens when $SS_{res} = 0$, i.e., your regression line makes **zero errors** — it passes through every single data point perfectly.
- In practice, this is rarely achievable for real (noisy) data, but it represents the **ideal/perfect** scenario.

**General Rule:**
- The closer $R^2$ is to **1**, the better your model is doing (it explains more of the variance than the naive mean baseline).
- The closer $R^2$ is to **0** (or below), the worse your model is doing.

### 4.4 Can R² Be Negative?

Yes — this is possible, though uncommon.
- This happens when $SS_{res} > SS_{mean}$, meaning your regression line is actually making **more errors than the simple mean line would** — i.e., your model is performing *worse* than just guessing the average every time!
- This typically indicates a **fundamental modeling mistake** — e.g., applying a **linear model to highly non-linear data**, where a straight line is a poor fit and does worse than the naive baseline.

### 4.5 A Deeper Interpretation — "Variance Explained"

If $R^2 = 0.8$ for a model predicting Package from CGPA:
> **"80% of the variance in the output (Package) column is being explained by the input (CGPA) column. The remaining 20% is unexplained — it could be due to factors like a good/bad interview, company-specific quirks, luck, etc., that we simply don't have data for."**

- If you added another meaningful input feature (e.g., IQ) and $R^2$ increased to say, 0.85 — that means: "85% of the variance in Package is now explained jointly by CGPA and IQ."
- **General rule of thumb:** higher $R^2$ = more of the variance is being explained by your inputs = generally a good sign, though this **always depends on the specific dataset/domain context**. There's no fixed universal "good" threshold — a particular $R^2$ value might be great in one context and poor in another. Domain context matters a lot when interpreting R².

---

## 5. Adjusted R² Score

### 5.1 The Problem with Plain R²

Here's a subtle flaw in R²: **adding more input columns to your model can artificially inflate R², even if those new columns are completely irrelevant/useless.**

**Example:**
- Say you have CGPA → Package, and R² = 0.80.
- Now you add IQ as a second input. If IQ is genuinely useful, R² might rise to something like 0.85 (which is fine and expected).
- But suppose instead you add a **completely meaningless column**, like "Temperature on the day of the interview" — this shouldn't logically affect Package at all.
- **The problem:** even adding this irrelevant column can cause R² to **increase (or at least not decrease)** — when ideally it should stay the same or decrease, since the column adds no real value. This happens purely due to the mathematics of how more parameters/features can slightly reduce the residual sum of squares just by chance, even without any real relationship.

This flaw makes plain R² **unreliable** when you're comparing models with different numbers of input features — hence, **Adjusted R²** was introduced to fix it.

### 5.2 Formula

$$\text{Adjusted } R^2 = 1 - \left[\frac{(1 - R^2)(n - 1)}{n - k - 1}\right]$$

Where:
- $n$ = number of rows (data points/students)
- $k$ = number of independent/input columns (e.g., k=1 for just CGPA; k=2 for CGPA + IQ; k=3 for CGPA + IQ + Temperature)

### 5.3 How the Formula "Fixes" the Problem

The formula is designed so that adding a **useless** column gets penalized:

**Case A: Adding a useless column (e.g., Temperature)**
- $k$ (number of input columns) **increases**.
- Since $k$ increases, the denominator $(n - k - 1)$ **decreases**.
- A smaller denominator (with the same or barely changed numerator) makes the whole fraction $\frac{(1-R^2)(n-1)}{n-k-1}$ **increase**.
- Since this whole fraction is being **subtracted from 1**, a bigger fraction means **Adjusted R² decreases** — correctly penalizing you for adding a useless feature.

**Case B: Adding a genuinely useful column (e.g., IQ)**
- $k$ increases (denominator decreases, pushing the fraction up) — **but** $R^2$ itself also increases meaningfully (since the new feature actually explains more variance), which makes $(1-R^2)$ **decrease significantly** — enough to outweigh the denominator's effect.
- Net result: the fraction **decreases overall** (because the numerator's drop is much bigger than the denominator's drop) → so **Adjusted R² goes up** — correctly rewarding you for a useful feature.

### 5.4 Key Takeaway
> **Adjusted R² increases only when you add a genuinely useful feature, and decreases (or stays flat) when you add a useless/irrelevant one — unlike plain R², which can misleadingly increase either way.**

### 5.5 When to Use It
- Especially useful/reliable when working with **Multiple Linear Regression** (many input features).
- **Best practice:** always calculate **both R² and Adjusted R²**. If there's a big gap between the two, it's a signal that R² might be giving a misleading/inflated picture, and you should trust Adjusted R² more in that case.

---

## 6. Code Walkthrough (Using the CGPA–Package Dataset)

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# y_test = actual values, y_pred = model's predictions

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)          # no direct rmse function; just take sqrt of mse
r2 = r2_score(y_test, y_pred)

print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)
```

**Example results from the lecture (CGPA → Package model):**
- MAE, MSE calculated directly.
- RMSE ≈ derived by taking `np.sqrt(mse)`.
- R² Score ≈ 0.78 → meaning ~78% of the variance in Package is explained by CGPA alone.

### 6.1 Manually Computing Adjusted R²

```python
r2 = r2_score(y_test, y_pred)
n = X_test.shape[0]      # number of rows
k = X_test.shape[1]      # number of input columns

adjusted_r2 = 1 - ((1 - r2) * (n - 1) / (n - k - 1))
print("Adjusted R2:", adjusted_r2)
```
- With only 1 input column (CGPA), Adjusted R² comes out **very close to plain R²** (e.g., 0.77 vs 0.78) — since with very few features, the penalty term barely has any impact.

### 6.2 Demonstrating the Flaw — Adding a Useless Column

The lecturer demonstrates this live:
1. **Add a random, meaningless column** (e.g., random numbers with no relation to Package) to the dataset.
2. Retrain the model with this new column included.
3. **Result:** R² actually goes up **very slightly** (e.g., from 0.78 → 0.782) — even though the random column has zero real explanatory power.
4. **However, Adjusted R² goes down** (e.g., from 0.774 → 0.760) — correctly reflecting that this addition didn't actually help, and penalizing the wasted "feature slot."

### 6.3 Demonstrating with a Genuinely Useful Column
1. Add a meaningful column, e.g., an "IQ" score that plausibly affects Package.
2. **Result:** R² increases more noticeably (e.g., a bigger jump), since this feature genuinely helps explain variance in Package.
3. **Adjusted R² also increases slightly** — confirming this was a worthwhile addition (unlike the random-column case).

> **Takeaway from the demo:** This is exactly the diagnostic value of Adjusted R² — comparing it against plain R² tells you whether a newly added feature is genuinely useful or just adding noise.

---

## 7. Key Takeaways (Summary Table)

| Metric | Formula | Units | Differentiable? | Sensitive to Outliers? | Best Use Case |
|---|---|---|---|---|---|
| **MAE** | $\frac{1}{n}\sum \lvert y_i - \hat y_i \rvert$ | Same as y | ❌ No (corner at 0) | Relatively robust | Easy interpretability, robust data |
| **MSE** | $\frac{1}{n}\sum (y_i - \hat y_i)^2$ | y² (squared) | ✅ Yes | Very sensitive | When optimization/differentiation is needed |
| **RMSE** | $\sqrt{MSE}$ | Same as y | ✅ Yes | Very sensitive | Most popular overall — combines interpretability + differentiability |
| **R² Score** | $1 - \dfrac{SS_{res}}{SS_{mean}}$ | Unitless (0 to 1, can go negative) | — | — | Understanding "% variance explained"; comparing to naive mean baseline |
| **Adjusted R²** | $1 - \dfrac{(1-R^2)(n-1)}{n-k-1}$ | Unitless | — | — | Multiple Linear Regression; detecting useless added features |

### Quick Mental Models
- **MAE/MSE/RMSE** all answer: *"On average, how wrong is my model?"* — just measured in slightly different ways (absolute vs squared vs square-rooted).
- **R²** answers: *"How much better am I doing than simply guessing the average every time?"*
- **Adjusted R²** answers: *"...and does that improvement actually justify the extra input features I added, or am I just gaming the metric?"*

### General Practical Advice
- Always compute **all metrics together** rather than relying on just one — context (domain, dataset, outliers, number of features) determines which one matters most in a given situation.
- For simple/single-feature regression, plain R² is usually fine.
- For multiple-feature regression, **always check Adjusted R² alongside R²** to catch misleading inflation from irrelevant features.
