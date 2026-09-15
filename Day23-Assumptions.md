# Assumptions of Linear Regression

---

## 0. Why This Topic Matters

- This is a **small but very important topic** — commonly asked in **interviews**.
- People who can't answer this question properly often get filtered out in interviews.
- There are **5 main assumptions** of Linear Regression (some sources list 6-7, but these 5 are the core/most essential ones):
  1. Linear relationship between input and output
  2. No multicollinearity
  3. Normal distribution of residuals/errors
  4. Homoscedasticity
  5. No autocorrelation of errors

- **Approach:** For each assumption, this lecture covers both the **theory** (what it means, why it matters) and **how to test/detect it in code** using Python.

### Setup Used in the Lecture
- A dataset with **3 input features** and **1 output** column.
- A `LinearRegression` model was already trained on this data.
- All 5 assumptions are then tested one by one, using this trained model.

---

## 1. Linear Relationship Between Input and Output

### 1.1 What It Means
- There should be a **clear (or roughly clear) linear relationship** between each input feature and the output/target.
- This can be:
  - **Positive linear** — as input increases, output also increases.
  - **Negative linear** — as input increases, output decreases.
  - Either direction is fine — what matters is that the relationship is **linear**, not curved/non-linear.

### 1.2 Visual Examples
- **Graph 1:** Clear upward trend → as X increases, Y increases too (with minor natural fluctuation) → **positive linear relationship**. Good.
- **Graph 2:** Clear downward trend → as X increases, Y decreases → **negative linear relationship**. Also good (still linear).
- **Graph 3:** A curved pattern (looks quadratic/squared) — as X increases, Y increases at a much faster, non-constant rate → **NOT a linear relationship**. This violates the assumption.

### 1.3 Applies to EVERY Input Column Individually
- If you have **multiple input columns**, this assumption applies to **each one separately** — i.e., you need to check the relationship between:
  - $X_1$ and $y$
  - $X_2$ and $y$
  - $X_3$ and $y$
  - ...and so on for every feature.
- Each pairwise relationship should individually look roughly linear.

### 1.4 How to Detect This in Code
- **Plot a scatter plot of each individual feature against the target column.**
```python
sns.scatterplot(x=df['feature1'], y=df['target'])
# repeat for feature2, feature3, etc.
```
- **Example finding from the lecture:**
  - Feature 1 → showed a clear linear relationship.
  - Feature 2 → not very clear whether it's linear or not.
  - Feature 3 → showed *some* linear relationship, but weaker/less clear than Feature 1.
- Simply plotting each feature vs. the target immediately tells you whether this assumption holds for that feature.

---

## 2. No Multicollinearity

### 2.1 What It Means
- **None of your input columns should be correlated with each other.**
- Example: if $X_1, X_2, X_3$ are your inputs and $y$ is your target, then $X_1$, $X_2$, and $X_3$ should all be **independent of one another** — none of them should depend on or predict the others.
- If changing $X_1$ causes $X_2$ to also change in some consistent way, that's called **multicollinearity** — and it's something you want to **avoid**.

### 2.2 Why Multicollinearity Is a Problem

**Geometric/mathematical explanation:**
- Linear Regression fits a hyperplane: $y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \beta_3 x_3$
- $\beta_1$ represents: *"how much does $y$ change when $x_1$ changes by a small amount, while $x_2$ and $x_3$ are held constant?"*
- Similarly, $\beta_2$ represents the effect of changing $x_2$ **alone**, keeping others constant.
- **The problem:** if $x_1$ and $x_3$ are related, then you literally **can't change $x_1$ while keeping $x_3$ constant** — changing one automatically causes the other to change too. This breaks the very premise on which $\beta_1$'s meaning depends, making the calculated coefficients unreliable/hard to interpret correctly.

**Intuitive analogy (from the lecture):**
- Imagine two scientists collaborating on a project, and afterward you want to figure out **how much each person individually contributed**.
- If the two scientists come from **completely different backgrounds** (e.g., one is a physicist, one is a chemist), it's relatively easy to separate out their individual contributions.
- But if both scientists have **identical, overlapping skill sets** (e.g., both know physics equally well), it becomes very hard to tell whose contribution was whose — their effects are tangled together.
- **This is exactly the problem multicollinearity causes in Linear Regression** — when input features are correlated with each other, it becomes difficult to isolate and correctly attribute each one's individual effect on the output.

### 2.3 How to Detect Multicollinearity in Code

**Method 1: Variance Inflation Factor (VIF)**
- A well-known, simple technique.
- **Rule of thumb:**
  - VIF values **around 1** → no multicollinearity problem.
  - VIF values **5 or above** → multicollinearity is present for that feature.
- If a feature's VIF is 5+, that feature might need to be **removed**, since whatever information it's contributing is likely already being captured by the other correlated features.

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor

# loop through each column and calculate VIF
for i in range(X.shape[1]):
    vif = variance_inflation_factor(X.values, i)
    print(vif)
```
- **Example result from the lecture:** All three features had VIF values close to 1 → **no multicollinearity** in this dataset.

**Method 2: Correlation Heatmap (a quicker visual check)**
- Calculate correlation between all pairs of input features, and visualize using a heatmap.
```python
sns.heatmap(df.corr(), annot=True)
```
- **What to look for:** Focus specifically on the correlation values **between the input features themselves** (not between inputs and the target) — i.e., Feature1-vs-Feature2, Feature1-vs-Feature3, Feature2-vs-Feature3.
- **Example finding:** All three pairwise correlations were very low/close to zero → suggests multicollinearity is unlikely.
- **Note:** This heatmap method gives you a **rough idea/first impression**, but **VIF is the more reliable, proper technique** to actually confirm multicollinearity.

---

## 3. Normal Distribution of Residuals (Errors)

### 3.1 Recap: What Is a Residual/Error?
- When you fit a regression line/hyperplane, for any given input $x$, there's:
  - The **actual** value from your dataset.
  - The **predicted** value from your regression line/model.
- The **difference between actual and predicted** is called the **residual** (or error).

### 3.2 What This Assumption States
- When you **plot all your residuals**, they should roughly follow a **Normal (Gaussian) Distribution** — the classic "bell curve" shape.
- Most residual values should cluster **close to zero**, with fewer and fewer values as you move further away in either direction (positive or negative).

### 3.3 How to Check This in Code

**Step 1: Calculate residuals for every test point**
```python
y_pred = model.predict(X_test)
residuals = y_test - y_pred
```

**Method A: Plot a Distribution Plot (KDE plot)**
```python
sns.distplot(residuals)   # or sns.kdeplot(residuals)
```
- **Example finding from the lecture:** The residuals' distribution had a small "bump" in it, but apart from that, it roughly resembled a normal distribution — not perfectly ideal, but the assumption roughly holds (a "shortcut/passable" version of following the assumption).

**Method B: Q-Q Plot (Quantile-Quantile Plot)**
- Draw a Q-Q plot on your residuals.
- **Interpretation:** the points should lie roughly **along the diagonal reference line**.
- If the residuals were *perfectly* normally distributed, all points would sit exactly on the line.
- In the lecture's example, the points were reasonably close to the line — not perfect, but good enough to say the residuals are **approximately normal**, satisfying this assumption reasonably well.

---

## 4. Homoscedasticity

### 4.1 What It Means
- **"Homo"** = same, **"scedasticity"** = spread/scatter. So **homoscedasticity = "same/equal spread."**
- This assumption is closely related to residuals: when you plot your residuals, their **spread (variance) should be roughly equal/consistent** across the entire range of predicted values — it shouldn't change or grow/shrink as predictions get larger or smaller.
- If the spread is **not** equal/consistent, this is called **heteroscedasticity**, which violates the assumption (this is what you want to **avoid**).

### 4.2 How to Check This in Code

**Step 1:** Plot predicted values ($\hat y$) on the X-axis, and residuals on the Y-axis (a scatter plot).
```python
plt.scatter(y_pred, residuals)
```

**What a GOOD plot looks like (Homoscedasticity present):**
- The spread of points should be **uniform/consistent across the entire width** of the plot.
- It shouldn't be lopsided or "funnel-shaped" toward one side.

**What a BAD plot looks like (Heteroscedasticity present):**
- The spread of points **visibly changes** as you move along the x-axis — e.g., points are tightly clustered on one side and much more spread out on the other side (a classic "funnel" or "cone" shape).
- If your plot looks like this, your Linear Regression model will **not fully satisfy this important assumption**.

**Example finding from the lecture:**
- The plotted graph didn't show any single, obvious "funnel" pattern — the spread was **mostly uniform**, though not perfectly so.
- Conclusion: this assumption was **reasonably satisfied** for this dataset (again, not perfect, but acceptable).

---

## 5. No Autocorrelation of Error (Residuals)

### 5.1 What It Means
- When you plot all your residual values (e.g., in the order they occur), **no clear pattern should emerge**.
- If a pattern (e.g., a wave-like or trending shape) appears when plotting residuals in sequence, this indicates **autocorrelation** — meaning that residuals are related/correlated with each other in some structured way — which is undesirable.

### 5.2 Positive vs Negative Autocorrelation
- **Positive autocorrelation:** plotting the residuals shows a clear, consistent pattern (e.g., residuals trend together or show a repeating/wave-like shape) — this means there **is** a relationship existing between the residual points, which is **not desirable**.
- **Negative autocorrelation (what you actually want):** the residuals, when plotted, look like a **jumbled/random mess with no discernible pattern** — this indicates the residuals are **not** related to each other, which is the desired outcome.

*(Note: despite the naming, "negative autocorrelation" here effectively just means "no meaningful pattern/relationship exists" — which is what you want. "Positive autocorrelation" is the undesirable case where a real relationship/pattern does exist between consecutive residuals.)*

### 5.3 How to Check This in Code
- Simply **plot your calculated residuals** (e.g., against their index/order) and visually inspect for any pattern.
```python
plt.plot(residuals)
```
- If the plot looks like random noise with no visible structure/pattern → the assumption holds (good).
- If a clear pattern (trend, wave, cycle) is visible → the assumption is violated (bad).

---

## 6. Key Takeaways (Summary Table)

| # | Assumption | What It Checks | How to Detect |
|---|---|---|---|
| 1 | **Linear Relationship** | Each input feature has a roughly linear relationship with the output | Scatter plot: each feature vs. target |
| 2 | **No Multicollinearity** | Input features should be independent of each other (not correlated) | VIF (Variance Inflation Factor — should be ~1, problematic if ≥5); Correlation heatmap (quick check) |
| 3 | **Normal Distribution of Residuals** | Residuals (errors) should roughly follow a bell-curve/Normal distribution | Distribution plot (KDE) of residuals; Q-Q plot (points should lie near the diagonal) |
| 4 | **Homoscedasticity** | Spread of residuals should be consistent/equal across all predicted values (not "funnel-shaped") | Scatter plot: predicted values vs. residuals |
| 5 | **No Autocorrelation of Error** | Residuals shouldn't show any pattern when plotted in sequence | Plot residuals in order; look for absence of pattern |

### Quick Mental Model
Think of these 5 assumptions as a **checklist for whether Linear Regression is even the "right tool" for your data**:
1. Do inputs and output move together in a straight-line way? *(Linear Relationship)*
2. Are your inputs actually independent clues, or are they secretly repeating the same information? *(No Multicollinearity)*
3. Are your mistakes (errors) random and bell-shaped, rather than skewed or unusual? *(Normal Residuals)*
4. Is your model's typical mistake-size consistent everywhere, rather than being much worse in certain ranges? *(Homoscedasticity)*
5. Are your mistakes truly random and independent of each other, with no hidden trend connecting them? *(No Autocorrelation)*

If most/all of these hold reasonably well, your Linear Regression model is on solid statistical footing. If several are badly violated, it's a sign that Linear Regression might not be the ideal choice for that particular dataset, or that some preprocessing/feature engineering is needed first.
