# Power Transformer

Continuing the **Mathematical Transformations** topic. Previously covered: `FunctionTransformer` (Log Transform, Reciprocal Transform, Square/Square Root Transform).

Today's new sklearn class: **`PowerTransformer`**, covering two new mathematical transforms:

1. **Box-Cox Transform**
2. 
3. **Yeo-Johnson Transform**

> Note: There's also a `QuantileTransformer` in sklearn, but it's not covered in this series since it's not commonly used in practice.

---

## Box-Cox Transform

### In Simple Words
Box-Cox is basically a **"smart formula machine"** — instead of you manually guessing whether to use log, square root, square, etc. on your column, Box-Cox automatically tests out a bunch of possible "power" values and picks whichever one makes your data look **most normally distributed**.

### The Formula (simplified)
Box-Cox applies a transformation on your column that depends on a value called **lambda (λ)** — basically, it raises your data to some power `λ`. Depending on what `λ` turns out to be:
- λ = 0 → it behaves like a **Log Transform**
- λ = 0.5 → it behaves like a **Square Root Transform**
- λ = 2 → it behaves like a **Square Transform**
- ...and so on for any value in between

**So Box-Cox is actually a "general/parent" transform** — Log Transform, Square Transform, Square Root Transform are all just *special cases* of Box-Cox, for specific λ values.

### How does it pick the best λ?
- Box-Cox tries out a range of λ values (typically between **-5 and +5**) — e.g., 1.5, 1.6, 1.7, 1.75, and so on.
- For each λ value, it checks: "if I transform the data using this λ, how close does it get to a normal distribution?"
- It then **automatically selects the λ that gives the best/closest result to a normal distribution**.

### How does it actually calculate the "best" λ? (just for awareness, not deeply needed)
Two statistical techniques are used internally:
1. **Maximum Likelihood** (a concept you'll encounter later, especially when studying Logistic Regression)
2. **Bayesian Statistics** (a completely separate branch of statistics with its own set of tools for this kind of problem)

> As a data scientist/ML practitioner, you don't need to know the deep internal math of *how* it calculates this — you just need to know *what* it does and *when* to use it. Sklearn handles the calculation for you.

### Important Restriction of Box-Cox
**Box-Cox only works on strictly POSITIVE numbers.**
- Data containing **0** → won't work
- Data containing **negative numbers** → definitely won't work

This is Box-Cox's main limitation — and it's exactly the problem that **Yeo-Johnson** was created to solve.

---

## Yeo-Johnson Transform

### In Simple Words
Yeo-Johnson is basically **"Box-Cox, but upgraded to also handle zero and negative numbers."** Two computer scientists (Yeo and Johnson) built this as an improved/extended version of Box-Cox specifically to remove Box-Cox's biggest limitation.

- Works on **positive numbers** (like Box-Cox)
- **Also works on 0 and negative numbers** (which Box-Cox cannot handle)

That's really the entire difference between the two — same core idea (find the best λ to make data normal), but Yeo-Johnson is more flexible about what kind of numbers it can accept.

### Bottom line
> If your data has only positive values → either Box-Cox or Yeo-Johnson will work.
> If your data has zero or negative values → you **must** use Yeo-Johnson (Box-Cox will fail/error out).

---

## How to Use This in Practice

You don't need to manually decide between Box-Cox and Yeo-Johnson or manually calculate λ — sklearn's **`PowerTransformer`** class handles both, and calculates the best λ for you automatically.

**General strategy:** If you're using an algorithm that performs better on normally distributed data (like Linear Regression, Logistic Regression), and your data isn't normally distributed:

- Just try applying `PowerTransformer` (Box-Cox or Yeo-Johnson) and check if your model's accuracy improves.
  
- This is basically a form of experimentation — try it, measure the result, keep it only if it helps.

---

## Code Example: Concrete Strength Dataset

### Dataset

A concrete material dataset (downloaded from Google/Kaggle) — a **regression problem**: predicting concrete strength based on its ingredients.

Input columns (ingredients): Cement, Blast Furnace Slag, Fly Ash, Water, Superplasticizer, Coarse Aggregate, Fine Aggregate, Age.

Target: Concrete compressive strength.

### Step 1: Imports

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stat
import pylab

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import PowerTransformer
```
scipy.stats and pylab are used for plotting Q-Q plots — to visually check how close a column's distribution is to normal.

Step 2: Load data and basic checks

```python
df = pd.read_csv('concrete_data.csv')
df.shape
df.isnull().sum()   # checked for missing values — none found
```
Checked shape (rows/columns look fine).

Checked for missing values — none found (transformations require complete/non-missing data).

Also checked whether any column has negative or zero values, since this determines whether Box-Cox can be used. Found: a couple of columns (like Blast Furnace Slag, Superplasticizer) have a minimum value of 0 — this becomes relevant later.

Step 3: Extract X and y, then Train-Test Split

```python
X = df.drop(columns=['Strength'])
y = df['Strength']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
Step 4: Baseline — Linear Regression WITHOUT any transformation

```python
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

r2_score(y_test, y_pred)   # ~0.62 (62%)
```
Baseline R² score: ~0.62

Also ran 10-fold cross-validation to double-check reliability:

cross_val_score(LinearRegression(), X_train, y_train, cv=10, scoring='r2').mean()

Result: slightly worse than the single-split value — around 0.46 — confirming the single-split score wasn't fully reliable on its own.

Step 5: Visualizing why performance isn't great (Q-Q plots for each column)

Plotted Q-Q plots for every input column (except the target) to see how far each is from normal:

Cement: reasonably close to a bell shape, no major issue

Blast Furnace Slag: clearly right-skewed

Fly Ash: okay, not too bad

Water: fairly decent

Superplasticizer: clearly right-skewed

Coarse Aggregate: looks fine

Fine Aggregate: looks fine

Age: worst of all — heavily right-skewed. Makes sense: many concrete samples have low age (recently made) while a few samples have very high age — creating a strong right-skew, similar to the Fare column example from the Titanic dataset in the previous note.

Conclusion: Since Age (and a couple of other columns) are far from normal, this is likely dragging down the Linear Regression model's performance — a good candidate for applying transformations.

Note: One thing intentionally not done here: no scaling was applied before/after — since the plan is to compare "without transformation" vs. "with transformation" fairly, and scaling would affect both cases equally anyway, it was skipped to keep the comparison clean and simple.

Applying Box-Cox Transform
Step 1: Create and apply the PowerTransformer (Box-Cox mode)

```python
pt = PowerTransformer(method='box-cox')

X_train_transformed = pt.fit_transform(X_train + 0.000001)  # small offset added
X_test_transformed = pt.transform(X_test + 0.000001)
```
Handling the zero-value problem

Box-Cox cannot handle 0 values (found earlier that some columns have a minimum of 0).

Workaround used: added a very tiny constant (0.000001) to the entire dataset before applying the transform — this nudges any 0 value to something just barely above 0, without meaningfully changing the actual data. (This trick was seen referenced in an online blog/article as a common workaround.)

Understanding what λ values got picked

pt.lambdas_

This returns the λ (lambda) value calculated for each column — remember, since we have 8 input columns, Box-Cox calculates a separate, best-fit λ for each individual column.

Example interpretation: if Cement's λ came out to be 0.170, that means each value in the Cement column effectively got raised to the power 0.170 as part of its transformation (i.e., cement_value ^ 0.170).

Similarly, Blast Furnace Slag's λ might come out as 0.021, meaning its values get raised to ^0.021.

Core idea to remember: each column gets its own custom "best power" (λ), and that column's values are transformed using that power — this is literally how the transform is applied under the hood.

Step 2: Train and evaluate model on Box-Cox transformed data

```python
lr = LinearRegression()
lr.fit(X_train_transformed, y_train)
y_pred = lr.predict(X_test_transformed)

r2_score(y_test, y_pred)   # improved to ~0.80
```

R² score improved from ~0.62 → ~0.80 — a solid improvement.
Double-checked with cross-validation again

cross_val_score(LinearRegression(), X_train_transformed, y_train, cv=10, scoring='r2').mean()

Result: improved from ~0.46 → ~0.60 — again confirming a genuine, reliable improvement (not just a lucky single split).

Step 3: Visual confirmation — Q-Q plots before vs. after
Plotted before/after Q-Q plots side by side for each column:

Cement: was slightly right-skewed before, now looks noticeably more normal (bell-shaped) after transform.
Blast Furnace Slag: still somewhat skewed even after transform, but couldn't be made perfectly normal — still, both before and after are "not fully normal," so no visible harm either.
Water: looked the same before and after — no meaningful change.
Superplasticizer: improved slightly.
Coarse Aggregate: looked the same, no meaningful change.
Fine Aggregate: looked the same, no meaningful change.
Age: the biggest improvement of all — went from a strongly right-skewed, clearly non-normal distribution to something that now looks close to a proper bell curve / normal distribution.
Conclusion: The improvement in R² score lines up directly with the visual improvement in the distributions — especially Age, which was the worst column before and got fixed the most.

Applying Yeo-Johnson Transform

pt2 = PowerTransformer()   # default method is 'yeo-johnson'

X_train_transformed2 = pt2.fit_transform(X_train)
X_test_transformed2 = pt2.transform(X_test)
PowerTransformer's default method is already 'yeo-johnson' — so if you don't specify method, you automatically get Yeo-Johnson.
Key advantage used here: unlike Box-Cox, no need to add that tiny +0.000001 offset trick — Yeo-Johnson handles 0 (and negative) values natively, without any workaround.
Results

lr.fit(X_train_transformed2, y_train)
y_pred = lr.predict(X_test_transformed2)
r2_score(y_test, y_pred)  # ~0.80, similar to Box-Cox
Single-split R²: similar result (~0.80), comparable to Box-Cox.
Cross-validation result: went from ~0.46 (baseline) → ~0.60ish, marginally better than or similar to Box-Cox — a genuinely good improvement over baseline either way.

Q-Q plots before vs. after (Yeo-Johnson)
Visually compared before/after distributions for all columns.
The differences looked broadly similar to Box-Cox's results overall — most columns showed no major change, but Age (and to a lesser extent a couple of others like Superplasticizer) again showed the clearest, biggest improvement toward a normal-looking distribution.
Since none of the columns had negative values in this dataset, Yeo-Johnson's extra ability to handle negatives wasn't really needed/tested here — the comparison ended up fairly similar to Box-Cox in this particular dataset.
Summary Table of Results (this dataset)
Approach	R² (single split)	R² (10-fold CV average)
No transformation	~0.62	~0.46
Box-Cox Transform	~0.80	~0.60
Yeo-Johnson Transform	~0.80	~0.60ish (similar/slightly better)
Both transforms gave a clear, meaningful improvement over the baseline (no transformation), especially by fixing the heavily right-skewed Age column.









