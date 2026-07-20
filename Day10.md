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
```
Visually plots the distribution — gives a rough/intuitive idea of how close it is to a bell curve (normal shape) vs. skewed.

Method 2: Skewness (skew())

```python
df['column_name'].skew()
```
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

## Square Transform (continued) & Square Root Transform

- **Square Transform (`x²`)** — used for a certain type of data (the specific use-case wasn't strongly emphasized in the continuation either — the creator notes they personally haven't found heavy real-world use for it in their own experience, though it's worth experimenting with).
- **Square Root Transform (`√x`)** — similarly, not something used very heavily by the creator personally, but still worth trying.

### Practical guidance on choosing a transform
- You do have **some rules of thumb**: e.g., if data is **right-skewed**, try **Log Transform**; if **left-skewed**, try **Square Transform** (this pairing was implied from context, though not stated with full certainty in the audio).
- However, in practice, the best approach is: **just try all of them and see which gives the best result** — this is exactly why sklearn provides tools like `FunctionTransformer`, since trying multiple transforms is just a one-line code change each time.

**Goal of today's example:** Demonstrate applying **4 different transforms** using the `FunctionTransformer` class, and compare model accuracy before and after each transform.

---

## Code Example: Titanic Dataset

### Columns used
Only 3 columns from the Titanic dataset this time:
- `Age`
- `Fare`
- `Survived` (target)

All other columns are ignored for this example.

### Imports

```python
import numpy as np
import pandas as pd
import seaborn as sns          # for Q-Q plots / distribution plots
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
Two models used for comparison: Logistic Regression and Decision Tree Classifier — deliberately chosen together, since one is a linear/statistical model (sensitive to data distribution) and the other is tree-based (not sensitive to data distribution) — useful for showing the contrast in how transforms affect each.
Step 1: Load data and select only 3 columns

df = pd.read_csv('train.csv')
df = df[['Age', 'Fare', 'Survived']]
Step 2: Handle missing values
The Age column has missing values (127 missing, in this case).
Since transformations require complete data (no missing values allowed), missing values must be handled first.

df['Age'].fillna(df['Age'].mean(), inplace=True)
Simple fix: filled missing Age values with the column mean using pandas' fillna().
Step 3: Extract X and y, then Train-Test Split

X = df.iloc[:, 0:2]   # Age, Fare
y = df.iloc[:, -1]    # Survived

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
As with every feature engineering technique, train-test split is done first, before any transformation.
Checking Normality of Age and Fare (Before Any Transform)
Using two methods: distribution plot (PDF) and Q-Q plot.

Age column

sns.distplot(X_train['Age'])
# Q-Q plot for Age
import scipy.stats as stat
import pylab
stat.probplot(X_train['Age'], dist='norm', plot=pylab)
The distribution looked reasonably close to a bell curve — fairly normal, though not perfectly so.
The Q-Q plot showed most points staying close to the reference line, with slight deviation in a couple of places — confirming: not perfectly normal, but reasonably close to normal distribution.
Fare column

sns.distplot(X_train['Fare'])
stat.probplot(X_train['Fare'], dist='norm', plot=pylab)
The distribution was clearly NOT normal at all — strongly right-skewed.
Makes intuitive sense: in the real Titanic dataset, a few passengers paid very high fares while most paid relatively low fares — creating a long tail toward higher values (right skew).
The Q-Q plot confirmed this — points deviated heavily from the reference line, clearly showing a right-skewed, non-normal distribution.
Conclusion: Since Fare is clearly right-skewed, a Log Transform is a good candidate to try on it.

Baseline: Model Performance WITHOUT Any Transformation

clf = LogisticRegression()
clf2 = DecisionTreeClassifier()

clf.fit(X_train, y_train)
clf2.fit(X_train, y_train)

y_pred = clf.predict(X_test)
y_pred2 = clf2.predict(X_test)

accuracy_score(y_test, y_pred)    # Logistic Regression: ~64%
accuracy_score(y_test, y_pred2)   # Decision Tree: ~68.7%
Baseline results (no transformation applied):

Logistic Regression: ~64%
Decision Tree: ~68.7%
Applying Log Transform

trf = FunctionTransformer(func=np.log1p)

X_train_transformed = trf.fit_transform(X_train)
X_test_transformed = trf.transform(X_test)
FunctionTransformer parameters
First parameter (func): the mathematical function to apply.
Can be a built-in function like np.log, np.log1p, np.sqrt, np.reciprocal, etc.
Can also be your own custom function.
np.log vs np.log1p — key difference
np.log(x) → simply applies log to each value. Problem: breaks/gives error (or -inf) if any value is 0, since log(0) is undefined.
np.log1p(x) → computes log(1 + x) — adds 1 to every value before taking the log, guaranteeing no value is ever exactly 0 going into the log function. This is why log1p is generally preferred over plain log.
If your data has no zero values at all, plain np.log can also be used safely.
Note: in this example, log transform was applied to both Age and Fare columns together (not just Fare) in this first attempt.

Train & evaluate models on log-transformed data

clf = LogisticRegression()
clf2 = DecisionTreeClassifier()

clf.fit(X_train_transformed, y_train)
clf2.fit(X_train_transformed, y_train)

y_pred = clf.predict(X_test_transformed)
y_pred2 = clf2.predict(X_test_transformed)

accuracy_score(y_test, y_pred)     
accuracy_score(y_test, y_pred2)   
Result after log transform (both Age & Fare):

Logistic Regression: noticeably improved
Decision Tree: almost no change — as expected, since tree-based models don't care about data distribution
Why Logistic Regression improved but Decision Tree didn't
Confirms the core theory discussed earlier: Logistic Regression (a statistical/linear model) benefits from data being closer to a normal distribution, so log transform helped it.
Decision Tree doesn't care about the distribution shape of the input data, so the transform had negligible effect on it — consistent with expectations.
Double-checking with Cross Validation
Since a single train-test split accuracy improvement can sometimes be due to random chance (not a fully reliable signal), cross-validation was used to verify:


cross_val_score(LogisticRegression(), X_train_transformed, y_train, cv=10, scoring='accuracy').mean()
cross_val_score(DecisionTreeClassifier(), X_train_transformed, y_train, cv=10, scoring='accuracy').mean()
Ran 10-fold cross-validation (splitting/training 10 different times and averaging results) for a more reliable estimate than a single split.
Logistic Regression: ~65% (average across 10 folds)
Decision Tree: ~67% (average across 10 folds)
Conclusion: the improvement direction held up under cross-validation too — Logistic Regression benefited more consistently from the transformation.

Visual confirmation via Q-Q plots (before vs after)

# Fare — before log transform
stat.probplot(X_train['Fare'], dist='norm', plot=pylab)

# Fare — after log transform
stat.probplot(X_train_transformed['Fare'], dist='norm', plot=pylab)
Fare before: far from the normal-distribution reference line (heavily right-skewed).
Fare after log transform: points moved noticeably closer to the reference line — confirming the transform pushed it toward normality, which explains the accuracy improvement.

# Age — before vs after log transform
Age before: was already reasonably close to normal.
Age after log transform: interestingly became slightly worse (moved a bit further from normal) — since Age was already fairly close to normal to begin with, forcefully applying log transform on it wasn't necessary and slightly hurt its distribution shape.
Refinement: Apply Log Transform ONLY on Fare (not Age)
Since Age didn't need the transform (it was already close to normal) and applying log to it actually made things slightly worse, the next step was to apply the log transform only to Fare, leaving Age untouched — using ColumnTransformer:


trf2 = ColumnTransformer([
    ('log', FunctionTransformer(np.log1p), ['Fare'])
], remainder='passthrough')

X_train_transformed2 = trf2.fit_transform(X_train)
X_test_transformed2 = trf2.transform(X_test)
Only Fare goes through the log transform; Age is passed through unchanged (remainder='passthrough').
Result after transforming only Fare

cross_val_score(LogisticRegression(), X_train_transformed2, y_train, cv=10, scoring='accuracy').mean()
cross_val_score(DecisionTreeClassifier(), X_train_transformed2, y_train, cv=10, scoring='accuracy').mean()
Logistic Regression: ~365%... (likely a transcription/spoken slip — intended as ~65%, consistent with earlier result) — essentially the same improved result as before (transforming just Fare was enough to get the full benefit).
Decision Tree: still no meaningful change, as expected.
Key takeaway from this refinement:

Since transforming Age didn't help (and slightly hurt), and transforming just Fare gave the same improvement as transforming both columns, the better/leaner approach is to transform only the columns that actually need it (i.e., only Fare here) — avoiding unnecessary computation and avoiding potentially harming columns that don't need transforming.

Summary of what was learned from the Log Transform experiment:

Fare was clearly right-skewed and far from normal → Log Transform helped bring it closer to normal → Logistic Regression accuracy improved.
Age was already close to normal → applying Log Transform on it was unnecessary and slightly worsened its distribution.
Decision Tree accuracy remained unaffected throughout, regardless of transformation — confirming tree-based models don't care about input data distribution, only linear/statistical models like Logistic Regression do.
Trying the Other Transforms (Reciprocal, Square, Square Root) on Fare
A helper function was created to streamline testing different transforms quickly:


def apply_transform(func):
    trf = FunctionTransformer(func=func)
    X_train_transformed = trf.fit_transform(X_train)
    X_test_transformed = trf.transform(X_test)
    
    clf = LogisticRegression()
    clf.fit(X_train_transformed, y_train)
    y_pred = clf.predict(X_test_transformed)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    
    # Q-Q plot before and after
    plt.figure(figsize=(14,4))
    plt.subplot(1,2,1)
    stat.probplot(X_train['Fare'], dist='norm', plot=pylab)
    plt.subplot(1,2,2)
    stat.probplot(X_train_transformed['Fare'], dist='norm', plot=pylab)
This function returns two things each time: the resulting accuracy, and a before/after Q-Q plot comparison.

Testing "no transform" (identity function) as a baseline check

apply_transform(lambda x: x)   # essentially no transformation
Confirmed baseline accuracy matches the original (no-transform) result — sanity check that the helper function works correctly.
Testing Reciprocal Transform (1/x)

apply_transform(np.reciprocal)
Result: accuracy dropped slightly (e.g., down to ~64%, worse than the log-transformed ~65-68% range).
Q-Q plot: after reciprocal transform, the distribution actually looked worse (further from normal) than before.
Conclusion: Reciprocal transform did not help for this particular Fare distribution — reinforces the point that not every transform works on every dataset; it depends on the specific data's shape. Reciprocal tends to work better on certain right-skewed data, but wasn't a good fit here.
Testing Square Transform (x²)

apply_transform(lambda x: x**2)
Result: also not helpful — no meaningful improvement over baseline for this data.
Testing Square Root Transform (√x)

apply_transform(np.sqrt)
Result: performed better than reciprocal and square, but still not as good as the Log Transform result.
Confirms: for this particular Fare distribution, Log Transform remains the best-performing choice among all the ones tried.
Testing a Custom Transform

apply_transform(lambda x: 1 / (x + 0.001))
A custom reciprocal-like function was tried, adding a small constant (0.001) to the denominator — done specifically to avoid division-by-zero errors in case any value is exactly 0.
Result: still no real improvement — didn't outperform the log transform.
Testing another custom function (sine)

apply_transform(np.sin)
Applying np.sin reduced the accuracy further — but visually, it made the Fare distribution look somewhat "weird"/different from before.
Point being made: you can pass literally ANY custom function into FunctionTransformer — it's fully flexible. The choice of function is entirely up to experimentation.
Key Takeaways / Revision Points
Mathematical Transformations aim to convert skewed/non-normal data distributions closer to a Normal Distribution.
This matters mainly for linear/statistical models (Linear Regression, Logistic Regression) — they tend to perform better on normally distributed data. Tree-based models (Decision Tree, Random Forest) are unaffected by data distribution.
Three ways to check normality:
Distribution plot (sns.distplot)
Skewness value (series.skew()) — closer to 0 = more normal
Q-Q Plot (most reliable) — the closer points hug the diagonal reference line, the more normal the data is
FunctionTransformer (from sklearn.preprocessing) lets you apply any mathematical function (built-in or custom) to your data:

FunctionTransformer(func=your_function)
Log Transform (np.log or np.log1p):
Best suited for right-skewed data
Cannot be applied to negative values (log undefined for negatives)
np.log1p (= log(1+x)) preferred over plain np.log to safely handle zero values
Converts an additive scale into a multiplicative scale — squashes large gaps between big values, making the distribution appear more even/normal
Reciprocal Transform (1/x): inverts large and small values; didn't help in this specific example, but can help on other right-skewed distributions depending on shape
Square Transform (x²) and Square Root Transform (√x): power-based transforms; less commonly impactful in the creator's personal experience, but worth trying
No fixed rule guarantees which transform will work best — the recommended approach is: just try all of them (they're cheap, one-line changes) and pick whichever gives the best validated result — this is exactly the convenience FunctionTransformer provides
Always cross-validate results (not just a single train-test split) before concluding a transform genuinely helped — single-split improvements can sometimes be due to randomness
Apply transforms only to the columns that actually need them — forcing a transform on an already-near-normal column (like Age here) can slightly hurt rather than help; use ColumnTransformer to target specific columns only
You can pass any custom function (lambda or named function) into FunctionTransformer — full flexibility to experiment beyond the standard built-in transforms
