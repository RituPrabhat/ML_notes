# Handling Missing Data — Notes (Part 1: Complete Case Analysis)

## Why Handle Missing Data?

Continuing with Feature Engineering. New topic: **How to Handle Missing Data**.

**Core reason this matters:** Machine learning algorithms generally **do not work well with missing data**. If you feed a dataset with missing values into most ML algorithms, they simply **cannot train** — almost all ML algorithms are **not capable of handling missing data** on their own.

**Data scientist's responsibility:** Before training any model, you must remove/handle all missing values in your data.

This topic will be covered across **5 videos/notes**, each covering a different technique.

---
## Two Broad Options When You Have Missing Values

### Option 1: Remove the row entirely
If any column in a particular row has a missing value, you simply **delete that entire row**.

**Example:** Suppose you have a dataset with 4 columns (3 input + 1 output) and 5 rows. If row 3 has a missing value in the first column, you just delete row 3 entirely — leaving you with 4 rows to work with.

- This is the simplest technique, but **not very commonly used/recommended** in practice.
- **Why not recommended:** when you delete a row because ONE column had a missing value, you're also throwing away the perfectly good data in the OTHER columns of that same row — potentially wasting a lot of useful information.

### Option 2: Fill in (Impute) the missing value
Instead of deleting data, you fill in ("impute") the missing value with some reasonable estimate.

Two broad imputation approaches:
1. **Univariate imputation** — when filling in a missing value, you only look at that **one column** in isolation (ignore all other columns).
2. **Multivariate imputation** — you use **multiple columns together** to make a smarter guess about what the missing value should be.

---
## Roadmap for This 5-Part Series

| Video/Note | Topic |
|---|---|
| 1 (this one) | **Removing rows entirely** — Complete Case Analysis (CCA) |
| 2 | **Univariate Imputation** — using sklearn's `SimpleImputer` class |
| 3 | **KNN Imputer** (a multivariate technique) |
| 4 | **Iterative Imputer** / **MICE** (Multivariate Imputation by Chained Equations) — another multivariate technique, uses an ML algorithm internally to predict missing values |
| 5 | **Missing Indicator** — a related concept, briefly touched on |

### Quick preview of Univariate Imputation techniques (for numerical vs categorical data)
- **Numerical missing data** — can be filled using:
  - Mean or Median
  - A random value from the existing distribution
  - "End of Distribution" value (explained in the next video)
- **Categorical missing data** — can be filled using:
  - Mode (most frequent value)
  - Or simply the string `"Missing"`

All of the above (univariate techniques) are handled via sklearn's `SimpleImputer` class (covered in the next note).

### Quick preview of Multivariate Imputation techniques
- **KNN Imputer** — uses a machine learning algorithm (K-Nearest Neighbors) to fill missing values, using information from other columns/rows.
- **Iterative Imputer** (a.k.a. **MICE**) — another ML-based approach, uses a more advanced iterative process, working across multiple columns.

---

## Today's Focus: Complete Case Analysis (CCA)

CCA = the technique of **removing rows** wherever data is missing.

---

## What is Complete Case Analysis (CCA)?

**Definition:** Complete Case Analysis (also called **List-wise Deletion of Cases**) means **discarding any row (observation) that has a missing value in ANY column**, and only keeping rows where **every single column has a value** (i.e., "complete" rows/cases).

- In simple terms: if even ONE column in a row is missing, delete the **entire row**.
- You end up working only with rows that have **complete information across all columns** — hence the name "Complete Case Analysis."

---

## Assumption Required for CCA (Very Important!)

Before applying CCA, there's a critical assumption you must verify: **the missing data must be "Missing Completely At Random" (MCAR)**.

### What does "Missing Completely At Random" (MCAR) mean?

Simple way to think about it: imagine your dataset has 1000 rows and 4 columns, and one column (say, `Age`) has 50 missing values.

- If you apply CCA, you'll delete all 50 rows where `Age` is missing → left with 950 rows.
- **This is only safe to do if those 50 missing values are scattered randomly throughout the dataset** — not concentrated in some meaningful pattern (like all missing values being from the first 50 rows, or the last 50 rows, or all belonging to a specific group/category).
- If the missing values follow some non-random pattern, then deleting them would **distort your dataset** — you'd unintentionally throw away a biased subset of your data, not just "random noise."

**In short:** you should only perform CCA when you're confident that the data you're removing is missing **purely due to random chance**, and not because of some underlying pattern or reason.

**Analogy given:** If data is truly Missing Completely At Random, then removing those 50 rows should feel exactly the same as if you had randomly picked and deleted any 50 rows from your dataset — the overall shape/distribution of your data would remain basically the same either way.

---

## Advantages of CCA

1. **Very easy to implement** — you literally just need to call a "drop" function; no complex data manipulation required.
2. **If the data truly is MCAR, the resulting distribution of your dataset (after removing rows) will closely match the original distribution** — since you're just removing "random noise," not any meaningful signal, the shape of your remaining data barely changes.

> **Important practice:** whenever you perform CCA, you should always double-check by comparing the data's distribution **before** vs. **after** removing the missing rows — if they look almost identical, that's a strong signal that your MCAR assumption held true and it's safe to proceed.

---

## Disadvantages of CCA

1. **Can result in discarding a large fraction of your original dataset.** If missing values are common across many columns, you might end up removing far too much data — a real waste of information.
2. **If the data is NOT actually Missing At Random,** CCA can distort your dataset's real distribution — this can seriously mislead your ML model's training, since your model would be trained on a biased/skewed subset that doesn't represent reality.
3. **The most important disadvantage — Production problem:** When you finally deploy your trained model into a real application/server, **you can't just delete rows with missing values there** — a real user's input might have a missing field, and your model needs to somehow still make a prediction. But since your model was ONLY trained on complete, no-missing-values data, **it has absolutely no idea how to handle missing values when it encounters them in production**. This is the single biggest reason CCA is not heavily used in real-world ML projects.

---

## When Should You Use CCA?

Two conditions must BOTH be satisfied:

1. **The missing data must be Missing Completely At Random (MCAR).** This is the first, non-negotiable requirement.
2. **The percentage of missing data should be low — commonly, less than 5% (per column).**
   - There's no strict universal rule for this threshold, but **5% is a widely used rule of thumb**.
   - If a column has **more than 5%** missing data, CCA (row removal based on that column) is generally **not recommended**.
   - **Special case:** if a particular column has an extremely high percentage of missing values (e.g., 95-98% missing), it usually makes more sense to **drop the entire column** instead of dropping rows — since the column barely has any useful information to begin with anyway.

**Summary rule to remember:**
> Only use CCA when: (1) missing data is random (MCAR), AND (2) the amount of missing data per column is small (typically under 5%).

---

## Code Example: Real-World Dataset (Data Science Jobs)

### Dataset Overview
A dataset about candidates who applied for a data science job — columns include:
- `enrollee_id` — candidate's ID
- `city` — city they're from
- `city_development_index` — development index of that city
- `gender`
- `relevant_experience`
- `enrolled_university` — what kind of university course (if any)
- `education_level`
- `major_discipline`
- `experience` (in years)
- `company_size`
- `company_type`
- `training_hours` — how many hours of data science training they've had
- `target` — 0 = not hired, 1 = hired (this is what we'd eventually want to predict)

### Step 1: Check missing values per column

```python
df.isnull().mean() * 100   # percentage of missing values per column
Columns with no missing values: enrollee_id, city, relevant_experience, target.
```
Columns with missing values (approximate percentages found):

Column	% Missing
gender	~23.1% (highest — a lot missing)
major_discipline	high (significant, mentioned as important-but-missing)
company_size	~30.8%
company_type	~32.2%
city_development_index	~2%
enrolled_university	~2%
education_level	~2%
experience	~0.3% (very low)
training_hours	~4%

Step 2: Decide which columns are safe for CCA

Since the rule of thumb is "less than 5% missing," we should NOT apply CCA based on gender (23.1%), company_size (30.8%), or company_type (32.2%) — doing so would delete way too many rows (potentially 20-30%+ of the entire ~19,000-row dataset).

Columns selected for CCA (all under 5% missing):

city_development_index
enrolled_university
education_level
experience
training_hours

Step 3: Programmatically identify columns under the 5% threshold

cols = [var for var in df.columns if df[var].isnull().mean() < 0.05 and df[var].isnull().mean() > 0]
This automatically extracts column names where missing percentage is greater than 0% but less than 5%.
Useful especially for large datasets with many columns — you don't have to manually eyeball each column's missing percentage.
Running this gave exactly the 5 columns listed above.

Step 4: Check the impact of dropping rows (before committing to it)

new_df = df[cols].dropna()

len(new_df) / len(df)   # what fraction of data survives
Before actually applying CCA, always check how much data you'd lose.
In this case: roughly 89% of the data survived (dataset went from ~19,020 rows to ~17,920 rows) — an acceptable loss, not too drastic.

Step 5: Compare distributions — Numerical Columns (Before vs. After CCA)
For numerical columns (e.g., training_hours, city_development_index, experience), compare histograms before and after CCA to verify the MCAR assumption held.


import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111)

df['training_hours'].hist(bins=50, ax=ax, density=True, color='red')
new_df['training_hours'].hist(bins=50, ax=ax, density=True, color='green', alpha=0.8)
Plotted the old distribution (red) and the new, post-CCA distribution (green) overlapping on the same graph.
Result: The green line almost completely overlapped the red line — in a few small spots, the red peeked out slightly, but overall the shape of the distribution remained essentially the same.
Conclusion: Removing rows with missing training_hours did NOT meaningfully distort the distribution — confirms that this missing data was indeed random (MCAR) — safe to proceed with CCA here.
Same check repeated for city_development_index and experience:

Both showed nearly identical (overlapping) distributions before and after — small, negligible bumps here and there, but overall shape preserved.
Conclusion: All three numerical columns pass the MCAR check — CCA is safe to apply on them.

Step 6: Compare distributions — Categorical Columns (Before vs. After CCA)
For categorical columns (enrolled_university, education_level), instead of histograms, compare the ratio/proportion of each category before vs. after removing rows.


df['enrolled_university'].value_counts() / len(df)
new_df['enrolled_university'].value_counts() / len(new_df)
What to check: the relative proportion of each category should stay roughly the same before and after CCA. If a category's ratio changes drastically, that's a red flag suggesting the missing data wasn't truly random for that column.

enrolled_university results (3 categories: No Enrollment, Full-time Course, Part-time Course):
Category	Before CCA	After CCA
No Enrollment	0.72	0.727
Full-time Course	0.194	~0.19-something
Part-time Course	0.06	0.06-something
Very minor shifts (e.g., 0.72 → 0.727) — negligible change, meaning the ratios were well-preserved. Safe to proceed.
(Rule of thumb mentioned: if a ratio had shifted drastically — e.g., from 0.19 to 0.25 — that would be a red flag suggesting something is wrong and CCA might be distorting this column's data.)
education_level results (5 categories: Graduate, Masters, High School, PhD, Primary School):
Category	Before CCA	After CCA
Graduate	~0.634 (example)	~0.634-ish, stayed close
Masters	~0.223	~0.223, stayed close
High School	same	same
PhD	same	same
Primary School	same	same
All ratios remained essentially unchanged — safe to apply CCA on this column too.
Conclusion for both categorical columns: since the category ratios stayed consistent before/after, this confirms the missing data in these columns was Missing Completely At Random — CCA is a valid approach here.

Final Takeaways / Key Points to Remember
CCA = deleting entire rows wherever any column has a missing value.
Only apply CCA when BOTH conditions hold:
Missing data is Missing Completely At Random (MCAR) — verify this by comparing distributions (histograms for numerical, category ratios for categorical) before and after removing rows. If they look nearly identical, the assumption holds.
Missing percentage is low (commonly under 5% per column) — check this first before even considering CCA on a column.
If a column has an extremely high percentage of missing data (like 95%+), consider dropping the entire column instead of dropping rows.
Even if both conditions are satisfied and CCA seems statistically "safe," there's still a major practical problem: once you deploy your model into production, real users' input may contain missing values — but since your model was trained ONLY on complete data, it has no built-in way to handle missing values at prediction time. This production limitation is the biggest reason CCA is not heavily used in real-world ML pipelines.
Because of this production issue, most practitioners prefer imputation (filling in missing values) over CCA (deleting rows) — which is exactly what the next 4 videos/notes in this series will cover.
