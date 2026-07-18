# Column Transformer 

## Why Column Transformer?

**Problem:** In a real dataset, different columns often need *different* preprocessing:
- Some columns may have missing values
- Some may need scaling
- Some may be categorical (needing encoding)

Handling each column separately (as done in previous notes) is tedious — you end up creating separate transformed arrays for each column, then manually merging them all back together (as seen with the `hstack` approach for One-Hot Encoding). This gets increasingly painful as the number of columns grows.

**Solution: `ColumnTransformer`** — a sklearn class that lets you apply different transformations to different columns and get back a single combined output, all in one step/line.

---

## Example Dataset

Columns:
| Column | Type | Notes |
|---|---|---|
| `age` | Numerical | — |
| `gender` | Nominal Categorical | Male/Female — no order |
| `fever` | Numerical | Has **missing values** (10 out of 100 missing) |
| `cough` | Ordinal Categorical | Mild / Strong — has order |
| `city` | Nominal Categorical | 4 cities: Kolkata, Bangalore, Delhi, Mumbai — no order |

(This is a fictional/practice dataset about patients — not to be taken seriously as real data.)

### Required preprocessing per column
- `age` → numerical, no preprocessing needed (kept as-is for this example)
- `fever` → has missing values → needs **imputation** (e.g., fill with mean)
- `cough` → ordinal → needs **Ordinal Encoding** (Mild < Strong)
- `gender` → nominal → needs **One-Hot Encoding**
- `city` → nominal → needs **One-Hot Encoding**

> Note: In this example, `gender` isn't actually being encoded (left as-is intentionally) — the focus is mainly on `fever`, `cough`, and `city`. Label Encoding for any target column isn't covered here either — this note focuses only on input feature transformation.

---

## Approach 1: The "Hard Way" (Without Column Transformer)

This is the manual, tedious way — done to show *why* Column Transformer is needed.

### Step 1: Train-Test Split (always first)

```python
X_train, X_test, y_train, y_test = train_test_split(...)
```
from sklearn.impute import SimpleImputer

si = SimpleImputer()
X_train_fever = si.fit_transform(X_train[['fever']])
X_test_fever = si.transform(X_test[['fever']])

