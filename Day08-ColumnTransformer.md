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

Step 2: Handle missing values in fever column

```python
from sklearn.impute import SimpleImputer

si = SimpleImputer()
X_train_fever = si.fit_transform(X_train[['fever']])
X_test_fever = si.transform(X_test[['fever']])
```

1) Only the fever column is selected and passed in.
   
2) Missing values get replaced with the column mean.

3) Result: a transformed array with all missing values filled, for just this one column.

Step 3: Ordinal Encoding on cough column

```python
from sklearn.preprocessing import OrdinalEncoder

oe = OrdinalEncoder(categories=[['Mild', 'Strong']])
X_train_cough = oe.fit_transform(X_train[['cough']])
X_test_cough = oe.transform(X_test[['cough']])
```

1)Order specified: Mild < Strong.
2)Again, only this single column is processed at a time.

Step 4: One-Hot Encoding on gender and city columns

```python
from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(drop='first')
X_train_gender_city = ohe.fit_transform(X_train[['gender', 'city']]).toarray()
X_test_gender_city = ohe.transform(X_test[['gender', 'city']]).toarray()
```

gender has 2 categories → 1 column after dropping first (dummy variable trap avoided)
city has 4 categories → 3 columns after dropping first
Total: 1 + 3 = 4 new columns

Step 5: Extract the untouched age column

```python
X_train_age = X_train.drop(columns=['fever', 'cough', 'gender', 'city']).values
X_test_age = X_test.drop(columns=['fever', 'cough', 'gender', 'city']).values
age needs no transformation, so it's extracted separately, as-is.
```

Step 6: Concatenate everything back together

```python
X_train_transformed = np.concatenate(
    (X_train_age, X_train_fever, X_train_gender_city, X_train_cough),
    axis=1
)
X_test_transformed = np.concatenate(
    (X_test_age, X_test_fever, X_test_gender_city, X_test_cough),
    axis=1
)
```

Final shape check: 1 (age) + 1 (fever) + 4 (gender+city) + 1 (cough) = 7 columns total.
Takeaway: This worked, but it was a LOT of separate steps for just 4-5 columns — each requiring its own fit/transform, and finally manual concatenation. Imagine doing this with 50 columns — extremely time-consuming and error-prone.

Approach 2: The "Easy Way" — Using ColumnTransformer
Step 1: Import and create the transformer

```python
from sklearn.compose import ColumnTransformer

transformer = ColumnTransformer(transformers=[
    ('tm1', SimpleImputer(), ['fever']),
    ('tm2', OrdinalEncoder(categories=[['Mild', 'Strong']]), ['cough']),
    ('tm3', OneHotEncoder(sparse=False, drop='first'), ['gender', 'city'])
], remainder='passthrough')
```

Breaking down ColumnTransformer's parameters:
1. transformers — a list of tuples, where each tuple has exactly 3 elements:

Name (any string you choose) — e.g., 'tm1', 'tm2', 'tm3'
Transformer object — e.g., SimpleImputer(), OrdinalEncoder(...), OneHotEncoder(...)
List of column(s) this transformer should apply to — e.g., ['fever'], ['cough'], ['gender', 'city']
So each tuple reads as: "apply THIS transformer to THESE columns, and call it THIS name."

2. remainder — tells the transformer what to do with columns that are not mentioned in any tuple (like age in this example). Two options:

'drop' (default) — drop the untouched columns entirely
'passthrough' — keep the untouched columns as-is, unchanged, in the final output
Since age needs no transformation but should still be part of the final output, remainder='passthrough' is used.

Step 2: Fit and transform in one line

```python
X_train_transformed = transformer.fit_transform(X_train)
X_test_transformed = transformer.transform(X_test)
```

Just pass the entire X_train dataframe (not individual columns!) — ColumnTransformer internally figures out which columns go to which transformer based on the tuples you defined.

It automatically:

Applies SimpleImputer to fever

Applies OrdinalEncoder to cough

Applies OneHotEncoder to gender and city

Passes age through unchanged

Combines all of this into a single final array — no manual hstack/concatenate needed!

Result

Same final output as the manual approach — 7 columns total — but achieved in a fraction of the code and effort.

For the test set, only .transform() (not .fit_transform()) is called — same rule as before: fit only on train data, transform both train and test.

Key Takeaways / Revision Points

1) Problem solved by ColumnTransformer: applying different preprocessing techniques (imputation, scaling, encoding, etc.) to different columns and combining results — without manually splitting data, transforming each part separately, and re-joining with hstack/concatenate.
   
2) Import: from sklearn.compose import ColumnTransformer
   
3) Syntax:

ColumnTransformer(transformers=[
    ('name1', transformer_object_1, ['col1']),
    ('name2', transformer_object_2, ['col2']),
    ('name3', transformer_object_3, ['col3', 'col4']),
], remainder='drop' or 'passthrough')

4) Each tuple in transformers = (name, transformer_object, list_of_columns)

5) remainder='passthrough' → keeps untouched columns in the output; remainder='drop' (default) → discards them
   
6) Just call .fit_transform(X_train) and .transform(X_test) on the whole dataframe — no need to manually select/split columns yourself

7) Massively simplifies code and reduces room for bugs, especially as the number of columns grows (imagine doing the manual approach with 50 columns)
   
8) Practice tip: pick any dataset and try applying ColumnTransformer yourself for hands-on understanding
