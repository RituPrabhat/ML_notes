# Encoding Categorical Data — Notes

## Types of Data

Data is broadly divided into two parts:

- **Numerical Data** — basically numbers (e.g., weight, height, sales figures)
- **Categorical Data** — data representing categories (e.g., gender, nationality, school branch/college branch)

### Categorical Data has two sub-types:

#### 1. Nominal Categorical Data
- Categories have **no relationship or order** among them.
- Example: **State** — West Bengal, Karnataka, Maharashtra. You cannot say one is "greater" or "better" than another.
- Example: **Engineering branch** — you can't say one branch is worth more than another.

#### 2. Ordinal Categorical Data
- Categories **do have a relationship/order** among them.
- Example: **Review** — Poor, Average, Good, Excellent. There is a clear ranking here (Excellent > Good > Average > Poor).

---

## Why Encode Categorical Data?

- Categorical data is mostly in the form of strings.
- Machine learning algorithms only understand numbers.
- So categorical data must be converted into numbers before feeding it to an ML algorithm.

There are many encoding techniques. This note focuses on **two main techniques**, plus one bonus:

1. **Ordinal Encoding** — used for **Ordinal** categorical data
2. **One-Hot Encoding** — used for **Nominal** categorical data (covered in next video/note)
3. **Label Encoding** — bonus technique, similar to Ordinal Encoding but used on the **target/output column**

---

## Label Encoding vs Ordinal Encoding (Key Difference)

- If you have a **categorical column in your input features (X)** that has an order → use **Ordinal Encoding**.
- If your **target/output column (y)** is categorical (as in classification problems: yes/no, rain/no rain, placement/no placement, image classes) → use **Label Encoder**, NOT Ordinal Encoder.
- Both do conceptually the same thing (convert categories to numbers), but:
  - `OrdinalEncoder` → designed for **input features (X)**
  - `LabelEncoder` → designed specifically for **output labels (y)**

> Note from experience: In the past, the video creator mistakenly used `LabelEncoder` on input columns too. After reading the sklearn documentation, this was corrected — `LabelEncoder` should only be used on the target column, not input features.

From sklearn docs on `LabelEncoder`:
> "Encode target labels with values between 0 and n_classes-1... This transformer should be used to encode target values, i.e., y, and not the input X."

---
## How Ordinal Encoding Works

Example column: **Education**, with values:
`High School`, `Under Graduate`, `Post Graduate` (and possibly more categories)

- First check: is this Ordinal or Nominal?
  - Since **Post Graduate > Under Graduate > High School** (an inherent order), this is **Ordinal data**.
  - If it were Nominal, you'd use One-Hot Encoding instead.

- To encode: you must **explicitly specify the order** — e.g., Post Graduate = highest, Under Graduate = middle, High School = lowest.
- You must tell the `OrdinalEncoder` the order — it cannot figure out the ranking on its own (it doesn't understand "meaning", only numbers you assign).
- Sklearn's `OrdinalEncoder` class handles the internal conversion once you provide the category order list.

Result: High School → 0, Under Graduate → 1, Post Graduate → 2 (example mapping).

---

## Code Example Walkthrough

### Dataset
Columns:
- `age`
- `gender` (Male/Female) → **Nominal**
- `review` (Poor / Average / Good) → **Ordinal**
- `education` (School / UG / PG) → **Ordinal**
- `purchased` (Yes/No) → target column, **Nominal**, classification problem

### Step 1: Identify column types
| Column | Type |
|---|---|
| gender | Nominal |
| review | Ordinal (Poor < Average < Good) |
| education | Ordinal (School < UG < PG) |
| purchased | Nominal (target/output) |

### Step 2: Plan of encoding
- `gender` → One-Hot Encoding (not covered in this note)
- `review`, `education` → Ordinal Encoding
- `purchased` → Label Encoding

> Ideally, all of this can be done together using `ColumnTransformer` (with different pipelines for different columns) — to be covered in a future note. For now, ignoring `gender` and demonstrating only `review`, `education` (Ordinal Encoding) and `purchased` (Label Encoding).

### Step 3: Train-Test Split (always do this FIRST before any transformation)

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    df.iloc[:, 0:2],      # input columns (review, education)
    df.iloc[:, -1],       # output column (purchased)
    test_size=0.2
)
Golden rule: fit transformer on train data only, but transform both train and test data.

Step 4: Ordinal Encoding

from sklearn.preprocessing import OrdinalEncoder

oe = OrdinalEncoder(categories=[
    ['Poor', 'Average', 'Good'],           # order for 'review' column
    ['School', 'UG', 'PG']                 # order for 'education' column
])

oe.fit(X_train)

X_train = oe.transform(X_train)
X_test = oe.transform(X_test)
The categories parameter takes a list of lists — one list per categorical column, in the order you want (lowest → highest).
If you don't specify order explicitly, categories may get assigned random/incorrect ranking (e.g., Poor could get a higher number than Excellent) — which defeats the purpose of ordinal encoding.
Always fit only on training data, then transform both train and test sets.
Output: categories converted to numbers, e.g., Good → 2, Average → 1, Poor → 0 (and similarly for education).
Step 5: Label Encoding (for target column purchased)

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
le.fit(y_train)

y_train = le.transform(y_train)
y_test = le.transform(y_test)
LabelEncoder has no categories parameter — it randomly assigns numbers to classes (e.g., Yes → 1, No → 0, or vice versa — not something you control).
Since there's no real order needed for a binary/multi-class target for most classifiers, this is fine.
Fit only on y_train, transform both y_train and y_test.
```

Key Takeaways / Revision Points
Nominal data → no order between categories → use One-Hot Encoding
Ordinal data → clear order between categories → use Ordinal Encoding (must manually specify category order)
Target/output column (categorical) → use Label Encoding, regardless of whether it's nominal or ordinal in nature
Always do train-test split before any transformation
Always fit on train data only, then transform both train and test data
OrdinalEncoder is for input features (X); LabelEncoder is for target labels (y) — don't mix these up
These encoding techniques are used in almost every ML project — a must-know fundamental
