## One-Hot Encoding — How It Works

Example: **Color** column with 3 categories — Yellow, Blue, Red (Nominal data — no order between colors).

- Since there's no order, Ordinal Encoding cannot be used — assigning Yellow=0, Blue=1, Red=2 would wrongly imply "Red > Blue > Yellow" to the ML algorithm, when in fact these are just categories with equal standing.

### Solution: Create one column per category

For a "Color" column with 3 categories, create 3 new columns:
- `color_Yellow`
- `color_Blue`
- `color_Red`

Each row gets a **1** in the column corresponding to its category and **0** in the rest.

Example:
| Original | color_Yellow | color_Blue | color_Red |
|---|---|---|---|
| Yellow | 1 | 0 | 0 |
| Blue | 0 | 1 | 0 |
| Red | 0 | 0 | 1 |

This converts each category into a **vector** representation:
- Yellow → [1, 0, 0]
- Blue → [0, 1, 0]
- Red → [0, 0, 1]

**Rule of thumb:** Whenever you have Nominal categorical data, use One-Hot Encoding.

### Scaling with number of categories
- If a column has 50 unique categories, you'll create 50 new columns.
- The number of new columns = number of unique categories.
- This increases dimensionality of the dataset — a known drawback of OHE (handled later via "most frequent categories" technique).

---

## Dummy Variable Trap

### What happens after One-Hot Encoding
After creating N columns for N categories, you should always **drop one column** (commonly the first one, though which one you drop doesn't matter much).

So, if you have N categories, keep only **(N-1) columns**.

### Why drop a column? (Multicollinearity)

- **Multicollinearity**: when your input columns have a mathematical relationship/dependency on each other. Input columns in ML should be **independent** of one another (independent variables), not dependent.
- In the color example: `color_Yellow + color_Blue + color_Red` always sums to 1 for every row — meaning if you know 2 of the 3 columns, you can always derive the 3rd. This creates a mathematical relationship between the columns → multicollinearity.
- Example: if `color_Blue = 0` and `color_Red = 0`, then `color_Yellow` MUST be 1 — so the Yellow column is redundant; it carries no additional information.
- This is problematic especially for **linear-based models** (Linear Regression, Logistic Regression) — a full explanation of why multicollinearity is a problem will be covered in a dedicated future note.

### Solution
- Drop one column entirely (e.g., drop the first one).
- With N-1 columns, you can still fully represent all N categories:
  - If both `color_Blue = 0` and `color_Red = 0` → it must be Yellow (the dropped category) — no information is lost.

### Terminology
- These newly created columns (from OHE) are called **Dummy Variables**.
- The problem of multicollinearity arising from including all N dummy columns is called the **Dummy Variable Trap**.
- **Fix:** always drop one of the dummy columns.

---

## Handling Columns With Many Categories (Most Frequent Categories Technique)

### Problem
Example: Car dataset with a **Brand** column having many unique brands (e.g., Maruti Suzuki, Land Rover, etc. — nominal data, no inherent order).

- If you apply OHE directly on `Brand`, you'll get one column per brand → high dimensionality → slower processing.

### Solution: Keep only most frequent categories
- Identify the **most frequent categories** (e.g., Maruti Suzuki, which has many cars).
- Keep those as individual columns.
- Merge all remaining (less frequent) categories into a single new category — e.g., "Uncommon" or "Others".
- This reduces the dataset's dimensionality significantly (e.g., instead of 32 brand columns, you might end up with just ~10).

This technique is useful when some categories are very frequent and others are rare.

---

## Code Example

### Dataset columns
- `brand` (many categories, nominal)
- `km_driven`
- `fuel` (Diesel, Petrol, CNG, LPG — 4 categories, nominal)
- `owner` (First, Second, Third, Fourth & above, Test Drive Car — 5 categories, nominal)
- `selling_price` (target)

### Exploring category counts

```python
df['brand'].value_counts()
df['brand'].nunique()   # e.g., 32 different brands
df['fuel'].value_counts()   # 4 categories — some very frequent, some rare
df['owner'].value_counts()  # 5 categories
Method 1: One-Hot Encoding using Pandas get_dummies

pd.get_dummies(df, columns=['fuel', 'owner'])
Only applying OHE on fuel and owner here (not on brand — that's handled separately later using the "most frequent" technique, since applying OHE directly on brand would create too many columns).
Original dataset had 5 columns. After OHE:
fuel column removed → replaced by 4 new columns (one per fuel category)
owner column removed → replaced by 5 new columns (one per owner category)
Net result: 5 - 2 + 4 + 5 = 12 columns
New column naming convention: columnname_categoryname (e.g., fuel_CNG, fuel_Diesel, owner_First Owner, etc.)

Handling Dummy Variable Trap with get_dummies

pd.get_dummies(df, columns=['fuel', 'owner'], drop_first=True)
drop_first=True drops the first category column for each encoded column, avoiding the dummy variable trap.
Example: fuel_CNG gets dropped (first alphabetically/positionally), owner_First Owner gets dropped similarly.
Caveat: pd.get_dummies() is convenient for EDA/data analysis, but should NOT be used for ML projects/production pipelines. Reason: it doesn't "remember" which column was placed at which position during encoding — running it again might produce a different column order/result, causing inconsistency between training and inference. For ML projects, use sklearn's OneHotEncoder instead, since it remembers the encoding logic (fit on train, transform on train/test consistently).

Method 2: One-Hot Encoding using sklearn OneHotEncoder

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Split first (always split before any transformation)
X_train, X_test, y_train, y_test = train_test_split(
    df.iloc[:, 0:4],   # input columns: brand, km_driven, fuel, owner
    df.iloc[:, -1],    # target: selling_price
    test_size=0.2
)

ohe = OneHotEncoder()

# Apply only on 'fuel' and 'owner' columns
X_train_new = ohe.fit_transform(X_train[['fuel', 'owner']]).toarray()
X_test_new = ohe.transform(X_test[['fuel', 'owner']]).toarray()
fit_transform returns a sparse matrix by default — use .toarray() to convert to a dense NumPy array for easy viewing.
Since OHE is applied only on 2 of the 4 input columns (fuel, owner), the remaining columns (brand, km_driven) must be extracted separately and then concatenated back with the encoded columns to form the complete input matrix.
Combining encoded + non-encoded columns

# Extract remaining columns (brand, km_driven) as array
X_train_remaining = X_train.iloc[:, 0:2].values
X_test_remaining = X_test.iloc[:, 0:2].values

# Concatenate remaining columns with the new one-hot encoded columns
X_train_final = np.hstack((X_train_remaining, X_train_new))
X_test_final = np.hstack((X_test_remaining, X_test_new))
Note: This manual splitting and merging is tedious. This exact problem is solved elegantly using ColumnTransformer (covered in a future video/note) — it lets you apply different transformers to different columns in a single line of code and get the final combined result directly, without needing to manually separate and re-join columns.

Avoiding Dummy Variable Trap in sklearn

ohe = OneHotEncoder(drop='first')
Setting drop='first' in OneHotEncoder drops the first category column for each feature — avoiding the dummy variable trap, similar to drop_first=True in pandas.
With this, both fuel and owner will each lose one category column (e.g., 4 fuel categories → 3 columns; 5 owner categories → 4 columns).
Avoiding dense array conversion — sparse=False

ohe = OneHotEncoder(drop='first', sparse=False)
By default, OneHotEncoder outputs a sparse matrix, requiring .toarray() conversion.
Setting sparse=False (or sparse_output=False in newer sklearn versions) directly returns a dense array, skipping the manual .toarray() step.
Controlling output data type

ohe = OneHotEncoder(drop='first', sparse=False, dtype=np.int32)
You can specify dtype (e.g., np.int32) so the resulting encoded columns are integers rather than floats.
Handling High-Cardinality Column (brand) Using Most Frequent Categories
Logic: keep individual columns only for brands with a car count above a threshold; merge all brands below that threshold into a single "uncommon" category.


threshold = 100  # example threshold — count of cars

counts = df['brand'].value_counts()

# Identify brands with car count below threshold
repl = counts[counts <= threshold].index

# Replace those rare brand names with a single common label
df['brand'] = df['brand'].replace(repl, 'uncommon')
counts = value counts of each brand.
repl = list/index of brand names whose count is below the threshold (i.e., rare brands like Jaguar, Volvo, Datsun, etc.).
df['brand'].replace(repl, 'uncommon') replaces all rare brand names with a single "uncommon" category.
Result: After this transformation, brands like BMW, Ford, Mahindra, Maruti, Renault, Toyota, Hyundai remain as individual categories (since they have enough cars), while rare brands (below threshold) are grouped into a single "uncommon" category — significantly reducing the number of unique categories, hence reducing dimensionality after OHE.

This becomes much easier and cleaner once you learn ColumnTransformer in the next video/note — you won't need this manual workaround; you'll be able to apply different transformers to different columns and build the entire pipeline in one go.

Key Takeaways / Revision Points
Nominal data (no order) → use One-Hot Encoding
OHE creates one new column per category; each row gets 1 in its category's column, 0 elsewhere
More categories → more new columns → increased dimensionality (a drawback of OHE)
Dummy Variable Trap: including all N dummy columns creates multicollinearity (columns become mathematically dependent on each other — e.g., they always sum to 1). Fix: drop one column (N categories → keep N-1 columns)
pd.get_dummies(df, columns=[...], drop_first=True) — quick and useful for EDA, but not recommended for ML pipelines since it doesn't persist encoding logic between train/test or across runs
sklearn.preprocessing.OneHotEncoder — preferred for actual ML projects since it can be fit on train data and consistently applied (transformed) on both train and test data
drop='first' → avoids dummy variable trap
sparse=False (or sparse_output=False) → returns dense array directly instead of sparse matrix
dtype=np.int32 → controls output data type
When a categorical column has too many unique categories (high cardinality, e.g., brand), avoid direct OHE — instead, keep only the most frequent categories as individual columns and group all rare/uncommon categories into a single new category (e.g., "uncommon") using a frequency threshold
Manual splitting/merging of encoded and non-encoded columns is tedious — this will be simplified using ColumnTransformer (next video/note)
