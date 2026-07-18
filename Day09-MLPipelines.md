# Sklearn Pipelines:

## What is a Pipeline?

**Definition:** A Pipeline in sklearn is basically a mechanism that aims to chain multiple steps together such that the output of one step is used as the input to the next step.

Visually:

Input → [Step 1] → [Step 2] → [Step 3] → Output

The key benefit: the output of one step automatically becomes the input of the next step — this chaining is handled internally/automatically. You don't manually pass data between steps yourself.

### Example scenario
Suppose you want to build an ML model on a dataset that has:
- Missing values
- Categorical variables

Normally you'd do this in 3 separate steps:
1. Remove/impute missing values
2. Encode categorical variables
3. Train the model

Using a Pipeline, you can chain all three steps together — you just give your raw input, and the pipeline automatically produces the final output (trained model / prediction), handling all intermediate steps internally.

## Why Use Pipelines? (Key Benefit)

**Main advantage:** Pipelines make it easy to apply the **same preprocessing to both train and test data**, and — more importantly — to **new/live data** once the model is deployed (e.g., in a website or application).

- When you deploy a model, live input data will come in (fresh, unseen data).
- Without a pipeline, you'd have to **manually re-write** all the same preprocessing steps (handling missing values, encoding categorical data, etc.) again in your production code.
- This means duplicating your training-time logic in your deployment/production code — repetitive and **error-prone** (any small mismatch between training preprocessing and production preprocessing causes bugs).
- With a pipeline, this preprocessing logic is bundled into a single object, eliminating this duplication and risk of inconsistency.

## Structure of This Video/Note

Two parts:
1. **Part 1 (this note):** A small project done **WITHOUT** pipelines — to show how much manual work and pain it involves.
2. **Part 2 (next):** The **SAME** project done **WITH** pipelines — to show how much simpler and cleaner the code becomes.

---

## Part 1: Titanic Dataset — WITHOUT Using Pipeline

Using the famous **Titanic dataset** (already familiar/previously discussed) — chosen because it's relatively simple, without too many complications, making it easy to focus purely on understanding pipelines later.

### Goal
The end goal here is **not** to build the best-performing model — the actual goal is to **learn how pipelines work**. So model performance/accuracy isn't the focus at this stage.

### Imports needed
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer      # for handling missing values
from sklearn.preprocessing import OneHotEncoder  # for categorical data
from sklearn.preprocessing import MinMaxScaler   # for feature scaling
from sklearn.tree import DecisionTreeClassifier  # for prediction
Dataset overview
Titanic passenger dataset.
Target column: Survived (0 = passenger died, 1 = passenger survived).
Goal: build a model that predicts survival based on passenger details — Passenger Class, Sex, Age, SibSp (siblings/spouses), Parch (parents/children), Fare, Embarked.
Step 1: Drop unnecessary columns
Dropped 4 columns not useful for prediction:

PassengerId
Name
Ticket
Cabin (has too many missing values)
Remaining dataframe: input features + Survived (target).

Step 2: Train-Test Split

X = df.drop(columns=['Survived'])
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
Step 3: Identify Missing Values
Known columns with missing values in this dataset:

Age — has missing values
Embarked — has missing values
Missing values must be handled before proceeding further.

Step 4: Applying Imputation
Two separate SimpleImputer objects were created — one for Age, one for Embarked:


si_age = SimpleImputer()  # default strategy = mean
si_embarked = SimpleImputer(strategy='most_frequent')
Age → filled using mean (default SimpleImputer behavior)
Embarked → filled using most frequent value (since Embarked has categories like Southampton, Cherbourg, Queenstown — mean doesn't make sense for categorical data; Southampton was the most frequent port, so missing values got replaced with "Southampton")

X_train_age = si_age.fit_transform(X_train[['Age']])
X_test_age = si_age.transform(X_test[['Age']])

X_train_embarked = si_embarked.fit_transform(X_train[['Embarked']])
X_test_embarked = si_embarked.transform(X_test[['Embarked']])
Why two separate imputer objects instead of one?

You cannot pass both Age and Embarked together into a single SimpleImputer with the same strategy, because they need different fill strategies (mean vs. most frequent).
Also, technically, doing them together would create complications since both columns simultaneously have missing values and one transformation choice doesn't fit both data types — so they must be handled with separate objects/transforms.
After this step: Age and Embarked no longer have missing values.

Step 5: One-Hot Encoding on Sex and Embarked
Two separate OneHotEncoder objects were created (again, due to similar reasons — can't cleanly combine differently-shaped categorical columns in a single object here):


ohe_sex = OneHotEncoder(sparse=False, handle_unknown='ignore')
X_train_sex = ohe_sex.fit_transform(X_train[['Sex']])
X_test_sex = ohe_sex.transform(X_test[['Sex']])

ohe_embarked = OneHotEncoder(sparse=False, handle_unknown='ignore')
X_train_embarked_ohe = ohe_embarked.fit_transform(X_train_embarked)
X_test_embarked_ohe = ohe_embarked.transform(X_test_embarked)
Why two separate calls? Because Sex has no missing values but Embarked had missing values that first needed to be filled (from Step 4) before applying one-hot encoding on top of it. The workflow needs to happen in sequence, and combining both into a single .fit_transform() call directly wasn't feasible at this stage without a pipeline/column transformer.
handle_unknown='ignore' parameter: this means if, in the future, the test data contains a completely new/unseen category not seen during training, the encoder won't throw an error — it'll just encode it as all zeros instead of crashing.
Note: no drop='first' was used here (i.e., dummy variable trap wasn't specifically avoided) — reasoning given: since a Decision Tree model is being used (not a linear model), multicollinearity from dummy variables doesn't meaningfully affect tree-based models, so it doesn't matter here.
Result: Sex → 2 new columns (male/female), Embarked → 3 new columns (Southampton/Cherbourg/Queenstown).

Step 6: Extract the remaining untouched columns

X_train_remaining = X_train.drop(columns=['Sex', 'Age', 'Embarked']).values
X_test_remaining = X_test.drop(columns=['Sex', 'Age', 'Embarked']).values
These are the columns that needed no special preprocessing at all (Pclass, SibSp, Parch, Fare) — left as-is.
Step 7: Concatenate everything back together

X_train_transformed = np.concatenate(
    (X_train_remaining, X_train_age, X_train_sex, X_train_embarked_ohe),
    axis=1
)
X_test_transformed = np.concatenate(
    (X_test_remaining, X_test_age, X_test_sex, X_test_embarked_ohe),
    axis=1
)
Final shape: 7 columns (input features) after all transformations combined.

Step 8: Train the model

clf = DecisionTreeClassifier()
clf.fit(X_train_transformed, y_train)

y_pred = clf.predict(X_test_transformed)
Observation: This entire process — no ColumnTransformer was used here, so every column had to be manually separated, individually transformed, and manually re-joined. It was tedious even with just 4-5 columns needing transformation. (Using ColumnTransformer — covered in a previous note — would have made this considerably easier, but the point here is to show the full pain, including deployment, without any automation.)

Deploying the Model (Pickling)
Goal: Deploy this trained model into a website/form, where a user can input a new passenger's details (Class, Sex, Age, SibSp, Parch, Fare, Embarked), and the model predicts survival.

Exporting the model
To export a trained model, the pickle library is used.


import pickle

pickle.dump(clf, open('models/clf.pkl', 'wb'))
Important: Along with the trained classifier (clf), you must also export the fitted transformer objects used during preprocessing:

ohe_sex (the fitted OneHotEncoder for Sex)
ohe_embarked (the fitted OneHotEncoder for Embarked)
Why? Because new incoming data will be raw (e.g., "male"/"female" as text) — this text must be encoded into numbers using the exact same encoding logic learned during training, otherwise the model (which only understands numbers) can't process it correctly.

Why NOT export the SimpleImputer objects (si_age, si_embarked)?

Deliberately not exported in this deployment example — reasoning given: at prediction time (a single new passenger submitting a form), it's assumed the user will always provide the Age and Embarked values (no missing values expected in this live single-row scenario), so imputation isn't necessary there. This was a deliberate simplification for this example, not a general rule.
So in this example, exported to disk (3 files saved in a models folder):

clf.pkl (trained Decision Tree Classifier)
ohe_sex.pkl (fitted OneHotEncoder for Sex)
ohe_embarked.pkl (fitted OneHotEncoder for Embarked)
These are binary files — not human-readable, only usable via code.

Simulating a "production" prediction (loading pickled files)

clf = pickle.load(open('models/clf.pkl', 'rb'))
ohe_sex = pickle.load(open('models/ohe_sex.pkl', 'rb'))
ohe_embarked = pickle.load(open('models/ohe_embarked.pkl', 'rb'))
Loaded back using the exact same variable names they were saved under.

New test input (manually created, simulating a form submission)
Example passenger:

Pclass = 2
Sex = male
Age = 31
SibSp = 0
Parch = 0
Fare = 10.50
Embarked = Southampton (S)

test_input = np.array([2, 'male', 31.0, 0, 0, 10.50, 'S'], dtype=object).reshape(1, 7)
Applying the same preprocessing steps manually (no imputation needed here, since no missing values in this new input)

test_input_sex = ohe_sex.transform(test_input[:, 1].reshape(1, 1))
test_input_embarked = ohe_embarked.transform(test_input[:, -1].reshape(1, 1))
Transform Sex column using the loaded ohe_sex → gives encoded output (e.g., array showing male=1, female=0 or similar).
Transform Embarked column using the loaded ohe_embarked → gives 3-column encoded output (e.g., Southampton=1, others=0).
Combine everything back together (again, manually, in the correct order)

test_input_age = test_input[:, 2].reshape(1, 1)

test_input_transformed = np.concatenate(
    (test_input[:, [0, 3, 4, 5]], test_input_age, test_input_sex, test_input_embarked),
    axis=1
).astype(float)
The order of columns matters a lot here — must exactly match the order used during training (remaining columns → age → sex → embarked).
Final shape: 1 row × 10 columns (matches the 10-column structure the model was trained on, after encoding expanded Sex → 2 cols and Embarked → 3 cols).
Final prediction

clf.predict(test_input_transformed)
# Output: array([1])  → passenger survives
Key Takeaway from Part 1 (Without Pipeline)
This entire manual process highlights the core problem:

A LOT of separate preprocessing steps had to be done individually (imputation, encoding, concatenation) — both during training AND again during "production"/deployment.
You have to remember the exact column order used during training and replicate it exactly at prediction time — a single mismatch or reordering breaks everything.
Any small change to the preprocessing pipeline (e.g., changing an imputation strategy) means you must go back and update the same logic in two places — training code AND production code — which is repetitive and highly error-prone (a classic real-world source of bugs when a training/serving mismatch occurs).
This is exactly the problem Pipelines solve — bundling preprocessing + model into a single object means you only write this logic once, and it stays consistent between training and deployment automatically.

Part 2 Preview: WITH Pipeline (strategy outlined so far)
New imports needed:


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, chi2
Planned Pipeline Strategy
The pipeline will be built as a sequence of steps:

Step 1 — Impute missing values (Age and Embarked) using a ColumnTransformer (trf1)
Step 2 — One-Hot Encode Sex and Embarked columns using another ColumnTransformer (trf2)
Step 3 — Feature Scaling using MinMaxScaler (chosen instead of StandardScaler because Feature Selection — the next step — works better/is more appropriate after Min-Max scaling in this case) — applied via another ColumnTransformer (trf3)
Step 4 — Feature Selection — select the best 5 features using SelectKBest (this step doesn't need a ColumnTransformer, just a direct function call)
Step 5 — Train the model using DecisionTreeClassifier (trf5)
All of these steps will then be chained together into a single Pipeline object using Pipeline() from sklearn.pipeline.

Details covered so far for the pipeline build:
trf1 — Imputation via ColumnTransformer:


trf1 = ColumnTransformer([
    ('impute_age', SimpleImputer(), [2]),
    ('impute_embarked', SimpleImputer(strategy='most_frequent'), [6])
], remainder='passthrough')
Columns referenced by index number (e.g., 2, 6), NOT by column name.
Important reasoning given: when a ColumnTransformer produces output, the result is a NumPy array (not a DataFrame) — meaning column names are lost after the first transformation step. So if subsequent steps in the pipeline tried to reference columns by name, the code would break. Best practice: always reference columns by index/position when building pipelines, since transformers pass along array-like data without column names between steps.
trf2 — One-Hot Encoding via ColumnTransformer:


trf2 = ColumnTransformer([
    ('ohe_sex_embarked', OneHotEncoder(sparse=False, handle_unknown='ignore'), [1, 6])
], remainder='passthrough')
Both Sex and Embarked handled together in a single OneHotEncoder call this time (unlike Part 1, where they were forced to be separate) — since everything is now happening cleanly within the pipeline's array-based flow.
handle_unknown='ignore' retained — same reasoning as before (avoid crashing on unseen categories in production).
No drop='first' used here either — same reasoning: Decision Tree model is not affected by multicollinearity.
trf3 — Feature Scaling via ColumnTransformer:


trf3 = ColumnTransformer([
    ('scale', MinMaxScaler(), slice(0, 10))
])
MinMaxScaler (not StandardScaler) chosen specifically because the next step, Feature Selection, pairs better with Min-Max scaled data in this case.
Column range: slice(0, 10) — applies to all 10 columns present at this point in the pipeline (since after trf1 and trf2, the column count grew from the original 7 to 10 — 2 columns replaced by encoding into 2 [Sex] + 3 [Embarked] = 5 new columns, replacing the original 2, i.e., 7 - 2 + 5 = 10).
Feature Selection step:


# uses SelectKBest with a scoring function (e.g., chi2), selecting top 5 features
No ColumnTransformer needed for this step — just calling the SelectKBest function directly with a chosen scoring function and k value (top 5 features in this case, though it could be adjusted, e.g., top 8).
Explicitly noted: the reasoning/internal mechanics of how Feature Selection works is not the focus here — the point is just to show how to slot a Feature Selection step into a pipeline.
Author's personal note: leaving out feature selection entirely wouldn't hurt much (results were noted to be only slightly worse without it) — this step is shown mainly for reference in case needed in the future, not because it's critical here.
Last step (model training) — mentioned that the classifier object will be the 5th step in the pipeline, joining everything together.

To combine all these steps, the Pipeline class needs to be imported from sklearn.pipeline, and a list of steps will be passed to it.

Note: This is Part 1 of the notes — the transcript was cut mid-explanation of building the final Pipeline object (steps list was about to be defined). Please send the second half of the transcript, and I'll continue the notes seamlessly from where this left off (covering the full Pipeline object construction, fitting, and comparison with the non-pipeline approach).



