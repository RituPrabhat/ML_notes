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

## Structure of The Note

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

## Creating the Pipeline Object

To combine all the individual transformer steps (`trf1`, `trf2`, `trf3`, feature selection, and the classifier) into one chained object, use the `Pipeline` class:

```python
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ('trf1', trf1),
    ('trf2', trf2),
    ('trf3', trf3),
    ('trf4', selector),      # feature selection object (SelectKBest)
    ('trf5', clf)            # DecisionTreeClassifier
])
Pipeline takes a list of tuples.
Each tuple has exactly 2 elements:
A name for that particular step (any string you choose)
The actual transformer/estimator object for that step
So the pipeline here has 5 steps: trf1 (imputation) → trf2 (one-hot encoding) → trf3 (scaling) → trf4 (feature selection) → trf5 (classifier).

You could name the steps anything you like — naming them trf1, trf2, etc. was just done for consistency/readability.

Pipeline (class) vs make_pipeline (function)
You may see other tutorials use make_pipeline() instead of the Pipeline class directly.


from sklearn.pipeline import make_pipeline

pipe = make_pipeline(trf1, trf2, trf3, selector, clf)
Difference: make_pipeline doesn't require you to manually specify a name for each step — you just pass the objects directly, and it auto-generates names internally.
This makes make_pipeline slightly simpler/shorter to write.
However, the video creator personally prefers using the Pipeline class directly (not make_pipeline) — reasoning: giving explicit, readable names to each step (like trf1, trf2) makes the pipeline easier to reference and debug later (as shown further below when inspecting individual steps).
Similarly, this same naming distinction applies to ColumnTransformer:

ColumnTransformer (class) → requires 3 things per tuple: name, object, columns
make_column_transformer (function) → only requires 2 things per tuple: object, columns (no name needed)
Choice between these is a matter of personal preference — either works.

Training the Pipeline
Once the pipeline object is created, training is as simple as:


pipe.fit(X_train, y_train)
Internally, this happens automatically in sequence:

X_train goes into trf1 (imputation) → produces some output
That output goes into trf2 (one-hot encoding) → produces some output
That output goes into trf3 (scaling) → produces some output
That output goes into trf4 (feature selection) → selects best features
Finally, the selected features go into trf5 (Decision Tree Classifier), which trains on them
All of this chaining happens automatically — you don't manually pass outputs between steps yourself; the Pipeline object handles it internally.

Important distinction: .fit() vs .fit_transform()
If your pipeline includes a final model/algorithm step (like a classifier) — i.e., you're doing actual model training, not just preprocessing — you call:

pipe.fit(X_train, y_train)
pipe.predict(X_test)
If your pipeline only contains preprocessing steps (imputation, encoding, scaling — no final model), then there's no "training" happening, just data transformation. In that case, you'd call:

pipe.fit_transform(X_train)
since you're not training a model, just processing data through the chained transformers, and you need the transformed output directly.
Summary: Two types of pipelines:

With an algorithm/model as the last step → use .fit() and .predict()
Without any algorithm (pure preprocessing pipeline) → use .fit_transform()
Visualizing the Pipeline
A very useful feature: sklearn can display a visual diagram of your pipeline's structure, showing each step in order.


from sklearn import set_config
set_config(display='diagram')

pipe   # just display the pipe object (e.g., in Jupyter) to see the diagram
This code must be written/run for the visual diagram to actually appear — it doesn't show up by default.
The diagram makes it easy for anyone (including you, later) to quickly understand what a pipeline is doing just by looking at it.
Example of what it shows for this pipeline:
trf1: applies 2 SimpleImputers — one for Age, one for Embarked
trf2: applies OneHotEncoder on 2 columns (Sex, Embarked)
trf3: scales everything using MinMaxScaler
trf4: selects the best features from the scaled output
trf5: trains the Decision Tree model on the selected best features
Exploring/Inspecting a Trained Pipeline (Debugging)
Once the pipeline (pipe) is trained, you can inspect its internal details — useful for debugging.

Get all steps with their names:

pipe.named_steps
Returns a dictionary where keys are the names you gave each step (e.g., 'trf1', 'trf2', 'trf3') and values are the actual fitted transformer/estimator objects.
This is the benefit of using the Pipeline class (with custom names) over make_pipeline — you can easily look up any step by its readable name.
Example: Inspecting what trf1 learned (mean used for Age imputation)

pipe.named_steps['trf1'].transformers_
Returns a list showing the 3 transformers inside trf1's ColumnTransformer: first is age (SimpleImputer), second is embarked (SimpleImputer), third is remainder.
To dig into the first one (Age imputer) and check what mean value it calculated:


pipe.named_steps['trf1'].transformers_[0][1].statistics_
transformers_[0] → gets the first transformer tuple (Age)
[1] → gets the actual object from that tuple (the SimpleImputer for Age)
.statistics_ → an attribute that shows the computed value used for imputation (the mean, in this case)
Example: Inspecting what trf1 learned for Embarked (most frequent value)

pipe.named_steps['trf1'].transformers_[1][1].statistics_
Same logic, but [1] instead of [0] to get the second transformer (Embarked imputer).
Output confirms: "Southampton" was the most frequent value, used to fill missing Embarked entries.
Why this matters: If you ever need to debug/verify your pipeline (check what values/parameters a particular step learned), you can drill into any step (trf1, trf2, trf3, etc.) this way and inspect its internal parameters/attributes. Very useful for debugging.

Predicting and Evaluating

y_pred = pipe.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)
Accuracy came out to be around 62% in this run — noted as slightly lower than expected, likely because Feature Selection removed some columns that were actually useful (over-aggressive feature reduction hurting performance here).
Suggestion: try removing the Feature Selection step from the pipeline and check if accuracy improves.
Cross Validation Using a Pipeline
You can also perform cross-validation directly on your pipeline object, just like you would with a standalone model.

What is cross-validation (brief): Instead of a single train-test split, you split your data multiple times (e.g., 5 different times) and train + evaluate the model on each split, then average the resulting scores — giving a more reliable performance estimate.


from sklearn.model_selection import cross_val_score

cross_val_score(pipe, X_train, y_train, cv=5, scoring='accuracy').mean()
The pipeline object itself can be passed directly into cross_val_score, just like a plain classifier.
Result in this case: 63% accuracy (called "cross accuracy" here).
Hyperparameter Tuning Using GridSearchCV with a Pipeline
What is hyperparameter tuning (brief): Every algorithm has "settings" (hyperparameters) that can be changed to improve or worsen its performance. Example: DecisionTreeClassifier has a hyperparameter called max_depth — tuning it (trying different values) can significantly change model performance. (A dedicated video on this topic already exists on the channel — worth watching separately.)

Setting up parameter grid for GridSearch

from sklearn.model_selection import GridSearchCV

params = {
    'trf5__max_depth': [1, 2, 3, 4, 5, None]
}

grid = GridSearchCV(pipe, params, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)
Important syntax note
Normally (without a pipeline), you'd just write the hyperparameter name directly, e.g., max_depth.
But inside a pipeline, you must prefix the hyperparameter name with the step's name, followed by double underscore (__):

'<step_name>__<parameter_name>'
Here, since the classifier step was named trf5, it becomes:

'trf5__max_depth'
This tells GridSearchCV exactly which step in the pipeline that hyperparameter belongs to.
Running and getting results

grid.best_score_     # e.g., 60% accuracy
grid.best_params_     # e.g., {'trf5__max_depth': 3}
This process takes a bit of time (tries all 6 different values for max_depth, running 5-fold cross-validation for each), then automatically picks the best-performing configuration.
The visual diagram (set_config(display='diagram')) also works when inspecting the grid object — same details displayed.
Deploying the Pipeline (Pickling) — Much Simpler Than Without a Pipeline
This is where the real payoff of using pipelines shows up.

Exporting

import pickle
pickle.dump(pipe, open('pipe.pkl', 'wb'))
Key benefit: you only need to export ONE single file — the entire pipe object.
Unlike the non-pipeline approach (Part 1), where you had to separately export the classifier AND each individual transformer (ohe_sex, ohe_embarked, etc.), here all that knowledge/logic is already bundled inside the single pipeline object — so there's nothing else to export separately.
Using in "production" code

import pickle
import numpy as np

pipe = pickle.load(open('pipe.pkl', 'rb'))

test_input2 = np.array([2, 'male', 31.0, 0, 0, 10.50, 'S'], dtype=object).reshape(1, 7)

pipe.predict(test_input2)
# Output: passenger does not survive (example result)
Notice how much shorter and simpler this production code is compared to Part 1 (no manual imputation, no manual encoding, no manual concatenation, no need to keep track of column order).
You just load the single pickled pipeline and call .predict() directly on the raw new input.
The real payoff: handling future changes
Scenario: If you later go back and make changes to your pipeline (e.g., change an imputation strategy, add/remove a preprocessing step, tweak parameters):

What you need to do: Just re-export (pickle.dump) the updated pipe object as the new pipe.pkl file, and place it in the same location on the server — that's it.
What you do NOT need to do: touch or modify the production code at all — no need to re-import anything new, no need to change function calls, no need to re-order columns, nothing.
This is the core advantage:

The production code stays completely untouched, no matter how much you change your preprocessing/pipeline logic internally. You only ever swap the .pkl file. This makes your deployed system extremely stable — production code "just works" like it did before, without any risk of a training/serving mismatch, since everything, including preprocessing logic, is bundled inside that one pickled file.

This is contrasted directly with Part 1 (without pipeline), where any small preprocessing change required you to manually go back and edit your production code as well — a fragile, error-prone process.

```
Key Takeaways / Revision Points:

Pipeline chains multiple steps (preprocessing + model) into a single object — output of each step automatically feeds into the next.

Import: from sklearn.pipeline import Pipeline

Syntax: Pipeline([('name1', obj1), ('name2', obj2), ...]) — list of (name, object) tuples

make_pipeline(obj1, obj2, ...) is a shortcut that skips naming — simpler, but less debuggable; naming your steps explicitly (using Pipeline class) is preferred for later inspection/debugging

Same distinction applies to ColumnTransformer vs make_column_transformer

Always reference columns by index, not name, inside pipeline steps — because intermediate outputs become plain NumPy arrays (column names get lost after the first transformation)

.fit() + .predict() → when pipeline includes a final model/algorithm step

.fit_transform() → when pipeline is preprocessing-only (no model at the end)

Visualize your pipeline using:

from sklearn import set_config

set_config(display='diagram')

then just display the pipeline object — extremely helpful for understanding/explaining pipeline structure at a glance

Debug/inspect any step using pipe.named_steps['<step_name>'], then drill into internal attributes (e.g., .transformers_, .statistics_) to check learned values (like computed mean, most frequent category, etc.)

Cross-validation works directly on pipeline objects: cross_val_score(pipe, X_train, y_train, cv=5)

Hyperparameter tuning works directly on pipeline objects too, via GridSearchCV — but hyperparameter names must be prefixed with the step name and double underscore: '<step_name>__<param_name>'

Deployment is dramatically simpler with pipelines:

Only ONE file needs to be pickled/exported (the whole pipeline), instead of separately exporting the model + every individual transformer

Production code becomes short and stable — it never needs to change even if you modify preprocessing logic later; you just swap out the updated .pkl file

Interview relevance: Pipelines are a commonly asked interview topic — understanding and being able to explain them well is considered valuable for job interviews in this field

Overall conclusion: Learn and use pipelines as early as possible in your ML workflow — they save enormous time/effort during both experimentation and especially production deployment, and prevent the exact bugs/mismatches shown in the "without pipeline" (Part 1) approach.





