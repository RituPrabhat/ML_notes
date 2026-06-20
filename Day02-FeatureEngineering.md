What is Feature Engineering?
<img width="862" height="362" alt="image" src="https://github.com/user-attachments/assets/642ae5a3-e7ca-429d-b48f-52f5020ce2ff" />

Before definition, let's understand the problem.
Suppose you have this dataset:
Age	Salary	Bought Laptop
22	25000	    No
45	90000	    Yes
You directly train a model.

Now another dataset:
Date of Birth	      Monthly Salary	      Bought Laptop
01-01-2002	           25000	            No
01-01-1980	           90000            	Yes

Humans immediately understand:
Age matters
Salary matters

But machine only sees numbers and text.

Feature Engineering is the process of converting raw data into more useful information for machine learning.
A feature is simply an input column.
Example: Age	Salary	Experience
Features:
Age
Salary
Experience

Target:
Purchased

In ML language:
X = Features
Y = Target

Why Feature Engineering Exists:
Real-world data is ugly. You may have:
Name
Address
Date
Gender
Salary
Comments

ML algorithms cannot properly understand many of these columns directly.
We transform them into meaningful numerical representations.
This transformation process is Feature Engineering.

A Famous Example:
Suppose we want to predict house prices.
Dataset:
House Size 1000
Model sees only size.
Now we create: Size	Bedrooms	Age
Performance improves. Why?Because we provided more useful information.This is feature engineering.

<img width="927" height="510" alt="image" src="https://github.com/user-attachments/assets/536c92e9-7bc7-43a9-bf31-a14982614d8d" />

Feature Transformation

Missing Value Imputation: Machine learning libraries (like Scikit-learn) do not accept missing values; you must remove or impute them using techniques like Mean, Median, or Mode.

Handling Categorical Values: Converting categorical text data into numerical formats (e.g., One-Hot Encoding) so the machine can process it.

Outlier Detection: Identifying and removing outliers is crucial, as they can distort results in algorithms like Linear Regression.

Feature Scaling: Standardizing data ranges (e.g., keeping values between -1 to 1). This is essential for algorithms based on Euclidean distance, like K-Nearest Neighbors (KNN), to ensure one feature doesn't dominate others.

Feature Construction: Involves creating new, meaningful features from existing ones.

Example: In the Titanic dataset, the speaker suggests combining "Siblings/Spouses" and "Parents/Children" counts to create a new "Family Size" feature.

 Feature Selection 

Selecting only the most important features from a large dataset to improve model speed and performance.
Example: In image data (like MNIST), most pixels (features) might be irrelevant; focusing on central pixels is more efficient.

Feature Extraction

Generating completely new, reduced sets of features from original ones (often using algorithms like PCA).
Example: In real estate, instead of using "Number of Rooms" and "Wall Count," you might create a single "Square Footage" feature.


