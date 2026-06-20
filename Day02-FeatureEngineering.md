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


