# Feature Scaling: Normalization 

## What is Normalization

The formal definition of Normalization is:

> Normalization is a technique applied as part of data preparation for machine learning. The goal of normalization is to change the values of numeric columns in the dataset to use a common scale, without distorting differences in the ranges of values or losing information.

In simple terms — if you're ever working with a dataset and you have some input columns, and you need to predict something (say, whether a person will buy sports equipment or not), you'll have a dataset with certain features.

Now, "weight" is a numerical quantity, and any such quantity has two components: a **magnitude** and a **unit**. So weight will have a magnitude and a unit — grams, kilograms, pounds, etc. In machine learning literature, it's said that whenever you're dealing with a numerical quantity, it's always a good idea to eliminate the units first — because if you work with units directly, things get messy. If you strip away the units from your columns and then apply machine learning algorithms on top, you'll always get better results. That's the core idea behind Normalization — to eliminate units, essentially.

Just like there are different types of Standardization, if you talk about "normal" machine learning transformations, the most famous technique is **Min-Max Scaling**. In fact, sometimes people just say "Normalize this" and assume it means Min-Max Scaling. But besides that, there are several other scaling/normalization techniques, such as:

- **Mean Normalization**
- **Max-Abs Scaling**
- **Robust Scaling**


## MinMax Scaling Intuition

Let's start with Min-Max Scaling — its intuition, when it's used, and what the formula is.

Suppose you have a numeric column called "Weight" in your dataset with values like: 130, 60, 63, 32, 54... and so on — let's say around 100 records.

Now if someone tells you to normalize this — in simple terms, apply Min-Max Scaling — it's actually quite simple. You apply a formula to every number in the distribution:

X_scaled = (X - X_min) / (X_max - X_min)



Where:
- `X_scaled` is the transformed value
- `X` is the original value
- `X_min` is the minimum value of the distribution
- `X_max` is the maximum value of the distribution

So you need to know two things from the entire set of numbers: what is the minimum value, and what is the maximum value. For example, here the minimum is 32 and the maximum is 130.

So, to convert 130: `(130 - 32) / (130 - 32) = 1`. And if you convert 32 itself: `(32 - 32) / (130 - 32) = 0`.

**Observation:** When you apply this transformation, the new distribution you get will always have a range strictly between 0 and 1. You can't go outside this range — because the smallest value will always map to 0 (since `X_min - X_min = 0`), and the largest value will always map to 1 (since `X_max - X_max` in numerator over the same range equals the denominator). So, basically, the maximum value of the column becomes 1, and the minimum value becomes 0.

**Bottom line:** Whenever you apply Min-Max Scaling, the output distribution will always lie between 0 and 1.

### Geometric Intuition

If you want to see the geometric intuition of what's happening — let's say you have 2 quantities: Weight and Height.

Initially, your data looks scattered — Height in centimeters on one axis, Weight on the other — and all your customers' data points are plotted somewhere in this space.

When we apply Min-Max Scaling on top of this, what the transformation does is: it takes this entire scattered data and squeezes/compresses it into a box where both axes range between 0 and 1.

So the geometric intuition of Min-Max Scaling is that you're taking your entire dataset and squeezing it into a **unit square** (or unit rectangle). If this were 3D data, you'd be compressing the entire dataset into a **unit cube**. If it had even more dimensions, you'd be compressing it into a **unit hypercube**.

Basically, whether the data is small or large, you're compressing it so that it fits within this unit structure. That's the geometric intuition for Min-Max Scaling, as opposed to what Standardization does.

## Code Example

Now let's do this hands-on with an actual **Wine dataset**, which I downloaded — you can find it on my GitHub, I'll share the link there. We'll apply normalization and see how it actually works.

Let me go to my browser — I've already written the code here. Let's go through it.

First, I imported the necessary libraries, then loaded my **wine.csv** file. This dataset has many columns, but I picked the first three: **Class label**, **Alcohol**, and **Malic acid**. So Alcohol and Malic acid are the input features, and Class is the output. Alcohol values appear somewhat larger in magnitude compared to Malic acid.

Next, I plotted a **distribution plot (distplot)** for the Alcohol column — it shows a somewhat bimodal-looking distribution, close to normal. And then I plotted the distribution of Malic acid as well.

I also made a **scatter plot** where the x-axis is Alcohol and the y-axis is Malic acid, and the points are colored by class labels — there are 3 different classes (1, 2, and 3) shown here. This was just to show you what the raw data looks like.

Next, the first thing you should always remember: whenever you're doing any kind of scaling, you must first do a **train-test split**. So I did that — split into train and test sets, and checked the shapes of X_train and X_test.

Then I imported `MinMaxScaler` from sklearn, created an object, and called `fit` on the training data — as I mentioned in the last video too, you always **fit** on the training set only, but you **transform** both train and test sets. Don't forget that.

Then I transformed both my train and test data using what the scaler learned — i.e., it computed the minimum and maximum of my distributions on both features.

Also, note that when you use sklearn's transformers, they convert your DataFrame into a NumPy array. So to handle that, I converted the NumPy array back into a DataFrame.

I then used `.describe()` on the original data — before scaling, the minimum for Alcohol was some value and the maximum was around 14.8, while Malic acid's minimum was some value and maximum was around 5.8 (approx).

Now, after applying Min-Max Scaling, both quantities' minimum becomes 0 and maximum becomes 1 — which you can see reflected in the describe() output of the scaled data. Alcohol's minimum became 0, and Malic acid's minimum also became 0, and both have a maximum value of 1.

**Important note:** Min-Max Scaling guarantees the min and max values — it does NOT guarantee anything about mean or standard deviation (that will vary dataset to dataset). It only guarantees the min and max.

I hope this transformation makes sense. Then, just like in the last video, I again plotted scatter plots comparing the original data and the Min-Max scaled data side by side. Visually, there isn't a huge difference in the scatter plot shape — you'll notice that (as I mentioned) the data gets compressed into a unit rectangle, and that's exactly what happened here. The original shape gets returned/preserved, just compressed into the unit box — the distribution shape stays almost identical.

If your distribution isn't very "normal" to begin with, you might notice slight shape changes. I'd recommend generating your own custom distributions and trying Min-Max Scaling on them yourself to build intuition — but generally, this scaling preserves the shape reasonably well, unlike some other transforms.

I also showed histograms/plots comparing before and after for both features individually — Malic acid ranged roughly from some value to 6, and Alcohol ranged from roughly some other range — and after scaling, both ranged from 0 to 1. You might notice negative values appearing sometimes in KDE plots — that's just because KDE plots make statistical inferences/extrapolations near the boundary, that's a KDE plotting artifact, not an actual data issue. Anyway, this brought both quantities onto the same scale (0 to 1), which is what we wanted.

Lastly, I showed individual distributions before and after scaling: for Alcohol, I didn't see much shape change apart from scale. For Malic acid too, I didn't see much difference in shape apart from the scale change — though it's not guaranteed that shape will always be preserved; some distributions can change shape after normalization. You need to keep this in mind.

One downside: since everything gets forced into the 0–1 range, **outliers get compressed too** (the effect of outliers persists in the scaled range). That's a slight drawback, but otherwise Min-Max Scaling works well for most other cases. That covers Min-Max Scaling.

## Mean Normalization

Now let's talk about **Mean Normalization** — I won't go into too much detail, I'll just give you the formula and tell you where it's used. The rest you can go try yourself.

The formula for Mean Normalization:

X_scaled = (X_i - mean) / (X_max - X_min)



So, similar to standardization, you're doing "mean-centering" — meaning wherever your data lies, you bring it so that it's centered around the mean (subtracting the mean), and then you also scale it by the range (max − min).

This typically gives you values roughly in the range of **-1 to 1**. If a value is less than the mean, you'll get a negative number; if greater than the mean, you'll get a positive number.

This technique isn't used very often. In fact, sklearn doesn't even have a dedicated class for it — you'd have to implement it yourself/hack it together, since sklearn's `StandardScaler` doesn't directly support this. It's rarely used, but it is used in algorithms where you specifically need **centered data**. But generally, people substitute Standardization instead for that purpose. So it's a bit less common, but I wanted to make you aware it exists.

## MaxAbs Scaling

Next is **Max-Abs Scaling** — this one's a bit interesting. If your distribution is already centered around zero and has both negative and positive values, and you want to scale it down, the formula is:

X_scaled = X / |X_max|



You simply divide the original value by the **maximum absolute value** in the dataset. That's it.

There's a dedicated class in sklearn called `MaxAbsScaler` which you can use directly.

This is typically used on **sparse data** — i.e., data that has a lot of zeros. If your dataset has a lot of zero values, you can use Max-Abs Scaling there. Beyond that, it's not used very often — personally, I haven't used it much, but wherever I've read about it, it's mentioned for sparse matrices/data with lots of zeros. I haven't personally used it on any project. What I have used is the next one — Robust Scaling.

## Robust Scaling

This one is quite interesting and often discussed. First, let me explain what Robust Scaling is: if you have some numeric column and you want to scale it using Robust Scaling, the formula is:

X_scaled = (X - median) / IQR



Where **IQR (Interquartile Range)** = 75th percentile value − 25th percentile value.

There's a direct class for this in sklearn — it's available as `RobustScaler`.

The biggest advantage of this technique is that it's **robust to outliers**. If your data has outliers, you should try this scaling — it handles outliers well. It performs well when your data has a good number of outliers, so it's worth trying.

You should experiment with all these techniques — none of these can be judged beforehand as to which will perform better. For some problems, one works better; for others, a different one works better. Generally, you can't predict in advance whether Min-Max, Standardization, or Robust Scaling will perform best on a given problem — you have to experiment. Machine learning is all about experimentation — the more experiments you run, the more knowledge/intuition you build about this.

## Normalization vs Standardization

Lastly, let's discuss **Normalization vs. Standardization** — a very confusing topic for many people. Let me share some practical, personal tips on when to use which.

I always say: you should deeply understand what kind of algorithm you're working with and what kind of data you're working with. The first question you should ask is:

**"Does this feature even need scaling?"**

You need to answer this first — sometimes scaling isn't required at all. For example, if you're working with tree-based models (like Decision Trees), you typically don't need to scale features. So the first question is always: do I need to scale features or not? This understanding will get clearer once you study a bit more and understand the internal working of algorithms — that will help you determine whether scaling is needed.

**Second question:** if you determine you do need scaling, should you use Standardization or Normalization? Here's a practical tip:

- **Most problems are solved better using Standardization.** In practice, you'll get better results with standardization most of the time, and it's used more often in general.

- **Normalization (Min-Max Scaling)** is typically used when you already know the guaranteed minimum and maximum values of your quantity in advance. For example — **Image Processing**. When I worked with CNNs, we applied algorithms on top of images, and in images, each color channel's value ranges between a fixed minimum (0) and maximum (255) — guaranteed. So when working with CNNs in real life, everyone typically uses Min-Max Scaling for this exact reason.

**So here's a solid tip:** if you already know that your numerical quantity will always lie within a fixed range (known min, known max), use **Min-Max Scaling**. When you don't have such guarantees, use **Standardization**. When you know there are **outliers**, use **Robust Scaling**. When you have **sparse data**, use **Max-Abs Scaling**.

These are some general tips — but if you're truly unsure and have no clue which to pick, just try all of them! It doesn't cost much — it'll just take a bit of extra computation time for your machine — and your intuition will develop: "oh, for this kind of data, this particular scaling worked best." That developed intuition/knowledge will sharpen your thinking and help you a lot going forward when you work on real-world datasets in an actual company setting.

---

