# Multiple Linear Regression — Deriving the Formula from Scratch (Matrix Approach)

## 0. Recap: The Three Types of Linear Regression
1. **Simple Linear Regression** — 1 input column, 1 output column.
2. **Multiple Linear Regression** — multiple input columns, 1 output column (e.g., CGPA, IQ, Gender → Package). *(Today's topic — deriving the math)*
3. **Polynomial Regression** — not covered yet.
**Goal of today's lecture:** Derive the formula to calculate **$\beta_0, \beta_1, \beta_2, \ldots, \beta_n$** (all the coefficients) for Multiple Linear Regression, entirely from scratch — using matrices this time (unlike Simple Linear Regression, where basic calculus was enough).
 
> **Lecturer's note:** This is a harder topic than usual — but understanding it thoroughly builds strong intuition that helps with *many* other ML algorithms later, since matrix-based derivations are common throughout ML.
 
---
 
## 1. Recap of the Hyperplane Equation
 
From the previous lecture:
$$y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots + \beta_n x_n$$
 
- Once you know $\beta_0, \beta_1, \ldots, \beta_n$, you can plug in any new input values and get a prediction $\hat y$.
- **Today's question: How do we actually calculate these $\beta$ values?**
---
 
## 2. Setting Up the Problem for Multiple Students (Building the Matrices)
 
### 2.1 Starting Small — One Student at a Time
 
For a single student (say, student 1), with 3 input columns (CGPA, IQ, Gender) labeled $x_1, x_2, x_3$:
 
$$\hat y_1 = \beta_0 + \beta_1 x_{11} + \beta_2 x_{12} + \beta_3 x_{13}$$
 
Here, the notation $x_{11}$ means: **row 1 (student 1), column 1 (feature 1 = CGPA)**. Similarly $x_{12}$ = student 1's IQ, $x_{13}$ = student 1's Gender value.
 
For student 2:
$$\hat y_2 = \beta_0 + \beta_1 x_{21} + \beta_2 x_{22} + \beta_3 x_{23}$$
 
...and so on, up to student $n$ (if there are $n$ students total):
$$\hat y_n = \beta_0 + \beta_1 x_{n1} + \beta_2 x_{n2} + \beta_3 x_{n3}$$
 
### 2.2 Generalizing to $n$ Students and $m$ Input Columns
 
If there are $n$ students (rows) and $m$ input columns (features) in general:
 
$$\hat y_i = \beta_0 + \beta_1 x_{i1} + \beta_2 x_{i2} + \cdots + \beta_m x_{im}$$
 
This is a repetitive, tedious set of $n$ separate equations — writing them all out individually is clunky. This is exactly why we switch to **matrix notation**: it lets us represent *all* these equations in **one single, compact equation**.
 
---
 
## 3. Converting to Matrix Form
 
### 3.1 The Prediction Matrix ($\hat Y$)
 
Stack all the predicted outputs into a single column vector:
 
$$\hat Y = \begin{bmatrix} \hat y_1 \\ \hat y_2 \\ \vdots \\ \hat y_n \end{bmatrix}$$
 
This is an $(n \times 1)$ matrix — $n$ rows, 1 column.
 
### 3.2 Decomposing the System as a Matrix Product
 
The key insight: this whole system of $n$ equations can be written as a **product of two matrices**:
 
$$\hat Y = X \cdot \beta$$
 
Where:
- **$X$** (the "input matrix" or "design matrix") contains all the input feature values — **plus an extra column of all 1's** added at the front. This is a clever trick that lets the $\beta_0$ (intercept) term get naturally included in the matrix multiplication (since $\beta_0 \times 1 = \beta_0$).
- **$\beta$** (the "coefficient matrix") contains all the unknowns we want to solve for: $\beta_0, \beta_1, \ldots, \beta_m$.
### 3.3 Shape of Matrix $X$
 
If you have $n$ rows (students) and $m$ input columns (features):
 
$$X = \begin{bmatrix} 1 & x_{11} & x_{12} & \cdots & x_{1m} \\ 1 & x_{21} & x_{22} & \cdots & x_{2m} \\ \vdots & \vdots & \vdots & & \vdots \\ 1 & x_{n1} & x_{n2} & \cdots & x_{nm} \end{bmatrix}$$
 
- **Shape of $X$: $n \times (m+1)$** — $n$ rows, and $(m+1)$ columns (the original $m$ feature columns, plus the extra "all 1's" column for the intercept).
### 3.4 Shape of Matrix $\beta$
 
$$\beta = \begin{bmatrix} \beta_0 \\ \beta_1 \\ \vdots \\ \beta_m \end{bmatrix}$$
 
- **Shape of $\beta$: $(m+1) \times 1$**.
### 3.5 Verifying the Matrix Multiplication Works
 
When you multiply $X$ (shape $n \times (m+1)$) by $\beta$ (shape $(m+1) \times 1$):
- The **inner dimensions match** ($m+1 = m+1$) → multiplication is valid.
- The **resulting shape** is $n \times 1$ → exactly matching $\hat Y$'s shape. ✅
This confirms: $\hat Y = X\beta$ correctly represents the entire system of equations for all $n$ students simultaneously, in one clean matrix equation.
 
### 3.6 The Actual Output Matrix ($Y$)
 
Separately, we also have the **actual** (real, known) output values from our dataset:
 
$$Y = \begin{bmatrix} y_1 \\ y_2 \\ \vdots \\ y_n \end{bmatrix}$$
 
*(This is just your original target/output column, e.g., actual Package values — shape $n \times 1$.)*
 
---
 
## 4. Setting Up the Loss Function (in Matrix Form)
 
### 4.1 Recap: The Loss Function from Simple Linear Regression
 
$$E = \sum_{i=1}^{n} (y_i - \hat y_i)^2$$
 
This summed up all the squared differences between actual and predicted values (called **residuals**).
 
### 4.2 Rewriting the Loss Function Using Matrices
 
Define a new matrix — the **residual matrix**:
 
$$Y - \hat Y = \begin{bmatrix} y_1 - \hat y_1 \\ y_2 - \hat y_2 \\ \vdots \\ y_n - \hat y_n \end{bmatrix}$$
 
**Key trick:** If you take this residual matrix, **transpose it**, and **multiply it by itself**, you get exactly the sum of squares we want:
 
$$(Y - \hat Y)^T (Y - \hat Y) = \sum_{i=1}^{n} (y_i - \hat y_i)^2$$
 
**Why this works:** Matrix multiplication of a row vector by a column vector (where both come from the same values) is exactly the dot product — and the dot product of a vector with itself is the sum of the squares of its elements. That's precisely what we need for the loss function.
 
So the **loss function in matrix form** becomes:
 
$$E = (Y - \hat Y)^T(Y - \hat Y)$$
 
Substituting $\hat Y = X\beta$:
 
$$E = (Y - X\beta)^T(Y - X\beta)$$
 
---
 
## 5. Expanding the Loss Function
 
Using the algebraic identity: $(A - B)^T = A^T - B^T$, expand:
 
$$E = (Y^T - (X\beta)^T)(Y - X\beta)$$
 
$$E = (Y^T - \beta^T X^T)(Y - X\beta)$$
 
Multiplying out (like expanding $(a-b)(c-d)$):
 
$$E = Y^T Y - Y^T X\beta - \beta^T X^T Y + \beta^T X^T X\beta$$
 
### 5.1 A Useful Simplification — Combining the Middle Two Terms
 
At first glance, the middle two terms look different: $Y^TX\beta$ and $\beta^T X^T Y$. But it turns out **these two terms are actually equal** to each other (both are just single numbers/scalars, and taking a transpose of a scalar doesn't change it).
 
**Proof sketch (from the lecture):**
- We want to show: $Y^T X \beta = \beta^T X^T Y$
- Take the transpose of the left side: $(Y^T X \beta)^T = \beta^T X^T Y$ (using the rule $(ABC)^T = C^T B^T A^T$, applied in reverse/general order).
- Since $Y^TX\beta$ is a **1×1 matrix (a single number/scalar)**, transposing a scalar doesn't change its value — a scalar equals its own transpose.
- Therefore: $Y^TX\beta = (Y^TX\beta)^T = \beta^TX^TY$ ✅ — they are indeed equal.
Since they're equal, we can just combine them:
 
$$E = Y^TY - 2\beta^TX^TY + \beta^TX^TX\beta$$
 
*(This is now our final, clean loss function ready for differentiation.)*
 
---
 
## 6. Differentiating and Solving for Beta
 
### 6.1 The Optimization Goal (Same as Before)
 
Just like in Simple Linear Regression: **take the derivative of the loss function with respect to $\beta$, and set it to 0** to find the minimum.
 
$$\frac{\partial E}{\partial \beta} = 0$$
 
### 6.2 Applying the Derivative Term-by-Term
 
Differentiating each term in $E = Y^TY - 2\beta^TX^TY + \beta^TX^TX\beta$ with respect to $\beta$ (using standard **matrix calculus / matrix differentiation** rules, which the lecturer notes is a separate topic worth learning on its own):
 
- $\frac{\partial}{\partial\beta}(Y^TY) = 0$ *(no $\beta$ term here at all — constant with respect to $\beta$)*
- $\frac{\partial}{\partial\beta}(-2\beta^TX^TY) = -2X^TY$
- $\frac{\partial}{\partial\beta}(\beta^TX^TX\beta) = 2X^TX\beta$
Putting it together:
 
$$\frac{\partial E}{\partial\beta} = -2X^TY + 2X^TX\beta = 0$$
 
> **Note:** The lecturer explicitly mentions that deriving *why* these matrix-calculus derivative rules work requires a separate dedicated topic (matrix differentiation), which isn't covered in full detail in this lecture — the results are stated directly here, with a promise to cover matrix differentiation in a future video.
 
### 6.3 Solving for Beta
 
Cancel the common factor of 2:
 
$$-X^TY + X^TX\beta = 0$$
 
Move $X^TY$ to the other side:
 
$$X^TX\beta = X^TY$$
 
Now, to isolate $\beta$, we need to "divide" by $X^TX$ — but since these are matrices, we don't divide, we **multiply by the inverse**:
 
$$\beta = (X^TX)^{-1} X^TY$$
 
### 6.4 🎯 Final Formula
 
$$\boxed{\beta = (X^TX)^{-1}X^TY}$$
 
This single matrix equation gives you **all** the coefficients ($\beta_0, \beta_1, \ldots, \beta_m$) at once — this is the **Ordinary Least Squares (OLS)** solution for Multiple Linear Regression.
 
---
 
## 7. Verifying the Shapes Match Up (Sanity Check)
 
It's good practice to check that the final formula's shapes make sense:
 
- $X$ has shape $n \times (m+1)$ → $X^T$ has shape $(m+1) \times n$
- $X^TX$ has shape $(m+1) \times (m+1)$ → a **square matrix** (necessary, since only square matrices can be inverted) → $(X^TX)^{-1}$ also has shape $(m+1)\times(m+1)$
- $X^TY$: $X^T$ is $(m+1)\times n$, $Y$ is $n \times 1$ → result is $(m+1) \times 1$
- Final multiplication: $(m+1)\times(m+1)$ times $(m+1)\times 1$ → result is $(m+1) \times 1$
This matches exactly the expected shape of $\beta$ (which should have one value per coefficient: $\beta_0$ through $\beta_m$, i.e., $m+1$ values total). ✅ The formula is dimensionally consistent.
 
---
 
## 8. Why Gradient Descent Also Exists (Even Though We Have a Direct Formula)
 
### 8.1 The Problem: Matrix Inversion is Computationally Expensive
 
- The formula requires calculating $(X^TX)^{-1}$ — a **matrix inverse**.
- Computing a matrix inverse (e.g., via Gaussian elimination) for an $n \times n$ matrix has a **time complexity of roughly $O(n^3)$** (cubic complexity).
- In our case, the matrix being inverted has shape $(m+1)\times(m+1)$, where $m$ = number of input columns/features.
### 8.2 Why This Becomes a Real Problem
 
- If your dataset has a **large number of input columns** (say, 1000 features — common with text data, image data, or other high-dimensional data), then $(m+1)$ ≈ 1000, and inverting a 1000×1000 matrix requires roughly $1000^3$ = **1 billion computations**.
- This makes the algorithm **very slow** for high-dimensional datasets — a serious practical bottleneck.
### 8.3 The Solution: Gradient Descent
 
- **Gradient Descent** is an **iterative approximation technique** — instead of directly solving a formula, it gradually adjusts $\beta$ values step-by-step, getting closer and closer to the correct answer over many iterations, without ever computing a matrix inverse.
- It won't always give the **exact** same answer as OLS, but it will get **very close** — often close enough for practical purposes.
- **When to use which:**
  - **OLS (`LinearRegression` class in scikit-learn)** → Works fine and is simple for **small-to-moderate** numbers of input columns. In practice, this covers the vast majority of everyday use cases (the lecturer notes it's rare to encounter datasets with such huge dimensionality that OLS becomes impractical).
  - **Gradient Descent (`SGDRegressor` class in scikit-learn)** → Needed specifically when dealing with **very high-dimensional data**, where computing the matrix inverse becomes too slow.
> **Practical takeaway:** You don't need to worry about picking between them too often — but it's important to understand *why* Gradient Descent exists as an alternative, since it's not just an arbitrary "different way to do the same thing" — it solves a real computational scalability problem that OLS has.
 
---
 
## 9. Key Takeaways (Summary)
 
| Concept | Meaning |
|---|---|
| **Matrix $X$ (design matrix)** | All input feature values, with an extra column of 1's prepended (for the intercept term). Shape: $n \times (m+1)$ |
| **Matrix $\beta$** | All unknown coefficients to solve for: $\beta_0, \beta_1, \ldots, \beta_m$. Shape: $(m+1) \times 1$ |
| **Matrix $Y$** | Actual/true output values. Shape: $n \times 1$ |
| **Prediction equation** | $\hat Y = X\beta$ |
| **Loss function (matrix form)** | $E = (Y-X\beta)^T(Y-X\beta) = Y^TY - 2\beta^TX^TY + \beta^TX^TX\beta$ |
| **Solving process** | Differentiate $E$ w.r.t. $\beta$, set to 0, solve |
| **Final OLS formula** | $\beta = (X^TX)^{-1}X^TY$ |
| **Limitation of OLS** | Requires matrix inversion → $O(n^3)$ time complexity → slow for datasets with many input columns |
| **Alternative** | Gradient Descent — iterative, avoids matrix inversion, scales better to high dimensions |
| **scikit-learn classes** | `LinearRegression` (uses OLS) vs `SGDRegressor` (uses Gradient Descent) |
 
### Quick Mental Model
Think of the entire derivation as doing **exactly the same thing** as Simple Linear Regression's calculus derivation (set derivative of squared error to zero, solve for the unknowns) — just **generalized to handle many inputs at once using matrix notation** instead of plain algebra. The final formula $\beta = (X^TX)^{-1}X^TY$ is the direct, one-shot "plug in your data and get all your coefficients at once" solution — but it has a hidden cost (matrix inversion) that becomes expensive when you have a huge number of input features, which is exactly the gap that Gradient Descent fills.
 
