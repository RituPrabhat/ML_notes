## 1. Goal of PCA (Recap)
 
PCA is a **dimensionality reduction** technique.
 
- If you have high-dimensional data, PCA lets you bring it down to a lower dimension.
- While doing so, it tries to **preserve the "essence" of the data** — i.e., the structure/information the data represents stays intact.
- Because the essence is preserved, ML algorithms trained on the reduced (low-dimensional) data give **results similar to** what you'd get from the original high-dimensional data.
Today's focus: **the actual math** — what mathematical objective function PCA is solving, and how that solution is used in code.
 
---
 
## 2. Problem Formulation
 
### 2.1 Intuition with a simple example
 
Imagine a 2D dataset that you want to reduce to 1D. The data points are scattered along a certain diagonal direction.
 
- You want to find a **single axis** (a line through the origin) onto which you can project all the data points.
- You want this axis to be the one where, after projecting, you still get **"good" results** — meaning the variance of the projected points is nearly as large as the variance in the original 2D space.
### 2.2 Projecting a point onto a unit vector
 
- Treat every data point as a **vector** $\vec{x}$ (e.g., 2 components: x and y).
- Let $\hat{u}$ be a **unit vector** (direction only matters, magnitude = 1) representing the axis you're projecting onto.
- The **projection formula** of vector $x$ onto vector $u$ in general is:
$$\text{proj}_u(x) = \frac{u \cdot x}{|u|^2}\, u$$
 
- Since $u$ is a **unit vector**, $|u| = 1$, so this simplifies to:
$$\text{proj}_u(x) = (u^T x)\, u$$
 
- The **scalar quantity** $u^T x$ (a dot product) gives the **length of the projection** of that particular point along direction $u$.
- This dot product expands as: if $x = (x_1, x_2)$ and $u = (u_1, u_2)$, then
$$u^T x = u_1 x_1 + u_2 x_2$$
 
which is just a number (a scalar).
 
### 2.3 The key idea — maximize variance
 
- Every point, once projected onto $\hat{u}$, gives a scalar value.
- Collecting the scalar projections of **all** points gives you a new 1D dataset.
- **PCA's job:** choose the unit vector $\hat{u}$ such that the **variance of these projected scalar values is maximum**.
- Intuition: many possible directions could be chosen, but only one particular direction preserves the most "spread" (information) of the original data — that's the one PCA picks.
### 2.4 Formal variance calculation
 
For $n$ points, once projected onto $\hat u$:
 
$$\sigma^2_{\hat u} = \frac{1}{n}\sum_{i=1}^{n} \left(u^T x_i - \bar{x}\right)^2$$
 
Since data is usually **mean-centered** first (mean = 0), this simplifies to:
 
$$\sigma^2_{\hat u} = \frac{1}{n}\sum_{i=1}^{n} (u^T x_i)^2$$
 
### 2.5 The Objective Function of PCA
 
> **Find the unit vector $\hat u$ that maximizes:**
> $$\frac{1}{n}\sum_{i=1}^{n} (u^T x_i)^2 \quad \text{subject to } \|u\| = 1$$
 
This is a **constrained optimization problem**. Solving it (using a technique called the **Rayleigh Quotient**, which involves higher-level math not covered in this lecture) leads directly to eigenvectors/eigenvalues — explained next.
 
---
 
## 3. Covariance and Covariance Matrix
 
### 3.1 Why not just use Mean/Variance?
 
- **Mean** tells you the "center" of data — but two very differently spread datasets can have the exact same mean, so mean alone can't describe spread.
- **Variance** tells you how spread out data is along a **single axis** — but variance alone doesn't tell you the **relationship between two variables** (e.g., X and Y).
### 3.2 Covariance — definition
 
**Covariance** measures how two variables change together (their relationship).
 
$$\text{Cov}(X, Y) = \frac{1}{n}\sum_{i=1}^{n}(x_i - \bar x)(y_i - \bar y)$$
 
- **Positive covariance** → as X increases, Y also tends to increase.
- **Negative covariance** → as X increases, Y tends to decrease.
- Unlike **correlation** (which is bounded between −1 and +1), **covariance has no fixed range**.
### 3.3 Covariance Matrix
 
For a dataset with multiple features (say $X_1, X_2$), the covariance matrix is:
 
$$
\Sigma =
\begin{bmatrix}
\text{Var}(X_1) & \text{Cov}(X_1, X_2) \\
\text{Cov}(X_2, X_1) & \text{Var}(X_2)
\end{bmatrix}
$$
 
Key properties:
- **Diagonal elements** = variance of each individual feature.
- **Off-diagonal elements** = covariance between pairs of features.
- The matrix is always **symmetric** (since $\text{Cov}(X_1,X_2) = \text{Cov}(X_2,X_1)$).
- For **3 features (X, Y, Z)**, it becomes a 3×3 symmetric matrix with variances on the diagonal and pairwise covariances elsewhere.
**Why the covariance matrix is special:** It gives *complete* information about the dataset —
1. **Spread** — via the variances on the diagonal.
2. **Orientation/relationship** — via the covariances off the diagonal (whether features move together or oppositely).
---
 
## 4. Eigenvectors and Eigenvalues
 
### 4.1 Matrices as Linear Transformations
 
- A **matrix**, when applied to a set of points (vectors) in a coordinate system, **transforms** the entire coordinate space.
- Every point can be thought of as a vector; applying a matrix changes *all* vectors' directions and/or magnitudes (rotates, stretches, squashes, reflects the space).
- Example: The **identity matrix** $\begin{bmatrix}1&0\\0&1\end{bmatrix}$ applied to any vector produces **no change at all** — every point stays exactly where it is.
### 4.2 What is an Eigenvector?
 
- When a linear transformation (matrix) is applied to space, **most vectors change direction**.
- **Eigenvectors** are the special vectors whose **direction does NOT change** after the transformation — only their **magnitude (scale)** may change (they may stretch, shrink, or flip along the same line/span).
**Example:**
- Vector $(1,0,2)$ transformed becomes $(3,0,4)$ — magnitude increased (scaled by 3), but direction (span) stayed the same → it's an eigenvector with **eigenvalue = 3**.
- Vector $(-1,2,1)$ transformed becomes $(-1.5, 1.5, 2)$ — scaled by 1.5 → eigenvalue = **1.5**.
### 4.3 The Eigen Equation
 
$$A\,\vec{v} = \lambda\,\vec{v}$$
 
- $A$ = the matrix (transformation).
- $\vec v$ = the eigenvector.
- $\lambda$ = the eigenvalue (a scalar).
**Interpretation:** Applying the full matrix transformation to $\vec v$ has the *same effect* as simply multiplying $\vec v$ by a scalar $\lambda$ — meaning direction doesn't change, only scale does.
 
- For an $n$-dimensional linear transformation, you get **multiple eigenvector–eigenvalue pairs** (e.g., 2 pairs for a 2D transformation, 3 for 3D, etc.).
---
 
## 5. Eigendecomposition of the Covariance Matrix (The Core Connection to PCA)
 
This is the key insight that ties everything together:
 
- The **objective function** from Section 2.5 (maximize variance of projections, subject to $\|u\|=1$) — when solved mathematically using the **Rayleigh Quotient** technique — turns out to have a solution:
> **The unit vector $\hat u$ that maximizes variance is exactly the eigenvector of the covariance matrix that corresponds to the LARGEST eigenvalue.**
 
- In other words:
  - Take the covariance matrix of your data.
  - Compute its **eigenvectors and eigenvalues** (this is called **eigendecomposition**).
  - The eigenvector with the **largest eigenvalue** points in the direction of **maximum variance/spread** in the data.
  - That eigenvector **is** your first **Principal Component**.
This is *why* eigenvectors and eigenvalues are central to how PCA works.
 
---
 
## 6. Step-by-Step Algorithm: How to Solve PCA
 
Given a dataset with features $F_1, F_2, F_3$ (say, 3D data):
 
**Step 1 — Mean Centering**
- Subtract the mean of each column from every value in that column.
- This shifts the data so it's centered around the origin.
- *Not mandatory*, but empirically improves PCA's performance.
**Step 2 — Compute the Covariance Matrix**
- Calculate variance of each feature and covariance between every pair of features.
- In code: this is a single call, e.g. `np.cov()`.
**Step 3 — Eigendecomposition**
- Find the **eigenvectors** and **eigenvalues** of the covariance matrix.
- For 3D data → you get **3 eigenvectors** and **3 eigenvalues**.
- In code: use a linear algebra sub-library (e.g., `numpy.linalg`, which has a direct function for eigendecomposition).
**Step 4 — Rank and Select Principal Components**
- The eigenvector with the **largest** eigenvalue → **PC1** (first Principal Component).
- The eigenvector with the **second largest** eigenvalue → **PC2**, and so on.
- You choose how many PCs to keep depending on how many dimensions you want to reduce to (e.g., keep PC1+PC2 to go from 3D→2D, or just PC1 to go 3D→1D).
---
 
## 7. How to Transform the Original Points into the New (Reduced) Space
 
Once you have your chosen principal components (eigenvectors):
 
1. Stack the chosen eigenvectors as columns (say PC1 and PC2) — this becomes your **transformation matrix**.
2. To project your original data matrix ($X$) onto the new PC axes:
$$X_{\text{new}} = X \cdot (\text{PC matrix})$$
 
  (In practice: take the transpose of the PC vectors and do a dot product with the original data — the exact operation is: `new_data = np.dot(original_data, pc_vectors)`.)
 
3. **Shape check:**
   - If original data is $(n \times 3)$ (n points, 3 features) and you pick 2 PCs each of dimension 3 → PC matrix shape is $(3 \times 2)$.
   - Dot product: $(n \times 3) \cdot (3 \times 2) = (n \times 2)$ — your new reduced dataset.
4. The **target/label column** (if any) is simply copied over unchanged into the new reduced dataframe.
**In short: Transforming points = original data dot-producted with the (transposed) principal component vectors.** This same logic applies regardless of how many dimensions you're reducing from/to.
 
---
 
## 8. Code Demo Summary (Conceptual Walkthrough)
 
The lecture demonstrates this on a synthetic 3D dataset (3 feature columns + 1 target/label column for classification), reducing it from 3D → 2D:
 
1. **Visualize** the raw 3D data using a 3D scatter plot, colored by target label.
2. **Mean-center** the data (e.g., using `StandardScaler` or similar).
3. **Compute covariance matrix** via `np.cov()` on the 3 feature columns → a 3×3 symmetric matrix.
4. **Eigendecomposition** via a linear algebra function (e.g., `numpy.linalg.eig`) → get 3 eigenvectors and 3 eigenvalues.
5. **Select top 2 eigenvectors** (those with the 2 largest eigenvalues) → these become PC1 and PC2.
6. **Transform** the original data: dot product of original data with the transposed PC vectors → new data shape becomes $(n \times 2)$.
7. Combine the 2 new PC columns with the (unchanged) target column into a new dataframe.
8. **Visualize** the new 2D data (scatter plot of PC1 vs PC2, colored by target) — this shows the same data now represented in fewer dimensions while preserving class separation/structure.
---
 
## 9. Key Takeaways (Summary)
 
| Concept | What it means |
|---|---|
| **PCA's objective** | Find unit vector(s) that maximize variance of projected data |
| **Covariance Matrix** | Captures both spread (variance) and orientation (covariance) of data |
| **Eigenvector** | Direction unchanged by a linear transformation |
| **Eigenvalue** | The scaling factor applied along that eigenvector's direction |
| **Core PCA result** | Eigenvector of the covariance matrix with the **largest eigenvalue** = direction of maximum variance = **1st Principal Component** |
| **Algorithm** | Mean-center → Covariance matrix → Eigendecomposition → Rank by eigenvalue → Select top-k → Project data |
| **Transforming data** | `new_data = original_data · (PC_vectors)ᵀ` (dot product) |
 
### Quick Mental Model
Think of PCA as: *"Rotate my coordinate axes so that the new axes point in the directions where my data is most spread out — then just keep the top few axes and throw away the rest."* The covariance matrix tells you exactly how to find that rotation (via its eigenvectors), and the eigenvalues tell you *how much* spread exists along each new axis — so you rank and keep the biggest ones.
 
