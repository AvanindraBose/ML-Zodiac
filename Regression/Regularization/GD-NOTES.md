# Batch Gradient Descent (BGD) vs Stochastic Gradient Descent (SGD)

Gradient Descent is an optimization algorithm used to minimize a loss function by repeatedly updating model parameters in the direction opposite to the gradient.

For Linear Regression or Ridge Regression, the parameter update rule is:

$$
w = w - \eta \frac{\partial L}{\partial w}
$$

where:

* $w$ = model weights
* $\eta$ = learning rate
* $L$ = loss function

---

# 1. Batch Gradient Descent (BGD)

In **Batch Gradient Descent**, the gradient is computed using the **entire training dataset** before every weight update.

Suppose:

$$
X \in \mathbb{R}^{n \times m}
$$

where:

* $n$ = number of samples
* $m$ = number of features

After adding the intercept column:

$$
X \in \mathbb{R}^{n \times (m+1)}
$$

Weights:

$$
w \in \mathbb{R}^{m+1}
$$

Predictions:

$$
\hat{y} = Xw
$$

Residual/Error:

$$
e = \hat{y} - y
$$

For Ridge Regression, the gradient is:

$$
\frac{\partial L}{\partial w}
=============================

\frac{1}{n}X^T(Xw-y)
+
\alpha w
$$

Since the intercept term should not be regularized:

$$
\text{penalty}[0] = 0
$$

Thus the update rule becomes:

$$
w
=

## w

\eta
\left(
\frac{1}{n}X^T(Xw-y)
+
\alpha w
\right)
$$

## BGD Code Pattern

```python
for epoch in range(epochs):

    error = np.dot(X_train, weights) - y_train

    penalty = alpha * weights
    penalty[0] = 0

    gradient = (
        np.dot(X_train.T, error) / X_train.shape[0]
        + penalty
    )

    weights = weights - learning_rate * gradient
```

### Key Points

* Uses the entire dataset to compute one gradient.
* Performs one weight update per epoch.
* Produces stable and accurate gradients.
* Loss decreases smoothly.
* Computationally expensive for large datasets.
* Requires loading all training samples for each update.

---

# 2. Stochastic Gradient Descent (SGD)

In **Stochastic Gradient Descent**, the gradient is computed using **one training sample at a time**.

For a single sample:

$$
x_i \in \mathbb{R}^{m+1}
$$

Prediction:

$$
\hat{y}_i = x_i^T w
$$

Residual/Error:

$$
e_i = \hat{y}_i - y_i
$$

Gradient for Ridge Regression:

$$
\frac{\partial L}{\partial w}
=============================

x_i(\hat{y}_i - y_i)
+
\alpha w
$$

Again, the intercept is not regularized:

$$
\text{penalty}[0] = 0
$$

Update rule:

$$
w
=

## w

\eta
\left(
x_i(\hat{y}_i-y_i)
+
\alpha w
\right)
$$

## SGD Code Pattern

```python
for epoch in range(epochs):

    indices = np.random.permutation(X_train.shape[0])

    for idx in indices:

        error = (
            np.dot(X_train[idx], weights)
            - y_train[idx]
        )

        penalty = alpha * weights
        penalty[0] = 0

        gradient = (
            X_train[idx] * error
            + penalty
        )

        weights = (
            weights
            - learning_rate * gradient
        )
```

### Key Points

* Uses one sample for each gradient computation.
* Performs $n$ updates per epoch.
* Updates are much more frequent than BGD.
* Faster and more memory-efficient for large datasets.
* Gradient estimates are noisy.
* Loss fluctuates instead of decreasing smoothly.
* Noise can help escape saddle points and poor regions of the optimization landscape.

---

# Why Does SGD Perform $n$ Updates per Epoch?

Suppose:

$$
n = 10,000
$$

training samples.

### Batch Gradient Descent

Compute gradient using all 10,000 rows:

$$
\nabla L =
\frac{1}{n}
X^T(Xw-y)
$$

Then perform:

$$
1
$$

weight update.

Therefore:

$$
\text{Updates per epoch} = 1
$$

### Stochastic Gradient Descent

Process rows one by one:

$$
x_1 \rightarrow \text{update}
$$

$$
x_2 \rightarrow \text{update}
$$

$$
x_3 \rightarrow \text{update}
$$

$$
\vdots
$$

$$
x_{10000} \rightarrow \text{update}
$$

Thus:

$$
\text{Updates per epoch} = n
$$

---

# BGD vs SGD Comparison

| Aspect               | Batch Gradient Descent (BGD) | Stochastic Gradient Descent (SGD) |
| -------------------- | ---------------------------- | --------------------------------- |
| Data used per update | Entire dataset               | One sample                        |
| Updates per epoch    | 1                            | $n$                               |
| Speed per update     | Slow                         | Fast                              |
| Memory usage         | High                         | Low                               |
| Gradient estimate    | Accurate                     | Noisy                             |
| Loss curve           | Smooth                       | Fluctuating                       |
| Convergence path     | Stable                       | Zig-zag                           |
| Scalability          | Poor for large datasets      | Excellent                         |
| Best suited for      | Small/Medium datasets        | Large datasets                    |
| Computational cost   | High                         | Low                               |

---

# Interview One-Liner

> Batch Gradient Descent computes the exact gradient using the entire dataset and performs one update per epoch, whereas Stochastic Gradient Descent approximates the gradient using a single sample and performs $n$ updates per epoch, making it significantly more scalable for large datasets but noisier during optimization.
