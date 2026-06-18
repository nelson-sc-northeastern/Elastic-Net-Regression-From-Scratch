# Elastic Net Regression From Scratch

An end-to-end implementation of **Elastic Net Regression** from scratch using only Python and NumPy. This project incorporates two optimization techniques—**Gradient Descent** and **Coordinate Descent**—to fit a regularized linear model, evaluates performance on the Kaggle **House Prices: Advanced Regression Techniques** dataset, and benchmarks the results directly against `scikit-learn`.

Developed by **Scott Nelson**.

---

## Project Overview

Regularization is a fundamental technique in machine learning used to prevent overfitting by penalizing large coefficients. While **Ridge Regression ($L_2$)** induces weight-sharing and stabilizes highly correlated features, **Lasso Regression ($L_1$)** enforces sparsity by driving irrelevant coefficients to exactly zero. 

**Elastic Net Regression** combines both worlds, finding a middle-ground compromise that handles multicollinearity robustly while maintaining feature selection capabilities. This project implements the complete algorithm, hyperparameter grid search, and evaluation pipeline without relying on external machine learning frameworks for the core logic.

---

## Mathematical Formulation

The loss function $\mathcal{L}(\beta)$ minimized by Elastic Net matches the native implementation format of `scikit-learn` to ensure precise benchmarking:

$$\mathcal{L}(\beta) = \frac{1}{2n} \|y - X\beta\|^2_2 + \alpha \rho \|\beta\|_1 + \frac{\alpha (1 - \rho)}{2} \|\beta\|^2_2$$

Where:
* $n$ is the number of samples.
* $y$ is the target outcome vector.
* $X$ is the feature matrix.
* $\beta$ represents the feature coefficient weights.
* $\alpha$ (Alpha) controls the overall strength of the regularization penalty.
* $\rho$ (Mixing Parameter / $L_1$ Ratio) dictates the balance between Lasso and Ridge penalties ($0 \le \rho \le 1$).
    * $\rho = 1$ simplifies the model to **Pure Lasso Regression**.
    * $\rho = 0$ simplifies the model to **Pure Ridge Regression**.

---

## Optimization Techniques Implemented

Two distinct optimization paradigms were developed from scratch to minimize the loss function:

### 1. Gradient Descent
* Updates all parameter weights simultaneously across full-batch passes.
* Leverages the structural gradient of the Mean Squared Error (MSE) and $L_2$ terms, alongside sub-gradient updates for the non-differentiable $L_1$ norm.
* **Characteristics:** Achieves fast visual convergence per epoch but can experience instability near the absolute minimum without meticulous learning rate tuning.

### 2. Coordinate Descent
* Optimizes one feature parameter weight at a time while keeping all other weights fixed, looping systematically through features.
* Utilizes the analytical soft-thresholding operator for the $L_1$ penalty step.
* **Characteristics:** Mathematically elegant and less computationally intense per parameter adjustment step. In this project, it ultimately mapped a higher fidelity fit compared to standard Gradient Descent.

---

## Dataset & Preprocessing

The model is evaluated using the popular **Kaggle Housing Dataset** (House Prices: Advanced Regression Techniques). 

* **Initial Feature Space:** 36 numeric features.
* **Data Cleaning:** 17 unhelpful or structural categorical identifier elements (such as `YearBuilt`, `YearRemodAdd`, `GarageYrBlt`) were filtered out or omitted for pure regression testing.
* **Final Feature Space:** 19 clean, continuous numerical features utilized to predict the target variable: `SalePrice`.
* **Key Discoveries:**
    * *Strongest Predictors:* Garage capacity metrics, 1st Floor Square Footage (`1stFlrSF`), and General Living Area (`GrLivArea`).
    * *Weakest Predictors:* Pool Area (`PoolArea`) and Porch dimensions.

---

## Hyperparameter Tuning & Results

A systematic **Grid Search** framework was built to map out the optimal tuning landscape:
* **Alpha Range Tested:** $[0.0, 1.0]$
* **Mixing Parameter ($\rho$) Range Tested:** $[0.0, 0.9]$

### Optimal Configuration Found:
* **Optimal Alpha:** $\approx 0.2308$
* **Optimal Mixing Ratio ($\rho$):** $\approx 0.8769$

> **Insight:** The optimal mixing ratio highly favors **Lasso Regression**, confirming that feature selection and sparsity are critical for parsing out pricing noise in this real estate dataset.

### Performance Summary:
All three models (Gradient Descent from scratch, Coordinate Descent from scratch, and Sklearn) converged successfully to highly uniform baselines:

| Model Implementation | Mean Absolute Error off correct Sale Price | Notes |
| :--- | :--- | :--- |
| **Coordinate Descent (Scratch)** | $~ \$28,000$ | Best overall scratch performance; outperformed GD by $\sim 10\%$. |
| **Gradient Descent (Scratch)** | $\~ \$31,000$ | Fast initial convergence but plateaued slightly higher. |
| **Scikit-Learn ElasticNet** | $\~ \$28,000$ | Verified identical convergence to the scratch Coordinate Descent model. |

---

These dollar values may seem highly off but due to the location of the data for this housing market the actual prediction is about ~10% off the true sale price.

##  Repository Directory Structure


Elastic-Net-Regression-From-Scratch-main/
│
├── house-prices-advanced-regression-techniques 3/
│   └── train.csv                          # Kaggle Housing training data file
│
├── Elastic Net Regression Project.ipynb   # Step-by-step notebook (Scratch implementation, EDA, & Benchmarks)
├── Elastic Net Regression.pptx           # Project presentation slides
└── Elastic Net Regression.pdf            # PDF export of project details & performance metrics

## Getting Started
Prerequisites

To execute the notebook and verify the benchmarks locally, ensure you have python with the following mathematical dependencies installed:
Bash

pip install numpy pandas matplotlib scikit-learn jupyter

## Execution Steps

    Clone or extract this workspace.

    Navigate to the root directory where the files reside:
    Bash

    cd Elastic-Net-Regression-From-Scratch-main

    Boot up the local notebook server:
    Bash

    jupyter notebook

    Click and open Elastic Net Regression Project.ipynb and run the cells sequentially to observe data engineering, matrix updates, loss metrics tracking, and side-by-side benchmark evaluations.

## References

    Zou, H., & Hastie, T. (2005). Regularization and variable selection via the elastic net. Journal of the Royal Statistical Society.

    Kaggle Dataset: House Prices - Advanced Regression Techniques

```text
