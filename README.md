## Implementation of Decision Tree Regressor Model for Predicting the Salary of the Employee

### AIM

To write and verify a Python program to implement a Decision Tree Regressor model for predicting the salary of an employee.

### Equipments Required

* **Hardware:** PC / Laptop
* **Software:** Anaconda / Python 3.7+ Installation, Jupyter Notebook (with `numpy`, `matplotlib`, and `scikit-learn`)
* <img width="550" height="558" alt="image" src="https://github.com/user-attachments/assets/9beaf2ef-e442-4368-8fc1-bd747dc26ef4" />


---

### Algorithm

1. Prepare the dataset containing employee features $X$ (such as Years of Experience, Performance Rating, Education Level, and Weekly Hours) and continuous target variable $y$ (Salary).
2. Split the dataset into training and testing sets using `train_test_split`.
3. Initialize a `DecisionTreeRegressor` (using squared error / MSE as the splitting criterion, and optional constraints like `max_depth` to avoid overfitting). For a candidate split dividing a region $R$ into left and right subsets $R_L$ and $R_R$, it minimizes the sum of squared residuals:

$$\text{MSE}(R) = \frac{1}{\vert{}R\vert{}} \sum_{i \in R} (y_i - \hat{y}_R)^2, \qquad \hat{y}_R = \frac{1}{\vert{}R\vert{}} \sum_{i \in R} y_i$$


4. Train the regressor on the training data so each leaf node stores the mean salary of the training samples reaching that leaf.
5. Predict salaries on the test set and for a new employee profile, and compute evaluation metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), and $R^2$ score.
6. Visualize the step-wise regression predictions against actual salaries and plot the learned decision tree.

---

### Program

```python
"""
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Darshan J
RegisterNumber: 212224020008
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Features X: [Years_of_Experience, Performance_Rating (1-5), Education_Level (1=Bachelors, 2=Masters, 3=PhD)]
feature_names = ["Years_Experience", "Performance_Rating", "Education_Level"]

X = np.array([
    [1.0, 3.0, 1],
    [1.5, 3.5, 1],
    [2.0, 3.0, 1],
    [2.5, 4.0, 2],
    [3.0, 3.5, 1],
    [3.5, 4.0, 2],
    [4.0, 4.5, 2],
    [4.5, 4.0, 2],
    [5.0, 4.5, 2],
    [6.0, 4.0, 2],
    [6.5, 4.5, 3],
    [7.0, 5.0, 2],
    [8.0, 4.5, 3],
    [8.5, 4.0, 2],
    [9.0, 5.0, 3],
    [10.0, 4.5, 3],
    [11.0, 5.0, 3],
    [12.0, 4.5, 3],
    [5.5, 3.5, 2],
    [7.5, 4.0, 2],
], dtype=float)

# Target y: Annual Salary (in $1,000s)
y = np.array([
    38.0, 42.0, 44.0, 54.0, 50.0,
    60.0, 66.0, 65.0, 72.0, 76.0,
    85.0, 88.0, 96.0, 92.0, 106.0,
    112.0, 120.0, 125.0, 68.0, 84.0
], dtype=float)

# 2. Split into training (75%) and testing (25%) sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# 3. Initialize and train Decision Tree Regressor
regressor = DecisionTreeRegressor(criterion="squared_error", max_depth=3, random_state=42)
regressor.fit(X_train, y_train)

# 4. Predict on test set and for a new employee (6.0 yrs exp, 4.5 rating, Masters=2)
y_pred = regressor.predict(X_test)

new_employee = np.array([[6.0, 4.5, 2.0]])
predicted_salary = regressor.predict(new_employee)[0]

# 5. Evaluate model performance
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Predicted Salary for {new_employee[0]} -> ${predicted_salary:.2f}k")
print(f"Mean Absolute Error (MAE): ${mae:.2f}k")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R² Score: {r2:.4f}")

print("\nFeature Importances:")
for name, imp in zip(feature_names, regressor.feature_importances_):
    print(f"  {name:20s}: {imp:.4f}")

# 6. Plot Actual vs. Predicted Salaries and Tree Structure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

ax1.scatter(y_test, y_pred, color="blue", s=70, label="Test Samples")
min_val = min(y_test.min(), y_pred.min()) - 5
max_val = max(y_test.max(), y_pred.max()) + 5
ax1.plot([min_val, max_val], [min_val, max_val], "r--", label="Ideal Fit")
ax1.set_title("Decision Tree Regressor: Actual vs. Predicted Salary")
ax1.set_xlabel("Actual Salary ($1,000s)")
ax1.set_ylabel("Predicted Salary ($1,000s)")
ax1.legend()
ax1.grid(True)

plot_tree(
    regressor,
    feature_names=feature_names,
    filled=True,
    rounded=True,
    fontsize=8,
    ax=ax2,
)
ax2.set_title("Learned Regression Tree")

plt.tight_layout()
plt.show()

```

---

### Output

```text
Predicted Salary for [6.  4.5 2. ] -> $76.00k
Mean Absolute Error (MAE): $4.83k
Mean Squared Error (MSE): 34.4833
R² Score: 0.9462

Feature Importances:
  Years_Experience    : 0.9686
  Performance_Rating  : 0.0000
  Education_Level     : 0.0314

```

---

### Result

Thus, the program to implement the Decision Tree Regressor model for predicting the salary of the employee was written and verified using Python programming.
