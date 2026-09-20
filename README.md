# House Price Prediction — Regression Analysis

Predicting median house prices using multiple regression techniques, with a focus on handling multicollinearity and improving model generalization through regularization.

## Objective
Pre-process real estate data and build a predictive model for median housing prices (PRICE), identify the key factors driving housing values, and translate the findings into practical recommendations for buyers and sellers.

## Dataset
506 records, 13 features including:
- **CRIM** – per capita crime rate
- **RM** – average number of rooms per dwelling
- **NOX** – nitric oxide concentration (pollution)
- **LSTAT** – % lower status population
- **DIS** – distance to employment centres
- **PTRATIO** – pupil-teacher ratio
- **PRICE** – median home value (target variable)

## Methodology
1. **Data Cleaning** — Imputed skewed missing values (CRIM, ZN, INDUS, AGE) using median, and LSTAT using mean; encoded the RIVER dummy variable.
2. **Baseline Model** — Fitted an OLS regression model as a baseline.
3. **Validation** — Ran 10-Fold Cross-Validation to check model stability.
4. **Optimization** — Tuned a Stochastic Gradient Descent (SGD) regressor via GridSearchCV.
5. **Regularization** — Applied Ridge, LASSO, and Elastic Net regression (with tuned alpha/l1_ratio) to address multicollinearity detected via correlation heatmap.

## Results

| Model | Test MSE / R² |
|---|---|
| OLS (baseline) | R² = 0.732 (train) |
| 10-Fold CV | Mean R² = 0.704 |
| Tuned SGD | MSE = 0.409 |
| Ridge (α=19) | MSE = 0.187 |
| LASSO (α=0.1) | MSE = 0.203 |
| **Elastic Net (α=0.1, l1_ratio=0.1)** | **MSE = 0.187 (best)** |

Elastic Net gave the strongest and most stable performance, effectively balancing multicollinearity reduction with predictive accuracy.

## Key Insights
- **RM** (number of rooms) is the strongest positive driver of price.
- **LSTAT** (% lower-status population) is the strongest negative driver.
- Pollution (NOX), distance from employment centres (DIS), and pupil-teacher ratio (PTRATIO) also reduce price.
- Proximity to the river shows a modest positive effect on price.

## Recommendations
- **Buyers**: Prioritize homes with more rooms, lower crime, and lower pollution for long-term value.
- **Sellers**: Highlight room count, low crime, and proximity to amenities/river when listing.

## Tools & Libraries
Python, Pandas, NumPy, scikit-learn, statsmodels, Matplotlib, Seaborn
