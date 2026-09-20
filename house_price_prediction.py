"""
House Price Prediction — Regression Analysis
Predicts median house prices (PRICE) using OLS, SGD, Ridge, LASSO,
and Elastic Net regression, with cross-validation and hyperparameter tuning.
"""

# ============================================================
# 1. Import Libraries
# ============================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression, SGDRegressor, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler


# ============================================================
# 2. Load and Explore Data
# ============================================================
data = pd.read_csv("Median_Prices.csv")
print(data.head())
print(data.info())
print(data.isnull().sum())


# ============================================================
# 3. Handle Missing Values
# ============================================================
# Impute skewed variables with median
data['CRIM'] = data['CRIM'].fillna(data['CRIM'].median())
data['ZN'] = data['ZN'].fillna(data['ZN'].median())
data['INDUS'] = data['INDUS'].fillna(data['INDUS'].median())
data['AGE'] = data['AGE'].fillna(data['AGE'].median())

# Impute normally distributed variable with mean
data['LSTAT'] = data['LSTAT'].fillna(data['LSTAT'].mean())

# Encode RIVER as dummy variable
data['RIVER'] = data['RIVER'].astype('object')
data['RIVER'] = pd.get_dummies(data=data['RIVER'], drop_first=True)
data['RIVER'] = data['RIVER'].fillna(data['RIVER'].mode()[0])
data['RIVER'] = data['RIVER'].astype('float')

print(data.isnull().sum())  # confirm no missing values remain


# ============================================================
# 4. Baseline Model — OLS Regression
# ============================================================
X = data.iloc[:, :12]
X = sm.add_constant(X).astype("float")
y = data['PRICE']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10, test_size=0.2)

MLR_model = sm.OLS(y_train, X_train).fit()
print(MLR_model.summary())

linreg = LinearRegression()
MLR_model = linreg.fit(X_train, y_train)

train_pred = MLR_model.predict(X_train)
test_pred = MLR_model.predict(X_test)

print("OLS Train MSE:", mean_squared_error(y_train, train_pred))
print("OLS Test MSE:", mean_squared_error(y_test, test_pred))


# ============================================================
# 5. 10-Fold Cross-Validation
# ============================================================
model = LinearRegression()
kf = KFold(n_splits=10, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=kf, scoring="r2")

print("R² scores per fold:", scores)
print("Average R² score:", np.mean(scores))


# ============================================================
# 6. Gradient Descent (SGD) — Hyperparameter Tuning
# ============================================================
X = data.iloc[:, :12]
y = data['PRICE']

X_scaler = StandardScaler()
X = X_scaler.fit_transform(X)
y = (y - y.mean()) / y.std()  # standardize target

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10, test_size=0.2)

sgd = SGDRegressor(max_iter=1000, tol=1e-3, penalty=None, learning_rate='constant', random_state=42)
param_grid = {'eta0': np.linspace(0.001, 1.0, 10)}
grid_search = GridSearchCV(estimator=sgd, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)

print("Best SGD learning rate:", grid_search.best_params_)

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
print("Tuned SGD Test MSE:", mean_squared_error(y_test, y_pred))


# ============================================================
# 7. Correlation Heatmap (Multicollinearity Check)
# ============================================================
df_features = data.iloc[:, :12]
plt.figure(figsize=(15, 10))
sns.heatmap(df_features.corr(), annot=True, annot_kws={"size": 10})
plt.yticks(rotation='horizontal', fontsize=15)
plt.xticks(fontsize=15)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()


# ============================================================
# 8. Regularization — Ridge Regression
# ============================================================
X = data.iloc[:, :12]
y = data['PRICE']
X = StandardScaler().fit_transform(X)
y = (y - y.mean()) / y.std()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=110, test_size=0.2)

ridge_grid = GridSearchCV(estimator=Ridge(), param_grid=[{'alpha': np.linspace(0.1, 100, 1000)}], cv=10)
ridge_grid.fit(X_train, y_train)
print("Best alpha (Ridge):", ridge_grid.best_params_)

ridge_model = Ridge(alpha=ridge_grid.best_params_['alpha']).fit(X_train, y_train)
print("Ridge Train MSE:", mean_squared_error(y_train, ridge_model.predict(X_train)))
print("Ridge Test MSE:", mean_squared_error(y_test, ridge_model.predict(X_test)))


# ============================================================
# 9. Regularization — LASSO Regression
# ============================================================
lasso_grid = GridSearchCV(estimator=Lasso(), param_grid=[{'alpha': np.linspace(0.1, 100, 1000)}], cv=10)
lasso_grid.fit(X_train, y_train)
print("Best alpha (LASSO):", lasso_grid.best_params_)

lasso_model = Lasso(alpha=lasso_grid.best_params_['alpha']).fit(X_train, y_train)
print("LASSO Train MSE:", mean_squared_error(y_train, lasso_model.predict(X_train)))
print("LASSO Test MSE:", mean_squared_error(y_test, lasso_model.predict(X_test)))

df_lasso_coeff = pd.DataFrame({'Variable': df_features.columns, 'Coefficient': lasso_model.coef_})
print("Variables eliminated by LASSO:", df_lasso_coeff.Variable[df_lasso_coeff.Coefficient == 0].tolist())


# ============================================================
# 10. Regularization — Elastic Net Regression
# ============================================================
enet_grid = GridSearchCV(
    estimator=ElasticNet(),
    param_grid=[{'alpha': np.linspace(0.1, 10, 100), 'l1_ratio': np.linspace(0.1, 1, 100)}],
    cv=10
)
enet_grid.fit(X_train, y_train)
print("Best params (Elastic Net):", enet_grid.best_params_)

enet_model = ElasticNet(**enet_grid.best_params_).fit(X_train, y_train)
print("Elastic Net Train MSE:", mean_squared_error(y_train, enet_model.predict(X_train)))
print("Elastic Net Test MSE:", mean_squared_error(y_test, enet_model.predict(X_test)))


# ============================================================
# 11. Final Model Comparison & Conclusion
# ============================================================
# Elastic Net (alpha=0.1, l1_ratio=0.1) gave the best generalization,
# closely followed by Ridge Regression (alpha=19).
# Key price drivers: RM (+), LSTAT (-), DIS (-), PTRATIO (-), NOX (-)
#
# Recommendations:
# - Buyers: prioritize more rooms, lower crime, lower pollution for long-term value
# - Sellers: highlight room count, low crime, proximity to river/amenities
