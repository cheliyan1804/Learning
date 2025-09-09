import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

housing = pd.read_csv('housing.csv')
#1203
print("Data head:")
print(housing.head())
print("\nData info:")
housing.info()

median_bedrooms = housing['total_bedrooms'].median()
housing['total_bedrooms'].fillna(median_bedrooms, inplace=True)

print("\nMissing values after cleaning:")
print(housing.isnull().sum())

housing.hist(bins=50, figsize=(20, 15))
plt.show()

from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd
import numpy as np 

housing['rooms_per_household'] = housing['total_rooms'] / housing['households']
housing['bedrooms_per_room'] = housing['total_bedrooms'] / housing['total_rooms']
housing['population_per_household'] = housing['population'] / housing['households']

housing_cat = housing[['ocean_proximity']]
housing_cat_encoded = pd.get_dummies(housing_cat, drop_first=True)
housing = pd.concat([housing.drop('ocean_proximity', axis=1), housing_cat_encoded], axis=1)

housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6.0, np.inf],
                               labels=[1, 2, 3, 4, 5])

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)

housing_train = strat_train_set.drop('median_house_value', axis=1)
housing_labels = strat_train_set['median_house_value'].copy()

housing_test = strat_test_set.drop('median_house_value', axis=1)
housing_test_labels = strat_test_set['median_house_value'].copy()

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np

housing_num_train = housing_train.drop(['ocean_proximity_INLAND', 'ocean_proximity_ISLAND', 'ocean_proximity_NEAR BAY', 'ocean_proximity_NEAR OCEAN'], axis=1)

scaler = StandardScaler()
housing_num_scaled = scaler.fit_transform(housing_num_train)
housing_train_prepared = pd.concat([pd.DataFrame(housing_num_scaled, columns=housing_num_train.columns), housing_train[['ocean_proximity_INLAND', 'ocean_proximity_ISLAND', 'ocean_proximity_NEAR BAY', 'ocean_proximity_NEAR OCEAN']].reset_index(drop=True)], axis=1)


forest_reg = RandomForestRegressor(n_estimators=100, random_state=42)
forest_reg.fit(housing_train_prepared, housing_labels)

housing_predictions = forest_reg.predict(housing_train_prepared)
forest_rmse = np.sqrt(mean_squared_error(housing_labels, housing_predictions))
print("\nTraining RMSE:", forest_rmse)

housing_num_test = housing_test.drop(['ocean_proximity_INLAND', 'ocean_proximity_ISLAND', 'ocean_proximity_NEAR BAY', 'ocean_proximity_NEAR OCEAN'], axis=1)
housing_num_test_scaled = scaler.transform(housing_num_test)
housing_test_prepared = pd.concat([pd.DataFrame(housing_num_test_scaled, columns=housing_num_test.columns), housing_test[['ocean_proximity_INLAND', 'ocean_proximity_ISLAND', 'ocean_proximity_NEAR BAY', 'ocean_proximity_NEAR OCEAN']].reset_index(drop=True)], axis=1)


final_predictions = forest_reg.predict(housing_test_prepared)
final_rmse = np.sqrt(mean_squared_error(housing_test_labels, final_predictions))
print("Final Test RMSE:", final_rmse)


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('housing.csv')

print("--- Statistical Summary of the DataFrame ---")
print(df.describe())
print("\n")

median_bedrooms = df['total_bedrooms'].median()
df['total_bedrooms'].fillna(median_bedrooms, inplace=True)

print("--- Count of non-null values after handling missing values ---")
df.info()
print("\n")

df.hist(column='median_house_value', bins=50)
plt.title('Distribution of Median House Value')
plt.xlabel('Median House Value')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('median_house_value_histogram.png')
plt.clf()

plt.figure(figsize=(10, 7))
plt.scatter(
    df["longitude"],
    df["latitude"],
    alpha=0.4,
    s=df["population"] / 100,
    c=df["median_house_value"],
    cmap='jet'
)
plt.colorbar(label='Median House Value')
plt.title('Geographic Scatter Plot of Housing Districts')
plt.ylabel('Latitude', fontsize=14)
plt.xlabel('Longitude', fontsize=14)
plt.legend()
plt.tight_layout()
plt.savefig('geographic_scatter_plot.png')

print("Histograms and scatter plots have been generated and saved as 'median_house_value_histogram.png' and 'geographic_scatter_plot.png'.")


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

df = pd.read_csv('housing.csv')

print("--- Statistical Summary of the DataFrame ---")
print(df.describe())
print("\n")

median_bedrooms = df['total_bedrooms'].median()
df['total_bedrooms'].fillna(median_bedrooms, inplace=True)

print("--- Count of non-null values after handling missing values ---")
df.info()
print("\n")

df.hist(column='median_house_value', bins=50)
plt.title('Distribution of Median House Value')
plt.xlabel('Median House Value')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('median_house_value_histogram.png')
plt.clf()

plt.figure(figsize=(10, 7))
plt.scatter(
    df["longitude"],
    df["latitude"],
    alpha=0.4,
    s=df["population"] / 100,
    c=df["median_house_value"],
    cmap='jet'
)
plt.colorbar(label='Median House Value')
plt.title('Geographic Scatter Plot of Housing Districts')
plt.ylabel('Latitude', fontsize=14)
plt.xlabel('Longitude', fontsize=14)
plt.legend()
plt.tight_layout()
plt.savefig('geographic_scatter_plot.png')
plt.clf()

df['income_cat'] = pd.cut(df['median_income'],
                           bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                           labels=[1, 2, 3, 4, 5])
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(df, df['income_cat']):
    train_set = df.loc[train_index]
    test_set = df.loc[test_index]

X_train = train_set.drop('median_house_value', axis=1)
y_train = train_set['median_house_value']
X_test = test_set.drop('median_house_value', axis=1)
y_test = test_set['median_house_value']

numerical_features = X_train.drop('ocean_proximity', axis=1).columns
categorical_features = ['ocean_proximity']

preprocessor = ColumnTransformer(
    transformers=[
        ('numerical', StandardScaler(), numerical_features),
        ('categorical', OneHotEncoder(), categorical_features)
    ])

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

lin_reg = LinearRegression()
lin_reg.fit(X_train_processed, y_train)

predictions_train = lin_reg.predict(X_train_processed)
lin_rmse_train = np.sqrt(mean_squared_error(y_train, predictions_train))

predictions_test = lin_reg.predict(X_test_processed)
lin_rmse_test = np.sqrt(mean_squared_error(y_test, predictions_test))

print("Histograms and scatter plots have been generated and saved as 'median_house_value_histogram.png' and 'geographic_scatter_plot.png'.")
print(f"Training set RMSE: {lin_rmse_train}")
print(f"Test set RMSE: {lin_rmse_test}")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

df = pd.read_csv('housing.csv')

print("--- Statistical Summary of the DataFrame ---")
print(df.describe())
print("\n")

median_bedrooms = df['total_bedrooms'].median()
df['total_bedrooms'].fillna(median_bedrooms, inplace=True)
print("--- Count of non-null values after handling missing values ---")
df.info()
print("\n")

df.hist(column='median_house_value', bins=50)
plt.title('Distribution of Median House Value')
plt.xlabel('Median House Value')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('median_house_value_histogram.png')
plt.clf()
plt.figure(figsize=(10, 7))
plt.scatter(
    df["longitude"],
    df["latitude"],
    alpha=0.4,
    s=df["population"] / 100,
    c=df["median_house_value"],
    cmap='jet'
)
plt.colorbar(label='Median House Value')
plt.title('Geographic Scatter Plot of Housing Districts')
plt.ylabel('Latitude', fontsize=14)
plt.xlabel('Longitude', fontsize=14)
plt.legend()
plt.tight_layout()
plt.savefig('geographic_scatter_plot.png')
plt.clf()

df['income_cat'] = pd.cut(df['median_income'],
                           bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                           labels=[1, 2, 3, 4, 5])
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(df, df['income_cat']):
    train_set = df.loc[train_index]
    test_set = df.loc[test_index]
X_train = train_set.drop('median_house_value', axis=1)
y_train = train_set['median_house_value']
X_test = test_set.drop('median_house_value', axis=1)
y_test = test_set['median_house_value']
numerical_features = X_train.drop('ocean_proximity', axis=1).columns
categorical_features = ['ocean_proximity']
preprocessor = ColumnTransformer(
    transformers=[
        ('numerical', StandardScaler(), numerical_features),
        ('categorical', OneHotEncoder(), categorical_features)
    ])
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(X_train_processed, y_train)

predictions_train = tree_reg.predict(X_train_processed)
tree_rmse_train = np.sqrt(mean_squared_error(y_train, predictions_train))
predictions_test = tree_reg.predict(X_test_processed)
tree_rmse_test = np.sqrt(mean_squared_error(y_test, predictions_test))

print("Histograms and scatter plots have been generated and saved as 'median_house_value_histogram.png' and 'geographic_scatter_plot.png'.")
print(f"Decision Tree Training set RMSE: {tree_rmse_train}")
print(f"Decision Tree Test set RMSE: {tree_rmse_test}")


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

df = pd.read_csv('housing.csv')

print("--- Statistical Summary of the DataFrame ---")
print(df.describe())
print("\n")

median_bedrooms = df['total_bedrooms'].median()
df['total_bedrooms'].fillna(median_bedrooms, inplace=True)
print("--- Count of non-null values after handling missing values ---")
df.info()
print("\n")

df.hist(column='median_house_value', bins=50)
plt.title('Distribution of Median House Value')
plt.xlabel('Median House Value')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('median_house_value_histogram.png')
plt.clf()
plt.figure(figsize=(10, 7))
plt.scatter(
    df["longitude"],
    df["latitude"],
    alpha=0.4,
    s=df["population"] / 100,
    c=df["median_house_value"],
    cmap='jet'
)
plt.colorbar(label='Median House Value')
plt.title('Geographic Scatter Plot of Housing Districts')
plt.ylabel('Latitude', fontsize=14)
plt.xlabel('Longitude', fontsize=14)
plt.legend()
plt.tight_layout()
plt.savefig('geographic_scatter_plot.png')
plt.clf()

df['income_cat'] = pd.cut(df['median_income'],
                           bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                           labels=[1, 2, 3, 4, 5])
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(df, df['income_cat']):
    train_set = df.loc[train_index]
    test_set = df.loc[test_index]
X_train = train_set.drop('median_house_value', axis=1)
y_train = train_set['median_house_value']
X_test = test_set.drop('median_house_value', axis=1)
y_test = test_set['median_house_value']
numerical_features = X_train.drop('ocean_proximity', axis=1).columns
categorical_features = ['ocean_proximity']
preprocessor = ColumnTransformer(
    transformers=[
        ('numerical', StandardScaler(), numerical_features),
        ('categorical', OneHotEncoder(), categorical_features)
    ])
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

forest_reg = RandomForestRegressor(n_estimators=100, random_state=42)
forest_reg.fit(X_train_processed, y_train)

predictions_train = forest_reg.predict(X_train_processed)
forest_rmse_train = np.sqrt(mean_squared_error(y_train, predictions_train))
predictions_test = forest_reg.predict(X_test_processed)
forest_rmse_test = np.sqrt(mean_squared_error(y_test, predictions_test))

print("Histograms and scatter plots have been generated and saved as 'median_house_value_histogram.png' and 'geographic_scatter_plot.png'.")
print(f"Random Forest Training set RMSE: {forest_rmse_train}")
print(f"Random Forest Test set RMSE: {forest_rmse_test}")


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

df = pd.read_csv('housing.csv')

print("--- Statistical Summary of the DataFrame ---")
print(df.describe())
print("\n")

median_bedrooms = df['total_bedrooms'].median()
df['total_bedrooms'].fillna(median_bedrooms, inplace=True)
print("--- Count of non-null values after handling missing values ---")
df.info()
print("\n")

df.hist(column='median_house_value', bins=50)
plt.title('Distribution of Median House Value')
plt.xlabel('Median House Value')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('median_house_value_histogram.png')
plt.clf()
plt.figure(figsize=(10, 7))
plt.scatter(
    df["longitude"],
    df["latitude"],
    alpha=0.4,
    s=df["population"] / 100,
    c=df["median_house_value"],
    cmap='jet'
)
plt.colorbar(label='Median House Value')
plt.title('Geographic Scatter Plot of Housing Districts')
plt.ylabel('Latitude', fontsize=14)
plt.xlabel('Longitude', fontsize=14)
plt.legend()
plt.tight_layout()
plt.savefig('geographic_scatter_plot.png')
plt.clf()

df['income_cat'] = pd.cut(df['median_income'],
                           bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                           labels=[1, 2, 3, 4, 5])
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(df, df['income_cat']):
    train_set = df.loc[train_index]
    test_set = df.loc[test_index]
X_train = train_set.drop('median_house_value', axis=1)
y_train = train_set['median_house_value']
X_test = test_set.drop('median_house_value', axis=1)
y_test = test_set['median_house_value']
numerical_features = X_train.drop('ocean_proximity', axis=1).columns
categorical_features = ['ocean_proximity']
preprocessor = ColumnTransformer(
    transformers=[
        ('numerical', StandardScaler(), numerical_features),
        ('categorical', OneHotEncoder(), categorical_features)
    ])
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [3, 5, 7]}
]
forest_reg = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train_processed, y_train)

best_params = grid_search.best_params_
best_score = grid_search.best_score_
print("Best hyperparameters found: ", best_params)
print("Best cross-validation score (negative MSE): ", best_score)
print(f"Best cross-validation RMSE: {np.sqrt(-best_score)}")

final_model = grid_search.best_estimator_
predictions_test = final_model.predict(X_test_processed)
final_rmse = np.sqrt(mean_squared_error(y_test, predictions_test))
print(f"Final test set RMSE with best model: {final_rmse}")


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from scipy.stats import randint

df = pd.read_csv('housing.csv')

print("--- Statistical Summary of the DataFrame ---")
print(df.describe())
print("\n")

median_bedrooms = df['total_bedrooms'].median()
df['total_bedrooms'].fillna(median_bedrooms, inplace=True)
print("--- Count of non-null values after handling missing values ---")
df.info()
print("\n")

df.hist(column='median_house_value', bins=50)
plt.title('Distribution of Median House Value')
plt.xlabel('Median House Value')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('median_house_value_histogram.png')
plt.clf()
plt.figure(figsize=(10, 7))
plt.scatter(
    df["longitude"],
    df["latitude"],
    alpha=0.4,
    s=df["population"] / 100,
    c=df["median_house_value"],
    cmap='jet'
)
plt.colorbar(label='Median House Value')
plt.title('Geographic Scatter Plot of Housing Districts')
plt.ylabel('Latitude', fontsize=14)
plt.xlabel('Longitude', fontsize=14)
plt.legend()
plt.tight_layout()
plt.savefig('geographic_scatter_plot.png')
plt.clf()

df['income_cat'] = pd.cut(df['median_income'],
                           bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                           labels=[1, 2, 3, 4, 5])
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(df, df['income_cat']):
    train_set = df.loc[train_index]
    test_set = df.loc[test_index]
X_train = train_set.drop('median_house_value', axis=1)
y_train = train_set['median_house_value']
X_test = test_set.drop('median_house_value', axis=1)
y_test = test_set['median_house_value']
numerical_features = X_train.drop('ocean_proximity', axis=1).columns
categorical_features = ['ocean_proximity']
preprocessor = ColumnTransformer(
    transformers=[
        ('numerical', StandardScaler(), numerical_features),
        ('categorical', OneHotEncoder(), categorical_features)
    ])
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

param_distribs = {
    'n_estimators': randint(low=1, high=200),
    'max_features': randint(low=1, high=8),
}
forest_reg = RandomForestRegressor(random_state=42)
rnd_search = RandomizedSearchCV(forest_reg, param_distributions=param_distribs,
                                n_iter=10, cv=5, scoring='neg_mean_squared_error',
                                random_state=42)
rnd_search.fit(X_train_processed, y_train)

best_params = rnd_search.best_params_
best_score = rnd_search.best_score_
print("Best hyperparameters found: ", best_params)
print("Best cross-validation score (negative MSE): ", best_score)
print(f"Best cross-validation RMSE: {np.sqrt(-best_score)}")

final_model = rnd_search.best_estimator_
predictions_test = final_model.predict(X_test_processed)
final_rmse = np.sqrt(mean_squared_error(y_test, predictions_test))
print(f"Final test set RMSE with best model: {final_rmse}") 