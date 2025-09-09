import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import StratifiedShuffleSplit, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from scipy.stats import randint

df = pd.read_csv('housing.csv')

median_bedrooms = df['total_bedrooms'].median()
df['total_bedrooms'].fillna(median_bedrooms, inplace=True)
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

param_distribs = {
    'n_estimators': randint(low=1, high=200),
    'max_features': randint(low=1, high=8),
}
forest_reg = RandomForestRegressor(random_state=42)
rnd_search = RandomizedSearchCV(forest_reg, param_distributions=param_distribs,
                                n_iter=10, cv=5, scoring='neg_mean_squared_error',
                                random_state=42)
rnd_search.fit(X_train_processed, y_train)

final_model = rnd_search.best_estimator_

joblib.dump(final_model, 'california_housing_model.pkl')
joblib.dump(preprocessor, 'california_housing_preprocessor.pkl')

print("Model and preprocessor saved successfully.")