import pandas as pd
import joblib
import numpy as np

model = joblib.load('california_housing_model.pkl')
preprocessor = joblib.load('california_housing_preprocessor.pkl')

new_data = pd.DataFrame([{
    'longitude': -122.27,
    'latitude': 37.83,
    'housing_median_age': 51.0,
    'total_rooms': 2665.0,
    'total_bedrooms': 574.0,
    'population': 1258.0, 
    'households': 536.0,
    'median_income': 2.7303,
    'ocean_proximity': 'NEAR BAY'
}])

new_data['income_cat'] = pd.cut(new_data['median_income'],
                                bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                                labels=[1, 2, 3, 4, 5])

new_data_processed = preprocessor.transform(new_data)

predicted_price = model.predict(new_data_processed)

print(f"The predicted median house value is: ${predicted_price[0]:,.2f}")