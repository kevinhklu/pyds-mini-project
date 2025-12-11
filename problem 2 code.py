import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print("\n")
print("QUESTION 2: Can weather predict next day's traffic?")

X_weather = dataset_2[['High Temp', 'Low Temp', 'Precipitation']]
y_weather = dataset_2['Total']

X_train, X_test, y_train, y_test = train_test_split(X_weather, y_weather, test_size=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X_weather, y_weather, test_size=0.2, random_state=42)

mean_train = X_train.mean()
std_train = X_train.std()
X_train_scaled = (X_train - mean_train) / std_train
X_test_scaled = (X_test - mean_train) / std_train

weather_model = LinearRegression()
weather_model.fit(X_train_scaled, y_train)

train_pred = weather_model.predict(X_train_scaled)
test_pred = weather_model.predict(X_test_scaled)

print("Weather Prediction Model Results:")
print(f"Train R²: {r2_score(y_train, train_pred):.4f}")
print(f"Test R²: {r2_score(y_test, test_pred):.4f}")
print(f"Test MSE: {mean_squared_error(y_test, test_pred):.2f}")
print(f"\nIndividual correlations with traffic:")
print(f"High Temp: {dataset_2['High Temp'].corr(dataset_2['Total']):.4f}")
print(f"Low Temp: {dataset_2['Low Temp'].corr(dataset_2['Total']):.4f}")
print(f"Precipitation: {dataset_2['Precipitation'].corr(dataset_2['Total']):.4f}")
