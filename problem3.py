from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd

# Problem 3: Trends by Day

days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

def trends(metric):
    avg_metrics = (dataset_2.groupby('Day')[metric].mean()).tolist()
    
    plt.bar(days, avg_metrics)
    plt.xlabel("Day")
    plt.ylabel("Avg. Amount Bikers")        
    plt.title(f"Traffic Trends ({metric})")
    plt.show()

useful_metrics = ['Total', 'Brooklyn Bridge', 'Manhattan Bridge', 'Williamsburg Bridge', 'Queensboro Bridge']
for metric in useful_metrics:
    trends(metric)

def day_prediction(metric, n):
    X = dataset_2[[metric]].values
    Y = dataset_2['Day'].values

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, stratify=Y)
    
    model = KNeighborsClassifier(n_neighbors = n)
    model.fit(X_train, Y_train) 
    Y_pred = model.predict(X_test)

    print(f"\nAccuracy for {metric}", accuracy_score(Y_test, Y_pred))

for metric in useful_metrics:
    day_prediction(metric, 7)

    