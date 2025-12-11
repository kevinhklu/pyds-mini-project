from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd


''' 
The following is the starting code for path2 for data reading to make your first step easier.
'dataset_2' is the clean data for path1.
'''
dataset_2 = pd.read_csv('NYC_Bicycle_Counts_2016.csv')
dataset_2['Brooklyn Bridge']      = pd.to_numeric(dataset_2['Brooklyn Bridge'].replace(',','', regex=True))
dataset_2['Manhattan Bridge']     = pd.to_numeric(dataset_2['Manhattan Bridge'].replace(',','', regex=True))
dataset_2['Queensboro Bridge']    = pd.to_numeric(dataset_2['Queensboro Bridge'].replace(',','', regex=True))
dataset_2['Williamsburg Bridge']  = pd.to_numeric(dataset_2['Williamsburg Bridge'].replace(',','', regex=True))
dataset_2['Total']  = pd.to_numeric(dataset_2['Total'].replace(',','', regex=True))

# print(dataset_2.to_string()) #This line will print out your data

# Problem 1: Sensor Installation

def sensor(bridge_1, bridge_2, bridge_3, not_included_bridge):

    included_bridges = f'{bridge_1} and {bridge_2} and {bridge_3}'
    dataset_2[included_bridges] = dataset_2[bridge_1] + dataset_2[bridge_2] + dataset_2[bridge_3]
    
    X = dataset_2[[included_bridges]]
    Y = dataset_2["Total"] 

    model = LinearRegression()
    model.fit(X, Y)
    pred = model.predict(X)
    r_squared = model.score(X, Y)
    mse = mean_squared_error(Y, pred) 

    print('\nNot Included Bridge: ', not_included_bridge)
    print('MSE: ', round(mse))
    print('R^2:', r_squared)

sensor("Manhattan Bridge", "Queensboro Bridge", "Williamsburg Bridge", "Brooklyn Bridge")  
sensor("Brooklyn Bridge", "Queensboro Bridge", "Williamsburg Bridge", "Manhattan Bridge")  
sensor("Brooklyn Bridge", "Manhattan Bridge", "Williamsburg Bridge", "Queensboro Bridge") 
sensor("Brooklyn Bridge", "Manhattan Bridge", "Queensboro Bridge", "Williamsburg Bridge")    
