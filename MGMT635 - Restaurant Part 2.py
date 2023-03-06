import pandas as pd
import numpy as np


sheet_url = "https://docs.google.com/spreadsheets/d/1lIdYmIr-GRxdYdohfO01i1PUrNoAk_AB/edit#gid=1233047613"
url = sheet_url.replace("/edit#gid=", "/export?format=csv&gid=")
df = pd.read_csv(url)
df.drop(columns=df.columns[-9:],axis=1, inplace=True)
df_before = df


sheet_url = "https://docs.google.com/spreadsheets/d/1lIdYmIr-GRxdYdohfO01i1PUrNoAk_AB/edit#gid=100581497"
url = sheet_url.replace("/edit#gid=", "/export?format=csv&gid=")
df_2 = pd.read_csv(url)
df = df.append(df_2, ignore_index = True)
df_before2 = df

columns = ["Area Income", "Monthly Revenue", "Town Population"]
for i in columns:
    df[i] = df[i].str.replace("$","", regex=True).str.replace(",","", regex=True)
    df[i] = df[i].apply(pd.to_numeric)   
one_hot = pd.get_dummies(df["Restaurants"])
df = df.join(one_hot)
df.drop(columns="Restaurants", inplace=True)

#Option to take out Restaurant Encoding

df.drop(columns=one_hot.columns, inplace=True)


# Train & Test Split
y = df["Monthly Revenue"]
x = df.drop(columns="Monthly Revenue")

# Standardize Values (0-1)
for i in x.columns:
    x[i] = x[i]/x[i].max()

y_train = y[0:-11]
x_train = x[0:-11]
x_test = x[-11:]



# Multiple Linear Regression
from sklearn.linear_model import LinearRegression
regr = LinearRegression()
regr.fit(x_train,y_train)
MLR_test_pred = regr.predict(x_test)
MLR_train_pred = regr.predict(x_train)

coefficients = regr.coef_
df_coef = pd.DataFrame(x_train.columns)
df_coef["Coefficient"] = coefficients.tolist()

MLR_ME_train = round(abs(np.subtract(y_train,MLR_train_pred)).mean(),2)
MLR_MSE_train = round(np.square(np.subtract(y_train,MLR_train_pred)).mean(),2)

# Neural Network SKLearn
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor

#Suppress warnings when applying GridSearchCV to find best Neural Network constants
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)


param_list = {"hidden_layer_sizes": [(1,),(50,)], "activation": ["identity", "logistic", "tanh", "relu"], "solver": ["lbfgs", "sgd", "adam"], "alpha": [0.00005,0.0005]}

clf = GridSearchCV(MLPRegressor(max_iter=1000),param_list, n_jobs=-1, cv=3)
clf = clf.fit(x_train, y_train)
NN_test_pred = clf.predict(x_test)
NN_train_pred = clf.predict(x_train)

NN_ME_train = round(abs(np.subtract(y_train,NN_train_pred)).mean(),2)
NN_MSE_train = round(np.square(np.subtract(y_train,NN_train_pred)).mean(),2)

df_results = pd.DataFrame(df_before2["Restaurants"])
df_results = df_results[-11:]
df_results["SKLearn Neural Network"] = NN_test_pred.tolist()
df_results["SKLearn Multiple Linear Regression"] = MLR_test_pred.tolist()

from sklearn.metrics import r2_score
score_NN = r2_score(y_train, NN_train_pred.tolist())
print("This is the R-squared score for the Neural Network "+str(score_NN))
score_MLR = r2_score(y_train, MLR_train_pred)

# Neural Network Tensorflow.keras
'''
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

model = Sequential()
model.add(Dense(64, input_dim=7, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='linear'))

model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))

model.fit(x_train, y_train, epochs=100, batch_size=32, validation_split=0.2)

loss = model.evaluate(x_train, y_train, verbose=0)
print(f'Mean Squared Error: {loss:.4f}')

predictions = model.predict(x_test)
print(f'Predictions: {predictions.flatten()}')
'''