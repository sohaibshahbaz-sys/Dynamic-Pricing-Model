import pandas as pd
import matplotlib.pyplot as plt
import seaborn as seabornInstance 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
import category_encoders as ce
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import math
from sklearn import preprocessing

dataset = pd.read_csv('AmazonBookReviews.csv')
encoder = ce.BackwardDifferenceEncoder(cols=['Genre'])
dataset = encoder.fit_transform(dataset)

dataset.plot(x='User Rating', y='Price', style='o')
plt.title('User Rating vs Price')
plt.xlabel('User Rating')
plt.ylabel('Price')
plt.show()

dataset.plot(x='Genre_0', y='Price', style='o')
plt.title('Genre vs Price')
plt.xlabel('Genre')
plt.ylabel('Price')
plt.show()

X = dataset[['User Rating', 'Genre_0']]
y = dataset['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)



regressor = LinearRegression()
regressor.fit(X,  y)
y_pred = regressor.predict(X_test)

R2 = r2_score(y_test, y_pred)
MAE = mean_absolute_error(y_test, y_pred)
RMSE = math.sqrt(mean_squared_error(y_test, y_pred))

df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(df)
print('regressor.coef: ', regressor.coef_, 'R2: ', R2, 'MAE: ', MAE, 'RMSE: ', RMSE)



from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np

regr = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2))
regr.fit(X, y)

y_pred = regr.predict(X_test)

R2 = r2_score(y_test, y_pred)
MAE = mean_absolute_error(y_test, y_pred)
RMSE = math.sqrt(mean_squared_error(y_test, y_pred))

df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(df)
#print('regressor.coef: ', regr.coef_, 'R2: ', R2, 'MAE: ', MAE, 'RMSE: ', RMSE)
print('R2: ', abs(R2), 'MAE: ', MAE, 'RMSE: ', RMSE)