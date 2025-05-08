import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn import linear_model
import numpy as np
import pandas as pd

data_home = 'https://raw.githubusercontent.com/dknife/ML/main/data/'
lin_data = pd.read_csv(data_home + 'pollution.csv')
print(lin_data)

# lin_data.plot.scatter(x='input', y='pollution', color='blue')
# # plt.show()

# w, b = 1, 1
# x0, x1 = 0.0, 1.0
# def h(x,w,b):
#     return w*x + b
# w, b = -3, 6
# x0, x1 = 0.0, 1.0
# lin_data.plot.scatter(x='input', y='pollution', color='blue')
# plt.plot([x0, x1], [h(x0,w,b), h(x1,w,b)])  
# # plt.show()

# # def h(x, param):
# #     return param[0] * x + param[1]

# # learning_iterations = 1000
# # learning_rate = 0.0025

# # param = [1, 1]

x = lin_data['input'].to_numpy().reshape(-1, 1)
y = lin_data['pollution'].to_numpy()
x = x[: np.newaxis]

# for i in range(learning_iterations):
#     if i % 200 == 0:
#         lin_data.plot.scatter(x='input', y='pollution', color='blue')
#         plt.plot([0,1], [h(0,param), h(1,param)])
#     error = (h(x, param) - y)
#     param[0] -= learning_rate * (error * x).sum()
#     param[1] -= learning_rate * error.sum()
# plt.show()

regr = linear_model.LinearRegression()
regr.fit(x, y)
lin_data.plot.scatter(x='input', y='pollution', color='blue')
y_pred = regr.predict([[0], [1]])
plt.plot([0,1], y_pred)
plt.show()