import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# ds1 = pd.read_csv("ds1.csv") #Total, Date: m/d/y
ds2 = pd.read_csv("diff.csv") #ORDERDATE: m/d/y, SALES
# ds3 = pd.read_csv("ds3.csv") #Month: y/m, Sales, DateVal

# print(ds2["SALES"].describe())
# dateparse = lambda dates: pd.datetime.strptime(dates, '%m/%d/%Y')
# dsp = pd.read_csv("New.csv",parse_dates=['Date'],squeeze=True,date_parser=dateparse)
# #Total, Date: m/d/y
#
# x1 = np.array(ds1["Values"]).reshape(-1,1)
# y1 = np.array(ds1["Total"]).reshape(-1,1)
#-------------------- SECOND DATASET --------------------
x2 = np.array(ds2["VALUES"]).reshape(-1,1)
y2 = np.array(ds2["SALES"]).reshape(-1,1)
#-------------------- THIRD DATASET ---------------------
# x3 = np.array(ds3["Values"]).reshape(-1,1)
# y3 = np.array(ds3["Sales"]).reshape(-1,1)

# print(x.shape)
# print(y.shape)

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
# scaledX = sc_x.fit_transform(x1)
# # scaledX = sc_x.fit_transform(x_p)
# scaledY = sc_y.fit_transform(y1)

#------------DS2------------------
scaledX2 = sc_x.fit_transform(x2)
# scaledX = sc_x.fit_transform(x_p)
scaledY2 = sc_y.fit_transform(y2)

#------------DS3------------------
# scaledX3 = sc_x.fit_transform(x3)
# # scaledX = sc_x.fit_transform(x_p)
# scaledY3 = sc_y.fit_transform(y3)


# print(scaledX.shape)
# print(scaledY.shape)

# x_train, x_test, y_train, y_test = train_test_split(x1, y1, test_size=0.3, random_state=42)

# x_train, x_test, y_train, y_test = train_test_split(scaledX, scaledY, test_size=0.3, random_state=42)

#---------DS2-----------
x_train2, x_test2, y_train2, y_test2 = train_test_split(scaledX2, scaledY2, test_size=0.3, random_state=42)

#---------DS3-----------
# x_train3, x_test3, y_train3, y_test3 = train_test_split(scaledX3, scaledY3, test_size=0.3, random_state=42)


from sklearn.svm import SVR

# reg = SVR(kernel="rbf",gamma='scale', C=100 , epsilon =0.05)
# reg = SVR(kernel="poly", C=10, degree=3, epsilon=0.2, coef0=5, gamma='scale')
# # reg = SVR(kernel="sigmoid")
# # reg = SVR(kernel='linear', C=100, gamma='scale')
# reg.fit(x_train, y_train)

#---------2----------
reg2 = SVR(kernel="rbf",gamma='scale', C=500 , epsilon =0.05)
reg2.fit(x_train2, y_train2)
# #---------3----------
# reg3 = SVR(kernel="rbf",gamma='scale', C=100 , epsilon =0.1)
# reg3.fit(x_train3, y_train3)

#-------------PREDICTIONS-----------------
# y_pred = reg.predict(y_test).reshape(-1,1)
y_pred2 = reg2.predict(y_test2).reshape(-1,1)
# y_pred3 = reg3.predict(y_test3).reshape(-1,1)

from sklearn.metrics import mean_squared_error
# print("First dataset : " , mean_squared_error(y_test, y_pred))
print("Second dataset: " , mean_squared_error(y_test2, y_pred2))
# print("Third dataset : " , mean_squared_error(y_test3, y_pred3))

#--------------VISUALIZE------------------

# fig = plt.figure()
# ax1 = fig.add_subplot(3,1,1)
# ax2 = fig.add_subplot(3,1,2)
# ax3 = fig.add_subplot(3,1,3)


# ax1.plot(scaledY, color='k')
# ax1.plot(reg.predict(scaledY), color='r')

plt.plot(scaledY2, color='k')
plt.plot(reg2.predict(scaledY2), color='r')

# ax3.plot(scaledY3, color='k')
# ax3.plot(reg3.predict(scaledY3), color='r')

plt.show()









# plt.plot(scaledY)
# plt.plot(reg.predict(scaledY), color='r')
# plt.show()

# plt.scatter(x1, y1, color = 'magenta')
# plt.plot(x1, reg.predict(x1), color = 'green')
# plt.title('SVR')
# plt.xlabel('Time')
# plt.ylabel('Sales')
# plt.show()
