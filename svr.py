import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

ds = pd.read_csv("d.csv") 
print(ds.describe())

# turn your dataset into a time series dataset, if necessary
dateparse = lambda dates: pd.datetime.strptime(dates, '%m/%d/%Y')
dsp = pd.read_csv("s.csv",parse_dates=['Date'],squeeze=True,date_parser=dateparse)

x = np.array(ds["V"]).reshape(-1,1)
y = np.array(ds["S"]).reshape(-1,1)

print(x.shape)
print(y.shape)

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
scaledX = sc_x.fit_transform(x)
scaledX = sc_x.fit_transform(y)
print(scaledX.shape)
print(scaledY.shape)

x_train, x_test, y_train, y_test = train_test_split(scaledX, scaledY, test_size=0.3, random_state=42)

from sklearn.svm import SVR

reg = SVR(kernel="rbf",gamma='scale', C=10 , epsilon =0.1)
reg.fit(x_train, y_train)


#-------------PREDICTIONS-----------------

y_pred = reg.predict(y_test).reshape(-1,1)

from sklearn.metrics import mean_squared_error
print(mean_squared_error(y_test, y_pred))


#--------------VISUALIZE------------------


plt.plot(scaledY, color='k')
plt.plot(reg.predict(scaledY), color='r')
plt.show()


