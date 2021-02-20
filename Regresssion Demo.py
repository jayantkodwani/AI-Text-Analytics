# 'Load in the dependencies
import pandas as pd
#import numpy as np
#import matplotlib.pyplot as plt  # To visualize

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv('advertising_sales.csv')
X=dataset.iloc[:,1:4]
y = dataset['Sales']




#split and train the dataset
X_train,X_test,y_train,y_test = train_test_split(X,y)

#Let the model predict results
lin = LinearRegression()
lin.fit(X_train,y_train)

y_pred = lin.predict(X_test)


coef=lin.coef_ 
components = pd.DataFrame(zip(X.columns,coef),columns=['component','value'])
components = components.append({'component':'intercept','value':lin.intercept_}, ignore_index=True)

#plt.scatter(X, y)
#plt.plot(X,y_pred, color='red')
#plt.show()


