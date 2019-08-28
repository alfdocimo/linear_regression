import pandas as pd
import matplotlib.pyplot as plt
import seaborn as seabornInstance
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

dataset = pd.read_csv('weather_data.csv')

"""
dataset will now hold the values from out CSV file as a DataFrame from pandas.
We can then use .plot and create a graph with that information.

In this case, we're trying to predict the Max temperature given a Mean of the temperature.
We can then say that Max Temp is our 'Y' value since its dependant on 'X' a.k.a our MeanTemp

"""

dataset.plot(x='MeanTemp', y='MaxTemp', style='o')
plt.title('MeanTemp vs MaxTemp')  # Here, we specify the title of our graph
plt.xlabel('MeanTemp')  # This will be our independent variable
plt.ylabel('MaxTemp')  # This will be our dependent variable
plt.show()  # show our graph

"""
Okay, one thing to keep in mind is that our dataset object is an array of values, or how cool people in the ML world call it, 'Vector'.

So in order to feed this data to our linear regresion model, we first need to shape it from a Vector into a Matrix

https://image.slidesharecdn.com/advancedsparkandtensorflowmeetupmay262016-160528013807/95/advanced-spark-and-tensorflow-meetup-may-26-2016-30-638.jpg?cb=1464402298

We'll do this for both X and y

"""


X = dataset['MinTemp'].values.reshape(-1, 1)
y = dataset['MaxTemp'].values.reshape(-1, 1)


"""

So far so good.

This next step basically does the following

from the input X and y, in oder words all the data inside those matrices, I want you 'train_test_split' to split them into two sets of data.

One of which the model will use to train and other which we'll use to predict.

In other words, one is data that the model will know, and other will be data that it has never seen before.

"""

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)


regressor = LinearRegression()  # Create an instance of the model
regressor.fit(X_train, y_train)  # training the Model with our training data

"""

This next step is where the sauce is at:

we're creating y_pred which is basically is another matrix of points, but this output points will come
from the input matrix X_test.

"""


y_pred = regressor.predict(X_test)

"""

Next, we just graph all the test data we used to test the model, and then
plot a line with the prediction for that data

"""

plt.scatter(X_test, y_test,  color='gray')
plt.plot(X_test, y_pred, color='red', linewidth=2)
plt.show()
