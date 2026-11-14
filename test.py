import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error
df = pd.read_csv('advertising.csv')

#print(df.head())

#print(df.shape)
#print(df.info())

x=df[['TV', 'Radio', 'Newspaper']]
y=df['Sales']
## split 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
lr = LinearRegression()
lr.fit(x_train, y_train)
y_pred=lr.predict(x_test)


#print(y_test)
print(y_pred)

