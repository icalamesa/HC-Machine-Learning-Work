import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
# Import train_test_split function
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error  

path = './insurance.csv' 
df = pd.read_csv(path)  

### START ENCODING BLOCK ###

df = pd.get_dummies(df, columns=['sex','smoker','region'])

### END ENCODING BLOCK###

print(df.head())

reg = LinearRegression(fit_intercept=True)
X = df.to_numpy()[:,:5]
y = df.to_numpy()[:,6]
print(X)

# Split data into the training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=5)
print("Size of training set: ", X_train.shape[0])
print("Size of validation set: ", X_val.shape[0])

reg.fit(X_train, y_train)

y_pred_train = reg.predict(X_train)
y_pred_val   = reg.predict(X_val)

#print(y_pred_train[:10])
#print(y_pred_val[:10])

err_train = mean_squared_error(y_train, y_pred_train)  
err_val = mean_squared_error(y_val, y_pred_val)

print(f"Training error is : {err_train:.2f}")
print(f"Validation error is : {err_val:.2f}")

fig, axes = plt.subplots(2, 3)

i = 0
print("\nManaging first row of subplots...")
for key in {'age', 'bmi', 'children'}:
    print(f"i index is {i} and the column key is {key}.")
    data = df[key].to_numpy().reshape(-1,1)
    target = df['charges'].to_numpy()

    print("Printing subplots...")
    axes[0, i].set_title(f'charges vs {key}')  # Set plot title
    axes[0, i].set_xlabel(key)    # Set x-axis label
    axes[0, i].set_ylabel("charges")  # Set y-axis label
    #axes[0, i].legend(loc='upper left')  # Set location of the legend to show in upper left corner
    axes[0, i].scatter(data, target)
    #axes[i].plot(X_train.append())

    i+=1

print("\nManaging second row of subplots(predicted values)...")
print(y_pred_train)
axes[1,0].scatter(X[:,0], np.append(y_pred_train, y_pred_val), color="black")


plt.show()

