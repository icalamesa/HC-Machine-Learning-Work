import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import logging

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',level=logging.INFO,datefmt='%Y-%m-%d %H:%M:%S')

df=pd.read_csv('data.csv')

# Quantifying the charges with quantiles
quant1, quant2 = [0] + list(df['charges'].quantile(list(np.arange(0.1, 1, 0.1)))), list(df['charges'].quantile(list(np.arange(0.1, 1, 0.1)))) + [df['charges'].max()]
i = 0
for q1, q2 in zip(quant1, quant2):
    df.loc[(df['charges']>q1) & (df['charges']<=q2), 'charges']=i
    i += 1
    print(i, q1,q2)


# Making all categorical values into integers
df.loc[df['sex']=='female','sex'],df.loc[df['sex']=='male','sex']=1,0
df.loc[df['region']=='southwest','region'],df.loc[df['region']=='southeast','region'],df.loc[df['region']=='northwest','region'],df.loc[df['region']=='northeast','region']=1,2,3,4
df.loc[df['smoker']=='yes','smoker'],df.loc[df['smoker']=='no','smoker']=1,0


label=df['charges'] # labels
df=df.drop(['charges'], axis=1) # Droping labels from traindata
print(df.columns)

feature=[column for column in df]
X = np.column_stack([list(df[col]) for col in feature ]) # Creating the feautre matrix for train_data
Y=label
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.8,random_state=42)


model = RandomForestClassifier(n_estimators=1000, random_state=42) # creating the model
logging.info('Start')
model.fit(X_train,Y_train) # Fitting the model
logging.info('Finish')
feature_importance=model.feature_importances_


ypred = model.predict(X_test)
print(metrics.classification_report(ypred, Y_test))

# Plotting the heat map
mat = confusion_matrix(Y_test, ypred)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)
plt.xlabel('true label')
plt.ylabel('predicted label')
plt.show()

Square_error=list((ypred-Y_test)**2)
mean_square=sum(Square_error)/len(Square_error)
print(mean_square, 1)
