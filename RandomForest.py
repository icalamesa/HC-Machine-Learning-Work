import pandas as pd
import numpy as np
import sklearn
from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier


df=pd.read_csv('data.csv')

# Quantifying the charges with quantiles
quant1, quant2 = [0] + list(df['charges'].quantile(list(np.arange(0.1, 1, 0.1)))), list(df['charges'].quantile(list(np.arange(0.1, 1, 0.1)))) + [df['charges'].max()]
i = 0
for q1, q2 in zip(quant1, quant2):
    df.loc[(df['charges']>q1) & (df['charges']<=q2), 'charges']=i
    i += 1

# Making all categorical values into integers
df.loc[df['sex']=='female','sex'],df.loc[df['sex']=='male','sex']=1,0
df.loc[df['region']=='southwest','region'],df.loc[df['region']=='southeast','region'],df.loc[df['region']=='northwest','region'],df.loc[df['region']=='northeast','region']=1,2,3,4
df.loc[df['smoker']=='yes','smoker'],df.loc[df['smoker']=='no','smoker']=1,0

# Spliting up a small sample of test data
df_test = df.tail(5)
df.drop(df.tail(5).index, inplace=True)
label=df['charges'] # labels


df=df.drop(['charges'], axis=1) # Droping labels from traindata

feature=[column for column in df]
X = np.column_stack([list(df[col]) for col in feature ]) # Creating the feautre matrix for train_data
Y=label

X_test=np.column_stack([list(df_test[col]) for col in feature ]) # Creating the feautre matrix for testing data

model = RandomForestClassifier(n_estimators=100, random_state=1) # creating the model
model.fit(X,Y) # Fitting the model
print(X_test)
print(model.predict(X_test[0].reshape(1,-1)))


# Tomas Nordström
