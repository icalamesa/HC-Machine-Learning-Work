import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

df=pd.read_csv('data.csv')
df['charges']=df['charges']/1000 # Maps the charges in to kilos instead

# Making all categorical values into integers
df.loc[df['sex']=='female','sex'],df.loc[df['sex']=='male','sex']=1,0
df.loc[df['region']=='southwest','region'],df.loc[df['region']=='southeast','region'],df.loc[df['region']=='northwest','region'],df.loc[df['region']=='northeast','region']=1,2,3,4
df.loc[df['smoker']=='yes','smoker'],df.loc[df['smoker']=='no','smoker']=1,0

label=df['charges'] # labels
df=df.drop(['charges'], axis=1) # Droping labels from traindata

feature=[column for column in df]
X = np.column_stack([list(df[col]) for col in feature ]) # Creating the feautre matrix for train_data
Y = label
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.8,random_state=42)

rf = RandomForestRegressor(n_estimators = 250, max_features = 'sqrt', max_depth = 7, random_state = 18)
rf = rf.fit(X_train, Y_train)

prediction = list(rf.predict(X_test))
Y_test=list(Y_test)

mse= metrics.mean_squared_error(Y_test,prediction)
print(mse)
rmse = mse**.5
print(rmse)

data={'age':list(X_test[:,0]),'sex':list(X_test[:,1]),'bmi':list(X_test[:,2]),'children':list(X_test[:,3]),'smoker':list(X_test[:,4]),'region':list(X_test[:,5]),'charge':list(Y_test),'Pred_charge':list(prediction)}
df=pd.DataFrame(data)

fig,ax=plt.subplots(4,2,sharex=True)
ax[0,0].plot(df[(df['smoker']==1)&(df['sex']==1)]['age' ],df[(df['smoker']==1)&(df['sex']==1)]['Pred_charge'],'o')
ax[0,0].plot(df[(df['smoker']==1)&(df['sex']==1)]['age'],df[(df['smoker']==1)&(df['sex']==1)]['charge'],'o')
ax[0,0].set_title('Male smokers, charge vs age')
ax[1,0].plot(df[(df['smoker']==1)&(df['sex']==0)]['age' ],df[(df['smoker']==1)&(df['sex']==0)]['Pred_charge'],'o')
ax[1,0].plot(df[(df['smoker']==1)&(df['sex']==0)]['age'],df[(df['smoker']==1)&(df['sex']==0)]['charge'],'o')
ax[1,0].set_title('Female smokers, charge vs age')
ax[2,0].plot(df[(df['smoker']==0)&(df['sex']==1)]['age' ],df[(df['smoker']==0)&(df['sex']==1)]['Pred_charge'],'o')
ax[2,0].plot(df[(df['smoker']==0)&(df['sex']==1)]['age'],df[(df['smoker']==0)&(df['sex']==1)]['charge'],'o')
ax[2,0].set_title('Male non-smokers, charge vs age')
ax[3,0].plot(df[(df['smoker']==0)&(df['sex']==0)]['age' ],df[(df['smoker']==0)&(df['sex']==0)]['Pred_charge'],'o')
ax[3,0].plot(df[(df['smoker']==0)&(df['sex']==0)]['age'],df[(df['smoker']==0)&(df['sex']==0)]['charge'],'o')
ax[3,0].set_title('Female non-smokers, charge vs age')
ax[0,1].plot(df[(df['smoker']==1)&(df['sex']==1)]['bmi' ],df[(df['smoker']==1)&(df['sex']==1)]['Pred_charge'],'o')
ax[0,1].plot(df[(df['smoker']==1)&(df['sex']==1)]['bmi'],df[(df['smoker']==1)&(df['sex']==1)]['charge'],'o')
ax[0,1].set_title('Male smokers, charge vs bmi')
ax[1,1].plot(df[(df['smoker']==1)&(df['sex']==0)]['bmi' ],df[(df['smoker']==1)&(df['sex']==0)]['Pred_charge'],'o')
ax[1,1].plot(df[(df['smoker']==1)&(df['sex']==0)]['bmi'],df[(df['smoker']==1)&(df['sex']==0)]['charge'],'o')
ax[1,1].set_title('Female smokers, charge vs bmi')
ax[2,1].plot(df[(df['smoker']==0)&(df['sex']==1)]['bmi' ],df[(df['smoker']==0)&(df['sex']==1)]['Pred_charge'],'o')
ax[2,1].plot(df[(df['smoker']==0)&(df['sex']==1)]['bmi'],df[(df['smoker']==0)&(df['sex']==1)]['charge'],'o')
ax[2,1].set_title('Male non-smokers, charge vs bmi')
ax[3,1].plot(df[(df['smoker']==0)&(df['sex']==0)]['bmi' ],df[(df['smoker']==0)&(df['sex']==0)]['Pred_charge'],'o')
ax[3,1].plot(df[(df['smoker']==0)&(df['sex']==0)]['bmi'],df[(df['smoker']==0)&(df['sex']==0)]['charge'],'o')
ax[3,1].set_title('Female non-smokers, charge vs bmi')


plt.show()
