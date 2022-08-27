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



ax[0,0].plot(df[(df['smoker']==1)&(df['sex']==1)]['age' ],df[(df['smoker']==1)&(df['sex']==1)]['Pred_charge'],'o',color='#004F7F')
ax[0,0].plot(df[(df['smoker']==1)&(df['sex']==1)]['age'],df[(df['smoker']==1)&(df['sex']==1)]['charge'],'o',      color='#FBA25C')
z1=np.polyfit(df[(df['smoker']==1)&(df['sex']==1)]['age' ],df[(df['smoker']==1)&(df['sex']==1)]['Pred_charge'],1)
z2=np.polyfit(df[(df['smoker']==1)&(df['sex']==1)]['age' ],df[(df['smoker']==1)&(df['sex']==1)]['charge'],1)
p1=np.poly1d(z1)
p2=np.poly1d(z2)
print('Male smokers, charge vs age' ,sum([(i-j)**2 for i,j in zip(p1(list(range(20,60))),p2(list(range(20,60))))])/len(list(range(20,60))))
ax[0,0].plot(df[(df['smoker']==1)&(df['sex']==1)]['age' ],p1(df[(df['smoker']==1)&(df['sex']==1)]['age' ]),color='#004F7F')
ax[0,0].plot(df[(df['smoker']==1)&(df['sex']==1)]['age' ],p2(df[(df['smoker']==1)&(df['sex']==1)]['age' ]),color='#FBA25C')
ax[0,0].set_title('Male smokers, charge vs age')


ax[1,0].plot(df[(df['smoker']==1)&(df['sex']==0)]['age' ],df[(df['smoker']==1)&(df['sex']==0)]['Pred_charge'],'o',color='#004F7F')
ax[1,0].plot(df[(df['smoker']==1)&(df['sex']==0)]['age'],df[(df['smoker']==1)&(df['sex']==0)]['charge'],'o',      color='#FBA25C')
z3=np.polyfit(df[(df['smoker']==1)&(df['sex']==0)]['age' ],df[(df['smoker']==1)&(df['sex']==0)]['Pred_charge'],1)
z4=np.polyfit(df[(df['smoker']==1)&(df['sex']==0)]['age' ],df[(df['smoker']==1)&(df['sex']==0)]['charge'],1)
p3=np.poly1d(z3)
p4=np.poly1d(z4)
print('Female smokers, charge vs age' ,sum([(i-j)**2 for i,j in zip(p3(list(range(20,60))),p4(list(range(20,60))))])/len(list(range(20,60))))
ax[1,0].plot(df[(df['smoker']==1)&(df['sex']==0)]['age' ],p3(df[(df['smoker']==1)&(df['sex']==0)]['age' ]),color='#004F7F')
ax[1,0].plot(df[(df['smoker']==1)&(df['sex']==0)]['age' ],p4(df[(df['smoker']==1)&(df['sex']==0)]['age' ]),color='#FBA25C')
ax[1,0].set_title('Female smokers, charge vs age')

ax[2,0].plot(df[(df['smoker']==0)&(df['sex']==1)]['age' ],df[(df['smoker']==0)&(df['sex']==1)]['Pred_charge'],'o',color='#004F7F')
ax[2,0].plot(df[(df['smoker']==0)&(df['sex']==1)]['age'],df[(df['smoker']==0)&(df['sex']==1)]['charge'],'o',      color='#FBA25C')
z5=np.polyfit(df[(df['smoker']==0)&(df['sex']==1)]['age' ],df[(df['smoker']==0)&(df['sex']==1)]['Pred_charge'],1)
z6=np.polyfit(df[(df['smoker']==0)&(df['sex']==1)]['age' ],df[(df['smoker']==0)&(df['sex']==1)]['charge'],1)
p5=np.poly1d(z5)
p6=np.poly1d(z6)
print('Male non-smokers, charge vs age', sum([(i-j)**2 for i,j in zip(p5(list(range(20,60))),p6(list(range(20,60))))])/len(list(range(20,60))))
ax[2,0].plot(df[(df['smoker']==0)&(df['sex']==1)]['age' ],p5(df[(df['smoker']==0)&(df['sex']==1)]['age' ]),color='#004F7F')
ax[2,0].plot(df[(df['smoker']==0)&(df['sex']==1)]['age' ],p6(df[(df['smoker']==0)&(df['sex']==1)]['age' ]),color='#FBA25C')
ax[2,0].set_title('Male non-smokers, charge vs age')


ax[3,0].plot(df[(df['smoker']==0)&(df['sex']==0)]['age' ],df[(df['smoker']==0)&(df['sex']==0)]['Pred_charge'],'o',color='#004F7F')
ax[3,0].plot(df[(df['smoker']==0)&(df['sex']==0)]['age'],df[(df['smoker']==0)&(df['sex']==0)]['charge'],'o'      ,color='#FBA25C')
z7=np.polyfit(df[(df['smoker']==0)&(df['sex']==0)]['age' ],df[(df['smoker']==0)&(df['sex']==0)]['Pred_charge'],1)
z8=np.polyfit(df[(df['smoker']==0)&(df['sex']==0)]['age' ],df[(df['smoker']==0)&(df['sex']==0)]['charge'],1)
p7=np.poly1d(z7)
p8=np.poly1d(z8)
print('Female non-smokers, charge vs age', sum([(i-j)**2 for i,j in zip(p7(list(range(20,60))),p8(list(range(20,60))))])/len(list(range(20,60))))
ax[3,0].plot(df[(df['smoker']==0)&(df['sex']==0)]['age' ],p7(df[(df['smoker']==0)&(df['sex']==0)]['age' ]),color='#004F7F')
ax[3,0].plot(df[(df['smoker']==0)&(df['sex']==0)]['age' ],p8(df[(df['smoker']==0)&(df['sex']==0)]['age' ]),color='#FBA25C')
ax[3,0].set_title('Female non-smokers, charge vs age')

ax[0,1].plot(df[(df['smoker']==1)&(df['sex']==1)]['bmi' ],df[(df['smoker']==1)&(df['sex']==1)]['Pred_charge'],'o',color='#004F7F')
ax[0,1].plot(df[(df['smoker']==1)&(df['sex']==1)]['bmi'],df[(df['smoker']==1)&(df['sex']==1)]['charge'],'o'      ,color='#FBA25C')
z9=np.polyfit(df[(df['smoker']==1)&(df['sex']==1)]['bmi'],df[(df['smoker']==1)&(df['sex']==1)]['Pred_charge'],1)
z10=np.polyfit(df[(df['smoker']==1)&(df['sex']==1)]['bmi'],df[(df['smoker']==1)&(df['sex']==1)]['charge'],1)
p9=np.poly1d(z9)
p10=np.poly1d(z10)
print('Male smokers, charge vs bmi' ,sum([(i-j)**2 for i,j in zip(p9(list(range(18,50))),p10(list(range(18,50))))])/len(list(range(18,50))))
ax[0,1].plot(df[(df['smoker']==1)&(df['sex']==1)]['bmi' ],p9(df[(df['smoker']==1)&(df['sex']==1)] ['bmi' ]),color='#004F7F')
ax[0,1].plot(df[(df['smoker']==1)&(df['sex']==1)]['bmi' ],p10(df[(df['smoker']==1)&(df['sex']==1)]['bmi' ]),color='#FBA25C')
ax[0,1].set_title('Male smokers, charge vs bmi')

ax[1,1].plot(df[(df['smoker']==1)&(df['sex']==0)]['bmi' ],df[(df['smoker']==1)&(df['sex']==0)]['Pred_charge'],'o',color='#004F7F')
ax[1,1].plot(df[(df['smoker']==1)&(df['sex']==0)]['bmi'],df[(df['smoker']==1)&(df['sex']==0)]['charge'],'o'      ,color='#FBA25C')
z11=np.polyfit(df[(df['smoker']==1)&(df['sex']==0)]['bmi'],df[(df['smoker']==1)&(df['sex']==0)]['Pred_charge'],1)
z12=np.polyfit(df[(df['smoker']==1)&(df['sex']==0)]['bmi'],df[(df['smoker']==1)&(df['sex']==0)]['charge'],1)
p11=np.poly1d(z11)
p12=np.poly1d(z12)
print('Female smokers, charge vs bmi',sum([(i-j)**2 for i,j in zip(p11(list(range(18,50))),p12(list(range(18,50))))])/len(list(range(18,50))))
ax[1,1].plot(df[(df['smoker']==1)&(df['sex']==0)]['bmi' ],p11(df[(df['smoker']==1)&(df['sex']==0)] ['bmi' ]),color='#004F7F')
ax[1,1].plot(df[(df['smoker']==1)&(df['sex']==0)]['bmi' ],p12(df[(df['smoker']==1)&(df['sex']==0)]['bmi' ]),color='#FBA25C')
ax[1,1].set_title('Female smokers, charge vs bmi')


ax[2,1].plot(df[(df['smoker']==0)&(df['sex']==1)]['bmi' ],df[(df['smoker']==0)&(df['sex']==1)]['Pred_charge'],'o',color='#004F7F')
ax[2,1].plot(df[(df['smoker']==0)&(df['sex']==1)]['bmi'],df[(df['smoker']==0)&(df['sex']==1)]['charge'],'o'      ,color='#FBA25C')
z13=np.polyfit(df[(df['smoker']==0)&(df['sex']==1)]['bmi'],df[(df['smoker']==0)&(df['sex']==1)]['Pred_charge'],1)
z14=np.polyfit(df[(df['smoker']==0)&(df['sex']==1)]['bmi'],df[(df['smoker']==0)&(df['sex']==1)]['charge'],1)
p13=np.poly1d(z13)
p14=np.poly1d(z14)
print('Male non-smokers, charge vs bmi',sum([(i-j)**2 for i,j in zip(p13(list(range(18,50))),p14(list(range(18,50))))])/len(list(range(18,50))))
ax[2,1].plot(df[(df['smoker']==0)&(df['sex']==1)]['bmi' ],p13(df[(df['smoker']==0)&(df['sex']==1)] ['bmi' ]),color='#004F7F')
ax[2,1].plot(df[(df['smoker']==0)&(df['sex']==1)]['bmi' ],p14(df[(df['smoker']==0)&(df['sex']==1)]['bmi' ]),color='#FBA25C')
ax[2,1].set_title('Male non-smokers, charge vs bmi')


ax[3,1].plot(df[(df['smoker']==0)&(df['sex']==0)]['bmi' ],df[(df['smoker']==0)&(df['sex']==0)]['Pred_charge'],'o',color='#004F7F')
ax[3,1].plot(df[(df['smoker']==0)&(df['sex']==0)]['bmi'],df[(df['smoker']==0)&(df['sex']==0)]['charge'],'o'      ,color='#FBA25C')
z15=np.polyfit(df[(df['smoker']==0)&(df['sex']==0)]['bmi'],df[(df['smoker']==0)&(df['sex']==0)]['Pred_charge'],1)
z16=np.polyfit(df[(df['smoker']==0)&(df['sex']==0)]['bmi'],df[(df['smoker']==0)&(df['sex']==0)]['charge'],1)
p15=np.poly1d(z15)
p16=np.poly1d(z16)
print('Female non-smokers, charge vs bmi', sum([(i-j)**2 for i,j in zip(p15(list(range(18,50))),p16(list(range(18,50))))])/len(list(range(18,50))))
ax[3,1].plot(df[(df['smoker']==0)&(df['sex']==0)]['bmi' ],p15(df[(df['smoker']==0)&(df['sex']==0)] ['bmi' ]),color='#004F7F')
ax[3,1].plot(df[(df['smoker']==0)&(df['sex']==0)]['bmi' ],p16(df[(df['smoker']==0)&(df['sex']==0)]['bmi' ]) ,color='#FBA25C')
ax[3,1].set_title('Female non-smokers, charge vs bmi')


plt.show()
