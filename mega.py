import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

dataset=pd.read_csv('Credit_Card_Applications.csv')
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values

from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler(feature_range=(0,1))
x=sc.fit_transform(x)

from minisom import MiniSom
som=MiniSom(x=10, y=10, input_len=15)
som.random_weights_init(x)
som.train_random(x, num_iteration=100)

from pylab import bone,pcolor,colorbar,plot,show
bone()
pcolor(som.distance_map().T)
colorbar()
markers=['o','s']
colors=['r','g']
for i ,X in enumerate(x):
    w=som.winner(X)
    plot(w[0]+0.5,
         w[1]+0.5,
         markers[y[i]],
         markeredgecolor=colors[y[i]],
         markerfacecolor='None',
         markersize=10,
         markeredgewidth=1)
show()
    
mappings=som.win_map(x)
frauds=np.concatenate((mappings[(8,3)],mappings[(7,3)]),axis=0)
frauds=sc.inverse_transform(frauds)

customers=dataset.iloc[:,1:].values

is_frauds=np.zeros(len(dataset))
for i in range(len(dataset)):
    if dataset.iloc[i,0] in frauds:
        is_frauds[i]=1

from sklearn.preprocessing import StandardScaler
sc1=StandardScaler()
customers=sc.fit_transform(customers)

from keras.models import Sequential
from keras.layers import Dense

ann=Sequential()

ann.add(Dense(units=2,activation='relu',input_dim=15))
        
ann.add(Dense(units=1,activation='sigmoid'))

ann.compile(optimizer='adam',loss='binary_crossentropy',metrics=['activation'])

ann.fit(customers,is_frauds,batch_size=1,epochs=2)

y_pred=ann.predict(customers)

y_pred=np.concatenate((dataset.iloc[:,0:1].values,y_pred),axis=1)
y_pred=y_pred[y_pred[:,1].argsort()]



        





