import pandas as pd
import numpy as np
data = pd.read_csv('D:\My Stuff\Arjun\FDP\Churn_Modelling.csv')
frame=pd.DataFrame(data)
frame
import keras
x=data.iloc[:,3:13].values
y=data.iloc[:,13:].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
lEncoder_x1= LabelEncoder()
lEncoder_x2= LabelEncoder()
x[:,1]= lEncoder_x1.fit_transform(x[:,1])
x[:,2]= lEncoder_x2.fit_transform(x[:,2])
ohEncoder= OneHotEncoder(categorical_features=[1])
x=ohEncoder.fit_transform(x).toarray()
x=x[:,1:]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20, random_state=0)

from sklearn.preprocessing import StandardScaler
sScaler = StandardScaler()
x_train=sScaler.fit_transform(x_train)
x_test=sScaler.transform(x_test)

from keras.models import Sequential
from keras.layers import Dense
classifier=Sequential()
classifier.add(Dense(input_dim=11, units=6, kernel_initializer='uniform', activation='relu'))
classifier.add(Dense(units=6,kernel_initializer='uniform', activation='sigmoid'))
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
classifier.fit(x_train,y_train,batch_size=10,epochs=100)


y_pred = classifier.predict(x_test)
y_pred = (y_pred > 0.5)
y_pred


newy=np.array([(0,1,600,0,25,4,50000,1,1,0,1)])
newy_predicted= classifier.predict(newy)
newy_predicted