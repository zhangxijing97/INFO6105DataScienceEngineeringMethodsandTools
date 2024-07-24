import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import SimpleRNN,Dropout,Dense
from sklearn.preprocessing import MinMaxScaler

#Read data
dataset_train = pd.read_csv('passengers_ds.csv')
train = dataset_train.loc[:, ['total_passengers']].values

#Normalize data
scaler = MinMaxScaler(feature_range = (0, 1))
train_scaled = scaler.fit_transform(train)

#Create Sequence with Length of 50
X_train = []
y_train = []
timesteps = 50
for i in range(timesteps, 144):
    X_train.append(train_scaled[i - timesteps:i, 0])
    y_train.append(train_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

#Reshape data
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

#Create RNN model
rnn = Sequential()
rnn.add(SimpleRNN(units = 50, return_sequences=True))
rnn.add(SimpleRNN(units = 50, return_sequences=True))
rnn.add(SimpleRNN(units = 50, return_sequences=True))
rnn.add(SimpleRNN(units = 50))
rnn.add(Dropout(0.2))
rnn.add(Dense(units = 1))

#Set optimizer and loss
rnn.compile(optimizer='adam', loss='mean_squared_error')

#Fitting the RNN to the Training set
rnn.fit(X_train, y_train, epochs=100, batch_size=32)

#Plot Real vs Predicted
predicted_y = rnn.predict(X_train)
predicted_y = scaler.inverse_transform(predicted_y)
y_train = scaler.inverse_transform(y_train.reshape(-1, 1))
plt.plot(y_train, color='red' )
plt.plot(predicted_y, color='blue' )
plt.xlabel('Time')
plt.ylabel('Passengers')
plt.legend(["Real", "Predicted"], loc="lower right")
plt.show()

