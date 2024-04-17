import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Input

#Input Dateset
org_df = pd.read_csv("diabetes.csv")

#Labels and Features
label_df = org_df.loc[:,org_df.columns == 'Outcome']
feat_df = org_df.loc[:,org_df.columns != 'Outcome']

#Normalize Features
feat_df = (feat_df - feat_df.mean()) / feat_df.std()

#Split Train and Test Data
x_train, x_test, y_train, y_test = train_test_split(feat_df, label_df, test_size = 0.3)

#Create NN Model
nn = Sequential()
nn.add(Input(shape=(8,)))
nn.add(Dense(units=5,  activation='relu'))
nn.add(Dense(units=1,  activation='sigmoid'))

# Set Optimizer, Loss Function, and metric
nn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#Train NN Model
nn.fit(x_train, y_train, epochs=100)
print(nn.summary())

#Accuracy of Model on Test data
loss,accuracy = nn.evaluate(x_test,y_test)
print('accuracy=',accuracy,' , loss=',loss)

