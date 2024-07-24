import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

#Input Dateset
org_df = pd.read_csv("high_income.csv")

#Labels and Features
label_df = org_df.loc[:,org_df.columns == 'label']
feat_df = org_df.loc[:,org_df.columns != 'label']

#Split Train and Test Data
x_train, x_test, y_train, y_test = train_test_split(feat_df, label_df, test_size = 0.2)

#Create SVM Model
svm = SVC(kernel='linear')
svm.fit(x_train, y_train)

#Accuracy of Model
accuracy = svm.score(x_test,y_test)
print("Test accuracy:  ", accuracy)

