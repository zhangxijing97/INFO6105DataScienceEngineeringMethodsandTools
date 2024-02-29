import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.tree import export_graphviz
import graphviz

#Input Dateset
org_df = pd.read_csv("diabetes.csv")

#Define features and label for KNN
label_df =  org_df.loc[:,org_df.columns == 'Outcome']
feat_df =  org_df.loc[:,org_df.columns != 'Outcome']

#Seperate test and train data
train_x,test_x,train_y,test_y = train_test_split(feat_df,label_df,test_size=0.25)

#Random Forest
rf = RandomForestClassifier(n_estimators=100)
rf.fit(train_x, train_y)

#Prediction on test data
test_pred_y = rf.predict(test_x)
accuracy = accuracy_score(test_y, test_pred_y)
print("Accuracy:", accuracy)

#Visualize confusion matrix
cf = confusion_matrix(test_y, test_pred_y)
sns.heatmap(cf,annot=True,yticklabels=['Negative','Positive']
               ,xticklabels=['Negative','Positive'],cmap='Blues', fmt='g')
plt.tight_layout()
plt.show()

# visualize Trees
for i in range(3):
    tree = rf.estimators_[i]
    dot_data = export_graphviz(tree,
                               feature_names=train_x.columns,
                               filled=True,
                               max_depth=2,
                               class_names=['Yes','No'],
                               proportion=True)
    graph = graphviz.Source(dot_data,filename='tr'+str(i)+'.png', format="png")
    graph.view()


#################################################

#Adaboost
ad = AdaBoostClassifier(n_estimators=50)
ad.fit(train_x, train_y)

#k-fold validation
k_folds = KFold(n_splits = 5)
scores = cross_val_score(ad, train_x, train_y, cv = k_folds)
print("Scores:", scores," Scores Mean:", scores.mean())

#Prediction on test data
test_pred_y = rf.predict(test_x)
accuracy = accuracy_score(test_y, test_pred_y)
print("Accuracy:", accuracy)

# Visualize confusion matrix
cf = confusion_matrix(test_y, test_pred_y)
sns.heatmap(cf, annot=True, yticklabels=['Negative', 'Positive']
            , xticklabels=['Negative', 'Positive'], cmap='Blues', fmt='g')
plt.tight_layout()
plt.show()

# visualize decision stump
for i in range(3):
    tree = ad.estimators_[i]
    dot_data = export_graphviz(tree,
                               feature_names=train_x.columns,
                               filled=True,
                               proportion=True)
    graph = graphviz.Source(dot_data, filename='tr' + str(i) + '.png', format="png")
    graph.view()
