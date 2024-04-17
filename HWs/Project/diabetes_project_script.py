import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score

# Load data
file_path = '/Users/jeffrey/Documents/Northeastern University/INFO 6105/Project/diabetes_project.csv'
data = pd.read_csv(file_path)

# Preprocess data: Remove outliers, impute missing values, normalize
def preprocess_data(data):
    for column in data.columns:
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        data = data[(data[column] >= (Q1 - 1.5 * IQR)) & (data[column] <= (Q3 + 1.5 * IQR))]

    imputer = KNNImputer(n_neighbors=5)
    data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

    scaler = StandardScaler()
    data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
    return data

data = preprocess_data(data)

# Unsupervised learning for label generation
def generate_labels(data):
    clustering_features = data[['Glucose', 'BMI', 'Age']]
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(clustering_features)
    data['Outcome'] = clusters
    cluster0_mean_glucose = data.loc[data['Outcome'] == 0, 'Glucose'].mean()
    cluster1_mean_glucose = data.loc[data['Outcome'] == 1, 'Glucose'].mean()
    if cluster0_mean_glucose > cluster1_mean_glucose:
        data['Outcome'] = data['Outcome'].replace({0: 1, 1: 0})
    return data

data = generate_labels(data)

# Split data
X = data.drop('Outcome', axis=1)
y = data['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ensemble model with cross-validation
def build_ensemble(X_train, y_train):
    models = [
        ('lr', LogisticRegression(max_iter=1000)),
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42))
    ]
    ensemble = VotingClassifier(estimators=models, voting='soft')
    ensemble.fit(X_train, y_train)
    return ensemble

ensemble = build_ensemble(X_train, y_train)

# Evaluate the model using cross-validation
scores = cross_val_score(ensemble, X, y, cv=5)
accuracy = scores.mean()
print(f'Cross-validated accuracy: {accuracy:.2f}')
