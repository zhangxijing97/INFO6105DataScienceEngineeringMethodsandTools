import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
from kmodes.kmodes import KModes
from pyclustering.cluster.kmedians import kmedians
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering
from sklearn import preprocessing


def normalize_data(org_df,col):
    col_array = np.array(org_df[col]).reshape(-1, 1)
    scaler = preprocessing.StandardScaler()
    scaler.fit(col_array)
    org_df[col] = scaler.transform(col_array)
    return org_df

def prepare_data(org_df):
    #select target attributes
    org_df = org_df.loc[:,org_df.columns.isin(['MIC','MIC_Interpretation', 'Antimicrobial', 'Patient_Age', 'Bacteria'])]

    # Categorical to Numeric
    org_df = pd.get_dummies(org_df, columns=['MIC_Interpretation', 'Antimicrobial', 'Bacteria'], dtype='int')

    #remove_outliers in Patient_Age
    values = [vl for vl in org_df['Patient_Age'] if not np.isnan(vl)]
    q3, q1 = np.percentile(values, [75, 25])
    fence = 1.5 * (q3 - q1)
    upper_band = q3 + fence
    lower_band = q1 - fence
    org_df.loc[(org_df['Patient_Age'] < lower_band) | (org_df['Patient_Age'] > upper_band), 'Patient_Age'] = None

    #impute dataset
    imputer = IterativeImputer(max_iter=10, random_state=0)
    imputed_dataset = imputer.fit_transform(org_df)
    imputed_dataframe = pd.DataFrame(imputed_dataset, columns=org_df.columns)
    return imputed_dataframe


#Input Dateset
org_df = pd.read_csv("DS_Dataset.csv")
train_feat = prepare_data(org_df)
# train_feat = normalize_data(train_feat,'MIC')
# train_feat = normalize_data(train_feat,'Patient_Age')

#kmeans
model = KMeans(n_clusters=2)
model.fit(train_feat)

# #kmode
# model =KModes(n_clusters=2)
# model.fit(train_feat)

# filter rows based on cluster
first_cluster = train_feat.loc[model.labels_ == 1,:]
second_cluster = train_feat.loc[model.labels_ == 0,:]

# Plotting the results
plt.scatter(first_cluster.loc[:, 'Patient_Age'], first_cluster.loc[:, 'MIC'], color='red')
plt.scatter(second_cluster.loc[:, 'Patient_Age'], second_cluster.loc[:, 'MIC'], color='black')
plt.show()

#kmedoids
# model = KMedoids(n_clusters=2)
# model.fit(train_feat)
# print(kmedoids.labels_)
#
inertias = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(train_feat)
    inertias.append(kmeans.inertia_) #inertia_ : Sum of squared distances of samples to their closest cluster center


plt.plot(range(1,11), inertias, marker='o')
plt.title('Elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('Sum of squared distances')
plt.show()

#Agnes
linkage_data = linkage(train_feat, method='single', metric='euclidean')
dendrogram(linkage_data, truncate_mode = 'level' ,p=5)
plt.show()


#%%
