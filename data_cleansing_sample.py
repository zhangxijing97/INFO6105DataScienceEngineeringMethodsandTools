import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

input_dataframe = pd.read_csv("Dataset_lec1.csv")
print(input_dataframe)

##MICE
imputer = IterativeImputer(max_iter=10, random_state=0)
imputed_dataset = imputer.fit_transform(input_dataframe)
imputed_dataframe = pd.DataFrame(imputed_dataset, columns=input_dataframe.columns)
print(imputed_dataframe)

##Bining
imputed_dataframe['fld2_bin'] = pd.qcut(imputed_dataframe['fld2'], q=3)
imputed_dataframe['fld2'] = pd.Series ([interval.mid for interval in imputed_dataframe['fld2_bin']])
print(imputed_dataframe)

##Detect Outliers
q3, q1 = np.percentile(imputed_dataframe['fld3'], [75 ,25])
fence = 1.5 * (q3 - q1)
upper_band = q3 + fence
lower_band = q1 - fence
print( 'q1=',q1,' q3=', q3, ' IQR=', q3-q1, ' upper=', upper_band, 'lower=',lower_band)
outliers = [val for val in imputed_dataframe['fld3'] if val < lower_band or val > upper_band]
print('Outliers=', outliers)

##Normalize
from sklearn import preprocessing
fld1_array = np.array(imputed_dataframe['fld1']).reshape(-1,1)
scaler = preprocessing.StandardScaler()
#scaler = preprocessing.MinMaxScaler()
scaler.fit(fld1_array)
scaled_fld1=scaler.transform(fld1_array)
print(scaled_fld1)
