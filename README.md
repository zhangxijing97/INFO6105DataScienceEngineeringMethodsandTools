# INFO6105DataScienceEngineeringMethodsandTools

## Table of Contents

1. [Introduction to Data Science](#1-Introduction-to-Data-Science)
- [Supervised Approaches](#Supervised-Approaches)
- [Training Model](#Training-Model)
- [Train / Test/ Validation](#Train-Test-Validation)
- [Unsupervised Approaches](#Unsupervised-Approaches)
- [Loss Function in Unsupervised](#Loss-Function-in-Unsupervised)

2. [Data Preprocessing](#2-Data-Preprocessing)
- [Why Data Preprocessing](#Why-Data-Preprocessing)
- [Multiple Imputation by Chained Equation(MICE)](#Multiple-Imputation-by-Chained-Equations)

3. [Linear Classifiers](#3-Linear-Classifiers)

4. [Non-Linear Classifiers](#4-Non-Linear-Classifiers)

5. [Decision Trees](#5-Decision-Trees)

6. [Ensembles and Super learners](#6-Ensembles-and-Super-learners)

7. [Dimensionality Reduction](#7-Dimensionality-Reduction)

8. [Clustering Methods](#8-Clustering-Methods)

9. [ Association Rules](#9-Association-Rules)

10. [Introduction to Neural Networks / Deep Learning](#10-Introduction-to-Neural-Networks-and-Deep-Learning)

11. [Introduction to Big-data Analysis](#11-Introduction-to-Big-data-Analysis)

## 1. Introduction to Data Science
### Supervised Approaches
**Regression:** Learn a line/curve (the model) using training data consisting of Input-output pairs. Use it to predict the outputs for new inputs<br>
SVM(Support Vector Machines): Support Vector Machine (SVM) models have the ability to perform a non-linear regression / classification by mapping their inputs into high-dimensional feature spaces<br>

**Classification:** Learn to separate different classes (the model) using training data consisting of input-output pairs Use it to identify the labels for new inputs<br>
Ensembles: Ensemble methods are machine learning techniques that combines several models in order to produce optimal models<br>

### Training Model
Our training data comes in pairs of inputs (x,y)<br>
D={(x1,y1),...,(xn,yn)}<br>

xi: input vector of the ith sample (feature vector)<br>
yi: label of the ith sample Training dataset<br>
D: Training dataset

The goal of supervised learning is to develop a model h:<br>
h(xi)≈yi for all (xi,yi)∈D<br>

### Train, Test, Validation
**Training Set:** The model learns patterns and relationships within the training set. It is the data on which the model is trained to make predictions.<br>

**Testing Set:** Once the model is trained, it is evaluated on the testing set to assess its performance and generalization to new, unseen data. This set helps to estimate how well the model is likely to perform on new, real-world data.<br>

**Validation Set:** The validation set is another independent subset used during the training phase to fine-tune the model and avoid overfitting.<br>

**Best practice:**<br>
Train : 70%-80%<br>
Validation : 5%-10%<br>
Test: 10%-25%<br>

### Unsupervised Approaches
**Clustering:** Learn the grouping structure for a given set of unlabeled inputs<br>
**Association rule:** Association rule mining is a rule- based machine learning method for discovering interesting relations between variables in transactional databases.<br>
Example: basket analysis, where the goal is to uncover associations between items frequently purchased together.<br>

**Apriori Algorithm:** The Apriori algorithm is a widely used algorithm for mining association rules. It works by iteratively discovering frequent itemsets (sets of items that occur together frequently) and generating association rules based on these itemsets.<br>

Rule: X => Y<br>
X: antecedent (or left-hand side) items that when observed<br>
Y: consequent (or right-hand side) items that are expected or likely to be present when the conditions in the antecedent are mets<br>

A, B => C: it suggests that when both items A and B are present (antecedent), there is a likelihood that item C will also be present (consequent).<br>
{Milk, Bread} => {Eggs}: customers who buy both milk and bread are likely to buy eggs as well.<br>

**Support:** Support measures the frequency of occurrence of a particular combination of items in a dataset. High support values indicate that the itemset is common in the dataset.<br>

Support = frq(X,Y)/N<br>
frq(X, Y): This is the count of transactions where the itemset (X, Y) is present.<br>
N: This represents the total number of transactions or instances in the dataset.<br>
Support({Milk,Bread})= Number of transactions containing both Milk and Bread/Total number of transactions in the dataset<br>

**Confidence:** Confidence measures the likelihood that an associated rule holds true. It is the conditional probability of finding the consequent (item B) given the antecedent (item A). High confidence indicates a strong association between the antecedent and consequent.<br>

Confidence = frq(X,Y)/frq(X)<br>
frq(X, Y): This is the count of transactions where both the antecedent (X) and the consequent (Y) are present.<br>
frq(X): This is the count of transactions where the antecedent (X) is present.<br>
Confidence({Milk, Bread}⇒{Eggs}) = Number of transactions containing Milk, Bread, and Eggs/Number of transactions containing Milk and Bread<br>

**Lift:** Lift measures the strength of association between an antecedent and consequent, taking into account the support of both itemsets. A lift greater than 1 indicates that the presence of the antecedent increases the likelihood of the consequent.<br>

Lift = Support(X,Y)/[Support(X)*Support(Y)]<br>
Support(X, Y): This is the support of the itemset containing both X and Y<br>
Support(X): This is the support of the antecedent X<br>
Support(Y): This is the support of the consequent Y<br>

The lift formula essentially compares the observed co-occurrence of X and Y (Support(X, Y)) to what would be expected if X and Y were independent events (Support(X) * Support(Y))<br>
Lift = 1: X and Y are independent.<br>
Lift > 1: There is a positive association between X and Y (X and Y are more likely to occur together than expected).<br>
Lift < 1: There is a negative association between X and Y (X and Y are less likely to occur together than expected).<br>

Lift({Milk, Bread}⇒{Eggs})= Support({Milk, Bread, Eggs})/Support({Milk, Bread})×Support({Eggs})<br>

### Loss Function in Unsupervised
Unsupervised learning is about modeling the world<br>
**K-Means Clustering:** In k-means clustering, the goal is to partition a dataset into k clusters, where each data point belongs to the cluster with the nearest centroid.<br>
**Loss Function:** The loss function in k-means is typically the sum of squared distances between each data point and its assigned cluster centroid. The objective is to minimize this sum.<br>

Loss = ∑ N i=1 (xi - cji)^2<br>
MSE(Mean Squared Error) Loss = 1/N * Loss<br>
N: the number of data points<br>
xi: a data point<br>
cji: centroid of the cluster to which xi is assigned<br>

## 2. Data Preprocessing
### Why Data Preprocessing
- Incomplete: e.g., occupation=“ ”<br>
- Noisy: e.g., Salary=“-10”<br>
- Inconsistent: e.g., Age=“42” Birthday=“03/07/1997”<br>

**Data Cleaning**<br>
- Fill in missing values
- Smooth noisy data
- Identify or remove outliers
- Remove duplicates
- Resolve inconsistencies and discrepancies

**Data Transformation**<br>
- Normalization 
- Discretization

**Data Reduction**<br>
- Dimensionality reduction
- Numerosity reduction

**Data Integration**<br>
- Combining data from multiple sources into a unified dataset.

### Multiple Imputation by Chained Equations

**Univariate Imputation**<br>
In univariate imputation, each missing value in a dataset is imputed (filled in) based on information from the same variable.<br>

- Mean/Median/Mode Imputation: Missing values are replaced with the mean, median, or mode of the observed values in the same variable. This is simple and often effective but can distort the distribution of the data and underestimate the variability.<br>
- Random Sampling: Missing values are replaced with a value drawn randomly from the observed values of the same variable. This maintains the distribution but doesn't use any other information that might be helpful.<br>
- Constant Value: All missing values are filled in with a constant value, such as zero. This is a basic approach and is rarely used unless there is a strong justification.<br>

Example:<br>
| Student | Age | Test Score |
|---------|-----|------------|
| A       | 14  | 85         |
| B       | 13  | Missing    |
| C       | 14  | 90         |
| D       | 13  | 75         |
| E       | 14  | Missing    |

Mean Test Score = (85 + 90 + 75) / 3 = 83.33<br>

| Student | Age | Test Score |
|---------|-----|------------|
| A       | 14  | 85         |
| B       | 13  | 83.33      |
| C       | 14  | 90         |
| D       | 13  | 75         |
| E       | 14  | 83.33      |

**Multivariate Imputation**<br>
Multivariate imputation considers the relationships between different variables in the dataset when imputing missing values.<br>

- Multiple Imputation: It involves creating multiple complete datasets by imputing the missing values multiple times. Statistical models (like regression models) are used, considering the relationships among the variables. The results from these multiple datasets are then combined to give a final estimate. This method is useful as it also estimates the uncertainty due to missing data.<br>

| Age | Experience | Salary |
|-----|------------|--------|
| 25  |            | 50     |
| 27  | 3          |        |
| 29  | 5          | 110    |
| 31  | 7          | 140    |
| 33  | 9          | 170    |
|     | 11         | 200    |

Step 1: Impute all missing values with the mean<br>
| Age | Experience | Salary |
|-----|------------|--------|
| 25  | 7          | 50     |
| 27  | 3          | 134    |
| 29  | 5          | 110    |
| 31  | 7          | 140    |
| 33  | 9          | 170    |
| 29  | 11         | 200    |

Step 2: Romve the 'Age' inputed value<br>

Step 3: Use LinearRegression to estimate the missing age, the predicted age is 36.2532<br>
| Age | Experience | Salary |
|-----|------------|--------|
| 25  | 7          | 50     |
| 27  | 3          | 134    |
| 29  | 5          | 110    |
| 31  | 7          | 140    |
| 33  | 9          | 170    |
| 36.2532  | 11    | 200    |

Step 4: Romve the 'Experience' inputed value, and use LinearRegression to estimate the missing age, the predicted Experience is 1.8538<br>

Step 5: Romve the 'Salary' inputed value, and use LinearRegression to estimate the missing age, the predicted Experience is 72.7748, iteration 1 done<br>
| Age | Experience | Salary |
|-----|------------|--------|
| 25  | 1.8538     | 50     |
| 27  | 3          | 72.7748|
| 29  | 5          | 110    |
| 31  | 7          | 140    |
| 33  | 9          | 170    |
| 36.2532  | 11    | 200    |

Step 6: 
| Age | Experience | Salary |     | Age | Experience | Salary |     | Age | Experience | Salary  |
|-----|------------|--------|-----|-----|------------|--------|-----|-----|------------|---------|
| 25  | 1.8538     | 50     |     | 25  | 7          | 50     |     | 0   | -5.1462    | 0       |
| 27  | 3          | 72.7748|     | 27  | 3          | 134    |     | 0   | 0          | -61.2252|
| 29  | 5          | 110    |  -  | 29  | 5          | 110    |  =  | 0   | 0          | 0       |
| 31  | 7          | 140    |     | 31  | 7          | 140    |     | 0   | 0          | 0       |
| 33  | 9          | 170    |     | 33  | 9          | 170    |     | 0   | 0          | 0       |
| 36.2532  | 11    | 200    |     | 29  | 11         | 200    |     | 7.2532  | 0      | 0       |

```
from sklearn.linear_model import LinearRegression
import numpy as np

# Example data
X = np.array([[7, 50], [3, 134], [5, 110], [7, 140], [9, 170]])  # Experience and Salary
y = np.array([25, 27, 29, 31, 33])  # Age

# Create linear regression model
model = LinearRegression()
model.fit(X, y)

# Predict the missing age
predicted_age = model.predict([[11, 200]])  # Experience = 11, Salary = 200
print("Predicted Age:", predicted_age[0])
# Predicted Age: 36.25316455696203
```

```
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# Read data
input_dataframe = pd.read_csv("/Users/zhangxijing/MasterNEU/INFO6105DataScienceEngineeringMethodsandTools/Dataset/Microbiology_Dataset.csv")
print(input_dataframe)

# MICE
imputer = IterativeImputer(max_iter=10, random_state=0)
imputed_dataset = imputer.fit_transform(input_dataframe)
imputed_dataframe = pd.DataFrame(imputed_dataset, columns=input_dataframe.columns)
print(imputed_dataframe)
```

## 3. Linear Classifiers
## 4. Non-Linear Classifiers
## 5. Decision Trees
## 6. Ensembles and Super learners
## 7. Dimensionality Reduction
## 8. Clustering Methods
## 9. Association Rules
## 10. Introduction to Neural Networks and Deep Learning
## 11. Introduction to Big-data Analysis

<!-- | Text with Color | Another Text |
| --------------- | ------------ |
| <span style="color:red;">Red Text</span> | <span style="color:blue;">Blue Text</span> | -->