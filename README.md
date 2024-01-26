# INFO6105DataScienceEngineeringMethodsandTools

## Table of Contents

1. [Introduction to Data Science](#1-Introduction-to-Data-Science)
- [Supervised Approaches](#Supervised-Approaches)
- [Training Model](#Training-Model)
- [Train / Test/ Validation](#Train-Test-Validation)
- [Unsupervised Approaches](#Unsupervised-Approaches)
- [Loss Function in Unsupervised](#Loss-Function-in-Unsupervised)

2. [Data Preprocessing](#2-Data-Preprocessing)

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
## 3. Linear Classifiers
## 4. Non-Linear Classifiers
## 5. Decision Trees
## 6. Ensembles and Super learners
## 7. Dimensionality Reduction
## 8. Clustering Methods
## 9. Association Rules
## 10. Introduction to Neural Networks and Deep Learning
## 11. Introduction to Big-data Analysis