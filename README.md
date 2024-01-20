# INFO6105DataScienceEngineeringMethodsandTools

## Table of Contents

1. [Introduction to Data Science](#1-Introduction-to-Data-Science)
- [Some Supervised Approaches](#Some-Supervised-Approaches)
- [Training Model](#Training-Model)
- [Train / Test/ Validation](#Train-Test-Validation)
- [Some Unsupervised Approaches](#Some-Unsupervised-Approaches)

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
### Some Supervised Approaches
**Regression:** Learn a line/curve (the model) using training data consisting of Input-output pairs. Use it to predict the outputs for new inputs<br>
SVM(Support Vector Machines): Support Vector Machine (SVM) models have the ability to perform a non-linear regression / classification by mapping their inputs into high-dimensional feature spaces<br>

**Classification:** Learn to separate different classes (the model) using training data consisting of input-output pairs Use it to identify the labels for new inputs<br>
Ensembles: Ensemble methods are machine learning techniques that combines several models in order to produce optimal models<br>

### Training Model
• Our training data comes in pairs of inputs (x,y)<br>
• D={(x1,y1),...,(xn,yn)}<br>

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

### Some Unsupervised Approaches
**Clustering:** Learn the grouping structure for a given set of unlabeled inputs<br>
**Association rule:** Association rule mining is a rule- based machine learning method for discovering interesting relations between variables in transactional databases.<br>
Example: basket analysis, where the goal is to uncover associations between items frequently purchased together.<br>


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