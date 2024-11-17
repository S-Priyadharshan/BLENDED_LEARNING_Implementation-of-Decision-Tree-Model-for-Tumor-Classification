# BLENDED_LEARNING
# Implementation of Decision Tree Model for Tumor Classification

## AIM:
To implement and evaluate a Decision Tree model to classify tumors as benign or malignant using a dataset of lab test results.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Data Preparation: Collect and preprocess tumor data, including features like size and texture, and label them as benign or malignant.
Model Training: Split the data into training and test sets, then train a decision tree classifier on the training set.
Model Evaluation: Evaluate the decision tree on the test set using metrics like accuracy, precision, and recall.
Prediction: Use the trained decision tree model to classify new tumor samples as benign or malignant.

## Program:
```
/*
Program to  implement a Decision Tree model for tumor classification.
Developed by: Priyadharshan S
RegisterNumber:  212223240127
*/

import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn import metrics
# Evaluation metrics related methods
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix, precision_recall_fscore_support, precision_score, recall_score

import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

rs = 123

dataset_url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML241EN-SkillsNetwork/labs/datasets/tumor.csv"
tumor_df = pd.read_csv(dataset_url)

# Get the input features
X = tumor_df.iloc[:, :-1]
# Get the target variable
y = tumor_df.iloc[:, -1:]

# Split the training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state = rs)

# Train a decision tree with all default arguments
model = DecisionTreeClassifier(random_state=rs)

model.fit(X_train, y_train.values.ravel())

preds = model.predict(X_test)

def evaluate_metrics(yt, yp):
    results_pos = {}
    results_pos['accuracy'] = accuracy_score(yt, yp)
    precision, recall, f_beta, _ = precision_recall_fscore_support(yt, yp, average='binary')
    results_pos['recall'] = recall
    results_pos['precision'] = precision
    results_pos['f1score'] = f_beta
    return results_pos

evaluate_metrics(y_test, preds)

custom_model = DecisionTreeClassifier(criterion='entropy', max_depth=10, min_samples_leaf=3, random_state=rs)
custom_model.fit(X_train, y_train.values.ravel())
preds = custom_model.predict(X_test)
evaluate_metrics(y_test, preds)
```

## Output:
![image](https://github.com/user-attachments/assets/51146adb-af32-4a74-b699-0f878e6b1e50)
![image](https://github.com/user-attachments/assets/7ccce2f1-abd8-4c80-931a-fe01ace133e6)
![image](https://github.com/user-attachments/assets/356433c0-fddd-4061-9f3b-4a98a6ccbef4)

## Result:
Thus, the Decision Tree model was successfully implemented to classify tumors as benign or malignant, and the model’s performance was evaluated.
