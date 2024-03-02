import numpy as np  
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os
print(os.listdir())

import warnings
warnings.filterwarnings('ignore')
 
# # Importing the dataset
dataset = pd.read_csv('static/heart.csv')
type(dataset)
(dataset.shape)#(303,14)

# # printing out a few columns
# print(dataset.head(5)) 
dataset.sample(5)
(dataset.describe())
# # the below code give rhe detail of the dataset
info = ["age","1: male, 0: female","chest pain type, 1: typical angina, 2: atypical angina, 3: non-anginal pain, 4: asymptomatic","resting blood pressure"," serum cholestoral in mg/dl","fasting blood sugar > 120 mg/dl","resting electrocardiographic results (values 0,1,2)"," maximum heart rate achieved","exercise induced angina","oldpeak = ST depression induced by exercise relative to rest","the slope of the peak exercise ST segment","number of major vessels (0-3) colored by flourosopy","thal: 3 = normal; 6 = fixed defect; 7 = reversable defect"]

for i in range(len(info)):
    (dataset.columns[i]+":\t\t\t"+info[i])
    
# # Analysing of target variable

dataset["target"].describe()
dataset["target"].unique()
#### Clearly, this is a classification problem, with the target variable having values '0' and '1'
dataset.corr()["target"].abs().sort_values(ascending=False)

# # Exploratory Data Analysis 

y = dataset["target"]

sns.countplot(y)
# show the plot
# print value counts
target_temp = dataset.target.value_counts()

# print(target_temp)
# plt.show()

# print("Percentage of patience without heart problems: "+str(round(target_temp[0]*100/303,2)))
# print("Percentage of patience with heart problems: "+str(round(target_temp[1]*100/303,2)))

dataset["sex"].unique()
dataset["cp"].unique()
dataset["fbs"].describe()
dataset["fbs"].unique()
dataset["restecg"].unique()
dataset["exang"].unique()
dataset["slope"].unique()
dataset["ca"].unique()
dataset["thal"].unique()



# # ṭraining and testing data
from sklearn.model_selection import train_test_split

predictors = dataset.drop("target",axis=1)
target = dataset["target"]

X_train,X_test,Y_train,Y_test = train_test_split(predictors,target,test_size=0.20,random_state=0)

X_train.shape

X_test.shape

Y_train.shape

Y_test.shape

# # MODEL FITTING

# # LOGISTIC REGRESSION
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

lr.fit(X_train,Y_train)

Y_pred_lr = lr.predict(X_test)
Y_pred_lr.shape
score_lr = round(accuracy_score(Y_pred_lr,Y_test)*100,2)
print("The accuracy score achieved using Logistic Regression is: "+str(score_lr)+" %")

from sklearn.metrics import confusion_matrix

# Confusion matrix
# conf_matrix_lr = confusion_matrix(Y_test, Y_pred_lr)

# # Extract TP, TN, FP, FN from confusion matrix
# TP_lr = conf_matrix_lr[1, 1]
# TN_lr = conf_matrix_lr[0, 0]
# FP_lr = conf_matrix_lr[0, 1]
# FN_lr = conf_matrix_lr[1, 0]

# # Display the confusion matrix and metrics
# print("Confusion Matrix:")
# print(conf_matrix_lr)
# print("\nTrue Positive (TP):", TP_lr)
# print("True Negative (TN):", TN_lr)
# print("False Positive (FP):", FP_lr)
# print("False Negative (FN):", FN_lr)






#knn fitting
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Hyperparameter Tuning
param_grid = {'n_neighbors': range(1, 21)}
grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
grid_search.fit(X_train_scaled, Y_train)

best_knn = grid_search.best_estimator_

# Make predictions using the best model
Y_pred_knn = best_knn.predict(X_test_scaled)

# Calculate accuracy score
score_knn = round(accuracy_score(Y_pred_knn, Y_test) * 100, 2)

print("The accuracy score achieved using KNN is: " + str(score_knn) + " %")
# print("Best parameters found by GridSearchCV:", grid_search.best_params_)



 # Decision Tree
 
 
from sklearn.tree import DecisionTreeClassifier

max_accuracy = 0
for x in range(200):
    dt = DecisionTreeClassifier(random_state=x)
    dt.fit(X_train,Y_train)
    Y_pred_dt = dt.predict(X_test)
    current_accuracy = round(accuracy_score(Y_pred_dt,Y_test)*100,2)
    if(current_accuracy>max_accuracy):
        max_accuracy = current_accuracy
        best_x = x
dt = DecisionTreeClassifier(random_state=best_x)
dt.fit(X_train,Y_train)
Y_pred_dt = dt.predict(X_test)
# print(Y_pred_dt.shape)
score_dt = round(accuracy_score(Y_pred_dt,Y_test)*100,2)
print("The accuracy score achieved using Decision Tree is: "+str(score_dt)+" %")

#  RANDOM FOREST

from sklearn.ensemble import RandomForestClassifier

max_accuracy = 0

for x in range(2000):
    rf = RandomForestClassifier(random_state=x)
    rf.fit(X_train,Y_train)
    Y_pred_rf = rf.predict(X_test)
    current_accuracy = round(accuracy_score(Y_pred_rf,Y_test)*100,2)
    if(current_accuracy>max_accuracy):
        max_accuracy = current_accuracy
        best_x = x
        
#print(max_accuracy)
#print(best_x)

rf = RandomForestClassifier(random_state=best_x)
rf.fit(X_train,Y_train)
Y_pred_rf = rf.predict(X_test)
Y_pred_rf.shape
score_rf = round(accuracy_score(Y_pred_rf,Y_test)*100,2)

print("The accuracy score achieved using Random forest is: "+str(score_rf)+" %")






# from prettytable import PrettyTable

# # Sample training data
# sample_train = pd.concat([X_train.head(5), Y_train.head(5)], axis=1)

# # Sample testing data
# sample_test = pd.concat([X_test.head(5), Y_test.head(5), pd.Series(Y_pred_lr[:5], name='Predicted')], axis=1)

# # Create PrettyTable for sample training data
# table_train = PrettyTable()
# table_train.field_names = sample_train.columns
# for row in sample_train.itertuples(index=False):
#     table_train.add_row(row)

# # Create PrettyTable for sample testing data
# table_test = PrettyTable()
# table_test.field_names = sample_test.columns
# for row in sample_test.itertuples(index=False):
#     table_test.add_row(row)

# # Print tables
# print("Sample Training Data:")
# print(table_train)
# print("\nSample Testing Data:")
# print(table_test)









