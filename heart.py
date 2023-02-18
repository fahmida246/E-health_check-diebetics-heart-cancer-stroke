# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 19:15:32 2022

@author: nus34
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
#%%
df = pd.read_csv('heart.csv')
print(df)
print(df.head())
#%%
#shape and column targets value count
print(df.shape)
print(df.target.value_counts() )
#%%
# is there any missing value?
print(df.isna().sum())
#%%
X = df.drop('target',1)
y = df['target']
print('shape of X and y respectively :', X.shape, y.shape)
#%%
#shape of x and y
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print('shape of X and y respectively (train) :', X_train.shape, y_train.shape)
print('shape of X and y respectively (test) :', X_test.shape, y_test.shape)
#%%



#%% 3
#************logistic regresssion*************************
print('Logistic Regression')
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
Y_pred = model.predict(X_test)

score1 = model.score(X_train, y_train)
print('Training Score:', score1)

score = model.score(X_test, y_test)
print('Testing Score:', score)

output = pd.DataFrame({'Predicted':Y_pred}) 
print('print first 5 predicted values:')
print(output.head())

people = output.loc[output.Predicted == 1]["Predicted"]
rate_people = 0
if len(people) > 0 :
    rate_people = len(people)/len(output)
print("% of people predicted with heart-disease:", rate_people)

score_logreg = score
out_logreg = output
a1=metrics.accuracy_score(y_test, Y_pred)

from sklearn.metrics import classification_report
print(classification_report(y_test,Y_pred))
print("Accuracy:",a1)

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test,Y_pred)
class_names = [0,1]
fig,ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks,class_names)
plt.yticks(tick_marks,class_names)
sns.heatmap(pd.DataFrame(confusion_matrix), annot = True, cmap = 'Greens', fmt = 'g')
ax.xaxis.set_label_position('top')
plt.tight_layout()
plt.title('Confusion matrix for logistic regression')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()




#%% 1
#************decision tree classifier *************************
print('DecisionTreeClassifier')
from sklearn.tree import DecisionTreeClassifier
decision_tree = DecisionTreeClassifier(max_depth=5) 
decision_tree.fit(X_train, y_train)  
Y_pred = model.predict(X_test)

score1 = model.score(X_train, y_train)
print('Training Score:', score1)

score = model.score(X_test, y_test)
print('Testing Score:', score)

output = pd.DataFrame({'Predicted':Y_pred}) 
print(output.head())

people = output.loc[output.Predicted == 1]["Predicted"]
rate_people = 0
if len(people) > 0 :
    rate_people = len(people)/len(output)
print("% of people predicted with heart-disease:", rate_people)

score_dtc = score
out_dtc = output

a2=metrics.accuracy_score(y_test, Y_pred)
from sklearn.metrics import classification_report
print(classification_report(y_test,Y_pred))
print("Accuracy:",metrics.accuracy_score(y_test, Y_pred))


from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test,Y_pred)
class_names = [0,1]
fig,ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks,class_names)
plt.yticks(tick_marks,class_names)
sns.heatmap(pd.DataFrame(confusion_matrix), annot = True, cmap = 'Blues', fmt = 'g')
ax.xaxis.set_label_position('top')
plt.tight_layout()
plt.title('Confusion matrix for decision tree')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()

from sklearn.metrics import roc_auc_score,roc_curve
y_probabilities = model.predict_proba(X_test)[:,1]
false_positive_rate_knn, true_positive_rate_knn, threshold_knn = roc_curve(y_test,y_probabilities)
plt.figure(figsize=(10,6))
plt.title('ROC for decision tree')
plt.plot(false_positive_rate_knn, true_positive_rate_knn, linewidth=5, color='green')
plt.plot([0,1],ls='--',linewidth=5)
plt.plot([0,0],[1,0],c='.5')
plt.plot([1,1],c='.5')
plt.text(0.2,0.6,'AUC: {:.2f}'.format(roc_auc_score(y_test,y_probabilities)),size= 16)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()
#%% 2
#************random forest tree classifier *************************
print('RandomForestClassifier')
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100) 
model.fit(X_train, y_train)
Y_pred = model.predict(X_test)
score1 = model.score(X_train, y_train)

print('Training Score:', score1)
score = model.score(X_test, y_test)

print('Testing Score:', score)
output = pd.DataFrame({'Predicted':Y_pred})
print(output.head())

people = output.loc[output.Predicted == 1]["Predicted"]
rate_people = 0
if len(people) > 0 :
    rate_people = len(people)/len(output)
print("% of people predicted with heart-disease:", rate_people)

score_rfc = score
out_rfc = output

a3=metrics.accuracy_score(y_test, Y_pred)

from sklearn.metrics import classification_report
print(classification_report(y_test,Y_pred))
print("Accuracy:",metrics.accuracy_score(y_test, Y_pred))

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test,Y_pred)
class_names = [0,1]
fig,ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks,class_names)
plt.yticks(tick_marks,class_names)
sns.heatmap(pd.DataFrame(confusion_matrix), annot = True, cmap = 'Reds', fmt = 'g')
ax.xaxis.set_label_position('top')
plt.tight_layout()
plt.title('Confusion matrix for random forest')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()
from sklearn.metrics import roc_auc_score,roc_curve
y_probabilities = model.predict_proba(X_test)[:,1]
false_positive_rate_knn, true_positive_rate_knn, threshold_knn = roc_curve(y_test,y_probabilities)
plt.figure(figsize=(10,6))
plt.title('ROC for random forest')
plt.plot(false_positive_rate_knn, true_positive_rate_knn, linewidth=5, color='green')
plt.plot([0,1],ls='--',linewidth=5)
plt.plot([0,0],[1,0],c='.5')
plt.plot([1,1],c='.5')
plt.text(0.2,0.6,'AUC: {:.2f}'.format(roc_auc_score(y_test,y_probabilities)),size= 16)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()
#%% 4
#************kneighbours classifier *************************
print('KNeighborsClassifier')
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()
model.fit(X_train, y_train)
Y_pred = model.predict(X_test)
score1 = model.score(X_train, y_train)

print('Training Score:', score1)
score = model.score(X_test, y_test)

print('Testing Score:', score)
output = pd.DataFrame({'Predicted':Y_pred})
print(output.head())

people = output.loc[output.Predicted == 1]["Predicted"]
rate_people = 0
if len(people) > 0 :
    rate_people = len(people)/len(output)
print("% of people predicted with heart-disease:", rate_people)

score_knc = score
out_knc = output
a4=metrics.accuracy_score(y_test, Y_pred)

from sklearn.metrics import classification_report
print(classification_report(y_test,Y_pred))
print("Accuracy:",metrics.accuracy_score(y_test, Y_pred))

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test,Y_pred)
class_names = [0,1]
fig,ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks,class_names)
plt.yticks(tick_marks,class_names)
sns.heatmap(pd.DataFrame(confusion_matrix), annot = True, cmap = 'Greens', fmt = 'g')
ax.xaxis.set_label_position('top')
plt.tight_layout()
plt.title('Confusion matrix for knc')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()
from sklearn.metrics import roc_auc_score,roc_curve
y_probabilities = model.predict_proba(X_test)[:,1]
false_positive_rate_knn, true_positive_rate_knn, threshold_knn = roc_curve(y_test,y_probabilities)
plt.figure(figsize=(10,6))
plt.title('ROC for kneighbour')
plt.plot(false_positive_rate_knn, true_positive_rate_knn, linewidth=5, color='green')
plt.plot([0,1],ls='--',linewidth=5)
plt.plot([0,0],[1,0],c='.5')
plt.plot([1,1],c='.5')
plt.text(0.2,0.6,'AUC: {:.2f}'.format(roc_auc_score(y_test,y_probabilities)),size= 16)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()
#%%  5
#*********************naive bayes*************
print('naive bayes')
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(X_train, y_train)
Y_pred = model.predict(X_test)
score1 = model.score(X_train, y_train)

print('Training Score:', score1)
score = model.score(X_test, y_test)

print('Testing Score:', score)
output = pd.DataFrame({'Predicted':Y_pred}) 

print(output.head())

people = output.loc[output.Predicted == 1]["Predicted"]
rate_people = 0
if len(people) > 0 :
    rate_people = len(people)/len(output)
print("% of people predicted with heart-disease:", rate_people)

score_nb = score
out_nb = output

a5=metrics.accuracy_score(y_test, Y_pred)

from sklearn.metrics import classification_report
print(classification_report(y_test,Y_pred))
print("Accuracy:",metrics.accuracy_score(y_test, Y_pred)) 


from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test,Y_pred)
class_names = [0,1]
fig,ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks,class_names)
plt.yticks(tick_marks,class_names)
sns.heatmap(pd.DataFrame(confusion_matrix), annot = True, cmap = 'Blues', fmt = 'g')
ax.xaxis.set_label_position('top')
plt.tight_layout()
plt.title('Confusion matrix for navie bayes')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()

from sklearn.metrics import roc_auc_score,roc_curve
y_probabilities = model.predict_proba(X_test)[:,1]
false_positive_rate_knn, true_positive_rate_knn, threshold_knn = roc_curve(y_test,y_probabilities)
plt.figure(figsize=(10,6))
plt.title('ROC for naive bayes')
plt.plot(false_positive_rate_knn, true_positive_rate_knn, linewidth=5, color='green')
plt.plot([0,1],ls='--',linewidth=5)
plt.plot([0,0],[1,0],c='.5')
plt.plot([1,1],c='.5')
plt.text(0.2,0.6,'AUC: {:.2f}'.format(roc_auc_score(y_test,y_probabilities)),size= 16)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()
#%%
#support vector classifier using 3 different tecniques linear,rbf and poly

from sklearn.svm import SVC
print('using linear kernel')
model = SVC( kernel='linear')
model.fit(X_train, y_train)
Y_pred = model.predict(X_test)
score = model.score(X_train, y_train)

print('Training Score:', score)
score = model.score(X_test, y_test)

print('Testing Score:', score)
output = pd.DataFrame({'Predicted':Y_pred}) 
print(output.head())

people = output.loc[output.Predicted == 1]["Predicted"]
rate_people = 0
if len(people) > 0 :
    rate_people = len(people)/len(output)
print("% of people predicted with heart-disease:", rate_people)

score_lin = score
out_lin = output

a6=metrics.accuracy_score(y_test, Y_pred)

from sklearn.metrics import classification_report
print(classification_report(y_test,Y_pred))
print("Accuracy:",metrics.accuracy_score(y_test, Y_pred)) 

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test,Y_pred)
class_names = [0,1]
fig,ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks,class_names)
plt.yticks(tick_marks,class_names)
sns.heatmap(pd.DataFrame(confusion_matrix), annot = True, cmap = 'Reds', fmt = 'g')
ax.xaxis.set_label_position('top')
plt.tight_layout()
plt.title('Confusion matrix for svm linear')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()
#%%
#************************
print('using poly kernel')
model = SVC( kernel='poly')
model.fit(X_train, y_train)
Y_pred = model.predict(X_test)
score = model.score(X_train, y_train)

print('Training Score:', score)
score = model.score(X_test, y_test)

print('Testing Score:', score)
output = pd.DataFrame({'Predicted':Y_pred}) 
print(output.head())

people = output.loc[output.Predicted == 1]["Predicted"]
rate_people = 0
if len(people) > 0 :
    rate_people = len(people)/len(output)
print("% of people predicted with heart-disease:", rate_people)

score_pol = score
out_pol = output

a7=metrics.accuracy_score(y_test, Y_pred)

from sklearn.metrics import classification_report
print(classification_report(y_test,Y_pred))
print("Accuracy:",metrics.accuracy_score(y_test, Y_pred))

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test,Y_pred)
class_names = [0,1]
fig,ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks,class_names)
plt.yticks(tick_marks,class_names)
sns.heatmap(pd.DataFrame(confusion_matrix), annot = True, cmap = 'Greens', fmt = 'g')
ax.xaxis.set_label_position('top')
plt.tight_layout()
plt.title('Confusion matrix for svm poly')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show() 
#%%
#************************
print('using rbf kernel')
model = SVC( kernel='rbf')
model.fit(X_train, y_train)
Y_pred = model.predict(X_test)

score = model.score(X_train, y_train)
print('Training Score:', score)

score = model.score(X_test, y_test)
print('Testing Score:', score)

output = pd.DataFrame({'Predicted':Y_pred}) 
print(output.head())

people = output.loc[output.Predicted == 1]["Predicted"]
rate_people = 0
if len(people) > 0 :
    rate_people = len(people)/len(output)
print("% of people predicted with heart-disease:", rate_people)

score_rbf = score
out_rbf = output

a8=metrics.accuracy_score(y_test, Y_pred)

from sklearn.metrics import classification_report
print(classification_report(y_test,Y_pred))
print("Accuracy:",metrics.accuracy_score(y_test, Y_pred)) 

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test,Y_pred)
class_names = [0,1]
fig,ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks,class_names)
plt.yticks(tick_marks,class_names)
sns.heatmap(pd.DataFrame(confusion_matrix), annot = True, cmap = 'Blues', fmt = 'g')
ax.xaxis.set_label_position('top')
plt.tight_layout()
plt.title('Confusion matrix for svm rbf')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()
#%%
#************************linear discriminante
print('using linear discriminante')
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
model = LinearDiscriminantAnalysis()
model.fit(X_train, y_train)
Y_pred = model.predict(X_test)
score = model.score(X_train, y_train)

print('Training Score:', score)
score = model.score(X_test, y_test)

print('Testing Score:', score)
output = pd.DataFrame({'Predicted':Y_pred})

print(output.head())
people = output.loc[output.Predicted == 1]["Predicted"]
rate_people = 0
if len(people) > 0 :
    rate_people = len(people)/len(output)
    
print("% of people predicted with heart-disease:", rate_people)
score_disc = score
out_disc = output

a9=metrics.accuracy_score(y_test, Y_pred)

from sklearn.metrics import classification_report
print(classification_report(y_test,Y_pred))
print("Accuracy:",metrics.accuracy_score(y_test, Y_pred)) 

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test,Y_pred)
class_names = [0,1]
fig,ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks,class_names)
plt.yticks(tick_marks,class_names)
sns.heatmap(pd.DataFrame(confusion_matrix), annot = True, cmap = 'Reds', fmt = 'g')
ax.xaxis.set_label_position('top')
plt.tight_layout()
plt.title('Confusion matrix for linear discriminante analysis')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()


#%% 8
#using adaboost
print('adaboost')
from sklearn.ensemble import AdaBoostClassifier
model = AdaBoostClassifier()
model.fit(X_train, y_train)
Y_pred = model.predict(X_test)
score = model.score(X_train, y_train)

print('Training Score:', score)
score = model.score(X_test, y_test)

print('Testing Score:', score)
output = pd.DataFrame({'Predicted':Y_pred}) 
print(output.head())

people = output.loc[output.Predicted == 1]["Predicted"]
rate_people = 0
if len(people) > 0 :
    rate_people = len(people)/len(output)
print("% of people predicted with heart-disease:", rate_people)

score_ada = score
out_ada = output

a10=metrics.accuracy_score(y_test, Y_pred)

from sklearn.metrics import classification_report
print(classification_report(y_test,Y_pred))
print("Accuracy:",metrics.accuracy_score(y_test, Y_pred)) 

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test,Y_pred)
class_names = [0,1]
fig,ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks,class_names)
plt.yticks(tick_marks,class_names)
sns.heatmap(pd.DataFrame(confusion_matrix), annot = True, cmap = 'Greens', fmt = 'g')
ax.xaxis.set_label_position('top')
plt.tight_layout()
plt.title('Confusion matrix for adaboost')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()

from sklearn.metrics import roc_auc_score,roc_curve
y_probabilities = model.predict_proba(X_test)[:,1]
false_positive_rate_knn, true_positive_rate_knn, threshold_knn = roc_curve(y_test,y_probabilities)
plt.figure(figsize=(10,6))
plt.title('ROC for adaboost')
plt.plot(false_positive_rate_knn, true_positive_rate_knn, linewidth=5, color='green')
plt.plot([0,1],ls='--',linewidth=5)
plt.plot([0,0],[1,0],c='.5')
plt.plot([1,1],c='.5')
plt.text(0.2,0.6,'AUC: {:.2f}'.format(roc_auc_score(y_test,y_probabilities)),size= 16)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()
#%%
#using xgboost
from xgboost import XGBClassifier
model = XGBClassifier(n_jobs=-1,n_estimators=12, random_state=4)
model.fit(X_train, y_train)
Y_pred = model.predict(X_test)
score = model.score(X_train, y_train)

print('Training Score:', score)
score = model.score(X_test, y_test)

print('Testing Score:', score)
output = pd.DataFrame({'Predicted':Y_pred}) # Heart-Disease yes or no? 1/0
print(output.head())

people = output.loc[output.Predicted == 1]["Predicted"]
rate_people = 0
if len(people) > 0 :
    rate_people = len(people)/len(output)
print("% of people predicted with heart-disease:", rate_people)

score_xgb = score
out_xgb = output
a11=metrics.accuracy_score(y_test, Y_pred)

from sklearn.metrics import classification_report
print(classification_report(y_test,Y_pred))
print("Accuracy:",metrics.accuracy_score(y_test, Y_pred)) 

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test,Y_pred)
class_names = [0,1]
fig,ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks,class_names)
plt.yticks(tick_marks,class_names)
sns.heatmap(pd.DataFrame(confusion_matrix), annot = True, cmap = 'Greens', fmt = 'g')
ax.xaxis.set_label_position('top')
plt.tight_layout()
plt.title('Confusion matrix for xgboost')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()

from sklearn.metrics import roc_auc_score,roc_curve
y_probabilities = model.predict_proba(X_test)[:,1]
false_positive_rate_knn, true_positive_rate_knn, threshold_knn = roc_curve(y_test,y_probabilities)
plt.figure(figsize=(10,6))
plt.title('ROC for logistic regression')
plt.plot(false_positive_rate_knn, true_positive_rate_knn, linewidth=5, color='green')
plt.plot([0,1],ls='--',linewidth=5)
plt.plot([0,0],[1,0],c='.5')
plt.plot([1,1],c='.5')
plt.text(0.2,0.6,'AUC: {:.2f}'.format(roc_auc_score(y_test,y_probabilities)),size= 16)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()

#%%
#using neural network
'''
from keras.models import Sequential
model = Sequential()
model.fit(X_train, y_train)
Y_pred = model.predict(X_test)
score = model.score(X_train, y_train)
print('Training Score:', score)
score = model.score(X_test, y_test)
print('Testing Score:', score)
output = pd.DataFrame({'Predicted':Y_pred}) # Heart-Disease yes or no? 1/0
print(output.head())
people = output.loc[output.Predicted == 1]["Predicted"]
rate_people = 0
if len(people) > 0 :
    rate_people = len(people)/len(output)
print("% of people predicted with heart-disease:", rate_people)
score_knc = score
out_knc = output
from sklearn.metrics import classification_report
print(classification_report(y_test,Y_pred))
print("Accuracy:",metrics.accuracy_score(y_test, Y_pred)) 

'''
#%% accuracy sort
results=pd.DataFrame(columns=['accuracy'])
results.loc['Logistic Regression']=[a1]
results.loc['Decision Tree Classifier']=[a2]
results.loc['Random Forest Classifier']=[a3]
results.loc['k nearest neighours']=[a4]
results.loc['naive bayes']=[a5]
results.loc['(svm)linear']=[a6]
results.loc['(svm)poly']=[a7]
results.loc['(svm)rbf']=[a8]
results.loc['linear discriminante']=[a9]
results.loc['adaboost']=[a10]
results.loc['xgboost']=[a11]
print('accuracy list')
b=results.sort_values('accuracy',ascending=False)
print(b)

#%%

from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
log_reg_grid = {"C": np.logspace(-4, 4, 20),
                "solver": ["liblinear"]}

#%%
np.random.seed(42)

# Setup random hyperparameter search for LogisticRegression
rs_log_reg = RandomizedSearchCV(LogisticRegression(),
                                param_distributions=log_reg_grid,
                                cv=5,
                                n_iter=20,
                                verbose=True)

# Fit random hyperparameter search model for LogisticRegression
rs_log_reg.fit(X_train, y_train)
#%%
score = rs_log_reg.score(X_test,y_test)
print(score)
score1 = rs_log_reg.score(X_train,y_train)
print(score1)

#%%
log_reg_grid = {'C': np.logspace(-4,4,30),
               "solver":["liblinear"]}

#setup  the gird cv
gs_log_reg = GridSearchCV(LogisticRegression(),
                          param_grid=log_reg_grid,
                          cv=5,
                          verbose=True)

#fit grid search cv
gs_log_reg.fit(X_train,y_train)
score2= gs_log_reg.score(X_test,y_test)
print(score2)
score3= gs_log_reg.score(X_train,y_train)
print(score3)

#%%
y_preds = gs_log_reg.predict(X_test)
print(y_preds)

print(rs_log_reg.best_params_)
#%%
import pickle

# Save trained model to file
pickle.dump(rs_log_reg, open("heart.pkl", "wb"))
loaded_model = pickle.load(open("heart.pkl", "rb"))
loaded_model.predict(X_test)
loaded_model.score(X_test,y_test)



#%% accuracy graph
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
classifiers = ['Logistic', 'DT','RF','KNN','naive ','ld','poly','rbf','disc','ada','xgb']
accuracies = [a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11]
ax.bar(classifiers,accuracies,align='center', width=0.4)
plt.ylim()
plt.show()

#%% accuracy graph
class_name = ('Logistic', 'DT','RF','KNN','naive ','ld','poly','rbf','disc','ada','xgb')
class_score = (a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11)
y_pos= np.arange(len(class_score))
colors = ("red","gray","purple","green","orange","blue")
plt.figure(figsize=(20,12))
plt.bar(y_pos,class_score,color=colors)
plt.xticks(y_pos,class_name,fontsize=20)
plt.yticks(np.arange(0.00, 1.05, step=0.05))
plt.ylabel('Accuracy')
plt.grid()
plt.title(" Accuracy Comparision of the Classes",fontsize=15)
plt.show()

#%% 
#some visualization
plt.figure(figsize=(10, 6))

# Scatter with positive example 
plt.scatter(df.age[df.target == 1], df.thalach[df.target ==1],
           c = "green")

# Scatter with negative example 

plt.scatter(df.age[df.target == 0], df.thalach[df.target==0],
           c = "lightblue")

# Add some helpful info
plt.title("Heart Disease in function of Age and Max Heart Rate")
plt.xlabel("Age")
plt.ylabel("Max Heart Rate")
plt.legend(["Disease", "No Disease"])
#%%
#correlation matrix graph
corr_matrix = df.corr()

fig, ax = plt.subplots(figsize = (15, 10))

ax = sns.heatmap(corr_matrix, 
                annot=True,
                linewidths= 0.5,
                fmt="0.2f",
                cmap="coolwarm");
#%% disease vs gender
pd.crosstab(df.target, df.sex).plot(kind = "bar",figsize=(10, 6), color = ["blue", "lightblue"] )
plt.title("Heart Disease Frequency for sex")
plt.xlabel("0 = No Disease, 1 = Disease")
plt.ylabel(" Amount")

plt.legend(["Female", "Male"])

plt.xticks(rotation = 0);
#%% percentage of target
disease = len(df[df['target'] == 1])
no_disease = len(df[df['target']== 0])
import matplotlib.pyplot as plt
y = ('Heart Disease', 'healthy')
y_pos = np.arange(len(y))
x = (disease, no_disease)
labels = 'Heart Disease', 'No Disease'
sizes = [disease, no_disease]
fig1, ax1 = plt.subplots()
ax1.pie(sizes,  labels=labels,autopct='%1.1f%%', startangle=90) 
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title('Percentage of target', size=16)
plt.show() # Pie chart, where the slices will be ordered and plotted counter-clockwise:
