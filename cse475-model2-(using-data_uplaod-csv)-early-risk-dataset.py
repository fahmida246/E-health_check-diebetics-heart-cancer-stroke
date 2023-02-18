
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
from sklearn.preprocessing import LabelEncoder
#%%
df = pd.read_csv('diabetes_data_upload.csv')
print(df)
print(df.head())
#%%
#shape and column targets value count
print(df.shape)
#print(df.class.value_counts() )
#%%
# is there any missing value?
print(df.isna().sum())

#%%
columns = ['Age','Gender','Polyuria','Polydipsia','sudden weight loss','weakness','Polyphagia','Genital thrush',	'visual blurring','Itching','Irritability',	'delayed healing',	'partial paresis'	,'muscle stiffness'	,'Alopecia'	,'Obesity'	,'class']
le = LabelEncoder()
df[columns] = df[columns].apply(le.fit_transform)
print(df.head())
#%%
X = df.drop('class',1)
y = df['class']
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
model1 = LogisticRegression(max_iter=1000)
model1.fit(X_train, y_train)
Y_pred = model1.predict(X_test)

score1 = model1.score(X_train, y_train)
print('Training Score:', score1)

score = model1.score(X_test, y_test)
print('Testing Score:', score)

output = pd.DataFrame({'Predicted':Y_pred}) 
print('print first 5 predicted values:')
print(output.head())


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
model2 = DecisionTreeClassifier(max_depth=5)
model2.fit(X_train, y_train)  
Y_pred = model2.predict(X_test)

score1 = model2.score(X_train, y_train)
print('Training Score:', score1)

score = model2.score(X_test, y_test)
print('Testing Score:', score)

output = pd.DataFrame({'Predicted':Y_pred}) 
print(output.head())


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


#%% 2
#************random forest tree classifier *************************
print('RandomForestClassifier')
from sklearn.ensemble import RandomForestClassifier
model3 = RandomForestClassifier(n_estimators=100) 
model3.fit(X_train, y_train)
Y_pred = model3.predict(X_test)
score1 = model3.score(X_train, y_train)

print('Training Score:', score1)
score = model3.score(X_test, y_test)

print('Testing Score:', score)
output = pd.DataFrame({'Predicted':Y_pred})
print(output.head())


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

#%% 4
#************kneighbours classifier *************************
print('KNeighborsClassifier')
from sklearn.neighbors import KNeighborsClassifier
model4 = KNeighborsClassifier()
model4.fit(X_train, y_train)
Y_pred = model4.predict(X_test)
score1 = model4.score(X_train, y_train)

print('Training Score:', score1)
score = model4.score(X_test, y_test)

print('Testing Score:', score)
output = pd.DataFrame({'Predicted':Y_pred})
print(output.head())

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

#%%  5
#*********************naive bayes*************
print('naive bayes')
from sklearn.naive_bayes import GaussianNB
model5 = GaussianNB()
model5.fit(X_train, y_train)
Y_pred = model5.predict(X_test)
score1 = model5.score(X_train, y_train)

print('Training Score:', score1)
score = model5.score(X_test, y_test)

print('Testing Score:', score)
output = pd.DataFrame({'Predicted':Y_pred}) 

print(output.head())


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

#%%
#support vector classifier using 3 different tecniques linear,rbf and poly

from sklearn.svm import SVC
print('using linear kernel')
model6 = SVC( kernel='linear')
model6.fit(X_train, y_train)
Y_pred = model6.predict(X_test)
score = model6.score(X_train, y_train)

print('Training Score:', score)
score = model6.score(X_test, y_test)

print('Testing Score:', score)
output = pd.DataFrame({'Predicted':Y_pred}) 
print(output.head())


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
model7 = SVC( kernel='poly')
model7.fit(X_train, y_train)
Y_pred = model7.predict(X_test)
score = model7.score(X_train, y_train)

print('Training Score:', score)
score = model7.score(X_test, y_test)

print('Testing Score:', score)
output = pd.DataFrame({'Predicted':Y_pred}) 
print(output.head())


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
model8 = SVC( kernel='rbf')
model8.fit(X_train, y_train)
Y_pred = model8.predict(X_test)

score = model8.score(X_train, y_train)
print('Training Score:', score)

score = model8.score(X_test, y_test)
print('Testing Score:', score)

output = pd.DataFrame({'Predicted':Y_pred}) 
print(output.head())


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
model9 = LinearDiscriminantAnalysis()
model9.fit(X_train, y_train)
Y_pred = model9.predict(X_test)
score = model9.score(X_train, y_train)

print('Training Score:', score)
score = model9.score(X_test, y_test)

print('Testing Score:', score)
output = pd.DataFrame({'Predicted':Y_pred})

print(output.head())

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
model10 = AdaBoostClassifier()
model10.fit(X_train, y_train)
Y_pred = model10.predict(X_test)
score = model10.score(X_train, y_train)

print('Training Score:', score)
score = model10.score(X_test, y_test)

print('Testing Score:', score)
output = pd.DataFrame({'Predicted':Y_pred}) 
print(output.head())

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

#%%
#using xgboost
from xgboost import XGBClassifier
model11 = XGBClassifier(n_jobs=-1,n_estimators=12, random_state=4)
model11.fit(X_train, y_train)
Y_pred = model11.predict(X_test)
score = model11.score(X_train, y_train)

print('Training Score:', score)
score = model11.score(X_test, y_test)

print('Testing Score:', score)
output = pd.DataFrame({'Predicted':Y_pred}) # Heart-Disease yes or no? 1/0
print(output.head())


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
#%%
results=pd.DataFrame(columns=['accuracy'])
results.loc['Logistic Regression']=[a1]
results.loc['Decision Tree Classifier']=[a2]
results.loc['Random Forest Classifier']=[a3]
results.loc['k nearest neighours']=[a4]
results.loc['naive bayes']=[a5]
results.loc['(svm)linear']=[a6]
results.loc['(svm)poly']=[a7]
results.loc['(svm)rbf']=[a8]
results.loc['Linear discriminant analysis']=[a9]
results.loc['adaboost']=[a10]
results.loc['xgboost']=[a11]
print('accuracy list')
b=results.sort_values('accuracy',ascending=False)
print(b)

#%%
import pickle
filename = 'die.pkl'
pickle.dump(model1, open(filename, 'wb'))



#%%
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
classifiers = ['Logistic', 'DT','RF','KNN','naive ','ld','poly','rbf','disc','ada','xgb']
accuracies = [a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11]
ax.bar(classifiers,accuracies,align='center', width=0.4)
plt.ylim()
plt.show()

#%%
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
#correlation matrix
corr_matrix = df.corr()

fig, ax = plt.subplots(figsize = (15, 10))

ax = sns.heatmap(corr_matrix, 
                annot=True,
                linewidths= 0.5,
                fmt="0.2f",
                cmap="coolwarm");

#%%
dia = len(df[df['class'] == 1])
hea = len(df[df['class']== 0])
import matplotlib.pyplot as plt
y = ('diabetes', 'healthy')
y_pos = np.arange(len(y))
x = (dia, hea)
labels = 'diabetes', 'healthy'
sizes = [dia, hea]
fig1, ax1 = plt.subplots()
ax1.pie(sizes,  labels=labels,autopct='%1.1f%%', startangle=90) 
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title('Percentage of target', size=16)
plt.show() # Pie chart, where the slices will be ordered and plotted counter-clockwise:
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
pickle.dump(rs_log_reg, open("Diabetes3.pkl", "wb"))
loaded_model = pickle.load(open("Diabetes3.pkl", "rb"))
loaded_model.predict(X_test)
loaded_model.score(X_test,y_test)
#%% if you want to take input and predict it uncomment from 614 to 641

#age = input()
#gender = input()
#polyuria = input()
#polydipsia = input()
#weightloss = input()
#weakness = input()
#polyphagia = input()
#genitalthrush=input()
#visualblurring= input()
#itching= input()
#irritability= input()
#delayedhealing= input()
#partialparesis= input()
#musclestiffness= input()
#alopecia= input()
#obesity= input()



#row_df = pd.DataFrame([pd.Series([age,gender,polyuria,polydipsia,weightloss,weakness,polyphagia,genitalthrush
#                                  ,visualblurring,itching,irritability,delayedhealing,partialparesis,
#                                  musclestiffness,alopecia,obesity])])
#print(row_df)
#prob = loaded_model.predict_proba(row_df)[0][1]
#print(f"The probability of you having Diabetes is {prob}")

#print(loaded_model.predict(row_df)[0])
#print(model1.predict(row_df)[0])









