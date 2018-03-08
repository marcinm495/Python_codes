import pandas as pd
import numpy as np
import math
import operator
import itertools
from sklearn import tree, svm, naive_bayes
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

data_file = pd.read_csv("train.csv")

data_file['age_num']=data_file['Age'].map(lambda x: data_file['Age'].median() if math.isnan(x) else x)
data_file['sex_num']=data_file['Sex'].map(lambda x: 1 if x == 'male' else 0)
data_file['emb_num']=data_file['Embarked'].map(lambda x: 1 if x=="C" else 2 if x=="S" else 3)

f_list = ['age_num', 'sex_num', 'emb_num', 'Parch', 'SibSp', 'Fare']

XTR, XTS, LTR, LTS = train_test_split(data_file[f_list], data_file['Survived'].values, test_size = 0.3) 

def class_fun(class_name, features_ind):
   clf = class_name
   if (len(features_ind) == 1):
       Xtr = XTR[[operator.itemgetter(*features_ind)(f_list)]].values
       Xts = XTS[[operator.itemgetter(*features_ind)(f_list)]].values
   else:
       Xtr = XTR[list(operator.itemgetter(*features_ind)(f_list))].values
       Xts = XTS[list(operator.itemgetter(*features_ind)(f_list))].values
   clf.fit(Xtr, LTR)
   pred_outcomes = clf.predict(Xts)
   return accuracy_score(pred_outcomes, LTS)

ind_list = range(0,len(f_list))

best_acc = 0
best_subset = []
num_it=10

for L in range(1, len(ind_list)+1):
  for subset in itertools.combinations(ind_list, L):
    acc_list=[]
    for k in range(1, num_it+1):
        acc_list.append(class_fun(RandomForestClassifier(n_estimators = 10), list(subset)))
    inst_acc = np.mean(acc_list)
    if (inst_acc > best_acc):
        best_acc = inst_acc 
        best_subset = list(subset)
        
print(best_acc)
if (len(best_subset)==1):
    print([operator.itemgetter(*best_subset)(f_list)])        
else:
    print(list(operator.itemgetter(*best_subset)(f_list)))        

