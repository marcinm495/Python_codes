
import numpy as np
import pandas as pd
import math
from sklearn import svm
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler



#data:
data_file = pd.read_csv("train.csv")

#preparation of feature columns:

data_file['age_num']=data_file['Age'].map(lambda x: data_file['Age'].median() if math.isnan(x) else x)
data_file['sex_num']=data_file['Sex'].map(lambda x: 1 if x == 'male' else 0)
data_file['emb_num']=data_file['Embarked'].map(lambda x: 1 if x=="C" else 2 if x=="S" else 3)


#function performing preparation of training and testing arrays with split into train/test specified by test_size_num:
def input_data(test_size_num, *feature_names):
    return train_test_split(data_file[list(feature_names)].values, data_file['Survived'].values, test_size = test_size_num, random_state = 0)

# principal component decomposition of the data:
def data_pca(test_size_num, n_princ_comp):
    xtr, xts, ytr, yts = input_data(test_size_num, 'age_num', 'sex_num', 'emb_num', 'Parch', 'SibSp', 'Fare')   
    from sklearn.decomposition import PCA
    scaler = StandardScaler()
    xtr = scaler.fit_transform(xtr)
    xts = scaler.transform(xts)
    pca = PCA(n_components = n_princ_comp).fit(xtr)
    print(pca.explained_variance_ratio_)  
    xtr_pca = pca.transform(xtr)
    xts_pca = pca.transform(xts)
    return xtr_pca, xts_pca, ytr, yts
    
#preparing train/test arrays:
features_train, features_test, labels_train, labels_test = data_pca(0.35, 6)


#function
def classifier_function(classifier_name):
    clf = classifier_name
    clf.fit(features_train, labels_train)
    pred_outcomes = clf.predict(features_test)
    return accuracy_score(pred_outcomes, labels_test)

def grid_class_function(class_name, params):
    from sklearn.model_selection import GridSearchCV
    pre_clf = class_name
    clf = GridSearchCV(pre_clf, params)
    clf.fit(features_train, labels_train)
    pred_outcomes = clf.predict(features_test)
    return (accuracy_score(pred_outcomes, labels_test), clf.best_params_)

    
print(classifier_function(svm.SVC(kernel = 'rbf')))
print(classifier_function(tree.DecisionTreeClassifier()))
print(classifier_function(RandomForestClassifier(n_estimators=10)))
print(classifier_function(AdaBoostClassifier(n_estimators=100)))

print(grid_class_function(svm.SVC(), {'kernel':('linear', 'rbf'), 'C':[0.1,1, 10, 100]}))
#print(grid_class_function(tree.DecisionTreeClassifier(), {'min_samples_split':[2,4,6,8]}))
#print(grid_class_function(RandomForestClassifier(), {'n_estimators':[5,10,15,20]}))
#print(classifier_function(AdaBoostClassifier(n_estimators=100)))


 


