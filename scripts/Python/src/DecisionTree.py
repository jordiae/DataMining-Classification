import pandas as pd
import numpy as np
import sklearn.model_selection as cv
import sklearn
url_learn = "../../../data/learning/BankCleanLearn.csv"
url_test = "../../../data/test/BankCleanTest.csv"
data_learn = pd.read_csv(url_learn,sep=';')
data_learn.head()
data_test = pd.read_csv(url_test,sep=';')
## Separate data from labels
X_learn=data_learn.drop(['y'], axis=1)
y_learn=data_learn['y']

print(X_learn.shape)
X_learn.head()

X_test=data_test.drop(['y'], axis=1)
y_test=data_test['y']
## Transform to numerical dataset
#Xn=pd.get_dummies(X)
#Xn.head()
Xn_learn = pd.get_dummies(X_learn)
Xn_test = pd.get_dummies(X_test)
Xn_learn.head()

## Train Naive Bayes model
"""
from sklearn.naive_bayes import BernoulliNB  # For binari features (f.i. word appears or not in document)
from statsmodels.stats.proportion import proportion_confint

clf = BernoulliNB()
pred = clf.fit(X_train, y_train).predict(X_test)
print(sklearn.metrics.confusion_matrix(y_test, pred))
print()
print("Accuracy:", sklearn.metrics.accuracy_score(y_test, pred))
print()
print(sklearn.metrics.classification_report(y_test, pred))
epsilon = sklearn.metrics.accuracy_score(y_test, pred)
proportion_confint(count=epsilon*X_test.shape[0], nobs=X_test.shape[0], alpha=0.05, method='binom_test')
scores = sklearn.model_selection.cross_val_score(clf, Xn, y, cv=10)
print(scores)
print(np.mean(scores))
"""

from statsmodels.stats.proportion import proportion_confint
from sklearn import tree
from sklearn.externals.six import StringIO
import pydot
from IPython.display import Image
#import cmdline

#from imblearn.under_sampling import RandomUnderSampler,CondensedNearestNeighbour,InstanceHardnessThreshold
#from imblearn.over_sampling import RandomOverSampler

Xx, yy = Xn_learn, y_learn
#rus = RandomUnderSampler(random_state=42)
#rus = CondensedNearestNeighbour(random_state=42)
#rus = RandomOverSampler(random_state=42)
#Xx, yy = rus.fit_resample(Xx, yy)
clf = tree.DecisionTreeClassifier(criterion='entropy',class_weight='balanced')
pred = clf.fit(Xx, yy).predict(Xn_test)
print(sklearn.metrics.confusion_matrix(y_test, pred))
print()
print("Accuracy on test set:", sklearn.metrics.accuracy_score(y_test, pred))
print()
print(sklearn.metrics.classification_report(y_test, pred))
epsilon = sklearn.metrics.accuracy_score(y_test, pred)
print("Confidence interval: ",proportion_confint(count=epsilon*X_test.shape[0], nobs=X_test.shape[0], alpha=0.05, method='binom_test'))

#scores = sklearn.model_selection.cross_val_score(clf, Xn, y, cv=10)
#print("Accuracy on 10 fold cross-validation:", scores)
#print(np.mean(scores))

## Print tree
dot_data = StringIO()
tree.export_graphviz(clf, out_file=dot_data,
                     filled=True, rounded=True,
                     feature_names=list(Xn_learn.columns.values),
                     special_characters=True)
graph = pydot.graph_from_dot_data(dot_data.getvalue())
Image(graph[0].create_png())

from sklearn.metrics import f1_score
from sklearn.metrics import fbeta_score, make_scorer
f_scorer = make_scorer(f1_score, pos_label='yes')
from sklearn.model_selection import GridSearchCV
params = {'min_impurity_split': list(np.linspace(0,1,21)),'min_samples_split':list(range(2,102,11))}
clf = GridSearchCV(tree.DecisionTreeClassifier(criterion='entropy',class_weight='balanced'), param_grid=params,cv=10,n_jobs=-1,scoring=f_scorer)  # If cv is integer, by default is Stratifyed
clf.fit(Xn_learn, y_learn)
print("Best Params=",clf.best_params_, "F1 score=", clf.best_score_)

from sklearn.metrics import confusion_matrix
clf=tree.DecisionTreeClassifier(criterion='entropy', min_samples_split=2,min_impurity_split=0.9,class_weight='balanced')
clf = clf.fit(Xn_learn, y_learn)
pred = clf.predict(Xn_test)

# Obtain accuracy score of learned classifier on test data
print(clf.score(Xn_test, y_test))
print(confusion_matrix(y_test, pred))
print()
print("Accuracy:", sklearn.metrics.accuracy_score(y_test, pred))
print()
print(sklearn.metrics.classification_report(y_test, pred))
epsilon = sklearn.metrics.accuracy_score(y_test, pred)
proportion_confint(count=epsilon*X_test.shape[0], nobs=X_test.shape[0], alpha=0.05, method='binom_test')
