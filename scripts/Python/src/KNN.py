import pandas as pd
import sklearn.model_selection as cv  # Pel Cross-validation
import sklearn.neighbors as nb  # Per fer servir el knn
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt  # per imprimir plots
import numpy as np
import utils

# Load digits dataset from scikit
url = "../../../data/clean/BankClean.csv"
dades = pd.read_csv(url, header=0, sep=";")
print(dades.head())

# HEM DE CONVERTIR LES CATEGORIQUES a numeriques!!
labelEncoders = utils.convert_cat_num(dades)

# Separate data from labels
X = dades.drop(['y'], axis=1).values  # Data
y = dades['y'].values  # Target
# Print range of values and dimensions of data
print(X.shape)
print(y.shape)

# Let's do a simple cross-validation: split data into training and test sets (test 30% of data)
(X_train, X_test, y_train, y_test) = cv.train_test_split(X, y, test_size=.3, random_state=1)

# Create a kNN classifier object
knc = nb.KNeighborsClassifier()

# Train the classifier
knc.fit(X_train, y_train)

# Obtain accuracy score of learned classifier on test data
print("Score Cross-validation: %s\n" % (knc.score(X_test, y_test)))
# More information with confussion matrix
y_pred = knc.predict(X_test)
print("Confusion Matrix: ")
print(pd.DataFrame(confusion_matrix(y_test, y_pred, labels=[1, 0]),
                   index=['true:yes', 'true:no'], columns=['pred:yes', 'pred:no']))
print("\nReport metrics: ")
print(metrics.classification_report(y_test, y_pred))

print("StratifiedKfold K-cross-Validation")
cv = StratifiedKFold(n_splits=10, random_state=1)

cv_scores = cross_val_score(nb.KNeighborsClassifier(), X=X, y=y, cv=cv, scoring='accuracy', n_jobs=-1)
print(np.mean(cv_scores))

# MILLOREM EL KNN
# Normalization in KNN
print("Millorem knn, normalització: ")
# One way is to standarize all data mean 0, std 1
scaler = preprocessing.StandardScaler().fit(X)
X2 = scaler.transform(X)
cv_scores = cross_val_score(nb.KNeighborsClassifier(), X=X2, y=y, cv=cv, scoring='accuracy', n_jobs=-1)
print("new accuracy: %s\n" % (np.mean(cv_scores)))
# Irrelevant columns
print("Effect of irrelevant columns:\n")
plt.subplots(figsize=(10, 10))
plt.subplots_adjust(hspace=0.27, wspace=0.5)
contador = 0
for i in dades:
    if i == 'y':
        continue
    plt.subplot(5, 4, 0 + contador + 1)
    dades[dades['y'] == 0][i].plot.hist(bins=10)
    dades[dades['y'] == 1][i].plot.hist(bins=10)
    contador += 1
plt.show()
# no he sacado ninguna conclusion xD
# Unfortunately, we don't know before hand the relevant feature.
# Select k best features following a given measure. Fit that on whole data set and return only relevant columns
print("Searching best K features plot...")
original = []
contador = 0
k_max = 0
X_new_best = None
for i in dades:
    if i == 'y':
        continue
    X_new = SelectKBest(mutual_info_classif, k=contador + 1).fit_transform(X2, y)
    cv_scores = cross_val_score(nb.KNeighborsClassifier(), X=X_new, y=y, cv=cv, scoring='accuracy', n_jobs=-1)
    if len(original) > 0 and np.mean(cv_scores) >= max(original):
        k_max = contador + 1
        X_new_best = X_new
    original.append(np.mean(cv_scores))
    contador += 1

plt.xticks(np.arange(0, 20, step=1))
plt.plot(range(1, 21), original)
plt.show()
print("K best feature:%s accuracy:%s\n" % (k_max, max(original)))
# K best features es 19, sembla que hi ha una feature que esta afegint error
X_new = X_new_best
# Buscarem els millors parametres PLOT:
# OJO! d'aqui fins al final TRIGA MOOOOLT i utilitza tots els nuclis
lr = []
for ki in range(1, 30, 2):
    cv_scores = cross_val_score(nb.KNeighborsClassifier(n_neighbors=ki), X=X_new, y=y, cv=10)
    lr.append(np.mean(cv_scores))
plt.plot(range(1, 30, 2), lr, 'b', label='No weighting')

lr = []
for ki in range(1, 30, 2):
    cv_scores = cross_val_score(nb.KNeighborsClassifier(n_neighbors=ki, weights='distance'), X=X_new, y=y, cv=10)
    lr.append(np.mean(cv_scores))
plt.plot(range(1, 30, 2), lr, 'r', label='Weighting')
plt.xlabel('k')
plt.ylabel('Accuracy')
plt.legend(loc='upper right')
plt.grid()
plt.tight_layout()
plt.show()

# Busquem els millors parametres amb un GridSearch
print("Searching best parameters for KNN\n")
params = {'n_neighbors': list(range(1, 30, 2)), 'weights': ('distance', 'uniform')}
knc = nb.KNeighborsClassifier()
clf = GridSearchCV(knc, param_grid=params, cv=cv, n_jobs=-1)  # If cv is integer, by default is Stratifyed
clf.fit(X_new, y)
print("Best Params=", clf.best_params_, "Accuracy=", clf.best_score_)
