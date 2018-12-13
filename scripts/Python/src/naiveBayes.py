import matplotlib.pyplot as  plt
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import sklearn.model_selection as cva
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.metrics import fbeta_score, make_scorer





def read_csv(url):
    # Load digits dataset from scikit
    data = pd.read_csv(url, sep=";")
    print(data.head())

    # HEM DE CONVERTIR LES CATEGORIQUES a numeriques!!

    # Separate data from labels
    x = data.drop(['y'], axis=1)  # Data
    y = data['y']  # Target
    x = pd.get_dummies(x)
    y = pd.get_dummies(y)['yes']
    print(x)
    return data, x, y

def filterp(th,ProbClass1):
    y=np.zeros(ProbClass1.shape[0])
    for i,v in enumerate(ProbClass1):
        if ProbClass1[i]>th:
            y[i]=1
    return y


def main():
    # Load digits dataset from scikit
    url_learn = "../../../data/learning/BankCleanLearn.csv"
    #url_test = "../../../data/test/BankCleanTest.csv"

    (dades, X, y) = read_csv(url_learn)

    #30% of data will be used for testing (.3)
    (X_train, X_test, y_train, y_test) = cva.train_test_split(X, y, test_size=.3, random_state=1)

    # Create a Naive Bayes classifier object
    clf = GaussianNB()
    lth=[]

    # We do a 10 fold crossvalidation with 10 iterations
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    for train_index, test_index in kf.split(X, y):
        X_train2, X_test2 = X.iloc[train_index], X.iloc[test_index]
        y_train2, y_test2 = y[train_index], y[test_index] # .iloc??

        # Train with the training data of the iteration
        clf.fit(X_train2, y_train2)
        # Obtaining probablity predictions for test data of the iterarion
        probs = clf.predict_proba(X_test2)
        # Collect probabilities of belonging to class 1
        ProbClass1 = probs[:,1]
        # Sort probabilities and generate pairs (threshold, f1-for-that-threshold)
        res = np.array([[th,f1_score(y_test2,filterp(th,ProbClass1),pos_label=1)] for th in np.sort(ProbClass1)])

        # Uncomment the following lines if you want to plot at each iteration how f1-score evolves increasing the threshold
        #plt.plot(res[:,0],res[:,1])
        #plt.show()

        # Find the threshold that has maximum value of f1-score
        maxF = np.max(res[:,1])
        optimal_th = res[res[:,1]==maxF,0]

        # Store the optimal threshold found for the current iteration
        lth.append(optimal_th)

    print("\nOptimal thresholds for every iteration")
    for t in lth:
        print(t[0])

    #There are very large (close to 1) thresholds and very low (close to 0)
    thdef = np.mean(lth)
    print("Selected threshold in 10-fold cross validation:", thdef)
    print()

    # Train a classifier with the whole training data
    clf.fit(X_train, y_train)
    # Obtain probabilities for data on test set
    probs = clf.predict_proba(X_test)
    # Generate predictions using probabilities and threshold found on 10 folds cross-validation
    pred = filterp(thdef,probs[:,1])
    # Print results with this prediction vector
    print("\n confusion matrix on test set:\n",confusion_matrix(y_test, pred))
    print("\n classification report on test set:\n",classification_report(y_test, pred))


main()
