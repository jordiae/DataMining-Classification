import pandas as pd
import sklearn.neighbors as nb  # Per fer servir el knn
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import GridSearchCV
from imblearn.under_sampling import RandomUnderSampler  # doctest: +NORMALIZE_WHITESPACE
from imblearn.over_sampling import RandomOverSampler  # doctest: +NORMALIZE_WHITESPACE
import matplotlib.pyplot as plt  # per imprimir plots
import numpy as np
import utils
import sklearn.model_selection as cva  # Pel Cross-validation
from sklearn.metrics import f1_score
from sklearn.metrics import fbeta_score, make_scorer

from sklearn.model_selection import StratifiedKFold


def filterp(th, ProbClass1):
    """ Given a treshold "th" and a set of probabilies of belonging to class 1 "ProbClass1", return predictions """
    y = ProbClass1.shape[0] * ['no']  # np.zeros(ProbClass1.shape[0])
    for i, v in enumerate(ProbClass1):
        if ProbClass1[i] > th:
            y[i] = 'yes'
    return y


def read_csv(url):
    # Load digits dataset from scikit
    data = pd.read_csv(url, sep=";")
    print(data.head())

    # HEM DE CONVERTIR LES CATEGORIQUES a numeriques!!

    # Separate data from labels
    x = data.drop(['y'], axis=1)  # Data
    y = data['y']  # Target
    # Print range of values and dimensions of data
    x = pd.get_dummies(x)
    # print(x.head())
    # print(y.head())
    print(x.shape)
    print(y.shape)
    return data, x, y


def main():
    # Load digits dataset from scikit
    url_learn = "../../../data/learning/BankCleanLearn.csv"
    url_test = "../../../data/test/BankCleanTest.csv"
    f_scorer = make_scorer(f1_score, pos_label='yes')

    (dades, X, y) = read_csv(url_learn)

    # rus = RandomOverSampler(sampling_strategy='auto', random_state=42)
    # X, y = rus.fit_resample(X, y)

    # rus = RandomUnderSampler(sampling_strategy='auto', random_state=42)
    # X, y = rus.fit_resample(X, y)

    # Let's do a simple cross-validation: split data into training and test sets (test 30% of data)
    (X_train, X_test, y_train, y_test) = cva.train_test_split(X, y, test_size=.3, random_state=1)

    # Create a kNN classifier object
    knc = nb.KNeighborsClassifier()

    # Train the classifier
    knc.fit(X_train, y_train)

    # Obtain accuracy score of learned classifier on test data
    print("Score Cross-validation: %s\n" % (knc.score(X_test, y_test)))
    # More information with confussion matrix
    y_pred = knc.predict(X_test)
    print("Confusion Matrix: ")
    print(pd.DataFrame(confusion_matrix(y_test, y_pred, labels=['yes', 'no']),
                       index=['true:yes', 'true:no'], columns=['pred:yes', 'pred:no']))
    print("\nReport metrics: ")
    print(metrics.classification_report(y_test, y_pred))

    print("StratifiedKfold K-cross-Validation")
    cv = StratifiedKFold(n_splits=10, random_state=1)

    cv_scores = cross_val_score(nb.KNeighborsClassifier(), X=X, y=y, cv=cv, scoring=f_scorer, n_jobs=-1)
    print(np.mean(cv_scores))

    # MILLOREM EL KNN
    # Normalization in KNN
    print("Millorem knn, normalització: ")
    # # One way is to standarize all data mean 0, std 1
    scaler = preprocessing.StandardScaler().fit(X)
    X2 = scaler.transform(X)
    cv_scores = cross_val_score(nb.KNeighborsClassifier(), X=X2, y=y, cv=cv, scoring=f_scorer, n_jobs=-1)
    print("new f_score: %s\n" % (np.mean(cv_scores)))
    # Irrelevant columns
    # print("Effect of irrelevant columns:\n")
    # plt.subplots(figsize=(10, 10))
    # plt.subplots_adjust(hspace=0.27, wspace=0.5)
    # contador = 0
    # for i in range(0,21):
    #     plt.subplot(5, 4, 0 + contador + 1)
    #     dades[dades['y'] == 'no'][i].plot.hist(bins=10, density=True)
    #     dades[dades['y'] == 'yes'][i].plot.hist(bins=10, density=True)
    #     contador += 1
    # plt.show()
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
        cv_scores = cross_val_score(nb.KNeighborsClassifier(), X=X_new, y=y, cv=cv, scoring=f_scorer, n_jobs=-1)
        if len(original) > 0 and np.mean(cv_scores) >= max(original):
            k_max = contador + 1
            X_new_best = X_new
        original.append(np.mean(cv_scores))
        contador += 1

    plt.xticks(np.arange(0, 20, step=1))
    plt.plot(range(1, 21), original)
    plt.show()
    print("K best feature:%s f-score:%s\n" % (k_max, max(original)))
    # K best features es 19, sembla que hi ha una feature que esta afegint error
    X_new = X_new_best
    # # Buscarem els millors parametres PLOT:
    # # OJO! d'aqui fins al final TRIGA MOOOOLT i utilitza tots els nuclis
    colors = ['b','r','g','y','m','c']
    contador = 0
    for pi in range(1,4):
        lr = []
        for ki in range(1, 30, 2):
            cv_scores = cross_val_score(nb.KNeighborsClassifier(n_neighbors=ki, p=pi), X=X_new, y=y, cv=10, scoring=f_scorer, n_jobs=-1)
            lr.append(np.mean(cv_scores))
        plt.plot(range(1, 30, 2), lr, colors[contador], label="No weighting p=%s" % pi)
        contador += 1
        lr = []
        for ki in range(1, 30, 2):
            cv_scores = cross_val_score(nb.KNeighborsClassifier(n_neighbors=ki, weights='distance', p=pi), X=X_new, y=y, cv=10,scoring=f_scorer, n_jobs=-1)
            lr.append(np.mean(cv_scores))
        plt.plot(range(1, 30, 2), lr, colors[contador], label="Weighting p=%s" % pi)
        contador += 1

    plt.xlabel('k')
    plt.ylabel('f1_score')
    plt.legend(loc='upper right')
    plt.grid()
    plt.tight_layout()
    plt.show()
    # Results of the plot: Default parameters of Knn classifier p=2 weight uniform
    # Busquem els millors parametres amb un GridSearch
    print("Searching best parameters for KNN\n")
    params = {'n_neighbors': list(range(1, 30, 2)), 'weights': ('distance', 'uniform'), 'p': list(range(1,4))}
    knc = nb.KNeighborsClassifier()
    clf = GridSearchCV(knc, param_grid=params, cv=cv, n_jobs=-1, scoring=f_scorer)  # If cv is integer, by default is Stratifyed
    clf.fit(X_new, y)
    print("Best Params=", clf.best_params_, "f-score=", clf.best_score_)

    
    #knc_test = nb.KNeighborsClassifier(n_neighbors=29,
    #                                   weights='uniform', p=3)
    (dades_test, X_testing, y_testing) = read_csv(url_test)
    scaler = preprocessing.StandardScaler().fit(X_testing)
    X_testing_norm = scaler.transform(X_testing)
    #knc_test.fit(X2, y)  # X2 conte dades originals normalitzades
    #print("Results without: ", knc_test.score(X_testing_norm, y_testing))
    # More information with confussion matrix
    #y_testing_pred = knc_test.predict(X_testing_norm)
    #print("Confusion Matrix: ")
    #print(pd.DataFrame(confusion_matrix(y_testing, y_testing_pred, labels=['yes', 'no']),
     #                  index=['true:yes', 'true:no'], columns=['pred:yes', 'pred:no']))
    #print("\nReport metrics: ")
    #print(metrics.classification_report(y_testing, y_testing_pred))

   # clf = GaussianNB()

    clf = nb.KNeighborsClassifier(n_neighbors=29,
                                  weights='uniform', p=3)
    """""
    lth = []

    # We do a 10 fold crossvalidation with 10 iterations
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    # Xx_array = Xx.values
    # yy_array = yy.values
    # yn_learn = pd.get_dummies(yy)
    # yn_test = pd.get_dummies(yy)
    for train_index, test_index in kf.split(X2, y):
        X_train2, X_test2 = X2[train_index], X2[test_index]
        y_train2, y_test2 = y[train_index], y[test_index]

        # Train with the training data of the iteration
        clf.fit(X_train2, y_train2)
        # Obtaining probablity predictions for test data of the iterarion
        probs = clf.predict_proba(X_test2)
        # Collect probabilities of belonging to class 1
        ProbClass1 = probs[:, 1]
        # Sort probabilities and generate pairs (threshold, f1-for-that-threshold)
        res = np.array(
            [[th, f1_score(y_test2, filterp(th, ProbClass1), pos_label='yes')] for th in np.sort(ProbClass1)])

        # Uncomment the following lines if you want to plot at each iteration how f1-score evolves increasing the threshold
        # plt.plot(res[:,0],res[:,1])
        # plt.show()

        # Find the threshold that has maximum value of f1-score
        maxF = np.max(res[:, 1])
        optimal_th = res[res[:, 1] == maxF, 0]

        # Store the optimal threshold found for the current iteration
        lth.append(optimal_th)

    # Compute the average threshold for all 10 iterations
    print(lth)
    # thdef = np.mean(np.ndarray.flatten(lth))
    # thdef = np.mean([l.tolist() for l in lth])
    # thdef = [item for sublist in thdef for item in sublist]
    # thdef = np.mean(lth)
    thdef = np.mean(np.concatenate(lth, axis=0))
    print("Selected threshold in 10-fold cross validation:", thdef)
    """
    thdef = 0.2043734230445753
    from sklearn.metrics import classification_report
    print("Training...")
    clf.fit(X2, y)
    # Obtain probabilities for data on test set
    print("proba...")
    probs = clf.predict_proba(X_testing_norm)
    # Generate predictions using probabilities and threshold found on 10 folds cross-validation
    print("filter...")
    pred = filterp(thdef, probs[:, 1])
    print("report....")
    # Print results with this prediction vector
    print(classification_report(y_testing, pred))
    print("Confusion Matrix: ")
    print(pd.DataFrame(confusion_matrix(y_testing, pred, labels=['yes', 'no']),
                       index=['true:yes', 'true:no'], columns=['pred:yes', 'pred:no']))


main()
