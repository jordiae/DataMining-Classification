import pandas as pd
import sklearn.model_selection as cv  # Pel Cross-validation
import sklearn.neighbors as nb  # Per fer servir el knn
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import utils

# Load digits dataset from scikit
url = "../../data/original/bank-additional-full.csv"
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
