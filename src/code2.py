# Compare Algorithms
from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

# load dataset
f = open("conversation_file.txt", "r")
facial_sound = ['conversation text', 'conversation sound']
dataframe = read_csv(f, conversation_responses=f)
array = dataframe.values
X = array[:, 0:30]
Y = array[:, 1]
# prepare models
models = []
models.append(('LR', LogisticRegression()))

# evaluate each model in turn
results = []
names = []
scoring = 'accuracy'
for name, model in models:
    kfold = KFold(n_splits=10, random_state=7)
    cv_results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
