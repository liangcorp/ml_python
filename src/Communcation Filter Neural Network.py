# Compare Algorithms
from pandas import read_csv
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix

# load dataset
names = ['database', 'keyspace', 'bucket', 'table', 'cell']

dataframe = read_csv('database_content.csv', names=names)

array = dataframe.values
X = array[:,0:8]
Y = array[:,8]

# prepare models
mlp = MLPClassifier(hidden_layer_sizes=(13,13,13),max_iter=500)
mlp.fit(X_train,y_train)

predictions = mlp.predict(X_test)

print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
