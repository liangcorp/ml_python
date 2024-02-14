# Import Algorithms  
from pandas import read_csv  
from sklearn.model_selection import train_test_split  
from sklearn.linear_model import LogisticRegression  
import pickle  
  
# load dataset  
names = ['id', 'location', 'speed', 'weight', 'battery_level', 'traffic', 'probability']  
dataset = read_csv('EV_Data.csv', names=names)  

mobile_prop = dataset  
del mobile_prop['probability']  
 
X = mobile_prop.values  
Y = dataset['probability'].values  
  
# Fit the model on 30%  
test_size = 0.30  
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)  
 
# prepare models  
model = LogisticRegression(n_estimators = 1000, random_state = 42)  
model.fit(X_train, Y_train)  

# load the model from disk  
loaded_model = pickle.load(open(filename, 'rb'))  
predicted = loaded_model.score(X_test, Y_test)  
print(predicted)  
