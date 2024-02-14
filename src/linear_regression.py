# Import Algorithms  
from pandas import read_csv  
from sklearn.model_selection import train_test_split  
from sklearn.linear_model import LinearRegression
import pickle  
  
# load dataset  
names = ["request sent time", 
	"content", 
	"size_in_bytes", 
	"action", 
	"destination", 
	"respond time", 
	"request received time", 
	"CPU", 
	"memory in mb",
    "disk I/O KB/S", 
    "complete time",
    "number of machines involved",
    "backup execution time", 
    "backup complete time", 
    "replication start time", 
    "replication complete time", 
    "data balance start time", 
    "data balance complete time", 
    "database repair start time", 
    "database complete time", 
    "repair failure",
	"db solution"]

dataset = read_csv('database_elements.csv', names=names)

mobile_prop = dataset  
del mobile_prop['db solution']  

X = mobile_prop.values  
Y = dataset['db solution'].values  

# Fit the model on 30%  
test_size = 0.30  
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)  

# prepare models  
model = LinearRegression()  
model.fit(X_train, Y_train)  

# save the model to disk  
filename = 'database_suggestion_model.sav'  
pickle.dump(model, open(filename, 'wb'))  

# load the model from disk  
loaded_model = pickle.load(open(filename, 'rb'))  
result = loaded_model.score(X_test, Y_test)  
print(result)  
