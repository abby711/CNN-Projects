from numpy import loadtxt

from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json

dataset = loadtxt('pima-indians-diabetes.csv', delimiter=',')
x=dataset[:,0:8] #full rows in csv
y=dataset[:,8]
json_file = open('model.json','r')
loaded_model_json=json_file.read()
json_file.close()
model=model_from_json(loaded_model_json)
model.load_weights("model.h5")
print("loaded model from disk")
prediction=model.predict_classes(x)
for i in range(0,11): # from 1'th row to 10'th row
    print("%s => %d (expected %d)" % (x[i].tolist(), prediction[i], y[i]))
