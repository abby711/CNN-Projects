'''
   1. Number of times pregnant
   2. Plasma glucose concentration a 2 hours in an oral glucose tolerance test
   3. Diastolic blood pressure (mm Hg)
   4. Triceps skin fold thickness (mm)
   5. 2-Hour serum insulin (mu U/ml)
   6. Body mass index (weight in kg/(height in m)^2)
   7. Diabetes pedigree function
   8. Age (years)
   9. Class variable (0 or 1)-0 means no diabetes 1 means yes
'''


from numpy import loadtxt

from keras.models import Sequential
#sequntial= create empty stack of layers where we'll customize each layers

from keras.layers import Dense
#dense==layerformat

from keras.models import model_from_json

dataset = loadtxt('pima-indians-diabetes.csv', delimiter=',')
x=dataset[:,0:8] #full rows in csv
y=dataset[:,8]
'''print(x)
print("xxxxxxxxxxxxxxxx")
print(y)'''

model=Sequential()
model.add(Dense(12,input_dim=8,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(1,activation='sigmoid'))
#model.summary()


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#line that does the training
model.fit(x,y, epochs=200, batch_size=10)
_,accuracy=model.evaluate(x,y)
print('ACCURACY= %.2f' % (accuracy*100))

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model.h5")
print("Saved model to disk")
