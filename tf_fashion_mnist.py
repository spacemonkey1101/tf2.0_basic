import tensorflow as tf
from tensorflow import keras #keras is a high level api ...that does things for us
import numpy as np
import matplotlib.pyplot as plt

#load in the data set from keras
data = keras.datasets.fashion_mnist

#splitting data into train and test
(train_images,train_lables),(test_images,test_labels) = data.load_data() #keras makes it easy for us by dividing data accordingly

#labels
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#observe your data using matplotlib
#plt.imshow(train_images[7])
#plt.show()

#normlize
train_images = train_images/255.0
test_images = test_images/255.0

#making the model
model = keras.Sequential([ #keras.sequestial means defining a sequence of layers
	keras.layers.Flatten(input_shape=(28,28)), #makes a 1-D input later from  28x28 layers
	keras.layers.Dense(128,activation="relu"),
	keras.layers.Dense(10,activation="softmax")
	])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy" , metrics=["accuracy"])

#train model
model.fit(train_images,train_lables, epochs=10) 
#epochs = number of times all 60000 images will be seen by the model in a randomised order 
#increasing the epoch is not always helpful

#see how well the model actually performs
test_loss , test_acc = model.evaluate(test_images,test_labels)
print("Tested acc ",test_acc )

#prediction using the model
prediction = model.predict(test_images)#give the input in a list

#show the images on screen
for i in range(5): 
	plt.grid(False)
	plt.imshow(test_images[i])
	plt.xlabel("Actual: " + class_names[test_labels[i]] )
	#index of neuron with the largest probability
	plt.title("prediction " + class_names[np.argmax(prediction[i])])
	plt.show()