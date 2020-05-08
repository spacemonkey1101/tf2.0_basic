import tensorflow as tf
from tensorflow import keras
import numpy as np

data = keras.datasets.imdb

(train_data , train_labels) , (test_data , test_labels) =data.load_data(num_words = 88000) #words with frequency 10000

#observe the data
#print(train_data[0])
#words are mapped to numbers

#mapping number to words
#this gives us a tuple that has the string and the word
word_index = data.get_word_index()

#get key(k) and value(v) from the tuple and store in the dict
word_index = {k:(v+3) for k,v in word_index.items()}
#the v+3 is there because we need to take care of some special keys used during mapping

#assign my own values for unique keys
#we do this to make all the movie list to the same length
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

#swaping all the values and the keys ||| values(int) ---> keys(words)
#get from tenorflow pager
reverse_word_index = dict([(value,key) for (key,value) in word_index.items()]) 

#compare length of 2 reviews ---->they are of diffenrent length ----> so we have to work with padding --->to make them the same length
#print(len(test_data[0]) , len(test_data[1]))

#pick an arbitrary length say 250 , thats the max num of words all reviews are going to be
#if a review has more words,we will get rid of the excess
#if a review has less words add <PAD>
#this is the preprocessing of data done by keras 
train_data = keras.preprocessing.sequence.pad_sequences(train_data , value=word_index["<PAD>"] , padding="post",maxlen=250)
test_data = keras.preprocessing.sequence.pad_sequences(test_data , value=word_index["<PAD>"] , padding="post",maxlen=250)

#check length after preprocessing
#print(len(test_data[0]) , len(test_data[1]))

#get from tenorflow page
def decode_review(text):
	return " ".join([reverse_word_index.get(i,"?") for i in text])#we try to get index i but if we cant find a value we put ?
	#return human readable word
#print(decode_review(test_data[0]))


#model definition
model = keras.Sequential([	
	keras.layers.Embedding(88000,16),				#similar words are grouped # we create 10000 word vectors #16 is the number of dimensiomn for our word vector
	keras.layers.GlobalAveragePooling1D(),			#scaling down the dimension #shrinks dimensions down
	keras.layers.Dense(16,activation="relu"),		
	keras.layers.Dense(1,activation="sigmoid")     # 1 output neuron of our model output: [0,1]
	])

model.summary()

model.compile(optimizer="adam" ,loss="binary_crossentropy",metrics=["accuracy"])

#validation data
x_val = train_data[:10000]
#training data
x_train = train_data[10000:]

#labels
y_val = train_labels[:10000]
y_train = train_labels[10000:]

model.fit(x_train , y_train , epochs=40 , batch_size=512 , validation_data=(x_val,y_val) , verbose =1)
#batch size = how many reviews will be loaded in one cycle

results = model.evaluate(test_data , test_labels)
print(results)

#saving our model so that we dont have to retrian the model every time we use it
model.save("imdb_saved_model.h5")