#testing out with some outside data
#imdb The lion King Review

import tensorflow as tf
from tensorflow import keras
import numpy as np

data = keras.datasets.imdb

model = keras.models.load_model("imdb_saved_model.h5")
(train_data , train_labels) , (test_data , test_labels) =data.load_data(num_words = 88000) #words with frequency 10000

print(train_data[0])

#mapping number to words

word_index = data.get_word_index()

word_index = {k:(v+3) for k,v in word_index.items()}
#assign my own values for diffenent things
#we do this to make all the movie list to the same length
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value,key) for (key,value) in word_index.items()]) #swaping all the values and the keys ||| values(int) ---> keys(words)

#compare length of 2 reviews ---->they are of diffenrent length ----> so we have to work with padding --->to make them the same length
#print(len(test_data[0]) , len(test_data[1]))

#pick an arbitrary length say 250 , thats the max num of words all reviews are going to be
train_data = keras.preprocessing.sequence.pad_sequences(train_data , value=word_index["<PAD>"] , padding="post",maxlen=250)
test_data = keras.preprocessing.sequence.pad_sequences(test_data , value=word_index["<PAD>"] , padding="post",maxlen=250)
print(len(test_data[0]) , len(test_data[1]))


def decode_review(text):
	return " ".join([reverse_word_index.get(i,"?") for i in text])#we try to get index i but if we cant find a value we put ?

def review_encode(s):
	encoded = [1]
	#loop through every word
	for word in s:
		if word.lower() in word_index: #if word is in our dict
			encoded.append(word_index[word.lower()])
		else:
			encoded.append(2) #if word is unknown add <UNK> tag

	return encoded

with open("lion_king.txt") as f:
	for line in f.readlines(): 
	#we need to convert the text into an encoded list of numbers
		nline = line.replace("," , "").replace("." , "").replace("(" , "").replace(")" , "").replace(":" , "").replace("\"" , "").strip().split(" ") 
		#replacing every word with . , ( / : 
		encode  = review_encode(nline) #this function returns an encoded list 
		encode = keras.preprocessing.sequence.pad_sequences([encode] , value=word_index["<PAD>"] , padding="post",maxlen=250)
		predict = model.predict(encode)
		print(line)
		print(encode)
		print(predict[0]) #final result

		