import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D, Bidirectional, CuDNNLSTM
from keras.utils.np_utils import to_categorical
import re
from sklearn.utils import shuffle

#Training
preprocessed = False
if preprocessed:
	# make positive and negative label files and concatenate them with shuffling
	data = pd.read_csv('training-data.csv')
	label_positive=data.loc[data['label'] == ('positive')]
	label_positive.to_csv('positive_label.csv')
	label_negative=data.loc[data['label'] == ('negative')]
	label_negative.to_csv('negative_label.csv')
	label_positive = pd.read_csv('positive_label.csv')
	label_negative = pd.read_csv('negative_label.csv')
	frame = [label_positive,label_negative]
	data_ = pd.concat(frame)
	data_ = shuffle(data_)
	data_.to_csv('training_preprocessed.csv')

data_ = pd.read_csv('training_preprocessed2.csv')
# data_['text'] = data_['text'].apply(lambda x:x.lower())
# data_.to_csv('training_preprocessed1.csv')
# data_['text'] = data_['text'].apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x)))
# data_.to_csv('training_preprocessed2.csv')
max_features=2000
# create a tozenizer to convert words to vectors
tokenizer = Tokenizer(num_words=max_features, split=' ')
tokenizer.fit_on_texts(data_['text'].values)
# convert these vectors to arrays
X = tokenizer.texts_to_sequences(data_['text'].values)
X = pad_sequences(X)
Y = pd.get_dummies(data_['label']).values

batch_size = 32
embed_dimension = 128
model = Sequential()
model.add(Embedding(max_features,embed_dimension,input_length=X.shape[1],dropout=0.2))
model.add(Bidirectional(CuDNNLSTM(200,kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, return_sequences=False, return_state=False, stateful=False)))
model.add(Dense(2,activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
print(model.summary())

#make validation data
val_X = X[-60000:]
val_Y = Y[-60000:]

model.fit(X, Y, batch_size=batch_size, epochs=2, verbose=1, shuffle=False, validation_data=(val_X,val_Y))
model.save_weights("model.h5")
print("Saved model to disk")

#Testing
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
# score,acc = model.evaluate(X_test, Y_test, verbose = 2, batch_size = batch_size)
# print("score: %.2f" % (score))
# print("acc: %.2f" % (acc))

