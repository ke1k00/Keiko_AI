# recursive neural network (expands with time)
# using LSTM structured neural network

import shopping_data
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
import chinese_vec
import numpy as np

x_train, y_train, x_test, y_test = shopping_data.load_data()

print('x train',x_train.shape)
print('y train',y_train.shape)
print('x test',x_test.shape)
print('y test',y_test.shape)
print(x_train[0])
print(y_train[0])

vocalen, word_index = shopping_data.createWordIndex(x_train, x_test)
print(word_index)
print('词典总词数：', vocalen)

x_train_index = shopping_data.word2Index(x_train, word_index)
x_test_index = shopping_data.word2Index(x_test, word_index)

maxlen = 25
x_train_index = sequence.pad_sequences(x_train_index, maxlen = maxlen)
x_test_index = sequence.pad_sequences(x_test_index, maxlen = maxlen)

#autotmatically generate matrix for each word
word_vecs = chinese_vec.load_word_vecs()
embedding_matrix = np.zeros((vocalen, 300)) # dimension 300
for word, i in word_index.items():
    embedding_vector = word_vecs.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

# insert the data into RNN
model = Sequential()
model.add(Embedding(trainable=False, weights=[embedding_matrix], input_dim=vocalen, output_dim=300, input_length=maxlen))
model.add(LSTM(128, return_sequences=True)) # dimension/no. of characteristics: 128
model.add(LSTM(128))

model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x_train_index, y_train,
          batch_size=512,
          epochs=200)
score, acc = model.evaluate(x_test_index, y_test)

print('Test score', score)
print('Test accuracy', acc)

# error: FileNotFoundError: [Errno 2] No such file or directory: 'C:\\Users\\zihui\\OneDrive\\Desktop\\Zihui_AI\\第13课资料\\lesson13/sgns.target.word-word.dynwin5.thr10.neg5.dim300.iter5'
