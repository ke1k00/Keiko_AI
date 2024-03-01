import shopping_data
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import Flatten

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

# from prev lesson
model = Sequential()
model.add(Embedding(trainable=True, input_dim=vocalen, output_dim=300, input_length=maxlen))
model.add(Flatten())
# 3 隐藏层
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))

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