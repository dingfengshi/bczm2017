from keras.layers import Activation
from keras.models import Sequential
from keras.datasets import mnist
from keras.utils import np_utils
import keras
import numpy as np
import config
import loaddata

option = config.getoption()

# 得到处理过的训练数据
print("loading data...\n")
load = loaddata.loaddata()
word_index_dict, index_word_dict = load.get_index_vocab()

s1_image, s2_image, flag_vec = load.get_aftertraindata()
s1_image = load.lookup_table(index_word_dict, s1_image)
s2_image = load.lookup_table(index_word_dict, s2_image)

print("building model...\n")
s1_input = keras.layers.Input(shape=(option["sen_max_len"], option["word2vec_dim"]), dtype="float32", name="sen1")
s2_input = keras.layers.Input(shape=(option["sen_max_len"], option["word2vec_dim"]), dtype="float32", name="sen2")

s1_lstm_out = keras.layers.LSTM(option["LSTM_unit"])(s1_input)
s2_lstm_out = keras.layers.LSTM(option["LSTM_unit"])(s2_input)

merge = keras.layers.concatenate([s1_lstm_out, s2_lstm_out])
merge = keras.layers.Dense(option["DENSE_unit"], activation="relu")(merge)
merge = keras.layers.Dense(option["DENSE_unit"], activation="relu")(merge)
merge = keras.layers.Dense(option["DENSE_unit"], activation="relu")(merge)

output = keras.layers.Dense(1, activation="sigmoid", name='output')(merge)

model = keras.models.Model(inputs=[s1_input, s2_input], output=[output])

model.compile(optimizer="rmsprop", loss="binary_crossentropy")

print("traing model...\n")
model.fit([s1_image, s2_image], flag_vec, epochs=option["epoch"], batch_size=option["batch_size"])

model.save(option["save_to"])