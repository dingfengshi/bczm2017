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
datanum = flag_vec.size

print("building model...\n")
s1_input = keras.layers.Input(shape=(option["sen_max_len"], option["word2vec_dim"]), dtype="float32", name="sen1")
s2_input = keras.layers.Input(shape=(option["sen_max_len"], option["word2vec_dim"]), dtype="float32", name="sen2")

s1_lstm_out = keras.layers.LSTM(option["LSTM_unit"], return_sequences=True,
                                input_shape=(option["sen_max_len"], option["word2vec_dim"]))(s1_input)
s1_lstm_out = keras.layers.LSTM(option["LSTM_unit"], return_sequences=True)(s1_lstm_out)
s1_lstm_out = keras.layers.LSTM(option["LSTM_unit2"])(s1_lstm_out)

s2_lstm_out = keras.layers.LSTM(option["LSTM_unit"], return_sequences=True,
                                input_shape=(option["sen_max_len"], option["word2vec_dim"])
                                )(s2_input)
s2_lstm_out = keras.layers.LSTM(option["LSTM_unit"], return_sequences=True)(s2_lstm_out)
s2_lstm_out = keras.layers.LSTM(option["LSTM_unit2"])(s2_lstm_out)

merge = keras.layers.concatenate([s1_lstm_out, s2_lstm_out])
merge = keras.layers.Dense(option["DENSE_unit"], activation="relu")(merge)

output = keras.layers.Dense(1, activation="sigmoid", name='output')(merge)

model = keras.models.Model(inputs=[s1_input, s2_input], output=[output])

model.compile(optimizer="rmsprop", loss="binary_crossentropy")

print("traing model...\n")
for batch in range(datanum // (option["batch_size"] * 100)):
    print("第%d个批次" % batch)
    start = batch * option["batch_size"] * 100
    end = start + option["batch_size"] * 100
    s1_image_batch = load.lookup_table(index_word_dict, s1_image, start, end)
    s2_image_batch = load.lookup_table(index_word_dict, s2_image, start, end)
    model.fit([s1_image_batch, s2_image_batch], flag_vec[start:end], epochs=option["epoch"],
              batch_size=option["batch_size"])
    model.save(option["save_to"])
