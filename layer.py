from keras.layers import Activation
from keras.models import Sequential
from keras.datasets import mnist
from keras.utils import np_utils
from keras import backend as K
import keras
import numpy as np
import config
import loaddata

option = config.getoption()

# 得到处理过的训练数据
print("loading data...\n")
load = loaddata.loaddata()
word_index_dict, index_word_dict = load.get_index_vocab()

s1_image, s2_image, flag_vec = load.get_aftertraindata(has_flag=True)
datanum = flag_vec.size

print("building model...\n")
s1_input = keras.layers.Input(shape=(option["sen_max_len"], option["word2vec_dim"]), dtype="float32", name="sen1")
s2_input = keras.layers.Input(shape=(option["sen_max_len"], option["word2vec_dim"]), dtype="float32", name="sen2")

s1_lstm_out = keras.layers.BatchNormalization()(s1_input)
s1_lstm_out = keras.layers.Bidirectional(keras.layers.LSTM(option["LSTM_unit"],
                                                           input_shape=(option["sen_max_len"], option["word2vec_dim"]),
                                                           dropout=0.5,
                                                           activation='tanh',
                                                           ), merge_mode="ave")(s1_lstm_out)

s2_lstm_out = keras.layers.BatchNormalization()(s2_input)
s2_lstm_out = keras.layers.Bidirectional(keras.layers.LSTM(option["LSTM_unit"],
                                                           input_shape=(option["sen_max_len"], option["word2vec_dim"]),
                                                           activation="tanh",
                                                           dropout=0.5), merge_mode="ave")(s2_lstm_out)

merge = keras.layers.concatenate([s1_lstm_out, s2_lstm_out])

output = keras.layers.BatchNormalization()(merge)
output = keras.layers.Dense(32, kernel_regularizer=keras.regularizers.l1(0.01))(output)
output = keras.layers.BatchNormalization()(output)
output = keras.layers.Activation("relu")(output)
output = keras.layers.Dense(32, kernel_regularizer=keras.regularizers.l1(0.01))(output)
output = keras.layers.BatchNormalization()(output)
output = keras.layers.Activation("relu")(output)
output = keras.layers.Dense(1, activation="sigmoid", kernel_regularizer=keras.regularizers.l2(0.01))(output)

model = keras.models.Model(inputs=[s1_input, s2_input], output=[output])

model.compile(optimizer="adam", loss="binary_crossentropy")

print("traing model...\n")
batchnum = datanum // (option["batch_size"] * option["split_batch"])

for epoch in range(option["epoch"]):
    for batch in range(batchnum):
        print("epoch%d,第%d个批次" % (epoch, batch))
        start = batch * option["batch_size"] * option["split_batch"]
        end = start + option["batch_size"] * option["split_batch"]
        s1_image_batch = load.lookup_table(index_word_dict, s1_image, start, end)
        s2_image_batch = load.lookup_table(index_word_dict, s2_image, start, end)
        model.fit([s1_image_batch, s2_image_batch], flag_vec[start:end], epochs=1,
                  batch_size=option["batch_size"])
    print("save model...\n")
    model.save(option["save_to"])

    s1_image_batch = load.lookup_table(index_word_dict, s1_image,
                                       batchnum * option["batch_size"] * option["split_batch"], datanum)
    s2_image_batch = load.lookup_table(index_word_dict, s2_image,
                                       batchnum * option["batch_size"] * option["split_batch"], datanum)
    model.fit([s1_image_batch, s2_image_batch],
              flag_vec[batchnum * option["batch_size"] * option["split_batch"]: datanum],
              epochs=1,
              batch_size=option["batch_size"])
    print("save model...\n")
    model.save(option["save_to"])
