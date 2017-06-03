import config
import loaddata
import keras
import pickle


def continue_train():
    option = config.getoption()
    print("loading data...\n")
    load = loaddata.loaddata()
    word_index_dict, index_word_dict = load.get_index_vocab()

    s1_image, s2_image, flag_vec = load.get_aftertraindata()
    datanum = flag_vec.size

    print("loading model...\n")
    model = keras.models.load_model(option["save_to"])

    for batch in range(datanum // (option["batch_size"] * 100)):
        print("第%d个批次" % batch)
        start = batch * option["batch_size"] * 100
        end = start + option["batch_size"] * 100
        s1_image_batch = load.lookup_table(index_word_dict, s1_image, start, end)
        s2_image_batch = load.lookup_table(index_word_dict, s2_image, start, end)
        model.fit([s1_image_batch, s2_image_batch], flag_vec[start:end], epochs=option["epoch"],
                  batch_size=option["batch_size"])
        model.save(option["save_to"])


def get_ans():
    option = config.getoption()
    fout = open(option["ans_save_to"], "w")
    print("loading data...\n")
    load = loaddata.loaddata()
    word_index_dict, index_word_dict = load.get_index_vocab()

    s1_image, s2_image, flag_vec = load.get_aftertraindata()
    datanum = flag_vec.size

    print("loading model...\n")
    model = keras.models.load_model(option["save_to"])

    batchnum = datanum // (option["batch_size"] * 100)

    for batch in range(batchnum):
        print("第%d个批次" % batch)
        start = batch * option["batch_size"] * 100
        end = start + option["batch_size"] * 100
        s1_image_batch = load.lookup_table(index_word_dict, s1_image, start, end)
        s2_image_batch = load.lookup_table(index_word_dict, s2_image, start, end)
        ans = model.predict([s1_image_batch, s2_image_batch])
        for each in ans.flatten():
            fout.write(str(each) + '\n')

    s1_image_batch = load.lookup_table(index_word_dict, s1_image, batchnum * option["batch_size"] * 100, datanum)
    s2_image_batch = load.lookup_table(index_word_dict, s2_image, batchnum * option["batch_size"] * 100, datanum)
    ans = model.predict([s1_image_batch, s2_image_batch])
    for each in ans.flatten():
        fout.write(str(each) + '\n')

    fout.close()

continue_train()

if __name__ == '__main__':
    pass
