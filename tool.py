import config
import loaddata
import keras
import pickle


def continue_train():
    option = config.getoption()
    print("loading data...\n")
    load = loaddata.loaddata()
    word_index_dict, index_word_dict = load.get_index_vocab()

    s1_image, s2_image, flag_vec = load.get_aftertraindata(has_flag=True)
    datanum = flag_vec.size

    print("loading model...\n")
    model = keras.models.load_model(option["save_to"])

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

        s1_image_batch = load.lookup_table(index_word_dict, s1_image,
                                           batchnum * option["batch_size"] * option["split_batch"], datanum)
        s2_image_batch = load.lookup_table(index_word_dict, s2_image,
                                           batchnum * option["batch_size"] * option["split_batch"], datanum)
        model.fit([s1_image_batch, s2_image_batch],
                  flag_vec[batchnum * option["batch_size"] * option["split_batch"]:datanum],
                  epochs=1,
                  batch_size=option["batch_size"],
                  shuffle=True)
        print("save model...\n")
        model.save(option["save_to"])


def get_ans():
    option = config.getoption()
    fout = open(option["ans_save_to"], "w")
    print("loading data...\n")
    load = loaddata.loaddata()
    word_index_dict, index_word_dict = load.get_index_vocab()

    s1_image, s2_image = load.get_aftertraindata(has_flag=False)
    datanum = s1_image.__len__()

    print("loading model...\n")
    model = keras.models.load_model(option["save_to"])

    batchnum = datanum // (option["batch_size"] * option["split_batch"])

    for batch in range(batchnum):
        print("第%d个批次" % batch)
        start = batch * option["batch_size"] * option["split_batch"]
        end = start + option["batch_size"] * option["split_batch"]
        s1_image_batch = load.lookup_table(index_word_dict, s1_image, start, end)
        s2_image_batch = load.lookup_table(index_word_dict, s2_image, start, end)
        ans = model.predict([s1_image_batch, s2_image_batch])
        for each in ans.flatten():
            fout.write(str(each) + '\n')

    s1_image_batch = load.lookup_table(index_word_dict, s1_image,
                                       batchnum * option["batch_size"] * option["split_batch"], datanum)
    s2_image_batch = load.lookup_table(index_word_dict, s2_image,
                                       batchnum * option["batch_size"] * option["split_batch"], datanum)
    ans = model.predict([s1_image_batch, s2_image_batch])
    for each in ans.flatten():
        fout.write(str(each) + '\n')

    fout.close()

continue_train()

if __name__ == '__main__':
    pass
