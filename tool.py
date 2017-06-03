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
    s1_image = load.lookup_table(index_word_dict, s1_image)
    s2_image = load.lookup_table(index_word_dict, s2_image)

    print("loading model...\n")
    model = keras.models.load_model(option["save_to"])
    model.fit([s1_image, s2_image], flag_vec, epochs=option["epoch"], batch_size=option["batch_size"])
    model.save(option["save_to"])
    print("done!")


def get_ans():
    option = config.getoption()
    print("loading data...\n")
    load = loaddata.loaddata()
    word_index_dict, index_word_dict = load.get_index_vocab()

    s1_image, s2_image, flag_vec = load.get_aftertraindata()
    s1_image = load.lookup_table(index_word_dict, s1_image)
    s2_image = load.lookup_table(index_word_dict, s2_image)

    print("loading model...\n")
    model = keras.models.load_model(option["save_to"])

    ans = model.predict([s1_image, s2_image])
    print(ans)
    with open(option["ans_save_to"], "wb") as f:
        for each in ans.flatten():
            f.write(str(each)+'\n')


if __name__ == '__main__':
    pass
