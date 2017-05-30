def getoption(Word2vecmodel="/home/ste/word2vec-for-wiki-master/wiki.model.300d.model",
              notraindata="/home/ste/word2vec-for-wiki-master/BoP2017-DBQA.train.txt",
              aftertraindata="/home/ste/word2vec-for-wiki-master/BoPcut.train.txt",
              word_index_dict="/home/ste/cnn/w_i_d.pkl",
              index_word_dict="/home/ste/cnn/i_w_d.pkl"
              ):
    option = {}
    option["Word2vecmodel"] = Word2vecmodel  # word2vec模型地址
    option["notraindata"] = notraindata  # 未处理的训练集地址
    option["aftertraindata"] = aftertraindata  # 分好词的训练集地址
    option["word_index_dict"] = word_index_dict  # key为单词,value为编号的字典
    option["index_word_dict"] = index_word_dict  # key为单词,value为编号的字典

    return option
