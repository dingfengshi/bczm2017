def getoption(Word2vecmodel="/home/ste/word2vec-for-wiki-master/wiki.model.300d.model",
              notraindata="/home/ste/word2vec-for-wiki-master/BoP2017-DBQA.train.txt",
              aftertraindata="/home/ste/word2vec-for-wiki-master/BoPcut.train.txt"
              ):
    option = {}
    option["Word2vecmodel"] = Word2vecmodel  # word2vec模型地址
    option["notraindata"] = notraindata  # 未处理的训练集地址
    option["aftertraindata"] =aftertraindata #分好词的训练集地址

    return option
