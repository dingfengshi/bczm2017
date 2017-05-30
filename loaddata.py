import gensim
import config
import jieba


class loaddata:
    def __init__(self):
        self.conf = config.getoption()

    def loadword(self):
        model = gensim.models.Word2Vec.load(self.conf["Word2vecmodel"])
        vocab = model.wv.vocab
        return model, vocab

    def get_index_word(self):
        model, vocab = self.loadword()
        index_word_dict = {}
        word_index_dict = {}
        i = 2  # i=0<EOF>  i=1<UNk>
        for key in vocab.keys():
            index_word_dict[i] = key
            word_index_dict[key]=i


    def pro_traingset(self):
        fin = open(self.conf["notraindata"])
        fout = open(self.conf["aftertraindata"], 'wb')

        for line in fin.readlines():
            flag, sen1, sen2 = line.split()
            word = jieba.cut(sen1)
            s1 = " ".join(word)
            word = jieba.cut(sen2)
            s2 = " ".join(word)
            fout.write(str(flag) + '\t' + s1 + '\t' + s2 + '\n')

        fin.close()
        fout.close()
