import gensim
import config
import jieba
import pickle


class loaddata:
    def __init__(self):
        self.conf = config.getoption()

    def loadword(self):
        model = gensim.models.Word2Vec.load(self.conf["Word2vecmodel"])
        vocab = model.wv.vocab
        return model, vocab

    def create_index_vocab(self):
        model, vocab = self.loadword()
        index_word_dict = {}
        word_index_dict = {}
        i = 1
        index_word_dict[0] = "UNK"
        word_index_dict["UNK"] = 0
        for key in vocab.keys():
            index_word_dict[i] = key
            word_index_dict[key] = i
            i = i + 1
        with open(self.conf["word_index_dict"], 'wb') as f1:
            pickle.dump(word_index_dict, f1)
        with open(self.conf["index_word_dict"], "wb") as f2:
            pickle.dump(index_word_dict, open(self.conf["index_word_dict"], "wb"))

    def get_index_vocab(self):
        # 加载字典
        with open(self.conf["word_index_dict"], 'rb') as f1:
            word_index_dict = pickle.load(f1)

        with open(self.conf["index_word_dict"], "rb") as f2:
            index_word_dict = pickle.load(f2)
        return word_index_dict, index_word_dict

    def pro_traingset(self):
        # 把每个句子中的单词数字化并保存在文件中
        fin = open(self.conf["notraindata"])
        fout = open(self.conf["aftertraindata"], 'wb')
        word_index_dict, index_word_dict = self.get_index_vocab()
        for line in fin.readlines():
            flag, sen1, sen2 = line.split('\t')
            words = jieba.cut(sen1)
            for word in words:
                if word_index_dict.has_key(word):
                    fout.write(word_index_dict[word] + ' ')
                else:
                    fout.write(str(0) + ' ')
            fout.write('\t')
            words = jieba.cut(sen2)
            for word in words:
                if word_index_dict.has_key(word):
                    fout.write(word_index_dict[word] + ' ')
                else:
                    fout.write(str(0) + ' ')
            fout.write('\n')
        fin.close()
        fout.close()


    def get_aftertraindata(self):
        fin=open(self.conf["aftertraindata"], 'rb')

        fin.close()
