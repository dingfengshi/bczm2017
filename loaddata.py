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

    def get_index_vocab(self):
        model, vocab = self.loadword()
        index_word_dict = {}
        word_index_dict = {}
        i = 2  # i=0<EOF>  i=1<UNk>
        index_word_dict[0]="EOF"
        index_word_dict[1]="UNK"
        word_index_dict["EOF"]=0
        word_index_dict["UNK"]=1
        for key in vocab.keys():
            index_word_dict[i] = key
            word_index_dict[key]=i
        with open(self.conf["word_index_dict"],'wb') as f1:
            pickle.dump(word_index_dict,f1 )
        with open(self.conf["index_word_dict"],"wb") as f2:
            pickle.dump(index_word_dict, open(self.conf["index_word_dict"], "wb"))

    def pro_traingset(self):
        fin = open(self.conf["notraindata"])
        fout = open(self.conf["aftertraindata"], 'wb')
        #加载字典
        with open(self.conf["word_index_dict"], 'rb') as f1:
            word_index_dict=pickle.load(f1)

        with open(self.conf["index_word_dict"],"rb") as f2:
            index_word_dict=pickle.load(f2)


        for line in fin.readlines():
            flag, sen1, sen2 = line.split()
            words = jieba.cut(sen1)
            for word in words:
                if

            word = jieba.cut(sen2)
            s2 = " ".join(word)
            fout.write(str(flag) + '\t' + s1 + '\t' + s2 + '\n')
        fin.close()
        fout.close()
