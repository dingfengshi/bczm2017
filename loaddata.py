# -*- coding:utf-8 -*-

import gensim
import config
import jieba
import tensorflow as tf

try:
    import cPickle as pickle
except ImportError:
    import pickle
import numpy as np


class loaddata:
    def __init__(self):
        self.conf = config.getoption()
        self.model = gensim.models.Word2Vec.load(self.conf["Word2vecmodel"])

    def loadword(self):
        vocab = self.model.wv.vocab
        return vocab

    def create_index_vocab(self):
        vocab = self.loadword()
        index_word_dict = {}
        word_index_dict = {}
        i = 0
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
        fin = open(self.conf["notraindata"], "r",encoding="utf-8")
        fout1 = open(self.conf["aftertraindata1"], 'wb')
        fout2 = open(self.conf["aftertraindata2"], 'wb')
        fout3 = open(self.conf["aftertraindata_flag"], 'wb')
        word_index_dict, index_word_dict = self.get_index_vocab()
        s1_image = []
        s2_image = []
        flagvec = []
        for line in fin.readlines():
            flag, sen1, sen2 = line.split('\t')
            vec1 = []
            vec2 = []
            words = list(jieba.cut(sen1))
            self.padding(words)
            for word in words:
                if word in word_index_dict:
                    vec1.append(word_index_dict[word])
                else:
                    vec1.append(9999999)
            words = list(jieba.cut(sen2))
            self.padding(words)
            for word in words:
                if word in word_index_dict:
                    vec2.append(word_index_dict[word])
                else:
                    vec2.append(9999999)
            s1_image.append(np.array(vec1))
            s2_image.append(np.array(vec2))
            flagvec.append(np.array(float(flag)))
        s1_image = np.array(s1_image)
        s2_image = np.array(s2_image)
        flagvec = np.array(flagvec, dtype='float32')

        pickle.dump(s1_image, fout1)
        pickle.dump(s2_image, fout2)
        pickle.dump(flagvec, fout3)
        fin.close()
        fout1.close()
        fout2.close()
        fout3.close()

    def get_aftertraindata(self):
        # 取出已经保存好的输入张量
        with open(self.conf["aftertraindata1"], 'rb') as f1:
            s1_image = pickle.load(f1)
        with open(self.conf["aftertraindata2"], 'rb') as f2:
            s2_image = pickle.load(f2)
        with open(self.conf["aftertraindata_flag"], 'rb') as f3:
            flagvec = pickle.load(f3)
        return s1_image, s2_image, flagvec

    def padding(self, sen1):
        maxlen = self.conf['sen_max_len']
        if len(sen1) < maxlen:
            for i in range(maxlen - len(sen1)):
                sen1.append("闊")

    def lookup_table(self, index_word_dict, sen, start_index, end_index):
        textvec = []
        for eachsen in sen[start_index:end_index]:
            senvec = []
            i = 0
            for index in eachsen:
                if i < self.conf["sen_max_len"]:
                    if int(index) != 9999999:
                        senvec.append(self.model[index_word_dict[index]])
                    else:
                        senvec.append(np.zeros((self.conf["word2vec_dim"])))
                    i = i + 1
            sentens = np.array(senvec)
            textvec.append(np.array(sentens))
        texttens = np.array(textvec)
        return texttens
