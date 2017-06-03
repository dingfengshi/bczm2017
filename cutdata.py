fin = open("E:\word2vec\BoP2017-DBQA.train.txt", encoding="utf-8")
fout = open("E:\word2vec\select.train.txt", 'w', encoding="utf-8")


def selectdata(tempset, posnum):
    pos = posnum
    for eachdata in tempset:
        _flag = eachdata[0]
        _sen1 = eachdata[1]
        _sen2 = eachdata[2]
        if (int(_flag) == 1):
            fout.write(str(_flag) + '\t' + _sen1 + '\t' + _sen2)
        else:
            if pos > 0:
                fout.write(str(_flag) + '\t' + _sen1 + '\t' + _sen2)
                pos = pos - 1


tempset = []
posnum = 0
now_sen = ""

for each in fin.readlines():
    flag, sen1, sen2 = each.split("\t")
    if sen1 != now_sen:
        selectdata(tempset, posnum)
        tempset = []
        posnum = 0
        count = 0
        now_sen = sen1
        if int(flag) == 1:
            posnum = posnum + 1
        tempset.append([flag, sen1, sen2])
    else:
        if int(flag) == 1:
            posnum = posnum + 1
        tempset.append([flag, sen1, sen2])
