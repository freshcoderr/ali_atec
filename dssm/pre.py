#/usr/bin/env python
#coding=utf-8
import jieba
import jieba.posseg as pseg
import sys,re
import numpy as np
from util import *

spliter1 = re.compile(r"([\&\ \!\:\(\)\*\-\"\/\;\#\+\~\{\[\]\}])")

def proline(line):
    array = line.strip().split('\t')
    s1 = array[1].decode('utf8').replace("***","*")
    s2 = array[2].decode('utf8').replace("***","*")
    s1l = [w for w in jieba.cut(s1) if w.strip()]
    s2l = [w for w in jieba.cut(s2) if w.strip()]
    res = []
    for word in s1l:
        res.append(wdic2.get(word,1))
        # if 1 < wdic2.get(word,1) < 6:
        #     print word,wdic2.get(word,1)
    while len(res) < maxlen:
        res.extend([0] * 5)
        # res.extend(res)
    res = res[:maxlen]
    for word in s2l:
        res.append(wdic2.get(word,1))
    while len(res) < 2 * maxlen:
        res.extend([0] * 5)
        # res.extend(res[maxlen:])
    res = res[:2 * maxlen]
    _, simword = bleu(s1,s2,3,wdic2)
    simword = map(lambda x:x.replace(" ",""),simword)
    simwordl = map(lambda x:wdic2[x],simword)
    simwordl.extend([0] * 5)
    return res,s1l,s2l,simword,simwordl[:5]

def prolinec(line):
    array = line.strip().split('\t')
    s1 = array[1].decode('utf8').replace("***","*")
    s2 = array[2].decode('utf8').replace("***","*")
    s1l = s1
    s2l = s2
    res = []
    for char in s1l:
        res.append(chardict2.get(char,1))
    while len(res) < maxlen2:
        res.extend([0] * 5)
        # res.extend(res)
    res = res[:maxlen2]
    for char in s2l:
        res.append(chardict2.get(char,1))
    while len(res) < 2 * maxlen2:
        res.extend([0] * 5)
        # res.extend(res[maxlen2:])
    res = res[:2 * maxlen2]
    return res

if __name__ == '__main__':
    # process(sys.argv[1], sys.argv[2])
    jieba.enable_parallel(4)
    print bleu("abcdefg","abcjhedfkaefg")
    wdic = {}

    for line in open('../data/atec_nlp_sim_train_add.csv'):
        if not line:
            continue
        array = line.strip().split('\t')
        s1 = array[1].decode('utf8')
        s2 = array[2].decode('utf8')
        label = array[3]
        _, res = bleu(s1, s2)
        for k in res:
            wdic[k.replace(' ','')] = wdic.get(k.replace(' ',''), 0) + 0.33

    for line in open('../data/atec_nlp_sim_train.csv'):
        if not line:
            continue
        array = line.strip().split('\t')
        s1 = array[1].decode('utf8')
        s2 = array[2].decode('utf8')
        label = array[3]
        _, res = bleu(s1, s2)
        for k in res:
            wdic[k.replace(' ','')] = wdic.get(k.replace(' ',''), 0) + 0.33

    f_out = open('./wdict','w')
    for k,v in wdic.items():
        if v > 15:
            print >> f_out,k.encode('utf8'),int(v),"n"
    f_out.close()

    # jieba.load_userdict('./wdict')
    chardict = {}
    for line in open('../data/atec_nlp_sim_train_add.csv'):
        if not line:
            continue
        array = line.strip().replace(' ','').split('\t')
        s1 = array[1].decode('utf8')
        s2 = array[2].decode('utf8')
        s1l = [w for w in jieba.cut(s1) if w.strip()]
        s2l = [w for w in jieba.cut(s2) if w.strip()]
        for word in s1l:
            wdic[word] = wdic.get(word, 0) + 1
        for word in s2l:
            wdic[word] = wdic.get(word, 0) + 1
        for char in s1:
            chardict[char] = chardict.get(char,0) + 1
        for char in s2:
            chardict[char] = chardict.get(char,0) + 1

    for line in open('../data/atec_nlp_sim_train.csv'):
        if not line:
            continue
        array = line.strip().replace(' ','').split('\t')
        s1 = array[1].decode('utf8')
        s2 = array[2].decode('utf8')
        s1l = [w for w in jieba.cut(s1) if w.strip()]
        s2l = [w for w in jieba.cut(s2) if w.strip()]
        for word in s1l:
            wdic[word] = wdic.get(word, 0) + 1
        for word in s2l:
            wdic[word] = wdic.get(word, 0) + 1
        for char in s1:
            chardict[char] = chardict.get(char,0) + 1
        for char in s2:
            chardict[char] = chardict.get(char,0) + 1

    wordflag = {}
    wdic2 = {}
    index = 2 + len(wordflag.items())
    limit = 5
    f_out = open('./wdict2','w')
    for k,v in wdic.items():
        if v > limit:
            wdic2[k] = index
            print >> f_out,k.encode('utf8'),int(v),"n",index
            index += 1
        else:
            wdic2[k] = 1
            l = [w.flag for w in pseg.cut(k)]
            if len(l) == 1:
                wdic2[k] = wordflag.get(l[0], 1)
            # print >> f_out, k.encode('utf8'), int(v), "n", 1
    f_out.close()
    print index

    chardict2 = {}
    index = 2
    f_out = open('./cdict2','w')
    for k,v in chardict.items():
        if v > limit:
            chardict2[k] = index
            print >> f_out,k.encode('utf8'),int(v),"n",index
            index += 1
        else:
            chardict2[k] = 1
    f_out.close()
    print index

    maxlen = 24
    maxlen2 = 48
    f_out = open('./debugtra','w')
    f_out2 = open('../data/train.csv','w')
    f_out3 = open('../data/trainc.csv', 'w')
    jieba.load_userdict('./dict')
    # jieba.load_userdict('./wdict2')
    for line in open('../data/atec_nlp_sim_train_add.csv'):
        if not line:
            continue
        line = line.replace(' ', '')
        array = line.strip().split('\t')
        res, s1l, s2l, simword, simwordl = proline(line)
        resc = prolinec(line)
        print >> f_out,(",".join(s1l)).encode('utf8'),(",".join(s2l)).encode('utf8'),array[3],"a" + array[0],(",".join(simword)).encode('utf8')
        print >> f_out2,array[3]+","+",".join(map(str,res))+","+",".join(map(str,simwordl)) + "," + "a" + array[0]
        print >> f_out3, ",".join(map(str, resc)) + "," + "a" + array[0]

    f_out = open('./debugtest', 'w')
    f_out2 = open('../data/test.csv', 'w')
    f_out3 = open('../data/testc.csv', 'w')
    for line in open('../data/atec_nlp_sim_train.csv'):
        if not line:
            continue
        line = line.replace(' ', '')
        array = line.strip().split('\t')
        res, s1l, s2l, simword, simwordl = proline(line)
        resc = prolinec(line)
        print >> f_out, (",".join(s1l)).encode('utf8'), (",".join(s2l)).encode('utf8'), array[3], array[0],(",".join(simword)).encode('utf8')
        print >> f_out2, array[3] + "," + ",".join(map(str, res))+","+",".join(map(str,simwordl)) + "," + array[0]
        print >> f_out3, ",".join(map(str, resc)) + "," + array[0]