#/usr/bin/env python
#coding=utf-8
import jieba
import sys
import numpy as np

def score(y_pred,y_true,t = 0.5):
    y_pred = np.array(y_pred).reshape(-1)
    y_true = np.array(y_true).reshape(-1)
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i in range(y_pred.shape[0]):
        if y_pred[i] > t and y_true[i] == 1:
            tp += 1
        elif  y_pred[i] <= t and y_true[i] != 1:
            tn += 1
        elif  y_pred[i] <= t and y_true[i] == 1:
            fn += 1
        elif  y_pred[i] > t and y_true[i] != 1:
            fp += 1

    print tp,tn,fp,fn
    print "acc:", tp*1.0/(tp+fp)
    print "recall:", tp*1.0/(tp+fn)
    print "F1:", 2.0*tp/(2*tp+fp+fn)


def bleu(c,r,length = 3, worddict = None):  # function to calc the simiarlity of two words
    if len(c) == 0 or len(r) == 0:
        return 0.0
    bp = 0
    sumpn = 0.0
    simwordt = []
    simword = []
    for i in range(1, 1 + length):
        rcount = {}
        for j in range(len(r) - i):
            w = 1
            rcount[" ".join(r[j:j+1+i])] = rcount.get(" ".join(r[j:j+1+i]),0) + w
        ccount = {}
        ctcount = {}
        for j in range(len(c) - i):
            w = 1
            t1 = " ".join(c[j:j+1+i])
            ccount[t1] = min(ccount.get(t1,0) + w,rcount.get(t1,0))
            if ccount[t1] > 0:
                simwordt.append(t1)
            ctcount[t1] = ctcount.get(t1, 0) + w
        temp = (1.0/length) * sum(map(lambda x:x[1],ccount.items()))*1.0/(sum(map(lambda x:x[1],ctcount.items()))+0.0001)
        sumpn += temp
    for word in simwordt:
        subword = False
        for otherword in simwordt:
            if otherword != word and word in otherword:
                subword = True
                break
        if not subword:
            if worddict and worddict.get(word.replace(' ',''),1) == 1:
                continue
            simword.append(word)
    return bp + sumpn, simword


# print bleu("abcdefg", "abcjhedfkaefg")