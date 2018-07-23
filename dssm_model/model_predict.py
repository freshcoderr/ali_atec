#/usr/bin/env python
#coding=utf-8
import jieba
import jieba.posseg as pseg
import sys,re
import os
import numpy as np
from util import *
import logging
import pandas as pd
from keras import backend as K
from keras.layers import *
from keras.layers import Activation, Input
from keras.layers.core import Dense, Lambda, Reshape
from keras.layers.convolutional import Convolution1D
from keras.layers.merge import concatenate, dot
from keras.models import Model, load_model
from JoinAttLayer import Attention
from keras.optimizers import *
from util import *
from JoinAttLayer import Attention
import numpy as np
import pandas as pd
import os,sys, random
random.seed(42)
np.random.seed(42)

from keras import backend as K
from keras.layers import *
from keras.layers import Activation, Input
from keras.layers.core import Dense, Lambda, Reshape
from keras.layers.convolutional import Convolution1D
from keras.layers.merge import concatenate, dot
from keras.models import Model
from JoinAttLayer import Attention
from keras.optimizers import *
from util import *
from sklearn.utils import shuffle
from sklearn.metrics import roc_auc_score
import cPickle as pkl
reload(sys)
sys.setdefaultencoding('utf8')

maxlen = 24
wordnum = 4482 + 1
maxlen2 = 48
charnum = 1237 + 1
embedsize = 200
lstmsize = 10



input1 = Input(shape=(maxlen,))
input2 = Input(shape=(maxlen,))
input3 = Input(shape=(5,))
embed1 = Embedding(wordnum,embedsize)
# lstm0 = CuDNNLSTM(lstmsize,return_sequences = True)
# lstm1 = Bidirectional(CuDNNLSTM(lstmsize))
# lstm2 = CuDNNLSTM(lstmsize)
lstm0 = LSTM(lstmsize,return_sequences = True)
lstm1 = Bidirectional(LSTM(lstmsize))
lstm2 = LSTM(lstmsize)
att1 = Attention(10)
den = Dense(64,activation = 'tanh')

# att1 = Lambda(lambda x: K.max(x,axis = 1))
v3 = embed1(input3)
v1 = embed1(input1)
v2 = embed1(input2)
v11 = lstm1(v1)
v22 = lstm1(v2)
v1ls = lstm2(lstm0(v1))
v2ls = lstm2(lstm0(v2))
v1 = Concatenate(axis=1)([att1(v1),v11])
v2 = Concatenate(axis=1)([att1(v2),v22])

input1c = Input(shape=(maxlen2,))
input2c = Input(shape=(maxlen2,))
embed1c = Embedding(charnum,embedsize)
# lstm1c = Bidirectional(CuDNNLSTM(6))
lstm1c = Bidirectional(LSTM(6))
att1c = Attention(10)
v1c = embed1(input1c)
v2c = embed1(input2c)
v11c = lstm1c(v1c)
v22c = lstm1c(v2c)
v1c = Concatenate(axis=1)([att1c(v1c),v11c])
v2c = Concatenate(axis=1)([att1c(v2c),v22c])


mul = Multiply()([v1,v2])
sub = Lambda(lambda x: K.abs(x))(Subtract()([v1,v2]))
maximum = Maximum()([Multiply()([v1,v1]),Multiply()([v2,v2])])
mulc = Multiply()([v1c,v2c])
subc = Lambda(lambda x: K.abs(x))(Subtract()([v1c,v2c]))
maximumc = Maximum()([Multiply()([v1c,v1c]),Multiply()([v2c,v2c])])
sub2 = Lambda(lambda x: K.abs(x))(Subtract()([v1ls,v2ls]))
matchlist = Concatenate(axis=1)([mul,sub,mulc,subc,maximum,maximumc,sub2])
matchlist = Dropout(0.05)(matchlist)

matchlist = Concatenate(axis=1)([Dense(32,activation = 'relu')(matchlist),Dense(48,activation = 'sigmoid')(matchlist)])
res = Dense(1, activation = 'sigmoid')(matchlist)


model = Model(inputs=[input1, input2, input3, input1c, input2c], outputs=res)
model.compile(optimizer=Adam(lr = 0.0015,epsilon=0.000001), loss="binary_crossentropy")
model.summary()

spliter1 = re.compile(r"([\&\ \!\:\(\)\*\-\"\/\;\#\+\~\{\[\]\}])")
jieba.enable_parallel(4)
wdict2_path = "./wdict2"
chardict2_path = "./cdict2"
maxlen = 24
wordnum = 4482 + 1
maxlen2 = 48
charnum = 1237 + 1
embedsize = 200
lstmsize = 10

def proline(line, wdic2):
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
    simwordl = map(lambda x:wdic2.get(x, 1),simword)
    simwordl.extend([0] * 5)
    return res,s1l,s2l,simword,simwordl[:5]

def prolinec(line, chardict2):
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

def load_wdict2(wdict2_path):
    assert os.path.exists(wdict2_path)
    wdict2 = {}
    with open(wdict2_path, 'r') as f_wdict2:
        for line in f_wdict2.readlines():
            line_list = line.strip('\n').split(' ')
            word = line_list[0].decode('utf-8')
            index = line_list[-1]
            wdict2[word] = index
    return wdict2

def load_chardict2(chardict2_path):
    assert os.path.exists(chardict2_path)
    chardict2 = {}
    with open(chardict2_path) as f_chardict2:
        for line in f_chardict2.readlines():
            line_list = line.strip('\n').split(' ')
            char = line_list[0].decode('utf-8')
            index = line_list[-1]
            chardict2[char] = index
    return chardict2

def process(input_path, output_path):
    # logging.info("loading wdcit2")
    wdic2 = load_wdict2(wdict2_path)
    # logging.info(wdic2)

    # logging.info("loading chardict2")
    chardict2 = load_chardict2(chardict2_path)
    # logging.info(chardict2)

    # f_out = open('./debugtest', 'w')
    f_out2 = open('./test.csv', 'w')
    f_out3 = open('./testc.csv', 'w')
    f_in_index = []
    label = []
    for line in open(input_path):
        if not line:
            print "empty line ! "
            continue
        line = line.replace(' ', '')
        array = line.strip().split('\t')
        label.extend(array[3])
        res, s1l, s2l, simword, simwordl = proline(line, wdic2)
        resc = prolinec(line, chardict2)
        f_in_index.append(array[0])
        # print >> f_out, (",".join(s1l)).encode('utf8'), (",".join(s2l)).encode('utf8'), array[3], array[0],(",".join(simword)).encode('utf8')
        print >> f_out2, "1" + "," + ",".join(map(str, res))+","+",".join(map(str,simwordl)) + "," + "a" + array[0]
        print >> f_out3, ",".join(map(str, resc)) + "," + "a" + array[0]
    f_out2.close()
    f_out3.close()

    test_data = pd.read_csv("./test.csv", header=None)

    test_datac = pd.read_csv("./testc.csv", header=None)

    test_data = pd.concat([test_data, test_datac], axis=1).reset_index(drop=True)

    # logging.info("test_data.shape:")
    # logging.info(test_data.shape)

    test_y = test_data.iloc[:, 0]
    test_x1 = test_data.iloc[:, 1:1 + maxlen]
    test_x1[np.isnan(test_x1)] = 0
    test_x1[np.isinf(test_x1)] = 0
    test_x1 = test_x1.astype(np.int32)
    test_x2 = test_data.iloc[:, 1 + maxlen:1 + 2 * maxlen]
    test_x2[np.isnan(test_x2)]=0
    test_x2[np.isinf(test_x2)]=0
    test_x2 = test_x2.astype(np.int32)
    test_x3 = test_data.iloc[:, 1 + 2 * maxlen:1 + 2 * maxlen + 5]
    test_x3[np.isnan(test_x3)] = 0
    test_x3[np.isinf(test_x3)] = 0
    test_x3 = test_x3.astype(int)
    test_index = test_data.iloc[:, 1 + 2 * maxlen + 5]
    # test_index[np.isnan(test_index)] = 0
    # test_index[np.isinf(test_index)] = 0
    # test_index.astype(np.int32)
    test_x1c = test_data.iloc[:, 1 + 1 + 2 * maxlen + 5:1 + 1 + 2 * maxlen + 5 + maxlen2]
    test_x1c[np.isnan(test_x1c)] = 0
    test_x1c[np.isinf(test_x1c)] = 0
    test_x1c = test_x1c.astype(np.int32)
    test_x2c = test_data.iloc[:, 1 + 1 + 2 * maxlen + 5 + maxlen2:1 + 1 + 2 * maxlen + 5 + 2 * maxlen2]
    test_x2c[np.isnan(test_x2c)] = 0
    test_x2c[np.isinf(test_x2c)] = 0
    test_x2c = test_x2c.astype(np.int32)
    # logging.info("test_x1:")
    # logging.info(test_x1.shape)
    # logging.info("test_x2:")
    # logging.info(test_x2.shape)
    # logging.info("test_x3:")
    # logging.info(test_x3.shape)
    # logging.info("test_x1c:")
    # logging.info(test_x1c.shape)
    # logging.info("test_x2c:")
    # logging.info(test_x2c.shape)

    # ------------------model structure-------------------
    #
    # input1 = Input(shape=(maxlen,))
    # input2 = Input(shape=(maxlen,))
    # input3 = Input(shape=(5,))
    # embed1 = Embedding(wordnum, embedsize)
    # # lstm0 = CuDNNLSTM(lstmsize,return_sequences = True)
    # # lstm1 = Bidirectional(CuDNNLSTM(lstmsize))
    # # lstm2 = CuDNNLSTM(lstmsize)
    # lstm0 = LSTM(lstmsize, return_sequences=True)
    # lstm1 = Bidirectional(LSTM(lstmsize))
    # lstm2 = LSTM(lstmsize)
    # att1 = Attention(10)
    # den = Dense(64, activation='tanh')
    #
    # # att1 = Lambda(lambda x: K.max(x,axis = 1))
    # v3 = embed1(input3)
    # v1 = embed1(input1)
    # v2 = embed1(input2)
    # v11 = lstm1(v1)
    # v22 = lstm1(v2)
    # v1ls = lstm2(lstm0(v1))
    # v2ls = lstm2(lstm0(v2))
    # v1 = Concatenate(axis=1)([att1(v1), v11])
    # v2 = Concatenate(axis=1)([att1(v2), v22])
    #
    # input1c = Input(shape=(maxlen2,))
    # input2c = Input(shape=(maxlen2,))
    # embed1c = Embedding(charnum, embedsize)
    # # lstm1c = Bidirectional(CuDNNLSTM(6))
    # lstm1c = Bidirectional(LSTM(6))
    # att1c = Attention(10)
    # v1c = embed1(input1c)
    # v2c = embed1(input2c)
    # v11c = lstm1c(v1c)
    # v22c = lstm1c(v2c)
    # v1c = Concatenate(axis=1)([att1c(v1c), v11c])
    # v2c = Concatenate(axis=1)([att1c(v2c), v22c])
    #
    # mul = Multiply()([v1, v2])
    # sub = Lambda(lambda x: K.abs(x))(Subtract()([v1, v2]))
    # maximum = Maximum()([Multiply()([v1, v1]), Multiply()([v2, v2])])
    # mulc = Multiply()([v1c, v2c])
    # subc = Lambda(lambda x: K.abs(x))(Subtract()([v1c, v2c]))
    # maximumc = Maximum()([Multiply()([v1c, v1c]), Multiply()([v2c, v2c])])
    # sub2 = Lambda(lambda x: K.abs(x))(Subtract()([v1ls, v2ls]))
    # matchlist = Concatenate(axis=1)([mul, sub, mulc, subc, maximum, maximumc, sub2])
    # matchlist = Dropout(0.05)(matchlist)
    #
    # matchlist = Concatenate(axis=1)(
    #     [Dense(32, activation='relu')(matchlist), Dense(48, activation='sigmoid')(matchlist)])
    # res = Dense(1, activation='sigmoid')(matchlist)
    #
    # model = Model(inputs=[input1, input2, input3, input1c, input2c], outputs=res)
    # model.compile(optimizer=Adam(lr=0.0015, epsilon=0.000001), loss="binary_crossentropy")
    # model.summary()

    # ------------------model structure-------------------
    # attention = Attention(10)
    # model.load_weights('./modelsub_weights.h5')
    model.load_weights('./modelsub_weights_10_fold.h5')
    # y_pred = model.predict([np.array(map(int, test_x1)), np.array((int, test_x2)), np.array((int, test_x3)), np.array((int, test_x1c)), np.array((int, test_x2c))])
    y_pred = model.predict([test_x1,test_x2, test_x3, test_x1c, test_x2c])
    # print "y_pred: \n", y_pred
    label = np.array(label).astype(int)
    # print labelm
    # for i in range(280,300,1):
    #     t = i/1000.0
    #     print "t=",t
    #     score(y_pred, label, t=t)
    #     print '\n'

    score(y_pred, label, t=0.3)
    # score(y_pred, label, t=0.286)
    # index = []
    # for _ in test_index:
    #     index.append(str(_).replace("a",""))


    seg_data = pd.DataFrame()
    seg_data["index"] = f_in_index
    seg_data['predict'] = [int(x + 0.714) for x in y_pred]
    with open(output_path, 'w') as fout:
        for index in seg_data.index:
            fout.write(str(seg_data['index'].loc[index]) + '\t' + str(seg_data['predict'].loc[index]) + '\n')
    fout.close()



if __name__ == '__main__':
    # logging.basicConfig(level=logging.INFO,
    #                     format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
    process(sys.argv[1], sys.argv[2])
