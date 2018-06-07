# /usr/bin/env python
# -*- coding:utf-8 -*-
import xgboost
import pandas as pd
import sys
import jieba


def jieba_cut(sentence):
    seg_list = jieba.cut(sentence)
    return ",".join(seg_list).split(',')


def process(input_path, output_path):

    jieba.enable_parallel(4)

    df_predict = pd.DataFrame()
    #input file:  id  s1  s2
    df_input = pd.read_table(input_path)



    stopwords_path = "stopwords.txt"
    stopwords = []
    for word in open(stopwords_path, "r").readlines():
        stopwords.append(word.strip('\n'))


    def word_match_share(row):
        q1words = {}
        q2words = {}
        for word in jieba_cut(row['s1']):
            if word not in stopwords:
                q1words[word] = 1
        for word in jieba_cut(row['s2']):
            if word not in stopwords:
                q2words[word] = 1
        if len(q1words) == 0 or len(q2words) == 0:
            # The computer-generated chaff includes a few questions that are nothing but stopwords
            return 0
        shared_words_in_q = [w for w in q1words.keys() if w in q2words]

        # 这里一定要注意 float，否则最后计算结果为 int
        R = float(len(shared_words_in_q) * 2) / (len(q1words) + len(q2words))
        return R



    input_qs = pd.Series(df_input['s1'].tolist() + df_input['s2'].tolist()).astype(str)
    from collections import Counter
    def get_weight(count, eps=10000, min_count=2):
        if count < min_count:
            return 0
        else:
            return 1.0 / (count + eps)

    eps = 5000
    total_words = jieba_cut(" ".join(input_qs))
    counts = Counter(total_words)
    weights = {word: get_weight(count) for word, count in counts.items()}


    def tfidf_word_match_share(row):
        q1words = {}
        q2words = {}
        for word in jieba_cut(row['s1']):
            if word not in stopwords:
                q1words[word] = 1
        for word in jieba_cut(row['s2']):
            if word not in stopwords:
                q2words[word] = 1
        if len(q1words) == 0 or len(q2words) == 0:
            # The computer-generated chaff includes a few questions that are nothing but stopwords
            return 0

        shared_weights = [weights.get(w, 0) for w in q1words.keys() if w in q2words] + [weights.get(w, 0) for w in
                                                                                        q2words.keys() if w in q1words]
        total_weights = [weights.get(w, 0) for w in q1words] + [weights.get(w, 0) for w in q2words]

        R = sum(shared_weights) / sum(total_weights)
        return R





    df_predict['word_match'] = df_input.apply(word_match_share, axis=1, raw=True)
    df_predict['tfidf_word_match'] = df_input.apply(tfidf_word_match_share, axis=1, raw=True)



    bst = xgboost.Booster(model_file="./model/xgboost.model")
    d_predict = xgboost.DMatrix(df_predict)
    predict = bst.predict(d_predict)

    df_output = pd.DataFrame()
    df_output['id'] = df_input['id']
    df_output['predict'] = [int(x+0.5) for x in predict]
    with open(output_path,'w') as fout:
        for index in df_output.index:
            fout.write(str(df_output['id'].loc[index]) + '\t' + str(df_output['predict'].loc[index]) + '\n')



if __name__ == '__main__':
    process(sys.argv[1], sys.argv[2])
