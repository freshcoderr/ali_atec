# -*- coding:utf-8 -*-

'''
python 2.7
feature engineering
author: Yuanxihao
date: 2018-06-15
'''

import pandas as pd
import numpy as np
import jieba
import bz2file as bz
import os
import logging

STOP_WORDS_PATH = "./../stopwords.txt"
USER_DICT_PATH = "./../user_dict.txt"
ORI_DATA_PATH = "./train.csv"
OUTPUT_PATH = "./cleaned_data.csv"
USE_USERDICT_FLAG = 1
USE_STOPWORDS_FLAG = 1
common_used_numerals_tmp ={u'零':0, u'一':1, u'二':2, u'两':2, u'三':3, u'四':4, u'五':5, u'六':6, u'七':7, u'八':8, u'九':9, u'十':10,
                           u'百':100, u'千':1000, u'万':10000, u'亿':100000000}
num_str_start_symbol = [u'一', u'二', u'两', u'三', u'四', u'五', u'六', u'七', u'八', u'九', u'十']
more_num_str_symbol = [u'零', u'一', u'二', u'两', u'三', u'四', u'五', u'六', u'七', u'八', u'九', u'十', u'百', u'千', u'万', u'亿']


def get_func_name(func):
    def run(*args):
        logging.info("processing function: {}".format(func.__name__))
        if args:
            return func(*args)
        else:
            return func()
    return run


class Feature(object):

    def __init__(self):
        self.df_data = pd.DataFrame()
        self.question1 = []
        self.question2 = []
        self.is_duplicate = []

        self.stopwords = []
        if USE_STOPWORDS_FLAG:
            self.load_stopwords()

        self.userdict = []
        if USE_USERDICT_FLAG:
            self.load_userdict2jieba()

    @get_func_name
    def loadData(self, datapath = ORI_DATA_PATH):
        self.df_data = pd.read_csv(datapath)
        self.question1 = self.df_data['s1']
        self.question2 = self.df_data['s2']
        self.is_duplicate = self.df_data['is_duplicate']
        logging.info("loading data: {}".format(self.df_data[:5]))

    # return list of str
    @get_func_name
    def load_stopwords(self, stopwords_path = STOP_WORDS_PATH):
        assert os.path.exists(stopwords_path)
        with open(stopwords_path, "r") as f:
            for word in f.readlines():
                self.stopwords.append(word.strip('\n'))

    @get_func_name
    def load_userdict2jieba(self, userdict_path = USER_DICT_PATH):
        assert os.path.exists(userdict_path)
        jieba.load_userdict(userdict_path)

    def jiebaCutS1(self, raw):
        tmp = []
        if not isinstance(raw[0], str):
            raw[0].encode('utf-8')
        seg_list = jieba.cut(raw[0])
        if USE_STOPWORDS_FLAG:      # 是否去停用词
            for word in seg_list:
                if word.encode('utf-8') not in self.stopwords:
                    tmp.append(word)
            return " ".join(tmp)
        else:
            return " ".join(seg_list)

    def jiebaCutS2(self, raw):
        tmp = []
        if isinstance(raw[1], unicode):
            raw[1].encode('utf-8')
        seg_list = jieba.cut(raw[1])
        if USE_STOPWORDS_FLAG:
            for word in seg_list:
                if word.encode('utf-8') not in self.stopwords:
                    tmp.append(word)
            return " ".join(tmp)
        else:
            return " ".join(seg_list)



    def chinese2digits(self, uchars_chinese):
        common_used_numerals = {}
        for key in common_used_numerals_tmp:
            common_used_numerals[key] = common_used_numerals_tmp[key]
        total = 0
        r = 1  # 表示单位：个十百千...
        for i in range(len(uchars_chinese) - 1, -1, -1):
            val = common_used_numerals.get(uchars_chinese[i])
            if val >= 10 and i == 0:  # 应对 十三 十四 十*之类
                if val > r:
                    r = val
                    total = total + val
                else:
                    r = r * val
                    # total =total + r * x
            elif val >= 10:
                if val > r:
                    r = val
                else:
                    r = r * val
            else:
                total = total + r * val
        return total

    # 汉字数字转阿拉伯数字
    def changeChineseNumToArabs1(self, raw):
        if not isinstance(raw['s1'], unicode):
            oriStr = raw['s1'].decode('utf-8')
        else:
            oriStr = raw['s1']
        # print oriStr, type(oriStr)
        lenStr = len(oriStr)
        aProStr = ''
        if lenStr == 0:
            return aProStr

        hasNumStart = False
        numberStr = ''
        for idx in range(lenStr):
            if oriStr[idx] in num_str_start_symbol:
                if not hasNumStart:
                    hasNumStart = True

                numberStr += oriStr[idx]
            else:
                if hasNumStart:
                    if oriStr[idx] in more_num_str_symbol:
                        numberStr += oriStr[idx]
                        continue
                    else:
                        numResult = str(self.chinese2digits(numberStr))
                        numberStr = ''
                        hasNumStart = False
                        aProStr += numResult

                aProStr += oriStr[idx]
                pass

        if len(numberStr) > 0:
            resultNum = self.chinese2digits(numberStr)
            aProStr += str(resultNum)

        return aProStr


    def changeChineseNumToArabs2(self, raw):
        if not isinstance(raw['s2'], unicode):
            oriStr = raw['s2'].decode('utf-8')
        else:
            oriStr = raw['s2']
        # print oriStr, type(oriStr)
        lenStr = len(oriStr)
        aProStr = ''
        if lenStr == 0:
            return aProStr

        hasNumStart = False
        numberStr = ''
        for idx in range(lenStr):
            if oriStr[idx] in num_str_start_symbol:
                if not hasNumStart:
                    hasNumStart = True

                numberStr += oriStr[idx]
            else:
                if hasNumStart:
                    if oriStr[idx] in more_num_str_symbol:
                        numberStr += oriStr[idx]
                        continue
                    else:
                        numResult = str(self.chinese2digits(numberStr))
                        numberStr = ''
                        hasNumStart = False
                        aProStr += numResult

                aProStr += oriStr[idx]
                pass

        if len(numberStr) > 0:
            resultNum = self.chinese2digits(numberStr)
            aProStr += str(resultNum)

        return aProStr

    @get_func_name
    def storeData(self, data, outputPath = OUTPUT_PATH):
        try:
            data.to_csv(outputPath, encoding='utf-8')
            logging.info("store data: {}".format(outputPath))
        except:
            try:
                data.to_csv(outputPath)
                logging.info("store data: {}".format(outputPath))
            except:
                logging.error("please check encoding of the data to store")


    def feature_engineering(self):

        # step 1, load origin data
        self.loadData()

        # step 2, chineseNum2digit
        logging.info("begin chinese num to Arabs")
        df_cn2dg = pd.DataFrame()
        df_cn2dg['s1'] = self.df_data.apply(self.changeChineseNumToArabs1, axis=1, raw=True)
        df_cn2dg['s2'] = self.df_data.apply(self.changeChineseNumToArabs2, axis=1, raw=True)
        logging.info("processing chinese num to Arabs: {}".format(df_cn2dg[:5]))
        logging.info("df_cn2dg sentence type: {}".format(type(df_cn2dg['s1'][0])))

        # step 3, jieba cut (param: stopwords)
        logging.info("begin jieba cut")
        df_jieba = pd.DataFrame()
        # print df_cn2dg,type(df_cn2dg), type(self.df_data)
        df_jieba['s1'] = df_cn2dg.apply(self.jiebaCutS1, axis=1, raw=True)
        df_jieba['s2'] = df_cn2dg.apply(self.jiebaCutS2, axis=1, raw=True)
        logging.info("processing jieba cut done: {}".format(df_jieba[:5]))
        del df_cn2dg

        # step 4, store data
        self.storeData(df_jieba)
        logging.info("feature engineering success!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
    feature_engineering = Feature()

    logging.info("begin feature engineering")
    feature_engineering.feature_engineering()