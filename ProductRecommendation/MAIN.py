import sys
import readConf as rc
import readData as rd
import handleData as hd
import FPrecommendation as fpr
import ALSrecommendation as alsr
import KMEANSrecommendation as kmeansr
from pyspark import SparkConf, SparkContext
"""
this func is the entry of the whole application
"""

if __name__ == '__main__':

        """
        need assign the path of drive file to argv[0]
        need assign the path of recommendation_conf.json to argv[1]
        """
        if len(sys.argv) != 6:
                print('parameter error!')
                exit(1)
        conf = SparkConf()
        sc = SparkContext(conf = conf)
        sc.setLogLevel("OFF")
        #初始化
        print("============read conf==============")
        # read conf from jsonfile
        conf = rc.readConf(sys.argv[1])  
        # where is data from? hdfs or hive or others
        mode = sys.argv[2]
        location = sys.argv[3]
        outputpath = sys.argv[4]
        algorighm = sys.argv[5]
        data = ""
        #读取数据
        if "hdfs" == mode :
                print("============read data from hdfs==============")
                data = rd.readDataFromHDFS(location)
        elif "hive" == mode:
                print("============read data from hive==============")
                data = rd.readDataFromHIVE(location)
        
        #处理数据
        print("============handle data==============")
        data = hd.handle(data,conf["actions"])
        if algorighm == "FP":
                print("============FP recommendation==============")
                fpr.recommendation(data,conf["FP"],outputpath)
        elif algorighm == "ALS":
                print("============ALS recommendation==============")
                alsr.recommendation(data,conf["ALS"],outputpath)
        elif algorighm == "KMEANS":
                print("============KMEANS recommendation==============")
                kmeansr.recommendation(data,conf["KMEANS"],outputpath)