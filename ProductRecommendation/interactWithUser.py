from pyspark.sql import SparkSession
from pyspark.ml.fpm import FPGrowthModel
from pyspark import SparkConf, SparkContext
import sys,json,os
import findResult as fr
import readConf as rc
import MyUtil


def transformForFP(path, productIds, predictions):
        # load model and data
        model_path = os.path.join("hdfs://{0}".format(path), "FPmodel")
        model = FPGrowthModel.load(model_path)
        data = model.associationRules.collect()
        # save confidences for result show
        confidences = []
        for prediction in predictions:
                tmp = []
                for antecedent, consequent, confidence in data:
                        if not set(consequent).issubset(set(productIds)) and set(antecedent).issubset(set(productIds)) and prediction == consequent[0]:
                                tmp.append(confidence)
                confidences.append(max(tmp))    
        # Conversion format
        result = []     
        for i in range(len(predictions)):
                result.append([predictions[i], confidences[i]])
        return sorted(result, key=lambda i:i[1],reverse=True)


def  filterInfo(productInfo,action,num,reverse,res):
        # save productids that need to sort by action
        productid = []
        for i in res:
                productid.append(int(i[0]))
        # save item whoes action that satisfies the parameter
        mid = []
        for item in productInfo:
                if item[0] == action:
                        mid.append(item)
        # save result
        result = {}
        for item in mid :
                if int(item[1]) not in result and int(item[1]) in productid:
                        result[int(item[1])] = item[2]
        # sort result
        return sorted(result.items(), key=lambda i:i[1],reverse=reverse)[:num]

if __name__ == '__main__':

        """
        need assign the path of drive file to argv[0]
        need assign the conf file path to argv[1]
        need assign the userid to argv[2]
        need assign the algorithm to argv[3]
        """
        if len(sys.argv) != 8:
                print('parameter error!there are 4 parameters :drive file path, conf file path ,userid ,algorithm')
                exit(1)

        conf = SparkConf()
        sc = SparkContext(conf = conf)
        sc.setLogLevel("OFF")
        sparkSession = SparkSession.builder.getOrCreate()

        print("============read argvs==============")
        data_path = sys.argv[1]
        originalData = sparkSession.read.csv("hdfs://{0}".format(data_path))
        userId = sys.argv[2]
        algorithm = sys.argv[3]
        path = sys.argv[4]
        resultpath = os.path.join("hdfs://{0}".format(path),algorithm+"result.json")
        productInfoPath = os.path.join("hdfs://{0}".format(path), "productInfo.json")
        productInfo = sparkSession.read.json(productInfoPath).collect()
        action = sys.argv[5]
        num = int(sys.argv[6])
        # Result in descending or ascending order, 0 means Ascending, 1 means descending
        reverse = int(sys.argv[7])
        # get their result's save path in hdfs and get data
        if algorithm == "FP": 
                data = sparkSession.read.json(resultpath)
                productIds, prediction = fr.findResultFromFP(data, userId)
                #conf = rc.readConf(conf_path)
                res = transformForFP(path, productIds, prediction)
                print("============FP==============")
                print(res)
                result = filterInfo(productInfo,action,num,reverse,res)
                print("========filter again========")
                print(result)
        elif algorithm == "ALS":
                data = sparkSession.read.json(resultpath)
                res = fr.findResultFromALS(data, userId)
                print("============ALS==============")
                print(res)
                print("========filter again========")
                result = filterInfo(productInfo,action,num,reverse,res)
                print(result)
        elif algorithm == "KMEANS":
                data = sparkSession.read.json(resultpath)
                res = fr.findResultFromKMEANS(data, userId )
                print("============KMEANS==============")
                print(res)
                print("========filter again========")
                result = filterInfo(productInfo,action,num,reverse,res)
                print(result)