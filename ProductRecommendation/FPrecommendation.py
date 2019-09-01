from pyspark.ml.fpm import FPGrowth
from pyspark.sql import SparkSession
import os,sys
import MyUtil


# transform data to find favorite product (score >= MyMinFavorScore) for every user and return "dataForFP"
def transformData(data,MyMinFavorScore):  
        print("============transform data to df==============")

        """
        result inclue tuples, in which has every userId and its favorite products in list, as follow :
        result = [(userId1,[productId1,productId1,...]),...]
        """
        result = []
        for userId in data:
                result.append((userId,[]))
                for productId in data[userId]:
                        if data[userId][productId] >= MyMinFavorScore:
                                result[-1][1].append(productId)

        # the format of "dataForFP" is the same to "result", just remove [](empty list)
        dataForFP = []
        for item in result:
                # dataForFP only append a list that length is not 0 
                if len(item[1]) != 0:
                        dataForFP.append(item)

        #print(len(dataForFP))
        return dataForFP        


def recommendation(data,conf,outputpath):
        
        sparkSession = SparkSession.builder.getOrCreate()       
        
        # extract algorithm parameters from conf file
        MyMinConfidence = conf["minConfidence"]
        MyMinSupport = conf["minSupport"]
        MyNumPartitions = conf["numPartitions"]
        MyMinFavorScore = conf["minfavorscore"]
        MyResultSavePath = os.path.join("hdfs://{0}".format(outputpath), "FPresult.json")
        MyModelSavePath = os.path.join("hdfs://{0}".format(outputpath), "FPmodel")

        print("============train FPmodel==============")

        df = sparkSession.createDataFrame(transformData(data,MyMinFavorScore),["userId","productIds"])
        fpGrowth = FPGrowth(itemsCol="productIds", minSupport = MyMinSupport, minConfidence = MyMinConfidence)
        model = fpGrowth.fit(df)
        
        print("============save association rules==============")
        # if the length of result is 0 
        if model.associationRules.count()==0:
                print("============no association rules! retry to change algorithm parameters ==============")
        else:
                # determine if the file exists
                (ret,out,err) = MyUtil.run_cmd(['hdfs', 'dfs', '-test','-e' ,MyModelSavePath])
                # if file already exists,then delete it
                if ret==0:
                        print(MyModelSavePath+" file alreay exits")
                        MyUtil.run_cmd(['hdfs', 'dfs', '-rm' , '-r',MyModelSavePath])
                print(MyModelSavePath+" file dosen't exit")
                model.save(MyModelSavePath)     
        print("============save association results==============")
        # if the length of result is 0 
        if model.associationRules.count()==0:
                print("============no association rules! retry to change algorithm parameters ==============")
        else:
                # determine if the file exists
                (ret,out,err) = MyUtil.run_cmd(['hdfs', 'dfs', '-test','-e' ,MyResultSavePath])
                # if file already exists,then delete it
                if ret==0:
                        print(MyResultSavePath+" file alreay exits")
                        MyUtil.run_cmd(['hdfs', 'dfs', '-rm' , '-r',MyResultSavePath])
                else:
                        print(MyResultSavePath+" file dosen't exit")
                model.transform(df).write.json(MyResultSavePath)