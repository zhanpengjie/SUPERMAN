from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
import os,sys
import MyUtil

def recommendation(data,conf,outputpath):

        sparkSession = SparkSession.builder.getOrCreate()
        # extract algorithm parameters from recommendation_conf.json 
        MyRank = conf["rank"]
        MyMaxIter = conf["maxIter"]
        MyRegParam = conf["regParam"]
        MyNumUserBlocks = conf["numUserBlocks"]
        MyNumItemBlocks = conf["numItemBlocks"]
        MySeed = conf["seed"]
        MyResultSavePath = os.path.join("hdfs://{0}".format(outputpath), "ALSresult.json")

        print("============train ALSmodel==============")

        df = sparkSession.createDataFrame(transformData(data),["userId","productIds","score"])
        als = ALS(rank = MyRank, maxIter = MyMaxIter, seed = MySeed, regParam = MyRegParam, numUserBlocks = MyNumUserBlocks,numItemBlocks = MyNumItemBlocks,userCol="userId", itemCol="productIds", ratingCol="score")
        model = als.fit(df)

        print("============find three recommendation for all user==============")

        # determine if the file exists
        (ret,out,err) = MyUtil.run_cmd(['hdfs', 'dfs', '-test','-e' ,MyResultSavePath])
        # if file already exists,then delete it
        if ret==0:
                print(MyResultSavePath+" file alreay exits")
                MyUtil.run_cmd(['hdfs', 'dfs', '-rm' , '-r',MyResultSavePath])
        else:
                print(MyResultSavePath+" file dosen't exit")
        # just recommend 3 products for all users 
        model.recommendForAllUsers(20).write.json(MyResultSavePath)
        

def transformData(data):
        print("============transform data to df==============")
        
        result = []
        for userId in data:
                for productId in data[userId]:
                        result.append((int(userId),))
                        # productId is the id of the string of "productId"
                        result[-1] += (int(productId), float(data[userId][productId]))                       

#       print(result)
        """
        the format of result as follow:
        result = [(userId1,productId1,score1),(userId2,productId2,score2),...]
        """
        return result