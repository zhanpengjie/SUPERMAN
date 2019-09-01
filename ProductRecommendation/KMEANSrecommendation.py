from pyspark.sql import SparkSession
from pyspark.ml.clustering import KMeans
import os,sys
import MyUtil
from pyspark.ml.linalg import Vectors


def recommendation(data,conf,outputpath):
        sparkSession = SparkSession.builder.getOrCreate()

        MyK= conf["k"]
        MyMaxIter = conf["maxIter"]
        MySeed = conf["seed"]
        MyResultSavePath = os.path.join("hdfs://{0}".format(outputpath), "KMEANSresult.json")

        print("============train KMEANSmodel==============")
        df = sparkSession.createDataFrame(transformData(data),["features","userId"])
        kmeans = KMeans(k=MyK, maxIter=MyMaxIter, seed=MySeed)
        model = kmeans.fit(df)
     
    # determine if the file exists
        (ret,out,err) = MyUtil.run_cmd(['hdfs', 'dfs', '-test','-e' ,MyResultSavePath])
        # if file already exists,then delete it
    if ret==0:
        print(MyResultSavePath+" file alreay exits")
                MyUtil.run_cmd(['hdfs', 'dfs', '-rm' ,'-r',MyResultSavePath])
    else:
        print(MyResultSavePath+" file dosen't exit")
    model.transform(df).write.json(MyResultSavePath)


def transformData(data):
    print("============transform data to df==============")
        """
    "result" like a matrix ,the number of row is userId's number ,the number of the column is the all productId's number ,
    for every user,Each user scores a product they have experienced, setting 0 for movies they have not experienced.the format as follow:
    result = [[userId1,socre1,socre2,...,]
         ,[userId2,socre1,socre2,...,]
         ,...]
    """

    result = []
    
        userNum = len(data)
        productList = []
    # find out all productId from data
        for userId in data:
                for productId in data[userId]:
                        if productId not in productList:
                            productList.append(productId)
        productNum = len(productList)

    # Generate a two-dimensional list "result" and initialize all values to 0
        for i in range(userNum):
            result.append([])
            for j in range(productNum+1):
                result[-1].append(0)
    # update two-dimensional list "result" ,each user scores a product they have experienced
        for userId in data:
                for productId in data[userId]:
                result[int(userId)-1][0] = int(userId)
                # productId is the id of string of productId
                        result[int(userId)-1][int(productId)] = float(data[userId][productId])   
    """
    Change the format and element type of "result" to generate "dataForKmeans",the format as follow:
    dataForKmeans = [(Vectors.dense(),userId1)
            ,(Vectors.dense(),userId2)
            ,...]
    """
        dataForKmeans = []
    for i in range(len(result)):
        dataForKmeans.append( (Vectors.dense(result[i][1:]), result[i][0]) ) 
    
    return dataForKmeans 