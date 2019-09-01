from pyspark.ml.linalg import Vectors
from pyspark.ml.stat import Correlation
from pyspark.sql import SparkSession
import math

def findResultFromFP(data,userId):
    for item in data.collect():
                # find the recommendations for the "userId" ,if find the "userId" successfully, just return the item["prediction"], that already calculated in FPresult.json in hdfs
                if item["userId"] == userId:
                         return item['productIds'], item["prediction"]
    return []

def findResultFromALS(data,userId):
    result = []
    for item in data.collect():
        # find the recommendations for the "userId" ,if find the "userId" successfully, just return the "result", which include recommendations those already calculated in ALSresult.json in hdfs
        if item["userId"] == int(userId):
            for product in item["recommendations"]:
                # formating string in "result"
                result.append([product["productIds"],product["rating"]])
    if not result:
        return []
    return sorted(result, key=lambda i:i[1],reverse=True)

def findResultFromKMEANS(data,userId):
    result = []
    clusterID = None
    baseRow = None
    # get "clusterID" which include "userId"
    for item in data.collect():
        if item["userId"] == int(userId):
            clusterID = item["prediction"]
            baseRow = item["features"]["values"]
    # get all the lists those are in the same cluster named as "clusterID", calculate the similarity between each list and "baseRow", get one list belonging to  one "userId" has the biggest similarity 
    listForCompare = []
    for item in data.collect():
        if item["prediction"] == clusterID:
            listForCompare.append( (item["features"]["values"],item["userId"]) )
    # the list that save similarities
    CorrList = []
    for compareRow in listForCompare:
        # cal euclidean distance representing the similarity between each list and "baseRow" 计算欧式距离
        total = 0
        for i in range(len(baseRow)):
            total += math.pow(baseRow[i]-compareRow[0][i],2)
        total = math.sqrt(total)
        # similarity
        Corr = total
        # userId
        userId = compareRow[1]
        # find the productid 
        productId = compareRow[0].index(max(compareRow[0]))+1
        CorrList.append([ productId, Corr ])
    # sort by similarities
    CorrList = sorted(CorrList, key=lambda i:i[1], reverse=True)
    similarities = []
    productId = []
    # result just have 3 values,select top 3 similarity in CorrList
    for item in CorrList:
        if item[0] not in productId:
            productId.append(item[0])
            similarities.append(item[1])
        if len(productId) == 20:
            break 
    for i in range(len(productId)):
        result.append([productId[i], similarities[i]])
    if not result:
        return []
    return result