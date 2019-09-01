from pyspark.sql import SparkSession,HiveContext
from pyspark import SparkConf, SparkContext

# read data from hdfs and return data of type DataFrame
def readDataFromHDFS(path): 

	sparkSession = SparkSession.builder.getOrCreate()

        dataPath = "hdfs://{0}".format(path)

        data = sparkSession.read.csv(dataPath)

        return data

# read data from hive
def readDataFromHIVE(path):

	sc = SparkContext(conf = SparkConf())

	sqlContext = HiveContext(sc)

	data = sqlContext.sql("Select * from {0}".format(path))

	return data