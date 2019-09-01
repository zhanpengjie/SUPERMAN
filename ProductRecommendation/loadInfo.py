from pyspark.sql import SparkSession
from pyspark import SparkConf,SparkContext
import sys,os
import handleData as hd
import readConf as rc
import MyUtil

if __name__ == '__main__':

	"""
	load user info
	"""
    	if len(sys.argv) != 6:
            	print('parameter error!')
           	exit(1)
	conf = SparkConf()
    	sc = SparkContext(conf = conf)
    	sc.setLogLevel("OFF")
	
    	# result  print user info
    	result = {} 
    	# read args
    	print("============read args=========")
	mode = sys.argv[1]
	path = sys.argv[2]
	userId = int(sys.argv[3])
	confpath = sys.argv[4]
	outputpath = sys.argv[5]	

	print("============read data=========")
	data = None
	sparkSession = None
	if mode == "hdfs":
    		datapath = "hdfs://{0}".format(path)
		sparkSession = SparkSession.builder.getOrCreate()
		data = sparkSession.read.csv(datapath)
	if mode == "hive":
		c = SparkContext(conf = SparkConf())
		sqlContext = HiveContext(sc)
		data = sqlContext.sql("Select * from {0}".format(path))
	"""
	every "item" in data include 5 column
    	item[0] is actionId
    	item[1] is productId
	item[2] is score
 	item[3] is timestamp
	item[4] is userId
	"""
	print("============User Info==================")
	for item in data.collect():
		if int(item[4]) == userId:
			if item[0] not in result:
				result[item[0]] = 1
			else:
				result[item[0]] += 1
	print(result)

	print("============Product Info==============")
	product = {}
        for item in data.collect():
                if item[1] not in product:
                        product[item[1]] = {}
                        product[item[1]][item[0]] = 1
                elif item[0] not in product[item[1]]:
                        product[item[1]][item[0]] = 1
                else:
                        product[item[1]][item[0]] += 1
        print(product)
	print("==========save Product Info===========")
        productInfoPath = os.path.join("hdfs://{0}".format(outputpath), "productInfo.json")
	actionScore = []
	Info = {}
	for productid in product:
		#actionScore = []
		for actionid in product[productid]:
			actionScore.append({"productid":productid,"actionid":actionid,"score":product[productid][actionid]})
		#Info["productid"] = productid
		#Info["info"] = actionScore
	df = sparkSession.createDataFrame(actionScore)
        (ret,out,err) = MyUtil.run_cmd(['hdfs', 'dfs', '-test','-e' ,productInfoPath])
        # if file already exists,then delete it
        if ret==0:
                print(productInfoPath+" file alreay exits")
                MyUtil.run_cmd(['hdfs', 'dfs', '-rm' , '-r',productInfoPath])
        else:
                print(productInfoPath+" file dosen't exit")
        df.write.json(productInfoPath)	

	print("============Product Score==============")	
	conf = rc.readConf(confpath)
	data = hd.handle(data,conf["actions"])
	score = {}
 	for userid in data:
		for productid in data[userid]:
			if productid not in score:
				score[productid] = data[userid][productid]
			else:
				score[productid] += data[userid][productid]
	print("============save Product Score===========")
	productScore = sorted(score.items(), key=lambda d: d[1],reverse=True) 
	print(productScore)	
	productScorePath = os.path.join("hdfs://{0}".format(outputpath), "productScore.csv")
	df = sparkSession.createDataFrame(productScore)
        (ret,out,err) = MyUtil.run_cmd(['hdfs', 'dfs', '-test','-e' ,productScorePath])
        # if file already exists,then delete it
        if ret==0:
                print(productScorePath+" file alreay exits")
                MyUtil.run_cmd(['hdfs', 'dfs', '-rm' , '-r',productScorePath])
        else:
                print(productScorePath+" file dosen't exit")
        df.write.csv(productScorePath) 